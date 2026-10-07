import os
import sys
import glob
import json
import pickle
import time
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from functools import partial

# Agregar el directorio raíz al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import RLMALASampler_Huziel, MALASampler
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.td3_modules import FlaxTD3Actor, build_td3_state
from src.local_energy import local_energies

# =====================================================================
# FUNCIONES AUXILIARES DE EVALUACIÓN
# =====================================================================
def autocorr(x):
    # PROTECCIÓN: Si la energía es constante (cadena congelada) o NaN, la autocorrelación es infinita.
    if np.isnan(x).any() or np.var(x) == 0:
        return np.ones(len(x))
        
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    
    # Doble protección contra divisiones por cero imprevistas
    norm_factor = result[result.size // 2]
    if norm_factor == 0:
        return np.ones(len(x))
        
    return result[result.size // 2:] / norm_factor

def calculate_iat(acf_array, c=5.0):
    tau = 1.0
    for t, rho in enumerate(acf_array[1:], 1):
        if rho < 0.05 or t >= c * tau: 
            break
        tau += 2.0 * rho
    return tau

@partial(jax.jit, static_argnums=(0, 5))
def _collect_chunk(sampler_obj, w_start, k_start, p_psi, p_act, chunk_size):
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        params_dict = {"ansatz": p_psi}
        if p_act is not None: params_dict["actor"] = p_act
        w_new, stats, new_cache, _ = sampler_obj.step(subkey, w, params_dict, cache, exploration_mode=False)
        return (w_new, k, new_cache), w_new
    
    sum_log = lambda w: jnp.sum(ansatz_apply(p_psi, w))
    init_cache = {"log_psi_curr": ansatz_apply(p_psi, w_start), "grad_curr": jnp.clip(jax.grad(sum_log)(w_start), -50.0, 50.0)}
    final_carry, hist_w = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, hist_w

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, chunk_size=1000):
    hist_w_list = []
    curr_w, curr_k = w_start, k_start
    final_carry = (curr_w, curr_k, None) 
    if total_steps == 0: return final_carry, np.array([])
    chunk_size = min(chunk_size, total_steps)
    if chunk_size > 0:
        for _ in range(total_steps // chunk_size):
            final_carry, h_w = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
            curr_w, curr_k, _ = final_carry
            hist_w_list.append(np.asarray(h_w))
    arr_w = np.concatenate(hist_w_list, axis=0) if hist_w_list else np.array([])
    return final_carry, arr_w

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=20000):
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

# =====================================================================
# EVALUADOR UNIVERSAL DE MODELOS
# =====================================================================
def evaluate_model_universal(dir_path, n_samples, n_ter, is_td3=False):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega = cfg["omega"]
    g_val = cfg.get("interaction_strength", cfg.get("g"))
    
    weights_prefix = "TD3" if is_td3 else "MALA"
    # Fallback seguro para buscar los pesos del Ansatz
    weights_path = os.path.join(dir_path, f"{weights_prefix}_ansatz_weights.pkl")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(dir_path, "MALA_ansatz_weights.pkl")
        
    with open(weights_path, "rb") as f: p_ansatz = pickle.load(f)
    p_act = None
    
    system = Harmonic_1D(name="Harm", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg.get("backflow_components", 8), backflow_r_max=cfg.get("backflow_r_max", 5.0), backflow_width=cfg.get("backflow_width", 0.7), backflow_envelope_length=cfg.get("backflow_envelope_length", 3.0))
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    if is_td3:
        with open(os.path.join(dir_path, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
        
        # Soportar arquitecturas legacy
        actor_type = cfg.get("actor_type", "full")
        actor_amp = cfg.get("actor_amplitude", 0.05)
        
        if actor_type == "flat":
            from src.td3_modules import FlaxFlatActor
            td3_model = FlaxFlatActor(n_elec=n_elec, actor_amplitude=actor_amp)
            def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                w_1d, grad_1d = w[..., 0], grad[..., 0] 
                target_w = jax.lax.dynamic_slice_in_dim(w_1d, e_idx, 1, axis=-1)
                target_g = jax.lax.dynamic_slice_in_dim(grad_1d, e_idx, 1, axis=-1)
                state = jnp.concatenate([target_w, target_g, w_1d, grad_1d], axis=-1)
                act = td3_model.apply(p_a, state)
                return act[:, 0:1]*actor_amp, act[:, 1:2], {}
        else:
            td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp)
            def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                st = build_td3_state(
                    w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn,
                    quantum_force_norm=cfg.get("force_scale", 5.0),
                    particle_factor=cfg.get("n_scale", 10.0),
                    dist_factor=cfg.get("r_scale", 5.0),
                    omega=omega, gs=g_val, softening=cfg["softening"]
                )
                act = td3_model.apply(p_a, st)
                drift = jnp.zeros_like(act[:, 0:1]) if actor_type == "scale_only" else act[:, 0:1]*actor_amp
                return drift, act[:, 1:2], {}

        sampler = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=cfg.get("penalty_factor", 0.1), max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-50.0, max_clip_grads_ansatz=50.0)
    else:
        sampler = MALASampler(ansatz.apply, cfg["batch_size"], cfg["tau"], system, 50.0, -50.0, 50.0, -50.0, -50.0, -50.0, 50.0)

    # Evaluación Única Estandarizada
    key = jax.random.PRNGKey(42)
    w_init = jax.random.normal(key, (cfg["batch_size"], n_elec, 1))
    
    carry, _ = collect_samples_safe(sampler, w_init, key, p_ansatz, p_act, n_ter)
    t0 = time.time()
    _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, n_samples)
    t_total = time.time() - t0
    
    eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    tau_int = calculate_iat(autocorr(eloc_w0))
    total_samples = len(hist_w) * cfg["batch_size"]
    ess_s = (total_samples / max(tau_int, 1.0)) / max(t_total, 1e-6)

    # Extracción profunda de hiperparámetros
    return {
        "dir_path": dir_path,
        "method": "TD3" if is_td3 else "MALA",
        "actor_type": cfg.get("actor_type", "full"),
        "train_steps": cfg.get("train_steps", 0),
        "batch_size": cfg.get("batch_size", 0),
        "base_tau": cfg.get("tau", 0.0),
        "actor_amplitude": cfg.get("actor_amplitude", 0.0),
        "penalty_factor": cfg.get("penalty_factor", 0.0),
        "sigma_scale": cfg.get("sigma_action_scale", 0.2),
        "phys_jump": cfg.get("max_phys_jump_factor", cfg.get("max_phys_jump_std", 5.0)),
        "tau_int": float(tau_int),
        "ess_s": float(ess_s),
        "N": n_elec,
        "g": g_val
    }

def main():
    parser = argparse.ArgumentParser()
    # Pasa todas las carpetas base separadas por espacio.
    parser.add_argument("--base_dirs", nargs='+', required=True)
    parser.add_argument("--out_csv", type=str, default="master_evaluation_N10_g5.csv")
    args = parser.parse_args()

    target_N = 10
    target_g = 5.0
    valid_mala_dirs = []
    valid_td3_dirs = []

    print(f"[{time.strftime('%H:%M:%S')}] Rastreando directorios proporcionados...")

    # RASTREO PROFUNDO
    for bdir in args.base_dirs:
        if not os.path.isdir(bdir): continue
        for root, dirs, files in os.walk(bdir):
            if "params.json" in files:
                try:
                    with open(os.path.join(root, "params.json"), "r") as f:
                        cfg = json.load(f)
                    
                    n = cfg.get("n_elec")
                    g = cfg.get("interaction_strength", cfg.get("g"))
                    
                    if n == target_N and float(g) == target_g:
                        method = cfg.get("method", "MALA")
                        # Verificar estrcitamente que la red terminó de entrenar (existen pesos)
                        if method == "MALA" and os.path.exists(os.path.join(root, "MALA_ansatz_weights.pkl")):
                            valid_mala_dirs.append(root)
                        elif method == "TD3" and os.path.exists(os.path.join(root, "TD3_actor_weights.pkl")):
                            valid_td3_dirs.append(root)
                except Exception as e:
                    print(f"Advertencia: No se pudo leer {root}/params.json ({e})")

    print(f"[{time.strftime('%H:%M:%S')}] REPORTE DE BÚSQUEDA:")
    print(f" -> Encontrados: {len(valid_mala_dirs)} modelos MALA listos para evaluar.")
    print(f" -> Encontrados: {len(valid_td3_dirs)} modelos TD3 listos para evaluar.\n")

    # BANDERA DE SEGURIDAD (Si no hay nada, abortar limpiamente)
    if len(valid_mala_dirs) == 0 and len(valid_td3_dirs) == 0:
        print("ERROR: No se encontraron modelos completos para N=10 y g=5.0 en las carpetas proporcionadas.")
        print("Asegúrate de que los entrenamientos hayan concluido y generado los archivos .pkl.")
        sys.exit(0)

    # 1. ESTABLECER LA LÍNEA BASE MALA
    best_mala = {"tau_int": float('inf'), "ess_s": -1}
    for mdir in valid_mala_dirs:
        try:
            print(f"Evaluando Línea Base MALA: {os.path.basename(mdir)}...")
            res = evaluate_model_universal(mdir, n_samples=20000, n_ter=2000, is_td3=False)
            if res["ess_s"] > best_mala["ess_s"]:
                best_mala = res
        except Exception as e:
            print(f" -> Error evaluando MALA {mdir}: {e}", file=sys.stderr)

    if best_mala["ess_s"] == -1:
        print("ADVERTENCIA CRÍTICA: Ningún MALA pudo ser evaluado correctamente. Se fijará un tau_int=1000 por defecto para evitar crashear.")
        best_mala = {"tau_int": 1000.0, "ess_s": 0.0, "dir_path": "Fallback"}

    print(f"\n=======================================================")
    print(f"LÍNEA BASE GLOBAL MALA: tau_int={best_mala['tau_int']:.2f}, ESS/s={best_mala['ess_s']:.0f}")
    print(f"Origen: {best_mala.get('dir_path', 'N/A')}")
    print(f"=======================================================\n")

    # 2. EVALUAR TODOS LOS MODELOS TD3
    master_records = [best_mala] if best_mala["ess_s"] > 0 else []
    
    for i, tdir in enumerate(valid_td3_dirs):
        try:
            print(f"[{i+1}/{len(valid_td3_dirs)}] Evaluando TD3: {os.path.basename(tdir)}...")
            res = evaluate_model_universal(tdir, n_samples=20000, n_ter=2000, is_td3=True)
            res["G_RL"] = best_mala["tau_int"] / res["tau_int"]
            master_records.append(res)
            print(f"    -> Éxito | G_RL = {res['G_RL']:.2f}x | ESS/s = {res['ess_s']:.0f}")
        except Exception as e:
            print(f"    -> FALLO al evaluar {tdir}: {e}", file=sys.stderr)

    # 3. EXPORTAR EL DATASET MAESTRO
    if len(master_records) > 0:
        df = pd.DataFrame(master_records)
        df.to_csv(args.out_csv, index=False)
        print(f"\n¡Cosecha masiva completada! {len(master_records)} modelos consolidados en '{args.out_csv}'.")
    else:
        print("\nNo se pudo consolidar ningún resultado.")

if __name__ == "__main__":
    main()