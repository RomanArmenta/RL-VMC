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
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

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

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / result[result.size // 2]

def calculate_iat_sokal(acf_array, c=5.0):
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

def evaluate_model(dir_path, args, is_td3=False):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega = cfg["omega"]
    g_val = cfg.get("interaction_strength", cfg.get("g"))
    
    weights_prefix = "TD3" if is_td3 else "MALA"
    with open(os.path.join(dir_path, f"{weights_prefix}_ansatz_weights.pkl"), "rb") as f: p_ansatz = pickle.load(f)
    p_act = None
    
    system = Harmonic_1D(name="Harm", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"], backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], backflow_envelope_length=cfg["backflow_envelope_length"])
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    if is_td3:
        with open(os.path.join(dir_path, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
        td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=cfg["actor_amplitude"])
        
        # INYECCIÓN CORRECTA DE HIPERPARÁMETROS DE ESTADO
        def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
            st = build_td3_state(
                w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn,
                quantum_force_norm=cfg.get("force_scale", 5.0),
                particle_factor=cfg.get("n_scale", 10.0),
                dist_factor=cfg.get("r_scale", 5.0),
                omega=omega, gs=g_val, softening=cfg["softening"]
            )
            act = td3_model.apply(p_a, st)
            return act[:, 0:1]*cfg["actor_amplitude"], act[:, 1:2], {}
            
        sampler = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=0.1, max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-20.0, max_clip_grads_ansatz=20.0)
    else:
        sampler = MALASampler(ansatz.apply, cfg["batch_size"], cfg["tau"], system, 20.0, -20.0, 50.0, -50.0, -50.0, -50.0, 50.0)

    eval_keys = jax.random.split(jax.random.PRNGKey(cfg["seed"]), args.n_evaluations)
    ess_results, tau_results = [], []

    for i in range(args.n_evaluations):
        k1, k2 = jax.random.split(eval_keys[i])
        w_init = jax.random.normal(k1, (cfg["batch_size"], n_elec, 1))
        
        carry, _ = collect_samples_safe(sampler, w_init, k2, p_ansatz, p_act, args.n_ter)
        
        t0 = time.time()
        _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, args.n_samples)
        t_total = time.time() - t0
        
        # Calculamos la energía sobre una trayectoria para la autocorrelación
        traj_w0 = hist_w[:, 0, :, 0]
        # Evaluamos por chunks usando un helper compatible
        elocs = []
        for i in range(0, len(traj_w0), 20000):
            chunk = jnp.array(traj_w0[i:i+20000])
            e = local_energies(p_ansatz, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
            elocs.append(np.asarray(e))
        eloc_w0 = np.concatenate(elocs, axis=0)
        
        # VENTANA DINÁMICA DE SOKAL (Sin truncamiento rígido)
        tau = calculate_iat_sokal(autocorr(eloc_w0))
        
        total_samples = len(hist_w) * cfg["batch_size"]
        ess_s = (total_samples / max(tau, 1.0)) / max(t_total, 1e-6)
        
        ess_results.append(ess_s)
        tau_results.append(tau)

    return {
        "dir_name": os.path.basename(dir_path),
        "N": n_elec,
        "g": g_val,
        "ess_mean": np.mean(ess_results), 
        "ess_std": np.std(ess_results), 
        "tau_mean": np.mean(tau_results),
        "actor_amplitude": cfg.get("actor_amplitude", 0),
        "penalty_factor": cfg.get("penalty_factor", 0),
        "sigma_scale": cfg.get("sigma_action_scale", 0),
        "max_phys_jump": cfg.get("max_phys_jump_factor", 0)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--n_evaluations", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    mala_base_dir = os.path.join(args.base_dir, "MALA_Baselines")
    mala_dirs = [d for d in glob.glob(os.path.join(mala_base_dir, "MALA_tau*")) if os.path.isdir(d)]
    # Busca cualquier carpeta que empiece con TD3_, pero ignora el symlink genérico "TD3"
    td3_dirs = [d for d in glob.glob(os.path.join(args.base_dir, "TD3_*")) if os.path.isdir(d) and os.path.basename(d) != "TD3"]

# 1. Evaluar MALA y encontrar la mejor línea base
    print("Evaluando baselines de MALA...")
    # CORRECCIÓN: Agregar dir_name por defecto para evitar KeyError
    best_mala = {"dir_name": "None", "ess_mean": -1, "tau_mean": float('inf')}
    for m_dir in mala_dirs:
        try:
            res = evaluate_model(m_dir, args, is_td3=False)
            if res["ess_mean"] > best_mala["ess_mean"]:
                best_mala = res
        except Exception as e:
            print(f"Error en {m_dir}: {e}")

    if best_mala["ess_mean"] == -1:
        print("[ERROR CRÍTICO] Ningún MALA generó pesos válidos. Abortando evaluación.", file=sys.stderr)
        os._exit(1)

    # 2. Evaluar todo el Grid Search de TD3
    print(f"Mejor MALA: {best_mala['dir_name']} (Tau={best_mala['tau_mean']:.2f}, ESS/s={best_mala['ess_mean']:.0f})")
    print("\nEvaluando malla de hiperparámetros TD3...")
    
    td3_results = []
    for td3_dir in td3_dirs:
        try:
            res = evaluate_model(td3_dir, args, is_td3=True)
            # Ganancia Relativa de Eficiencia (Topológica, sin sesgo de wall-clock time)
            res["G_RL"] = best_mala["tau_mean"] / res["tau_mean"]
            td3_results.append(res)
            print(f" -> Evaluado {res['dir_name']}: G_RL = {res['G_RL']:.2f}x")
        except Exception as e:
            print(f" -> Falla en {os.path.basename(td3_dir)}: {e}")

    # 3. Guardar resultados (¡BLINDADO!)
    if len(td3_results) == 0:
        print("ERROR CRÍTICO: Ningún modelo TD3 generó resultados válidos. Revisa los logs de entrenamiento.")
        os._exit(1)
        
    df = pd.DataFrame(td3_results).sort_values("G_RL", ascending=False)
    df.to_csv(os.path.join(args.base_dir, "hyperparam_sweep_results.csv"), index=False)
    
    best_td3 = df.iloc[0]
    with open(os.path.join(args.base_dir, "best_td3_model.txt"), "w") as f:
        f.write(best_td3["dir_name"])

    # 4. Generar Dashboard del Grid Search
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Tomamos el Top 10 para claridad visual
    top_df = df.head(10)
    labels = [f"Amp:{r.actor_amplitude} | Pen:{r.penalty_factor}\nSig:{r.sigma_scale} | Jmp:{r.max_phys_jump}" for _, r in top_df.iterrows()]
    
    # Eje 1: G_RL (Eficiencia algorítmica)
    color = '#2c3e50'
    ax1.set_xlabel('Hyperparameter Configurations (Top 10)', weight='bold')
    ax1.set_ylabel(r'Algorithmic Efficiency Gain ($G_{RL}$)', color=color, weight='bold')
    bars = ax1.bar(labels, top_df["G_RL"], color=color, alpha=0.8, label="Topological Gain (Sokal Tau)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(1.0, color='r', linestyle='--', linewidth=2, label="MALA Baseline")
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.2f}x', ha='center', va='bottom', fontsize=10, weight='bold')

    # Eje 2: ESS/s (Eficiencia computacional wall-clock)
    ax2 = ax1.twinx()
    color = '#e74c3c'
    ax2.set_ylabel('Effective Sample Size / sec (ESS/s)', color=color, weight='bold')
    ax2.plot(labels, top_df["ess_mean"], color=color, marker='o', linewidth=2, markersize=8, label="Wall-clock ESS/s")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    sys_N = top_df.iloc[0]["N"]
    sys_g = top_df.iloc[0]["g"]
    fig.suptitle(fr"Hyperparameter Grid Search Evaluation: Strongly Correlated Wigner Regime ($N={sys_N}, g={sys_g}$)", weight='bold', fontsize=15)
    fig.tight_layout()
    plt.savefig(os.path.join(args.base_dir, "Hyperparam_Sweep_Dashboard.pdf"), dpi=300, format='pdf')
    print("Dashboard del Grid Search generado y guardado con éxito.")

    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()