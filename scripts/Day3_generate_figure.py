import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import json
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns

# Importaciones de tu framework (Asegúrate de que estas rutas coincidan con tu repo)
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import RLMALASampler_Huziel, MALASampler
from src.local_energy import local_energies
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.td3_modules import FlaxTD3Actor, build_td3_state
from functools import partial

# --- FUNCIONES MATEMÁTICAS ---
def autocorr(x):
    """Calcula la función de autocorrelación."""
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / result[result.size // 2]

def calculate_iat(acf_array):
    """Calcula el tiempo de autocorrelación integrado (tau_int)."""
    tau = 1.0
    for rho in acf_array[1:]:
        if rho < 0: break
        tau += 2.0 * rho
    return tau

# --- REFACTORIZACIÓN ANTI-OOM (CHUNKING) ---
@partial(jax.jit, static_argnums=(0, 5))
def _collect_chunk(sampler_obj, w_start, k_start, p_psi, p_act, chunk_size):
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        params_dict = {"ansatz": p_psi}
        if p_act is not None: params_dict["actor"] = p_act
        w_new, stats, new_cache, _ = sampler_obj.step(subkey, w, params_dict, cache, exploration_mode=False)
        return (w_new, k, new_cache), (w_new, stats)
    
    def sum_log(w): return jnp.sum(ansatz_apply(p_psi, w))
    init_cache = {"log_psi_curr": ansatz_apply(p_psi, w_start), "grad_curr": jnp.clip(jax.grad(sum_log)(w_start), -50.0, 50.0)}
    
    final_carry, (hist_w, hist_stats) = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, hist_w, hist_stats

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, subsample=1, chunk_size=1000):
    """Muestrea en bloques, descarga a RAM y devuelve trayectorias de posiciones y estadísticas (aceptación)."""
    hist_w_list = []
    hist_acc_list = []
    curr_w, curr_k = w_start, k_start
    
    chunk_size = max(subsample, (chunk_size // subsample) * subsample)
    if total_steps == 0:
        return (w_start, k_start, None), np.array([]), np.array([])

    for _ in range(total_steps // chunk_size):
        final_carry, h_w, h_stats = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
        curr_w, curr_k, _ = final_carry
        
        # Submuestreo directo antes de almacenar en la lista de RAM
        hist_w_list.append(np.asarray(h_w[::subsample]))
        
        # Extraer métrica de aceptación si está disponible, si no, rellenar con unos (como dummy)
        if "acceptance" in h_stats:
            hist_acc_list.append(np.asarray(h_stats["acceptance"][::subsample]))
        elif "acceptance_rate" in h_stats:
             hist_acc_list.append(np.asarray(h_stats["acceptance_rate"][::subsample]))
        else:
             hist_acc_list.append(np.ones_like(h_w[::subsample, 0, 0, 0]))

    arr_w = np.concatenate(hist_w_list, axis=0) if hist_w_list else np.array([])
    arr_acc = np.concatenate(hist_acc_list, axis=0) if hist_acc_list else np.array([])
    return final_carry, arr_w, arr_acc

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=20000):
    """Calcula energías locales en bloques seguros."""
    if len(flat_w) == 0: return np.array([])
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

# --- SCRIPT PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Carpeta principal, ej. results_1D_N4_sweep_g_100kTrain_PaperReward")
    parser.add_argument("--perm_dir", type=str, required=True, help="Carpeta de permutación, ej. Perm_g4.0_Om0.1_Amp0.01_Run1")
    parser.add_argument("--n_samples", type=int, default=50000, help="Pasos totales de muestreo a registrar")
    parser.add_argument("--n_subsamples", type=int, default=1, help="Dejar en 1 para medir saltos físicos exactos")
    parser.add_argument("--n_ter", type=int, default=5000, help="Pasos de burn-in silenciosos")
    parser.add_argument("--max_clip_grads", type=float, default=20.0)
    args = parser.parse_args()

    full_perm_dir = os.path.join(args.base_dir, args.perm_dir)
    dir_mala = os.path.join(full_perm_dir, "MALA")
    dir_td3 = os.path.join(full_perm_dir, "TD3")

    # Carga de Configuraciones y Pesos (Ansatz Congelado y Actor Congelado)
    with open(os.path.join(dir_td3, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["omega"], cfg["interaction_strength"]
    
    with open(os.path.join(dir_mala, "MALA_ansatz_weights.pkl"), "rb") as f: p_ansatz_mala = pickle.load(f)
    with open(os.path.join(dir_td3, "TD3_ansatz_weights.pkl"), "rb") as f: p_ansatz_td3 = pickle.load(f)
    with open(os.path.join(dir_td3, "TD3_actor_weights.pkl"), "rb") as f: p_actor_td3 = pickle.load(f)

    # Inicializar Entorno Físico
    system = Harmonic_1D(name="Harmonic1D", n_elec=n_elec, n_up=n_up, n_dn=n_dn, 
                         coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, 
                         interaction_strength=g_val, softening=cfg["softening"], dim=1)
    
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"],
                                backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], 
                                backflow_envelope_length=cfg["backflow_envelope_length"])
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    # Inicialización del Actor (Modo Inferencia Congelado)
    td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=cfg["actor_amplitude"])
    def td3_wrapper(p_act, w, e_idx, r_nuc, exp_mode, grad):
        state = build_td3_state(w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn)
        action = td3_model.apply(p_act, state)
        # exp_mode = False, el actor es completamente determinista aquí
        return action[:, 0:1] * cfg["actor_amplitude"], action[:, 1:2], {}

    # Inicialización Segura
    sampler_mala = MALASampler(
        ansatz.apply, cfg["batch_size"], cfg["tau"], system, 
        args.max_clip_grads, -args.max_clip_grads, 50.0, -50.0, -50.0, -50.0, 50.0
    )
    
    sampler_td3 = RLMALASampler_Huziel(
        ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=td3_wrapper, 
        tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=0.1, max_clip_log_ratio=50.0, 
        min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, 
        posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, 
        min_clip_grads_ansatz=-args.max_clip_grads, max_clip_grads_ansatz=args.max_clip_grads
    )

    key = jax.random.PRNGKey(42)
    w_init = jax.random.normal(key, (cfg["batch_size"], n_elec, 1))

    # --- FASE 1: TERMALIZACIÓN ---
    print(f"[{g_val}] Termalizando {args.n_ter} pasos...")
    carry_mala, _, _ = collect_samples_safe(sampler_mala, w_init, key, p_ansatz_td3, None, args.n_ter, chunk_size=1000)
    carry_td3, _, _ = collect_samples_safe(sampler_td3, w_init, key, p_ansatz_td3, p_actor_td3, args.n_ter, chunk_size=1000)
    
    w_ter_mala, k_ter_mala, _ = carry_mala
    w_ter_td3, k_ter_td3, _ = carry_td3

    # --- FASE 2: MUESTREO DE PRODUCCIÓN CRONOMETRADO ---
    print(f"[{g_val}] Muestreando {args.n_samples} pasos de producción...")
    
    start_time = time.time()
    _, hist_mala, acc_mala = collect_samples_safe(sampler_mala, w_ter_mala, k_ter_mala, p_ansatz_td3, None, args.n_samples, subsample=args.n_subsamples, chunk_size=1000)
    t_mala = time.time() - start_time

    start_time = time.time()
    _, hist_td3, acc_td3 = collect_samples_safe(sampler_td3, w_ter_td3, k_ter_td3, p_ansatz_td3, p_actor_td3, args.n_samples, subsample=args.n_subsamples, chunk_size=1000)
    t_td3 = time.time() - start_time

    # --- CÁLCULO DE MÉTRICAS (Las 7 métricas del Asesor) ---
    print(f"[{g_val}] Calculando métricas físicas...")
    
    # 1. Acceptance
    mean_acc_mala = np.mean(acc_mala)
    mean_acc_td3 = np.mean(acc_td3)
    
    # Cálculos de distancias físicas
    # hist_shape = (steps, batch_size, n_elec, 1) -> extraemos solo el caminante cero para la métrica en el tiempo
    traj_mala_w0 = hist_mala[:, 0, :, 0]
    traj_td3_w0 = hist_td3[:, 0, :, 0]
    
    # 2. Expected Jump Distance EJD (<|dx|>)
    dx_mala = np.abs(traj_mala_w0[1:] - traj_mala_w0[:-1])
    dx_td3 = np.abs(traj_td3_w0[1:] - traj_td3_w0[:-1])
    ejd_mala = np.mean(dx_mala)
    ejd_td3 = np.mean(dx_td3)

    # 3. Expected Squared Jump Distance ESJD (<(dx)^2>)
    esjd_mala = np.mean(dx_mala**2)
    esjd_td3 = np.mean(dx_td3**2)

    # Evaluación de Energía en todo el batch
    flat_mala = hist_mala[..., 0].reshape(-1, n_elec)
    flat_td3 = hist_td3[..., 0].reshape(-1, n_elec)

    eloc_mala_all = get_eloc_safe(p_ansatz_td3, flat_mala, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3_all = get_eloc_safe(p_ansatz_td3, flat_td3, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    # Para el análisis de autocorrelación usamos la traza del caminante cero
    eloc_mala_w0 = get_eloc_safe(p_ansatz_td3, traj_mala_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3_w0 = get_eloc_safe(p_ansatz_td3, traj_td3_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    # 4. Tiempo de Autocorrelación (Tau_int)
    acf_mala = autocorr(eloc_mala_w0)[:2000]
    acf_td3 = autocorr(eloc_td3_w0)[:2000]
    tau_mala = calculate_iat(acf_mala)
    tau_td3 = calculate_iat(acf_td3)

    # 5. Effective Sample Size (N_eff)
    total_samples = len(eloc_mala_all)
    neff_mala = total_samples / max(tau_mala, 1.0)
    neff_td3 = total_samples / max(tau_td3, 1.0)

    # 6. Eficiencia Computacional (N_eff / t)
    ess_t_mala = neff_mala / t_mala
    ess_t_td3 = neff_td3 / t_td3

    # 7. Standard Error SE(E)
    var_mala = np.var(eloc_mala_all)
    var_td3 = np.var(eloc_td3_all)
    se_mala = np.sqrt((var_mala * tau_mala) / max(total_samples, 1))
    se_td3 = np.sqrt((var_td3 * tau_td3) / max(total_samples, 1))

    # El Santo Grial: G_RL
    G_RL = ess_t_td3 / ess_t_mala

    # --- IMPRESIÓN DE LA TABLA ---
    print("\n" + "="*80)
    print(f" TABLA DEFINITIVA DE PRODUCCIÓN CONGELADA (g={g_val}, N={n_elec}, {args.n_samples} pasos)")
    print("="*80)
    print(f"{'Métrica':<30} | {'MALA':<20} | {'RL-MALA (TD3)':<20}")
    print("-" * 80)
    print(f"{'Acceptance':<30} | {mean_acc_mala:.4f} | {mean_acc_td3:.4f}")
    print(f"{'<|dx|> (EJD)':<30} | {ejd_mala:.4f} | {ejd_td3:.4f}")
    print(f"{'<(dx)^2> (ESJD)':<30} | {esjd_mala:.4f} | {esjd_td3:.4f}")
    print(f"{'Tau_int (E_L)':<30} | {tau_mala:.2f} | {tau_td3:.2f}")
    print(f"{'N_eff':<30} | {int(neff_mala):<20} | {int(neff_td3):<20}")
    print(f"{'N_eff / sec (ESS/t)':<30} | {ess_t_mala:.2f} | {ess_t_td3:.2f}")
    print(f"{'SE(E)':<30} | {se_mala:.6f} | {se_td3:.6f}")
    print("="*80)
    print(f" >>> VENTAJA DE EFICIENCIA DEL MUESTREADOR: G_RL = {G_RL:.3f}x <<<")
    print("="*80 + "\n")

    # --- GENERACIÓN DE FIGURE 2 (Los Cuatro Paneles) ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(fr"Figure 2: Performance Evaluation of Trained RL-Sampler vs Standard MALA ($N={n_elec}, g={g_val}$)", weight='bold', fontsize=16)

    # Panel A: Time Series (Energía Local, primeros 500 pasos por claridad visual)
    ax_a = axs[0, 0]
    ax_a.plot(eloc_mala_w0[:500], color='black', alpha=0.6, label='MALA')
    ax_a.plot(eloc_td3_w0[:500], color='red', alpha=0.8, label='TD3')
    ax_a.set_title("A. Local Energy Time Series (Trace excerpt)")
    ax_a.set_xlabel("MCMC Steps")
    ax_a.set_ylabel(r"$E_L$ (Hartrees)")
    ax_a.legend()

    # Panel B: Autocorrelation Function
    ax_b = axs[0, 1]
    lags = np.arange(len(acf_mala))
    # Buscar el índice donde la ACF cae debajo de 0.05 para centrar la gráfica visualmente
    max_lag = min(len(acf_mala), int(max(tau_mala, tau_td3) * 5) + 10)
    
    ax_b.plot(lags[:max_lag], acf_mala[:max_lag], color='black', linestyle='--', linewidth=2, label=fr'MALA ($\tau={tau_mala:.1f}$)')
    ax_b.plot(lags[:max_lag], acf_td3[:max_lag], color='red', linewidth=2, label=fr'TD3 ($\tau={tau_td3:.1f}$)')
    ax_b.axhline(0, color='gray', linestyle=':')
    ax_b.set_title("B. Autocorrelation Function of $E_L$")
    ax_b.set_xlabel("Lag (Steps)")
    ax_b.set_ylabel("Autocorrelation Coefficient")
    ax_b.legend()

    # Panel C: Energy Histogram
    ax_c = axs[1, 0]
    # Tomamos un submuestreo de las energías globales para pintar el KDE suave
    idx = np.random.choice(len(eloc_mala_all), size=min(10000, len(eloc_mala_all)), replace=False)
    sns.kdeplot(eloc_mala_all[idx], ax=ax_c, color='black', linestyle='--', linewidth=2, label=f'MALA')
    sns.kdeplot(eloc_td3_all[idx], ax=ax_c, color='red', alpha=0.6, linewidth=2, fill=True, label=f'TD3')
    ax_c.set_title("C. Invariant Energy Distribution")
    ax_c.set_xlabel(r"Local Energy $E_L$")
    ax_c.set_ylabel("Density")
    ax_c.legend()

    # Panel D: Running Standard Error
    ax_d = axs[1, 1]
    # Calculamos el SE progresivo
    steps_array = np.arange(100, len(eloc_mala_w0), 100) # Evaluamos cada 100 pasos
    running_se_mala = []
    running_se_td3 = []
    for s in steps_array:
        v_m = np.var(eloc_mala_w0[:s])
        v_t = np.var(eloc_td3_w0[:s])
        # Asumimos que tau_int ya converge rápidamente a su valor estimado
        running_se_mala.append(np.sqrt((v_m * tau_mala) / s))
        running_se_td3.append(np.sqrt((v_t * tau_td3) / s))

    ax_d.plot(steps_array, running_se_mala, color='black', linestyle='--', linewidth=2, label='MALA')
    ax_d.plot(steps_array, running_se_td3, color='red', linewidth=2, label='TD3')
    ax_d.set_title("D. Convergence of Running Standard Error")
    ax_d.set_xlabel("Production Steps")
    ax_d.set_ylabel(r"Running $SE(E)$")
    ax_d.set_yscale('log')
    ax_d.legend()

    plt.tight_layout()
    out_name = f"Figure2_Production_N{n_elec}_g{g_val}_{args.n_samples}s.png"
    out_path = os.path.join(full_perm_dir, out_name)
    plt.savefig(out_path, dpi=300)
    print(f"[{g_val}] Figura 2 generada exitosamente: {out_path}")

if __name__ == "__main__":
    main()