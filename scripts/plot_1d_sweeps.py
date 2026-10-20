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
from src.samplers import RLMALASampler_Huziel
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.local_energy import local_energies

# =====================================================================
# 1. MÓDULO ESTADÍSTICO (ACF y ESJD)
# =====================================================================
def autocorr(x):
    if np.isnan(x).any() or np.var(x) == 0: return np.ones(len(x))
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    norm_factor = result[result.size // 2]
    if norm_factor == 0: return np.ones(len(x))
    return result[result.size // 2:] / norm_factor

def calculate_iat(acf_array, c=5.0):
    tau = 1.0
    for t, rho in enumerate(acf_array[1:], 1):
        if rho < 0.05 or t >= c * tau: break
        tau += 2.0 * rho
    return tau

def compute_esjd(hist_w):
    diffs = hist_w[1:] - hist_w[:-1]
    return float(np.mean(np.sum(diffs**2, axis=(2, 3))))

@partial(jax.jit, static_argnums=(0, 5))
def _collect_chunk(sampler_obj, w_start, k_start, p_psi, p_act, chunk_size):
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        w_new, stats, new_cache, _ = sampler_obj.step(subkey, w, {"ansatz": p_psi, "actor": p_act}, cache, exploration_mode=False)
        return (w_new, k, new_cache), (w_new, stats['acceptance'])
    sum_log = lambda w: jnp.sum(ansatz_apply(p_psi, w))
    init_cache = {"log_psi_curr": ansatz_apply(p_psi, w_start), "grad_curr": jnp.clip(jax.grad(sum_log)(w_start), -50.0, 50.0)}
    final_carry, (hist_w, hist_acc) = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, hist_w, hist_acc

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, chunk_size=1000):
    hist_w_list, hist_acc_list = [], []
    curr_w, curr_k = w_start, k_start
    final_carry = (curr_w, curr_k, None)
    if total_steps == 0: return final_carry, np.array([]), np.array([])
    for _ in range(total_steps // chunk_size):
        final_carry, h_w, h_acc = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
        curr_w, curr_k, _ = final_carry
        hist_w_list.append(np.asarray(h_w)); hist_acc_list.append(np.asarray(h_acc))
    return final_carry, np.concatenate(hist_w_list, axis=0), np.concatenate(hist_acc_list, axis=0)

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=20000):
    elocs = [np.asarray(calc_eloc_chunk(p_psi, jnp.array(flat_w[i:i+chunk_size]), sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)) for i in range(0, len(flat_w), chunk_size)]
    return np.concatenate(elocs, axis=0)

# =====================================================================
# 2. MOTOR DE EVALUACIÓN NEURONAL
# =====================================================================
def evaluate_td3_model(dir_path, n_eval, n_samples, n_ter, csv_out_dir):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["omega"], cfg.get("interaction_strength", cfg.get("g", 0.0))
    
    with open(os.path.join(dir_path, "TD3_ansatz_weights.pkl"), "rb") as f: p_ansatz = pickle.load(f)
    with open(os.path.join(dir_path, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
    
    system = Harmonic_1D(name="Harm", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg.get("backflow_components", 8), backflow_r_max=cfg.get("backflow_r_max", 5.0), backflow_width=cfg.get("backflow_width", 0.7), backflow_envelope_length=cfg.get("backflow_envelope_length", 3.0))
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    actor_amp = cfg.get("actor_amplitude", 0.05)
    from src.td3_modules import FlaxTD3Actor, build_td3_state
    td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp)
    def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
        st = build_td3_state(w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn, quantum_force_norm=cfg.get("force_scale", 5.0), particle_factor=cfg.get("n_scale", 10.0), dist_factor=cfg.get("r_scale", 5.0), omega=omega, gs=g_val, softening=cfg["softening"])
        act = td3_model.apply(p_a, st)
        return act[:, 0:1]*actor_amp, act[:, 1:2], {}
    
    sampler = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=cfg.get("penalty_factor", 0.1), max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-50.0, max_clip_grads_ansatz=50.0)

    eval_keys = jax.random.split(jax.random.PRNGKey(cfg.get("seed", 42)), n_eval)
    metrics = {"E": [], "Tau": [], "Acc": []}
    model_id = os.path.basename(dir_path)
    
    for i in range(n_eval):
        k1, k2 = jax.random.split(eval_keys[i])
        w_init = jax.random.normal(k1, (cfg["batch_size"], n_elec, 1))
        
        carry, _, _ = collect_samples_safe(sampler, w_init, k2, p_ansatz, p_act, n_ter)
        _, hist_w, hist_acc = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, n_samples)
        
        eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        tau_int = calculate_iat(autocorr(eloc_w0))
        
        metrics["E"].append(float(np.mean(eloc_w0)))
        metrics["Tau"].append(tau_int)
        metrics["Acc"].append(float(np.mean(hist_acc)))

        # Trazabilidad
        acc_data = hist_acc if hist_acc.ndim == 1 else hist_acc[:, 0]
        csv_filename = os.path.join(csv_out_dir, f"Trace_{model_id}_run{i+1}.csv")
        pd.DataFrame({"Step": np.arange(n_samples), "Energy": eloc_w0, "Acceptance": acc_data}).to_csv(csv_filename, index=False)

    return {
        "dir_path": dir_path, "Model_ID": model_id,
        "tau": cfg["tau"], "amp": cfg["actor_amplitude"], "pen": cfg["penalty_factor"], 
        "sig": cfg.get("sigma_action_scale", 0.2), "jmp": cfg.get("max_phys_jump_factor", 3.0),
        "Tau_mean": np.mean(metrics["Tau"]), "Tau_err": np.std(metrics["Tau"]) / np.sqrt(n_eval),
        "E_mean": np.mean(metrics["E"]), "E_err": np.std(metrics["E"]) / np.sqrt(n_eval),
    }

# =====================================================================
# 3. ALGORITMO DE HIPER-CORTES Y GENERACIÓN DE GRÁFICAS
# =====================================================================
def extract_1d_sweep(df, target_param, other_params, optimal_row):
    """ Encuentra el subconjunto 1D que varía 'target_param' y pasa lo más cerca posible del Óptimo Global """
    groups = df.groupby(other_params)
    best_group, max_unique, min_dist = None, 0, float('inf')
    
    for name, group in groups:
        n_unique = group[target_param].nunique()
        # Normalizar distancia Euclideana en el hiper-espacio
        dist = sum(((name[i] - optimal_row[other_params[i]]) / (optimal_row[other_params[i]] + 1e-8))**2 for i in range(len(other_params)))
        if n_unique > max_unique or (n_unique == max_unique and dist < min_dist):
            max_unique = n_unique; min_dist = dist; best_group = group
            
    return best_group.sort_values(target_param)

def plot_1d_sweeps(df, out_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    optimal_row = df.loc[df['Tau_mean'].idxmin()]
    
    print(f"\n[Óptimo Global Encontrado]")
    print(f" -> Directorio: {optimal_row['Model_ID']}")
    print(f" -> T: {optimal_row['tau']}, A: {optimal_row['amp']}, P: {optimal_row['pen']}, S: {optimal_row['sig']}, J: {optimal_row['jmp']}")
    print(f" -> Tau_int: {optimal_row['Tau_mean']:.2f}\n")

    params = ['tau', 'amp', 'pen', 'sig', 'jmp']
    labels = ['Integration Step ($\\tau$)', 'Actor Amplitude (A)', 'Penalty Factor (P)', 'Sigma Scale ($\\sigma$)', 'Max Phys Jump (J)']
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("1D Hyperparameter Sensitivity Sweeps (Centered near Global Optimum)", weight='bold', fontsize=16)
    axs = axs.flatten()

    for idx, target in enumerate(params):
        other_params = [p for p in params if p != target]
        sweep_df = extract_1d_sweep(df, target, other_params, optimal_row)
        
        ax = axs[idx]
        if sweep_df is None or len(sweep_df) < 2:
            ax.text(0.5, 0.5, "Datos Insuficientes\npara Sweep 1D", ha='center', va='center', weight='bold')
            continue
            
        fixed_str = ", ".join([f"{p.capitalize()[0]}={sweep_df.iloc[0][p]}" for p in other_params])
        
        # Ploteo de línea con barras de error
        ax.errorbar(sweep_df[target], sweep_df['Tau_mean'], yerr=sweep_df['Tau_err'], 
                    fmt='-o', color='#2c3e50', linewidth=2, capsize=5, markersize=8, label='Evaluated Models')
        
        # Resaltar si el óptimo global está en esta línea
        opt_in_sweep = sweep_df[sweep_df['Model_ID'] == optimal_row['Model_ID']]
        if not opt_in_sweep.empty:
            ax.scatter(opt_in_sweep[target], opt_in_sweep['Tau_mean'], color='#e74c3c', s=250, marker='*', zorder=5, label='Global Optimum', edgecolor='black')

        ax.set_xlabel(labels[idx], weight='bold')
        ax.set_ylabel(r"Integrated Autocorrelation ($\tau_{int}$)", weight='bold')
        ax.set_title(f"Sensitivity along {labels[idx]}\nFixed: [{fixed_str}]", fontsize=11)
        if target in ['tau', 'amp', 'pen']: ax.set_xscale('log')
        ax.legend()

    axs[5].axis('off') # El sexto panel queda vacío
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "1D_Hyperparam_Sweeps_TD3.pdf"), dpi=300)

# =====================================================================
# 4. MAIN EJECUTIVO
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".", help="Ruta base donde están los results_massive_a")
    parser.add_argument("--n_evaluations", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    plots_dir = os.path.join(args.base_dir, "results_massive_plots_1D")
    csv_dir = os.path.join(plots_dir, "csv_traces")
    os.makedirs(csv_dir, exist_ok=True)

    # 1. Recolector Inteligente de Directorios TD3
    td3_folders = []
    for root, dirs, files in os.walk(args.base_dir):
        if "params.json" in files and "TD3_actor_weights.pkl" in files:
            td3_folders.append(root)

    if not td3_folders:
        print("[ERROR CRÍTICO] No se encontraron carpetas con modelos TD3 válidos (params.json + pesos pkl).")
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando Evaluación 1D en {len(td3_folders)} modelos TD3 encontrados...")
    
    # 2. Evaluación Múltiple de todos los modelos del espacio de hiperparámetros
    all_results = []
    for idx, folder in enumerate(td3_folders, 1):
        print(f" -> [{idx}/{len(td3_folders)}] Evaluando: {os.path.basename(folder)}")
        try:
            res = evaluate_td3_model(folder, args.n_evaluations, args.n_samples, args.n_ter, csv_dir)
            all_results.append(res)
        except Exception as e:
            print(f"    ! Error ignorado en {os.path.basename(folder)}: {e}")

    df_results = pd.DataFrame(all_results)
    
    # Consolidar runs duplicados si la misma configuración de hiperparámetros fue probada en carpetas distintas
    df_grouped = df_results.groupby(['tau', 'amp', 'pen', 'sig', 'jmp']).agg({
        'Model_ID': 'first', 'Tau_mean': 'mean', 'Tau_err': lambda x: np.sqrt(np.sum(x**2))/len(x),
        'E_mean': 'mean', 'E_err': lambda x: np.sqrt(np.sum(x**2))/len(x)
    }).reset_index()

    summary_csv = os.path.join(plots_dir, "1d_evaluation_summary.csv")
    df_grouped.to_csv(summary_csv, index=False)
    
    print(f"\n[{time.strftime('%H:%M:%S')}] Extrayendo Hiper-Cortes y Graficando...")
    plot_1d_sweeps(df_grouped, plots_dir)
    print(f"[{time.strftime('%H:%M:%S')}] ¡Misión Cumplida! Gráficas en {plots_dir}/1D_Hyperparam_Sweeps_TD3.pdf")

if __name__ == "__main__":
    main()