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

# Importaciones de tu arquitectura
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import RLMALASampler_Huziel, MALASampler
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.local_energy import local_energies

# =====================================================================
# 1. FUNCIONES ESTADÍSTICAS MÚLTIPLES
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
    # ESJD = E[ ||X_{t+1} - X_t||^2 ]
    diffs = hist_w[1:] - hist_w[:-1]
    squared_distances = np.sum(diffs**2, axis=(2, 3)) # Suma sobre electrones y dimensiones
    return float(np.mean(squared_distances))

@partial(jax.jit, static_argnums=(0, 5))
def _collect_chunk(sampler_obj, w_start, k_start, p_psi, p_act, chunk_size):
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        params_dict = {"ansatz": p_psi}
        if p_act is not None: params_dict["actor"] = p_act
        w_new, stats, new_cache, _ = sampler_obj.step(subkey, w, params_dict, cache, exploration_mode=False)
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
    chunk_size = min(chunk_size, total_steps)
    if chunk_size > 0:
        for _ in range(total_steps // chunk_size):
            final_carry, h_w, h_acc = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
            curr_w, curr_k, _ = final_carry
            hist_w_list.append(np.asarray(h_w)); hist_acc_list.append(np.asarray(h_acc))
    return final_carry, np.concatenate(hist_w_list, axis=0), np.concatenate(hist_acc_list, axis=0)

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
# 2. MOTOR DE EVALUACIÓN MASIVA
# =====================================================================
def evaluate_single_model(dir_path, n_eval, n_samples, n_ter, is_td3, csv_out_dir, massive_id):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["omega"], cfg.get("interaction_strength", cfg.get("g", 0.0))
    
    weights_prefix = "TD3" if is_td3 else "MALA"
    w_path = os.path.join(dir_path, f"{weights_prefix}_ansatz_weights.pkl")
    if not os.path.exists(w_path): w_path = os.path.join(dir_path, "MALA_ansatz_weights.pkl")
    with open(w_path, "rb") as f: p_ansatz = pickle.load(f)
    
    system = Harmonic_1D(name="Harm", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg.get("backflow_components", 8), backflow_r_max=cfg.get("backflow_r_max", 5.0), backflow_width=cfg.get("backflow_width", 0.7), backflow_envelope_length=cfg.get("backflow_envelope_length", 3.0))
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    p_act = None
    if is_td3:
        with open(os.path.join(dir_path, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
        actor_amp = cfg.get("actor_amplitude", 0.05)
        from src.td3_modules import FlaxTD3Actor, build_td3_state
        td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp)
        def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
            st = build_td3_state(w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn, quantum_force_norm=cfg.get("force_scale", 5.0), particle_factor=cfg.get("n_scale", 10.0), dist_factor=cfg.get("r_scale", 5.0), omega=omega, gs=g_val, softening=cfg["softening"])
            act = td3_model.apply(p_a, st)
            return act[:, 0:1]*actor_amp, act[:, 1:2], {}
        sampler = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=cfg.get("penalty_factor", 0.1), max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-50.0, max_clip_grads_ansatz=50.0)
    else:
        sampler = MALASampler(ansatz.apply, cfg["batch_size"], cfg["tau"], system, 50.0, -50.0, 50.0, -50.0, -50.0, -50.0, 50.0)

    eval_keys = jax.random.split(jax.random.PRNGKey(cfg.get("seed", 42) + (10 if is_td3 else 0)), n_eval)
    
    metrics = {"E": [], "ESS": [], "Tau": [], "ESJD": [], "Acc": []}
    
    for i in range(n_eval):
        k1, k2 = jax.random.split(eval_keys[i])
        w_init = jax.random.normal(k1, (cfg["batch_size"], n_elec, 1))
        
        # Termalización
        carry, _, _ = collect_samples_safe(sampler, w_init, k2, p_ansatz, p_act, n_ter)
        
        # Producción
        t0 = time.time()
        _, hist_w, hist_acc = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, n_samples)
        t_total = max(time.time() - t0, 1e-6)
        
        eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        
        tau_int = calculate_iat(autocorr(eloc_w0))
        ess_s = ((len(hist_w) * cfg["batch_size"]) / max(tau_int, 1.0)) / t_total
        esjd = compute_esjd(hist_w)
        acc_mean = float(np.mean(hist_acc))
        
        metrics["E"].append(float(np.mean(eloc_w0)))
        metrics["ESS"].append(ess_s)
        metrics["Tau"].append(tau_int)
        metrics["ESJD"].append(esjd)
        metrics["Acc"].append(acc_mean)

        # Exportar CSV por evaluación para trazabilidad (Protección para la dimensión de hist_acc)
        acc_data = hist_acc if hist_acc.ndim == 1 else hist_acc[:, 0]
        csv_filename = os.path.join(csv_out_dir, f"Eval_{weights_prefix}_{massive_id}_N{n_elec}_g{g_val}_run{i+1}.csv")
        pd.DataFrame({
            "Step": np.arange(n_samples), "Local_Energy_W0": eloc_w0, "Acceptance": acc_data,
            "Walker0_Elec0_X": hist_w[:, 0, 0, 0]
        }).to_csv(csv_filename, index=False)

    return {
        "Model_ID": massive_id, "Method": weights_prefix, "N": n_elec, "g": g_val,
        "E_mean": np.mean(metrics["E"]), "E_err": np.std(metrics["E"]) / np.sqrt(n_eval),
        "ESS_mean": np.mean(metrics["ESS"]), "ESS_err": np.std(metrics["ESS"]) / np.sqrt(n_eval),
        "Tau_mean": np.mean(metrics["Tau"]), "Tau_err": np.std(metrics["Tau"]) / np.sqrt(n_eval),
        "ESJD_mean": np.mean(metrics["ESJD"]), "ESJD_err": np.std(metrics["ESJD"]) / np.sqrt(n_eval),
        "Acc_mean": np.mean(metrics["Acc"])
    }

# =====================================================================
# 3. GENERACIÓN DE FIGURAS MASIVAS
# =====================================================================
def generate_massive_plots(df, out_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    df['System_Label'] = df.apply(lambda x: f"{x['Model_ID']}\n(N={x['N']}, g={x['g']})", axis=1)
    
    mala_df = df[df['Method'] == 'MALA'].set_index('System_Label')
    td3_df = df[df['Method'] == 'TD3'].set_index('System_Label')
    systems = mala_df.index.intersection(td3_df.index)

    # Figura 1: Brecha de Energía del Estado Base
    fig, ax = plt.subplots(figsize=(14, 6))
    delta_E = mala_df.loc[systems, 'E_mean'] - td3_df.loc[systems, 'E_mean']
    err_E = np.sqrt(mala_df.loc[systems, 'E_err']**2 + td3_df.loc[systems, 'E_err']**2)
    
    ax.errorbar(systems, delta_E, yerr=err_E, fmt='-o', color='purple', markersize=8, capsize=5, linewidth=2)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel(r"$\Delta E = E_{MALA} - E_{TD3}$ (Ha)", weight='bold')
    ax.set_title("Ground State Energy Gap (Positivo = TD3 encontró mejor estado)", weight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Fig1_Energy_Accuracy.pdf"), dpi=300); plt.close()

    # Figura 2: Eficiencia Computacional Absoluta
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(systems))
    width = 0.35
    
    ax.bar(x - width/2, mala_df.loc[systems, 'ESS_mean'], width, yerr=mala_df.loc[systems, 'ESS_err'], label='MALA', color='#7f8c8d', capsize=5, edgecolor='black')
    ax.bar(x + width/2, td3_df.loc[systems, 'ESS_mean'], width, yerr=td3_df.loc[systems, 'ESS_err'], label='RL-MALA', color='#2c3e50', capsize=5, edgecolor='black')
    
    ax.set_yscale('log')
    ax.set_ylabel("Effective Sample Size / sec", weight='bold')
    ax.set_title("Absolute Computational Scaling across Models", weight='bold')
    ax.set_xticks(x); ax.set_xticklabels(systems, rotation=45)
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Fig2_ESS_Scaling.pdf"), dpi=300); plt.close()

    # Figura 3: Ganancia Algorítmica (G_RL)
    fig, ax = plt.subplots(figsize=(14, 6))
    G_RL = mala_df.loc[systems, 'Tau_mean'] / td3_df.loc[systems, 'Tau_mean']
    
    sns.barplot(x=systems, y=G_RL, ax=ax, palette="viridis", edgecolor='black')
    ax.axhline(1, color='red', linestyle='--', label='Baseline (MALA)')
    ax.set_ylabel(r"Algorithmic Gain $G_{RL} = \tau_{MALA} / \tau_{TD3}$", weight='bold')
    ax.set_title("Neural Proposal Advantage (Topological Unblocking)", weight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Fig3_Algorithmic_Gain.pdf"), dpi=300); plt.close()

    # Figura 4: ESJD vs Autocorrelación (Scatter Dual)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.scatter(mala_df.loc[systems, 'ESJD_mean'], mala_df.loc[systems, 'Tau_mean'], color='#7f8c8d', s=150, label='MALA', edgecolors='black', marker='o')
    ax.scatter(td3_df.loc[systems, 'ESJD_mean'], td3_df.loc[systems, 'Tau_mean'], color='#3498db', s=200, label='RL-MALA', edgecolors='black', marker='*')
    
    ax.set_xlabel("Expected Squared Jump Distance (ESJD)", weight='bold')
    ax.set_ylabel(r"Integrated Autocorrelation Time ($\tau_{int}$)", weight='bold')
    ax.set_title("Phase Space Traversal Efficiency", weight='bold')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "Fig4_Exploration_Metrics.pdf"), dpi=300); plt.close()

# =====================================================================
# 4. MAIN EJECUTIVO
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=".", help="Carpeta superior que contiene los results_massive_a")
    parser.add_argument("--n_evaluations", type=int, default=5, help="Múltiples evaluaciones para barras de error")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    plots_dir = os.path.join(args.base_dir, "results_massive_plots")
    csv_dir = os.path.join(plots_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)

    massive_folders = sorted(glob.glob(os.path.join(args.base_dir, "results_example_N10_g20_*")))
    massive_folders = [f for f in massive_folders if os.path.isdir(f) and "plots" not in f]

    if not massive_folders:
        print("ERROR: No se encontraron carpetas con el prefijo 'results_massive_'")
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando Evaluación Masiva en {len(massive_folders)} modelos...")
    
    all_results = []
    for folder in massive_folders:
        massive_id = os.path.basename(folder)
        print(f"\n-> Procesando: {massive_id}")
        
        # Buscar la subcarpeta de sistema
        sys_dirs = glob.glob(os.path.join(folder, "results_N*_g*"))
        if not sys_dirs:
            print(f"    ! Error: No se encontró subcarpeta de sistema dentro de {folder}")
            continue
        sys_dir = sys_dirs[0]
        
        # ----------------------------------------------------
        # 1. ENCONTRAR TD3 ROBUSTAMENTE
        # ----------------------------------------------------
        td3_dir = os.path.join(sys_dir, "TD3")
        if not os.path.exists(td3_dir) or not os.path.isdir(td3_dir):
            best_td3_txt = os.path.join(sys_dir, "best_td3_model.txt")
            if os.path.exists(best_td3_txt):
                with open(best_td3_txt, "r") as f:
                    td3_name = f.read().strip()
                td3_dir = os.path.join(sys_dir, td3_name)
        
        # ----------------------------------------------------
        # 2. ENCONTRAR MALA ROBUSTAMENTE
        # ----------------------------------------------------
        mala_dirs = []
        for d in glob.glob(os.path.join(sys_dir, "MALA_tau*")):
            if os.path.isdir(d): mala_dirs.append(d)
        for d in glob.glob(os.path.join(sys_dir, "MALA_Baselines", "MALA_tau*")):
            if os.path.isdir(d): mala_dirs.append(d)
            
        mala_dir = None
        stats_path = os.path.join(sys_dir, "stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f: s = json.load(f)
                best_tau = s.get("MALA_best_tau")
                if best_tau:
                    for d in mala_dirs:
                        if f"MALA_tau{best_tau}" in d:
                            mala_dir = d
                            break
            except: pass
            
        if mala_dir is None and mala_dirs:
            mala_dir = mala_dirs[0] # Agarra el primero si no hay info en stats.json
            
        # ----------------------------------------------------
        # 3. EVALUAR
        # ----------------------------------------------------
        if not os.path.exists(td3_dir):
            print(f"    ! Error: No se pudo localizar una carpeta TD3 válida en {sys_dir}")
            continue
        if mala_dir is None or not os.path.exists(mala_dir):
            print(f"    ! Error: No se pudo localizar una carpeta MALA válida en {sys_dir}")
            continue
            
        try:
            print("    Evaluando MALA...")
            res_mala = evaluate_single_model(mala_dir, args.n_evaluations, args.n_samples, args.n_ter, False, csv_dir, massive_id)
            all_results.append(res_mala)
            
            print("    Evaluando TD3...")
            res_td3 = evaluate_single_model(td3_dir, args.n_evaluations, args.n_samples, args.n_ter, True, csv_dir, massive_id)
            all_results.append(res_td3)
            print("    ✅ Evaluación completada.")
        except Exception as e:
            print(f"    ! Excepción al evaluar el modelo en {folder}: {e}")

    df_results = pd.DataFrame(all_results)
    
    if df_results.empty:
        print("\n[ERROR CRÍTICO] Ningún modelo pudo ser evaluado. Revisa la estructura de carpetas.")
        sys.exit(1)
        
    summary_csv = os.path.join(plots_dir, "massive_evaluation_summary.csv")
    df_results.to_csv(summary_csv, index=False)
    print(f"\n[{time.strftime('%H:%M:%S')}] Evaluaciones terminadas. CSV maestro guardado en: {summary_csv}")

    print(f"[{time.strftime('%H:%M:%S')}] Generando figuras maestras...")
    generate_massive_plots(df_results, plots_dir)
    print(f"[{time.strftime('%H:%M:%S')}] ¡Todas las producciones y figuras guardadas en {plots_dir}!")

if __name__ == "__main__":
    main()