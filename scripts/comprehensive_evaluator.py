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

# Agregar el directorio raíz al path para poder importar src
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
# 1. FUNCIONES ESTADÍSTICAS Y MCMC
# =====================================================================
def autocorr(x):
    if np.isnan(x).any() or np.var(x) == 0:
        return np.ones(len(x))
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
# 2. MOTOR DE EVALUACIÓN INDEPENDIENTE
# =====================================================================
def run_full_evaluation(dir_path, n_eval, n_samples, n_ter, is_td3=False):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["omega"], cfg.get("interaction_strength", cfg.get("g"))
    
    weights_prefix = "TD3" if is_td3 else "MALA"
    w_path = os.path.join(dir_path, f"{weights_prefix}_ansatz_weights.pkl")
    if not os.path.exists(w_path): w_path = os.path.join(dir_path, "MALA_ansatz_weights.pkl")
    with open(w_path, "rb") as f: p_ansatz = pickle.load(f)
    p_act = None

    # Etiquetado de Hiperparámetros Dinámico
    if is_td3:
        label = f"RL-MALA (TD3)\n$\\tau={cfg['tau']}, Amp={cfg.get('actor_amplitude',0)}, Pen={cfg.get('penalty_factor',0)}, Sig={cfg.get('sigma_action_scale',0)}, Jmp={cfg.get('max_phys_jump_factor',5.0)}$"
    else:
        label = f"Optimized MALA ($\\tau={cfg['tau']}$)"
    
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
        actor_type = cfg.get("actor_type", "full")
        actor_amp = cfg.get("actor_amplitude", 0.05)
        
        if actor_type == "flat":
            from src.td3_modules import FlaxFlatActor
            td3_model = FlaxFlatActor(actor_amplitude=actor_amp)
            def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                w_1d, grad_1d = w[..., 0], grad[..., 0] 
                target_w = jax.lax.dynamic_slice_in_dim(w_1d, e_idx, 1, axis=-1)
                target_g = jax.lax.dynamic_slice_in_dim(grad_1d, e_idx, 1, axis=-1)
                state = jnp.concatenate([target_w, target_g, w_1d, grad_1d], axis=-1)
                act = td3_model.apply(p_a, state)
                return act[:, 0:1]*actor_amp, act[:, 1:2], {}
        else:
            from src.td3_modules import FlaxTD3Actor, build_td3_state
            td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp)
            def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                st = build_td3_state(w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn, quantum_force_norm=cfg.get("force_scale", 5.0), particle_factor=cfg.get("n_scale", 10.0), dist_factor=cfg.get("r_scale", 5.0), omega=omega, gs=g_val, softening=cfg["softening"])
                act = td3_model.apply(p_a, st)
                drift = jnp.zeros_like(act[:, 0:1]) if actor_type == "scale_only" else act[:, 0:1]*actor_amp
                return drift, act[:, 1:2], {}

        sampler = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=cfg.get("penalty_factor", 0.1), max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-50.0, max_clip_grads_ansatz=50.0)
    else:
        sampler = MALASampler(ansatz.apply, cfg["batch_size"], cfg["tau"], system, 50.0, -50.0, 50.0, -50.0, -50.0, -50.0, 50.0)

    # Ciclo de Evaluaciones Independientes
    eval_keys = jax.random.split(jax.random.PRNGKey(cfg.get("seed", 42)), n_eval)
    all_ess, all_tau, all_E_means = [], [], []
    
    # Variables para guardar observalbes de la última evaluación
    last_hist_w, last_eloc = None, None

    for i in range(n_eval):
        k1, k2 = jax.random.split(eval_keys[i])
        w_init = jax.random.normal(k1, (cfg["batch_size"], n_elec, 1))
        
        carry, _ = collect_samples_safe(sampler, w_init, k2, p_ansatz, p_act, n_ter)
        t0 = time.time()
        _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, n_samples)
        t_total = max(time.time() - t0, 1e-6)
        
        eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        
        tau_int = calculate_iat(autocorr(eloc_w0))
        ess_s = ((len(hist_w) * cfg["batch_size"]) / max(tau_int, 1.0)) / t_total
        
        all_ess.append(ess_s)
        all_tau.append(tau_int)
        all_E_means.append(float(np.mean(eloc_w0)))
        
        last_hist_w = hist_w
        last_eloc = eloc_w0

    # Procesar observables (Usando la última evaluación)
    acf_array = autocorr(last_eloc)
    rolling_var = pd.Series(last_eloc).rolling(window=100).var().values
    
    # Distribuciones Físicas
    flat_w = last_hist_w.reshape(-1, n_elec, 1)
    downsampled_w = flat_w[::max(1, len(flat_w)//50000)]
    density_1body = downsampled_w.flatten()
    
    coords = downsampled_w[..., 0]
    diffs = coords[:, :, np.newaxis] - coords[:, np.newaxis, :]
    distances = np.abs(diffs)
    idx = np.triu_indices(n_elec, k=1)
    pair_distances = distances[:, idx[0], idx[1]].flatten()

    # Extraer el Trace (Primeros 2000 pasos, Walker 0, Electron 0)
    trace = last_hist_w[:min(2000, n_samples), 0, 0, 0]

    return {
        "dir": dir_path, "label": label, "cfg": cfg,
        "ess_mean": np.mean(all_ess), "ess_std": np.std(all_ess),
        "tau_mean": np.mean(all_tau),
        "E_mean": np.mean(all_E_means), "E_std": np.std(all_E_means), # Para barra de error
        "acf": acf_array, "rolling_var": rolling_var,
        "density": density_1body, "pair_distances": pair_distances,
        "trace": trace
    }

# =====================================================================
# 3. GENERADORES DE FIGURAS
# =====================================================================
def plot_fig4(res_mala, res_td3, out_dir, n_eval, n_ter):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = [res_mala["label"], res_td3["label"]]
    means = [res_mala["ess_mean"], res_td3["ess_mean"]]
    stds = [res_mala["ess_std"], res_td3["ess_std"]]
    colors = ['#7f8c8d', '#2c3e50']
    
    bars = ax.bar(labels, means, yerr=stds, capsize=10, color=colors, edgecolor='black', linewidth=1.5, error_kw={'elinewidth': 2})
    ax.set_ylabel("Effective Sample Size / sec (ESS/s)", weight='bold')
    ax.set_title(f"Computational Efficiency (N={res_mala['cfg']['n_elec']}, g={res_mala['cfg']['interaction_strength']})\nEvaluations: {n_eval} | Thermalization: {n_ter} steps", weight='bold', pad=20)
    
    if max(means) > 1000:
        ax.set_yscale('log')
        
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval * 1.1, f'{int(yval):,}', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Final_Fig4_Efficiency.pdf"), dpi=300, format='pdf')

def plot_training_metrics(res_mala, res_td3, out_dir):
    sns.set_theme(style="whitegrid", context="paper")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Training Metrics Synthesis: MALA vs RL-MALA", weight='bold', fontsize=16)

    # 1. MALA
    csv_mala = os.path.join(res_mala["dir"], "training_metrics.csv")
    if os.path.exists(csv_mala):
        df_m = pd.read_csv(csv_mala)
        axs[0, 0].plot(df_m['Step'], df_m['E_mean'], color='#7f8c8d', linewidth=2, alpha=0.9, label="MALA")
        if 'acceptance' in df_m.columns:
            axs[0, 1].plot(df_m['Step'], df_m['acceptance'], color='#7f8c8d', linewidth=2, alpha=0.9, label="MALA")

    # 2. TD3
    csv_td3 = os.path.join(res_td3["dir"], "training_metrics.csv")
    if os.path.exists(csv_td3):
        df_t = pd.read_csv(csv_td3)
        axs[0, 0].plot(df_t['Step'], df_t['E_mean'], color='#2c3e50', linewidth=2, label="RL-MALA")
        if 'acceptance' in df_t.columns:
            axs[0, 1].plot(df_t['Step'], df_t['acceptance'], color='#2c3e50', linewidth=2, label="RL-MALA")
        
        if 'sig_scale' in df_t.columns and 'd_act' in df_t.columns:
            axs[1, 0].plot(df_t['Step'], df_t['sig_scale'], color='#3498db', linewidth=2, label="Var Scale")
            axs[1, 0].plot(df_t['Step'], df_t['d_act'], color='#27ae60', linewidth=2, label="Drift Mag")
            
        # Panel 4: Recompensas Separadas (CORREGIDO)
        if 'mean_reward' in df_t.columns:
            l1 = axs[1, 1].plot(df_t['Step'], df_t['mean_reward'], color='purple', linewidth=2, label="Total Reward")
            lns = l1
            
            if 'reward_dist' in df_t.columns:
                l2 = axs[1, 1].plot(df_t['Step'], df_t['reward_dist'], color='green', linestyle='--', alpha=0.7, label="Exploration")
                lns += l2
                
            # Soporte para versiones nuevas y antiguas del nombre de la penalización
            if 'reward_entropy' in df_t.columns:
                l3 = axs[1, 1].plot(df_t['Step'], -df_t['reward_entropy'], color='red', linestyle=':', alpha=0.7, label="Penalty (Entropy)")
                lns += l3
            elif 'reward_penalty' in df_t.columns:
                l3 = axs[1, 1].plot(df_t['Step'], -df_t['reward_penalty'], color='red', linestyle=':', alpha=0.7, label="Penalty (Force)")
                lns += l3

            axs[1, 1].axhline(0, color='gray', linestyle='-')
            
            ax_loss = axs[1, 1].twinx()
            if 'Actor_Loss' in df_t.columns:
                l4 = ax_loss.plot(df_t['Step'], df_t['Actor_Loss'], color='orange', alpha=0.5, label="Actor Loss")
                ax_loss.set_yscale('symlog')
                ax_loss.set_ylabel("Loss", color='orange')
                lns += l4
                
            labs = [l.get_label() for l in lns]
            axs[1, 1].legend(lns, labs, loc='center right')

    axs[0, 0].set_title("VMC Energy Convergence", weight='bold'); axs[0, 0].set_ylabel("Energy (Ha)")
    # Añadir caja de texto con energía final
    textstr = f"Final Energies (Evaluations):\nTD3: {res_td3['E_mean']:.3f} $\pm$ {res_td3['E_std']:.3f}\nMALA: {res_mala['E_mean']:.3f} $\pm$ {res_mala['E_std']:.3f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    axs[0, 0].text(0.95, 0.95, textstr, transform=axs[0, 0].transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    axs[0, 1].set_title("MCMC Acceptance Rate", weight='bold'); axs[0, 1].set_ylabel("Acceptance"); axs[0, 1].legend()
    axs[1, 0].set_title("Actor Learned Actions", weight='bold'); axs[1, 0].legend()
    axs[1, 1].set_title("RL Optimization Stats", weight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Final_Training_Metrics.pdf"), dpi=300, format='pdf')

def plot_observables(res_mala, res_td3, out_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Winning Config Observables\n{res_td3['label']}", weight='bold', fontsize=14)

    axs[0, 0].plot(res_mala["acf"], color='#7f8c8d', linewidth=2, label="MALA")
    axs[0, 0].plot(res_td3["acf"], color='#2c3e50', linewidth=2, label="TD3")
    axs[0, 0].axhline(0.05, color='r', linestyle='--', alpha=0.5, label='Sokal Threshold')
    axs[0, 0].set_title("Energy Autocorrelation Function (ACF)", weight='bold'); axs[0, 0].set_xlim(0, 500); axs[0, 0].set_ylim(-0.2, 0.4); axs[0, 0].legend()

    axs[0, 1].plot(res_mala["rolling_var"], color='#7f8c8d', alpha=0.8, label="MALA")
    axs[0, 1].plot(res_td3["rolling_var"], color='#2c3e50', alpha=0.8, label="TD3")
    axs[0, 1].set_yscale('log'); axs[0, 1].set_title("Local Energy Rolling Variance", weight='bold')

    sns.kdeplot(res_mala["density"], ax=axs[1, 0], color='#7f8c8d', fill=True, alpha=0.3, label="MALA")
    sns.kdeplot(res_td3["density"], ax=axs[1, 0], color='#2c3e50', fill=True, alpha=0.3, label="TD3")
    axs[1, 0].set_title(r"One-Body Density $\rho(x)$", weight='bold')
    min_x, max_x = np.percentile(res_td3["density"], 1) - 1.0, np.percentile(res_td3["density"], 99) + 1.0
    axs[1, 0].set_xlim(min_x, max_x)

    sns.kdeplot(res_mala["pair_distances"], ax=axs[1, 1], color='#7f8c8d', linewidth=2, label="MALA")
    sns.kdeplot(res_td3["pair_distances"], ax=axs[1, 1], color='#2c3e50', linewidth=2, label="TD3")
    axs[1, 1].set_title(r"Pairwise Distance Distribution $P(r_{ij})$", weight='bold')
    axs[1, 1].set_xlim(0, np.percentile(res_td3["pair_distances"], 99) + 0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Final_Observables.pdf"), dpi=300, format='pdf')

def plot_trace(res_mala, res_td3, out_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(res_mala["trace"], color='#7f8c8d', alpha=0.8, linewidth=1.5, label="MALA")
    ax.plot(res_td3["trace"], color='#2c3e50', alpha=0.8, linewidth=1.5, label="RL-MALA")
    
    ax.set_title(f"Walker Trajectory (Nodal Tunneling Analysis)\n{res_td3['label']}", weight='bold')
    ax.set_xlabel("MCMC Step (First 2000 steps)")
    ax.set_ylabel("Electron Position (x)")
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Final_Trace_Plot.pdf"), dpi=300, format='pdf')

# =====================================================================
# 4. MAIN & PARSEO DE ARGUMENTOS
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None, help="Si se proporciona, auto-detecta el mejor MALA y TD3")
    parser.add_argument("--mala_dir", type=str, default=None, help="Carpeta explícita de MALA")
    parser.add_argument("--td3_dir", type=str, default=None, help="Carpeta explícita de TD3")
    parser.add_argument("--n_evaluations", type=int, default=3)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    mala_dir = args.mala_dir
    td3_dir = args.td3_dir
    out_dir = args.base_dir if args.base_dir else "."

    if args.base_dir:
        # Resolver automáticamente los mejores modelos
        best_td3_path = os.path.join(args.base_dir, "best_td3_model.txt")
        if os.path.exists(best_td3_path):
            with open(best_td3_path, "r") as f: td3_name = f.read().strip()
            td3_dir = os.path.join(args.base_dir, td3_name)
        else:
            # Fallback a Symlink
            td3_dir = os.path.join(args.base_dir, "TD3")
            
        stats_path = os.path.join(args.base_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f: s = json.load(f)
            mala_dir = os.path.join(args.base_dir, "MALA_Baselines", f"MALA_tau{s['MALA_best_tau']}")
        else:
            # Fallback: buscar el que tenga mayores ESS si no hay json o agarrar el symlink
            mala_syms = glob.glob(os.path.join(args.base_dir, "MALA_tau*"))
            if mala_syms: mala_dir = mala_syms[0]

    if not mala_dir or not td3_dir or not os.path.exists(mala_dir) or not os.path.exists(td3_dir):
        print("ERROR: No se pudieron encontrar las carpetas MALA y TD3. Verifica las rutas.")
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] Iniciando Evaluación Comprensiva...")
    print(f" -> MALA: {mala_dir}\n -> TD3: {td3_dir}")
    print(f" -> Evaluaciones: {args.n_evaluations} | Samples: {args.n_samples} | Ter: {args.n_ter}")

    res_mala = run_full_evaluation(mala_dir, args.n_evaluations, args.n_samples, args.n_ter, is_td3=False)
    res_td3 = run_full_evaluation(td3_dir, args.n_evaluations, args.n_samples, args.n_ter, is_td3=True)

    print(f"[{time.strftime('%H:%M:%S')}] Evaluaciones terminadas. Generando Figuras...")
    plot_fig4(res_mala, res_td3, out_dir, args.n_evaluations, args.n_ter)
    plot_training_metrics(res_mala, res_td3, out_dir)
    plot_observables(res_mala, res_td3, out_dir)
    plot_trace(res_mala, res_td3, out_dir)

    print(f"[{time.strftime('%H:%M:%S')}] ¡Todas las figuras guardadas exitosamente en {out_dir}!")

if __name__ == "__main__":
    main()