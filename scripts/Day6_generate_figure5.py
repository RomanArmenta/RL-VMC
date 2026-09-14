import os
import sys
import json
import pickle
import time
import argparse
import numpy as np
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

# ==============================================================================
# FUNCIONES DE SEGURIDAD Y FÍSICA
# ==============================================================================
def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / result[result.size // 2]

def calculate_iat(acf_array):
    tau = 1.0
    for rho in acf_array[1:]:
        if rho < 0: break
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

def collect_production_chain(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, chunk_size=1000):
    hist_w_list = []
    curr_w, curr_k = w_start, k_start
    final_carry = (curr_w, curr_k, None) 
    
    if total_steps == 0: return final_carry, np.array([])
    chunk_size = min(chunk_size, total_steps)
    
    t0 = time.time()
    for _ in range(total_steps // chunk_size):
        final_carry, h_w = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
        curr_w, curr_k, _ = final_carry
        hist_w_list.append(np.asarray(h_w))
    wall_time = time.time() - t0
    
    arr_w = np.concatenate(hist_w_list, axis=0) if hist_w_list else np.array([])
    return final_carry, arr_w, wall_time

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=10000):
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

# ==============================================================================
# SCRIPT PRINCIPAL: TIME-TO-SOLUTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./results_Ablation_N4_g5_PaperReward")
    parser.add_argument("--n_samples", type=int, default=100000, help="Pasos largos sin thinning")
    parser.add_argument("--n_ter", type=int, default=2000, help="Burn-in independiente")
    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--clip_grads", type=float, default=20.0)
    args = parser.parse_args()

    dir_full = os.path.join(args.base_dir, "TD3_full")
    dir_mala = os.path.join(args.base_dir, "MALA")
    
    with open(os.path.join(dir_full, "params.json"), "r") as f: cfg = json.load(f)
    with open(os.path.join(dir_full, "TD3_ansatz_weights.pkl"), "rb") as f: p_ansatz_master = pickle.load(f)

    system = Harmonic_1D(name="Harm", n_elec=cfg["n_elec"], n_up=cfg["n_up"], n_dn=cfg["n_dn"], coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=cfg["omega"], interaction_strength=cfg["interaction_strength"], softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=cfg["n_elec"], n_up=cfg["n_up"], omega=cfg["omega"], interaction_strength=cfg["interaction_strength"], softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"], backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], backflow_envelope_length=cfg["backflow_envelope_length"])
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    results = {}
    models = ["MALA", "TD3_full"]

    print("="*70)
    print(f" CALCULANDO TIEMPO DE SOLUCIÓN DEFINITIVO (N={cfg['n_elec']}, g={cfg['interaction_strength']})")
    print("="*70)

    for mod in models:
        print(f"\n---> Muestreando {mod} ({args.n_samples} pasos continuos)...")
        if mod == "MALA":
            sampler = MALASampler(ansatz.apply, cfg["batch_size"], args.tau, system, args.clip_grads, -args.clip_grads, 50.0, -50.0, -50.0, -50.0, 50.0)
            p_act = None
        else:
            with open(os.path.join(dir_full, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
            td3_model = FlaxTD3Actor(n_up=cfg["n_up"], n_dn=cfg["n_dn"], actor_amplitude=cfg["actor_amplitude"])
            def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                st = build_td3_state(w[..., 0], grad[..., 0], e_idx, cfg["n_elec"], cfg["n_up"], cfg["n_dn"])
                act = td3_model.apply(p_a, st)
                return act[:, 0:1]*cfg["actor_amplitude"], act[:, 1:2], {}
            
            # CORRECCIÓN DE LA INSTANCIACIÓN AQUÍ
            sampler = RLMALASampler_Huziel(
                ansatz_apply=ansatz.apply, 
                system=system, 
                batch_size=cfg["batch_size"], 
                actor_apply=wrapper, 
                tau=args.tau, 
                gamma=cfg["gamma"], 
                beta=cfg["beta"], 
                kappa=cfg["kappa"], 
                penalty_factor=cfg["penalty_factor"], 
                max_clip_log_ratio=cfg["max_clip_log_ratio"], 
                min_clip_log_ratio=cfg["min_clip_log_ratio"], 
                nan_clip_log_ratio=cfg["nan_clip_log_ratio"], 
                neginf_clip_log_ratio=cfg["neginf_clip_log_ratio"], 
                posinf_clip_log_ratio=cfg["posinf_clip_log_ratio"], 
                scale_clip_min=cfg["scale_clip_min"], 
                scale_clip_max=cfg["scale_clip_max"], 
                min_clip_grads_ansatz=-args.clip_grads, 
                max_clip_grads_ansatz=args.clip_grads
            )

        key = jax.random.PRNGKey(42)
        w_init = jax.random.normal(key, (cfg["batch_size"], cfg["n_elec"], 1))
        
        # 1. Independent Burn-in
        carry, _, _ = collect_production_chain(sampler, w_init, key, p_ansatz_master, p_act, args.n_ter)
        
        # 2. Production Run (No Thinning)
        _, hist_w, wall_time = collect_production_chain(sampler, carry[0], carry[1], p_ansatz_master, p_act, args.n_samples)
        
        # 3. Energy Evaluation
        traj_w0 = hist_w[:, 0, :, 0]
        eloc_w0 = get_eloc_safe(p_ansatz_master, traj_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        
        tau = calculate_iat(autocorr(eloc_w0)[:2000])
        var_e = np.var(eloc_w0)
        
        results[mod] = {
            "tau": tau,
            "var": var_e,
            "time_per_step": wall_time / args.n_samples,
            "batch_size": cfg["batch_size"]
        }
        
        print(f"     Tau_int = {tau:.2f} pasos")
        print(f"     Var(E)  = {var_e:.6f} Eh^2")
        print(f"     Tiempo  = {wall_time:.2f} seg")

    # ==============================================================================
    # CÁLCULOS TEÓRICOS DE SE(E) Y TIEMPO (T_eps)
    # ==============================================================================
    # Target Errors
    epsilons = [1e-3, 5e-4]
    
    print("\n" + "="*70)
    print(" ANÁLISIS DE TIEMPO-A-SOLUCIÓN (T_epsilon)")
    print("="*70)
    
    for eps in epsilons:
        print(f"\nObjetivo: SE(E) < {eps} Hartrees")
        t_eps_mala = (results["MALA"]["var"] * results["MALA"]["tau"] * results["MALA"]["time_per_step"]) / (eps**2 * results["MALA"]["batch_size"])
        t_eps_td3 = (results["TD3_full"]["var"] * results["TD3_full"]["tau"] * results["TD3_full"]["time_per_step"]) / (eps**2 * results["TD3_full"]["batch_size"])
        
        ratio = t_eps_mala / t_eps_td3
        print(f"  T_eps (MALA):     {t_eps_mala:.2f} segundos")
        print(f"  T_eps (RL-MALA):  {t_eps_td3:.2f} segundos")
        print(f"  Aceleración Neta: {ratio:.2f}x más rápido")

    # ==============================================================================
    # RENDERIZAR FIGURA 5
    # ==============================================================================
    steps = np.arange(1, args.n_samples + 1)
    
    # Construcción de curvas
    se_mala_steps = np.sqrt((results["MALA"]["var"] * results["MALA"]["tau"]) / (steps * results["MALA"]["batch_size"]))
    time_mala = steps * results["MALA"]["time_per_step"]
    
    se_td3_steps = np.sqrt((results["TD3_full"]["var"] * results["TD3_full"]["tau"]) / (steps * results["TD3_full"]["batch_size"]))
    time_td3 = steps * results["TD3_full"]["time_per_step"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(fr"Figure 5: Standard Error Convergence Rates ($N={cfg['n_elec']}, g={cfg['interaction_strength']}$)", weight='bold', fontsize=16)

    # Panel A: SE vs Sweeps
    axs[0].plot(steps, se_mala_steps, 'k--', linewidth=2, label=fr"MALA ($\tau={results['MALA']['tau']:.1f}$)")
    axs[0].plot(steps, se_td3_steps, 'r-', linewidth=2, label=fr"RL-MALA ($\tau={results['TD3_full']['tau']:.1f}$)")
    axs[0].axhline(1e-3, color='gray', linestyle=':', label=r"Target $\epsilon=10^{-3}$")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_xlabel("MCMC Sweeps (Production)")
    axs[0].set_ylabel(r"Running Standard Error $\mathrm{SE}(E)$")
    axs[0].set_title("Statistical Convergence (Step Domain)")
    axs[0].legend()

    # Panel B: SE vs Wall Time
    axs[1].plot(time_mala, se_mala_steps, 'k--', linewidth=2, label="Standard MALA")
    axs[1].plot(time_td3, se_td3_steps, 'r-', linewidth=2, label="Proposed RL-MALA")
    axs[1].axhline(1e-3, color='gray', linestyle=':')
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_xlabel("Wall-Clock Time (seconds)")
    axs[1].set_ylabel(r"Running Standard Error $\mathrm{SE}(E)$")
    axs[1].set_title("Absolute Computational Efficiency (Time Domain)")
    axs[1].legend()

    plt.tight_layout()
    out_path = os.path.join(args.base_dir, "Figure5_Time_To_Solution.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nFigura 5 generada exitosamente en: {out_path}")

if __name__ == "__main__":
    main()