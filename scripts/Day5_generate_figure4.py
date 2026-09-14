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
from src.td3_modules import FlaxTD3Actor, build_td3_state, FlaxFlatActor
from src.local_energy import local_energies

# ==============================================================================
# FUNCIONES DE SEGURIDAD, MUESTREO Y FÍSICA
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
    
    def sum_log(w): return jnp.sum(ansatz_apply(p_psi, w))
    init_cache = {"log_psi_curr": ansatz_apply(p_psi, w_start), "grad_curr": jnp.clip(jax.grad(sum_log)(w_start), -50.0, 50.0)}
    final_carry, hist_w = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, hist_w

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, subsample=1, chunk_size=1000):
    hist_w_list = []
    curr_w, curr_k = w_start, k_start
    final_carry = (curr_w, curr_k, None) 
    if total_steps == 0: return final_carry, np.array([])
    chunk_size = min(chunk_size, total_steps)
    chunk_size = max(subsample, (chunk_size // subsample) * subsample)
    if chunk_size > 0:
        for _ in range(total_steps // chunk_size):
            final_carry, h_w = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
            curr_w, curr_k, _ = final_carry
            hist_w_list.append(np.asarray(h_w[::subsample]))
    arr_w = np.concatenate(hist_w_list, axis=0) if hist_w_list else np.array([])
    return final_carry, arr_w

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=20000):
    if len(flat_w) == 0: return np.array([])
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

# ==============================================================================
# SCRIPT PRINCIPAL DE ABLACIÓN (FIGURE 4)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./results_Ablation_N4_g5_HuzielReward_0tertrain")
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--n_subsamples", type=int, default=1)
    parser.add_argument("--n_ter", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--clip_grads", type=float, default=20.0)
    args = parser.parse_args()

    dir_full = os.path.join(args.base_dir, "TD3_full")
    if not os.path.exists(dir_full):
        print(f"Error: No se encontró {dir_full}.")
        return

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
    models = ["MALA", "TD3_flat", "TD3_scale_only", "TD3_full"]
    labels = ["Standard\nMALA", "Flat MLP\n(No DeepSet)", "Scale-Only\n(No Dir. Shift)", "Proposed\nRL-MALA"]

    print("="*60)
    print(f" INICIANDO EVALUACIÓN DE ABLACIÓN (N={cfg['n_elec']}, g={cfg['interaction_strength']})")
    print("="*60)

    for mod in models:
        print(f"\n---> Evaluando {mod} ...")
        mod_dir = os.path.join(args.base_dir, mod)
        
        if mod == "MALA":
            sampler = MALASampler(ansatz.apply, cfg["batch_size"], args.tau, system, args.clip_grads, -args.clip_grads, 50.0, -50.0, -50.0, -50.0, 50.0)
            p_act = None
        else:
            with open(os.path.join(mod_dir, "TD3_actor_weights.pkl"), "rb") as f: p_act = pickle.load(f)
            act_type = mod.replace("TD3_", "")
            
            if act_type == "flat":
                td3_model = FlaxFlatActor(actor_amplitude=cfg["actor_amplitude"])
                def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                    w_1d, grad_1d = w[..., 0], grad[..., 0] 
                    target_w = jax.lax.dynamic_slice_in_dim(w_1d, e_idx, 1, axis=-1)
                    target_g = jax.lax.dynamic_slice_in_dim(grad_1d, e_idx, 1, axis=-1)
                    state = jnp.concatenate([target_w, target_g, w_1d, grad_1d], axis=-1)
                    action = td3_model.apply(p_a, state)
                    drift_res = action[:, 0:1] * cfg["actor_amplitude"]
                    return drift_res, action[:, 1:2], {}
            elif act_type == "scale_only":
                td3_model = FlaxTD3Actor(n_up=cfg["n_up"], n_dn=cfg["n_dn"], actor_amplitude=cfg["actor_amplitude"])
                def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                    w_1d, grad_1d = w[..., 0], grad[..., 0] 
                    state = build_td3_state(w_1d, grad_1d, e_idx, cfg["n_elec"], cfg["n_up"], cfg["n_dn"])
                    action = td3_model.apply(p_a, state)
                    drift_res = jnp.zeros_like(action[:, 0:1])
                    return drift_res, action[:, 1:2], {}
            else: # full
                td3_model = FlaxTD3Actor(n_up=cfg["n_up"], n_dn=cfg["n_dn"], actor_amplitude=cfg["actor_amplitude"])
                def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
                    w_1d, grad_1d = w[..., 0], grad[..., 0] 
                    state = build_td3_state(w_1d, grad_1d, e_idx, cfg["n_elec"], cfg["n_up"], cfg["n_dn"])
                    action = td3_model.apply(p_a, state)
                    drift_res = action[:, 0:1] * cfg["actor_amplitude"]
                    return drift_res, action[:, 1:2], {}
            
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
        carry, _ = collect_samples_safe(sampler, w_init, key, p_ansatz_master, p_act, args.n_ter)
        
        t0 = time.time()
        _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz_master, p_act, args.n_samples, subsample=args.n_subsamples)
        t_total = time.time() - t0
        
        traj_w0 = hist_w[:, 0, :, 0]
        eloc_w0 = get_eloc_safe(p_ansatz_master, traj_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        tau = calculate_iat(autocorr(eloc_w0)[:2000])
        
        total_samples = len(hist_w) * cfg["batch_size"]
        neff = total_samples / max(tau, 1.0)
        ess_s = neff / max(t_total, 1e-6)
        
        results[mod] = ess_s
        print(f"     Tau_int = {tau:.2f}")
        print(f"     ESS/s   = {ess_s:,.2f}")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(10, 7))
    colors = ['#7f8c8d', '#e74c3c', '#c0392b', '#2c3e50']
    bar_heights = [results[m] for m in models]
    bars = plt.bar(labels, bar_heights, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.title(fr"Figure 4: Algorithmic Ablation Study ($N={cfg['n_elec']}, g={cfg['interaction_strength']}$)", weight='bold', pad=20)
    plt.ylabel("Effective Sample Size / sec (ESS/s)", weight='bold')
    plt.yscale('log')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.15, f'{int(yval):,}', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.ylim(top=max(bar_heights) * 5)
    plt.tight_layout()
    out_path = os.path.join(args.base_dir, "Figure4_Ablation.png")
    plt.savefig(out_path, dpi=300)
    print("\n" + "="*60)
    print(f"Figura 4 generada exitosamente en: {out_path}")
    print("="*60)

if __name__ == "__main__":
    main()