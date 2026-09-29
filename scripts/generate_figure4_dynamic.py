import os
import sys
import glob
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
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

def evaluate_model(dir_path, args, is_td3=False):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["interaction_strength"] if "interaction_strength" in cfg else cfg["g"], cfg.get("interaction_strength", cfg.get("g"))
    
    #with open(os.path.join(dir_path, f"{'TD3' if is_td3 else 'MALA'}_ansatz_weights.pkl"), "rb") as f: p_ansatz = pickle.load(f)
    td3_master_path = os.path.join(os.path.dirname(dir_path), "TD3", "TD3_ansatz_weights.pkl")
    with open(td3_master_path, "rb") as f: p_ansatz = pickle.load(f)
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

        def wrapper(p_a, w, e_idx, r_nuc, exp, grad):
            st = build_td3_state(
                w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn,
                quantum_force_norm=cfg.get("force_scale", 5.0),
                particle_factor=cfg.get("n_scale", 10.0),
                dist_factor=cfg.get("r_scale", 5.0),
                omega=omega, gs=g_val, softening=cfg["softening"]
            )
            act = td3_model.apply(p_a, st)
            return act[:, 0:1]*cfg.get("actor_amplitude", 0.15), act[:, 1:2], {}

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
        
        traj_w0 = hist_w[:, 0, :, 0]
        eloc_w0 = get_eloc_safe(p_ansatz, traj_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        tau = calculate_iat(autocorr(eloc_w0)[:2000])
        
        total_samples = len(hist_w) * cfg["batch_size"]
        ess_s = (total_samples / max(tau, 1.0)) / max(t_total, 1e-6)
        
        ess_results.append(ess_s)
        tau_results.append(tau)

    return {"ess_mean": np.mean(ess_results), "ess_std": np.std(ess_results), "tau_mean": np.mean(tau_results), "tau": cfg.get("tau", 0.0), "cfg": cfg}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_dir", type=str, required=True)
    parser.add_argument("--n_evaluations", type=int, default=5)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    # --- LA CORRECCIÓN CLAVE ESTÁ AQUÍ: Ignoramos los archivos .log ---
    mala_dirs = [d for d in glob.glob(os.path.join(args.sys_dir, "MALA_tau*")) if os.path.isdir(d)]
    
    if not mala_dirs:
        print("Aún no hay directorios MALA listos o válidos.")
        return

    best_mala = {"ess_mean": -1, "tau": 0}

    # 1. Encontrar el mejor MALA
    for m_dir in mala_dirs:
        try:
            res = evaluate_model(m_dir, args, is_td3=False)
            if res["ess_mean"] > best_mala["ess_mean"]:
                best_mala = res
        except Exception as e:
            print(f"Saltando evaluación de {m_dir} por error: {e}")

    # 2. Evaluar TD3
    td3_dir = os.path.join(args.sys_dir, "TD3")
    if not os.path.isdir(td3_dir):
        print("El modelo TD3 aún no termina. Saliendo.")
        return

    res_td3 = evaluate_model(td3_dir, args, is_td3=True)

    cfg = res_td3["cfg"]
    g_val = cfg.get("interaction_strength", cfg.get("g"))
    
    stats_data = {
        "N": cfg["n_elec"], "g": g_val, "omega": cfg["omega"],
        "MALA_best_tau": best_mala["tau"], 
        "MALA_ESS_mean": float(best_mala["ess_mean"]), 
        "MALA_tau_int_mean": float(best_mala["tau_mean"]),
        "TD3_ESS_mean": float(res_td3["ess_mean"]), 
        "TD3_tau_int_mean": float(res_td3["tau_mean"])
    }
    
    with open(os.path.join(args.sys_dir, "stats.json"), "w") as f:
        json.dump(stats_data, f, indent=4)

    # 3. Plotear y guardar PDF
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    plt.figure(figsize=(9, 7))
    
    labels = [fr"Opt. MALA ($\tau={best_mala['tau']}$)", "Proposed RL-MALA"]
    means = [best_mala["ess_mean"], res_td3["ess_mean"]]
    stds = [best_mala.get("ess_std", 0), res_td3["ess_std"]]
    
    bars = plt.bar(labels, means, yerr=stds, capsize=10, color=['#7f8c8d', '#2c3e50'], edgecolor='black', linewidth=1.5)
    
    plt.title(fr"Figure 4: Computational Efficiency ($N={cfg['n_elec']}, g={g_val}$)" + "\n" + fr"Evaluations: {args.n_evaluations} | Thermalization: {args.n_ter} steps", weight='bold', pad=15)
    plt.ylabel("Effective Sample Size / sec (ESS/s)", weight='bold')
    plt.yscale('log')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.15, f'{int(yval):,}', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.ylim(top=max(means) * 5)
    plt.tight_layout()
    plt.savefig(os.path.join(args.sys_dir, "Figure4_Efficiency.pdf"), dpi=300, format='pdf')
    plt.close()

if __name__ == "__main__":
    main()