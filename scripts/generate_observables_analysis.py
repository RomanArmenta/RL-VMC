import os
import sys
import glob
import json
import pickle
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

def extract_observables(dir_path, args, is_td3=False):
    with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega = cfg["omega"]
    g_val = cfg.get("interaction_strength", cfg.get("g"))
    
    weights_name = "TD3_ansatz_weights.pkl" if is_td3 else "MALA_ansatz_weights.pkl"
    ansatz_path = os.path.join(dir_path, weights_name)
    with open(ansatz_path, "rb") as f: p_ansatz = pickle.load(f)
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

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    w_init = jax.random.normal(k1, (cfg["batch_size"], n_elec, 1))
    
    carry, _ = collect_samples_safe(sampler, w_init, k2, p_ansatz, p_act, args.n_ter)
    _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, p_act, args.n_samples)
    
    # 1. Energia y Autocorrelación
    eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    acf = autocorr(eloc_w0)
    
    # 2. Varianza Móvil
    window = min(1000, len(eloc_w0)//10)
    rolling_var = np.array([np.var(eloc_w0[i:i+window]) for i in range(len(eloc_w0)-window)])
    
    # 3. Densidad (1D)
    flat_traj = hist_w[..., 0].reshape(-1) # shape: (steps * batch * N_elec)
    
    # 4. Distribución de pares
    traj_N = hist_w[..., 0].reshape(-1, n_elec)
    i_idx, j_idx = np.triu_indices(n_elec, 1)
    pair_distances = np.abs(traj_N[:, i_idx] - traj_N[:, j_idx]).flatten()
    
    return {
        "acf": acf[:2500],
        "rolling_var": rolling_var,
        "density": flat_traj,
        "pair_distances": pair_distances,
        "label": "RL-MALA (TD3)" if is_td3 else f"Optimized MALA (tau={cfg['tau']})"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mala_dir", type=str, required=True)
    parser.add_argument("--td3_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=15000)
    parser.add_argument("--n_ter", type=int, default=2000)
    args = parser.parse_args()

    print(f"Extrayendo observables de la línea base MALA...")
    res_mala = extract_observables(args.mala_dir, args, is_td3=False)
    
    print(f"Extrayendo observables del modelo ganador RL-MALA...")
    res_td3 = extract_observables(args.td3_dir, args, is_td3=True)

    # Plotting
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Deep Statistical Scan", weight='bold', fontsize=16)

    # Panel 1: Función de Autocorrelación (ACF)
    axs[0, 0].plot(res_mala["acf"], color='#7f8c8d', linewidth=2, label=res_mala["label"])
    axs[0, 0].plot(res_td3["acf"], color='#2c3e50', linewidth=2, label=res_td3["label"])
    axs[0, 0].axhline(0.05, color='r', linestyle='--', alpha=0.5, label='Sokal Threshold')
    axs[0, 0].set_title("Energy Autocorrelation Function (ACF)", weight='bold')
    axs[0, 0].set_xlabel("Lag (steps)")
    axs[0, 0].set_ylabel(r"$\rho(t)$")
    axs[0, 0].set_xlim(0, 2500)
    axs[0, 0].legend()

    # Panel 2: Varianza Móvil de la Energía
    axs[0, 1].plot(res_mala["rolling_var"], color='#7f8c8d', alpha=0.8, label=res_mala["label"])
    axs[0, 1].plot(res_td3["rolling_var"], color='#2c3e50', alpha=0.8, label=res_td3["label"])
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title("Local Energy Rolling Variance", weight='bold')
    axs[0, 1].set_xlabel("MCMC Step")
    axs[0, 1].set_ylabel(r"$Var(E_L)$")

    # Panel 3: Perfil de Densidad Unidimensional
    sns.kdeplot(res_mala["density"], ax=axs[1, 0], color='#7f8c8d', fill=True, alpha=0.3, label=res_mala["label"])
    sns.kdeplot(res_td3["density"], ax=axs[1, 0], color='#2c3e50', fill=True, alpha=0.3, label=res_td3["label"])
    axs[1, 0].set_title(r"One-Body Density $\rho(x)$", weight='bold')
    axs[1, 0].set_xlabel("Position (x)")
    axs[1, 0].set_ylabel("Density")
    axs[1, 0].set_xlim(-6, 6)

    # Panel 4: Distribución de Pares (Hueco de Fermi/Coulomb)
    sns.kdeplot(res_mala["pair_distances"], ax=axs[1, 1], color='#7f8c8d', linewidth=2, label=res_mala["label"])
    sns.kdeplot(res_td3["pair_distances"], ax=axs[1, 1], color='#2c3e50', linewidth=2, label=res_td3["label"])
    axs[1, 1].set_title(r"Pairwise Distance Distribution $P(r_{ij})$", weight='bold')
    axs[1, 1].set_xlabel(r"Distance $r_{ij}$")
    axs[1, 1].set_ylabel("Density")
    axs[1, 1].set_xlim(0, 8)

    plt.tight_layout()
    out_file = os.path.join(args.out_dir, "Winning_Config_Observables.pdf")
    plt.savefig(out_file, dpi=300, format='pdf')
    print(f"Escaneo de observables generado: {out_file}")

    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()