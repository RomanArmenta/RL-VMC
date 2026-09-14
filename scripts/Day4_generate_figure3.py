import os
import sys
import glob
import re
import pickle
import json
import argparse
import time
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
from src.local_energy import local_energies
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.td3_modules import FlaxTD3Actor, build_td3_state

# --- FUNCIONES CORE (Idénticas a Figure 2 para garantizar rigor) ---
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
    
    # 1. Valor por defecto seguro para evitar UnboundLocalError
    final_carry = (curr_w, curr_k, None) 
    
    if total_steps == 0:
        return final_carry, np.array([])
        
    # 2. Si nos piden menos pasos que el chunk_size, reducimos el chunk_size
    chunk_size = min(chunk_size, total_steps)
    chunk_size = max(subsample, (chunk_size // subsample) * subsample)
    
    # 3. Solo iteramos si el chunk_size es válido
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

# --- EVALUACIÓN DE UN SOLO g ---
def evaluate_single_g(perm_dir, args):
    dir_mala = os.path.join(perm_dir, "MALA")
    dir_td3 = os.path.join(perm_dir, "TD3")

    with open(os.path.join(dir_td3, "params.json"), "r") as f: cfg = json.load(f)
    n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
    omega, g_val = cfg["omega"], cfg["interaction_strength"]
    
    with open(os.path.join(dir_td3, "TD3_ansatz_weights.pkl"), "rb") as f: p_ansatz_td3 = pickle.load(f)
    with open(os.path.join(dir_td3, "TD3_actor_weights.pkl"), "rb") as f: p_actor_td3 = pickle.load(f)

    system = Harmonic_1D(name="Harmonic1D", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
    sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"], backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], backflow_envelope_length=cfg["backflow_envelope_length"])
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    td3_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=cfg["actor_amplitude"])
    def td3_wrapper(p_act, w, e_idx, r_nuc, exp_mode, grad):
        state = build_td3_state(w[..., 0], grad[..., 0], e_idx, n_elec, n_up, n_dn)
        action = td3_model.apply(p_act, state)
        return action[:, 0:1] * cfg["actor_amplitude"], action[:, 1:2], {}

    sampler_mala = MALASampler(ansatz.apply, cfg["batch_size"], cfg["tau"], system, args.max_clip_grads, -args.max_clip_grads, 50.0, -50.0, -50.0, -50.0, 50.0)
    sampler_td3 = RLMALASampler_Huziel(ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=td3_wrapper, tau=cfg["tau"], gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=0.1, max_clip_log_ratio=50.0, min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, min_clip_grads_ansatz=-args.max_clip_grads, max_clip_grads_ansatz=args.max_clip_grads)

    key = jax.random.PRNGKey(42)
    w_init = jax.random.normal(key, (cfg["batch_size"], n_elec, 1))

    # Thermalization (FROZEN ANSATZ: usamos p_ansatz_td3 para MALA también)
    carry_mala, _ = collect_samples_safe(sampler_mala, w_init, key, p_ansatz_td3, None, args.n_ter, chunk_size=1000)
    carry_td3, _ = collect_samples_safe(sampler_td3, w_init, key, p_ansatz_td3, p_actor_td3, args.n_ter, chunk_size=1000)
    
    # Production
    start_time = time.time()
    _, hist_mala = collect_samples_safe(sampler_mala, carry_mala[0], carry_mala[1], p_ansatz_td3, None, args.n_samples, subsample=args.n_subsamples)
    t_mala = time.time() - start_time

    start_time = time.time()
    _, hist_td3 = collect_samples_safe(sampler_td3, carry_td3[0], carry_td3[1], p_ansatz_td3, p_actor_td3, args.n_samples, subsample=args.n_subsamples)
    t_td3 = time.time() - start_time

    traj_mala_w0 = hist_mala[:, 0, :, 0]
    traj_td3_w0 = hist_td3[:, 0, :, 0]
    eloc_mala_w0 = get_eloc_safe(p_ansatz_td3, traj_mala_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3_w0 = get_eloc_safe(p_ansatz_td3, traj_td3_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    tau_mala = calculate_iat(autocorr(eloc_mala_w0)[:2000])
    tau_td3 = calculate_iat(autocorr(eloc_td3_w0)[:2000])

    total_samples = len(hist_mala) * cfg["batch_size"]
    neff_mala = total_samples / max(tau_mala, 1.0)
    neff_td3 = total_samples / max(tau_td3, 1.0)

    ess_t_mala = neff_mala / t_mala
    ess_t_td3 = neff_td3 / t_td3
    G_RL = ess_t_td3 / ess_t_mala

    lambda_val = g_val / np.sqrt(omega)

    return {"g": g_val, "lambda": lambda_val, "omega": omega, "N": n_elec,
            "tau_mala": tau_mala, "tau_td3": tau_td3,
            "ess_s_mala": ess_t_mala, "ess_s_td3": ess_t_td3, "G_RL": G_RL}

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Carpeta base (ej. results_1D_N4_sweep_g...)")
    parser.add_argument("--n_samples", type=int, default=20000, help="Pasos de producción")
    parser.add_argument("--n_subsamples", type=int, default=1, help="Submuestreo")
    parser.add_argument("--n_ter", type=int, default=2000, help="Burn-in")
    parser.add_argument("--max_clip_grads", type=float, default=20.0)
    args = parser.parse_args()

    # Buscar todas las carpetas de permutación en la base_dir
    perm_dirs = glob.glob(os.path.join(args.base_dir, "Perm_g*_Om*_Run*"))
    
    results = []
    print(f"Encontradas {len(perm_dirs)} carpetas de evaluación. Iniciando extracción rigurosa de Ansatz Congelado...")

    for pdir in sorted(perm_dirs):
        match = re.search(r'g([0-9\.]+)_Om([0-9\.]+)', pdir)
        if not match: continue
        g_val = float(match.group(1))
        
        print(f"\n---> Evaluando g = {g_val} ...")
        res = evaluate_single_g(pdir, args)
        results.append(res)
        print(f"     G_RL = {res['G_RL']:.3f}x")

    df = pd.DataFrame(results).sort_values("lambda")
    
    # Guardar resultados en CSV
    csv_path = os.path.join(args.base_dir, "Figure3_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDatos guardados en {csv_path}")

    # --- PLOT FIGURE 3 ---
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    N_val = df['N'].iloc[0]
    
    fig.suptitle(fr"Figure 3: Sampler Efficiency across Interaction Strengths ($N={N_val}$)", weight='bold')

    # Panel A: ESS/s vs Lambda
    ax0 = axs[0]
    ax0.plot(df['lambda'], df['ess_s_mala'], 'k--o', linewidth=2, label="Standard MALA")
    ax0.plot(df['lambda'], df['ess_s_td3'], 'r-s', linewidth=2, markersize=8, label="RL-MALA (TD3)")
    ax0.set_yscale('log')
    ax0.set_xlabel(r"Effective Interaction $\lambda = g/\sqrt{\omega}$")
    ax0.set_ylabel("Effective Sample Size / second (ESS/s)")
    ax0.set_title("Absolute Computational Efficiency")
    ax0.legend()

    # Panel B: G_RL vs Lambda
    ax1 = axs[1]
    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=2)
    ax1.plot(df['lambda'], df['G_RL'], 'b-^', linewidth=2, markersize=9, label=f"N = {N_val}")
    ax1.set_xlabel(r"Effective Interaction $\lambda = g/\sqrt{\omega}$")
    ax1.set_ylabel(r"Relative Efficiency Gain ($G_{RL}$)")
    ax1.set_title("Algorithmic Advantage of Learned Proposals")
    ax1.legend()

    plt.tight_layout()
    out_path = os.path.join(args.base_dir, f"Figure3_LambdaSweep_N{N_val}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Figura 3 generada exitosamente en: {out_path}")

if __name__ == "__main__":
    main()