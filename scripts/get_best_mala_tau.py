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
from functools import partial

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import MALASampler
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.local_energy import local_energies

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / result[result.size // 2]

def calculate_iat(acf_array, c=5.0):
    tau = 1.0
    for t, rho in enumerate(acf_array[1:], 1):
        if rho < 0.05 or t >= c * tau: 
            break
        tau += 2.0 * rho
    return tau

@partial(jax.jit, static_argnums=(0, 4))
def _collect_chunk(sampler_obj, w_start, k_start, p_psi, chunk_size):
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        w_new, _, new_cache, _ = sampler_obj.step(subkey, w, {"ansatz": p_psi}, cache, exploration_mode=False)
        return (w_new, k, new_cache), w_new
    
    sum_log = lambda w: jnp.sum(ansatz_apply(p_psi, w))
    init_cache = {"log_psi_curr": ansatz_apply(p_psi, w_start), "grad_curr": jnp.clip(jax.grad(sum_log)(w_start), -50.0, 50.0)}
    final_carry, hist_w = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, hist_w

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, total_steps, chunk_size=1000):
    hist_w_list = []
    curr_w, curr_k = w_start, k_start
    final_carry = (curr_w, curr_k, None) 
    if total_steps == 0: return final_carry, np.array([])
    chunk_size = min(chunk_size, total_steps)
    if chunk_size > 0:
        for _ in range(total_steps // chunk_size):
            final_carry, h_w = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, chunk_size)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_ter", type=int, default=1000)
    args = parser.parse_args()

    mala_dirs = [d for d in glob.glob(os.path.join(args.sys_dir, "MALA_tau*")) if os.path.isdir(d)]
    best_tau = 0.01
    best_ess = -1

    for dir_path in mala_dirs:
        try:
            with open(os.path.join(dir_path, "params.json"), "r") as f: cfg = json.load(f)
            with open(os.path.join(dir_path, "MALA_ansatz_weights.pkl"), "rb") as f: p_ansatz = pickle.load(f)
            
            n_elec, n_up, n_dn = cfg["n_elec"], cfg["n_up"], cfg["n_dn"]
            omega, g_val = cfg["omega"], cfg.get("interaction_strength", cfg.get("g"))
            tau_val = cfg["tau"]

            system = Harmonic_1D(name="Harm", n_elec=n_elec, n_up=n_up, n_dn=n_dn, coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=omega, interaction_strength=g_val, softening=cfg["softening"], dim=1)
            sys_cfg = SystemConfig(n_electrons=n_elec, n_up=n_up, omega=omega, interaction_strength=g_val, softening=cfg["softening"])
            wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
            orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"], backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], backflow_envelope_length=cfg["backflow_envelope_length"])
            correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
            
            global ansatz_apply
            ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
            ansatz_apply = ansatz.apply

            sampler = MALASampler(ansatz.apply, cfg["batch_size"], tau_val, system, 20.0, -20.0, 50.0, -50.0, -50.0, -50.0, 50.0)

            key = jax.random.PRNGKey(42)
            w_init = jax.random.normal(key, (cfg["batch_size"], n_elec, 1))
            carry, _ = collect_samples_safe(sampler, w_init, key, p_ansatz, args.n_ter)
            
            t0 = time.time()
            _, hist_w = collect_samples_safe(sampler, carry[0], carry[1], p_ansatz, args.n_samples)
            t_total = time.time() - t0
            
            eloc_w0 = get_eloc_safe(p_ansatz, hist_w[:, 0, :, 0], sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
            
            tau_sokal = calculate_iat(autocorr(eloc_w0))
            ess_s = ((len(hist_w) * cfg["batch_size"]) / max(tau_sokal, 1.0)) / max(t_total, 1e-6)
            
            if ess_s > best_ess:
                best_ess = ess_s
                best_tau = tau_val
        except Exception as e:
            print(f"Error procesando {dir_path}: {e}", file=sys.stderr)

    print(best_tau, flush=True)
    os._exit(0)

if __name__ == "__main__":
    main()