import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================================
# 1. RUTAS Y DEPENDENCIAS
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_VMC_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
sys.path.append(RL_VMC_SRC)
from src.td3_modules import TD3Trainer

def compute_autocorr_time(series):
    n = len(series)
    mu = np.mean(series)
    var = np.var(series)
    if var == 0: return 0.0
    t_max = min(1000, n // 2)
    rho = np.zeros(t_max)
    for t in range(t_max):
        rho[t] = np.mean((series[:n-t] - mu) * (series[t:] - mu)) / var
    tau_int = 0.5
    for t in range(1, t_max):
        if rho[t] < 0.05: break
        tau_int += rho[t]
    return tau_int

# =====================================================================
# 2. FACTOR DE JASTROW NEURONAL (CORRELACIÓN DINÁMICA)
# =====================================================================
class NeuralJastrow(nn.Module):
    hidden_dim: int = 64
    
    @nn.compact
    def __call__(self, x):
        x_flat = x.reshape(-1)
        h = nn.Dense(self.hidden_dim)(x_flat)
        h = nn.tanh(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.tanh(h)
        out = nn.Dense(1)(h)
        return out[0]

# =====================================================================
# 3. ANSATZ TOTAL (CASSCF CONGELADO + JASTROW) Y ENERGÍA LOCAL
# =====================================================================
def build_ho_basis_jax(r, n_basis, omega):
    z = jnp.sqrt(omega) * r
    H = [jnp.ones_like(z), 2.0 * z]
    for n in range(2, n_basis):
        H.append(2.0 * z * H[n-1] - 2.0 * (n - 1) * H[n-2])
    H_stack = jnp.stack(H[:n_basis], axis=0) 
    ns = jnp.arange(n_basis)
    fact = jnp.exp(gammaln(ns + 1.0))
    norm = jnp.sqrt(2.0**ns * fact)
    pref = (omega / jnp.pi)**0.25 * jnp.exp(-0.5 * z**2)
    return (pref * (H_stack / norm[:, None])).T 

def cargar_ansatz_total(npz_path):
    data = np.load(npz_path)
    meta = json.loads(str(data["metadata_json"]))
    cfg = meta["config"]
    omega = cfg["omega"]
    n_up, n_dn = cfg["n_up"], cfg["n_down"]
    n_basis = cfg["n_basis"]
    
    g_int = cfg.get("g", 0.1)
    softening = cfg.get("softening", 0.1)
    
    import re
    match = re.search(r'c(\d+)-a(\d+)', npz_path)
    fallback_core = int(match.group(1)) if match else 0
    fallback_active = int(match.group(2)) if match else 2
    n_core = cfg.get("n_core", cfg.get("n_core_orbitals", fallback_core))
    n_active = cfg.get("n_active_orbitals", cfg.get("n_active", fallback_active))
    
    orbitals = jnp.array(data["orbital_coefficients"])
    ci_coeffs = jnp.array(data["selected_ci_coefficients"])
    dets = data["selected_ci_determinants"]
    
    occ_up_list, occ_dn_list = [], []
    for d_str in dets:
        val = int(d_str)
        act_up = [i for i in range(n_active) if val & (1 << i)]
        act_dn = [i for i in range(n_active) if val & (1 << (n_active + i))]
        occ_up_list.append(list(range(n_core)) + [n_core + i for i in act_up])
        occ_dn_list.append(list(range(n_core)) + [n_core + i for i in act_dn])
        
    occ_up_idx, occ_dn_idx = jnp.array(occ_up_list), jnp.array(occ_dn_list)
    sign_c, log_c = jnp.sign(ci_coeffs), jnp.log(jnp.abs(ci_coeffs) + 1e-15)
    
    def log_psi_casscf(x_1d):
        x_up, x_dn = x_1d[:n_up], x_1d[n_up:]
        basis_up = build_ho_basis_jax(x_up, n_basis, omega)
        basis_dn = build_ho_basis_jax(x_dn, n_basis, omega)
        def eval_det(idx_up, idx_dn, s_c, l_c):
            mat_up = basis_up @ orbitals[:, idx_up] + jnp.eye(n_up) * 1e-12
            mat_dn = basis_dn @ orbitals[:, idx_dn] + jnp.eye(n_dn) * 1e-12
            s_u, l_u = jnp.linalg.slogdet(mat_up)
            s_d, l_d = jnp.linalg.slogdet(mat_dn)
            return s_c * s_u * s_d, l_c + l_u + l_d
        signs, log_mags = jax.vmap(eval_det)(occ_up_idx, occ_dn_idx, sign_c, log_c)
        max_log = jnp.max(log_mags)
        psi_val = jnp.sum(signs * jnp.exp(log_mags - max_log))
        return max_log + jnp.log(jnp.abs(psi_val) + 1e-15)
        
    jastrow_net = NeuralJastrow()
    
    @jax.jit
    def ansatz_apply(params, x):
        x_1d = x[..., 0]
        def evaluate_single(x_single):
            l_casscf = log_psi_casscf(x_single)
            j_val = jastrow_net.apply(params["jastrow"], x_single) if "jastrow" in params else 0.0
            return jnp.clip(l_casscf + j_val, -1000.0, 1000.0)
        return jax.vmap(evaluate_single)(x_1d) if x_1d.ndim == 2 else evaluate_single(x_1d)

    @jax.jit
    def vmc_local_energy(params, x_walkers):
        def single_local_energy(x):
            # 1. Aplanamos x para evitar el bug de traza 4D
            x_1d = x[:, 0]
            
            def log_psi_1d(x_in):
                return ansatz_apply(params, x_in[:, None])
            
            # 2. Gradiente (N,) y Hessiano (N, N)
            grad_psi = jax.grad(log_psi_1d)(x_1d)
            hess_psi = jax.hessian(log_psi_1d)(x_1d)
            
            # 3. Laplaciano real
            lapl_psi = jnp.trace(hess_psi)
            kin = -0.5 * (lapl_psi + jnp.sum(grad_psi**2))
            
            # 4. Cálculo de potenciales con vector 1D
            v_harm = 0.5 * (omega**2) * jnp.sum(x_1d**2)
            n = x_1d.shape[0]
            v_ee = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dist = jnp.sqrt((x_1d[i] - x_1d[j])**2 + softening**2)
                    v_ee += g_int / dist
                    
            return kin + v_harm + v_ee

        eloc = jax.vmap(single_local_energy)(x_walkers)
        median_e = jnp.median(eloc)
        mad_e = jnp.mean(jnp.abs(eloc - median_e))
        eloc_clipped = jnp.clip(eloc, median_e - 5.0 * mad_e, median_e + 5.0 * mad_e)
        e_mean = jnp.mean(eloc_clipped)
        
        def loss_fn(jastrow_p):
            p_temp = {"jastrow": jastrow_p}
            log_psis = ansatz_apply(p_temp, x_walkers)
            return 2.0 * jnp.mean((jax.lax.stop_gradient(eloc_clipped) - jax.lax.stop_gradient(e_mean)) * log_psis)
        
        if "jastrow" in params:
            grads_jastrow = jax.grad(loss_fn)(params["jastrow"])
        else:
            grads_jastrow = None
            
        return e_mean, grads_jastrow, eloc

    # LÍNEA FALTANTE RECUPERADA
    return ansatz_apply, vmc_local_energy, jastrow_net, cfg

# =====================================================================
# 4. FUNCIONES DE EJECUCIÓN (BASELINE Y OPTIMIZACIÓN)
# =====================================================================
def run_baseline_evaluation(trainer, vmc_energy_fn, key, walkers, params, n_steps, is_rl=False):
    energy_history = []
    logp, gradp = trainer.get_logp_and_grad(params, walkers)
    
    print(f"--- Baseline CASSCF Evaluation ({'RL-MALA' if is_rl else 'MALA'}) ---")
    for step in range(n_steps):
        key, subk = jax.random.split(key)
        exp_std = 0.2 if is_rl else 0.0
        p_act = params["actor"] if is_rl else trainer.actor_model.init(subk, jnp.zeros((1, trainer.state_dim)))
        
        out = trainer.td3_update_all_electrons(subk, walkers, logp, gradp, params, p_act, exp_std)
        walkers, logp, gradp = out[0], out[1], out[2]
        
        e_mean, _, _ = vmc_energy_fn(params, walkers)
        energy_history.append(float(e_mean))
        
        if step % 500 == 0:
            print(f"Step {step:4d}/{n_steps} | Energy: {e_mean:.4f} | Accept: {float(jnp.mean(out[10])):.3f}")
            
    return energy_history, walkers

def run_vmc_optimization(trainer, vmc_energy_fn, key, walkers, params, n_steps, lr_jastrow=1e-3, is_rl=False):
    tx_jas = optax.adam(learning_rate=lr_jastrow)
    opt_state_jas = tx_jas.init(params["jastrow"])
    energy_history = []
    
    logp, gradp = trainer.get_logp_and_grad(params, walkers)
    
    print(f"\n--- Starting VMC Optimization ({'RL-MALA' if is_rl else 'MALA'}) ---")
    for step in range(n_steps):
        key, subk = jax.random.split(key)
        exp_std = 0.2 if is_rl else 0.0
        p_act = params["actor"] if is_rl else trainer.actor_model.init(subk, jnp.zeros((1, trainer.state_dim)))
        
        out = trainer.td3_update_all_electrons(subk, walkers, logp, gradp, params, p_act, exp_std)
        walkers, logp, gradp = out[0], out[1], out[2]
        
        e_mean, grads_jas, _ = vmc_energy_fn(params, walkers)
        energy_history.append(float(e_mean))
        
        updates, opt_state_jas = tx_jas.update(grads_jas, opt_state_jas, params["jastrow"])
        params["jastrow"] = optax.apply_updates(params["jastrow"], updates)
        
        if step % 500 == 0:
            print(f"Step {step:4d}/{n_steps} | Energy: {e_mean:.4f} | Accept: {float(jnp.mean(out[10])):.3f}")
            
    return params, energy_history, walkers

def evaluate_optimized_ansatz(trainer, vmc_energy_fn, key, walkers, params, n_eval_steps, is_rl=False):
    logp, gradp = trainer.get_logp_and_grad(params, walkers)
    all_traj, all_energies = [], []
    
    for _ in range(n_eval_steps):
        key, subk = jax.random.split(key)
        p_act = params["actor"] if is_rl else trainer.actor_model.init(subk, jnp.zeros((1, trainer.state_dim)))
        out = trainer.td3_update_all_electrons(subk, walkers, logp, gradp, params, p_act, 0.0)
        walkers, logp, gradp = out[0], out[1], out[2]
        
        e_mean, _, _ = vmc_energy_fn(params, walkers)
        all_traj.append(np.array(walkers))
        all_energies.append(float(e_mean))
        
    return np.array(all_traj), np.array(all_energies)

# =====================================================================
# 5. VISUALIZACIÓN
# =====================================================================
def plot_baseline_verification(hist_mala, hist_rl, target_energy, out_dir):
    plt.figure(figsize=(10, 6))
    df = pd.DataFrame({"MALA": hist_mala, "RL": hist_rl})
    
    plt.plot(df["MALA"], color='#e74c3c', alpha=0.3)
    plt.plot(df["MALA"].rolling(50).mean(), color='#c0392b', lw=2, label=f'MALA (Final: {np.mean(hist_mala[-500:]):.4f})')
    
    plt.plot(df["RL"], color='#2ecc71', alpha=0.3)
    plt.plot(df["RL"].rolling(50).mean(), color='#27ae60', lw=2, label=f'RL-MALA (Final: {np.mean(hist_rl[-500:]):.4f})')
    
    if target_energy is not None:
        plt.axhline(target_energy, color='k', linestyle='--', lw=2.5, label=f'CASSCF Target (summary.json): {target_energy:.4f}')
        
    plt.title("Baseline Energy Verification (Bare CASSCF)", weight='bold')
    plt.xlabel("MCMC Steps")
    plt.ylabel("Local Energy (Hartree)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Baseline_Energy_Verification.png"), dpi=300)
    plt.close()

def plot_vmc_comparison(hist_mala, hist_rl, traj_mala, traj_rl, n_up, n_dn, target_energy, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    ax = axes[0]
    ax.plot(hist_mala, color='#e74c3c', alpha=0.8, label=f'MALA Sampler (Final: {np.mean(hist_mala[-1000:]):.3f})')
    ax.plot(hist_rl, color='#2ecc71', alpha=0.8, label=f'RL-MALA Sampler (Final: {np.mean(hist_rl[-1000:]):.3f})')
    
    df_e = pd.DataFrame({"MALA": hist_mala, "RL": hist_rl})
    ax.plot(df_e["MALA"].rolling(100).mean(), color='#c0392b', lw=2)
    ax.plot(df_e["RL"].rolling(100).mean(), color='#27ae60', lw=2)
    
    if target_energy is not None:
        ax.axhline(target_energy, color='k', linestyle='--', lw=2, label='CASSCF Baseline')
        
    ax.set_title("VMC Energy Minimization (CASSCF + Jastrow)", weight='bold')
    ax.set_xlabel("Optimization Steps")
    ax.set_ylabel("Local Energy (Hartree)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    burn = int(len(traj_mala) * 0.2)
    x_mala = traj_mala[burn:].reshape(-1, n_up+n_dn)
    bins = np.linspace(-15, 15, 100)
    ax.hist(x_mala.flatten(), bins=bins, density=True, color='k', histtype='step', lw=2, label='Total Density')
    ax.hist(x_mala[:, :n_up].flatten(), bins=bins, density=True, color='#3498db', alpha=0.5, label=r'Spin $\uparrow$')
    ax.hist(x_mala[:, n_up:].flatten(), bins=bins, density=True, color='#e74c3c', alpha=0.5, label=r'Spin $\downarrow$')
    ax.set_title("Fine-Tuned Density (MALA Sampler)", weight='bold')
    ax.set_xlabel("Position (Bohr)")
    ax.legend()
    
    ax = axes[2]
    x_rl = traj_rl[burn:].reshape(-1, n_up+n_dn)
    ax.hist(x_rl.flatten(), bins=bins, density=True, color='k', histtype='step', lw=2, label='Total Density')
    ax.hist(x_rl[:, :n_up].flatten(), bins=bins, density=True, color='#3498db', alpha=0.5, label=r'Spin $\uparrow$')
    ax.hist(x_rl[:, n_up:].flatten(), bins=bins, density=True, color='#e74c3c', alpha=0.5, label=r'Spin $\downarrow$')
    ax.set_title("Fine-Tuned Density (RL-MALA Sampler)", weight='bold')
    ax.set_xlabel("Position (Bohr)")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "VMC_FineTuning_Comparison.png"), dpi=300)
    plt.close()

# =====================================================================
# 6. FUNCIÓN PRINCIPAL
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="VMC Fine-Tuning de CASSCF incorporando un Factor Jastrow Neuronal.")
    parser.add_argument("--npz_path", type=str, required=True, help="Ruta al archivo .npz del Ansatz CASSCF base.")
    parser.add_argument("--out_dir", type=str, default="results_vmc_opt_jastrow", help="Directorio de salida.")
    parser.add_argument("--n_walkers", type=int, default=256, help="Número de caminantes (batch size de VMC).")
    parser.add_argument("--n_steps_opt", type=int, default=30_000, help="Pasos de optimización variacional.")
    parser.add_argument("--n_steps_eval", type=int, default=50_000, help="Pasos de evaluación MCMC.")
    parser.add_argument("--lr_jastrow", type=float, default=5e-4, help="Learning rate para la red Jastrow.")
    parser.add_argument("--tau_base", type=float, default=0.10, help="Tamaño de paso base para MALA.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    summary_path = os.path.join(os.path.dirname(args.npz_path), "summary.json")
    target_energy = None
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary_data = json.load(f)
            target_energy = summary_data.get("energy", None)
            print(f"-> Baseline Target Energy found in summary.json: {target_energy}")
    
    print(f"-> Loading CASSCF (Frozen) & building Jastrow from: {args.npz_path}")
    ansatz_apply, vmc_energy_fn, jastrow_net, cfg = cargar_ansatz_total(args.npz_path)
    n_elec, n_up, n_dn = cfg["n_up"] + cfg["n_down"], cfg["n_up"], cfg["n_down"]
    
    key = jax.random.PRNGKey(42)
    key, k_w, k_j, k_a = jax.random.split(key, 4)
    walkers_init = jax.random.normal(k_w, (args.n_walkers, n_elec, 1)) * 3.0
    
    dummy_x = jnp.zeros((n_elec, 1))
    jastrow_params = jastrow_net.init(k_j, dummy_x)
    
    trainer = TD3Trainer(ansatz_apply, n_elec, n_up, n_dn, tau=args.tau_base, actor_amp=0.15, lambda_=0.25)
    
    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0
    trainer.state_dim = 4 + 5 + 3 * max_same + 3 * max_opp
    
    actor_params = trainer.actor_model.init(k_a, jnp.zeros((1, trainer.state_dim)))
    
    print("\n[PHASE 1] Executing pure CASSCF baseline verification...")
    base_params_mala = {}
    base_params_rl = {"actor": actor_params}
    
    base_e_mala, _ = run_baseline_evaluation(trainer, vmc_energy_fn, key, walkers_init.copy(), base_params_mala, 2000, is_rl=False)
    base_e_rl, _ = run_baseline_evaluation(trainer, vmc_energy_fn, key, walkers_init.copy(), base_params_rl, 2000, is_rl=True)
    
    plot_baseline_verification(base_e_mala, base_e_rl, target_energy, args.out_dir)
    
    print("\n[PHASE 2] Starting Jastrow Optimization...")
    params_mala = {"jastrow": jastrow_params.copy()}
    params_rl = {"jastrow": jastrow_params.copy(), "actor": actor_params}
    
    params_mala_opt, hist_e_mala, w_mala = run_vmc_optimization(
        trainer, vmc_energy_fn, key, walkers_init.copy(), params_mala, 
        args.n_steps_opt, lr_jastrow=args.lr_jastrow, is_rl=False
    )
    
    params_rl_opt, hist_e_rl, w_rl = run_vmc_optimization(
        trainer, vmc_energy_fn, key, walkers_init.copy(), params_rl, 
        args.n_steps_opt, lr_jastrow=args.lr_jastrow, is_rl=True
    )
    
    print("\n-> Evaluando Ansatz Optimizado (MALA)...")
    traj_mala, e_eval_mala = evaluate_optimized_ansatz(trainer, vmc_energy_fn, key, w_mala, params_mala_opt, args.n_steps_eval, is_rl=False)
    
    print("-> Evaluando Ansatz Optimizado (RL-MALA)...")
    traj_rl, e_eval_rl = evaluate_optimized_ansatz(trainer, vmc_energy_fn, key, w_rl, params_rl_opt, args.n_steps_eval, is_rl=True)
    
    plot_vmc_comparison(hist_e_mala, hist_e_rl, traj_mala, traj_rl, n_up, n_dn, target_energy, args.out_dir)
    
    t_int_mala = compute_autocorr_time(e_eval_mala)
    t_int_rl = compute_autocorr_time(e_eval_rl)
    
    print("\n--- VMC FINE-TUNING SUMMARY ---")
    print(f"MALA    | Final Energy: {np.mean(e_eval_mala):.4f} +/- {np.std(e_eval_mala)/np.sqrt(len(e_eval_mala)):.4f} | tau_int: {t_int_mala:.2f}")
    print(f"RL-MALA | Final Energy: {np.mean(e_eval_rl):.4f} +/- {np.std(e_eval_rl)/np.sqrt(len(e_eval_rl)):.4f} | tau_int: {t_int_rl:.2f}")

if __name__ == "__main__":
    main()