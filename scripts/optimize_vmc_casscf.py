import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
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
# 2. ANSATZ CASSCF ENTRENABLE Y ENERGÍA LOCAL
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

def cargar_ansatz_diferenciable(npz_path):
    data = np.load(npz_path)
    meta = json.loads(str(data["metadata_json"]))
    cfg = meta["config"]
    omega, n_up, n_dn, n_basis = cfg["omega"], cfg["n_up"], cfg["n_down"], cfg["n_basis"]
    
    import re
    match = re.search(r'c(\d+)-a(\d+)', npz_path)
    fallback_core = int(match.group(1)) if match else 0
    fallback_active = int(match.group(2)) if match else 2
    n_core = cfg.get("n_core", cfg.get("n_core_orbitals", fallback_core))
    n_active = cfg.get("n_active_orbitals", cfg.get("n_active", fallback_active))
    
    # Parámetros Variacionales Iniciales (Extraídos del CASSCF estático)
    orbitals_init = jnp.array(data["orbital_coefficients"])
    ci_coeffs_init = jnp.array(data["selected_ci_coefficients"])
    dets = data["selected_ci_determinants"]
    
    occ_up_list, occ_dn_list = [], []
    for d_str in dets:
        val = int(d_str)
        act_up = [i for i in range(n_active) if val & (1 << i)]
        act_dn = [i for i in range(n_active) if val & (1 << (n_active + i))]
        occ_up_list.append(list(range(n_core)) + [n_core + i for i in act_up])
        occ_dn_list.append(list(range(n_core)) + [n_core + i for i in act_dn])
        
    occ_up_idx, occ_dn_idx = jnp.array(occ_up_list), jnp.array(occ_dn_list)
    
    @jax.jit
    def ansatz_apply(params, x):
        """
        Función de onda parametrizada. 
        `params` debe contener 'orbitals' y 'ci_coeffs'.
        """
        orbitals = params["orbitals"]
        ci_coeffs = params["ci_coeffs"]
        x_1d = x[..., 0]
        
        def evaluate_single(x_single):
            x_up, x_dn = x_single[:n_up], x_single[n_up:]
            basis_up = build_ho_basis_jax(x_up, n_basis, omega)
            basis_dn = build_ho_basis_jax(x_dn, n_basis, omega)
            
            def eval_det(idx_up, idx_dn, c_i):
                mat_up = basis_up @ orbitals[:, idx_up] + jnp.eye(n_up) * 1e-12
                mat_dn = basis_dn @ orbitals[:, idx_dn] + jnp.eye(n_dn) * 1e-12
                s_u, l_u = jnp.linalg.slogdet(mat_up)
                s_d, l_d = jnp.linalg.slogdet(mat_dn)
                return c_i * s_u * s_d, l_u + l_d
                
            c_vals, log_mags = jax.vmap(eval_det)(occ_up_idx, occ_dn_idx, ci_coeffs)
            max_log = jnp.max(log_mags)
            psi_val = jnp.sum(c_vals * jnp.exp(log_mags - max_log))
            
            return jnp.clip(max_log + jnp.log(jnp.abs(psi_val) + 1e-15), -1000.0, 1000.0)
            
        return jax.vmap(evaluate_single)(x_1d) if x_1d.ndim == 2 else evaluate_single(x_1d)

    @jax.jit
    def vmc_local_energy(params, x_walkers, g_int=4.0, softening=0.1):
        """Calcula la energía local y los gradientes variacionales de los parámetros CASSCF."""
        def single_local_energy(x):
            def log_psi_flat(x_f):
                return ansatz_apply(params, x_f)
            
            # Cinética
            grad_psi = jax.grad(log_psi_flat)(x)
            lapl_psi = jnp.trace(jax.hessian(log_psi_flat)(x))
            kin = -0.5 * (lapl_psi + jnp.sum(grad_psi**2))
            
            # Potencial
            v_harm = 0.5 * (omega**2) * jnp.sum(x**2)
            n = x.shape[0]
            v_ee = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dist = jnp.sqrt((x[i, 0] - x[j, 0])**2 + softening**2)
                    v_ee += g_int / dist
            return kin + v_harm + v_ee

        eloc = jax.vmap(single_local_energy)(x_walkers)
        
        # Clipping para estabilidad de gradiente
        median_e = jnp.median(eloc)
        mad_e = jnp.mean(jnp.abs(eloc - median_e))
        eloc_clipped = jnp.clip(eloc, median_e - 5.0 * mad_e, median_e + 5.0 * mad_e)
        e_mean = jnp.mean(eloc_clipped)
        
        # Loss function de VMC para optimizar los coeficientes del CASSCF
        def loss_fn(casscf_params):
            p_temp = params.copy()
            p_temp["orbitals"] = casscf_params["orbitals"]
            p_temp["ci_coeffs"] = casscf_params["ci_coeffs"]
            
            log_psis = ansatz_apply(p_temp, x_walkers)
            # Regla del gradiente de la energía variacional
            return 2.0 * jnp.mean((jax.lax.stop_gradient(eloc_clipped) - jax.lax.stop_gradient(e_mean)) * log_psis)
            
        casscf_dict = {"orbitals": params["orbitals"], "ci_coeffs": params["ci_coeffs"]}
        grads_casscf = jax.grad(loss_fn)(casscf_dict)
        
        return e_mean, grads_casscf, eloc

    return ansatz_apply, vmc_local_energy, orbitals_init, ci_coeffs_init, cfg

# =====================================================================
# 3. CICLO DE OPTIMIZACIÓN VMC
# =====================================================================
def run_vmc_optimization(trainer, vmc_energy_fn, key, walkers, params, n_steps, lr_casscf=1e-4, is_rl=False, actor_amp=0.15):
    # Optimizador Adam para los parámetros del CASSCF
    tx_casscf = optax.adam(learning_rate=lr_casscf)
    
    casscf_params = {"orbitals": params["orbitals"], "ci_coeffs": params["ci_coeffs"]}
    opt_state = tx_casscf.init(casscf_params)
    
    energy_history = []
    
    # Pre-termalización
    logp, gradp = trainer.get_logp_and_grad(params, walkers)
    
    print(f"\n--- Starting VMC Fine-Tuning ({'RL-MALA' if is_rl else 'MALA'}) ---")
    for step in range(n_steps):
        key, subk = jax.random.split(key)
        
        exp_std = 0.2 if is_rl else 0.0
        p_act = params["actor"] if is_rl else trainer.actor_model.init(subk, jnp.zeros((1, trainer.state_dim)))
        
        # 1. Muestreo
        out = trainer.td3_update_all_electrons(subk, walkers, logp, gradp, params, p_act, exp_std)
        walkers, logp, gradp = out[0], out[1], out[2]
        
        # 2. Computar Gradientes de Energía respecto a CI y Orbitales
        e_mean, grads_casscf, _ = vmc_energy_fn(params, walkers, g_int=4.0)
        energy_history.append(float(e_mean))
        
        # 3. Actualizar Parámetros CASSCF
        updates, opt_state = tx_casscf.update(grads_casscf, opt_state, casscf_params)
        casscf_params = optax.apply_updates(casscf_params, updates)
        
        # Asignar los nuevos valores al diccionario general
        params["orbitals"] = casscf_params["orbitals"]
        params["ci_coeffs"] = casscf_params["ci_coeffs"]
        
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
        
        e_mean, _, _ = vmc_energy_fn(params, walkers, g_int=4.0)
        all_traj.append(np.array(walkers))
        all_energies.append(float(e_mean))
        
    return np.array(all_traj), np.array(all_energies)

# =====================================================================
# 4. VISUALIZACIÓN
# =====================================================================
def plot_vmc_comparison(hist_mala, hist_rl, traj_mala, traj_rl, n_up, n_dn, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    ax = axes[0]
    ax.plot(hist_mala, color='#e74c3c', alpha=0.8, label=f'MALA Optimized (Final: {np.mean(hist_mala[-1000:]):.3f})')
    ax.plot(hist_rl, color='#2ecc71', alpha=0.8, label=f'RL-MALA Optimized (Final: {np.mean(hist_rl[-1000:]):.3f})')
    
    df_e = pd.DataFrame({"MALA": hist_mala, "RL": hist_rl})
    ax.plot(df_e["MALA"].rolling(100).mean(), color='#c0392b', lw=2)
    ax.plot(df_e["RL"].rolling(100).mean(), color='#27ae60', lw=2)
    
    ax.set_title("CASSCF Parameters Optimization (VMC Fine-Tuning)", weight='bold')
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
    plt.savefig(os.path.join(out_dir, "VMC_CASSCF_FineTuning_Comparison.png"), dpi=300)

# =====================================================================
# 5. FUNCIÓN PRINCIPAL
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="VMC Fine-Tuning de Parámetros Orbitales y CI del Ansatz CASSCF.")
    parser.add_argument("--npz_path", type=str, required=True, help="Ruta al archivo .npz del Ansatz CASSCF base.")
    parser.add_argument("--out_dir", type=str, default="results_vmc_opt_casscf", help="Directorio de salida.")
    parser.add_argument("--n_walkers", type=int, default=256, help="Número de caminantes (batch size de VMC).")
    parser.add_argument("--n_steps_opt", type=int, default=3000, help="Pasos de optimización variacional de la energía.")
    parser.add_argument("--n_steps_eval", type=int, default=5000, help="Pasos de evaluación MCMC posteriores.")
    parser.add_argument("--lr_casscf", type=float, default=1e-4, help="Learning rate para optimizar los coeficientes CASSCF.")
    parser.add_argument("--tau_base", type=float, default=0.10, help="Tamaño de paso base para MALA.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"-> Loading CASSCF & Unlocking Parameters from: {args.npz_path}")
    ansatz_apply, vmc_energy_fn, orb_init, ci_init, cfg = cargar_ansatz_diferenciable(args.npz_path)
    n_elec, n_up, n_dn = cfg["n_up"] + cfg["n_down"], cfg["n_up"], cfg["n_down"]
    
    key = jax.random.PRNGKey(42)
    key, k_w, k_a = jax.random.split(key, 3)
    walkers_init = jax.random.normal(k_w, (args.n_walkers, n_elec, 1)) * 3.0
    
    trainer = TD3Trainer(ansatz_apply, n_elec, n_up, n_dn, tau=args.tau_base, actor_amp=0.15, lambda_=0.25)
    
    # --- FIX: Calcular la dimensión del estado manualmente e inyectarla al trainer ---
    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0
    trainer.state_dim = 4 + 5 + 3 * max_same + 3 * max_opp
    # --------------------------------------------------------------------------------
    
    actor_params = trainer.actor_model.init(k_a, jnp.zeros((1, trainer.state_dim)))
    
    # 1. Optimización VMC Guiada por MALA
    params_mala = {"orbitals": orb_init.copy(), "ci_coeffs": ci_init.copy()}
    params_mala_opt, hist_e_mala, w_mala = run_vmc_optimization(
        trainer, vmc_energy_fn, key, walkers_init.copy(), params_mala, 
        args.n_steps_opt, lr_casscf=args.lr_casscf, is_rl=False
    )
    
    # 2. Optimización VMC Guiada por RL-MALA
    params_rl = {"orbitals": orb_init.copy(), "ci_coeffs": ci_init.copy(), "actor": actor_params}
    params_rl_opt, hist_e_rl, w_rl = run_vmc_optimization(
        trainer, vmc_energy_fn, key, walkers_init.copy(), params_rl, 
        args.n_steps_opt, lr_casscf=args.lr_casscf, is_rl=True
    )
    
    print("\n-> Evaluando Ansatz Optimizado (MALA)...")
    traj_mala, e_eval_mala = evaluate_optimized_ansatz(trainer, vmc_energy_fn, key, w_mala, params_mala_opt, args.n_steps_eval, is_rl=False)
    
    print("-> Evaluando Ansatz Optimizado (RL-MALA)...")
    traj_rl, e_eval_rl = evaluate_optimized_ansatz(trainer, vmc_energy_fn, key, w_rl, params_rl_opt, args.n_steps_eval, is_rl=True)
    
    plot_vmc_comparison(hist_e_mala, hist_e_rl, traj_mala, traj_rl, n_up, n_dn, args.out_dir)
    
    t_int_mala = compute_autocorr_time(e_eval_mala)
    t_int_rl = compute_autocorr_time(e_eval_rl)
    
    print("\n--- CASSCF FINE-TUNING SUMMARY ---")
    print(f"MALA    | Final Energy: {np.mean(e_eval_mala):.4f} +/- {np.std(e_eval_mala)/np.sqrt(len(e_eval_mala)):.4f} | tau_int: {t_int_mala:.2f}")
    print(f"RL-MALA | Final Energy: {np.mean(e_eval_rl):.4f} +/- {np.std(e_eval_rl)/np.sqrt(len(e_eval_rl)):.4f} | tau_int: {t_int_rl:.2f}")

if __name__ == "__main__":
    main()