import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt

os.environ["JAX_ENABLE_X64"] = "True"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp

from src.system import PhysicalSystem
from src.wavefunctions import DeepQMCAnsatz
from src.hamiltonians import FullCoulombHamiltonian
from src.samplers import MALASampler, RLMALASampler
from src.actors import DirectionalActor, DirectionalActorIsotropic, DirectionalActorCovariant
from src.aux_functions import calculate_iat
from scripts.run_experiment import SYSTEMS 

def load_weights(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def run_measurement(sampler, walkers, params_dict, cache, key, n_samples, hamiltonian, ansatz, exp_mode=False):
    e_time_series = []
    walkers_history = []
    
    # --- NUEVO: Acumulador para la tasa de aceptación ---
    acc_accumulator = 0.0 
    
    @jax.jit
    def get_local_energy(p, w):
        return hamiltonian.local_energy(ansatz.apply, p, w)
    
    for step in range(n_samples):
        key, k_step = jax.random.split(key)
        
        walkers, stats, cache, _ = sampler.step(k_step, walkers, params_dict, cache, exploration_mode=exp_mode)
        
        e_local_batch = get_local_energy(params_dict["ansatz"], walkers)
        e_mean_step = float(jnp.mean(e_local_batch))
        e_time_series.append(e_mean_step)
        
        # --- NUEVO: Extraemos la aceptación del paso actual ---
        current_acc = float(stats["acceptance"])
        acc_accumulator += current_acc
        
        # Guardamos las posiciones cada 10 pasos
        if step % 10 == 0:
            walkers_history.append(np.array(walkers))
        
        if step % 100 == 0: 
            mode_str = "Actor ON " if exp_mode else "Actor OFF"
            
            # --- NUEVO: Calculamos el promedio de aceptación del bloque ---
            mean_acc = acc_accumulator / 100.0 if step > 0 else current_acc
            
            print(f"  -> Sample {step}/{n_samples} | Mode: {mode_str} | E_mean: {e_mean_step:.4f} Ha | Acc: {mean_acc:.2%}")
            
            # Reiniciamos el acumulador para el siguiente bloque
            acc_accumulator = 0.0 
            
    return np.array(e_time_series), np.stack(walkers_history)

def calculate_progressive_error(e_series, ref_energy, num_points=50):
    n_total = len(e_series)
    indices = np.linspace(100, n_total, num_points, dtype=int)
    std_errors, abs_errors = [], []
    
    for idx in indices:
        sub_series = e_series[:idx]
        tau_int = calculate_iat(sub_series)
        tau_int = max(tau_int, 0.5) 
        n_eff = len(sub_series) / (2.0 * tau_int)
        
        std_err = np.std(sub_series, ddof=1) / np.sqrt(n_eff)
        mean_e = np.mean(sub_series)
        abs_err = np.abs(mean_e - ref_energy)
        
        std_errors.append(std_err)
        abs_errors.append(abs_err)
        
    return indices, np.array(std_errors), np.array(abs_errors)

def plot_comparison_row(ax_std, ax_abs, 
                        idx1, std1, abs1, label1, color1, ls1,
                        idx2, std2, abs2, label2, color2, ls2,
                        title_prefix):
    ax_std.plot(idx1, std1, label=label1, color=color1, linewidth=2, linestyle=ls1)
    ax_std.plot(idx2, std2, label=label2, color=color2, linewidth=2, linestyle=ls2)
    ax_std.set_title(f"{title_prefix} - Std Error", fontsize=11)
    ax_std.set_xlabel("Samples", fontsize=9)
    ax_std.set_ylabel("Std Error (Ha)", fontsize=9)
    ax_std.set_yscale('log')
    ax_std.grid(True, which="both", ls="--", alpha=0.5)
    ax_std.legend(fontsize=8)

    ax_abs.plot(idx1, abs1, label=label1, color=color1, linewidth=2, linestyle=ls1)
    ax_abs.plot(idx2, abs2, label=label2, color=color2, linewidth=2, linestyle=ls2)
    ax_abs.axhline(y=0.0016, color='green', linestyle=':', linewidth=2, label='Chemical Accuracy')
    ax_abs.set_title(f"{title_prefix} - Abs Error", fontsize=11)
    ax_abs.set_xlabel("Samples", fontsize=9)
    ax_abs.set_ylabel("Abs Error (Ha)", fontsize=9)
    ax_abs.set_yscale('log')
    ax_abs.grid(True, which="both", ls="--", alpha=0.5)
    ax_abs.legend(fontsize=8)

def plot_physical_distributions(w_mm, w_rr, w_rm, w_mr, folder_name, system):
    """
    Genera gráficas de la densidad radial electrónica y proyecciones 2D
    usando los historiales de los caminantes.
    """
    print("\nGenerating physical distribution plots...")
    
    # 1. Calcular las distancias radiales (norma espacial de cada electrón respecto al origen)
    # w_shape = (steps, batch, n_elec, 3) -> aplanamos a un vector 1D masivo
    r_mm = np.linalg.norm(w_mm, axis=-1).flatten()
    r_rr = np.linalg.norm(w_rr, axis=-1).flatten()
    r_rm = np.linalg.norm(w_rm, axis=-1).flatten()
    r_mr = np.linalg.norm(w_mr, axis=-1).flatten()

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # --- Panel 1: Distribución Radial Completa (Baseline vs RL) ---
    axs[0, 0].hist(r_mm, bins=200, density=True, alpha=0.6, color='blue', label='MALA Completo')
    axs[0, 0].hist(r_rr, bins=200, density=True, alpha=0.5, color='red', label='RL-MALA Completo')
    axs[0, 0].set_title(fr"Radial Electron Density $|\Psi|^2$ ({system})", fontsize=12)
    axs[0, 0].set_xlabel("Distance from Nucleus (Bohrs)", fontsize=10)
    axs[0, 0].set_ylabel("Probability Density", fontsize=10)
    axs[0, 0].set_xlim(0, 6) # Cortamos la cola larga para ver el detalle
    axs[0, 0].grid(True, ls="--", alpha=0.5)
    axs[0, 0].legend()

    # --- Panel 2: Aislamiento del Sampler (Ansatz MALA Fijo) ---
    axs[0, 1].hist(r_mm, bins=200, density=True, alpha=0.6, color='blue', label='Sampler MALA')
    axs[0, 1].hist(r_mr, bins=200, density=True, alpha=0.5, color='orange', label='Sampler RL')
    axs[0, 1].set_title("Sampler Effect on a Classic Wavefunction", fontsize=12)
    axs[0, 1].set_xlabel("Distance from Nucleus (Bohrs)", fontsize=10)
    axs[0, 1].set_xlim(0, 6)
    axs[0, 1].grid(True, ls="--", alpha=0.5)
    axs[0, 1].legend()

    # --- Panel 3: Mapa de Calor Espacial 2D (MALA Clásico) ---
    # Usamos los ejes X e Y de todos los electrones
    x_mm = w_mm[..., 0].flatten()
    y_mm = w_mm[..., 1].flatten()
    h1 = axs[1, 0].hist2d(x_mm, y_mm, bins=150, cmap='inferno', density=True, range=[[-4, 4], [-4, 4]])
    fig.colorbar(h1[3], ax=axs[1, 0])
    axs[1, 0].set_title("2D Spatial Projection (MALA + MALA)", fontsize=12)
    axs[1, 0].set_xlabel("X (Bohrs)")
    axs[1, 0].set_ylabel("Y (Bohrs)")

    # --- Panel 4: Mapa de Calor Espacial 2D (RL-MALA) ---
    x_rr = w_rr[..., 0].flatten()
    y_rr = w_rr[..., 1].flatten()
    h2 = axs[1, 1].hist2d(x_rr, y_rr, bins=150, cmap='inferno', density=True, range=[[-4, 4], [-4, 4]])
    fig.colorbar(h2[3], ax=axs[1, 1])
    axs[1, 1].set_title("2D Spatial Projection (RL + RL)", fontsize=12)
    axs[1, 1].set_xlabel("X (Bohrs)")
    axs[1, 1].set_ylabel("Y (Bohrs)")

    plt.tight_layout(pad=3.0)
    plot_path = os.path.join(folder_name, f"physical_distributions_{system}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Distribution plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Cross-Evaluation of VMC models")
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--system", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_energy", type=float, required=True)
    args = parser.parse_args()

    sys_config = SYSTEMS[args.system]
    molecule = PhysicalSystem(
        name=args.system, n_elec=sys_config["n_elec"], n_up=sys_config["n_up"],
        n_dn=sys_config["n_dn"], coords=jnp.array(sys_config["coords"]),
        charges=jnp.array(sys_config["charges"], dtype=jnp.int32)
    )
    
    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_run = jax.random.split(key, 3)

    ansatz = DeepQMCAnsatz(molecule)
    actor = DirectionalActorCovariant(n_elec=molecule.n_elec, n_nuclei=len(molecule.coords))
    hamiltonian = FullCoulombHamiltonian(molecule)

    clip_args = {
        "min_clip_grads_ansatz": -20.0, "max_clip_grads_ansatz": 20.0,
        "max_clip_log_ratio": 20.0, "min_clip_log_ratio": -20.0,
        "nan_clip_log_ratio": -20.0, "neginf_clip_log_ratio": -20.0, "posinf_clip_log_ratio": 20.0
    }
    
    sampler_mala = MALASampler(ansatz_apply=ansatz.apply, batch_size=args.batch_size, tau=args.tau, system=molecule, **clip_args)
    sampler_rl = RLMALASampler(ansatz_apply=ansatz.apply, actor_apply=actor.apply, batch_size=args.batch_size, 
                               tau=args.tau, system=molecule, gamma=0.1, beta=10.0, kappa=0.1, penalty_factor=0.1,
                               scale_clip_min=-5.0, scale_clip_max=1.0, **clip_args)

    walkers_init = jax.random.normal(k_init, (args.batch_size, molecule.n_elec, 3)) * 0.2 + molecule.coords[0]

    params_mala_ansatz = load_weights(os.path.join(args.folder_name, "MALA_ansatz_weights.pkl"))
    params_rl_ansatz = load_weights(os.path.join(args.folder_name, "RL-MALA_ansatz_weights.pkl"))
    params_rl_actor = load_weights(os.path.join(args.folder_name, "RL-MALA_actor_weights.pkl"))

    def get_init_cache(params_psi):
        def sum_log_psi(w): return jnp.sum(ansatz.apply(params_psi, w))
        return {
            "log_psi_curr": ansatz.apply(params_psi, walkers_init),
            "grad_curr": jnp.clip(jax.grad(sum_log_psi)(walkers_init), -20.0, 20.0)
        }

    cache_mala_ansatz = get_init_cache(params_mala_ansatz)
    cache_rl_ansatz = get_init_cache(params_rl_ansatz)

    # 1. MALA + MALA
    print("\n[1/4] Ansatz MALA + Sampler MALA")
    e_mm, w_mm = run_measurement(sampler_mala, walkers_init, {"ansatz": params_mala_ansatz, "actor": None}, 
                                 cache_mala_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=False)

    # 2. RL + RL
    print("\n[2/4] Ansatz RL-MALA + Sampler RL-MALA")
    e_rr, w_rr = run_measurement(sampler_rl, walkers_init, {"ansatz": params_rl_ansatz, "actor": params_rl_actor}, 
                                 cache_rl_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=True)

    # 3. RL + MALA
    print("\n[3/4] Ansatz RL-MALA + Sampler MALA")
    e_rm, w_rm = run_measurement(sampler_mala, walkers_init, {"ansatz": params_rl_ansatz, "actor": None}, 
                                 cache_rl_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=False)

    # 4. MALA + RL
    print("\n[4/4] Ansatz MALA + Sampler RL-MALA")
    e_mr, w_mr = run_measurement(sampler_rl, walkers_init, {"ansatz": params_mala_ansatz, "actor": params_rl_actor}, 
                                 cache_mala_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=True)

    # --- GUARDAR TRAYECTORIAS ---
    data_path = os.path.join(args.folder_name, f"trajectories_{args.system}.npz")
    np.savez_compressed(data_path, 
                        e_mala_mala=e_mm, walkers_mala_mala=w_mm,
                        e_rl_rl=e_rr, walkers_rl_rl=w_rr,
                        e_rl_mala=e_rm, walkers_rl_mala=w_rm,
                        e_mala_rl=e_mr, walkers_mala_rl=w_mr)
    print(f"\nTrajectories and energies saved to: {data_path}")

    # --- CÁLCULO DE ERRORES ---
    idx_mm, std_mm, abs_mm = calculate_progressive_error(e_mm, args.ref_energy)
    idx_rr, std_rr, abs_rr = calculate_progressive_error(e_rr, args.ref_energy)
    idx_rm, std_rm, abs_rm = calculate_progressive_error(e_rm, args.ref_energy)
    idx_mr, std_mr, abs_mr = calculate_progressive_error(e_mr, args.ref_energy)

    # ==========================================
    # FIGURA 1: Análisis Original (3 filas)
    # ==========================================
    fig1, axs1 = plt.subplots(3, 2, figsize=(12, 14))
    plot_comparison_row(axs1[0, 0], axs1[0, 1], idx_rr, std_rr, abs_rr, "Ansatz RL + Sampler RL", "red", "--", idx_mm, std_mm, abs_mm, "Ansatz MALA + Sampler MALA", "blue", "-", "Row 1: Proposed vs Baseline")
    plot_comparison_row(axs1[1, 0], axs1[1, 1], idx_rm, std_rm, abs_rm, "Ansatz RL + Sampler MALA", "purple", "-.", idx_mm, std_mm, abs_mm, "Ansatz MALA + Sampler MALA", "blue", "-", "Row 2: Wavefunction Isolation")
    plot_comparison_row(axs1[2, 0], axs1[2, 1], idx_rr, std_rr, abs_rr, "Ansatz RL + Sampler RL", "red", "--", idx_mr, std_mr, abs_mr, "Ansatz MALA + Sampler RL", "orange", ":", "Row 3: Sampler Isolation")
    fig1.tight_layout(pad=2.0)
    fig1.savefig(os.path.join(args.folder_name, f"cross_evaluation_panel_1_{args.system}.png"), dpi=300)

    # ==========================================
    # FIGURA 2: Análisis por Ansatz (2 filas)
    # ==========================================
    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 9))
    plot_comparison_row(axs2[0, 0], axs2[0, 1], idx_mm, std_mm, abs_mm, "Ansatz MALA + Sampler MALA", "blue", "-", idx_mr, std_mr, abs_mr, "Ansatz MALA + Sampler RL", "orange", ":", "Row 1: Ansatz MALA Fixed")
    plot_comparison_row(axs2[1, 0], axs2[1, 1], idx_rm, std_rm, abs_rm, "Ansatz RL + Sampler MALA", "purple", "-.", idx_rr, std_rr, abs_rr, "Ansatz RL + Sampler RL", "red", "--", "Row 2: Ansatz RL Fixed")
    fig2.tight_layout(pad=2.0)
    fig2.savefig(os.path.join(args.folder_name, f"cross_evaluation_panel_2_{args.system}.png"), dpi=300)

    plot_physical_distributions(w_mm, w_rr, w_rm, w_mr, args.folder_name, args.system)

    print("All plots generated successfully.")

if __name__ == "__main__":
    main()