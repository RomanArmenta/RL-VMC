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
from src.actors import DirectionalActor
from src.aux_functions import calculate_iat
from scripts.run_experiment import SYSTEMS 

def load_weights(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

# Añadimos el argumento exp_mode para controlar el Actor desde aquí
def run_measurement(sampler, walkers, params_dict, cache, key, n_samples, hamiltonian, ansatz, exp_mode=False):
    e_time_series = []
    
    @jax.jit
    def get_local_energy(p, w):
        return hamiltonian.local_energy(ansatz.apply, p, w)
    
    for step in range(n_samples):
        key, k_step = jax.random.split(key)
        
        walkers, stats, cache, _ = sampler.step(k_step, walkers, params_dict, cache, exploration_mode=exp_mode)
        
        e_local_batch = get_local_energy(params_dict["ansatz"], walkers)
        e_mean_step = float(jnp.mean(e_local_batch))
        e_time_series.append(e_mean_step)
        
        if step % 100 == 0: 
            mode_str = "Actor ON" if exp_mode else "Actor OFF"
            print(f"  -> Sample {step}/{n_samples} | Mode: {mode_str} | E_mean: {e_mean_step:.4f} Ha")
            
    return np.array(e_time_series)

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
    
    # Gráfica Error Estándar
    ax_std.plot(idx1, std1, label=label1, color=color1, linewidth=2, linestyle=ls1)
    ax_std.plot(idx2, std2, label=label2, color=color2, linewidth=2, linestyle=ls2)
    ax_std.set_title(f"{title_prefix} - Standard Error", fontsize=12)
    ax_std.set_xlabel("Samples", fontsize=10)
    ax_std.set_ylabel("Std Error (Ha)", fontsize=10)
    ax_std.set_yscale('log')
    ax_std.grid(True, which="both", ls="--", alpha=0.5)
    ax_std.legend(fontsize=9)

    # Gráfica Error Absoluto
    ax_abs.plot(idx1, abs1, label=label1, color=color1, linewidth=2, linestyle=ls1)
    ax_abs.plot(idx2, abs2, label=label2, color=color2, linewidth=2, linestyle=ls2)
    ax_abs.axhline(y=0.0016, color='green', linestyle=':', linewidth=2, label='Chemical Accuracy')
    ax_abs.set_title(f"{title_prefix} - Absolute Error", fontsize=12)
    ax_abs.set_xlabel("Samples", fontsize=10)
    ax_abs.set_ylabel("Abs Error (Ha)", fontsize=10)
    ax_abs.set_yscale('log')
    ax_abs.grid(True, which="both", ls="--", alpha=0.5)
    ax_abs.legend(fontsize=9)

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

    if args.system not in SYSTEMS:
        raise ValueError(f"System {args.system} not found.")

    sys_config = SYSTEMS[args.system]
    molecule = PhysicalSystem(
        name=args.system, n_elec=sys_config["n_elec"], n_up=sys_config["n_up"],
        n_dn=sys_config["n_dn"], coords=jnp.array(sys_config["coords"]),
        charges=jnp.array(sys_config["charges"], dtype=jnp.int32)
    )
    
    key = jax.random.PRNGKey(args.seed)
    key, k_init, k_run = jax.random.split(key, 3)

    ansatz = DeepQMCAnsatz(molecule)
    actor = DirectionalActor(n_elec=molecule.n_elec, n_nuclei=len(molecule.coords))
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

    path_mala_ansatz = os.path.join(args.folder_name, "MALA_ansatz_weights.pkl")
    path_rl_ansatz = os.path.join(args.folder_name, "RL-MALA_ansatz_weights.pkl")
    path_rl_actor = os.path.join(args.folder_name, "RL-MALA_actor_weights.pkl")

    params_mala_ansatz = load_weights(path_mala_ansatz)
    params_rl_ansatz = load_weights(path_rl_ansatz)
    params_rl_actor = load_weights(path_rl_actor)

    def get_init_cache(params_psi):
        def sum_log_psi(w): return jnp.sum(ansatz.apply(params_psi, w))
        return {
            "log_psi_curr": ansatz.apply(params_psi, walkers_init),
            "grad_curr": jnp.clip(jax.grad(sum_log_psi)(walkers_init), -20.0, 20.0)
        }

    # Cachés iniciales
    cache_mala_ansatz = get_init_cache(params_mala_ansatz)
    cache_rl_ansatz = get_init_cache(params_rl_ansatz)

    # =========================================================================
    # EJECUCIÓN DE LAS 4 COMBINACIONES
    # =========================================================================
    
    # 1. Ansatz MALA + Sampler MALA (Línea Base Clásica)
    print("\n[1/4] Running: Ansatz MALA + Sampler MALA")
    e_mala_mala = run_measurement(sampler_mala, walkers_init, {"ansatz": params_mala_ansatz, "actor": None}, 
                                  cache_mala_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=False)

    # 2. Ansatz RL + Sampler RL (Evaluación completa del modelo propuesto)
    print("\n[2/4] Running: Ansatz RL-MALA + Sampler RL-MALA (Actor ON)")
    e_rl_rl = run_measurement(sampler_rl, walkers_init, {"ansatz": params_rl_ansatz, "actor": params_rl_actor}, 
                              cache_rl_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=True)

    # 3. Ansatz RL + Sampler MALA (Aislamiento de la calidad de la función de onda RL)
    print("\n[3/4] Running: Ansatz RL-MALA + Sampler MALA Clásico")
    e_rl_mala = run_measurement(sampler_mala, walkers_init, {"ansatz": params_rl_ansatz, "actor": None}, 
                                cache_rl_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=False)

    # 4. Ansatz MALA + Sampler RL (Aislamiento del Sampler RL sobre una función de onda clásica)
    print("\n[4/4] Running: Ansatz MALA + Sampler RL-MALA (Actor ON)")
    e_mala_rl = run_measurement(sampler_rl, walkers_init, {"ansatz": params_mala_ansatz, "actor": params_rl_actor}, 
                                cache_mala_ansatz, k_run, args.n_samples, hamiltonian, ansatz, exp_mode=True)

    # =========================================================================
    # CÁLCULO DE ERRORES
    # =========================================================================
    print("\nCalculating progressive errors for all combinations...")
    idx_mala_mala, std_mala_mala, abs_mala_mala = calculate_progressive_error(e_mala_mala, args.ref_energy)
    idx_rl_rl, std_rl_rl, abs_rl_rl = calculate_progressive_error(e_rl_rl, args.ref_energy)
    idx_rl_mala, std_rl_mala, abs_rl_mala = calculate_progressive_error(e_rl_mala, args.ref_energy)
    idx_mala_rl, std_mala_rl, abs_mala_rl = calculate_progressive_error(e_mala_rl, args.ref_energy)

    # =========================================================================
    # GENERACIÓN DEL PANEL (3 filas x 2 columnas)
    # =========================================================================
    fig, axs = plt.subplots(3, 2, figsize=(14, 18))

    # Fila 1: RL-MALA (Completo) vs MALA Clásico (Completo)
    plot_comparison_row(axs[0, 0], axs[0, 1],
                        idx_rl_rl, std_rl_rl, abs_rl_rl, "Ansatz RL + Sampler RL", "red", "--",
                        idx_mala_mala, std_mala_mala, abs_mala_mala, "Ansatz MALA + Sampler MALA", "blue", "-",
                        "Row 1: Full Proposed Model vs Baseline")

    # Fila 2: Ansatz RL (con Sampler MALA) vs Ansatz MALA (con Sampler MALA)
    plot_comparison_row(axs[1, 0], axs[1, 1],
                        idx_rl_mala, std_rl_mala, abs_rl_mala, "Ansatz RL + Sampler MALA", "purple", "-.",
                        idx_mala_mala, std_mala_mala, abs_mala_mala, "Ansatz MALA + Sampler MALA", "blue", "-",
                        "Row 2: Wavefunction Quality Isolation")

    # Fila 3: Ansatz RL (con Sampler RL) vs Ansatz MALA (con Sampler RL)
    plot_comparison_row(axs[2, 0], axs[2, 1],
                        idx_rl_rl, std_rl_rl, abs_rl_rl, "Ansatz RL + Sampler RL", "red", "--",
                        idx_mala_rl, std_mala_rl, abs_mala_rl, "Ansatz MALA + Sampler RL", "orange", ":",
                        "Row 3: Sampler Efficiency Isolation")

    plt.tight_layout(pad=3.0)
    plot_path = os.path.join(args.folder_name, f"cross_evaluation_panel_{args.system}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nEvaluation complete! Panel saved to: {plot_path}")

if __name__ == "__main__":
    main()