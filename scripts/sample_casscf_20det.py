#!/usr/bin/env python3
"""
scripts/sample_casscf_20det.py
Muestreador MCMC y Análisis Comparativo del Ansatz CASSCF (20 determinantes, base 256).
Sistema: N=6 (3 up, 3 down), omega=0.1, g=1.0, softening=0.1.
"""

import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

# Añadir raíz al sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_vmc.config import Experiment1DConfig
from rl_vmc.ansatze.ansatz_1d import CASSCF1DAnsatz
from rl_vmc.physics.harmonic_1d import Harmonic1DHamiltonian


def compute_autocorrelation(series: np.ndarray, max_lag: int = 200) -> np.ndarray:
    """Calcula la función de autocorrelación normalizada (ACF)."""
    n = len(series)
    max_lag = min(max_lag, n // 3)
    s = series - np.mean(series)
    var = np.var(series)
    if var < 1e-14:
        return np.ones(max_lag)
    acf = np.zeros(max_lag)
    for k in range(max_lag):
        acf[k] = np.mean(s[: n - k] * s[k:]) / var
    return acf


def compute_integrated_autocorr_time(acf: np.ndarray) -> float:
    """Calcula el tiempo de autocorrelación integrado tau_int."""
    tau = 0.5
    for k in range(1, len(acf)):
        if acf[k] + acf[k - 1] <= 0:
            break
        tau += acf[k]
    return max(0.5, float(tau))


def run_sampling(
    npz_path: str,
    n_walkers: int = 128,
    burn_in: int = 1000,
    n_steps: int = 3000,
    step_size: float = 0.08,
    output_dir: str = "results_casscf_20det_sampling",
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"===============================================================")
    print(f" MUESTREO MCMC: CASSCF 20 DETERMINANTES (BASE 256)")
    print(f" Archivo: {npz_path}")
    print(f" Caminantes: {n_walkers} | Burn-in: {burn_in} | Pasos: {n_steps}")
    print(f"===============================================================\n")

    cfg = Experiment1DConfig()
    cfg.ansatz.casscf_npz_path = npz_path
    cfg.ansatz.casscf_trainable = False
    cfg.system.n_elec = 6
    cfg.system.n_up = 3
    cfg.system.n_dn = 3
    cfg.system.omega = 0.1
    cfg.system.interaction_strength = 1.0
    cfg.system.softening = 0.1

    # 1. Cargar Ansatz y Hamiltoniano
    ansatz = CASSCF1DAnsatz(cfg)
    hamiltonian = Harmonic1DHamiltonian(cfg.system)
    
    # 2. Inicialización de caminantes en posiciones tipo red de Wigner
    # Evita singularidades nodales colocando las 6 partículas en equilibrio armónico
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    base_lattice = jnp.linspace(-6.0, 6.0, 6).reshape(1, 6, 1)
    walkers = jnp.repeat(base_lattice, n_walkers, axis=0) + jax.random.normal(k1, shape=(n_walkers, 6, 1)) * 0.15

    params = ansatz.init(k2, walkers)
    raw_data = np.load(npz_path, allow_pickle=True)
    target_e = float(raw_data.get("selected_energy", 4.524052))
    print(f"[+] Ansatz inicializado. Energía CI seleccionada de referencia: {target_e:.6f} Ha")

    # 3. Compilación de pasos MCMC (Metropolis-Hastings con gradiente o random walk)
    # Definir función de log-probabilidad y gradientes para MALA
    def log_prob_fn(x):
        return 2.0 * ansatz.apply(params, x)

    @jax.jit
    def mala_step(state, subkey):
        x, curr_log_p = state
        sk1, sk2 = jax.random.split(subkey)
        
        # Propuesta Langevin
        # grad_log_p = grad_fn(x)
        grad_log_p = jax.vmap(jax.grad(lambda w: jnp.squeeze(log_prob_fn(w[None, ...]))))(x)
        drift = 0.5 * step_size * grad_log_p
        drift = jnp.clip(drift, -2.0, 2.0) # Clipping estabilizador
        
        noise = jax.random.normal(sk1, shape=x.shape) * jnp.sqrt(step_size)
        x_prop = x + drift + noise
        
        prop_log_p = log_prob_fn(x_prop)
        grad_prop = jax.vmap(jax.grad(lambda w: jnp.squeeze(log_prob_fn(w[None, ...]))))(x_prop)
        drift_prop = jnp.clip(0.5 * step_size * grad_prop, -2.0, 2.0)
        
        # Probabilidades de transición hacia adelante y hacia atrás
        fwd_diff = x_prop - x - drift
        bwd_diff = x - x_prop - drift_prop
        log_q_fwd = -0.5 * jnp.sum(fwd_diff ** 2, axis=(1, 2)) / step_size
        log_q_bwd = -0.5 * jnp.sum(bwd_diff ** 2, axis=(1, 2)) / step_size
        
        log_alpha = prop_log_p - curr_log_p + log_q_bwd - log_q_fwd
        log_u = jnp.log(jax.random.uniform(sk2, shape=(x.shape[0],)))
        
        accept = log_u < log_alpha
        accept_exp = accept[:, None, None]
        
        new_x = jnp.where(accept_exp, x_prop, x)
        new_log_p = jnp.where(accept, prop_log_p, curr_log_p)
        
        # Distancia cuadrática de salto (ESJD)
        sq_jump = jnp.mean(jnp.sum((new_x - x) ** 2, axis=(1, 2)))
        acc_rate = jnp.mean(accept.astype(jnp.float64))
        
        return (new_x, new_log_p), (acc_rate, sq_jump)

    # 4. Fase de Burn-in
    print(f"\n[+] Iniciando Burn-in ({burn_in} pasos)...")
    curr_log_p = log_prob_fn(walkers)
    state = (walkers, curr_log_p)
    
    t0 = time.time()
    for step in range(burn_in):
        key, subkey = jax.random.split(key)
        state, (acc, _) = mala_step(state, subkey)
        if (step + 1) % (burn_in // 4) == 0 or step == burn_in - 1:
            print(f"    Burn-in paso {step+1}/{burn_in} - Aceptación media: {float(acc)*100:.1f}%")
    t_burn = time.time() - t0
    print(f"[+] Burn-in completado en {t_burn:.2f} s.")

    # 5. Fase de Producción y Medición de Energía Local
    print(f"\n[+] Iniciando Producción ({n_steps} pasos)...")
    t0_prod = time.time()
    
    energies_per_step = []
    kinetics_per_step = []
    potentials_per_step = []
    all_sampled_positions = []
    acceptances = []

    # Medir cada k pasos para aligerar memoria
    measure_every = 2
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        state, (acc, esjd) = mala_step(state, subkey)
        acceptances.append(float(acc))
        
        if step % measure_every == 0:
            curr_x = state[0]
            e_loc, t_loc, v_loc = hamiltonian.local_energy_and_components(ansatz.apply, params, curr_x)
            
            # Filtro robusto de finitud
            mask_finite = jnp.isfinite(e_loc) & (jnp.abs(e_loc) < 100.0)
            if jnp.any(mask_finite):
                energies_per_step.append(float(jnp.mean(e_loc[mask_finite])))
                kinetics_per_step.append(float(jnp.mean(t_loc[mask_finite])))
                potentials_per_step.append(float(jnp.mean(v_loc[mask_finite])))
                all_sampled_positions.append(np.asarray(curr_x).reshape(-1))

        if (step + 1) % (n_steps // 5) == 0:
            curr_e_mean = np.mean(energies_per_step) if energies_per_step else 0.0
            print(f"    Paso {step+1}/{n_steps} | <E_L> = {curr_e_mean:.5f} Ha | Aceptación: {np.mean(acceptances[-100:])*100:.1f}%")

    t_prod = time.time() - t0_prod
    print(f"[+] Producción completada en {t_prod:.2f} s.\n")

    # 6. Post-procesamiento y Métricas Estadísticas
    e_series = np.array(energies_per_step)
    t_series = np.array(kinetics_per_step)
    v_series = np.array(potentials_per_step)
    
    e_mean = float(np.mean(e_series))
    e_var = float(np.var(e_series))
    acf_e = compute_autocorrelation(e_series, max_lag=150)
    tau_int = compute_integrated_autocorr_time(acf_e)
    
    n_eff = len(e_series) / (2.0 * tau_int)
    se_e = float(np.sqrt(e_var / max(1.0, n_eff)))
    t_mean = float(np.mean(t_series))
    v_mean = float(np.mean(v_series))
    acc_mean = float(np.mean(acceptances))
    
    ess_per_sec = n_eff / max(0.1, t_prod)
    diff_vs_target = e_mean - target_e

    # 7. Resumen Comparativo con Benchmarks Anteriores
    # Valores del Benchmark 1 y 2 para N=6, omega=0.1, g=1.0:
    e_b1_casscf = 5.529180   # uhf1d, 36 det, base 32
    e_b2_sjb = 4.462724      # VMC + MALA (SJB), single det + neural Jastrow

    print("=" * 68)
    print(" CUADRO COMPARATIVO: CASSCF (20 DET) vs BENCHMARK 1 & 2")
    print("=" * 68)
    print(f" Modelo / Método                         | Energía (Ha)         | Delta vs Obj. ")
    print("-" * 68)
    print(f" CASSCF CI Objetivo (20 det, base 256)   | {target_e:.6f}          | 0.000000 Ha")
    print(f" Muestreo MCMC 20 det (Este Script)      | {e_mean:.5f} ± {se_e:.5f} | {diff_vs_target:+.6f} Ha")
    print(f" CASSCF B1/B2 (36 det, base 32)          | {e_b1_casscf:.6f}          | {e_b1_casscf - target_e:+.6f} Ha")
    print(f" Variacional SJB B2 (1 det + Jastrow)    | {e_b2_sjb:.6f}          | {e_b2_sjb - target_e:+.6f} Ha")
    print("-" * 68)
    print(f" Observables MCMC:")
    print(f"   <T_L> = {t_mean:.5f} Ha  |  <V_L> = {v_mean:.5f} Ha  |  Var(E_L) = {e_var:.5f}")
    print(f"   tau_int = {tau_int:.2f} pasos  |  ESS/s = {ess_per_sec:.1f}  |  Aceptación = {acc_mean*100:.1f}%")
    print("=" * 68)

    # 8. Guardar Métricas en CSV
    df_metrics = pd.DataFrame([{
        "model": "CASSCF_20det_b256",
        "N": 6,
        "omega": 0.1,
        "g": 1.0,
        "E_mean": e_mean,
        "SE_E": se_e,
        "E_var": e_var,
        "T_mean": t_mean,
        "V_mean": v_mean,
        "tau_int": tau_int,
        "acceptance": acc_mean,
        "target_ci_energy": target_e,
        "delta_vs_target": diff_vs_target,
        "E_b1_casscf_36det_b32": e_b1_casscf,
        "E_b2_sjb_variational": e_b2_sjb,
        "wall_time_prod_s": t_prod,
    }])
    csv_path = os.path.join(output_dir, "comparison_summary_20det.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"\n[+] Resultados guardados en CSV: {csv_path}")

    # 9. Generación de Gráficas Comparativas
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    
    # Panel 1: Traza y Media Acumulada
    steps_meas = np.arange(len(e_series)) * measure_every
    cum_mean = np.cumsum(e_series) / (np.arange(len(e_series)) + 1)
    axes[0].plot(steps_meas, e_series, color="#3498db", alpha=0.35, label="E_L por paso")
    axes[0].plot(steps_meas, cum_mean, color="#2980b9", lw=2.2, label=f"Media Acum. ({e_mean:.4f} Ha)")
    axes[0].axhline(target_e, color="red", ls="--", lw=1.8, label=f"CASSCF CI 20-det ({target_e:.4f} Ha)")
    axes[0].axhline(e_b2_sjb, color="green", ls=":", lw=1.8, label=f"SJB Neural B2 ({e_b2_sjb:.4f} Ha)")
    axes[0].set_title("Evolución de Energía Local", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Paso de Medición")
    axes[0].set_ylabel("Energía (Ha)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8.5)

    # Panel 2: Función de Autocorrelación
    lags = np.arange(len(acf_e))
    axes[1].plot(lags, acf_e, color="#8e44ad", lw=2.0, label=f"$\\rho_E(k)$ ($\\tau_{{int}}={tau_int:.1f}$)")
    axes[1].axhline(0, color="gray", ls="--", lw=0.8)
    axes[1].set_title("Autocorrelación de Energía Local", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Lag $k$")
    axes[1].set_ylabel("ACF $\\rho(k)$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    # Panel 3: Densidad Muestreada (Wigner 6-Peaks)
    all_x = np.concatenate(all_sampled_positions)
    axes[2].hist(all_x, bins=60, density=True, color="#16a085", alpha=0.6, label="Densidad MCMC $\\rho(x)$")
    axes[2].set_title("Densidad Monoparticular Muestreada", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Posición $x$ ($a_0$)")
    axes[2].set_ylabel("Densidad $\\rho(x)$")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "fig_comparison_casscf_20det.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"[+] Gráfica comparativa guardada: {fig_path}")

    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Muestreo MCMC de CASSCF 20 determinantes")
    parser.add_argument(
        "--npz_path",
        type=str,
        default="figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz",
        help="Ruta al archivo .npz de CASSCF",
    )
    parser.add_argument("--walkers", type=int, default=64, help="Número de caminantes")
    parser.add_argument("--burn_in", type=int, default=800, help="Pasos de burn-in")
    parser.add_argument("--steps", type=int, default=2000, help="Pasos de producción")
    parser.add_argument("--step_size", type=float, default=0.08, help="Tamaño de paso MALA")
    parser.add_argument("--out_dir", type=str, default="results_casscf_20det_sampling", help="Directorio de salida")
    args = parser.parse_args()

    run_sampling(
        npz_path=args.npz_path,
        n_walkers=args.walkers,
        burn_in=args.burn_in,
        n_steps=args.steps,
        step_size=args.step_size,
        output_dir=args.out_dir,
    )
