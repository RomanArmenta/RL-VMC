import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

import jax
import jax.numpy as jnp
from functools import partial

# Importaciones de tu arquitectura
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Huziel, Ansatz1D_Backflow
from src.actors import Actor_Huziel
from src.samplers import RLMALASampler_Huziel, MALASampler
from src.wavefunction import WavefunctionConfig
from src.local_energy import local_energies
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.physics import SystemConfig
from src.td3_modules import FlaxTD3Actor, build_td3_state

def find_folder(base_dir, method, g, omega, amp, run_id):
    pattern = f"*_1D__{method}_g{g}_Om{omega}_Amp{amp}_{run_id}"
    matches = glob.glob(os.path.join(base_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No se encontró carpeta para: {pattern}")
    return matches[0]

def load_json_params(folder):
    with open(os.path.join(folder, "params.json"), "r") as f:
        return json.load(f)

def load_weights(folder, method):
    with open(os.path.join(folder, f"{method}_ansatz_weights.pkl"), "rb") as f:
        p_ansatz = pickle.load(f)
    
    # Manejo seguro: MALA no tiene actor, por lo tanto no habrá archivo
    actor_path = os.path.join(folder, f"{method}_actor_weights.pkl")
    if os.path.exists(actor_path):
        with open(actor_path, "rb") as f:
            p_actor = pickle.load(f)
    else:
        p_actor = None
        
    return p_ansatz, p_actor

@partial(jax.jit, static_argnums=(0, 6, 7))
def collect_samples(sampler_obj, w_start, k_start, p_psi, p_act, init_cache, n_steps, subsample):
    """Bucle JIT para recolectar las posiciones de los electrones."""
    def step_fn(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        
        # En MALA, p_act será None, pero MALASampler ignora esa llave de todas formas
        params_dict = {"ansatz": p_psi}
        if p_act is not None:
            params_dict["actor"] = p_act
            
        w_new, stats, new_cache, _ = sampler_obj.step(
            subkey, w, params_dict, cache, exploration_mode=False
        )
        return (w_new, k, new_cache), w_new
    
    _, history = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(n_steps))
    return history[::subsample]

def autocorr(x):
    """Función de autocorrelación rápida."""
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:] / result[result.size // 2]

def calculate_iat(acf_array):
    """Calcula tau_int usando el estimador de ventana (hasta que la ACF se vuelve negativa o ruidosa)."""
    tau = 1.0 # Equivalente a la convención: 1 + 2 * sum(rho)
    for rho in acf_array[1:]:
        if rho < 0: # Cortamos en el primer cruce por cero para evitar sumar ruido estadístico
            break
        tau += 2.0 * rho
    return tau

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="./results_1D_new")
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--amp", type=float, required=True)
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--subsample", type=int, default=10)
    args = parser.parse_args()

    # 1. Localizar y Cargar
    dir_mala = find_folder(args.base_dir, "MALA", args.g, args.omega, args.amp, args.run_id)
    dir_td3 = find_folder(args.base_dir, "TD3", args.g, args.omega, args.amp, args.run_id)
    cfg = load_json_params(dir_td3) 
    
    out_dir = os.path.join(dir_mala, "compiled_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    p_ansatz_mala, p_actor_mala = load_weights(dir_mala, "MALA")
    p_ansatz_td3, p_actor_td3 = load_weights(dir_td3, "TD3")

    # 2. Configurar Física (Ansatz con Backflow)
    system = Harmonic_1D(name="Harmonic1D", n_elec=cfg["n_elec"], n_up=cfg["n_up"], n_dn=cfg["n_dn"], 
                         coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=cfg["omega"], 
                         interaction_strength=cfg["interaction_strength"], softening=cfg["softening"], dim=1)
    
    sys_cfg = SystemConfig(
        n_electrons=cfg["n_elec"], 
        n_up=cfg["n_up"], 
        omega=cfg["omega"], 
        interaction_strength=cfg["interaction_strength"], 
        softening=cfg["softening"]
    )
                         
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(
        model=cfg["orbital_model"], backflow_components=cfg.get("backflow_components", 8),
        backflow_r_max=cfg.get("backflow_r_max", 5.0), backflow_width=cfg.get("backflow_width", 0.7), 
        backflow_envelope_length=cfg.get("backflow_envelope_length", 3.0)
    )
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)

    # 3. Wrappers y Samplers
    td3_model = FlaxTD3Actor(n_up=cfg["n_up"], n_dn=cfg["n_dn"], actor_amplitude=cfg["actor_amplitude"])
    
    def td3_wrapper(p_act, w, e_idx, r_nuc, exp_mode, grad):
        w_1d, grad_1d = w[..., 0], grad[..., 0]
        state = build_td3_state(w_1d, grad_1d, e_idx, cfg["n_elec"], cfg["n_up"], cfg["n_dn"])
        action = td3_model.apply(p_act, state)
        drift_res, raw_sigma = action[:, 0:1] * cfg["actor_amplitude"], action[:, 1:2]
        return drift_res, raw_sigma, {}

    sampler_mala = MALASampler(
        ansatz.apply, cfg["batch_size"], cfg["tau"], system, 
        cfg.get("max_clip_grads_ansatz", 50.0), cfg.get("min_clip_grads_ansatz", -50.0), 
        cfg.get("max_clip_log_ratio", 50.0), cfg.get("min_clip_log_ratio", -50.0), 
        cfg.get("nan_clip_log_ratio", -50.0), cfg.get("neginf_clip_log_ratio", -50.0), 
        cfg.get("posinf_clip_log_ratio", 50.0)
    )
                                      
    sampler_td3 = RLMALASampler_Huziel(
        ansatz.apply, system, cfg["batch_size"], actor_apply=td3_wrapper, 
        tau=cfg["tau"], gamma=cfg["gamma"], beta=cfg["beta"], kappa=cfg["kappa"], penalty_factor=0.1,
        max_clip_log_ratio=50., min_clip_log_ratio=-50., nan_clip_log_ratio=-50., neginf_clip_log_ratio=-50., posinf_clip_log_ratio=50.,
        scale_clip_min=-5., scale_clip_max=1., min_clip_grads_ansatz=-50., max_clip_grads_ansatz=50.
    )

    # 4. Muestreo Cronometrado
    print("Muestreando MALA Clásico...")
    key = jax.random.PRNGKey(42)
    w_init = jax.random.normal(key, (cfg["batch_size"], cfg["n_elec"], 1))
    
    def sum_log_mala(w): return jnp.sum(ansatz.apply(p_ansatz_mala, w))
    cache_mala = {"log_psi_curr": ansatz.apply(p_ansatz_mala, w_init), "grad_curr": jnp.clip(jax.grad(sum_log_mala)(w_init), -50.0, 50.0)}
    
    _ = collect_samples(sampler_mala, w_init, key, p_ansatz_mala, None, cache_mala, 2, 1)
    
    start_mala = time.time()
    hist_mala = collect_samples(sampler_mala, w_init, key, p_ansatz_mala, None, cache_mala, args.n_samples, args.subsample)
    hist_mala.block_until_ready() 
    time_mala = time.time() - start_mala

    print("Muestreando TD3...")
    def sum_log_td3(w): return jnp.sum(ansatz.apply(p_ansatz_td3, w))
    cache_td3 = {"log_psi_curr": ansatz.apply(p_ansatz_td3, w_init), "grad_curr": jnp.clip(jax.grad(sum_log_td3)(w_init), -50.0, 50.0)}
    
    _ = collect_samples(sampler_td3, w_init, key, p_ansatz_td3, p_actor_td3, cache_td3, 2, 1)
    
    start_td3 = time.time()
    hist_td3 = collect_samples(sampler_td3, w_init, key, p_ansatz_td3, p_actor_td3, cache_td3, args.n_samples, args.subsample)
    hist_td3.block_until_ready()
    time_td3 = time.time() - start_td3


    # 5. Generación de Gráficos y Métricas
    sns.set_theme(style="whitegrid")
    x_grid = np.linspace(-15, 15, 500)
    v_ho = 0.5 * (cfg["omega"]**2) * x_grid**2

    flat_w_mala = np.array(hist_mala).reshape(-1, cfg["n_elec"])
    flat_w_td3 = np.array(hist_td3).reshape(-1, cfg["n_elec"])

    # --- PLOT 1: SUPERPOSICIÓN DE DENSIDADES ---
    print("Generando Gráfico de Densidades Superpuestas...")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(fr"Validación de Distribución: MALA (Relieve) vs TD3 (Líneas) (g={args.g})", fontsize=14)
    
    ax.hist(flat_w_mala[:, :cfg["n_up"]].flatten(), bins=100, density=True, alpha=0.3, color='gray', label='MALA Ground Truth (Up)')
    ax.hist(flat_w_mala[:, cfg["n_up"]:].flatten(), bins=100, density=True, alpha=0.3, color='darkgray', label='MALA Ground Truth (Dn)')
    
    ax.hist(flat_w_td3[:, :cfg["n_up"]].flatten(), bins=100, density=True, histtype='step', linewidth=2.5, color='orange', label=r'TD3 $\rho_{\uparrow}(x)$')
    ax.hist(flat_w_td3[:, cfg["n_up"]:].flatten(), bins=100, density=True, histtype='step', linewidth=2.5, color='green', label=r'TD3 $\rho_{\downarrow}(x)$')
    
    ax2 = ax.twinx()
    ax2.plot(x_grid, v_ho, color='black', alpha=0.2, linestyle='--', label=r'$V_{HO}(x)$')
    ax2.set_ylim(0, max(v_ho))
    ax.set_xlabel("x"); ax.set_ylabel("Densidad Probabilidad")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.savefig(os.path.join(out_dir, "densities_overlay.png"), dpi=300)

    # --- PLOT 2: PERFIL DEL ACTOR ---
    print("Generando Perfil del Actor...")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Perfil del Drift Propuesto vs Fuerza Cuántica")
    
    n_fixed = cfg["n_elec"] - 1
    if n_fixed > 0:
        fixed_pos = jnp.linspace(-5.0, 5.0, n_fixed)
    else:
        fixed_pos = jnp.array([])
    probe_x = jnp.linspace(-15, 15, 300)
    batch_probe = jnp.zeros((300, cfg["n_elec"], 1))
    batch_probe = batch_probe.at[:, 0, 0].set(probe_x)
    batch_probe = batch_probe.at[:, 1:, 0].set(fixed_pos)

    for ax, p_psi, p_act, name in zip(axs, [p_ansatz_mala, p_ansatz_td3], [None, p_actor_td3], ["MALA Clásico", "TD3"]):
        def sum_log(w): return jnp.sum(ansatz.apply(p_psi, w))
        
        batch_grad = jnp.clip(jax.grad(sum_log)(batch_probe), -50.0, 50.0)
        q_force = batch_grad[:, 0, 0] 
        mala_drift = cfg["tau"] * q_force

        if name == "MALA Clásico":
            actor_drift = jnp.zeros_like(q_force)
        else:
            drift, _, _ = td3_wrapper(p_act, batch_probe, 0, system.coords, False, batch_grad)
            actor_drift = drift[:, 0]

        ax.plot(probe_x, q_force, 'k--', alpha=0.5, label=r'Fuerza Cuántica $\nabla \ln|\Psi|^2$')
        ax.plot(probe_x, mala_drift, 'r-', label=r'Drift MALA ($\tau \nabla$)')
        
        if name != "MALA Clásico":
            ax.plot(probe_x, actor_drift, 'b-', label='Drift Actor RL', linewidth=2)
            
        ax.set_title(f"Método: {name}")
        ax.set_xlabel("Posición Electrón 0 (x)")
        ax.legend()
    plt.savefig(os.path.join(out_dir, "actor_profile.png"), dpi=300)

    # --- PLOT 3: AUTOCORRELACIÓN Y MÉTRICAS FINALES ---
    print("Calculando Energías Locales y Autocorrelación...")
    
    com_mala = np.mean(np.array(hist_mala)[:, 0, :, 0], axis=1)
    com_td3 = np.mean(np.array(hist_td3)[:, 0, :, 0], axis=1)
    
    traj_mala = jnp.array(hist_mala)[:, 0, :, 0] 
    traj_td3 = jnp.array(hist_td3)[:, 0, :, 0]
    
    eloc_mala = local_energies(p_ansatz_mala, traj_mala, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3 = local_energies(p_ansatz_td3, traj_td3, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    acf_com_mala = autocorr(com_mala)[:200]
    acf_com_td3 = autocorr(com_td3)[:200]
    acf_e_mala = autocorr(eloc_mala)[:200]
    acf_e_td3 = autocorr(eloc_td3)[:200]

    tau_com_mala = calculate_iat(acf_com_mala)
    tau_com_td3 = calculate_iat(acf_com_td3)
    tau_e_mala = calculate_iat(acf_e_mala)
    tau_e_td3 = calculate_iat(acf_e_td3)

    N_total = args.n_samples
    neff_e_mala = N_total / tau_e_mala
    neff_e_td3 = N_total / tau_e_td3

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(fr"Decaimiento de Memoria MCMC (Lag de {args.subsample} pasos físicos)", fontsize=16)

    axs[0].plot(acf_com_mala, 'b-', label=fr'MALA ($\tau_{{int}} \approx {tau_com_mala:.1f}$)')
    axs[0].plot(acf_com_td3, 'r-', label=fr'TD3 ($\tau_{{int}} \approx {tau_com_td3:.1f}$)')
    axs[0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[0].set_title("Autocorrelación: Centro de Masa")
    axs[0].set_xlabel("Lag (Puntos guardados)")
    axs[0].legend()

    axs[1].plot(acf_e_mala, 'b-', label=fr'MALA ($N_{{eff}} \approx {int(neff_e_mala)}$)')
    axs[1].plot(acf_e_td3, 'r-', label=fr'TD3 ($N_{{eff}} \approx {int(neff_e_td3)}$)')
    axs[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axs[1].set_title("Autocorrelación: Energía Local $E_L(x)$")
    axs[1].set_xlabel("Lag (Puntos guardados)")
    axs[1].legend()

    plt.savefig(os.path.join(out_dir, "autocorrelation_advanced.png"), dpi=300)
    
    ess_sec_mala = neff_e_mala / time_mala
    ess_sec_td3 = neff_e_td3 / time_td3

    print(f"==================================================")
    print(f"MÉTRICAS FINALES (g={args.g}):")
    print(f"--------------------------------------------------")
    print(f"Energía MALA: {np.mean(eloc_mala):.5f} Ha")
    print(f"Energía TD3:  {np.mean(eloc_td3):.5f} Ha")
    print(f"Tau_int (E) MALA: {tau_e_mala:.2f}")
    print(f"Tau_int (E) TD3:  {tau_e_td3:.2f}")
    print(f"--------------------------------------------------")
    print(f"N_eff MALA: {int(neff_e_mala)} muestras útiles")
    print(f"N_eff TD3:  {int(neff_e_td3)} muestras útiles")
    print(f"Tiempo Muestreo MALA: {time_mala:.2f} s")
    print(f"Tiempo Muestreo TD3:  {time_td3:.2f} s")
    print(f"--------------------------------------------------")
    print(f"ESS/sec MALA: {ess_sec_mala:.2f} muestras/segundo")
    print(f"ESS/sec TD3:  {ess_sec_td3:.2f} muestras/segundo")
    print(f"==================================================")
    
    print(f"Evaluación completada. Gráficos guardados en {out_dir}")

if __name__ == "__main__":
    main()