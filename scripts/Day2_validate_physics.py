import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import json
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

# Importaciones del framework
from src.system import Harmonic_1D
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import RLMALASampler_Huziel, MALASampler
from src.local_energy import local_energies
from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.td3_modules import FlaxTD3Actor, build_td3_state
from functools import partial

# --- FUNCIONES MATEMÁTICAS ---
def exact_noninteracting_energy(n_up, n_dn, omega):
    """Calcula E_0 analítico para gas de fermiones ideal (g=0)."""
    e_up = sum([(n + 0.5) for n in range(n_up)])
    e_dn = sum([(n + 0.5) for n in range(n_dn)])
    return omega * (e_up + e_dn)

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

# --- REFACTORIZACIÓN ANTI-OOM CON SUBMUESTREO ---
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
    
    final_carry, history = jax.lax.scan(step_fn, (w_start, k_start, init_cache), jnp.arange(chunk_size))
    return final_carry, history

def collect_samples_safe(sampler_obj, w_start, k_start, p_psi, p_act, total_steps, subsample=1, chunk_size=1000):
    """Muestrea en bloques, aplica submuestreo en la GPU y descarga a la RAM para evitar Out Of Memory."""
    histories = []
    curr_w, curr_k = w_start, k_start
    
    # Aseguramos que el chunk_size sea un múltiplo limpio del subsample
    chunk_size = max(subsample, (chunk_size // subsample) * subsample)
    
    if total_steps == 0:
        return (w_start, k_start, None), np.array([])

    for _ in range(total_steps // chunk_size):
        final_carry, hist_chunk = _collect_chunk(sampler_obj, curr_w, curr_k, p_psi, p_act, chunk_size)
        curr_w, curr_k, _ = final_carry
        # Submuestreo directo antes de almacenar en la lista de RAM
        histories.append(np.asarray(hist_chunk[::subsample]))
        
    return final_carry, np.concatenate(histories, axis=0) if histories else np.array([])

@partial(jax.jit, static_argnums=(2, 3, 4, 5))
def calc_eloc_chunk(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg):
    return local_energies(p_psi, w_chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

def get_eloc_safe(p_psi, flat_w, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg, chunk_size=20000):
    """Calcula energías en bloques para evitar que el vmap colapse la memoria VRAM."""
    if len(flat_w) == 0: return np.array([])
    elocs = []
    for i in range(0, len(flat_w), chunk_size):
        chunk = jnp.array(flat_w[i:i+chunk_size])
        e = calc_eloc_chunk(p_psi, chunk, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
        elocs.append(np.asarray(e))
    return np.concatenate(elocs, axis=0)

# --- DIBUJO DE LA FIGURA 1 ---
def draw_methodology_figure(ax):
    ax.axis('off')
    ax.set_title("Methodology: RLMH Proposal Formulation", fontsize=14, weight='bold')
    nodes = {
        'Psi': (0.1, 0.5, r"$\Psi(X)$"),
        'Feat': (0.3, 0.5, r"$F_i, \rho_{loc}$"),
        'Actor': (0.5, 0.5, "Actor\n(TD3)"),
        'Action': (0.7, 0.5, r"$(\Delta\mu_i, \sigma_i)$"),
        'Prop': (0.9, 0.7, "Proposal\n$X^*$"),
        'MH': (0.9, 0.3, "MH\nCorrection")
    }
    for key, (x, y, label) in nodes.items():
        box = FancyBboxPatch((x-0.06, y-0.08), 0.12, 0.16, boxstyle="round,pad=0.02", 
                             ec="black", fc="aliceblue", lw=1.5)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=12)
        
    arrow_props = dict(facecolor='black', edgecolor='black', arrowstyle='->', lw=2)
    ax.annotate("", xy=(0.24, 0.5), xytext=(0.16, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.44, 0.5), xytext=(0.36, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.64, 0.5), xytext=(0.56, 0.5), arrowprops=arrow_props)
    ax.annotate("", xy=(0.84, 0.65), xytext=(0.76, 0.55), arrowprops=arrow_props)
    ax.annotate("", xy=(0.84, 0.35), xytext=(0.76, 0.45), arrowprops=arrow_props)
    ax.annotate("", xy=(0.9, 0.46), xytext=(0.9, 0.54), arrowprops=arrow_props)

# --- SCRIPT PRINCIPAL ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--n_elec", type=int, required=True)
    parser.add_argument("--n_up", type=int, required=True)
    parser.add_argument("--omega", type=float, default=0.1)
    parser.add_argument("--g", type=float, default=0.0)
    
    # Argumentos parametrizados desde el .sh
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--n_subsamples", type=int, default=10)
    parser.add_argument("--n_ter_eval", type=int, default=5000)
    parser.add_argument("--tau", type=float, default=0.002)
    parser.add_argument("--min_clip_grads", type=float, default=-20.0)
    parser.add_argument("--max_clip_grads", type=float, default=20.0)
    args = parser.parse_args()

    n_dn = args.n_elec - args.n_up
    perm_dir = os.path.join(args.base_dir, f"Perm_Ne{args.n_elec}_Nup{args.n_up}_g{args.g}_Om{args.omega}")
    dir_mala = os.path.join(perm_dir, "MALA")
    dir_td3 = os.path.join(perm_dir, "TD3")

    with open(os.path.join(dir_td3, "params.json"), "r") as f: cfg = json.load(f)
    
    with open(os.path.join(dir_mala, "MALA_ansatz_weights.pkl"), "rb") as f: p_ansatz_mala = pickle.load(f)
    with open(os.path.join(dir_td3, "TD3_ansatz_weights.pkl"), "rb") as f: p_ansatz_td3 = pickle.load(f)
    with open(os.path.join(dir_td3, "TD3_actor_weights.pkl"), "rb") as f: p_actor_td3 = pickle.load(f)

    # Inicializar Entorno Físico
    system = Harmonic_1D(name="Harmonic1D", n_elec=args.n_elec, n_up=args.n_up, n_dn=n_dn, 
                         coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=args.omega, 
                         interaction_strength=args.g, softening=cfg["softening"], dim=1)
    
    sys_cfg = SystemConfig(n_electrons=args.n_elec, n_up=args.n_up, omega=args.omega, 
                           interaction_strength=args.g, softening=cfg["softening"])
    wf_cfg = WavefunctionConfig(name=cfg["wf_model"])
    orbital_cfg = OrbitalConfig(model=cfg["orbital_model"], backflow_components=cfg["backflow_components"],
                                backflow_r_max=cfg["backflow_r_max"], backflow_width=cfg["backflow_width"], 
                                backflow_envelope_length=cfg["backflow_envelope_length"])
    correlation_cfg = CorrelationConfig(model=cfg["correlation_model"])
    
    global ansatz_apply
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    ansatz_apply = ansatz.apply

    td3_model = FlaxTD3Actor(n_up=args.n_up, n_dn=n_dn, actor_amplitude=cfg["actor_amplitude"])
    def td3_wrapper(p_act, w, e_idx, r_nuc, exp_mode, grad):
        state = build_td3_state(w[..., 0], grad[..., 0], e_idx, args.n_elec, args.n_up, n_dn)
        action = td3_model.apply(p_act, state)
        return action[:, 0:1] * cfg["actor_amplitude"], action[:, 1:2], {}

    # Inicialización Segura usando los recortes dinámicos recibidos por argparse
    sampler_mala = MALASampler(
        ansatz.apply, cfg["batch_size"], args.tau, system, 
        args.max_clip_grads, args.min_clip_grads, 50.0, -50.0, -50.0, -50.0, 50.0
    )
    
    sampler_td3 = RLMALASampler_Huziel(
        ansatz_apply=ansatz.apply, system=system, batch_size=cfg["batch_size"], actor_apply=td3_wrapper, 
        tau=args.tau, gamma=0.1, beta=10.0, kappa=0.5, penalty_factor=0.1, max_clip_log_ratio=50.0, 
        min_clip_log_ratio=-50.0, nan_clip_log_ratio=-50.0, neginf_clip_log_ratio=-50.0, 
        posinf_clip_log_ratio=50.0, scale_clip_min=-5.0, scale_clip_max=1.0, 
        min_clip_grads_ansatz=args.min_clip_grads, max_clip_grads_ansatz=args.max_clip_grads
    )

    key = jax.random.PRNGKey(42)
    w_init = jax.random.normal(key, (cfg["batch_size"], args.n_elec, 1))

    # --- FASE 1: TERMALIZACIÓN ---
    print(f"Termalizando N={args.n_elec} ({args.n_ter_eval} pasos)...")
    carry_mala, _ = collect_samples_safe(sampler_mala, w_init, key, p_ansatz_mala, None, args.n_ter_eval, chunk_size=1000)
    carry_td3, _ = collect_samples_safe(sampler_td3, w_init, key, p_ansatz_td3, p_actor_td3, args.n_ter_eval, chunk_size=1000)
    
    w_ter_mala, k_ter_mala, _ = carry_mala
    w_ter_td3, k_ter_td3, _ = carry_td3

    # --- FASE 2: MUESTREO REAL CON SUBMUESTREO ---
    print(f"Muestreando y Submuestreando {args.n_samples} pasos...")
    _, hist_mala = collect_samples_safe(sampler_mala, w_ter_mala, k_ter_mala, p_ansatz_mala, None, args.n_samples, subsample=args.n_subsamples, chunk_size=1000)
    _, hist_td3 = collect_samples_safe(sampler_td3, w_ter_td3, k_ter_td3, p_ansatz_td3, p_actor_td3, args.n_samples, subsample=args.n_subsamples, chunk_size=1000)
    
    # 1. Cálculos de Energía y Z-Score
    e_exact = exact_noninteracting_energy(args.n_up, n_dn, args.omega)
    
    flat_mala = hist_mala[..., 0].reshape(-1, args.n_elec)
    flat_td3 = hist_td3[..., 0].reshape(-1, args.n_elec)

    eloc_mala_all = get_eloc_safe(p_ansatz_mala, flat_mala, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3_all = get_eloc_safe(p_ansatz_td3, flat_td3, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    e_mean_mala, e_var_mala = np.mean(eloc_mala_all), np.var(eloc_mala_all)
    e_mean_td3, e_var_td3 = np.mean(eloc_td3_all), np.var(eloc_td3_all)
    
    traj_mala_w0 = hist_mala[:, 0, :, 0]
    traj_td3_w0 = hist_td3[:, 0, :, 0]
    
    eloc_mala_w0 = get_eloc_safe(p_ansatz_mala, traj_mala_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    eloc_td3_w0 = get_eloc_safe(p_ansatz_td3, traj_td3_w0, sys_cfg, wf_cfg, orbital_cfg, correlation_cfg)

    tau_mala = calculate_iat(autocorr(eloc_mala_w0)[:500])
    tau_td3 = calculate_iat(autocorr(eloc_td3_w0)[:500])
    
    # Se utiliza la cantidad final de muestras independientes post-subsampling para el Error Estándar
    total_samples = len(eloc_mala_all)
    se_mala = np.sqrt((e_var_mala * tau_mala) / max(total_samples, 1))
    se_td3 = np.sqrt((e_var_td3 * tau_td3) / max(total_samples, 1))
    
    z_score = (e_mean_td3 - e_mean_mala) / np.sqrt(se_td3**2 + se_mala**2)
    
    # 2. Submuestreo Visual para KDEs
    sample_size = min(50000, len(flat_mala))
    idx_plot = np.random.choice(len(flat_mala), size=sample_size, replace=False)
    plot_mala = flat_mala[idx_plot]
    plot_td3 = flat_td3[idx_plot]

    def get_pairwise(flat_w):
        r_ij = []
        for i in range(args.n_elec):
            for j in range(i+1, args.n_elec):
                r_ij.append(np.abs(flat_w[:, i] - flat_w[:, j]))
        return np.concatenate(r_ij)
        
    r_ij_mala = get_pairwise(plot_mala)
    r_ij_td3 = get_pairwise(plot_td3)

    # --- RENDERIZADO DEL PLOT ACADÉMICO ---
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    fig.suptitle(fr"Validation of Sampler Correctness & Physics ($N={args.n_elec}, g={args.g}, \omega={args.omega}$)", fontsize=18, weight='bold')

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis('off')
    table_data = [
        ["Metric", "MALA", "RL-MALA (TD3)"],
        [r"$\langle E \rangle$ (Ha)", f"{e_mean_mala:.6f}", f"{e_mean_td3:.6f}"],
        [r"Std Error (SE)", f"{se_mala:.6e}", f"{se_td3:.6e}"],
        [r"Theoretical $E_0$", f"{e_exact:.6f}", f"{e_exact:.6f}"],
        [r"Deviation $|E - E_0|$", f"{abs(e_mean_mala - e_exact):.2e}", f"{abs(e_mean_td3 - e_exact):.2e}"]
    ]
    table = ax0.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.5)
    
    z_color = "green" if abs(z_score) <= 2.0 else "red"
    ax0.text(0.5, -0.1, fr"Equivalence Z-Score: $Z_E = {z_score:.3f}$", ha='center', fontsize=16, color=z_color, weight='bold', bbox=dict(facecolor='white', edgecolor=z_color, boxstyle='round,pad=0.5'))

    ax1 = fig.add_subplot(gs[0, 1])
    draw_methodology_figure(ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    sns.kdeplot(plot_mala.flatten(), ax=ax2, color='black', linestyle='--', linewidth=2, label='MALA (Baseline)')
    sns.kdeplot(plot_td3.flatten(), ax=ax2, color='red', alpha=0.6, linewidth=2, label='RL-MALA (Actor)')
    ax2.set_title(r"One-body Density $\rho(x)$", fontsize=14)
    ax2.set_xlabel("x (Bohr)")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(r_ij_mala, ax=ax3, color='black', linestyle='--', linewidth=2, label='MALA (Baseline)')
    sns.kdeplot(r_ij_td3, ax=ax3, color='red', alpha=0.6, linewidth=2, label='RL-MALA (Actor)')
    ax3.set_title(r"Pairwise Distance Distribution $P(r_{ij})$", fontsize=14)
    ax3.set_xlabel(r"$r_{ij}$ (Bohr)")
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(args.base_dir, f"Validation_Ne{args.n_elec}_Nup{args.n_up}_g{args.g}.png")
    plt.savefig(out_path, dpi=300)
    print(f"Gráfica de validación generada exitosamente: {out_path}")

if __name__ == "__main__":
    main()