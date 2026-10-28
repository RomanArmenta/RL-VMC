import os
# ¡LÍNEA CRÍTICA!: Fuerza a JAX a usar precisión doble para evitar NaNs en los determinantes
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import matplotlib.pyplot as plt
import optax
import pandas as pd

# =====================================================================
# 1. RESOLUCIÓN DE RUTAS
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_VMC_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
UHF1D_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../uhf_1d_ho_v270826_m-determinants/src"))
WIGNER_POST_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../wigner_postprocessing_v014"))

sys.path.append(RL_VMC_SRC)
sys.path.append(UHF1D_SRC)
sys.path.append(WIGNER_POST_DIR)

from src.td3_modules import TD3Trainer, ReplayBuffer
from wigner_post.analysis import analyze_orbital_state
from wigner_post.models import GridConfig

# =====================================================================
# 2. ADAPTADOR: CASSCF -> Wavefunction de RL-VMC (JAX Puro de 64 bits)
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

def cargar_ansatz_casscf(npz_path):
    data = np.load(npz_path)
    meta = json.loads(str(data["metadata_json"]))
    cfg = meta["config"]
    omega, n_up, n_dn, n_basis = cfg["omega"], cfg["n_up"], cfg["n_down"], cfg["n_basis"]
    n_core = cfg.get("n_core", 0)
    n_active = cfg.get("n_active_orbitals", 0)
    
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
        
    occ_up_idx = jnp.array(occ_up_list)
    occ_dn_idx = jnp.array(occ_dn_list)
    sign_c, log_c = jnp.sign(ci_coeffs), jnp.log(jnp.abs(ci_coeffs) + 1e-15)
    
    @jax.jit
    def ansatz_apply(params, x):
        x_1d = x[..., 0] 
        def evaluate_single(x_single):
            x_up, x_dn = x_single[:n_up], x_single[n_up:]
            basis_up = build_ho_basis_jax(x_up, n_basis, omega)
            basis_dn = build_ho_basis_jax(x_dn, n_basis, omega)
            
            def eval_det(idx_up, idx_dn, s_c, l_c):
                # Añadimos un minúsculo jitter para estabilizar la inversa de matrices
                mat_up = basis_up @ orbitals[:, idx_up] + jnp.eye(n_up) * 1e-12
                mat_dn = basis_dn @ orbitals[:, idx_dn] + jnp.eye(n_dn) * 1e-12
                s_u, l_u = jnp.linalg.slogdet(mat_up)
                s_d, l_d = jnp.linalg.slogdet(mat_dn)
                return s_c * s_u * s_d, l_c + l_u + l_d
            
            signs, log_mags = jax.vmap(eval_det)(occ_up_idx, occ_dn_idx, sign_c, log_c)
            max_log = jnp.max(log_mags)
            psi_val = jnp.sum(signs * jnp.exp(log_mags - max_log))
            
            # Recorte seguro para evitar derivadas infinitas (log(0))
            log_psi = max_log + jnp.log(jnp.abs(psi_val) + 1e-15)
            return jnp.clip(log_psi, -1000.0, 1000.0)
        
        return jax.vmap(evaluate_single)(x_1d) if x_1d.ndim == 2 else evaluate_single(x_1d)
        
    return ansatz_apply, cfg

# =====================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN 
# =====================================================================
def termalizar_caminantes(trainer, key, walkers_init, params_psi, p_act, n_steps):
    """Fase obligatoria: Llevamos los caminantes a los pozos físicos sin recolectar basura."""
    print(f"-> Thermalizing walkers for {n_steps} steps...")
    curr_logp, curr_grad = trainer.get_logp_and_grad(params_psi, walkers_init)
    
    def step_fn(carry, _):
        rng, w, logp, grad = carry
        rng, subk = jax.random.split(rng)
        out = trainer.td3_update_all_electrons(subk, w, logp, grad, params_psi, p_act, 0.0)
        return (rng, out[0], out[1], out[2]), None
        
    final_carry, _ = jax.lax.scan(step_fn, (key, walkers_init, curr_logp, curr_grad), jnp.arange(n_steps))
    return final_carry[1], final_carry[2], final_carry[3]

def entrenar_actor_td3(trainer, key, w_term, logp_term, grad_term, params_psi, n_sweeps, n_elec, n_up, n_dn):
    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0
    state_dim = 4 + 5 + 3 * max_same + 3 * max_opp
    action_dim = 2

    key, k_act, k_c1, k_c2 = jax.random.split(key, 4)
    dummy_state = jnp.zeros((1, state_dim))
    dummy_action = jnp.zeros((1, action_dim))

    p_act = trainer.actor_model.init(k_act, dummy_state)
    p_c1 = trainer.critic_model.init(k_c1, dummy_state, dummy_action)
    p_c2 = trainer.critic_model.init(k_c2, dummy_state, dummy_action)
    t_act, t_c1, t_c2 = jax.tree_util.tree_map(lambda x: x.copy(), p_act), p_c1, p_c2

    tx_act, tx_crit = optax.adam(learning_rate=1e-5), optax.adam(learning_rate=3e-5)
    opt_act, opt_c1, opt_c2 = tx_act.init(p_act), tx_crit.init(p_c1), tx_crit.init(p_c2)
    trainer.opt_a, trainer.opt_c = tx_act, tx_crit

    buffer = ReplayBuffer(capacity=50000, state_dim=state_dim, action_dim=action_dim)
    batch_size = 256

    # DICCIONARIO PARA MÉTRICAS
    hist = {
        "reward": [], "exploit": [], "entropy": [], "explore": [], 
        "loss_critic": [], "loss_actor": [], "accept": [], "disp": []
    }

    print(f"\n--- Starting TD3 Training ({n_sweeps} Sweeps) ---")
    walkers, curr_logp, curr_grad = w_term, logp_term, grad_term

    for sweep in range(n_sweeps):
        key, subk, k_train = jax.random.split(key, 3)
        exp_std = 0.2 * (1.0 - sweep / max(1, n_sweeps))

        out = trainer.td3_update_all_electrons(
            subk, walkers, curr_logp, curr_grad, params_psi, p_act, exp_std
        )
        walkers, curr_logp, curr_grad = out[0], out[1], out[2]
        
        # Aplanar para Replay Buffer
        s_flat = np.array(out[3]).reshape(-1, state_dim)
        a_flat = np.array(out[4]).reshape(-1, action_dim)
        r_flat = np.repeat(np.array(out[5]), n_elec)
        ns_flat = np.array(out[9]).reshape(-1, state_dim)
        buffer.add_batch(s_flat, a_flat, r_flat, ns_flat)
        
        # Registrar componentes de la recompensa CDLB
        hist["reward"].append(float(jnp.mean(out[5])))
        hist["exploit"].append(float(jnp.mean(out[6])))
        hist["entropy"].append(float(jnp.mean(out[7])))
        hist["explore"].append(float(jnp.mean(out[8])))
        hist["accept"].append(float(jnp.mean(out[10])))
        hist["disp"].append(float(jnp.mean(out[11])))

        l1, l_act = 0.0, 0.0
        if buffer.size > batch_size:
            batch = buffer.sample(batch_size)
            update_actor = jnp.array(sweep % 2 == 0)
            p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_act = trainer.train_networks(
                k_train, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_actor
            )
        
        hist["loss_critic"].append(float(l1))
        hist["loss_actor"].append(float(l_act))

        if sweep % 500 == 0 or sweep == n_sweeps - 1:
            print(f"Sweep {sweep:4d}/{n_sweeps} | Reward: {hist['reward'][-1]:.4f} | Accept: {hist['accept'][-1]:.3f} | Disp: {hist['disp'][-1]:.3f}")

    print("--- Training Completed ---\n")
    return p_act, hist

def run_evaluation_chain(trainer, key, walkers, params_psi, params_actor, n_steps, chunk_size=10000):
    """
    Ejecuta la cadena MCMC en fragmentos (chunks) para evitar errores OOM (Out Of Memory) en la GPU.
    Descarga la trayectoria a la CPU en tiempo real.
    """
    curr_logp, curr_grad = trainer.get_logp_and_grad(params_psi, walkers)
    
    # 1. Definir la función internar SIN el decorador @jax.jit
    def run_chunk_fn(rng, w, logp, grad, current_chunk_size):
        def sweep_fn(carry, _):
            r, cw, clp, cg = carry
            r, subkey = jax.random.split(r)
            out = trainer.td3_update_all_electrons(
                subkey, cw, clp, cg, params_psi, params_actor, 0.0
            )
            return (r, out[0], out[1], out[2]), (out[0], jnp.stack([jnp.mean(out[10].astype(jnp.float32)), jnp.mean(out[11])]))
            
        final_state, (traj, metrics_trace) = jax.lax.scan(sweep_fn, (rng, w, logp, grad), jnp.arange(current_chunk_size))
        return final_state, traj, metrics_trace[:, 0], metrics_trace[:, 1]

    # 2. Compilar manualmente pasándole la función como primer argumento y usando static_argnums
    # El índice 4 corresponde a 'current_chunk_size' (0: rng, 1: w, 2: logp, 3: grad, 4: current_chunk_size)
    run_chunk = jax.jit(run_chunk_fn, static_argnums=4)

    # 3. Bucle externo en Python para gestionar la memoria RAM
    all_traj = []
    all_acc = []
    all_disp = []
    
    state = (key, walkers, curr_logp, curr_grad)
    steps_remaining = n_steps
    
    while steps_remaining > 0:
        current_steps = min(chunk_size, steps_remaining)
        
        # Ejecutar en GPU
        state, traj_chunk, acc_chunk, disp_chunk = run_chunk(state[0], state[1], state[2], state[3], current_steps)
        
        # Transferir inmediatamente a RAM (CPU) para vaciar la VRAM
        all_traj.append(np.array(jax.device_get(traj_chunk)))
        all_acc.append(np.array(jax.device_get(acc_chunk)))
        all_disp.append(np.array(jax.device_get(disp_chunk)))
        
        steps_remaining -= current_steps
        
    # Concatenar todos los fragmentos almacenados en la RAM
    return np.concatenate(all_traj, axis=0), np.concatenate(all_acc, axis=0), np.concatenate(all_disp, axis=0)

# =====================================================================
# 4. GRÁFICAS Y MÉTRICAS (ENGLISH)
# =====================================================================
def plot_convergencia_energia(traj_mala, traj_rl, cfg, out_name="Article_Convergence.pdf"):
    omega, g, softening, n_elec = cfg["omega"], cfg.get("g", 4.0), cfg.get("softening", 0.1), cfg["n_up"]+cfg["n_down"]
    
    @jax.jit
    def calc_potential_traj(x_traj):
        v_harm = 0.5 * (omega**2) * jnp.sum(x_traj**2, axis=-1)
        diff_ee = x_traj[:, :, :, None] - x_traj[:, :, None, :]
        dist_sq = diff_ee**2 + softening**2
        mask_ee = jnp.triu(jnp.ones((n_elec, n_elec)), k=1)
        v_int = g * jnp.sum((1.0 / jnp.sqrt(dist_sq)) * mask_ee, axis=(2, 3))
        return v_harm + v_int

    n_ver = min(10000, traj_mala.shape[0])
    v_mala = calc_potential_traj(traj_mala[:n_ver, ..., 0])
    v_rl = calc_potential_traj(traj_rl[:n_ver, ..., 0])
    
    # --- EXPORTAR A CSV ---
    df_energy = pd.DataFrame({
        "step": np.arange(n_ver),
        "energy_mala": np.array(jnp.mean(v_mala, axis=1)),
        "energy_rl": np.array(jnp.mean(v_rl, axis=1))
    })
    df_energy.to_csv("data_convergence.csv", index=False)
    # ----------------------

    plt.figure(figsize=(10, 5))
    plt.plot(df_energy["energy_mala"], color='#e74c3c', alpha=0.9, lw=2, label='Standard MALA')
    plt.plot(df_energy["energy_rl"], color='#2ecc71', alpha=0.9, lw=2, label='RL-MALA')
    plt.title("Burn-in Convergence Time to Target Distribution", weight='bold')
    plt.xlabel("MCMC Steps")
    plt.ylabel("Average Potential Energy (Hartree)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)

def plot_resultados_articulo(traj_mala, acc_mala, traj_rl, acc_rl, cfg, df_exact_density=None):
    n_up, n_down, omega = cfg["n_up"], cfg["n_down"], cfg["omega"]
    burn_in = int(0.2 * traj_mala.shape[0])
    x_mala = np.array(traj_mala[burn_in:]).reshape(-1, n_up + n_down)
    x_rl = np.array(traj_rl[burn_in:]).reshape(-1, n_up + n_down)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    espacio, potencial = np.linspace(-15, 15, 500), 0.5 * (omega**2) * (np.linspace(-15, 15, 500)**2)
    
    # --- EXPORTAR DENSIDADES A CSV ---
    bins = np.linspace(-15, 15, 120)
    d_tot_m, _ = np.histogram(x_mala.flatten(), bins=120, range=(-15, 15), density=True)
    d_up_m, _ = np.histogram(x_mala[:, :n_up].flatten(), bins=120, range=(-15, 15), density=True)
    d_dn_m, _ = np.histogram(x_mala[:, n_up:].flatten(), bins=120, range=(-15, 15), density=True)

    d_tot_rl, _ = np.histogram(x_rl.flatten(), bins=120, range=(-15, 15), density=True)
    d_up_rl, _ = np.histogram(x_rl[:, :n_up].flatten(), bins=120, range=(-15, 15), density=True)
    d_dn_rl, _ = np.histogram(x_rl[:, n_up:].flatten(), bins=120, range=(-15, 15), density=True)

    df_dens = pd.DataFrame({
        "x_bin": bins,
        "mala_tot": d_tot_m * (n_up + n_down),
        "mala_up": d_up_m * n_up,
        "mala_dn": d_dn_m * n_down,
        "rl_tot": d_tot_rl * (n_up + n_down),
        "rl_up": d_up_rl * n_up,
        "rl_dn": d_dn_rl * n_down
    })
    df_dens.to_csv("data_densities.csv", index=False)
    # ---------------------------------

    def plot_dens(ax, d_tot, d_up, d_dn, title):
        ax.plot(bins, d_tot * (n_up + n_down), 'k-', lw=2.5, label='Sampled Total Density')
        ax.fill_between(bins, d_up * n_up, color='#3498db', alpha=0.5, label=r'Sampled Spin $\uparrow$')
        ax.fill_between(bins, d_dn * n_down, color='#e74c3c', alpha=0.5, label=r'Sampled Spin $\downarrow$')

        if df_exact_density is not None:
            x_ex = df_exact_density["x"].values
            ax.plot(x_ex, df_exact_density["rho_total"].values, 'k:', lw=2.5, alpha=0.9, label='Exact CASSCF Total')
            ax.plot(x_ex, df_exact_density["rho_up"].values, color='#2980b9', linestyle=':', lw=2.5, alpha=0.9, label=r'Exact Spin $\uparrow$')
            ax.plot(x_ex, df_exact_density["rho_down"].values, color='#c0392b', linestyle=':', lw=2.5, alpha=0.9, label=r'Exact Spin $\downarrow$')

        ax_pot = ax.twinx()
        ax_pot.plot(espacio, potencial, '--', color='gray', lw=1.5, alpha=0.6, label='V(x)')
        ax_pot.set_yticks([]) 
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel("x (Bohr)")
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    plot_dens(axes[0], d_tot_m, d_up_m, d_dn_m, "Standard MALA")
    plot_dens(axes[1], d_tot_rl, d_up_rl, d_dn_rl, "Adaptive Policy RL-MALA")
    plt.tight_layout()
    plt.savefig("Article_Densities.pdf", dpi=300)
    
    # --- EXPORTAR OBSERVABLES A CSV ---
    msd_mala = np.mean((traj_mala - traj_mala[0])**2, axis=(1, 2, 3))
    msd_rl = np.mean((traj_rl - traj_rl[0])**2, axis=(1, 2, 3))
    
    df_obs = pd.DataFrame({
        "step": np.arange(len(acc_mala)),
        "acc_mala": acc_mala,
        "acc_rl": acc_rl,
        "msd_mala": msd_mala,
        "msd_rl": msd_rl
    })
    # Guarda solo 1 de cada 100 pasos
    df_obs.iloc[::100].to_csv("data_observables.csv", index=False)
    # ----------------------------------

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes2[0].plot(acc_mala, color='#e74c3c', alpha=0.6, label='MALA')
    axes2[0].plot(acc_rl, color='#2ecc71', alpha=0.6, label='RL-MALA')
    axes2[0].set_title("MCMC Acceptance Rate", weight='bold')
    axes2[0].axhline(0.574, color='k', linestyle='--', label='Theoretical Optimum')
    axes2[0].legend()
    
    axes2[1].plot(msd_mala, color='#e74c3c', lw=2, label='Standard MALA')
    axes2[1].plot(msd_rl, color='#2ecc71', lw=2, label='RL-MALA')
    axes2[1].set_title("Wigner Topological Mobility (MSD)", weight='bold')
    axes2[1].set_xlabel("MCMC Steps")
    axes2[1].set_ylabel(r"MSD (Bohr$^2$)")
    axes2[1].legend()
    
    plt.tight_layout()
    plt.savefig("Article_Ergodicity_Observables.pdf", dpi=300)

def plot_metricas_entrenamiento(hist, out_name="Article_Training_Metrics.pdf"):
    # --- EXPORTAR A CSV ---
    pd.DataFrame(hist).to_csv("data_training_metrics.csv", index_label="sweep")
    # ----------------------
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    steps = np.arange(len(hist["reward"]))

    axes[0].plot(steps, hist["reward"], label="Total Reward", color="black", lw=2)
    axes[0].plot(steps, hist["exploit"], label="Exploitation", alpha=0.7)
    axes[0].plot(steps, hist["entropy"], label="Entropy", alpha=0.7)
    axes[0].plot(steps, hist["explore"], label="Exploration", alpha=0.7)
    axes[0].set_title("CDLB Reward Components", weight="bold")
    axes[0].set_ylabel("Reward Value")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(steps, hist["loss_critic"], label="Critic 1 Loss", color="#e74c3c")
    ax_actor = axes[1].twinx()
    ax_actor.plot(steps, hist["loss_actor"], label="Actor Loss", color="#3498db")
    axes[1].set_title("Network Losses", weight="bold")
    axes[1].set_ylabel("Critic MSE", color="#e74c3c")
    ax_actor.set_ylabel("Actor Loss", color="#3498db")
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, hist["accept"], label="Acceptance Rate", color="#2ecc71")
    ax_disp = axes[2].twinx()
    ax_disp.plot(steps, hist["disp"], label="Proposed Disp.", color="#9b59b6")
    axes[2].axhline(0.574, color='k', linestyle='--', alpha=0.5, label="Optimum")
    axes[2].set_title("Training Phase MCMC Diagnostics", weight="bold")
    axes[2].set_xlabel("TD3 Sweeps")
    axes[2].set_ylabel("Acceptance Rate", color="#2ecc71")
    ax_disp.set_ylabel("Displacement (Bohr)", color="#9b59b6")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_name, dpi=300)

def plot_walker_ergodicity(df, n_up, title, out_name, df_exact_density=None):
    import matplotlib.gridspec as gridspec
    
    # Submuestreo visual solo para la gráfica de líneas (para que no tarde una eternidad en renderizar)
    df_plot = df.iloc[::100] 
    x_vals = df_plot["step"] if "step" in df_plot.columns else df_plot.index
    
    # Averiguar n_down leyendo cuántas columnas de electrones hay
    elec_cols = [c for c in df.columns if c != "step"]
    n_elec = len(elec_cols)
    n_down = n_elec - n_up
    
    # Crear la figura y el GridSpec (Panel principal 80% ancho, Histograma 20%)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    
    ax_main = plt.subplot(gs[0])
    ax_hist = plt.subplot(gs[1], sharey=ax_main)
    
    # Listas para recolectar TODAS las posiciones (sin submuestrear) para un histograma fiel
    all_up = []
    all_dn = []
    
    for col in elec_cols:
        elec_idx = int(col.split('_')[1])
        color = '#3498db' if elec_idx < n_up else '#e74c3c' 
        
        # Trazar la trayectoria en el panel principal
        ax_main.plot(x_vals, df_plot[col], color=color, alpha=0.6, linewidth=0.5)
        
        # Acumular para el histograma
        if elec_idx < n_up:
            all_up.extend(df[col].values)
        else:
            all_dn.extend(df[col].values)
            
    ax_main.set_title(f"Walker Trajectory - {title}", weight='bold')
    ax_main.set_xlabel("MCMC Steps")
    ax_main.set_ylabel("Electron Position (Bohr)")
    ax_main.grid(alpha=0.3)
    
    # --- PANEL DERECHO: Histograma Marginal ---
    bins = np.linspace(-15, 15, 150)
    
    # Histogramas muestreados (density=True normaliza el área a 1.0)
    ax_hist.hist(all_up, bins=bins, orientation='horizontal', density=True, 
                 color='#3498db', alpha=0.4, label=r'Sampled $\uparrow$')
    ax_hist.hist(all_dn, bins=bins, orientation='horizontal', density=True, 
                 color='#e74c3c', alpha=0.4, label=r'Sampled $\downarrow$')
                 
    # Superponer la densidad analítica exacta si se proporciona
    if df_exact_density is not None:
        x_ex = df_exact_density["x"].values
        # Normalizamos dividiendo entre n_up/n_down para que su área sea 1.0 igual que el histograma
        rho_up_norm = df_exact_density["rho_up"].values / n_up
        rho_dn_norm = df_exact_density["rho_down"].values / n_down
        
        ax_hist.plot(rho_up_norm, x_ex, color='#2980b9', linestyle='--', lw=2, label=r'Exact $\uparrow$')
        ax_hist.plot(rho_dn_norm, x_ex, color='#c0392b', linestyle='--', lw=2, label=r'Exact $\downarrow$')
        
    ax_hist.set_title("Marginal Density", weight='bold')
    ax_hist.set_xlabel("Normalized $\\rho(x)$")
    ax_hist.tick_params(axis='y', labelleft=False) # Ocultar los ticks Y para mantenerlo limpio
    ax_hist.grid(alpha=0.3)
    ax_hist.legend(loc='upper right', fontsize=9)
    
    # Guardar
    # fig.tight_layout() no funciona bien con gridspec a veces, bbox_inches es más seguro
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig) # Liberar memoria

def plot_kl_divergence_analytic(traj_mala, traj_rl, cfg, df_exact_density, out_name="Article_KL_Divergence.pdf"):
    print("-> Calculando evolución de Divergencia KL resuelta por espín...")
    
    n_up, n_down = cfg["n_up"], cfg["n_down"]
    bins, hist_range = 200, (-15.0, 15.0)
    edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    epsilon = 1e-12
    
    p_tot_exact = np.interp(centers, df_exact_density["x"].values, df_exact_density["p_total"].values) + epsilon
    p_up_exact = np.interp(centers, df_exact_density["x"].values, df_exact_density["p_up"].values) + epsilon
    p_dn_exact = np.interp(centers, df_exact_density["x"].values, df_exact_density["p_down"].values) + epsilon
    
    p_tot_exact /= np.sum(p_tot_exact)
    p_up_exact /= np.sum(p_up_exact)
    p_dn_exact /= np.sum(p_dn_exact)
    
    def get_kl_evolutions(traj):
        steps = np.arange(5000, traj.shape[0], 5000)
        kl_tot, kl_up, kl_dn = [], [], []
        burn_in = 5000
        
        for t in steps:
            if t <= burn_in:
                kl_tot.append(np.nan); kl_up.append(np.nan); kl_dn.append(np.nan)
                continue
                
            data_t = np.array(traj[burn_in:t]).reshape(-1, n_up + n_down)
            
            h_tot, _ = np.histogram(data_t.flatten(), bins=edges, density=True)
            h_up, _ = np.histogram(data_t[:, :n_up].flatten(), bins=edges, density=True)
            h_dn, _ = np.histogram(data_t[:, n_up:].flatten(), bins=edges, density=True)
            
            p_tot = (h_tot + epsilon) / np.sum(h_tot + epsilon)
            p_up = (h_up + epsilon) / np.sum(h_up + epsilon)
            p_dn = (h_dn + epsilon) / np.sum(h_dn + epsilon)
            
            kl_tot.append(np.sum(p_tot * np.log(p_tot / p_tot_exact)))
            kl_up.append(np.sum(p_up * np.log(p_up / p_up_exact)))
            kl_dn.append(np.sum(p_dn * np.log(p_dn / p_dn_exact)))
            
        return steps, kl_tot, kl_up, kl_dn

    steps_m, kl_tot_m, kl_up_m, kl_dn_m = get_kl_evolutions(traj_mala)
    steps_rl, kl_tot_rl, kl_up_rl, kl_dn_rl = get_kl_evolutions(traj_rl)
    
    # --- EXPORTAR A CSV ---
    df_kl = pd.DataFrame({
        "step": steps_m,
        "kl_tot_mala": kl_tot_m,
        "kl_up_mala": kl_up_m,
        "kl_dn_mala": kl_dn_m,
        "kl_tot_rl": kl_tot_rl,
        "kl_up_rl": kl_up_rl,
        "kl_dn_rl": kl_dn_rl
    })
    df_kl.to_csv("data_kl_divergence.csv", index=False)
    # ----------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    def plot_panel(ax, kl_m, kl_rl, title):
        ax.plot(steps_m, kl_m, color='#e74c3c', lw=2.5, label='Standard MALA')
        ax.plot(steps_rl, kl_rl, color='#2ecc71', lw=2.5, label='RL-MALA')
        ax.set_yscale('log')
        ax.set_title(title, weight='bold', fontsize=14)
        ax.set_xlabel("MCMC Samples Evaluated", fontsize=12)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=11)

    plot_panel(axes[0], kl_tot_m, kl_tot_rl, "Total Density $D_{KL}$")
    plot_panel(axes[1], kl_up_m, kl_up_rl, r"Spin $\uparrow$ Density $D_{KL}$")
    plot_panel(axes[2], kl_dn_m, kl_dn_rl, r"Spin $\downarrow$ Density $D_{KL}$")
    
    axes[0].set_ylabel(r"KL Divergence $D_{KL}(P_t \parallel P_{exact})$", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_name, dpi=300)

########################################################################
#################################### mainsillos ########################
########################################################################

if __name__ == "__main__":
    npz_path = os.path.join(UHF1D_SRC, "../results_N4/g_4.0/cas_n4_g4_omega0.1_soft0.1_b32_c0-a4/casscf_state.npz")
    print(f"1. Loading CASSCF Ansatz from: {npz_path}")
    ansatz_apply, cfg = cargar_ansatz_casscf(npz_path)
    n_elec, n_up, n_dn = cfg["n_up"] + cfg["n_down"], cfg["n_up"], cfg["n_down"]
    
    # =====================================================================
    # INTEGRACIÓN CON LA PAQUETERÍA DEL ASESOR PARA GROUND TRUTH ANALÍTICO
    # =====================================================================
    print("-> Extrayendo la Densidad Analítica CASSCF Exacta...")
    casscf_outdir = "casscf_exact_analysis"
    overrides = {
        "n_up": n_up, 
        "n_down": n_dn, 
        "omega": cfg["omega"], 
        "g": cfg.get("g", 4.0), 
        "a": cfg.get("softening", 0.1)
    }

    # Creamos un grid extendido para que cuadre con tus gráficas (-15 a 15)
    grid_config = GridConfig(x_min=-15.0, x_max=15.0, n_grid=500)

    analyze_orbital_state(
        path=npz_path, 
        outdir=casscf_outdir, 
        method="casscf", 
        grid=grid_config, 
        pin_x=None, 
        overrides=overrides
    )

    # Cargamos el CSV que su paquetería generó
    df_exact_density = pd.read_csv(os.path.join(casscf_outdir, "one_body_density.csv"))
    print("-> Densidad analítica cargada con éxito.")

    n_walkers = 100  
    n_steps_eval = 500_000   # ¡Aumentado de 20k a 100k! Deja que la estadística haga su trabajo.
    n_sweeps_train = 20_000
    
    # HIPERPARÁMETROS DESBLOQUEADOS (Para permitir cruce de nodos)
    tau_base_mala = 0.25
    tau_base_rl = 0.1      
    actor_amp = 0.1        # Reducido de 0.25. Un empuje direccional más sutil y quirúrgico.
    penalty = 0.25          # Aumentado de 0.1. ¡Crucial! Obliga a la red a no actuar dentro de los pozos.
    sigma_scale = 1.1       # Reducido de 1.8. Mantén la varianza de la propuesta casi idéntica a Langevin puro.
    max_jump = 2.0          # Reducido de 3.0. Evita pasos catastróficos que dañen la aceptación local.
    
    key = jax.random.PRNGKey(43)
    key, subk = jax.random.split(key)
    walkers_init = jax.random.normal(subk, (n_walkers, n_elec, 1)) * 3.0

    # print(f"Initial positions: {walkers_init}")

    print("2. Configuring Trainers...")
    trainer_mala = TD3Trainer(
        ansatz_apply, n_elec, n_up, n_dn, tau=tau_base_mala, actor_amp=0.0, actor_type="full",
        lambda_=penalty, sigma_action_scale=sigma_scale, max_phys_jump_factor=max_jump, drift_limit=10.0, action_rev_limit=1.0
    )
    trainer_rl = TD3Trainer(
        ansatz_apply, n_elec, n_up, n_dn, tau=tau_base_rl, actor_amp=actor_amp, actor_type="full",
        lambda_=penalty, sigma_action_scale=sigma_scale, max_phys_jump_factor=max_jump, drift_limit=10.0, action_rev_limit=1.0
    )

    key, subk = jax.random.split(key)
    dummy_p_act = trainer_mala.actor_model.init(subk, jnp.zeros((1, 4 + 5 + 3 * (max(max(n_up-1,0), max(n_dn-1,0)) + max(n_up, n_dn)))))
    
    w_term, logp_term, grad_term = termalizar_caminantes(
        trainer_mala, subk, walkers_init, params_psi=None, p_act=dummy_p_act, n_steps=2000
    )
    
    key, subk = jax.random.split(key)
    params_actor, hist_train = entrenar_actor_td3(
        trainer_rl, subk, w_term, logp_term, grad_term, params_psi=None, 
        n_sweeps=n_sweeps_train, n_elec=n_elec, n_up=n_up, n_dn=n_dn
    )

    print("3. Executing MALA Evaluation (Baseline)...")
    offset = jnp.arange(n_elec).reshape(1, n_elec, 1) * 1.0
    walkers_eval_init = jnp.ones((n_walkers, n_elec, 1)) * 5.0 + offset

    key, subk = jax.random.split(key)
    traj_mala, acc_mala, disp_mala = run_evaluation_chain(
        trainer_mala, subk, walkers_eval_init, params_psi=None, params_actor=dummy_p_act, n_steps=n_steps_eval
    )
    
    print("4. Executing RL-MALA Evaluation...")
    key, subk = jax.random.split(key)
    traj_rl, acc_rl, disp_rl = run_evaluation_chain(
        trainer_rl, subk, walkers_eval_init, params_psi=None, params_actor=params_actor, n_steps=n_steps_eval
    )

    # =====================================================================
    # LIBERACIÓN DE VRAM: Mover todas las trayectorias a la RAM (CPU)
    # =====================================================================
    print("-> Transfiriendo tensores masivos de GPU a CPU RAM...")
    import jax
    import numpy as np
    
    traj_mala = np.array(jax.device_get(traj_mala))
    traj_rl = np.array(jax.device_get(traj_rl))
    acc_mala = np.array(jax.device_get(acc_mala))
    acc_rl = np.array(jax.device_get(acc_rl))
    
    # =====================================================================

    print("5. Generating paper figures...")
    
    # Ahora las funciones de graficado usarán NumPy tradicional en la RAM
    plot_resultados_articulo(traj_mala, acc_mala, traj_rl, acc_rl, cfg, df_exact_density=df_exact_density)
    plot_convergencia_energia(traj_mala, traj_rl, cfg)
    plot_metricas_entrenamiento(hist_train)
    plot_kl_divergence_analytic(traj_mala, traj_rl, cfg, df_exact_density)

    # Guardar la trayectoria temporal de un solo walker (índice 0)
    print("-> Exportando trayectoria del caminante 0 para análisis de ergodicidad...")
    
    # Extraemos: [todos_los_pasos, walker_0, todos_los_electrones, dimension_x]
    df_walker_mala = pd.DataFrame(traj_mala[:, 0, :, 0], columns=[f"elec_{i}" for i in range(n_elec)])
    df_walker_mala.to_csv("data_single_walker_mala.csv", index_label="step")

    df_walker_rl = pd.DataFrame(traj_rl[:, 0, :, 0], columns=[f"elec_{i}" for i in range(n_elec)])
    df_walker_rl.to_csv("data_single_walker_rl.csv", index_label="step")

    plot_walker_ergodicity(df_walker_mala, overrides["n_up"], "MALA", "Article_walker_MALA.pdf", df_exact_density=df_exact_density)
    plot_walker_ergodicity(df_walker_rl, overrides["n_up"], "RL", "Article_walker_RL.pdf", df_exact_density=df_exact_density)

    print("All done. Figures saved successfully.")