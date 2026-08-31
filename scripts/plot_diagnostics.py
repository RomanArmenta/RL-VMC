import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_autocorrelation(x, max_lag=200):
    """Calcula la función de autocorrelación discreta hasta un lag máximo."""
    x = x - np.mean(x)
    var = np.sum(x**2)
    if var == 0:
        return np.zeros(max_lag)
    acf = [np.sum(x[:len(x)-k] * x[k:]) / var for k in range(max_lag)]
    return np.array(acf)

def plot_telemetry(folder, system, out_dir):
    """Genera el panel de telemetría del entrenamiento RL."""
    csv_path = os.path.join(folder, f"{system}_training_metrics_RL-MALA.csv")
    if not os.path.exists(csv_path):
        print(f"  [!] No telemetry data found at {csv_path}. Skipping.")
        return

    print("  Generating Training Telemetry panel...")
    df = pd.read_csv(csv_path)
    steps = df['Step']

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL-MALA Training Telemetry", fontsize=16)

    # 1. Reward vs Actor Loss
    ax1 = axs[0, 0]
    ax1.plot(steps, df['Reward'], color='green', label='Mean Reward')
    ax1.set_ylabel("Reward", color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.set_xlabel("Steps")
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(steps, df['Actor_Loss'], color='purple', alpha=0.6, label='Actor Loss')
    ax1_twin.set_ylabel("Policy Loss", color='purple')
    ax1_twin.tick_params(axis='y', labelcolor='purple')
    ax1.set_title("Reward & Policy Loss Convergence")
    ax1.grid(True, alpha=0.3)

    # 2. EJD vs Acceptance
    ax2 = axs[0, 1]
    ax2.plot(steps, df['EJD'], color='blue', label='Expected Jump Dist (EJD)')
    ax2.set_ylabel("EJD (Bohrs)", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_xlabel("Steps")

    ax2_twin = ax2.twinx()
    ax2_twin.plot(steps, df['Acceptance'], color='orange', alpha=0.8, label='Acceptance Rate')
    ax2_twin.set_ylabel("Acceptance Ratio", color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.set_ylim(0, 1)
    ax2.set_title("Jump Dynamics")
    ax2.grid(True, alpha=0.3)

    # 3. Reward Breakdown
    ax3 = axs[1, 0]
    if 'reward_dist' in df.columns:
        ax3.plot(steps, df['reward_dist'], label='Distance Reward', color='blue')
        ax3.plot(steps, -df['reward_penalty'], label='Grad Penalty (Negative)', color='red')
        ax3.plot(steps, -df['penalty_for_reject'], label='Rejection Penalty', color='orange')
        ax3.set_title("Reward Signal Breakdown")
        ax3.set_xlabel("Steps")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Energy Convergence (Zoom in to see stability)
    ax4 = axs[1, 1]
    # Ignorar el warmup para no deformar la gráfica con energías muy altas
    safe_e = df['E_mean'].clip(lower=df['E_mean'].median() - 2, upper=df['E_mean'].median() + 2)
    ax4.plot(steps, safe_e, color='black', alpha=0.8)
    ax4.set_title("Local Energy Convergence")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Energy (Ha)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(out_dir, "training_telemetry.pdf"), dpi=300)
    plt.close()

def plot_acf_and_traces(folder, system, out_dir):
    """Genera gráficas de Función de Autocorrelación (ACF) y Trace Plots."""
    mala_csv = os.path.join(folder, f"{system}_measurement_data_MALA.csv")
    rl_csv = os.path.join(folder, f"{system}_measurement_data_RL-MALA.csv")
    
    print("  Generating Autocorrelation & Trace plots...")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    max_lag = 200
    
    if os.path.exists(mala_csv):
        df_mala = pd.read_csv(mala_csv)
        e_mala = df_mala['E_Local_Mean'].values
        acf_mala = compute_autocorrelation(e_mala, max_lag)
        axs[0].plot(acf_mala, label='MALA', color='blue', linewidth=2)
        axs[1].plot(e_mala[:1000], label='MALA', color='blue', alpha=0.7)

    if os.path.exists(rl_csv):
        df_rl = pd.read_csv(rl_csv)
        e_rl = df_rl['E_Local_Mean'].values
        acf_rl = compute_autocorrelation(e_rl, max_lag)
        axs[0].plot(acf_rl, label='RL-MALA', color='red', linestyle='--', linewidth=2)
        axs[1].plot(e_rl[:1000], label='RL-MALA', color='red', alpha=0.7)

    axs[0].set_title("Energy Autocorrelation Function (ACF)")
    axs[0].set_xlabel("Lag (Steps)")
    axs[0].set_ylabel(r"Autocorrelation $\rho(k)$")
    axs[0].axhline(0, color='black', linestyle='-', linewidth=1)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].set_title("Energy Trace Plot (First 1000 Steps)")
    axs[1].set_xlabel("Measurement Steps")
    axs[1].set_ylabel("Batch Mean Energy (Ha)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acf_and_traces.pdf"), dpi=300)
    plt.close()

def plot_step_displacements(folder, system, out_dir):
    """Calcula y grafica el histograma de los desplazamientos físicos reales de los caminantes."""
    traj_path = os.path.join(folder, f"trajectories_{system}.npz")
    if not os.path.exists(traj_path):
        print(f"  [!] No trajectory file found at {traj_path}. Skipping displacements.")
        return

    print("  Generating Step Displacement Histograms...")
    data = np.load(traj_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))

    if 'walkers_mala_mala' in data:
        w_mm = data['walkers_mala_mala']
        # Calcular el vector de desplazamiento entre pasos consecutivos
        dr_mm = np.linalg.norm(w_mm[1:] - w_mm[:-1], axis=-1).flatten()
        ax.hist(dr_mm, bins=150, density=True, alpha=0.6, color='blue', label='MALA', range=(0, 1.0))

    if 'walkers_rl_rl' in data:
        w_rr = data['walkers_rl_rl']
        dr_rr = np.linalg.norm(w_rr[1:] - w_rr[:-1], axis=-1).flatten()
        ax.hist(dr_rr, bins=150, density=True, alpha=0.6, color='red', label='RL-MALA', range=(0, 1.0))

    ax.set_title("Physical Step Displacement Distribution", fontsize=14)
    ax.set_xlabel(r"Step Magnitude $\|\Delta \mathbf{R}\|$ (Bohrs)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "step_displacements.pdf"), dpi=300)
    plt.close()
    data.close()

def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for RL-VMC.")
    parser.add_argument("--folder_name", type=str, required=True, help="Path to the results directory")
    parser.add_argument("--system", type=str, required=True, help="Name of the physical system (e.g., B, LiH)")
    args = parser.parse_args()

    # Create output directory
    out_dir = os.path.join(args.folder_name, "diagnostics_plot")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving diagnostic plots to: {out_dir}")

    # Generate plots
    plot_telemetry(args.folder_name, args.system, out_dir)
    plot_acf_and_traces(args.folder_name, args.system, out_dir)
    plot_step_displacements(args.folder_name, args.system, out_dir)

    print("\nDiagnostics complete.")

if __name__ == "__main__":
    main()