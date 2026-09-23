import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_dir", type=str, required=True, help="Carpeta del sistema (ej. results_N2_g0.1)")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Training Diagnostics: {os.path.basename(args.sys_dir)}", weight='bold', fontsize=16)

    mala_dirs = [d for d in glob.glob(os.path.join(args.sys_dir, "MALA_tau*")) if os.path.isdir(d)]
    td3_dir = os.path.join(args.sys_dir, "TD3")

    colors = sns.color_palette("Greys", len(mala_dirs) + 2)[1:] # Escala de grises para MALAs

    for i, m_dir in enumerate(sorted(mala_dirs)):
        csv_path = os.path.join(m_dir, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            tau_label = os.path.basename(m_dir).replace("MALA_tau", "MALA (τ=") + ")"
            axs[0, 0].plot(df['Step'], df['E_mean'], color=colors[i], label=tau_label, linewidth=1.5)
            if 'acceptance' in df.columns:
                axs[0, 1].plot(df['Step'], df['acceptance'], color=colors[i], label=tau_label, linewidth=1.5)

    if os.path.isdir(td3_dir) and os.path.exists(os.path.join(td3_dir, "training_metrics.csv")):
        df_td3 = pd.read_csv(os.path.join(td3_dir, "training_metrics.csv"))
        axs[0, 0].plot(df_td3['Step'], df_td3['E_mean'], 'r-', label="TD3", linewidth=2.5)
        if 'acceptance' in df_td3.columns:
            axs[0, 1].plot(df_td3['Step'], df_td3['acceptance'], 'r-', label="TD3", linewidth=2.5)
        
        # Panel 3: Acciones del Actor (Solo TD3)
        if 'sig_scale' in df_td3.columns and 'd_act' in df_td3.columns:
            axs[1, 0].plot(df_td3['Step'], df_td3['sig_scale'], 'b-', label="Var Scale (sig_scale)")
            axs[1, 0].plot(df_td3['Step'], df_td3['d_act'], 'g-', label="Drift Magnitude (d_act)")
            
        # Panel 4: Recompensa y Pérdidas (Solo TD3)
        if 'mean_reward' in df_td3.columns:
            axs[1, 1].plot(df_td3['Step'], df_td3['mean_reward'], 'purple', label="Mean Reward (CDLB)")
            axs[1, 1].axhline(0, color='gray', linestyle='--')
        if 'Actor_Loss' in df_td3.columns:
            ax_loss = axs[1, 1].twinx()
            ax_loss.plot(df_td3['Step'], df_td3['Actor_Loss'], 'orange', alpha=0.6, label="Actor Loss")
            ax_loss.set_yscale('symlog')
            ax_loss.set_ylabel("Actor Loss", color='orange')

    # Ajustes estéticos
    axs[0, 0].set_title("VMC Convergence (Energy)"); axs[0, 0].set_ylabel("Energy (Ha)"); axs[0, 0].legend()
    axs[0, 1].set_title("MCMC Acceptance Rate"); axs[0, 1].set_ylabel("Acceptance"); axs[0, 1].legend()
    axs[1, 0].set_title("Learned Actor Behavior"); axs[1, 0].legend()
    axs[1, 1].set_title("RL Optimization (Reward & Loss)"); axs[1, 1].legend(loc='upper left')

    plt.tight_layout()
    out_file = os.path.join(args.sys_dir, "Training_Metrics_Dashboard.png")
    plt.savefig(out_file, dpi=300)
    print(f"Dashboard guardado en: {out_file}")

if __name__ == "__main__":
    main()