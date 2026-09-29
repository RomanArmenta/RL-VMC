import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_dir", type=str, required=True, help="Carpeta del sistema (ej. results_N4_g1.0)")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Training Metrics Synthesis: {os.path.basename(args.sys_dir)}", weight='bold', fontsize=16)

    mala_dirs = [d for d in glob.glob(os.path.join(args.sys_dir, "MALA_tau*")) if os.path.isdir(d)]
    td3_dir = os.path.join(args.sys_dir, "TD3")

    # Paleta dinámica de grises para MALA
    colors_mala = sns.color_palette("Greys", len(mala_dirs) + 2)[1:] 

    # 1. Graficar todos los modelos MALA
    for i, m_dir in enumerate(sorted(mala_dirs)):
        csv_path = os.path.join(m_dir, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            tau_label = os.path.basename(m_dir).replace("MALA_tau", "MALA (τ=") + ")"
            
            axs[0, 0].plot(df['Step'], df['E_mean'], color=colors_mala[i], alpha=0.8, label=tau_label)
            if 'acceptance' in df.columns:
                axs[0, 1].plot(df['Step'], df['acceptance'], color=colors_mala[i], alpha=0.8, label=tau_label)

    # 2. Graficar modelo TD3
    if os.path.isdir(td3_dir) and os.path.exists(os.path.join(td3_dir, "training_metrics.csv")):
        df_td3 = pd.read_csv(os.path.join(td3_dir, "training_metrics.csv"))
        
        # Paneles 1 y 2
        axs[0, 0].plot(df_td3['Step'], df_td3['E_mean'], 'r-', linewidth=2, label="RL-MALA (TD3)")
        if 'acceptance' in df_td3.columns:
            axs[0, 1].plot(df_td3['Step'], df_td3['acceptance'], 'r-', linewidth=2, label="RL-MALA (TD3)")
        
        # Panel 3: Físicas del Actor
        if 'sig_scale' in df_td3.columns and 'd_act' in df_td3.columns:
            axs[1, 0].plot(df_td3['Step'], df_td3['sig_scale'], color='#3498db', linewidth=2, label="Var Scale (sig_scale)")
            axs[1, 0].plot(df_td3['Step'], df_td3['d_act'], color='#27ae60', linewidth=2, label="Drift Mag (d_act)")
            if 'mag_mala' in df_td3.columns:
                axs[1, 0].plot(df_td3['Step'], df_td3['mag_mala'], color='gray', linestyle=':', label="Base Langevin Mag")
            
        # Panel 4: Salud de RL
        if 'mean_reward' in df_td3.columns:
            l1 = axs[1, 1].plot(df_td3['Step'], df_td3['mean_reward'], color='purple', label="Mean Reward (CDLB)")
            axs[1, 1].axhline(0, color='gray', linestyle='--')
        if 'Actor_Loss' in df_td3.columns:
            ax_loss = axs[1, 1].twinx()
            l2 = ax_loss.plot(df_td3['Step'], df_td3['Actor_Loss'], color='orange', alpha=0.6, label="Actor Loss")
            ax_loss.set_yscale('symlog')
            ax_loss.set_ylabel("Loss", color='orange')
            
            # Unir leyendas del panel dual
            lns = l1 + l2
            labs = [l.get_label() for l in lns]
            axs[1, 1].legend(lns, labs, loc='upper center')

    # Estética final
    axs[0, 0].set_title("VMC Energy Convergence", weight='bold'); axs[0, 0].set_ylabel("Energy (Ha)"); axs[0, 0].legend()
    axs[0, 1].set_title("MCMC Acceptance Rate", weight='bold'); axs[0, 1].set_ylabel("Acceptance"); axs[0, 1].legend()
    axs[1, 0].set_title("Actor Learned Actions", weight='bold'); axs[1, 0].legend()
    axs[1, 1].set_title("RL Optimization Stats", weight='bold')

    plt.tight_layout()
    out_file = os.path.join(args.sys_dir, "Training_Metrics_Dashboard.pdf")
    plt.savefig(out_file, dpi=300, format='pdf')
    print(f"Dashboard global generado en: {out_file}")

if __name__ == "__main__":
    main()