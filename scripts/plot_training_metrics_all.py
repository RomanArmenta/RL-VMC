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
    fig.suptitle(f"Training Metrics Synthesis: MALA vs RL-MALA", weight='bold', fontsize=16)

    # 1. Buscar el único MALA optimizado
    mala_base = os.path.join(args.sys_dir, "MALA_Baselines")
    mala_dirs = [d for d in glob.glob(os.path.join(mala_base, "MALA_tau*")) if os.path.isdir(d)]
    
    if mala_dirs:
        # Tomamos el primero (solo debería haber uno después del clear)
        m_dir = mala_dirs[0]
        csv_path = os.path.join(m_dir, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            tau_label = "Optimized " + os.path.basename(m_dir).replace("_", " ")
            
            axs[0, 0].plot(df['Step'], df['E_mean'], color='#7f8c8d', linewidth=2, alpha=0.9, label=tau_label)
            if 'acceptance' in df.columns:
                axs[0, 1].plot(df['Step'], df['acceptance'], color='#7f8c8d', linewidth=2, alpha=0.9, label=tau_label)

    # 2. Graficar modelo TD3
    td3_dir = os.path.join(args.sys_dir, "TD3")
    if os.path.isdir(td3_dir) and os.path.exists(os.path.join(td3_dir, "training_metrics.csv")):
        df_td3 = pd.read_csv(os.path.join(td3_dir, "training_metrics.csv"))
        
        # Paneles 1 y 2
        axs[0, 0].plot(df_td3['Step'], df_td3['E_mean'], color='#2c3e50', linewidth=2, label="RL-MALA (TD3)")
        if 'acceptance' in df_td3.columns:
            axs[0, 1].plot(df_td3['Step'], df_td3['acceptance'], color='#2c3e50', linewidth=2, label="RL-MALA (TD3)")
        
        # Panel 3: Físicas del Actor
        if 'sig_scale' in df_td3.columns and 'd_act' in df_td3.columns:
            axs[1, 0].plot(df_td3['Step'], df_td3['sig_scale'], color='#3498db', linewidth=2, label="Var Scale (sig_scale)")
            axs[1, 0].plot(df_td3['Step'], df_td3['d_act'], color='#27ae60', linewidth=2, label="Drift Mag (d_act)")
            if 'mag_mala' in df_td3.columns:
                axs[1, 0].plot(df_td3['Step'], df_td3['mag_mala'], color='gray', linestyle=':', label="Base Langevin Mag")
            
        # Panel 4: Salud de RL (Tolerante a columnas faltantes)
        if 'mean_reward' in df_td3.columns:
            l1 = axs[1, 1].plot(df_td3['Step'], df_td3['mean_reward'], color='purple', linewidth=2, label="Total Reward (CDLB)")
            lns = l1
            
            # Si existen las componentes desglosadas, graficarlas
            if 'reward_dist' in df_td3.columns and 'reward_penalty' in df_td3.columns:
                l2 = axs[1, 1].plot(df_td3['Step'], df_td3['reward_dist'], color='green', linestyle='--', alpha=0.7, label="Exploration (Dist)")
                l3 = axs[1, 1].plot(df_td3['Step'], -df_td3['reward_penalty'], color='red', linestyle=':', alpha=0.7, label="Penalty (Force)")
                lns = l1 + l2 + l3
            
            axs[1, 1].axhline(0, color='gray', linestyle='-')
            
            ax_loss = axs[1, 1].twinx()
            if 'Actor_Loss' in df_td3.columns:
                l4 = ax_loss.plot(df_td3['Step'], df_td3['Actor_Loss'], color='orange', alpha=0.5, label="Actor Loss")
                ax_loss.set_yscale('symlog')
                ax_loss.set_ylabel("Loss", color='orange')
                lns += l4
                
            labs = [l.get_label() for l in lns]
            axs[1, 1].legend(lns, labs, loc='center right')

    # Estética final
    axs[0, 0].set_title("VMC Energy Convergence", weight='bold'); axs[0, 0].set_ylabel("Energy (Ha)"); axs[0, 0].legend()
    axs[0, 1].set_title("MCMC Acceptance Rate", weight='bold'); axs[0, 1].set_ylabel("Acceptance"); axs[0, 1].legend()
    axs[1, 0].set_title("Actor Learned Actions", weight='bold'); axs[1, 0].legend()
    axs[1, 1].set_title("RL Optimization Stats", weight='bold')

    plt.tight_layout()
    out_file = os.path.join(args.sys_dir, "Training_Metrics_Dashboard.pdf")
    plt.savefig(out_file, dpi=300, format='pdf')
    print(f"Dashboard global generado en: {out_file}")
    
    import sys
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()