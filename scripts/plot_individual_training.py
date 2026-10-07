import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--td3_dir", type=str, required=True, help="Directorio del modelo TD3 recién entrenado")
    parser.add_argument("--mala_dir", type=str, required=True, help="Directorio del mejor modelo MALA base")
    args = parser.parse_args()

    td3_csv = os.path.join(args.td3_dir, "training_metrics.csv")
    mala_csv = os.path.join(args.mala_dir, "training_metrics.csv")

    if not os.path.exists(td3_csv):
        print(f"Error: No se encontró {td3_csv}")
        return

    df_td3 = pd.read_csv(td3_csv)
    # Filtrar solo la etapa VMC para que el eje X tenga sentido temporal
    df_td3 = df_td3[df_td3['Stage'] == 'VMC'] if 'Stage' in df_td3.columns else df_td3

    has_mala = os.path.exists(mala_csv)
    if has_mala:
        df_mala = pd.read_csv(mala_csv)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    td3_name = os.path.basename(args.td3_dir)
    fig.suptitle(f"Training Metrics Synthesis: {td3_name}", weight='bold', fontsize=14)

    # -----------------------------------------------------------
    # Panel 1: Energía
    # -----------------------------------------------------------
    if has_mala:
        axs[0, 0].plot(df_mala['Step'], df_mala['E_mean'], color='gray', alpha=0.7, label=f"MALA ({os.path.basename(args.mala_dir)})")
    axs[0, 0].plot(df_td3['Step'], df_td3['E_mean'], color='#34495e', linewidth=2, label="RL-MALA (TD3)")
    axs[0, 0].set_title("VMC Energy Convergence", weight='bold')
    axs[0, 0].set_ylabel("Energy (Ha)")
    axs[0, 0].legend()

    # -----------------------------------------------------------
    # Panel 2: Aceptación
    # -----------------------------------------------------------
    if has_mala and 'acceptance' in df_mala.columns:
        axs[0, 1].plot(df_mala['Step'], df_mala['acceptance'], color='gray', alpha=0.7, label="MALA")
    if 'acceptance' in df_td3.columns:
        axs[0, 1].plot(df_td3['Step'], df_td3['acceptance'], color='#2980b9', linewidth=2, label="RL-MALA")
    axs[0, 1].set_title("MCMC Acceptance Rate", weight='bold')
    axs[0, 1].set_ylabel("Acceptance")
    axs[0, 1].set_ylim(0, 1.05)
    axs[0, 1].legend()

    # -----------------------------------------------------------
    # Panel 3: Acciones del Actor (Dinámico)
    # -----------------------------------------------------------
    actor_cols = ['sig_scale', 'd_act', 'actor_w_norm']
    colors = ['#16a085', '#d35400', '#8e44ad']
    for idx, col in enumerate(actor_cols):
        if col in df_td3.columns:
            axs[1, 0].plot(df_td3['Step'], df_td3[col], color=colors[idx % len(colors)], label=col)
            
    axs[1, 0].set_title("Actor Actions & Weight Norm", weight='bold')
    axs[1, 0].legend()

    # -----------------------------------------------------------
    # Panel 4: Recompensas (Extracción Automática de Partes)
    # -----------------------------------------------------------
    reward_cols = [c for c in df_td3.columns if 'reward' in c.lower()]
    
    if reward_cols:
        colors_rw = sns.color_palette("husl", len(reward_cols))
        for idx, col in enumerate(reward_cols):
            axs[1, 1].plot(df_td3['Step'], df_td3[col], color=colors_rw[idx], linewidth=2, label=col)
        axs[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axs[1, 1].set_ylabel("Reward Components")
    
    # Añadir el Actor Loss en un eje secundario si existe
    if 'Actor_Loss' in df_td3.columns:
        ax_loss = axs[1, 1].twinx()
        ax_loss.plot(df_td3['Step'], df_td3['Actor_Loss'], color='orange', alpha=0.5, linestyle=':', label="Actor Loss")
        ax_loss.set_yscale('symlog')
        ax_loss.set_ylabel("Loss", color='orange')
        
    axs[1, 1].set_title("RL Optimization & Reward Breakdown", weight='bold')
    
    # Combinar leyendas del eje principal y secundario
    lines_1, labels_1 = axs[1, 1].get_legend_handles_labels()
    lines_2, labels_2 = ax_loss.get_legend_handles_labels() if 'Actor_Loss' in df_td3.columns else ([], [])
    axs[1, 1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', prop={'size': 8})

    plt.tight_layout()
    out_pdf = os.path.join(args.td3_dir, "Training_Metrics.pdf")
    plt.savefig(out_pdf, dpi=300, format='pdf')
    print(f"Gráfica generada y guardada en {out_pdf}")

if __name__ == "__main__":
    main()