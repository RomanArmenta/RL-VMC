import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_file = "master_evaluation_N10_g5.csv"
    if not os.path.exists(csv_file):
        print(f"Error: No se encuentra el archivo {csv_file}")
        sys.exit(1)

    df = pd.read_csv(csv_file)
    
    # Limpieza: Separar MALA y TD3
    mala_df = df[df['method'] == 'MALA'].copy()
    td3_df = df[df['method'] == 'TD3'].copy()
    
    # Limpieza: Descartar los modelos TD3 totalmente colapsados (G_RL casi 0 por amplitudes masivas)
    # para no arruinar la escala visual de los modelos "sanos"
    td3_sanos = td3_df[td3_df['G_RL'] > 0.02].copy()
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("Global Hyperparameter & Architecture Analysis (N=10, g=5.0, Backflow Enabled)", weight='bold', fontsize=16)

    # =======================================================
    # PANEL 1: Impacto de la Amplitud Neuronal
    # =======================================================
    ax1 = plt.subplot(1, 3, 1)
    
    # Aquí usamos todos los TD3 para mostrar el colapso a altas amplitudes
    sns.scatterplot(data=td3_df, x='actor_amplitude', y='G_RL', hue='ess_s', palette='flare', size='ess_s', sizes=(20, 200), ax=ax1, edgecolor='k', alpha=0.8)
    ax1.set_xscale('log')
    ax1.set_title("Sensitivity to Actor Amplitude", weight='bold')
    ax1.set_xlabel("Actor Amplitude (Log Scale)")
    ax1.set_ylabel(r"Topological Gain ($G_{RL}$)")
    ax1.axhline(1.0, color='r', linestyle='--', label="MALA Baseline")
    ax1.legend(title="ESS/s", bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 8})
    
    # =======================================================
    # PANEL 2: Ablación de Arquitectura (Boxplot)
    # =======================================================
    ax2 = plt.subplot(1, 3, 2)
    
    # Combinamos MALA y los TD3 sanos para comparar arquitecturas
    mala_df['actor_type'] = 'MALA (No Actor)'
    combined_df = pd.concat([td3_sanos, mala_df])
    
    # Orden personalizado
    order = ['MALA (No Actor)', 'full', 'scale_only', 'flat']
    order = [o for o in order if o in combined_df['actor_type'].unique()]
    
    sns.violinplot(data=combined_df, x='actor_type', y='G_RL', order=order, ax=ax2, inner="point", palette="Set2", cut=0)
    ax2.set_title("Architecture Ablation Performance", weight='bold')
    ax2.set_xlabel("TD3 Actor Architecture")
    ax2.set_ylabel(r"Topological Gain ($G_{RL}$)")
    ax2.axhline(1.0, color='r', linestyle='--', alpha=0.5)

    # =======================================================
    # PANEL 3: Frontera de Pareto (Tau vs ESS/s)
    # =======================================================
    ax3 = plt.subplot(1, 3, 3)
    
    sns.scatterplot(data=td3_sanos, x='tau_int', y='ess_s', hue='actor_type', palette="Set2", s=100, edgecolor='k', ax=ax3, alpha=0.8)
    
    # Agregar MALA como la estrella de referencia
    if not mala_df.empty:
        mala_tau = mala_df['tau_int'].iloc[0]
        mala_ess = mala_df['ess_s'].iloc[0]
        ax3.scatter(mala_tau, mala_ess, color='red', marker='*', s=400, edgecolor='k', label='Optimized MALA')
        
        # Anotación para MALA
        ax3.annotate(f"MALA\n$\\tau={mala_tau:.2f}$", (mala_tau, mala_ess), xytext=(10, -20), textcoords='offset points', color='red', weight='bold')

    ax3.set_yscale('log')
    # ax3.set_xscale('log') # Opcional si tau_int tiene rangos muy amplios
    ax3.set_title("Computational Pareto Front", weight='bold')
    ax3.set_xlabel(r"Integrated Autocorrelation Time ($\tau_{int}$) $\rightarrow$ Lower is Better")
    ax3.set_ylabel(r"Effective Sample Size / sec (ESS/s) $\uparrow$")
    ax3.legend(title="Architecture")

    plt.tight_layout()
    out_file = "Global_Evaluation_Dashboard_N10_g5.pdf"
    plt.savefig(out_file, dpi=300, format='pdf')
    print(f"\n¡Análisis gráfico generado con éxito! Revisa el archivo: {out_file}")

if __name__ == "__main__":
    main()