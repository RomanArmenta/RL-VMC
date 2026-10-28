import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_grid_search_results(csv_path, out_file="GridSearch_Analysis.png"):
    df = pd.read_csv(csv_path)
    
    # Separar runs de MALA puro (actor_amp = 0) y RL-MALA
    df_mala = df[df['actor_amp'] == 0.0].copy()
    df_rl = df[df['actor_amp'] > 0.0].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Panel 1: tau_base vs tau_int (Comparación directa)
    ax1 = axes[0]
    
    # Línea base de MALA
    ax1.plot(df_mala['tau_base'], df_mala['mala_tau_int'], 
             marker='o', linestyle='-', color='#e74c3c', linewidth=2.5, markersize=8, label='MALA (Baseline)')
    
    # Scatter plot de todas las combinaciones de RL-MALA
    sns.scatterplot(data=df_rl, x='tau_base', y='rl_tau_int', 
                    hue='actor_amp', size='penalty', sizes=(50, 200),
                    palette='viridis', ax=ax1, alpha=0.8)
    
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel('Base Time Step ($\tau$)', fontweight='bold')
    ax1.set_ylabel('Integrated Autocorrelation Time ($\tau_{int}$)', fontweight='bold')
    ax1.set_title('Sampling Efficiency: MALA vs RL-MALA', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right')
    
    # Panel 2: Tasa de aceptación vs tau_int para RL-MALA
    ax2 = axes[1]
    scatter = sns.scatterplot(data=df_rl, x='rl_avg_accept', y='rl_tau_int', 
                              hue='tau_base', style='penalty', size='actor_amp',
                              palette='plasma', sizes=(50, 200), ax=ax2, alpha=0.9)
    
    # Marcar el óptimo teórico de MALA (~0.574)
    ax2.axvline(0.574, color='k', linestyle='--', alpha=0.5, label='Theoretical Optimum (0.574)')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('Average Acceptance Rate', fontweight='bold')
    ax2.set_ylabel('Integrated Autocorrelation Time ($\tau_{int}$)', fontweight='bold')
    ax2.set_title('Autocorrelation vs Acceptance Rate (RL-MALA)', fontweight='bold', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada en {out_file}")

if __name__ == '__main__':
    plot_grid_search_results('master_summary.csv')
