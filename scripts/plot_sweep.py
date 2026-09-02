import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def smooth_curve(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()

def find_folder(base_dir, method, g, omega, amp, run_id):
    """Busca la carpeta que coincida con el patrón de nomenclatura."""
    pattern = f"*_1D__{method}_g{g}_Om{omega}_Amp{amp}_{run_id}"
    search_path = os.path.join(base_dir, pattern)
    matches = glob.glob(search_path)
    
    if not matches:
        raise FileNotFoundError(f"No se encontró carpeta para: {pattern}")
    return matches[0] # Retorna la primera coincidencia (usualmente la única)

def main():
    parser = argparse.ArgumentParser(description="Barrido y comparación MALA vs TD3")
    parser.add_argument("--base_dir", type=str, default="./results_1D")
    parser.add_argument("--g", type=float, required=True, help="Interaction strength")
    parser.add_argument("--omega", type=float, required=True, help="Frecuencia del pozo")
    parser.add_argument("--amp", type=float, required=True, help="Amplitud del actor")
    parser.add_argument("--run_id", type=int, default=1, help="Sufijo del experimento (1, 2, etc)")
    parser.add_argument("--smooth_window", type=int, default=15)
    args = parser.parse_args()

    # Localizar carpetas automáticamente
    dir_re = find_folder(args.base_dir, "MALA", args.g, args.omega, args.amp, args.run_id)
    dir_td3 = find_folder(args.base_dir, "TD3", args.g, args.omega, args.amp, args.run_id)

    print(f"MALA: {os.path.basename(dir_re)}")
    print(f"TD3:       {os.path.basename(dir_td3)}")

    df_re = pd.read_csv(os.path.join(dir_re, "training_metrics.csv"))
    df_td3 = pd.read_csv(os.path.join(dir_td3, "training_metrics.csv"))

    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(fr"Entrenamiento RL-VMC: MALA vs TD3\n($g={args.g}$, $\omega={args.omega}$, Amp Actor={args.amp})", fontsize=16, y=0.98)

    # 1: Energía
    ax = axs[0, 0]
    ax.plot(df_re['Step'], df_re['E_mean'], alpha=0.2, color='blue')
    ax.plot(df_td3['Step'], df_td3['E_mean'], alpha=0.2, color='red')
    ax.plot(df_re['Step'], smooth_curve(df_re['E_mean'], args.smooth_window), color='blue', label='MALA')
    ax.plot(df_td3['Step'], smooth_curve(df_td3['E_mean'], args.smooth_window), color='red', label='TD3')
    ax.set_title('Energía Local (E_mean)')
    ax.set_xlabel('Pasos VMC'); ax.set_ylabel('Hartrees')
    ax.legend()

    # 2: Expected Jump Distance (EJD)
    ax = axs[0, 1]
    if 'ejd' in df_re.columns and 'ejd' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['ejd'], args.smooth_window), color='blue', label='MALA')
        ax.plot(df_td3['Step'], smooth_curve(df_td3['ejd'], args.smooth_window), color='red', label='TD3')
        ax.set_title('Distancia Esperada de Salto (EJD)')
        ax.set_xlabel('Pasos VMC'); ax.set_ylabel('EJD')
        ax.legend()

    # 3: Tasa de Aceptación
    ax = axs[1, 0]
    if 'acceptance' in df_re.columns and 'acceptance' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['acceptance'], args.smooth_window), color='blue', label='MALA')
        ax.plot(df_td3['Step'], smooth_curve(df_td3['acceptance'], args.smooth_window), color='red', label='TD3')
        ax.set_title('Tasa de Aceptación Metropolis')
        ax.set_xlabel('Pasos VMC'); ax.set_ylabel('Probabilidad')
        ax.set_ylim(0.4, 1.0)
        ax.legend()
        
    # 4: Recompensa
    ax = axs[1, 1]
    if 'mean_reward' in df_re.columns and 'mean_reward' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['mean_reward'], args.smooth_window), color='blue', label='MALA')
        ax.plot(df_td3['Step'], df_td3['mean_reward'], color='red', label='TD3', linestyle='--')
        ax.set_title('Recompensa del Actor (Reward)')
        ax.set_xlabel('Pasos VMC'); ax.set_ylabel('Recompensa Media')
        ax.legend()

    # 5: Pérdida del Actor
    ax = axs[2, 0]
    if 'Actor_Loss' in df_re.columns and 'Actor_Loss' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['Actor_Loss'], args.smooth_window), color='blue', label='MALA')
        # Línea punteada para TD3 porque se actualiza en bloques
        ax.plot(df_td3['Step'], df_td3['Actor_Loss'], color='red', label='TD3', linestyle='--')
        ax.set_title('Pérdida del Actor (Policy Loss)')
        ax.set_xlabel('Pasos VMC'); ax.set_ylabel('Loss')
        ax.legend()
        
    # 6: Pérdida de los Críticos (Solo existe en TD3)
    ax = axs[2, 1]
    if 'Critic1_Loss' in df_td3.columns:
        ax.plot(df_td3['Step'], smooth_curve(df_td3['Critic1_Loss'], args.smooth_window), color='darkred', label='TD3 Critic 1')
        ax.plot(df_td3['Step'], smooth_curve(df_td3['Critic2_Loss'], args.smooth_window), color='salmon', label='TD3 Critic 2', linestyle='--')
        ax.set_title('Pérdida de los Críticos (Solo TD3)')
        ax.set_xlabel('Pasos VMC'); ax.set_ylabel('MSE Loss')
        ax.set_yscale('log') # Escala logarítmica suele ser mejor para el MSE
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Críticos no disponibles', ha='center', va='center')

    plt.tight_layout()
    out_name = f"Compare_g{args.g}_Om{args.omega}_Amp{args.amp}_{args.run_id}.png"
    out_path = os.path.join(args.base_dir, out_name)
    plt.savefig(out_path, dpi=300)
    print(f"Gráfico comparativo guardado en: {out_path}")

if __name__ == "__main__":
    main()