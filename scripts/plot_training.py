import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def smooth_curve(series, window=10):
    """Aplica una media móvil para suavizar las curvas de RL."""
    return series.rolling(window=window, min_periods=1).mean()

def load_data(folder_path):
    """Carga los datos y los parámetros de una carpeta."""
    csv_path = os.path.join(folder_path, "training_metrics.csv")
    json_path = os.path.join(folder_path, "params.json")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    params = {}
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            
    return df, params

def main():
    parser = argparse.ArgumentParser(description="Compara MALA vs TD3")
    parser.add_argument("--dir_MALA", type=str, required=True, help="Carpeta de resultados MALA")
    parser.add_argument("--dir_td3", type=str, required=True, help="Carpeta de resultados TD3")
    parser.add_argument("--out_name", type=str, default="comparison_plot.png", help="Nombre del archivo de salida")
    parser.add_argument("--smooth_window", type=int, default=15, help="Ventana de suavizado (media móvil)")
    args = parser.parse_args()

    # Cargar datos
    print(f"Cargando MALA desde: {args.dir_MALA}")
    df_re, params_re = load_data(args.dir_MALA)
    
    print(f"Cargando TD3 desde: {args.dir_td3}")
    df_td3, params_td3 = load_data(args.dir_td3)

    # Configuración de estilo
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extraer variables para el título
    g = params_re.get('interaction_strength', 'N/A')
    omega = params_re.get('omega', 'N/A')
    amp = params_re.get('actor_amplitude', 'N/A')
    fig.suptitle(f"Comparación de Entrenamiento: MALA vs TD3\n(g={g}, $\omega$={omega}, Amp={amp})", fontsize=16, y=0.98)

    # --- Plot 1: Energía ---
    ax = axs[0, 0]
    ax.plot(df_re['Step'], df_re['E_mean'], alpha=0.3, color='blue')
    ax.plot(df_td3['Step'], df_td3['E_mean'], alpha=0.3, color='red')
    ax.plot(df_re['Step'], smooth_curve(df_re['E_mean'], args.smooth_window), color='blue', label='MALA', linewidth=2)
    ax.plot(df_td3['Step'], smooth_curve(df_td3['E_mean'], args.smooth_window), color='red', label='TD3', linewidth=2)
    ax.set_title('Convergencia de la Energía (E_mean)')
    ax.set_xlabel('Pasos VMC')
    ax.set_ylabel('Energía Local Promedio (Ha)')
    ax.legend()

    # --- Plot 2: Expected Jump Distance (EJD) ---
    ax = axs[0, 1]
    if 'ejd' in df_re.columns and 'ejd' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['ejd'], args.smooth_window), color='blue', label='MALA', linewidth=2)
        ax.plot(df_td3['Step'], smooth_curve(df_td3['ejd'], args.smooth_window), color='red', label='TD3', linewidth=2)
        ax.set_title('Expected Jump Distance (EJD)')
        ax.set_xlabel('Pasos VMC')
        ax.set_ylabel('EJD')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'Métrica EJD no disponible', ha='center', va='center')

    # --- Plot 3: Tasa de Aceptación ---
    ax = axs[1, 0]
    if 'acceptance' in df_re.columns and 'acceptance' in df_td3.columns:
        ax.plot(df_re['Step'], smooth_curve(df_re['acceptance'], args.smooth_window), color='blue', label='MALA', linewidth=2)
        ax.plot(df_td3['Step'], smooth_curve(df_td3['acceptance'], args.smooth_window), color='red', label='TD3', linewidth=2)
        ax.set_title('Tasa de Aceptación')
        ax.set_xlabel('Pasos VMC')
        ax.set_ylabel('Aceptación (0 a 1)')
        ax.set_ylim(0.4, 1.0)
        ax.legend()
        
    # --- Plot 4: Tiempo de Ejecución (Si existe) ---
    ax = axs[1, 1]
    if 'Time(s)' in df_re.columns and 'Time(s)' in df_td3.columns:
        ax.plot(df_re['Step'], df_re['Time(s)'], color='blue', label='MALA', linewidth=2)
        ax.plot(df_td3['Step'], df_td3['Time(s)'], color='red', label='TD3', linewidth=2)
        ax.set_title('Tiempo de Ejecución por Bloque')
        ax.set_xlabel('Pasos VMC')
        ax.set_ylabel('Tiempo Acumulado (s)')
        ax.legend()
    else:
        # Si aún no hay timestamps, ploteamos la Recompensa de MALA como fallback
        if 'mean_reward' in df_re.columns:
            ax.plot(df_re['Step'], smooth_curve(df_re['mean_reward'], args.smooth_window), color='blue', label='MALA Reward', linewidth=2)
            ax.set_title('Recompensa')
            ax.set_xlabel('Pasos VMC')
            ax.set_ylabel('Recompensa Media')
            ax.legend()

    plt.tight_layout()
    out_path = os.path.join(args.dir_MALA, args.out_name)
    plt.savefig(out_path, dpi=300)
    print(f"Gráfico guardado exitosamente en: {out_path}")
    
if __name__ == "__main__":
    main()