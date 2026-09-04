import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(logs_dir):
    pattern = os.path.join(logs_dir, "out_evaluate_*.txt")
    log_files = glob.glob(pattern)
    
    if not log_files:
        print(f"No se encontraron archivos de evaluación en {logs_dir}")
        return None

    data = []

    for file_path in log_files:
        # Extraer el valor de g del nombre del archivo (ej: 1D__TD3_g5.0_Om0.1...)
        match_g = re.search(r'_g([0-9\.]+)_', file_path)
        if not match_g:
            continue
        g_val = float(match_g.group(1))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            try:
                # Métricas TD3
                e_td3 = float(re.search(r'Energía TD3:\s*([-\d\.]+)', content).group(1))
                tau_td3 = float(re.search(r'Tau_int \(E\) TD3:\s*([-\d\.]+)', content).group(1))
                neff_td3 = float(re.search(r'N_eff TD3:\s*(\d+)', content).group(1))
                ess_td3 = float(re.search(r'ESS/sec TD3:\s*([-\d\.]+)', content).group(1))
                
                # Métricas MALA
                e_mala = float(re.search(r'Energía MALA:\s*([-\d\.]+)', content).group(1))
                tau_mala = float(re.search(r'Tau_int \(E\) MALA:\s*([-\d\.]+)', content).group(1))
                neff_mala = float(re.search(r'N_eff MALA:\s*(\d+)', content).group(1))
                ess_mala = float(re.search(r'ESS/sec MALA:\s*([-\d\.]+)', content).group(1))
                
                data.append({
                    'g': g_val,
                    'E_MALA': e_mala, 'E_TD3': e_td3,
                    'Tau_MALA': tau_mala, 'Tau_TD3': tau_td3,
                    'Neff_MALA': neff_mala, 'Neff_TD3': neff_td3,
                    'ESS_MALA': ess_mala, 'ESS_TD3': ess_td3
                })
                
            except AttributeError:
                print(f"Faltan métricas en {file_path}. Ignorando...")
                continue
                
    if not data:
        return None
        
    # Crear DataFrame, promediar repetidos y ordenar por g
    df = pd.DataFrame(data)
    df = df.groupby('g').mean().reset_index()
    df = df.sort_values('g').reset_index(drop=True)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--subsample", type=int, default=10)
    parser.add_argument("--n_ter", type=int, default=1000) # Recepción del argumento n_ter
    args = parser.parse_args()
    
    df = parse_logs(args.logs_dir)
    if df is None:
        return
        
    # 1. Exportar Tabla CSV y mostrar en consola (incorporando n_ter en el nombre)
    csv_filename = f"summary_metrics_g_{args.n_samples}s_{args.subsample}sub_{args.n_ter}ter.csv"
    csv_path = os.path.join(args.out_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*90)
    print(f"TABLA DE RESULTADOS: MALA vs TD3 ({args.n_samples} samples, {args.subsample} subsample, {args.n_ter} termalización)")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90 + "\n")
    
    # 2. Generar Gráficos
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    # Incorporando n_ter al título superior de la gráfica
    fig.suptitle(fr"Resumen: Desempeño RL-VMC vs Fuerza de Interacción ($N=10$, Amp=0.01, $N_{{ter}}$={args.n_ter})", fontsize=16)

    g_vals = df['g']

    # --- Panel 1: Energía Local ---
    ax = axs[0, 0]
    ax.plot(g_vals, df['E_MALA'], 'k--o', label='MALA')
    ax.plot(g_vals, df['E_TD3'], 'r-o', label='TD3')
    ax.set_title("Energía Local Variacional (Menor es mejor)")
    ax.set_ylabel("Hartrees")
    ax.legend()

    # --- Panel 2: Tau ---
    ax = axs[0, 1]
    ax.plot(g_vals, df['Tau_MALA'], 'k--o', label='MALA')
    ax.plot(g_vals, df['Tau_TD3'], 'g-o', label='TD3')
    ax.set_title(r"Tiempo de Autocorrelación ($\tau_{int}$) - Menor es mejor")
    ax.set_ylabel(r"$\tau_{int}$")
    ax.legend()

    # --- Panel 3: N_eff ---
    ax = axs[1, 0]
    ax.plot(g_vals, df['Neff_MALA'], 'k--o', label='MALA')
    ax.plot(g_vals, df['Neff_TD3'], 'b-o', label='TD3')
    ax.set_title("Muestras Útiles Extraídas ($N_{eff}$) - Mayor es mejor")
    ax.set_xlabel("Fuerza de Interacción (g)")
    ax.set_ylabel("Muestras")
    ax.set_yscale('log')
    ax.legend()

    # --- Panel 4: ESS / sec ---
    ax = axs[1, 1]
    ax.plot(g_vals, df['ESS_MALA'], 'k--o', label='MALA')
    ax.plot(g_vals, df['ESS_TD3'], 'm-o', label='TD3')
    ax.set_title("Eficiencia Computacional Neta (ESS / Segundo)")
    ax.set_xlabel("Fuerza de Interacción (g)")
    ax.set_ylabel("ESS / segundo")
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout()
    
    # Incorporando n_ter al nombre del archivo de la imagen
    plot_filename = f"sweep_summary_g_{args.n_samples}s_{args.subsample}sub_{args.n_ter}ter.png"
    plot_path = os.path.join(args.out_dir, plot_filename)
    plt.savefig(plot_path, dpi=300)
    
    print(f"-> Tabla CSV guardada en: {csv_path}")
    print(f"-> Gráfica resumen guardada en: {plot_path}")

if __name__ == "__main__":
    main()