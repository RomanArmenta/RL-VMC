import os
import glob
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_logs(logs_dir):
    """Parsea los archivos txt de evaluación y devuelve un DataFrame agrupado por g."""
    pattern = os.path.join(logs_dir, "out_evaluate_*.txt")
    log_files = glob.glob(pattern)
    
    if not log_files:
        return None

    data = []
    for file_path in log_files:
        match_g = re.search(r'_g([0-9\.]+)_', file_path)
        if not match_g:
            continue
        g_val = float(match_g.group(1))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                # TD3
                e_td3 = float(re.search(r'Energía TD3:\s*([-\d\.]+)', content).group(1))
                tau_td3 = float(re.search(r'Tau_int \(E\) TD3:\s*([-\d\.]+)', content).group(1))
                neff_td3 = float(re.search(r'N_eff TD3:\s*(\d+)', content).group(1))
                ess_td3 = float(re.search(r'ESS/sec TD3:\s*([-\d\.]+)', content).group(1))
                # MALA
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
                continue
                
    if not data:
        return None
        
    df = pd.DataFrame(data)
    df = df.groupby('g').mean().reset_index()
    return df.sort_values('g').reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir_prefix", type=str, default="./logs_1D_N4_sweep_g_100kTrain")
    parser.add_argument("--out_dir", type=str, default="./results_1D_N4_sweep_g_100kTrain_Comparisons")
    parser.add_argument("--n_samples", type=int, default=100000)
    parser.add_argument("--subsample", type=int, default=1000)
    parser.add_argument("--n_ter", type=int, default=5000)
    args = parser.parse_args()

    # Nombres exactos de los sufijos de las carpetas
    rewards = ["MyReward", "HuzielReward", "PaperReward"]
    colors = {"MyReward": "blue", "HuzielReward": "green", "PaperReward": "red"}
    labels = {"MyReward": "TD3 (My Rwd)", "HuzielReward": "TD3 (Huziel Rwd)", "PaperReward": "TD3 (Paper Rwd)"}

    df_dict = {}
    for r in rewards:
        ldir = f"{args.base_dir_prefix}_{r}/eval_{args.n_samples}s_{args.subsample}sub_{args.n_ter}ter"
        df = parse_logs(ldir)
        if df is not None and not df.empty:
            df_dict[r] = df
        else:
            print(f"[Advertencia] No se encontraron datos para {r} en: {ldir}")

    if not df_dict:
        print("Error: No se encontró información para graficar en ninguna carpeta.")
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # Extraemos el Baseline de MALA del primer DataFrame disponible
    first_r = list(df_dict.keys())[0]
    g_vals = df_dict[first_r]['g']
    e_mala = df_dict[first_r]['E_MALA']
    
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(fr"Comparación de Recompensas TD3 ($N=4$, {args.n_samples}s, {args.subsample}sub, {args.n_ter}ter)", fontsize=18)

    # --- Panel 1: Energía Local con Tabla ---
    ax = axs[0, 0]
    ax.plot(g_vals, e_mala, 'k--o', label='MALA Baseline', linewidth=2)
    for r, df in df_dict.items():
        ax.plot(df['g'], df['E_TD3'], marker='o', color=colors[r], label=labels[r])
        
    ax.set_title("Energía Local Variacional (Menor es mejor)")
    ax.set_ylabel("Hartrees")
    ax.legend(loc="lower right")

    # Creación del Cuadrito Numérico en el gráfico
    table_data = []
    col_labels = ["g", "MALA"] + [labels[r].replace("TD3 ", "") for r in rewards if r in df_dict]
    for idx, g in enumerate(g_vals):
        row = [f"{g:.1f}", f"{e_mala.iloc[idx]:.4f}"]
        for r in rewards:
            if r in df_dict:
                val = df_dict[r].loc[df_dict[r]['g'] == g, 'E_TD3']
                row.append(f"{val.values[0]:.4f}" if not val.empty else "N/A")
        table_data.append(row)

    # Posicionado en la esquina superior izquierda (vacía porque las curvas suben hacia la derecha)
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='upper left', bbox=[0.02, 0.55, 0.45, 0.43])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # --- Panel 2: Tau ---
    ax = axs[0, 1]
    ax.plot(g_vals, df_dict[first_r]['Tau_MALA'], 'k--o', label='MALA')
    for r, df in df_dict.items():
        ax.plot(df['g'], df['Tau_TD3'], marker='o', color=colors[r], label=labels[r])
    ax.set_title(r"Tiempo de Autocorrelación ($\tau_{int}$)")
    ax.set_ylabel(r"$\tau_{int}$")
    ax.legend()

    # --- Panel 3: N_eff ---
    ax = axs[1, 0]
    ax.plot(g_vals, df_dict[first_r]['Neff_MALA'], 'k--o', label='MALA')
    for r, df in df_dict.items():
        ax.plot(df['g'], df['Neff_TD3'], marker='o', color=colors[r], label=labels[r])
    ax.set_title("Muestras Útiles Extraídas ($N_{eff}$)")
    ax.set_xlabel("Fuerza de Interacción (g)")
    ax.set_ylabel("Muestras")
    ax.set_yscale('log')
    ax.legend()

    # --- Panel 4: ESS / sec ---
    ax = axs[1, 1]
    ax.plot(g_vals, df_dict[first_r]['ESS_MALA'], 'k--o', label='MALA')
    for r, df in df_dict.items():
        ax.plot(df['g'], df['ESS_TD3'], marker='o', color=colors[r], label=labels[r])
    ax.set_title("Eficiencia Computacional Neta (ESS / Segundo)")
    ax.set_xlabel("Fuerza de Interacción (g)")
    ax.set_ylabel("ESS / segundo")
    ax.set_yscale('log')
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plot_filename = f"reward_comparison_{args.n_samples}s_{args.subsample}sub_{args.n_ter}ter.png"
    out_path = os.path.join(args.out_dir, plot_filename)
    plt.savefig(out_path, dpi=300)
    print(f"Gráfica comparativa maestra guardada en: {out_path}")

if __name__ == "__main__":
    main()