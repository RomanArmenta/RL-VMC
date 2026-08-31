import os
import re
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    """
    Lee un archivo .txt de salida de terminal, detecta el penalty factor
    y extrae la telemetría del entrenamiento de RL-MALA y las métricas de eficiencia.
    """
    data = []
    tau_ints = {}
    pf_value = None
    in_rl_phase = False
    current_method = None
    
    with open(filepath, 'r') as f:
        for line in f:
            # 1. Detectar el Penalty Factor del encabezado
            if "penalty_factor:" in line:
                try:
                    pf_value = float(line.split(":")[1].strip())
                except ValueError:
                    pass
            
            # 2. Controlar en qué fase del log estamos
            if "Starting training with RL-MALA" in line:
                in_rl_phase = True
            elif "Starting training with MALA" in line:
                in_rl_phase = False
                
            # 3. Extraer telemetría paso a paso (Solo para RL-MALA)
            if in_rl_phase and line.startswith("Step:"):
                parts = line.split("|")
                row = {}
                for p in parts:
                    p = p.strip()
                    if p.startswith("Step:"):
                        row["Step"] = int(p.split(":")[1].strip())
                    elif ":" in p:
                        k, v = p.split(":", 1)
                        try:
                            row[k.strip()] = float(v.strip())
                        except ValueError:
                            row[k.strip()] = np.nan
                data.append(row)
                
            # 4. Extraer resultados finales (Tau_int)
            if "Final results for[" in line:
                m = re.search(r'\[(.*?)\]', line)
                if m:
                    current_method = m.group(1)
            
            if "Tau_int:" in line and current_method:
                m = re.search(r'Tau_int:\s*([\d\.]+)', line)
                if m:
                    tau_ints[current_method] = float(m.group(1))
                    current_method = None # Reset
                    
    df = pd.DataFrame(data)
    return pf_value, df, tau_ints

def main():
    parser = argparse.ArgumentParser(description="Generate Comparative Analysis for Different Penalty Factors")
    parser.add_argument("--folders", nargs='+', required=True, help="List of result folders to compare")
    parser.add_argument("--out_dir", type=str, default="./results/comparative_analysis", help="Output directory for the plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    results = []
    mala_baseline_tau = None

    # Procesar todas las carpetas
    print("Parsing result folders...")
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f" [!] Warning: '{folder}' is not a valid directory. Skipping.")
            continue
            
        txt_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        if not txt_files:
            print(f" [!] Warning: No .txt log file found in '{folder}'. Skipping.")
            continue
            
        log_path = os.path.join(folder, txt_files[0]) # Asumimos que hay 1 log por carpeta
        pf, df, tau_dict = parse_log_file(log_path)
        
        if df.empty or pf is None:
            print(f" [!] Warning: Could not extract RL data from '{log_path}'.")
            continue
            
        print(f"  -> Loaded PF = {pf} | RL-MALA Tau_int = {tau_dict.get('RL-MALA', 'N/A')}")
        results.append({"pf": pf, "df": df, "tau_dict": tau_dict})
        
        # Guardar la baseline de MALA si existe (asumimos que es casi idéntica en todas las corridas)
        if 'MALA' in tau_dict and mala_baseline_tau is None:
            mala_baseline_tau = tau_dict['MALA']

    if not results:
        print("Error: No valid data found to plot.")
        return

    # Ordenar por Penalty Factor de menor a mayor
    results.sort(key=lambda x: x['pf'])
    
    # Colores visualmente distinguibles
    colors = ['#e63946', '#2a9d8f', '#1d3557', '#f4a261', '#e76f51']
    
    print("\nGenerating Comparative Panel...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("RL-MALA Comparative Analysis: Effect of Penalty Factor", fontsize=18, y=0.97)

    # Gráfica 1: Evolución de s_para (La confianza espacial del Actor)
    ax1 = axs[0, 0]
    for i, res in enumerate(results):
        df = res['df']
        ax1.plot(df['Step'], df['s_para'], label=f"PF = {res['pf']}", color=colors[i%len(colors)], linewidth=2)
    ax1.set_title("Actor Spatial Variance ($\sigma_{\parallel}$) Evolution", fontsize=13)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Standard Deviation Scalar")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Gráfica 2: Trade-off de Aceptación vs EJD
    ax2 = axs[0, 1]
    for i, res in enumerate(results):
        df = res['df']
        # Tomamos solo la segunda mitad del entrenamiento para ver el comportamiento asintótico
        half_idx = len(df) // 2
        stable_df = df.iloc[half_idx:]
        ax2.scatter(stable_df['EJD'], stable_df['Acc'], alpha=0.5, s=20, 
                    label=f"PF = {res['pf']}", color=colors[i%len(colors)])
    
    ax2.set_title("Exploration vs Acceptance Trade-off", fontsize=13)
    ax2.set_xlabel("Expected Jump Distance (EJD)")
    ax2.set_ylabel("Acceptance Rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Gráfica 3: Recompensa Pura por Distancia (reward_dist)
    ax3 = axs[1, 0]
    for i, res in enumerate(results):
        df = res['df']
        # Suavizado para mejor legibilidad
        smoothed_dist = df['Rew_dist'].rolling(window=10, min_periods=1).mean()
        ax3.plot(df['Step'], smoothed_dist, label=f"PF = {res['pf']}", color=colors[i%len(colors)], linewidth=2)
        
    ax3.set_title("Pure Distance Reward Component", fontsize=13)
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("Reward Signal (Distance)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Gráfica 4: Eficiencia Final (Tau_int)
    ax4 = axs[1, 1]
    labels = [f"PF={res['pf']}" for res in results]
    tau_values = [res['tau_dict'].get('RL-MALA', 0) for res in results]
    
    bars = ax4.bar(labels, tau_values, color=[colors[i%len(colors)] for i in range(len(results))], alpha=0.8)
    
    if mala_baseline_tau:
        ax4.axhline(mala_baseline_tau, color='black', linestyle='--', linewidth=2, label=f'MALA Baseline ({mala_baseline_tau})')
        
    ax4.set_title(r"Measurement Efficiency (Integrated Autocorrelation Time $\tau_{int}$)", fontsize=13)
    ax4.set_ylabel(r"$\tau_{int}$ (Steps) - Lower is Better")
    ax4.set_ylim(0, max(tau_values + [mala_baseline_tau or 0]) * 1.2)
    ax4.grid(axis='y', alpha=0.3)
    ax4.legend()
    
    # Añadir valores numéricos sobre las barras
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout(pad=3.0)
    out_file = os.path.join(args.out_dir, "comparative_panel.pdf")
    plt.savefig(out_file, format='pdf', dpi=300)
    plt.close()
    
    print(f"\nComparative panel saved successfully at: {out_file}")

if __name__ == "__main__":
    main()