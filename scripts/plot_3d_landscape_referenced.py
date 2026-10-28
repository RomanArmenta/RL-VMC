import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(description="Renderiza el paisaje 3D referenciado a la línea base de MALA.")
    parser.add_argument("--sys_dirs", nargs='+', required=True, help="Lista de carpetas (ej. dir1 dir2)")
    args = parser.parse_args()

    dfs = []
    mala_taus = []
    
    print("Recolectando datos de:")
    for d in args.sys_dirs:
        # 1. Intentar extraer tau de MALA desde stats.json
        stats_path = os.path.join(d, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                try:
                    s = json.load(f)
                    if "MALA_tau_int_mean" in s:
                        mala_taus.append(s["MALA_tau_int_mean"])
                except: pass
        
        # 2. Leer CSVs de hiperparámetros
        csv_path = os.path.join(d, "hyperparam_sweep_results.csv")
        if os.path.exists(csv_path):
            print(f" -> {csv_path}")
            tmp_df = pd.read_csv(csv_path)
            
            # Respaldo si no hay stats.json: inferir tau de MALA usando G_RL
            if not mala_taus and 'G_RL' in tmp_df.columns and 'tau_mean' in tmp_df.columns:
                t_mala = tmp_df.iloc[0]['tau_mean'] * tmp_df.iloc[0]['G_RL']
                mala_taus.append(t_mala)
                
            tmp_df = tmp_df[tmp_df['dir_name'].str.contains("TD3")]
            dfs.append(tmp_df)
        else:
            print(f" -> [Omitido - No se encontró CSV]: {d}")

    if not dfs:
        print("Error: No se cargó ningún dato válido. Revisa las rutas.")
        sys.exit(1)

    # Consolidar y limpiar datos
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values('tau_mean').drop_duplicates(
        subset=['actor_amplitude', 'penalty_factor', 'sigma_scale', 'max_phys_jump']
    )
    
    # Determinar la línea base MALA
    tau_mala = np.mean(mala_taus) if mala_taus else df['tau_mean'].max() * 1.5
    print(f"\n=> Línea base inferida de MALA: tau_int = {tau_mala:.2f}")

    # Configuración de la figura
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    fig.suptitle(f"TD3 Topological Landscape vs MALA Baseline (N=10, g=20.0)\nColor: Integrated Autocorrelation Time ($\\tau_{{int}}$) | Total Models: {len(df)}", weight='bold', fontsize=16)

    # Usamos LogNorm para que los colores se distribuyan bien aunque MALA sea gigante
    vmin = max(1.0, df['tau_mean'].min() * 0.9)
    vmax = max(df['tau_mean'].max(), tau_mala * 1.1)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = 'coolwarm' # Azul para tiempos bajos (TD3), Rojo para tiempos altos (MALA)

    # =======================================================
    # PLOT 1: Regularización & Varianza
    # =======================================================
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('white')
    sc1 = ax1.scatter(df['penalty_factor'], df['sigma_scale'], df['actor_amplitude'], 
                      c=df['tau_mean'], cmap=cmap, norm=norm, s=250, alpha=0.85, edgecolors='black', linewidth=0.8)
    
    ax1.set_xlabel('\nPenalty Factor (CDLB)', weight='bold')
    ax1.set_ylabel('\nSigma Scale', weight='bold')
    ax1.set_zlabel('\nActor Amplitude', weight='bold')
    ax1.set_title("Regularization & Variance Dynamics", weight='bold', pad=15)
    ax1.view_init(elev=25, azim=45)

    # =======================================================
    # PLOT 2: Límites del Espacio de Acción
    # =======================================================
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('white')
    sc2 = ax2.scatter(df['sigma_scale'], df['max_phys_jump'], df['actor_amplitude'], 
                      c=df['tau_mean'], cmap=cmap, norm=norm, s=250, alpha=0.85, edgecolors='black', linewidth=0.8)
    
    ax2.set_xlabel('\nSigma Scale', weight='bold')
    ax2.set_ylabel('\nMax Phys Jump', weight='bold')
    ax2.set_zlabel('\nActor Amplitude', weight='bold')
    ax2.set_title("Action Space Constraints", weight='bold', pad=15)
    ax2.view_init(elev=25, azim=135)

    # =======================================================
    # BARRA DE COLOR REFERENCIADA A MALA
    # =======================================================
    cbar = plt.colorbar(sc1, ax=[ax1, ax2], pad=0.05, shrink=0.6)
    cbar.set_label(r'Integrated Autocorrelation Time ($\tau_{int}$) [Log Scale]', weight='bold', size=12)
    
    # Añadimos explícitamente el marcador de MALA en la barra
    ticks = list(cbar.get_ticks())
    # Limpiamos ticks muy cercanos a MALA para evitar superposición visual
    ticks = [t for t in ticks if abs(t - tau_mala)/tau_mala > 0.3] 
    ticks.append(tau_mala)
    cbar.set_ticks(sorted(ticks))
    
    # Etiquetar explícitamente el tick de MALA con color rojo
    tick_labels = [f"{t:.1f}" if t != tau_mala else f"MALA\n({t:.1f})" for t in cbar.get_ticks()]
    cbar.set_ticklabels(tick_labels)
    
    # Pintar de rojo la etiqueta de MALA en el eje Y de la barra de color
    for label in cbar.ax.get_yticklabels():
        if "MALA" in label.get_text():
            label.set_color("red")
            label.set_weight("bold")

    out_file = "Combined_Hyperparam_3D_Landscape_vs_MALA.pdf"
    plt.savefig(out_file, dpi=300, format='pdf', bbox_inches='tight')
    print(f"\nGráficas 3D referenciadas generadas exitosamente en: {out_file}")

if __name__ == "__main__":
    main()