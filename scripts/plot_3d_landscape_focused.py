import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Renderiza el paisaje 3D dual para RL-MALA.")
    parser.add_argument("--sys_dir", type=str, default="results_example_N10_g20/results_N10_g20.0", help="Carpeta del sistema")
    args = parser.parse_args()

    csv_path = os.path.join(args.sys_dir, "hyperparam_sweep_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo {csv_path}.", file=sys.stderr)
        print("El script evaluador debe terminar primero.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    # Filtrar estrictamente los TD3
    df = df[df['dir_name'].str.contains("TD3")].copy()

    # Extraer el Tau de Integración desde el nombre de la carpeta (ya que no era una columna explícita)
    df['integration_tau'] = df['dir_name'].str.extract(r'tau([0-9.]+)').astype(float)

    # Configuración de la figura dual
    fig = plt.figure(figsize=(20, 8))
    sys_name = os.path.basename(args.sys_dir).replace('results_', '').replace('_', ', ')
    fig.suptitle(f"TD3-MALA Topological Landscape ({sys_name})\nColor: Integrated Autocorrelation Time ($\\tau_{int}$) [Lower is Better/Blue]", weight='bold', fontsize=16)

    # Colormap: coolwarm invertido (queremos que los Taus BAJOS sean Azules y los ALTOS sean Rojos)
    cmap = 'coolwarm'

    # =======================================================
    # PLOT 1: Integración & Restricción (Tau vs Pen vs Amp)
    # =======================================================
    ax1 = fig.add_subplot(121, projection='3d')
    # Usamos Log10 para el Tau porque abarca varios órdenes de magnitud (0.0001 a 0.01)
    x1 = np.log10(df['integration_tau']) 
    y1 = df['penalty_factor']
    z1 = df['actor_amplitude']
    c1 = df['tau_mean']
    
    sc1 = ax1.scatter(x1, y1, z1, c=c1, cmap=cmap, s=200, alpha=0.9, edgecolors='black', linewidth=0.8)
    
    ax1.set_xlabel('\nLog10(Integration Tau)', weight='bold')
    ax1.set_ylabel('\nPenalty Factor (CDLB)', weight='bold')
    ax1.set_zlabel('\nActor Amplitude', weight='bold')
    ax1.set_title("Integration & Regularization Dynamics", weight='bold', pad=15)
    ax1.view_init(elev=25, azim=45)

    # =======================================================
    # PLOT 2: Espacio de Acción (Sigma vs Jump vs Amp)
    # =======================================================
    ax2 = fig.add_subplot(122, projection='3d')
    x2 = df['sigma_scale']
    y2 = df['max_phys_jump']
    z2 = df['actor_amplitude']
    c2 = df['tau_mean']
    
    sc2 = ax2.scatter(x2, y2, z2, c=c2, cmap=cmap, s=200, alpha=0.9, edgecolors='black', linewidth=0.8)
    
    ax2.set_xlabel('\nSigma Scale', weight='bold')
    ax2.set_ylabel('\nMax Phys Jump', weight='bold')
    ax2.set_zlabel('\nActor Amplitude', weight='bold')
    ax2.set_title("Actor Action Space Geometry", weight='bold', pad=15)
    ax2.view_init(elev=25, azim=135)

    # Barra de color compartida
    cbar = plt.colorbar(sc1, ax=[ax1, ax2], pad=0.05, shrink=0.6)
    cbar.set_label(r'Topological Autocorrelation Time ($\tau_{int}$)', weight='bold', size=12)

    # Guardado
    out_file = os.path.join(args.sys_dir, "Hyperparam_3D_Landscape_Dual.pdf")
    plt.savefig(out_file, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Gráficas 3D generadas exitosamente en: {out_file}")

if __name__ == "__main__":
    main()