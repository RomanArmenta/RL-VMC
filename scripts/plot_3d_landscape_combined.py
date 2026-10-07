import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Renderiza el paisaje 3D combinando múltiples carpetas.")
    # Permite recibir una lista indefinida de carpetas
    parser.add_argument("--sys_dirs", nargs='+', required=True, help="Lista de carpetas (ej. dir1 dir2 dir3)")
    args = parser.parse_args()

    dfs = []
    print("Recolectando datos de:")
    for d in args.sys_dirs:
        csv_path = os.path.join(d, "hyperparam_sweep_results.csv")
        if os.path.exists(csv_path):
            print(f" -> {csv_path}")
            tmp_df = pd.read_csv(csv_path)
            # Filtrar solo los modelos de RL (TD3)
            tmp_df = tmp_df[tmp_df['dir_name'].str.contains("TD3")]
            dfs.append(tmp_df)
        else:
            print(f" -> [Omitido - No se encontró CSV]: {d}")

    if not dfs:
        print("Error: No se cargó ningún dato válido. Revisa las rutas.")
        sys.exit(1)

    # Fusionar todos los dataframes
    df = pd.concat(dfs, ignore_index=True)
    
    # Eliminar configuraciones duplicadas (por si ambos scripts evaluaron lo mismo)
    # Conservamos el que tenga menor tau_mean (el mejor intento)
    df = df.sort_values('tau_mean').drop_duplicates(
        subset=['actor_amplitude', 'penalty_factor', 'sigma_scale', 'max_phys_jump']
    )

    print(f"\nTotal de configuraciones únicas consolidadas: {len(df)}")

    # Configuración de la figura dual
    fig = plt.figure(figsize=(20, 8))
    fig.patch.set_facecolor('white')
    
    fig.suptitle(f"Combined TD3-MALA Topological Landscape (N=10, g=20.0)\nColor: Integrated Autocorrelation Time ($\\tau_{int}$) [Lower is Better/Blue] | Total Models: {len(df)}", weight='bold', fontsize=16)

    # Colormap: coolwarm invertido (Taus BAJOS/buenos = Azul, Taus ALTOS/malos = Rojo)
    cmap = 'coolwarm'

    # =======================================================
    # PLOT 1: Regularización & Varianza (Pen vs Sig vs Amp)
    # =======================================================
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('white')
    
    x1 = df['penalty_factor']
    y1 = df['sigma_scale']
    z1 = df['actor_amplitude']
    c1 = df['tau_mean']
    
    sc1 = ax1.scatter(x1, y1, z1, c=c1, cmap=cmap, s=250, alpha=0.85, edgecolors='black', linewidth=0.8)
    
    ax1.set_xlabel('\nPenalty Factor (CDLB)', weight='bold')
    ax1.set_ylabel('\nSigma Scale', weight='bold')
    ax1.set_zlabel('\nActor Amplitude', weight='bold')
    ax1.set_title("Regularization & Variance Dynamics", weight='bold', pad=15)
    ax1.view_init(elev=25, azim=45)

    # =======================================================
    # PLOT 2: Límites del Espacio de Acción (Sig vs Jump vs Amp)
    # =======================================================
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_facecolor('white')
    
    x2 = df['sigma_scale']
    y2 = df['max_phys_jump']
    z2 = df['actor_amplitude']
    c2 = df['tau_mean']
    
    sc2 = ax2.scatter(x2, y2, z2, c=c2, cmap=cmap, s=250, alpha=0.85, edgecolors='black', linewidth=0.8)
    
    ax2.set_xlabel('\nSigma Scale', weight='bold')
    ax2.set_ylabel('\nMax Phys Jump', weight='bold')
    ax2.set_zlabel('\nActor Amplitude', weight='bold')
    ax2.set_title("Action Space Constraints", weight='bold', pad=15)
    ax2.view_init(elev=25, azim=135)

    # Barra de color compartida
    cbar = plt.colorbar(sc1, ax=[ax1, ax2], pad=0.05, shrink=0.6)
    cbar.set_label(r'Integrated Autocorrelation Time ($\tau_{int}$)', weight='bold', size=12)

    # Guardado en el directorio actual para fácil acceso
    out_file = "Combined_Hyperparam_3D_Landscape_N10_g20.pdf"
    plt.savefig(out_file, dpi=300, format='pdf', bbox_inches='tight')
    print(f"Gráficas 3D combinadas generadas exitosamente en: {out_file}")

if __name__ == "__main__":
    main()