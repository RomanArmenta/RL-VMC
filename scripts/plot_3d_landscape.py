import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    parser = argparse.ArgumentParser(description="Renderiza el paisaje 3D de tau_int para TD3.")
    parser.add_argument("--sys_dir", type=str, required=True, help="Carpeta del sistema (ej. results_linear_search_4/results_N10_g10.0)")
    args = parser.parse_args()

    csv_path = os.path.join(args.sys_dir, "hyperparam_sweep_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo de evaluaciones {csv_path}.", file=sys.stderr)
        print("Asegúrate de que evaluate_grid_search_2.py haya terminado exitosamente en esa carpeta.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Filtrar solo los modelos TD3 (por si MALA se coló en el CSV)
    df = df[df['dir_name'].str.contains("TD3")]

    # Configuración de la figura 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Extracción de dimensiones
    x = df['penalty_factor']
    y = df['sigma_scale']
    z = df['max_phys_jump']
    c = df['tau_mean']               # Color = Eficiencia topológica
    s = df['actor_amplitude'] * 3000 # Escalar el tamaño para que sea visible

    # Renderizado (Usamos coolwarm: Azul para tau bajo/bueno, Rojo para tau alto/malo)
    sc = ax.scatter(x, y, z, c=c, cmap='coolwarm', s=s, alpha=0.85, edgecolors='black', linewidth=0.8)

    # Estética y Etiquetas
    ax.set_xlabel('\nPenalty Factor (CDLB)', weight='bold')
    ax.set_ylabel('\nSigma Scale', weight='bold')
    ax.set_zlabel('\nMax Phys Jump', weight='bold')
    
    sys_name = os.path.basename(args.sys_dir).replace('results_', '').replace('_', ', ')
    ax.set_title(f"RL-MALA Topological Landscape ({sys_name})\n"
                 "Color: Autocorrelation Time | Size: Actor Amplitude", weight='bold', fontsize=14, pad=20)

    # Barra de color con formato científico si los números son muy grandes
    cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label(r'Integrated Autocorrelation Time ($\tau_{int}$)', weight='bold', size=12)

    # Ajustar el ángulo de visión inicial para una mejor perspectiva
    ax.view_init(elev=25, azim=45)

    # Guardado
    out_file = os.path.join(args.sys_dir, "Hyperparam_3D_Landscape.pdf")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, format='pdf')
    print(f"Paisaje 3D topológico generado y guardado en: {out_file}")

if __name__ == "__main__":
    main()