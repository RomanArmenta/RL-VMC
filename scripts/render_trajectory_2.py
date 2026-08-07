import os
import glob
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
os.environ["JAX_ENABLE_X64"] = "True"

import jax
import jax.numpy as jnp
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.system import PhysicalSystem
from src.wavefunctions import DeepQMCAnsatz
from scripts.run_experiment import SYSTEMS

def evaluate_grid_in_batches(ansatz_apply, params, bg_walker, grid_pts, batch_size=5000):
    n_pts = len(grid_pts)
    densities = []
    
    @jax.jit
    def compute_density(w):
        log_psi = ansatz_apply(params, w)
        return jnp.exp(2.0 * log_psi) # |Psi|^2

    for i in range(0, n_pts, batch_size):
        chunk = grid_pts[i:i+batch_size]
        batched_w = jnp.repeat(bg_walker[None, ...], len(chunk), axis=0)
        batched_w = batched_w.at[:, 0, :].set(chunk)
        densities.append(np.array(compute_density(batched_w)))
        print(f"\r  Evaluating slice... {min(i+batch_size, n_pts)}/{n_pts} points", end="")
        
    print()
    return np.concatenate(densities)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_folder", type=str, required=True, help="Ruta a la carpeta /traj")
    parser.add_argument("--system", type=str, default="B", help="Nombre del sistema físico")
    parser.add_argument("--grid_res", type=int, default=150, help="Resolución del plano 2D (ej. 150x150)")
    parser.add_argument("--box_size", type=float, default=4.0, help="Tamaño de la caja en Bohrs")
    args = parser.parse_args()

    # 1. Configurar Sistema
    sys_config = SYSTEMS[args.system]
    molecule = PhysicalSystem(
        name=args.system, n_elec=sys_config["n_elec"], n_up=sys_config["n_up"],
        n_dn=sys_config["n_dn"], coords=jnp.array(sys_config["coords"]),
        charges=jnp.array(sys_config["charges"], dtype=jnp.int32)
    )
    ansatz = DeepQMCAnsatz(molecule)

    # 2. Crear un plano 2D (Z=0)
    print(f"Generating 2D spatial slice ({args.grid_res}x{args.grid_res} points)...")
    lin = jnp.linspace(-args.box_size, args.box_size, args.grid_res)
    X, Y = jnp.meshgrid(lin, lin)
    Z = jnp.zeros_like(X) # Fijamos Z=0 para cortar por el centro
    grid_pts = jnp.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # 3. Leer archivos
    pkl_files = sorted(glob.glob(os.path.join(args.traj_folder, "snapshot_*.pkl")))
    if not pkl_files:
        raise FileNotFoundError(f"No snapshot files found in {args.traj_folder}")

    n_files = len(pkl_files)
    fig, axes = plt.subplots(1, n_files, figsize=(4 * n_files, 4.5), constrained_layout=True)
    if n_files == 1:
        axes = [axes]

    # Para normalizar el color entre todas las tramas
    vmax = 0 

    for idx, (ax, file_path) in enumerate(zip(axes, pkl_files)):
        print(f"\nProcessing {os.path.basename(file_path)}...")
        
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        params = data["params_psi"]
        bg_walker = jnp.array(data["background_walker"])

        # Evaluar
        density_1d = evaluate_grid_in_batches(ansatz.apply, params, bg_walker, grid_pts)
        density_2d = density_1d.reshape(args.grid_res, args.grid_res)
        
        vmax = max(vmax, np.max(density_2d)) # Guardar el máximo para la barra de color

        # Graficar contornos rellenos
        contour = ax.contourf(X, Y, density_2d, levels=60, cmap='inferno')
        
        # Dibujar el núcleo
        ax.plot(0, 0, marker='+', color='white', markersize=12, markeredgewidth=2)
        ax.scatter(bg_walker[1:, 0], bg_walker[1:, 1], color='cyan', 
                   marker='o', s=30, edgecolors='white', zorder=5)
        ax.set_title(f"Step {data['step']}")
        ax.set_xlabel("X (Bohrs)")
        if idx == 0:
            ax.set_ylabel("Y (Bohrs)")
        else:
            ax.set_yticks([]) # Ocultar eje Y para los demás subplots
            
        ax.set_aspect('equal')

    # Añadir barra de color global
    fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label=r'Density $|\Psi|^2$')

    out_image_path = os.path.join(args.traj_folder, f"slice_evolution_{args.system}.png")
    fig.savefig(out_image_path, dpi=300, bbox_inches='tight')
    print(f"\nRender saved successfully to: {out_image_path}")

if __name__ == "__main__":
    main()