import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Energías de referencia aproximadas (Hartree) para calcular la energía de correlación
REFERENCE_ENERGIES = {
    "H2":  {"E_HF": -1.1336, "E_exact": -1.1745},
    "He":  {"E_HF": -2.8617, "E_exact": -2.9037},
    "LiH": {"E_HF": -7.9873, "E_exact": -8.0705},
    "Be":  {"E_HF": -14.5730, "E_exact": -14.6673},
    "Li2": {"E_HF": -14.8715, "E_exact": -14.9954},
    "B":   {"E_HF": -24.5290, "E_exact": -24.6539},
    "cyclobutadiene": {"E_HF": -150.0, "E_exact": -151.0} # Placeholder para pruebas
}

COLORS = {"MALA": "gray", "RL-MALA": "#1f77b4"}

def get_correlation_energy(e_calc, sys_name):
    if sys_name not in REFERENCE_ENERGIES:
        return e_calc # Si no hay referencia, retorna la energía cruda
    e_hf = REFERENCE_ENERGIES[sys_name]["E_HF"]
    e_exact = REFERENCE_ENERGIES[sys_name]["E_exact"]
    return 100.0 * (e_calc - e_hf) / (e_exact - e_hf)

def main():
    parser = argparse.ArgumentParser(description="Graficar E_corr, Recompensa y Aceptación desde los resultados.")
    parser.add_argument("folders", nargs="+", help="Rutas a las carpetas de resultados dentro de results/")
    parser.add_argument("--smoothing", type=float, default=0.05, help="Factor alpha para el suavizado EMA (default: 0.05)")
    args = parser.parse_args()

    n_systems = len(args.folders)
    fig, axes = plt.subplots(nrows=n_systems, ncols=3, figsize=(16, 4 * n_systems), squeeze=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    for i, folder_path in enumerate(args.folders):
        folder_name = os.path.basename(os.path.normpath(folder_path))
        sys_name = folder_name.split("_")[0] # Asume formato Sistema_config...

        for method in ["MALA", "RL-MALA"]:
            # El log original asume "training_metrics_{method}.csv" pero revisa la estructura
            csv_file = [f for f in os.listdir(folder_path) if f.endswith(f"training_metrics_{method}.csv")]
            
            if not csv_file:
                continue
            
            df = pd.read_csv(os.path.join(folder_path, csv_file[0]))
            
            # --- Columna 1: Energía de Correlación (%) ---
            ax_e = axes[i, 0]
            # Extraemos la energía (si la columna se llama 'E_mean' o 'E', ajusta según tu df)
            e_col = 'E_mean' if 'E_mean' in df.columns else 'E' 
            corr_energy = get_correlation_energy(df[e_col], sys_name)
            corr_energy_smoothed = corr_energy.ewm(alpha=args.smoothing).mean()
            
            ax_e.plot(df["Step"], corr_energy_smoothed, label=method, color=COLORS[method], linewidth=2.0, alpha=0.8)
            ax_e.set_xscale("log")
            
            # Formateo si se encontró energía de referencia
            if sys_name in REFERENCE_ENERGIES:
                ax_e.set_ylim(-5, 105)
                ax_e.set_yticks([0, 90, 99, 100])
            ax_e.grid(True, which="both", ls="--", alpha=0.4)

            # --- Columna 2: Historial de Recompensas ---
            ax_r = axes[i, 1]
            rew_col = 'Rew' if 'Rew' in df.columns else 'Reward'
            if rew_col in df.columns and method == "RL-MALA":
                rew_smoothed = df[rew_col].ewm(alpha=args.smoothing).mean()
                ax_r.plot(df["Step"], rew_smoothed, color=COLORS[method], linewidth=2.0)
            ax_r.set_xscale("log")
            ax_r.grid(True, which="both", ls="--", alpha=0.4)

            # --- Columna 3: Tasa de Aceptación ---
            ax_a = axes[i, 2]
            acc_col = 'Acc' if 'Acc' in df.columns else 'Acceptance'
            if acc_col in df.columns:
                acc_smoothed = df[acc_col].ewm(alpha=args.smoothing).mean()
                ax_a.plot(df["Step"], acc_smoothed, color=COLORS[method], linewidth=2.0, alpha=0.8)
            ax_a.set_ylim(0, 1.0)
            ax_a.set_xscale("log")
            ax_a.grid(True, which="both", ls="--", alpha=0.4)

        # Títulos de los ejes Y (solo en los bordes para mantenerlo limpio)
        axes[i, 0].set_ylabel(f"{sys_name}\nCorr. Energy (%)", fontsize=12, fontweight='bold')
        if i == 0:
            axes[0, 1].set_title("Actor Reward (RL-MALA)", fontsize=14, fontweight='bold')
            axes[0, 2].set_title("MCMC Acceptance Rate", fontsize=14, fontweight='bold')
            axes[0, 0].set_title("Correlation Energy", fontsize=14, fontweight='bold')

    # Etiquetas X y Leyenda general
    for ax in axes[-1, :]:
        ax.set_xlabel("Training Steps", fontsize=12)
        
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=12)

    out_file = "Fig4_Replication_Detailed.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Gráfica generada y guardada como: {out_file}")

if __name__ == "__main__":
    main()