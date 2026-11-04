#!/usr/bin/env python3
"""
scripts/plot_benchmark_all_models_summary.py
==============================================================================
Genera las figuras representativas de síntesis global de todos los modelos
obtenidos a partir del archivo maestro `live_all_models_master.csv` generado
por el benchmark de 3 pipelines.

Produce 4 figuras diagnósticas de calidad de publicación:
  1. fig_all_models_speedup_ess.png:
     Aceleración del Actor G_RL y Eficiencia de Muestreo (ESS/seg) vs g para cada N.
  2. fig_all_models_acceptance_esjd.png:
     Tasa de aceptación MCMC y distancia de salto ESJD en función de la correlación g.
  3. fig_all_models_fidelity_kl_energy.png:
     Fidelidad física: Divergencia KL respecto a CASSCF y error de energía Delta E vs g.
  4. fig_all_models_executive_dashboard.png:
     Dashboard panorámico comparando los 4 regímenes (P2 MALA, P2 RL, P3 VMC, P3 Coupled).

Uso:
  python3 scripts/plot_benchmark_all_models_summary.py --csv results_csv_b2/live_all_models_master.csv --out_dir figures/benchmark_summary
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuración estética general
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
})

N_COLORS = {
    2: "#1f77b4",   # Azul
    4: "#2ca02c",   # Verde
    6: "#ff7f0e",   # Naranja
    10: "#d62728",  # Rojo
}
N_MARKERS = {
    2: "o",
    4: "s",
    6: "^",
    10: "D",
}


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Carga y normaliza los tipos de datos del CSV maestro."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[+] Cargando {len(df)} modelos desde: {csv_path}")

    # Columnas numéricas
    num_cols = [
        "N", "g", "omega", "softening", "E_mean", "SE_E", "E_var",
        "acceptance", "ESJD", "accepted_disp", "tau_int_E", "tau_int_pos",
        "G_RL_speedup", "ESS", "ESS_per_sec", "casscf_energy",
        "delta_E_vs_CASSCF", "kl_casscf_total_mean", "kl_casscf_up_mean",
        "kl_casscf_down_mean", "time_to_solution_sec"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clasificación por método y pipeline
    def classify_row(label: str):
        l = str(label)
        if "P2_MALA" in l or "MALA_Fixed" in l:
            return "P2: MALA (CASSCF Fijo)"
        elif "P2_RL" in l or "RL_TD3_Fixed" in l:
            return "P2: RL-MALA TD3 (CASSCF Fijo)"
        elif "P3_VMC_MALA" in l or "VMC_MALA" in l:
            return "P3: VMC + MALA (SJB)"
        elif "P3_Coupled_RL" in l or "Coupled_RL" in l:
            return "P3: Coupled VMC + RL-MALA (SJB)"
        return "Otro"

    df["pipeline_category"] = df["label"].apply(classify_row)
    return df


def plot_speedup_and_ess(df: pd.DataFrame, out_dir: str):
    """Figura 1: Speedup G_RL y Eficiencia ESS/seg vs fuerza de acoplamiento g."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    p3_rl = df[df["pipeline_category"] == "P3: Coupled VMC + RL-MALA (SJB)"]
    p3_mala = df[df["pipeline_category"] == "P3: VMC + MALA (SJB)"]

    n_values = sorted(df["N"].dropna().unique())

    # Panel (0, 0): Speedup G_RL en Pipeline 2
    ax = axes[0, 0]
    for n in n_values:
        sub = p2_rl[p2_rl["N"] == n].sort_values("g")
        if not sub.empty:
            ax.plot(sub["g"], sub["G_RL_speedup"], color=N_COLORS.get(n, "black"),
                    marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Paridad MALA (G=1)")
    ax.set_ylabel(r"Factor de Aceleración $G_{\mathrm{RL}} = \tau_{\mathrm{int}}^{\mathrm{MALA}} / \tau_{\mathrm{int}}^{\mathrm{RL}}$")
    ax.set_title("Pipeline 2: Aceleración de Muestreo sobre CASSCF Congelado", weight="bold")
    ax.grid(True)
    ax.legend()

    # Panel (0, 1): Speedup G_RL en Pipeline 3
    ax = axes[0, 1]
    for n in n_values:
        sub = p3_rl[p3_rl["N"] == n].sort_values("g")
        if not sub.empty:
            ax.plot(sub["g"], sub["G_RL_speedup"], color=N_COLORS.get(n, "black"),
                    marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Paridad MALA (G=1)")
    ax.set_ylabel(r"Factor de Aceleración $G_{\mathrm{RL}}$")
    ax.set_title("Pipeline 3: Aceleración Acoplada en Entrenamiento VMC (SJB)", weight="bold")
    ax.grid(True)
    ax.legend()

    # Panel (1, 0): ESS/segundo en Pipeline 2 (MALA vs RL)
    ax = axes[1, 0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["ESS_per_sec"], color=c, linestyle="--", marker="x", alpha=0.7)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["ESS_per_sec"], color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel("Muestras Efectivas por Segundo (ESS/s)")
    ax.set_title("Pipeline 2: Rendimiento Absoluto ESS/s (Línea Continua: RL, Discontinua: MALA)", weight="bold")
    ax.grid(True, which="both")
    ax.legend()

    # Panel (1, 1): ESS/segundo en Pipeline 3 (MALA vs RL)
    ax = axes[1, 1]
    for n in n_values:
        sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
        sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["ESS_per_sec"], color=c, linestyle="--", marker="x", alpha=0.7)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["ESS_per_sec"], color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel("Muestras Efectivas por Segundo (ESS/s)")
    ax.set_title("Pipeline 3: Rendimiento Absoluto ESS/s (Línea Continua: RL, Discontinua: MALA)", weight="bold")
    ax.grid(True, which="both")
    ax.legend()

    fig.suptitle("Comportamiento Global del Actor RL: Aceleración de Autocorrelación y Rendimiento ESS", weight="bold")
    fig.tight_layout()
    out_file = os.path.join(out_dir, "fig_all_models_speedup_ess.png")
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"[+] Figura guardada: {out_file}")


def plot_acceptance_and_esjd(df: pd.DataFrame, out_dir: str):
    """Figura 2: Tasa de Aceptación MCMC y Distancia de Salto ESJD vs g."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True)

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    p3_rl = df[df["pipeline_category"] == "P3: Coupled VMC + RL-MALA (SJB)"]
    p3_mala = df[df["pipeline_category"] == "P3: VMC + MALA (SJB)"]

    n_values = sorted(df["N"].dropna().unique())

    # Panel (0, 0): Tasa de Aceptación P2
    ax = axes[0, 0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["acceptance"] * 100.0, color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["acceptance"] * 100.0, color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.set_title("Pipeline 2: Aceptación MCMC (Continua: RL, Discontinua: MALA)", weight="bold")
    ax.grid(True)
    ax.legend()

    # Panel (0, 1): Tasa de Aceptación P3
    ax = axes[0, 1]
    for n in n_values:
        sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
        sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["acceptance"] * 100.0, color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["acceptance"] * 100.0, color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.set_title("Pipeline 3: Aceptación MCMC en VMC (Continua: RL, Discontinua: MALA)", weight="bold")
    ax.grid(True)
    ax.legend()

    # Panel (1, 0): ESJD en P2
    ax = axes[1, 0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["ESJD"], color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["ESJD"], color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Distancia Cuadrática de Salto ($\mathrm{ESJD}$)")
    ax.set_title("Pipeline 2: Movilidad Espacial ESJD (Salto Efectivo en Configuración)", weight="bold")
    ax.grid(True)
    ax.legend()

    # Panel (1, 1): ESJD en P3
    ax = axes[1, 1]
    for n in n_values:
        sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
        sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty:
            ax.plot(sub_m["g"], sub_m["ESJD"], color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty:
            ax.plot(sub_r["g"], sub_r["ESJD"], color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Distancia Cuadrática de Salto ($\mathrm{ESJD}$)")
    ax.set_title("Pipeline 3: Movilidad Espacial ESJD en VMC Acoplado", weight="bold")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Exploración del Espacio de Configuración: Aceptación y Movilidad Cuadrática ESJD", weight="bold")
    fig.tight_layout()
    out_file = os.path.join(out_dir, "fig_all_models_acceptance_esjd.png")
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"[+] Figura guardada: {out_file}")


def plot_fidelity_and_energy(df: pd.DataFrame, out_dir: str):
    """Figura 3: Fidelidad respecto a CASSCF (D_KL y Delta E) vs g."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    n_values = sorted(df["N"].dropna().unique())

    # Panel 0: Divergencia KL Total respecto a CASSCF
    ax = axes[0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty and "kl_casscf_total_mean" in sub_m.columns:
            ax.plot(sub_m["g"], sub_m["kl_casscf_total_mean"], color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty and "kl_casscf_total_mean" in sub_r.columns:
            ax.plot(sub_r["g"], sub_r["kl_casscf_total_mean"], color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Divergencia KL Total $D_{\mathrm{KL}}(\rho \parallel \rho_{\mathrm{CASSCF}})$")
    ax.set_title("Fidelidad de la Densidad MCMC vs CASSCF Analítico (Menor es Mejor)", weight="bold")
    ax.grid(True, which="both")
    ax.legend()

    # Panel 1: Error Absoluto de Energía |E - E_CASSCF|
    ax = axes[1]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(n, "black")
        if not sub_m.empty and "delta_E_vs_CASSCF" in sub_m.columns:
            ax.plot(sub_m["g"], np.abs(sub_m["delta_E_vs_CASSCF"]), color=c, linestyle="--", marker="x", alpha=0.6)
        if not sub_r.empty and "delta_E_vs_CASSCF" in sub_r.columns:
            ax.plot(sub_r["g"], np.abs(sub_r["delta_E_vs_CASSCF"]), color=c, linestyle="-", marker=N_MARKERS.get(n, "o"), label=f"N = {int(n)}")
    ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Error Absoluto de Energía $|E_{\mathrm{MCMC}} - E_{\mathrm{CASSCF}}|$ (Ha)")
    ax.set_title("Precisión de la Energía Local MCMC vs Referencia CASSCF", weight="bold")
    ax.grid(True, which="both")
    ax.legend()

    fig.suptitle("Precisión Física y Convergencia Asintótica del Sampler MALA vs RL-MALA", weight="bold")
    fig.tight_layout()
    out_file = os.path.join(out_dir, "fig_all_models_fidelity_kl_energy.png")
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"[+] Figura guardada: {out_file}")


def plot_executive_dashboard(df: pd.DataFrame, out_dir: str):
    """Figura 4: Dashboard Ejecutivo Comparativo de los 4 Regímenes."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 0: Distribución de Tiempos de Autocorrelación tau_int por Método
    methods = [
        "P2: MALA (CASSCF Fijo)",
        "P2: RL-MALA TD3 (CASSCF Fijo)",
        "P3: VMC + MALA (SJB)",
        "P3: Coupled VMC + RL-MALA (SJB)"
    ]
    data_tau = [df[df["pipeline_category"] == m]["tau_int_E"].dropna() for m in methods]
    labels_clean = ["P2 MALA", "P2 RL-MALA", "P3 VMC+MALA", "P3 Coupled RL"]

    ax = axes[0, 0]
    ax.boxplot(data_tau, labels=labels_clean, patch_artist=True,
               boxprops=dict(facecolor="#cce5ff", color="#004085"),
               medianprops=dict(color="red", lw=2))
    ax.set_yscale("log")
    ax.set_ylabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int},E}$ (pasos)")
    ax.set_title("Comparativa de Autocorrelación entre los 4 Regímenes", weight="bold")
    ax.grid(True, which="both")

    # Panel 1: Tasa de Aceptación MCMC por Método
    data_acc = [df[df["pipeline_category"] == m]["acceptance"].dropna() * 100.0 for m in methods]
    ax = axes[0, 1]
    ax.boxplot(data_acc, labels=labels_clean, patch_artist=True,
               boxprops=dict(facecolor="#d4edda", color="#155724"),
               medianprops=dict(color="darkgreen", lw=2))
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.set_title("Distribución de Tasa de Aceptación MCMC", weight="bold")
    ax.grid(True)

    # Panel 2: ESS/s promedio por Sistema N
    ax = axes[1, 0]
    n_values = sorted(df["N"].dropna().unique())
    x_pos = np.arange(len(n_values))
    width = 0.35

    p2_rl_means = [df[(df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)") & (df["N"] == n)]["ESS_per_sec"].mean() for n in n_values]
    p2_m_means = [df[(df["pipeline_category"] == "P2: MALA (CASSCF Fijo)") & (df["N"] == n)]["ESS_per_sec"].mean() for n in n_values]

    ax.bar(x_pos - width/2, p2_m_means, width, label="P2 MALA", color="gray", alpha=0.7)
    ax.bar(x_pos + width/2, p2_rl_means, width, label="P2 RL-MALA TD3", color="crimson", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"N = {int(n)}" for n in n_values])
    ax.set_yscale("log")
    ax.set_ylabel("ESS/segundo Promedio")
    ax.set_title("Eficiencia de Producción Promedio vs Tamaño del Sistema (N)", weight="bold")
    ax.grid(True, which="both")
    ax.legend()

    # Panel 3: Diagrama de Dispersión: Error de Energía vs Autocorrelación tau_int
    ax = axes[1, 1]
    for cat, color, marker in [
        ("P2: MALA (CASSCF Fijo)", "gray", "o"),
        ("P2: RL-MALA TD3 (CASSCF Fijo)", "crimson", "*"),
        ("P3: VMC + MALA (SJB)", "blue", "s"),
        ("P3: Coupled VMC + RL-MALA (SJB)", "purple", "^"),
    ]:
        sub = df[df["pipeline_category"] == cat]
        if not sub.empty:
            ax.scatter(sub["tau_int_E"], np.abs(sub["delta_E_vs_CASSCF"]), color=color, marker=marker,
                       s=70, alpha=0.75, label=cat)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int},E}$ (Menor es Mejor)")
    ax.set_ylabel(r"Error Absoluto de Energía $|E - E_{\mathrm{CASSCF}}|$ (Ha)")
    ax.set_title("Frente de Pareto: Precisión de Energía vs Rapidez MCMC", weight="bold")
    ax.grid(True, which="both")
    ax.legend(fontsize=8.5)

    fig.suptitle("Dashboard Ejecutivo: Evaluación Panorámica de los 96 Modelos del Benchmark", weight="bold")
    fig.tight_layout()
    out_file = os.path.join(out_dir, "fig_all_models_executive_dashboard.png")
    fig.savefig(out_file, dpi=250)
    plt.close(fig)
    print(f"[+] Figura guardada: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Genera figuras panorámicas de todos los modelos del benchmark.")
    parser.add_argument("--csv", type=str, default="results_csv_b2/live_all_models_master.csv", help="Ruta al CSV maestro.")
    parser.add_argument("--out_dir", type=str, default="figures/benchmark_summary", help="Directorio donde guardar las figuras.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_clean_data(args.csv)

    print("=" * 80)
    print("    GENERACIÓN DE FIGURAS REPRESENTATIVAS DE TODOS LOS MODELOS")
    print(f"    Archivo Entrada   : {args.csv}")
    print(f"    Directorio Salida : {args.out_dir}")
    print(f"    Total Modelos     : {len(df)}")
    print("=" * 80)

    plot_speedup_and_ess(df, args.out_dir)
    plot_acceptance_and_esjd(df, args.out_dir)
    plot_fidelity_and_energy(df, args.out_dir)
    plot_executive_dashboard(df, args.out_dir)

    print("=" * 80)
    print("    FIGURAS GENERADAS EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
