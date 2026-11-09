#!/usr/bin/env python3
"""
scripts/plot_benchmark_all_models_summary.py
==============================================================================
Genera las figuras representativas de síntesis global de todos los modelos
obtenidos a partir del archivo maestro `live_all_models_master.csv` (o
`live_pipeline2_sampling_comparison.csv`) generado en tiempo real por
`run_production_benchmark_3pipelines.sh`.

Produce 6 figuras diagnósticas de calidad de publicación (PNG por defecto,
con banderas para PDF, PPI, títulos y escalas):
  1. fig_all_models_vs_g_evolution:
     Dashboard de 6 paneles con barras de error mostrando cómo evolucionan
     D_KL^spin (cruce nodal), D_KL^tot, tau_int(E), ESJD, ESS/s y Delta E +/- SE(E)
     respecto a la fuerza de interacción g para cada número de electrones N.
  2. fig_all_models_vs_N_evolution:
     Dashboard de 6 paneles con barras de error mostrando cómo evolucionan
     D_KL^spin (cruce nodal), D_KL^tot, tau_int(E), ESJD, ESS/s y Delta E +/- SE(E)
     respecto al número de electrones N para cada acoplamiento g.
  3. fig_all_models_speedup_ess:
     Factor de aceleración G_RL y rendimiento ESS/s respecto a g y respecto a N.
  4. fig_all_models_acceptance_esjd:
     Tasa de aceptación MCMC y movilidad cuadrática ESJD respecto a g y a N.
  5. fig_all_models_fidelity_kl_energy:
     Fidelidad física: D_KL^spin, D_KL^tot y energía <E> +/- SE(E) vs E_CASSCF.
  6. fig_all_models_executive_dashboard:
     Dashboard panorámico comparando MALA vs RL-MALA TD3 con remarcado de mejores modelos.

Convención visual:
  - MALA Clásico  : líneas discontinuas (`--`), marcadores abiertos/cruces
  - RL-MALA (TD3) : líneas continuas (`-`), marcadores sólidos rellenos
==============================================================================
"""

import os
import re
import sys
import argparse
import tempfile
import numpy as np
import pandas as pd
import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_COLORS = {
    2: "#1f77b4",   # Azul
    4: "#2ca02c",   # Verde
    6: "#ff7f0e",   # Naranja
    8: "#9467bd",   # Morado
    10: "#d62728",  # Rojo
}
N_MARKERS = {
    2: "o",
    4: "s",
    6: "^",
    8: "P",
    10: "D",
}

G_CMAP_COLORS = [
    "#1b9e77",
    "#2b8cbe",
    "#2ca02c",
    "#ff7f0e",
    "#e7298a",
    "#7570b3",
    "#d62728",
    "#66a61e",
]


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Carga, desduplica y normaliza los tipos de datos del CSV maestro."""
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"No se encontró o está vacío el archivo CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"El archivo CSV no contiene filas: {csv_path}")

    if "label" in df.columns:
        df = df.drop_duplicates(subset=["label"], keep="last").reset_index(drop=True)

    # Inferir N y g desde label si faltan en alguna fila
    if "N" not in df.columns:
        df["N"] = np.nan
    if "g" not in df.columns:
        df["g"] = np.nan

    for idx, row in df.iterrows():
        lbl = str(row.get("label", ""))
        if pd.isna(row.get("N")):
            m_n = re.search(r"N(\d+)", lbl)
            if m_n:
                df.at[idx, "N"] = float(m_n.group(1))
        if pd.isna(row.get("g")):
            m_g = re.search(r"_g([0-9]+(?:\.[0-9]+)?)", lbl)
            if m_g:
                df.at[idx, "g"] = float(m_g.group(1))

    num_cols = [
        "N", "g", "omega", "softening", "E_mean", "SE_E", "E_std_across_chains", "E_var",
        "acceptance", "acceptance_std", "ESJD", "ESJD_std", "accepted_disp",
        "tau_int_E", "tau_int_E_std", "tau_int_pos", "tau_int_pos_std",
        "G_RL_speedup", "ESS", "ESS_per_sec", "ESS_per_sec_std",
        "casscf_energy", "delta_E_vs_CASSCF",
        "kl_casscf_total_mean", "kl_casscf_total_std",
        "kl_casscf_up_mean", "kl_casscf_up_std",
        "kl_casscf_down_mean", "kl_casscf_down_std",
        "kl_casscf_spin_mean", "kl_casscf_spin_std",
        "time_to_solution_sec",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "kl_casscf_spin_mean" not in df.columns or df["kl_casscf_spin_mean"].isna().all():
        if "kl_casscf_up_mean" in df.columns and "kl_casscf_down_mean" in df.columns:
            df["kl_casscf_spin_mean"] = 0.5 * (df["kl_casscf_up_mean"] + df["kl_casscf_down_mean"])
        elif "kl_casscf_total_mean" in df.columns:
            df["kl_casscf_spin_mean"] = df["kl_casscf_total_mean"]

    def classify_row(label: str) -> str:
        l = str(label)
        if "P3_VMC_MALA" in l or "VMC_MALA" in l or "VMC + MALA" in l:
            return "P3: VMC + MALA (SJB)"
        elif "P3_Coupled_RL" in l or "Coupled_RL" in l or "Coupled VMC" in l:
            return "P3: Coupled VMC + RL-MALA (SJB)"
        elif "P2_RL" in l or "RL_TD3" in l or "RL_MALA" in l or "RL-MALA" in l:
            return "P2: RL-MALA TD3 (CASSCF Fijo)"
        elif "P2_MALA" in l or "MALA" in l:
            return "P2: MALA (CASSCF Fijo)"
        return "Otro"

    df["pipeline_category"] = df["label"].apply(classify_row)
    print(f"[+] Cargando {len(df)} modelos únicos desde: {csv_path}")
    return df


def _atomic_savefig(fig: plt.Figure, out_path: str, fmt: str, dpi: int, transparent: bool) -> None:
    """Guarda una figura de matplotlib de forma atómica (temporal + os.replace)."""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(out_path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{stem}.tmp.", suffix=f".{fmt}", dir=out_dir)
    os.close(fd)
    try:
        fig.savefig(tmp_name, format=fmt, dpi=dpi, bbox_inches="tight", transparent=transparent)
        os.replace(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _save_fig_custom(fig: plt.Figure, out_dir: str, stem: str, args: argparse.Namespace) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if getattr(args, "no_titles", False):
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
        for ax in fig.axes:
            ax.set_title("")
    else:
        custom_sup = str(getattr(args, "custom_suptitle", "") or "").strip()
        t_prefix = str(getattr(args, "title_prefix", "") or "").strip()
        f_scale = float(getattr(args, "font_scale", 1.0))
        if custom_sup:
            full_sup = f"{t_prefix} — {custom_sup}" if t_prefix else custom_sup
            fig.suptitle(full_sup, fontsize=14.0 * f_scale, weight="bold", y=1.02)
        elif t_prefix:
            if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
                cur = fig._suptitle.get_text()
                if not cur.startswith(t_prefix):
                    fig._suptitle.set_text(f"{t_prefix} — {cur}")
            else:
                fig.suptitle(t_prefix, fontsize=13.5 * f_scale, weight="bold", y=1.02)

    fmt = getattr(args, "figure_format", "png")
    dpi = int(getattr(args, "dpi", 300))
    transparent = bool(getattr(args, "transparent_bg", False))
    if fmt in ("png", "both"):
        out_png = os.path.join(out_dir, f"{stem}.png")
        _atomic_savefig(fig, out_png, "png", dpi=dpi, transparent=transparent)
        print(f"[+] Figura guardada: {out_png}")
    if fmt in ("pdf", "both"):
        out_pdf = os.path.join(out_dir, f"{stem}.pdf")
        _atomic_savefig(fig, out_pdf, "pdf", dpi=dpi, transparent=transparent)
        print(f"[+] Figura guardada: {out_pdf}")
    plt.close(fig)


def _get_err_series(sub: pd.DataFrame, err_col: str) -> np.ndarray | None:
    if err_col in sub.columns:
        vals = pd.to_numeric(sub[err_col], errors="coerce").to_numpy(dtype=float)
        if np.any(np.isfinite(vals) & (vals > 0)):
            return np.where(np.isfinite(vals) & (vals >= 0), vals, 0.0)
    return None


def _finite_xy(sub: pd.DataFrame, x_col: str, y_col: str, err_col: str | None = None):
    """Extrae pares finitos (x, y) y barras de error alineadas cuando existen."""
    if x_col not in sub.columns or y_col not in sub.columns:
        return np.array([]), np.array([]), None
    x = pd.to_numeric(sub[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([]), np.array([]), None
    x_fin = x[mask]
    y_fin = y[mask]
    yerr_fin = None
    if err_col is not None:
        yerr_all = _get_err_series(sub, err_col)
        if yerr_all is not None:
            cand = yerr_all[mask]
            if np.any(cand > 0):
                yerr_fin = cand
    return x_fin, y_fin, yerr_fin


def _legend_if_nonempty(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kwargs)


def plot_evolution_vs_g(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 1: Dashboard de 6 paneles de evolución de métricas vs g para cada N."""
    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    if p2_rl.empty and p2_mala.empty:
        return

    n_values = sorted(df["N"].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True)

    metrics = [
        (axes[0, 0], "kl_casscf_spin_mean", "kl_casscf_spin_std", r"Cruce Nodal $D_{\mathrm{KL}}^{\mathrm{spin}}$", "Divergencia KL de Espín vs $g$ (Menor es Mejor)", True),
        (axes[0, 1], "kl_casscf_total_mean", "kl_casscf_total_std", r"Fidelidad $D_{\mathrm{KL}}^{\mathrm{tot}}(\rho \parallel \rho_{\mathrm{CASSCF}})$", "Divergencia KL Total vs $g$ (Menor es Mejor)", True),
        (axes[0, 2], "tau_int_E", "tau_int_E_std", r"Autocorrelación $\tau_{\mathrm{int}}(E)$ (pasos)", r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ vs $g$", False),
        (axes[1, 0], "ESJD", "ESJD_std", r"Movilidad Cuadrática $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$)", "Salto Cuadrático Esperado (ESJD) vs $g$", False),
        (axes[1, 1], "ESS_per_sec", "ESS_per_sec_std", r"Velocidad Efectiva $\mathrm{ESS/s}$ ($\mathrm{s}^{-1}$)", "Rendimiento Real ESS/s vs $g$ (Mayor es Mejor)", True),
        (axes[1, 2], "delta_E_vs_CASSCF", "SE_E", r"$\Delta E = \langle E \rangle - E_{\mathrm{CASSCF}}$ (Ha)", r"Precisión de Energía $\Delta E \pm \mathrm{SE}(E)$ vs $g$", False),
    ]

    for ax, col, err_col, ylabel, title, use_log in metrics:
        pos_plotted = False
        if col in df.columns:
            for n in n_values:
                c = N_COLORS.get(int(n), "#333333")
                mk = N_MARKERS.get(int(n), "o")
                sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
                sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")

                xm, ym, yerrm = _finite_xy(sub_m, "g", col, err_col)
                if xm.size > 0:
                    ax.errorbar(
                        xm, ym, yerr=yerrm,
                        color=c, linestyle="--", marker="x", markersize=6.5,
                        capsize=3, alpha=0.72, label=f"MALA N={int(n)}",
                    )
                    if np.any(ym > 0):
                        pos_plotted = True

                xr, yr, yerrr = _finite_xy(sub_r, "g", col, err_col)
                if xr.size > 0:
                    ax.errorbar(
                        xr, yr, yerr=yerrr,
                        color=c, linestyle="-", marker=mk, markersize=7.0,
                        capsize=3, alpha=0.95, label=f"RL-MALA N={int(n)}",
                    )
                    if np.any(yr > 0):
                        pos_plotted = True

        if col == "delta_E_vs_CASSCF":
            ax.axhline(0.0, color="black", linestyle=":", lw=1.6, label=r"Exacto $\Delta E = 0$")
        elif use_log and pos_plotted:
            ax.set_yscale("log")

        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold")
        ax.grid(True, which="both", alpha=0.32)

    for ax in axes[1, :]:
        ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")

    _legend_if_nonempty(axes[0, 0], fontsize=8.5, ncol=2, loc="best")
    _legend_if_nonempty(axes[1, 2], fontsize=8.5, ncol=2, loc="best")

    fig.suptitle(
        "Evolución Global respecto al Acoplamiento g (Por Sistema N): RL-MALA vs MALA",
        weight="bold",
    )
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_vs_g_evolution", args)


def plot_evolution_vs_N(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 2: Dashboard de 6 paneles de evolución de métricas vs N para cada g."""
    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    if p2_rl.empty and p2_mala.empty:
        return

    g_values = sorted(df["g"].dropna().unique())
    n_values = sorted(df["N"].dropna().unique())
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), sharex=True)

    metrics = [
        (axes[0, 0], "kl_casscf_spin_mean", "kl_casscf_spin_std", r"Cruce Nodal $D_{\mathrm{KL}}^{\mathrm{spin}}$", "Divergencia KL de Espín vs $N$ (Menor es Mejor)", True),
        (axes[0, 1], "kl_casscf_total_mean", "kl_casscf_total_std", r"Fidelidad $D_{\mathrm{KL}}^{\mathrm{tot}}(\rho \parallel \rho_{\mathrm{CASSCF}})$", "Divergencia KL Total vs $N$ (Menor es Mejor)", True),
        (axes[0, 2], "tau_int_E", "tau_int_E_std", r"Autocorrelación $\tau_{\mathrm{int}}(E)$ (pasos)", r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ vs $N$", False),
        (axes[1, 0], "ESJD", "ESJD_std", r"Movilidad Cuadrática $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$)", "Salto Cuadrático Esperado (ESJD) vs $N$", False),
        (axes[1, 1], "ESS_per_sec", "ESS_per_sec_std", r"Velocidad Efectiva $\mathrm{ESS/s}$ ($\mathrm{s}^{-1}$)", "Rendimiento Real ESS/s vs $N$ (Mayor es Mejor)", True),
        (axes[1, 2], "delta_E_vs_CASSCF", "SE_E", r"$\Delta E = \langle E \rangle - E_{\mathrm{CASSCF}}$ (Ha)", r"Precisión de Energía $\Delta E \pm \mathrm{SE}(E)$ vs $N$", False),
    ]

    for ax, col, err_col, ylabel, title, use_log in metrics:
        pos_plotted = False
        if col in df.columns:
            for idx_g, g_val in enumerate(g_values):
                c = G_CMAP_COLORS[idx_g % len(G_CMAP_COLORS)]
                sub_m = p2_mala[np.isclose(p2_mala["g"], g_val)].sort_values("N")
                sub_r = p2_rl[np.isclose(p2_rl["g"], g_val)].sort_values("N")

                xm, ym, yerrm = _finite_xy(sub_m, "N", col, err_col)
                if xm.size > 0:
                    ax.errorbar(
                        xm, ym, yerr=yerrm,
                        color=c, linestyle="--", marker="x", markersize=6.0,
                        capsize=3, alpha=0.68, label=f"MALA g={g_val:g}",
                    )
                    if np.any(ym > 0):
                        pos_plotted = True

                xr, yr, yerrr = _finite_xy(sub_r, "N", col, err_col)
                if xr.size > 0:
                    ax.errorbar(
                        xr, yr, yerr=yerrr,
                        color=c, linestyle="-", marker="o", markersize=6.5,
                        capsize=3, alpha=0.95, label=f"RL-MALA g={g_val:g}",
                    )
                    if np.any(yr > 0):
                        pos_plotted = True

        if col == "delta_E_vs_CASSCF":
            ax.axhline(0.0, color="black", linestyle=":", lw=1.6, label=r"Exacto $\Delta E = 0$")
        elif use_log and pos_plotted:
            ax.set_yscale("log")

        if len(n_values) > 0:
            ax.set_xticks([int(n) for n in n_values])
        ax.set_ylabel(ylabel)
        ax.set_title(title, weight="bold")
        ax.grid(True, which="both", alpha=0.32)

    for ax in axes[1, :]:
        ax.set_xlabel("Número de Electrones $N$")

    _legend_if_nonempty(axes[0, 0], fontsize=7.8, ncol=2, loc="best")
    _legend_if_nonempty(axes[1, 2], fontsize=7.8, ncol=2, loc="best")

    fig.suptitle(
        "Evolución Global respecto al Número de Electrones N (Por Acoplamiento g): RL-MALA vs MALA",
        weight="bold",
    )
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_vs_N_evolution", args)


def plot_speedup_and_ess(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 3: Speedup G_RL y Eficiencia ESS/seg vs g y vs N (o P2 vs P3)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    p3_rl = df[df["pipeline_category"] == "P3: Coupled VMC + RL-MALA (SJB)"]
    p3_mala = df[df["pipeline_category"] == "P3: VMC + MALA (SJB)"]
    has_p3 = (not p3_rl.empty) or (not p3_mala.empty)

    n_values = sorted(df["N"].dropna().unique())
    g_values = sorted(df["g"].dropna().unique())

    # Panel (0, 0): Speedup G_RL vs g (Pipeline 2)
    ax = axes[0, 0]
    for n in n_values:
        sub = p2_rl[p2_rl["N"] == n].sort_values("g")
        xr, yr, _ = _finite_xy(sub, "g", "G_RL_speedup")
        if xr.size > 0:
            ax.plot(
                xr, yr,
                color=N_COLORS.get(int(n), "black"),
                marker=N_MARKERS.get(int(n), "o"),
                linestyle="-",
                label=f"RL-MALA N = {int(n)}",
            )
    ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Baseline MALA (G=1.0)")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Factor de Aceleración $G_{\mathrm{RL}} = \tau_{\mathrm{int}}^{\mathrm{MALA}} / \tau_{\mathrm{int}}^{\mathrm{RL}}$")
    ax.set_title(r"Aceleración de Muestreo $G_{\mathrm{RL}}$ vs Acoplamiento $g$", weight="bold")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=9)

    # Panel (0, 1): Si hay P3 muestra P3 vs g; si solo hay P2 muestra G_RL vs N por cada g
    ax = axes[0, 1]
    if has_p3:
        for n in n_values:
            sub = p3_rl[p3_rl["N"] == n].sort_values("g")
            xr, yr, _ = _finite_xy(sub, "g", "G_RL_speedup")
            if xr.size > 0:
                ax.plot(
                    xr, yr,
                    color=N_COLORS.get(int(n), "black"),
                    marker=N_MARKERS.get(int(n), "o"),
                    linestyle="-",
                    label=f"N = {int(n)}",
                )
        ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Baseline MALA (G=1.0)")
        ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
        ax.set_ylabel(r"Factor de Aceleración $G_{\mathrm{RL}}$")
        ax.set_title("Pipeline 3: Aceleración Acoplada en Entrenamiento VMC (SJB)", weight="bold")
    else:
        for idx_g, g_val in enumerate(g_values):
            sub = p2_rl[np.isclose(p2_rl["g"], g_val)].sort_values("N")
            xr, yr, _ = _finite_xy(sub, "N", "G_RL_speedup")
            if xr.size > 0:
                ax.plot(
                    xr, yr,
                    color=G_CMAP_COLORS[idx_g % len(G_CMAP_COLORS)],
                    marker="o",
                    linestyle="-",
                    label=f"g = {g_val:g}",
                )
        ax.axhline(1.0, color="gray", linestyle="--", lw=1.5, label="Baseline MALA (G=1.0)")
        if len(n_values) > 0:
            ax.set_xticks([int(n) for n in n_values])
        ax.set_xlabel("Número de Electrones $N$")
        ax.set_ylabel(r"Factor de Aceleración $G_{\mathrm{RL}} = \tau_{\mathrm{int}}^{\mathrm{MALA}} / \tau_{\mathrm{int}}^{\mathrm{RL}}$")
        ax.set_title(r"Aceleración de Muestreo $G_{\mathrm{RL}}$ vs Tamaño del Sistema $N$", weight="bold")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel (1, 0): ESS/segundo en Pipeline 2 vs g
    ax = axes[1, 0]
    pos_ess_g = False
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, _ = _finite_xy(sub_m, "g", "ESS_per_sec")
        if xm.size > 0:
            ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.72, label=f"MALA N={int(n)}")
            if np.any(ym > 0):
                pos_ess_g = True
        xr, yr, _ = _finite_xy(sub_r, "g", "ESS_per_sec")
        if xr.size > 0:
            ax.plot(xr, yr, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"RL-MALA N={int(n)}")
            if np.any(yr > 0):
                pos_ess_g = True
    if pos_ess_g:
        ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel("Muestras Efectivas por Segundo (ESS/s)")
    ax.set_title("Rendimiento Absoluto ESS/s vs $g$ (RL-MALA vs MALA)", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel (1, 1): ESS/segundo vs N (o P3 si existe)
    ax = axes[1, 1]
    pos_ess_n = False
    if has_p3:
        for n in n_values:
            sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
            sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
            c = N_COLORS.get(int(n), "black")
            xm, ym, _ = _finite_xy(sub_m, "g", "ESS_per_sec")
            if xm.size > 0:
                ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.72)
                if np.any(ym > 0):
                    pos_ess_n = True
            xr, yr, _ = _finite_xy(sub_r, "g", "ESS_per_sec")
            if xr.size > 0:
                ax.plot(xr, yr, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"N = {int(n)}")
                if np.any(yr > 0):
                    pos_ess_n = True
        ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
        ax.set_title("Pipeline 3: Rendimiento Absoluto ESS/s vs $g$", weight="bold")
    else:
        for idx_g, g_val in enumerate(g_values):
            c = G_CMAP_COLORS[idx_g % len(G_CMAP_COLORS)]
            sub_m = p2_mala[np.isclose(p2_mala["g"], g_val)].sort_values("N")
            sub_r = p2_rl[np.isclose(p2_rl["g"], g_val)].sort_values("N")
            xm, ym, _ = _finite_xy(sub_m, "N", "ESS_per_sec")
            if xm.size > 0:
                ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.68)
                if np.any(ym > 0):
                    pos_ess_n = True
            xr, yr, _ = _finite_xy(sub_r, "N", "ESS_per_sec")
            if xr.size > 0:
                ax.plot(xr, yr, color=c, linestyle="-", marker="o", label=f"g = {g_val:g}")
                if np.any(yr > 0):
                    pos_ess_n = True
        if len(n_values) > 0:
            ax.set_xticks([int(n) for n in n_values])
        ax.set_xlabel("Número de Electrones $N$")
        ax.set_title("Rendimiento Absoluto ESS/s vs $N$ (RL-MALA vs MALA)", weight="bold")
    if pos_ess_n:
        ax.set_yscale("log")
    ax.set_ylabel("Muestras Efectivas por Segundo (ESS/s)")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    fig.suptitle("Comportamiento Global del Actor RL-MALA: Aceleración y Velocidad Efectiva ESS/s", weight="bold")
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_speedup_ess", args)


def plot_acceptance_and_esjd(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 4: Tasa de Aceptación MCMC y Distancia de Salto ESJD vs g y vs N."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    p3_rl = df[df["pipeline_category"] == "P3: Coupled VMC + RL-MALA (SJB)"]
    p3_mala = df[df["pipeline_category"] == "P3: VMC + MALA (SJB)"]
    has_p3 = (not p3_rl.empty) or (not p3_mala.empty)

    n_values = sorted(df["N"].dropna().unique())
    g_values = sorted(df["g"].dropna().unique())

    # Panel (0, 0): Tasa de Aceptación P2 vs g
    ax = axes[0, 0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, _ = _finite_xy(sub_m, "g", "acceptance")
        if xm.size > 0:
            ax.plot(xm, ym * 100.0, color=c, linestyle="--", marker="x", alpha=0.68, label=f"MALA N={int(n)}")
        xr, yr, _ = _finite_xy(sub_r, "g", "acceptance")
        if xr.size > 0:
            ax.plot(xr, yr * 100.0, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"RL-MALA N={int(n)}")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.set_title("Tasa de Aceptación MCMC vs $g$", weight="bold")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel (0, 1): Aceptación vs N (o P3 si existe)
    ax = axes[0, 1]
    if has_p3:
        for n in n_values:
            sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
            sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
            c = N_COLORS.get(int(n), "black")
            xm, ym, _ = _finite_xy(sub_m, "g", "acceptance")
            if xm.size > 0:
                ax.plot(xm, ym * 100.0, color=c, linestyle="--", marker="x", alpha=0.68)
            xr, yr, _ = _finite_xy(sub_r, "g", "acceptance")
            if xr.size > 0:
                ax.plot(xr, yr * 100.0, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"N = {int(n)}")
        ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
        ax.set_title("Pipeline 3: Tasa de Aceptación MCMC vs $g$", weight="bold")
    else:
        for idx_g, g_val in enumerate(g_values):
            c = G_CMAP_COLORS[idx_g % len(G_CMAP_COLORS)]
            sub_m = p2_mala[np.isclose(p2_mala["g"], g_val)].sort_values("N")
            sub_r = p2_rl[np.isclose(p2_rl["g"], g_val)].sort_values("N")
            xm, ym, _ = _finite_xy(sub_m, "N", "acceptance")
            if xm.size > 0:
                ax.plot(xm, ym * 100.0, color=c, linestyle="--", marker="x", alpha=0.68)
            xr, yr, _ = _finite_xy(sub_r, "N", "acceptance")
            if xr.size > 0:
                ax.plot(xr, yr * 100.0, color=c, linestyle="-", marker="o", label=f"g = {g_val:g}")
        if len(n_values) > 0:
            ax.set_xticks([int(n) for n in n_values])
        ax.set_xlabel("Número de Electrones $N$")
        ax.set_title("Tasa de Aceptación MCMC vs $N$", weight="bold")
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel (1, 0): ESJD en P2 vs g
    ax = axes[1, 0]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, _ = _finite_xy(sub_m, "g", "ESJD")
        if xm.size > 0:
            ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.68, label=f"MALA N={int(n)}")
        xr, yr, _ = _finite_xy(sub_r, "g", "ESJD")
        if xr.size > 0:
            ax.plot(xr, yr, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"RL-MALA N={int(n)}")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Distancia Cuadrática de Salto $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$)")
    ax.set_title("Movilidad Espacial ESJD vs $g$", weight="bold")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel (1, 1): ESJD vs N (o P3 si existe)
    ax = axes[1, 1]
    if has_p3:
        for n in n_values:
            sub_m = p3_mala[p3_mala["N"] == n].sort_values("g")
            sub_r = p3_rl[p3_rl["N"] == n].sort_values("g")
            c = N_COLORS.get(int(n), "black")
            xm, ym, _ = _finite_xy(sub_m, "g", "ESJD")
            if xm.size > 0:
                ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.68)
            xr, yr, _ = _finite_xy(sub_r, "g", "ESJD")
            if xr.size > 0:
                ax.plot(xr, yr, color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), label=f"N = {int(n)}")
        ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
        ax.set_title("Pipeline 3: Movilidad Espacial ESJD vs $g$", weight="bold")
    else:
        for idx_g, g_val in enumerate(g_values):
            c = G_CMAP_COLORS[idx_g % len(G_CMAP_COLORS)]
            sub_m = p2_mala[np.isclose(p2_mala["g"], g_val)].sort_values("N")
            sub_r = p2_rl[np.isclose(p2_rl["g"], g_val)].sort_values("N")
            xm, ym, _ = _finite_xy(sub_m, "N", "ESJD")
            if xm.size > 0:
                ax.plot(xm, ym, color=c, linestyle="--", marker="x", alpha=0.68)
            xr, yr, _ = _finite_xy(sub_r, "N", "ESJD")
            if xr.size > 0:
                ax.plot(xr, yr, color=c, linestyle="-", marker="o", label=f"g = {g_val:g}")
        if len(n_values) > 0:
            ax.set_xticks([int(n) for n in n_values])
        ax.set_xlabel("Número de Electrones $N$")
        ax.set_title("Movilidad Espacial ESJD vs $N$", weight="bold")
    ax.set_ylabel(r"Distancia Cuadrática de Salto $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$)")
    ax.grid(True)
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    fig.suptitle("Exploración del Espacio de Configuración: Aceptación y Movilidad Cuadrática ESJD", weight="bold")
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_acceptance_esjd", args)


def plot_fidelity_and_energy(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 5: Fidelidad respecto a CASSCF (D_KL^spin, D_KL^tot y Energía <E> +/- SE vs E_CASSCF)."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))

    p2_rl = df[df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)"]
    p2_mala = df[df["pipeline_category"] == "P2: MALA (CASSCF Fijo)"]
    n_values = sorted(df["N"].dropna().unique())

    # Panel 0: Divergencia KL de Espín (Cruce Nodal) vs g
    ax = axes[0]
    pos_spin = False
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, yerrm = _finite_xy(sub_m, "g", "kl_casscf_spin_mean", "kl_casscf_spin_std")
        if xm.size > 0:
            ax.errorbar(
                xm, ym, yerr=yerrm,
                color=c, linestyle="--", marker="x", capsize=3, alpha=0.72, label=f"MALA N={int(n)}"
            )
            if np.any(ym > 0):
                pos_spin = True
        xr, yr, yerrr = _finite_xy(sub_r, "g", "kl_casscf_spin_mean", "kl_casscf_spin_std")
        if xr.size > 0:
            ax.errorbar(
                xr, yr, yerr=yerrr,
                color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), capsize=3, label=f"RL-MALA N={int(n)}"
            )
            if np.any(yr > 0):
                pos_spin = True
    if pos_spin:
        ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Divergencia KL de Espín $D_{\mathrm{KL}}^{\mathrm{spin}}$")
    ax.set_title("Cruce Nodal y Simetría de Espín vs CASSCF (Menor es Mejor)", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    # Panel 1: Divergencia KL Total respecto a CASSCF
    ax = axes[1]
    pos_tot = False
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, yerrm = _finite_xy(sub_m, "g", "kl_casscf_total_mean", "kl_casscf_total_std")
        if xm.size > 0:
            ax.errorbar(
                xm, ym, yerr=yerrm,
                color=c, linestyle="--", marker="x", capsize=3, alpha=0.72,
            )
            if np.any(ym > 0):
                pos_tot = True
        xr, yr, yerrr = _finite_xy(sub_r, "g", "kl_casscf_total_mean", "kl_casscf_total_std")
        if xr.size > 0:
            ax.errorbar(
                xr, yr, yerr=yerrr,
                color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), capsize=3, label=f"N = {int(n)}"
            )
            if np.any(yr > 0):
                pos_tot = True
    if pos_tot:
        ax.set_yscale("log")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"Divergencia KL Total $D_{\mathrm{KL}}^{\mathrm{tot}}(\rho \parallel \rho_{\mathrm{CASSCF}})$")
    ax.set_title("Fidelidad de la Densidad Total vs CASSCF (Menor es Mejor)", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=9)

    # Panel 2: Energía MCMC <E> +/- SE(E) y Desviación Delta E respecto a CASSCF
    ax = axes[2]
    for n in n_values:
        sub_m = p2_mala[p2_mala["N"] == n].sort_values("g")
        sub_r = p2_rl[p2_rl["N"] == n].sort_values("g")
        c = N_COLORS.get(int(n), "black")
        xm, ym, yerrm = _finite_xy(sub_m, "g", "delta_E_vs_CASSCF", "SE_E")
        if xm.size > 0:
            ax.errorbar(
                xm, ym * 1e3,
                yerr=(yerrm * 1e3 if yerrm is not None else None),
                color=c, linestyle="--", marker="x", capsize=3, alpha=0.72, label=f"MALA N={int(n)}"
            )
        xr, yr, yerrr = _finite_xy(sub_r, "g", "delta_E_vs_CASSCF", "SE_E")
        if xr.size > 0:
            ax.errorbar(
                xr, yr * 1e3,
                yerr=(yerrr * 1e3 if yerrr is not None else None),
                color=c, linestyle="-", marker=N_MARKERS.get(int(n), "o"), capsize=3, label=f"RL-MALA N={int(n)}"
            )
    ax.axhline(0.0, color="black", linestyle=":", lw=1.6, label=r"Referencia $E_{\mathrm{CASSCF}}$ ($\Delta E = 0$)")
    ax.set_xlabel("Fuerza de Interacción $g$ (Hartree)")
    ax.set_ylabel(r"$\Delta E = \langle E \rangle - E_{\mathrm{CASSCF}} \pm \mathrm{SE}(E)$ (mHa)")
    ax.set_title("Precisión de Energía Local con Barras de Error vs CASSCF", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=8.5, ncol=2)

    fig.suptitle("Fidelidad Física, Cruce Nodal y Convergencia de Energía respecto a CASSCF", weight="bold")
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_fidelity_kl_energy", args)


def plot_executive_dashboard(df: pd.DataFrame, out_dir: str, args: argparse.Namespace) -> None:
    """Figura 6: Dashboard Ejecutivo Comparativo con remarcado de mejores modelos."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    candidate_methods = [
        ("P2: MALA (CASSCF Fijo)", "P2 MALA"),
        ("P2: RL-MALA TD3 (CASSCF Fijo)", "P2 RL-MALA"),
        ("P3: VMC + MALA (SJB)", "P3 VMC+MALA"),
        ("P3: Coupled VMC + RL-MALA (SJB)", "P3 Coupled RL"),
    ]

    # Panel 0: Distribución de Tiempos de Autocorrelación tau_int por Método
    ax = axes[0, 0]
    if "tau_int_E" in df.columns:
        active_tau = []
        for m, lbl in candidate_methods:
            s = pd.to_numeric(df[df["pipeline_category"] == m]["tau_int_E"], errors="coerce").dropna()
            s = s[np.isfinite(s)]
            if not s.empty:
                active_tau.append((lbl, s))
        if active_tau:
            labels_tau = [lbl for lbl, _ in active_tau]
            data_tau = [s for _, s in active_tau]
            ax.boxplot(
                data_tau, tick_labels=labels_tau, patch_artist=True,
                boxprops=dict(facecolor="#cce5ff", color="#004085"),
                medianprops=dict(color="red", lw=2),
            )
            if any((s > 0).any() for s in data_tau):
                ax.set_yscale("log")
    ax.set_ylabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ (pasos)")
    ax.set_title("Distribución de Autocorrelación por Método (Menor es Mejor)", weight="bold")
    ax.grid(True, which="both")

    # Panel 1: Distribución de D_KL^spin (Cruce Nodal) por Método (filtrado independiente)
    ax = axes[0, 1]
    if "kl_casscf_spin_mean" in df.columns:
        active_kl = []
        for m, lbl in candidate_methods:
            s = pd.to_numeric(df[df["pipeline_category"] == m]["kl_casscf_spin_mean"], errors="coerce").dropna()
            s = s[np.isfinite(s)]
            if not s.empty:
                active_kl.append((lbl, s))
        if active_kl:
            labels_kl = [lbl for lbl, _ in active_kl]
            data_kl = [s for _, s in active_kl]
            ax.boxplot(
                data_kl, tick_labels=labels_kl, patch_artist=True,
                boxprops=dict(facecolor="#d4edda", color="#155724"),
                medianprops=dict(color="darkgreen", lw=2),
            )
            if any((s > 0).any() for s in data_kl):
                ax.set_yscale("log")
    ax.set_ylabel(r"Divergencia KL de Espín $D_{\mathrm{KL}}^{\mathrm{spin}}$")
    ax.set_title(r"Cruce Nodal: Distribución de $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Menor es Mejor)", weight="bold")
    ax.grid(True, which="both")

    # Panel 2: ESS/s promedio por Sistema N
    ax = axes[1, 0]
    n_values = sorted(df["N"].dropna().unique())
    x_pos = np.arange(len(n_values))
    width = 0.36

    if "ESS_per_sec" in df.columns and len(n_values) > 0:
        p2_m_means = [
            pd.to_numeric(
                df[(df["pipeline_category"] == "P2: MALA (CASSCF Fijo)") & (df["N"] == n)]["ESS_per_sec"],
                errors="coerce",
            ).mean()
            for n in n_values
        ]
        p2_r_means = [
            pd.to_numeric(
                df[(df["pipeline_category"] == "P2: RL-MALA TD3 (CASSCF Fijo)") & (df["N"] == n)]["ESS_per_sec"],
                errors="coerce",
            ).mean()
            for n in n_values
        ]
        m_pairs = [(float(x) - width / 2, float(v)) for x, v in zip(x_pos, p2_m_means) if np.isfinite(v)]
        r_pairs = [(float(x) + width / 2, float(v)) for x, v in zip(x_pos, p2_r_means) if np.isfinite(v)]
        if m_pairs:
            ax.bar(
                [p[0] for p in m_pairs], [p[1] for p in m_pairs], width,
                label="MALA (CASSCF Fijo)", color="#7f8c8d", edgecolor="black", linestyle="--", alpha=0.8,
            )
        if r_pairs:
            ax.bar(
                [p[0] for p in r_pairs], [p[1] for p in r_pairs], width,
                label="RL-MALA TD3 (CASSCF Fijo)", color="#c0392b", edgecolor="black", alpha=0.9,
            )
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"N = {int(n)}" for n in n_values])
        if any(p[1] > 0 for p in m_pairs + r_pairs):
            ax.set_yscale("log")
    ax.set_ylabel("ESS/segundo Promedio")
    ax.set_title("Velocidad Efectiva Promedio vs Tamaño del Sistema $N$", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax)

    # Panel 3: Frente de Pareto: Cruce Nodal D_KL^spin vs Autocorrelación tau_int(E)
    ax = axes[1, 1]
    y_col = None
    if "kl_casscf_spin_mean" in df.columns and df["kl_casscf_spin_mean"].notna().any():
        y_col = "kl_casscf_spin_mean"
    elif "delta_E_vs_CASSCF" in df.columns and df["delta_E_vs_CASSCF"].notna().any():
        y_col = "delta_E_vs_CASSCF"

    pos_x = False
    pos_y = False
    if y_col is not None and "tau_int_E" in df.columns:
        for cat, color, marker in [
            ("P2: MALA (CASSCF Fijo)", "#7f8c8d", "o"),
            ("P2: RL-MALA TD3 (CASSCF Fijo)", "#c0392b", "*"),
            ("P3: VMC + MALA (SJB)", "#2980b9", "s"),
            ("P3: Coupled VMC + RL-MALA (SJB)", "#8e44ad", "^"),
        ]:
            sub = df[df["pipeline_category"] == cat]
            if not sub.empty:
                x_vals = pd.to_numeric(sub["tau_int_E"], errors="coerce").to_numpy(dtype=float)
                y_vals = np.abs(pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float))
                mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                if np.any(mask):
                    xp, yp = x_vals[mask], y_vals[mask]
                    ax.scatter(
                        xp, yp,
                        color=color, marker=marker, s=95 if marker == "*" else 65,
                        alpha=0.85, edgecolors="black", linewidths=0.5, label=cat,
                    )
                    if np.any(xp > 0):
                        pos_x = True
                    if np.any(yp > 0):
                        pos_y = True
    if pos_x:
        ax.set_xscale("log")
    if pos_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ (Menor es Mejor)")
    if y_col == "kl_casscf_spin_mean":
        ax.set_ylabel(r"Divergencia KL de Espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Menor es Mejor)")
        ax.set_title("Frente de Pareto: Cruce Nodal vs Autocorrelación MCMC", weight="bold")
    else:
        ax.set_ylabel(r"Error Absoluto de Energía $|E - E_{\mathrm{CASSCF}}|$ (Ha)")
        ax.set_title("Frente de Pareto: Precisión de Energía vs Rapidez MCMC", weight="bold")
    ax.grid(True, which="both")
    _legend_if_nonempty(ax, fontsize=8.5)

    fig.suptitle("Dashboard Ejecutivo: Evaluación Panorámica de los Modelos del Benchmark", weight="bold")
    fig.tight_layout()
    _save_fig_custom(fig, out_dir, "fig_all_models_executive_dashboard", args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera figuras panorámicas de todos los modelos del benchmark respecto a N y g.")
    parser.add_argument("--csv", type=str, default="results_csv_b2/live_all_models_master.csv", help="Ruta al CSV maestro.")
    parser.add_argument("--out_dir", type=str, default="figures/benchmark_summary", help="Directorio donde guardar las figuras.")
    parser.add_argument(
        "--figure_format",
        "--format",
        dest="figure_format",
        type=str,
        default="png",
        choices=["png", "pdf", "both"],
        help="Formato de exportación gráfica ('png' por defecto para ahorrar espacio).",
    )
    parser.add_argument(
        "--dpi",
        "--ppi",
        dest="dpi",
        type=int,
        default=300,
        help="Resolución en píxeles/puntos por pulgada (PPI / DPI).",
    )
    parser.add_argument("--title_prefix", type=str, default="", help="Prefijo de título.")
    parser.add_argument("--custom_suptitle", type=str, default="", help="Suptitle personalizado.")
    parser.add_argument(
        "--no_titles",
        "--hide_titles",
        dest="no_titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ocultar títulos para LaTeX.",
    )
    parser.add_argument("--font_scale", type=float, default=1.0, help="Escala de fuente.")
    parser.add_argument("--line_width_scale", type=float, default=1.0, help="Escala de grosor de línea.")
    parser.add_argument(
        "--transparent_bg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fondo transparente.",
    )
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 11 * args.font_scale,
        "axes.labelsize": 12 * args.font_scale,
        "axes.titlesize": 12 * args.font_scale,
        "xtick.labelsize": 10 * args.font_scale,
        "ytick.labelsize": 10 * args.font_scale,
        "legend.fontsize": 9.5 * args.font_scale,
        "figure.titlesize": 14 * args.font_scale,
        "lines.linewidth": 2.0 * args.line_width_scale,
        "lines.markersize": 6.5,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
    })

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_and_clean_data(args.csv)

    print("=" * 80)
    print("    GENERACIÓN DE FIGURAS GLOBALES DE TODOS LOS MODELOS (VS N Y G)")
    print(f"    Archivo Entrada   : {args.csv}")
    print(f"    Directorio Salida : {args.out_dir}")
    print(f"    Formato / PPI     : {args.figure_format} @ {args.dpi} PPI")
    print(f"    Total Modelos     : {len(df)}")
    print("=" * 80)

    plot_evolution_vs_g(df, args.out_dir, args)
    plot_evolution_vs_N(df, args.out_dir, args)
    plot_speedup_and_ess(df, args.out_dir, args)
    plot_acceptance_and_esjd(df, args.out_dir, args)
    plot_fidelity_and_energy(df, args.out_dir, args)
    plot_executive_dashboard(df, args.out_dir, args)

    print("=" * 80)
    print("    FIGURAS GLOBALES ACTUALIZADAS EXITOSAMENTE")
    print("=" * 80)


if __name__ == "__main__":
    main()
