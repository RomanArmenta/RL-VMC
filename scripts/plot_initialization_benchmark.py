#!/usr/bin/env python3
"""
scripts/plot_initialization_benchmark.py
==============================================================================
Generador de figuras y tablas estadísticas con barras de error para el
Benchmark de Estrategias de Inicialización de Caminantes (MALA vs RL-MALA).

Estrategias de Inicialización analizadas:
  1) alternating_spin   : Un electrón en cada Gaussiana (con jitter cuántico)
  2) symmetric_gaussian : Gaussiana centrada en x=0 con sigma que cubre la envolvente
  3) uniform            : Distribución uniforme en caja [-L, L] que encierra la nube

Estadística:
  - Agrega sobre n_exp réplicas (diferentes seeds) y n_eval cadenas MCMC.
  - Calcula la media (mean) y el error estándar de la media (SEM = std / sqrt(n)).
  - Produce barras de error en todas las métricas de eficiencia, autocorrelación,
    sesgo energético y ergodocidad nodal.
==============================================================================
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Colores y estilo de publicación
COLOR_MALA = "#D95F02"      # Naranja rojizo para MALA clásico
COLOR_RL_MALA = "#1B9E77"   # Verde azulado para RL-MALA
COLOR_CASSCF = "#000000"    # Negro para referencia exacta

INIT_LABELS = {
    "alternating_spin": "1 e⁻ por Gaussiana\n(Alternating Spin)",
    "symmetric_gaussian": "Gaussiana x=0\n(Envolvente)",
    "uniform": "Uniforme\n(Caja [-L, L])",
}

INIT_ORDER = ["alternating_spin", "symmetric_gaussian", "uniform"]


def setup_matplotlib_style(font_scale: float = 1.0, line_width_scale: float = 1.0) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "lines.linewidth": 1.75 * line_width_scale,
        "lines.markersize": 6 * font_scale,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.autolayout": True,
    })


def parse_run_info(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Lee el archivo summary.json o production_metrics.csv de una corrida individual."""
    summary_path = run_dir / "summary.json"
    data = {}
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Aviso] No se pudo leer {summary_path}: {e}")

    # Fallback o lectura complementaria desde CSV
    csv_candidates = list(run_dir.glob("production_metrics_*.csv"))
    csv_row = {}
    if csv_candidates:
        try:
            df = pd.read_csv(csv_candidates[0])
            if not df.empty:
                csv_row = df.iloc[0].to_dict()
        except Exception:
            pass

    if not data and not csv_row:
        return None

    # Extraer metadatos
    obs = data.get("observables", {})
    cass = data.get("casscf_comparison", {})
    train_h = data.get("training_hyperparameters", {})
    samp_cfg = train_h.get("sampler", {})
    eval_cfg = data.get("evaluation_config", {})
    prop_cfg = data.get("model_hyperparameters", {}).get("proposal", {})
    sys_cfg = data.get("system", {})

    init_mode = (
        samp_cfg.get("init_mode")
        or csv_row.get("init_mode")
        or eval_cfg.get("eval_init_mode")
        or "unknown"
    )
    if init_mode == "same_as_sampler":
        init_mode = samp_cfg.get("init_mode", "unknown")

    # Si aún es unknown, inferir del nombre de la carpeta
    dirname = run_dir.name
    if "alt_spin" in dirname or "alternating_spin" in dirname:
        init_mode = "alternating_spin"
    elif "sym_gauss" in dirname or "symmetric_gaussian" in dirname:
        init_mode = "symmetric_gaussian"
    elif "uniform" in dirname:
        init_mode = "uniform"

    method = (
        csv_row.get("proposal_type")
        or prop_cfg.get("proposal_type")
        or ("TD3" if "RL_MALA" in dirname or "TD3" in dirname else "MALA")
    )
    if method in ("neural_deepset", "TD3", "td3"):
        model_type = "RL_MALA"
    else:
        model_type = "MALA"

    n_elec = int(sys_cfg.get("n_elec") or csv_row.get("N") or (4 if "N4" in dirname else 8))
    g_val = float(sys_cfg.get("interaction_strength") or csv_row.get("g") or 1.0)
    omega = float(sys_cfg.get("omega") or csv_row.get("omega") or 0.1)
    seed = int(train_h.get("seed") or csv_row.get("seed") or 0)

    e_mean = float(obs.get("E_mean", csv_row.get("E_mean", float("nan"))))
    e_var = float(obs.get("E_var", csv_row.get("E_var", float("nan"))))
    se_e = float(obs.get("SE_E", csv_row.get("SE_E", float("nan"))))
    tau_e = float(obs.get("tau_int_E", csv_row.get("tau_int_E", float("nan"))))
    tau_pos = float(obs.get("tau_int_pos", csv_row.get("tau_int_pos", float("nan"))))
    ess = float(obs.get("ESS", csv_row.get("ESS", float("nan"))))
    ess_s = float(obs.get("ESS_per_sec", csv_row.get("ESS_per_sec", float("nan"))))
    acc = float(obs.get("acceptance", csv_row.get("acceptance", float("nan"))))
    esjd = float(obs.get("ESJD", csv_row.get("ESJD", float("nan"))))
    wall_time = float(obs.get("wall_time_sec", csv_row.get("wall_time_sec", float("nan"))))

    cas_e = float(cass.get("casscf_energy", csv_row.get("casscf_energy", float("nan"))))
    delta_e = float(cass.get("delta_E_vs_CASSCF", csv_row.get("delta_E_vs_CASSCF", float("nan"))))
    kl_tot = float(cass.get("kl_casscf_total_mean", csv_row.get("kl_casscf_total_mean", float("nan"))))
    kl_up = float(cass.get("kl_casscf_up_mean", csv_row.get("kl_casscf_up_mean", float("nan"))))
    kl_dn = float(cass.get("kl_casscf_down_mean", csv_row.get("kl_casscf_down_mean", float("nan"))))
    kl_spin = 0.5 * (kl_up + kl_dn) if (np.isfinite(kl_up) and np.isfinite(kl_dn)) else float("nan")

    return {
        "run_dir": str(run_dir),
        "run_name": dirname,
        "n_elec": n_elec,
        "g": g_val,
        "omega": omega,
        "system_tag": f"N{n_elec}_g{g_val}",
        "init_mode": init_mode,
        "model_type": model_type,
        "seed": seed,
        "E_mean": e_mean,
        "SE_E": se_e,
        "E_var": e_var,
        "tau_int_E": tau_e,
        "tau_int_pos": tau_pos,
        "ESS": ess,
        "ESS_per_sec": ess_s,
        "acceptance": acc,
        "ESJD": esjd,
        "wall_time_sec": wall_time,
        "casscf_energy": cas_e,
        "delta_E_vs_CASSCF": delta_e,
        "kl_total": kl_tot,
        "kl_spin": kl_spin,
    }


def collect_benchmark_data(root_dir: Path) -> pd.DataFrame:
    """Recolecta métricas de todas las subcarpetas de corridas."""
    records = []
    # Buscar carpetas que contengan summary.json o production_metrics_*.csv
    for sfile in root_dir.glob("**/summary.json"):
        info = parse_run_info(sfile.parent)
        if info is not None:
            records.append(info)

    # Si no se encontraron por summary.json, buscar por csv
    if not records:
        for cfile in root_dir.glob("**/production_metrics_*.csv"):
            info = parse_run_info(cfile.parent)
            if info is not None:
                records.append(info)

    # También verificar si existe un master_csv en root_dir
    master_csvs = list(root_dir.glob("*master*.csv"))
    if not records and master_csvs:
        try:
            mdf = pd.read_csv(master_csvs[0])
            for _, r in mdf.iterrows():
                row_d = r.to_dict()
                dirname = str(row_d.get("run_name", ""))
                init_m = "unknown"
                if "alt_spin" in dirname or "alternating_spin" in dirname:
                    init_m = "alternating_spin"
                elif "sym_gauss" in dirname or "symmetric_gaussian" in dirname:
                    init_m = "symmetric_gaussian"
                elif "uniform" in dirname:
                    init_m = "uniform"

                method = str(row_d.get("proposal_type", ""))
                m_type = "RL_MALA" if ("TD3" in dirname or "RL_MALA" in dirname or method == "neural_deepset") else "MALA"
                n_el = int(row_d.get("N", 4 if "N4" in dirname else 8))
                g_v = float(row_d.get("g", 1.0))

                records.append({
                    "run_dir": str(row_d.get("output_h5", "")),
                    "run_name": dirname,
                    "n_elec": n_el,
                    "g": g_v,
                    "omega": float(row_d.get("omega", 0.1)),
                    "system_tag": f"N{n_el}_g{g_v}",
                    "init_mode": init_m,
                    "model_type": m_type,
                    "seed": int(row_d.get("seed", 0)),
                    "E_mean": float(row_d.get("E_mean", float("nan"))),
                    "SE_E": float(row_d.get("SE_E", float("nan"))),
                    "E_var": float(row_d.get("E_var", float("nan"))),
                    "tau_int_E": float(row_d.get("tau_int_E", float("nan"))),
                    "tau_int_pos": float(row_d.get("tau_int_pos", float("nan"))),
                    "ESS": float(row_d.get("ESS", float("nan"))),
                    "ESS_per_sec": float(row_d.get("ESS_per_sec", float("nan"))),
                    "acceptance": float(row_d.get("acceptance", float("nan"))),
                    "ESJD": float(row_d.get("ESJD", float("nan"))),
                    "wall_time_sec": float(row_d.get("wall_time_sec", float("nan"))),
                    "casscf_energy": float(row_d.get("casscf_energy", float("nan"))),
                    "delta_E_vs_CASSCF": float(row_d.get("delta_E_vs_CASSCF", float("nan"))),
                    "kl_total": float(row_d.get("kl_casscf_total_mean", float("nan"))),
                    "kl_spin": float(row_d.get("kl_casscf_spin_mean", float("nan"))),
                })
        except Exception as e:
            print(f"[Aviso] No se pudo leer master_csv: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        # Deduplicar por run_name
        df = df.drop_duplicates(subset=["run_name"]).reset_index(drop=True)
    return df


def aggregate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas sobre seeds/réplicas para cada (system_tag, model_type, init_mode)."""
    metrics = [
        "E_mean", "SE_E", "E_var", "tau_int_E", "tau_int_pos",
        "ESS", "ESS_per_sec", "acceptance", "ESJD", "wall_time_sec",
        "delta_E_vs_CASSCF", "kl_total", "kl_spin"
    ]

    grouped = df.groupby(["system_tag", "n_elec", "model_type", "init_mode"])
    agg_rows = []

    for (sys_tag, n_el, model_t, init_m), grp in grouped:
        n_samples = len(grp)
        row = {
            "system_tag": sys_tag,
            "n_elec": n_el,
            "model_type": model_t,
            "init_mode": init_m,
            "n_exp_runs": n_samples,
        }

        for m in metrics:
            vals = grp[m].dropna().values
            if len(vals) > 0:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{m}_mean"] = mean_v
                row[f"{m}_std"] = std_v
                row[f"{m}_sem"] = sem_v
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_sem"] = float("nan")

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def plot_system_initialization_comparison(
    agg_df: pd.DataFrame,
    system_tag: str,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera el panel principal de 6 subplots comparando MALA vs RL-MALA en las
    3 inicializaciones con barras de error rigurosas (SEM).
    """
    sub_df = agg_df[agg_df["system_tag"] == system_tag].copy()
    if sub_df.empty:
        print(f"[Aviso] No hay datos agregados para el sistema {system_tag}")
        return

    init_keys = [k for k in INIT_ORDER if k in sub_df["init_mode"].unique()]
    if not init_keys:
        init_keys = sorted(sub_df["init_mode"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    ax_ess = axes[0, 0]
    ax_tau = axes[0, 1]
    ax_de  = axes[0, 2]
    ax_acc = axes[1, 0]
    ax_esjd= axes[1, 1]
    ax_kl  = axes[1, 2]

    x_idx = np.arange(len(init_keys))
    bar_width = 0.35

    models = ["MALA", "RL_MALA"]
    colors = {"MALA": COLOR_MALA, "RL_MALA": COLOR_RL_MALA}
    labels = {"MALA": "MALA Clásico", "RL_MALA": "RL-MALA (TD3)"}

    for m_idx, m_name in enumerate(models):
        offset = (m_idx - 0.5) * bar_width
        m_sub = sub_df[sub_df["model_type"] == m_name].set_index("init_mode")

        def get_series(col_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
            means, errs = [], []
            for k in init_keys:
                if k in m_sub.index:
                    m_val = m_sub.loc[k, f"{col_prefix}_mean"]
                    e_val = m_sub.loc[k, f"{col_prefix}_sem"]
                    means.append(m_val if np.isfinite(m_val) else 0.0)
                    errs.append(e_val if np.isfinite(e_val) else 0.0)
                else:
                    means.append(0.0)
                    errs.append(0.0)
            return np.array(means), np.array(errs)

        # 1. ESS / sec
        ess_m, ess_e = get_series("ESS_per_sec")
        ax_ess.bar(
            x_idx + offset, ess_m, bar_width, yerr=ess_e, capsize=4,
            color=colors[m_name], label=labels[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 2. tau_int(E)
        tau_m, tau_e = get_series("tau_int_E")
        ax_tau.bar(
            x_idx + offset, tau_m, bar_width, yerr=tau_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 3. |Delta E| vs CASSCF (en mHa = 1e-3 Ha)
        de_m, de_e = get_series("delta_E_vs_CASSCF")
        # Mostrar magnitud del sesgo en mHa
        de_m_mha = np.abs(de_m) * 1000.0
        de_e_mha = de_e * 1000.0
        ax_de.bar(
            x_idx + offset, de_m_mha, bar_width, yerr=de_e_mha, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 4. Acceptance (%)
        acc_m, acc_e = get_series("acceptance")
        ax_acc.bar(
            x_idx + offset, acc_m * 100.0, bar_width, yerr=acc_e * 100.0, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 5. ESJD
        esjd_m, esjd_e = get_series("ESJD")
        ax_esjd.bar(
            x_idx + offset, esjd_m, bar_width, yerr=esjd_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 6. Spin KL Divergence (D_KL^spin)
        kl_m, kl_e = get_series("kl_spin")
        # En caso de que kl_spin sea 0 o NaN, intentar con kl_total
        if np.all(kl_m == 0.0):
            kl_m, kl_e = get_series("kl_total")
        ax_kl.bar(
            x_idx + offset, kl_m, bar_width, yerr=kl_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

    # Configuración de ejes y etiquetas
    x_tick_labels = [INIT_LABELS.get(k, k) for k in init_keys]
    for ax in axes[1, :]:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_tick_labels, rotation=0, ha="center")

    ax_ess.set_title("(a) Eficiencia de Muestreo (ESS/s)")
    ax_ess.set_ylabel("Muestras Efectivas / s (mayor es mejor)")
    ax_ess.legend(loc="upper right", framealpha=0.9)

    ax_tau.set_title("(b) Tiempo Autocorrelación $\\tau_{\\mathrm{int}}(E)$")
    ax_tau.set_ylabel("Pasos MCMC (menor es mejor)")

    ax_de.set_title("(c) Sesgo Energético $|\\Delta E|$ vs CASSCF")
    ax_de.set_ylabel("mHa (menor es mejor)")

    ax_acc.set_title("(d) Tasa de Aceptación M-H")
    ax_acc.set_ylabel("Aceptación (%)")
    ax_acc.set_ylim(0, 105)

    ax_esjd.set_title("(e) Distancia Cuadrática de Salto (ESJD)")
    ax_esjd.set_ylabel("$\\mathrm{Bohr}^2$ (mayor es mejor)")

    ax_kl.set_title("(f) Divergencia $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ (Cruce Nodal)")
    ax_kl.set_ylabel("Divergencia KL (menor es mejor)")
    ax_kl.set_yscale("log")

    plt.suptitle(
        f"Benchmark de Inicialización: MALA vs RL-MALA en {system_tag} (Barras de Error: SEM sobre Réplicas)",
        fontsize=14, fontweight="bold", y=0.99
    )

    out_file = out_dir / f"benchmark_initialization_{system_tag}.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráfica] Guardada figura comparativa en: {out_file}")


def plot_speedup_scaling_summary(
    agg_df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera figura comparativa resumen mostrando la ganancia de velocidad (Speedup G_RL = ESS_RL / ESS_MALA)
    y reducción de autocorrelación a través de todos los sistemas e inicializaciones.
    """
    systems = sorted(agg_df["system_tag"].unique())
    if not systems:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bar_width = 0.25

    for s_idx, sys_tag in enumerate(systems):
        s_df = agg_df[agg_df["system_tag"] == sys_tag]
        init_keys = [k for k in INIT_ORDER if k in s_df["init_mode"].unique()]
        x_base = np.arange(len(init_keys))
        offset = (s_idx - (len(systems) - 1) / 2.0) * bar_width

        speedups = []
        speedup_errs = []
        tau_reductions = []

        for k in init_keys:
            m_sub = s_df[(s_df["model_type"] == "MALA") & (s_df["init_mode"] == k)]
            r_sub = s_df[(s_df["model_type"] == "RL_MALA") & (s_df["init_mode"] == k)]

            if not m_sub.empty and not r_sub.empty:
                ess_m = float(m_sub["ESS_per_sec_mean"].iloc[0])
                ess_r = float(r_sub["ESS_per_sec_mean"].iloc[0])
                sem_m = float(m_sub["ESS_per_sec_sem"].iloc[0])
                sem_r = float(r_sub["ESS_per_sec_sem"].iloc[0])

                if ess_m > 0 and np.isfinite(ess_r):
                    sp = ess_r / ess_m
                    # Propagación de error para razón sp = r / m
                    sp_err = sp * np.sqrt((sem_r / max(ess_r, 1e-9))**2 + (sem_m / max(ess_m, 1e-9))**2)
                    speedups.append(sp)
                    speedup_errs.append(sp_err)
                else:
                    speedups.append(1.0)
                    speedup_errs.append(0.0)

                tau_m = float(m_sub["tau_int_E_mean"].iloc[0])
                tau_r = float(r_sub["tau_int_E_mean"].iloc[0])
                if tau_r > 0 and np.isfinite(tau_m):
                    tau_reductions.append(tau_m / tau_r)
                else:
                    tau_reductions.append(1.0)
            else:
                speedups.append(0.0)
                speedup_errs.append(0.0)
                tau_reductions.append(0.0)

        # Plot en Panel 1: Factor de Aceleración G_RL
        ax1.bar(
            x_base + offset, speedups, bar_width, yerr=speedup_errs, capsize=4,
            label=f"Sistema {sys_tag}", alpha=0.85, edgecolor="black", linewidth=0.8
        )

        # Plot en Panel 2: Reducción de Autocorrelación
        ax2.bar(
            x_base + offset, tau_reductions, bar_width, capsize=4,
            label=f"Sistema {sys_tag}", alpha=0.85, edgecolor="black", linewidth=0.8
        )

    init_labels = [INIT_LABELS.get(k, k) for k in INIT_ORDER if any(k in agg_df["init_mode"].values for _ in [1])]
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(init_labels)))
        ax.set_xticklabels(init_labels)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Paridad MALA (1.0×)")

    ax1.set_title("Factor de Aceleración Neta $G_{\\mathrm{RL}} = \\mathrm{ESS/s}_{\\mathrm{RL}} / \\mathrm{ESS/s}_{\\mathrm{MALA}}$")
    ax1.set_ylabel("Aceleración Relativa (veces mayor)")
    ax1.legend(loc="upper left")

    ax2.set_title("Factor de Descorrelación $\\tau_{\\mathrm{int}}(E)_{\\mathrm{MALA}} / \\tau_{\\mathrm{int}}(E)_{\\mathrm{RL}}$")
    ax2.set_ylabel("Reducción de Autocorrelación (veces menor)")
    ax2.legend(loc="upper left")

    plt.suptitle("Resumen Global del Impacto de la Inicialización: Resiliencia y Aceleración de RL-MALA", fontsize=13, fontweight="bold")
    out_file = out_dir / f"benchmark_initialization_summary_speedups.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráfica] Guardada figura resumen en: {out_file}")


def export_markdown_and_csv_reports(agg_df: pd.DataFrame, out_dir: Path) -> None:
    """Genera tabla ejecutiva en CSV y en Markdown."""
    csv_file = out_dir / "initialization_benchmark_summary_table.csv"
    agg_df.to_csv(csv_file, index=False)
    print(f"[Tabla] Tabla CSV agregada exportada en: {csv_file}")

    md_file = out_dir / "INITIALIZATION_BENCHMARK_REPORT.md"
    lines = [
        "# Reporte Estadístico: Benchmark de Estrategias de Inicialización (MALA vs RL-MALA)",
        "",
        "Este reporte sintetiza los resultados experimentales evaluando la sensibilidad, estabilidad y aceleración",
        "de **MALA clásico** frente a **RL-MALA (TD3)** bajo tres condiciones de frontera iniciales:",
        "",
        "1. **`alternating_spin`**: Un electrón en cada Gaussiana (red cristalina con jitter térmico local).",
        "2. **`symmetric_gaussian`**: Distribución normal isotrópica $\\mathcal{N}(0, \\sigma^2)$ cubriendo la envolvente completa.",
        "3. **`uniform`**: Distribución uniforme $\\mathcal{U}(-L, L)$ sobre la caja del sistema confinante.",
        "",
        "---",
        "",
        "## Tabla Ejecutiva de Resultados (Media ± Error Estándar de la Media)",
        "",
        "| Sistema | Modelo | Modo Inicialización | Réplicas | $\\Delta E$ vs CASSCF (mHa) | $\\tau_{\\mathrm{int}}(E)$ | Aceptación (%) | $\\mathrm{ESS/s}$ | $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in agg_df.iterrows():
        sys_t = row["system_tag"]
        mod_t = row["model_type"]
        init_m = row["init_mode"]
        n_exp = int(row["n_exp_runs"])

        de_val = row.get("delta_E_vs_CASSCF_mean", float("nan")) * 1000.0
        de_sem = row.get("delta_E_vs_CASSCF_sem", 0.0) * 1000.0

        tau_val = row.get("tau_int_E_mean", float("nan"))
        tau_sem = row.get("tau_int_E_sem", 0.0)

        acc_val = row.get("acceptance_mean", float("nan")) * 100.0
        acc_sem = row.get("acceptance_sem", 0.0) * 100.0

        ess_val = row.get("ESS_per_sec_mean", float("nan"))
        ess_sem = row.get("ESS_per_sec_sem", 0.0)

        kl_val = row.get("kl_spin_mean", row.get("kl_total_mean", float("nan")))
        kl_sem = row.get("kl_spin_sem", row.get("kl_total_sem", 0.0))

        lines.append(
            f"| `{sys_t}` | **{mod_t}** | `{init_m}` | {n_exp} | "
            f"${de_val:+.2f} \\pm {de_sem:.2f}$ | "
            f"${tau_val:.2f} \\pm {tau_sem:.2f}$ | "
            f"${acc_val:.1f}\\% \\pm {acc_sem:.1f}\\%$ | "
            f"${ess_val:.1f} \\pm {ess_sem:.1f}$ | "
            f"${kl_val:.4f} \\pm {kl_sem:.4f}$ |"
        )

    lines.extend([
        "",
        "---",
        "Reporte generado automáticamente por `scripts/plot_initialization_benchmark.py`."
    ])

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Reporte] Reporte Markdown guardado en: {md_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficador y Generador de Tablas del Benchmark de Inicializaciones.")
    parser.add_argument("results_dir", type=str, help="Carpeta raíz con los resultados del benchmark.")
    parser.add_argument("--out_dir", type=str, default=None, help="Carpeta de salida para figuras y reportes (por defecto: results_dir/figures).")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "pdf", "both"], help="Formato de exportación gráfica.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolución en puntos por pulgada.")
    parser.add_argument("--font_scale", type=float, default=1.0, help="Escala de fuentes tipográficas.")
    parser.add_argument("--line_width_scale", type=float, default=1.0, help="Escala de grosor de línea.")
    args = parser.parse_args()

    root_dir = Path(args.results_dir)
    if not root_dir.exists():
        print(f"Error: La carpeta de resultados '{root_dir}' no existe.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else root_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib_style(args.font_scale, args.line_width_scale)

    print(f"[Benchmark] Recolectando corridas de inicialización desde: {root_dir}...")
    raw_df = collect_benchmark_data(root_dir)
    if raw_df.empty:
        print(f"[Aviso] No se encontraron corridas en {root_dir}. Verifica la ruta.")
        sys.exit(0)

    print(f"[Benchmark] Encontradas {len(raw_df)} corridas individuales.")
    agg_df = aggregate_statistics(raw_df)

    export_markdown_and_csv_reports(agg_df, out_dir)

    fmts = ["png", "pdf"] if args.format == "both" else [args.format]
    for fmt in fmts:
        for sys_tag in agg_df["system_tag"].unique():
            plot_system_initialization_comparison(agg_df, sys_tag, out_dir, fmt=fmt, dpi=args.dpi)
        plot_speedup_scaling_summary(agg_df, out_dir, fmt=fmt, dpi=args.dpi)

    print("[Benchmark] Post-procesamiento y graficación completados con éxito.")


if __name__ == "__main__":
    main()
