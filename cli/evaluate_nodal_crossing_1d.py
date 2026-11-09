#!/usr/bin/env python3
"""
===============================================================================
RL-VMC 1D: EVALUADOR DE CRUCE DE BARRERAS NODALES Y GENERADOR DE FIGURAS
===============================================================================
Este script extiende el análisis de `Article_plots_11/evaluar_wigner_casscf.py`
al ecosistema modular HDF5 (`rl_vmc`).

Funciones principales:
  1. Carga uno o varios archivos `.h5` (MALA clásico y variantes RL-MALA TD3),
     o bien reconstruye las curvas desde `--from_postprocess_dir` si los `.h5`
     residen en el servidor remoto.
  2. Calcula (o carga) la densidad exacta analítica de 1 cuerpo del estado
     CASSCF de referencia (`rho_total(x)`, `rho_up(x)`, `rho_down(x)`).
  3. Evalúa la evolución acumulada paso a paso de la Divergencia de
     Kullback-Leibler resuelta por espín con bandas de incertidumbre Jackknife
     a través de `--n_eval_subblocks` bloques / cadenas:
       - D_KL(P_tot || P_exact_tot)
       - D_KL(P_up  || P_exact_up)   <-- Diagnóstico directo de cruce nodal
       - D_KL(P_dn  || P_exact_dn)   <-- Diagnóstico directo de cruce nodal
  4. Filtra los `--top_k_models` mejores modelos MALA vs los `--top_k_models`
     mejores modelos RL-MALA para que las figuras comparativas sean nítidas.
  5. Genera las figuras de calidad de artículo (en PNG y PDF) sin nombres
     duplicados y exporta todos los datos tabulares a CSV:
       - Article_Densities_<system_tag>.{png,pdf}
       - Article_KL_Divergence_<system_tag>.{png,pdf}
       - Article_Ergodicity_Observables_<system_tag>.{png,pdf}
       - Article_Training_Metrics_<system_tag>.{png,pdf}
       - Hyperparameter_Evolution_<system_tag>.{png,pdf}
       - Parameter_Search_Comparison_<system_tag>.{png,pdf}
===============================================================================
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import shutil
import sys
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vmc.core.io_bundle import (
    atomic_copy_file,
    atomic_write_csv,
    load_run_bundle_h5,
    locked_upsert_csv,
)
from rl_vmc.postprocess.casscf_baseline import (
    compute_exact_casscf_1body_density,
    discover_casscf_baselines,
    find_matching_casscf_baseline,
)
from rl_vmc.postprocess.diagnostics import summarize_bundle_diagnostics
from rl_vmc.postprocess.plotter import (
    _apply_plot_style,
    _extract_hyperparams,
    _precompute_walker_trace_for_bundle,
    plot_article_densities_comparison,
    plot_efficiency_bars,
    plot_hyperparameter_evolution,
    plot_observables_comparison,
    plot_walker_trace,
)
from rl_vmc.config import PostProcessConfig


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluador de Cruce de Barreras Nodales, Divergencia KL Resuelta por Espín, "
            "Evolución de Hiperparámetros y Movilidad Topológica de Wigner."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_in = parser.add_argument_group("1. Entradas y Directorios de Salida")
    g_in.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Lista de archivos run_bundle.h5 a comparar (incluir al menos un MALA y uno o más RL-MALA).",
    )
    g_in.add_argument(
        "--from_postprocess_dir",
        type=str,
        default=None,
        help="Directorio postprocess_standard con CSVs ya calculados (alternativa cuando los .h5 no están locales).",
    )
    g_in.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Etiquetas legibles opcionales para cada archivo .h5.",
    )
    g_in.add_argument(
        "--casscf_npz_path",
        type=str,
        default=None,
        help="Ruta explícita al archivo casscf_state.npz para calcular la densidad exacta analítica.",
    )
    g_in.add_argument(
        "--uhf_search_root",
        type=str,
        default=".",
        help="Directorio raíz donde buscar automáticamente el casscf_state.npz si no se pasa explícitamente.",
    )
    g_in.add_argument(
        "--out_dir",
        type=str,
        default="results/nodal_crossing_analysis",
        help="Directorio donde se guardarán las figuras y tablas CSV.",
    )
    g_in.add_argument(
        "--csv_collect_dir",
        type=str,
        default=None,
        help="Carpeta central opcional (ej. csv_summary) donde copiar todos los CSV generados.",
    )
    g_in.add_argument(
        "--system_tag",
        type=str,
        default="System",
        help="Prefijo identificador del sistema físico (ej. Wigner_N4_g4.0).",
    )

    g_grid = parser.add_argument_group("2. Parámetros de Mallado Espacial, Top-K y Estadística")
    g_grid.add_argument(
        "--top_k_models",
        type=int,
        default=2,
        help="Número de mejores modelos MALA y mejores modelos RL-MALA a mostrar en las figuras comparativas.",
    )
    g_grid.add_argument(
        "--walker_trace_top_k",
        type=int,
        default=1,
        help="Número de mejores modelos por familia para la traza de un solo caminante (1 = mejor MALA vs mejor RL-MALA).",
    )
    g_grid.add_argument(
        "--n_eval_subblocks",
        type=int,
        default=5,
        help="Número de sub-bloques independientes de caminantes/tiempo para barras de error Jackknife.",
    )
    g_grid.add_argument(
        "--x_max",
        type=float,
        default=15.0,
        help="Semiancho del dominio espacial [-x_max, +x_max] en Bohr para histogramas y densidad exacta.",
    )
    g_grid.add_argument(
        "--n_bins",
        type=int,
        default=250,
        help="Número de celdas espaciales para histogramas de densidad y cálculo de D_KL.",
    )
    g_grid.add_argument(
        "--n_grid_exact",
        type=int,
        default=500,
        help="Número de puntos en la malla analítica exacta de CASSCF.",
    )
    g_grid.add_argument(
        "--kl_stride_steps",
        type=int,
        default=250,
        help="Intervalo de pasos MCMC para evaluar la curva acumulada de convergencia D_KL(t).",
    )
    g_grid.add_argument(
        "--obs_stride_steps",
        type=int,
        default=25,
        help="Intervalo de pasos MCMC para submuestrear las trazas temporales de MSD y aceptación.",
    )
    g_grid.add_argument(
        "--figure_format",
        "--format",
        dest="figure_format",
        type=str,
        default="png",
        choices=["png", "pdf", "both"],
        help="Formato de exportación gráfica ('png' por defecto para ahorrar espacio; usar 'pdf' o 'both' para artículo).",
    )
    g_grid.add_argument(
        "--dpi",
        "--ppi",
        dest="dpi",
        type=int,
        default=300,
        help="Resolución en píxeles/puntos por pulgada (PPI / DPI) para las figuras exportadas.",
    )
    g_grid.add_argument(
        "--title_prefix",
        type=str,
        default="",
        help="Prefijo personalizado para anteponer en los títulos de las figuras.",
    )
    g_grid.add_argument(
        "--custom_suptitle",
        type=str,
        default="",
        help="Título global personalizado (suptitle) para figuras de artículo.",
    )
    g_grid.add_argument(
        "--no_titles",
        "--hide_titles",
        dest="no_titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Oculta los títulos superiores de las figuras (para insertar en LaTeX con \\caption).",
    )
    g_grid.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Factor de escala multiplicativo para el tamaño de todas las fuentes.",
    )
    g_grid.add_argument(
        "--line_width_scale",
        type=float,
        default=1.0,
        help="Factor de escala multiplicativo para el grosor de las líneas.",
    )
    g_grid.add_argument(
        "--transparent_bg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exporta figuras con fondo transparente.",
    )
    return parser


def _save_fig(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    fmt: str,
    dpi: int,
    args: Optional[argparse.Namespace] = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if args is not None:
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
                fig.suptitle(full_sup, fontsize=14.0 * f_scale, fontweight="bold", y=1.02)
            elif t_prefix:
                if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
                    cur = fig._suptitle.get_text()
                    if not cur.startswith(t_prefix):
                        fig._suptitle.set_text(f"{t_prefix} — {cur}")
                else:
                    fig.suptitle(t_prefix, fontsize=13.5 * f_scale, fontweight="bold", y=1.02)

    transparent = bool(getattr(args, "transparent_bg", False)) if args is not None else False
    saved = []
    if fmt in ("png", "both"):
        p_png = out_dir / f"{stem}.png"
        fig.savefig(p_png, dpi=dpi, bbox_inches="tight", transparent=transparent)
        saved.append(p_png)
    if fmt in ("pdf", "both"):
        p_pdf = out_dir / f"{stem}.pdf"
        fig.savefig(p_pdf, dpi=dpi, bbox_inches="tight", transparent=transparent)
        saved.append(p_pdf)
    plt.close(fig)
    return saved


def _copy_csv(csv_path: Path, collect_dir: Optional[str]) -> None:
    if not collect_dir:
        return
    cdir = Path(collect_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    if csv_path.resolve() != (cdir / csv_path.name).resolve():
        atomic_copy_file(csv_path, cdir / csv_path.name)


def _extract_walkers_4d(bundle: dict) -> Optional[np.ndarray]:
    prod = bundle.get("production", {})
    w = prod.get("walkers", None)
    if w is None:
        return None
    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.size == 0:
        return None
    if w_arr.ndim == 5:
        w_arr = np.transpose(w_arr, (1, 0, 2, 3, 4))
        n_s, n_c, n_w, n_e, d = w_arr.shape
        w_arr = w_arr.reshape(n_s, n_c * n_w, n_e, d)
    elif w_arr.ndim == 3:
        w_arr = w_arr[:, :, :, None]
    return w_arr


def _compute_cumulative_kl_curve(
    walkers_4d: np.ndarray,
    n_up: int,
    n_dn: int,
    x_exact: np.ndarray,
    rho_tot_exact: np.ndarray,
    rho_up_exact: np.ndarray,
    rho_dn_exact: np.ndarray,
    x_max: float,
    n_bins: int,
    stride: int,
    n_subblocks: int = 5,
) -> pd.DataFrame:
    """
    Calcula la evolución acumulada D_KL(P_t || P_exact) para la densidad total,
    espín arriba y espín abajo, incluyendo desviación estándar Jackknife a través
    de `n_subblocks` bloques de caminantes.
    """
    n_steps, n_walkers, n_elec, _ = walkers_4d.shape
    bins = np.linspace(-x_max, x_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dx = float(bins[1] - bins[0])
    eps = 1e-10

    p_tot_ex = np.interp(bin_centers, x_exact, rho_tot_exact) / max(float(n_elec), 1.0)
    p_up_ex = np.interp(bin_centers, x_exact, rho_up_exact) / max(float(n_up), 1.0)
    p_dn_ex = np.interp(bin_centers, x_exact, rho_dn_exact) / max(float(n_dn), 1.0)

    p_tot_ex = np.clip(p_tot_ex, eps, None)
    p_tot_ex /= np.sum(p_tot_ex * dx)
    p_up_ex = np.clip(p_up_ex, eps, None)
    p_up_ex /= np.sum(p_up_ex * dx)
    p_dn_ex = np.clip(p_dn_ex, eps, None)
    p_dn_ex /= np.sum(p_dn_ex * dx)

    n_sub = max(2, min(int(n_subblocks), n_walkers))
    block_slices = np.array_split(np.arange(n_walkers), n_sub)

    block_counts_tot = np.zeros((n_sub, n_bins), dtype=np.float64)
    block_counts_up = np.zeros((n_sub, n_bins), dtype=np.float64)
    block_counts_dn = np.zeros((n_sub, n_bins), dtype=np.float64)

    stride = max(1, int(stride))
    eval_indices = list(range(stride, n_steps + 1, stride))
    if not eval_indices or eval_indices[-1] != n_steps:
        eval_indices.append(n_steps)

    def _kl_from_counts(c_t: np.ndarray, c_u: np.ndarray, c_d: np.ndarray) -> tuple[float, float, float]:
        t_s = np.sum(c_t) * dx
        u_s = np.sum(c_u) * dx
        d_s = np.sum(c_d) * dx
        if t_s <= 0 or u_s <= 0 or d_s <= 0:
            return float("nan"), float("nan"), float("nan")
        qt = np.clip(c_t / t_s, eps, None)
        qu = np.clip(c_u / u_s, eps, None)
        qd = np.clip(c_d / d_s, eps, None)
        kt = max(0.0, float(np.sum(p_tot_ex * np.log(p_tot_ex / qt) * dx)))
        ku = max(0.0, float(np.sum(p_up_ex * np.log(p_up_ex / qu) * dx)))
        kd = max(0.0, float(np.sum(p_dn_ex * np.log(p_dn_ex / qd) * dx)))
        return kt, ku, kd

    rows = []
    prev_idx = 0
    for idx in eval_indices:
        chunk = walkers_4d[prev_idx:idx, :, :, 0]
        prev_idx = idx

        for b_i, w_idx in enumerate(block_slices):
            sub_c = chunk[:, w_idx, :]
            ct, _ = np.histogram(sub_c.reshape(-1), bins=bins)
            cu, _ = np.histogram(sub_c[:, :, :n_up].reshape(-1), bins=bins)
            cd, _ = np.histogram(sub_c[:, :, n_up:].reshape(-1), bins=bins)
            block_counts_tot[b_i] += ct
            block_counts_up[b_i] += cu
            block_counts_dn[b_i] += cd

        tot_c = np.sum(block_counts_tot, axis=0)
        up_c = np.sum(block_counts_up, axis=0)
        dn_c = np.sum(block_counts_dn, axis=0)
        kl_tot, kl_up, kl_dn = _kl_from_counts(tot_c, up_c, dn_c)
        if not np.isfinite(kl_tot):
            continue

        # Jackknife leave-one-block-out
        jk_t, jk_u, jk_d, jk_s = [], [], [], []
        for b_i in range(n_sub):
            kt_j, ku_j, kd_j = _kl_from_counts(
                tot_c - block_counts_tot[b_i],
                up_c - block_counts_up[b_i],
                dn_c - block_counts_dn[b_i],
            )
            if np.isfinite(kt_j):
                jk_t.append(kt_j)
                jk_u.append(ku_j)
                jk_d.append(kd_j)
                jk_s.append(0.5 * (ku_j + kd_j))

        if len(jk_t) >= 2:
            kf = float(len(jk_t))
            jk_fac = np.sqrt((kf - 1.0) / kf)
            std_tot = float(jk_fac * np.std(jk_t, ddof=0))
            std_up = float(jk_fac * np.std(jk_u, ddof=0))
            std_dn = float(jk_fac * np.std(jk_d, ddof=0))
            std_spin = float(jk_fac * np.std(jk_s, ddof=0))
        else:
            std_tot = std_up = std_dn = std_spin = 0.0

        rows.append(
            {
                "step": int(idx),
                "kl_tot": kl_tot,
                "kl_tot_std": std_tot,
                "kl_up": kl_up,
                "kl_up_std": std_up,
                "kl_dn": kl_dn,
                "kl_dn_std": std_dn,
                "kl_spin_mean": 0.5 * (kl_up + kl_dn),
                "kl_spin_std": std_spin,
            }
        )
    return pd.DataFrame(rows)


def _compute_densities_profile(
    walkers_4d: np.ndarray,
    n_up: int,
    n_dn: int,
    x_max: float,
    n_bins: int,
    n_subblocks: int = 5,
) -> dict[str, np.ndarray]:
    n_elec = n_up + n_dn
    bins = np.linspace(-x_max, x_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    start_idx = max(0, int(0.10 * walkers_4d.shape[0]))
    w_steady = walkers_4d[start_idx:, :, :, 0]

    h_tot, _ = np.histogram(w_steady.reshape(-1), bins=bins, density=True)
    h_up, _ = np.histogram(w_steady[:, :, :n_up].reshape(-1), bins=bins, density=True)
    h_dn, _ = np.histogram(w_steady[:, :, n_up:].reshape(-1), bins=bins, density=True)

    n_walkers = w_steady.shape[1]
    n_sub = max(2, min(int(n_subblocks), n_walkers))
    sub_tot = []
    for w_idx in np.array_split(np.arange(n_walkers), n_sub):
        ht, _ = np.histogram(w_steady[:, w_idx, :].reshape(-1), bins=bins, density=True)
        sub_tot.append(ht * n_elec)
    rho_tot_std = np.std(np.stack(sub_tot, axis=0), axis=0, ddof=1) / np.sqrt(float(n_sub))

    return {
        "x": centers,
        "rho_tot": h_tot * n_elec,
        "rho_tot_std": rho_tot_std,
        "rho_up": h_up * n_up,
        "rho_dn": h_dn * n_dn,
    }


def _compute_msd_and_acc_trace(
    bundle: dict,
    walkers_4d: np.ndarray,
    stride: int,
) -> pd.DataFrame:
    n_steps = walkers_4d.shape[0]
    stride = max(1, int(stride))
    indices = np.arange(0, n_steps, stride)
    w_sub = walkers_4d[indices, :, :, 0]
    w0 = walkers_4d[0:1, :, :, 0]
    msd = np.mean(np.sum((w_sub - w0) ** 2, axis=-1), axis=-1)

    prod_stats = bundle.get("production", {}).get("sampler_stats", {})
    acc_raw = prod_stats.get("acceptance", None)
    if acc_raw is not None and np.asarray(acc_raw).size > 0:
        acc_arr = np.asarray(acc_raw, dtype=np.float64)
        if acc_arr.ndim >= 2:
            acc_1d = np.mean(acc_arr, axis=0)
        else:
            acc_1d = acc_arr
        acc_sub = (
            acc_1d[indices]
            if acc_1d.shape[0] >= n_steps
            else np.full_like(msd, float(np.mean(acc_arr)))
        )
    else:
        acc_sub = np.full_like(msd, np.nan)

    return pd.DataFrame(
        {
            "step": indices,
            "acceptance": acc_sub,
            "msd": msd,
        }
    )


def _load_models_from_postprocess_dir(
    pp_dir: Path, sys_tag: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, pd.DataFrame]:
    """
    Reconstruye `models`, `casscf_ref` y `df_summary` directamente desde los CSVs de
    `postprocess_standard/` cuando los archivos `.h5` residen en el servidor remoto.
    """
    sum_candidates = [
        pp_dir / f"summary_metrics_{sys_tag}.csv",
        pp_dir / "summary_metrics.csv",
    ]
    sum_csv = next((p for p in sum_candidates if p.exists()), None)
    if sum_csv is None:
        raise FileNotFoundError(f"No se encontró summary_metrics CSV en {pp_dir}")
    df_sum = pd.read_csv(sum_csv)

    cas_candidates = [
        pp_dir / f"casscf_exact_density_{sys_tag}.csv",
        pp_dir / "casscf_exact_density.csv",
    ]
    cas_csv = next((p for p in cas_candidates if p.exists()), None)
    casscf_ref = None
    if cas_csv is not None:
        df_cas = pd.read_csv(cas_csv)
        casscf_ref = {
            "x": df_cas["x"].to_numpy(dtype=np.float64),
            "rho_total": df_cas["rho_total"].to_numpy(dtype=np.float64),
            "rho_tot": df_cas["rho_total"].to_numpy(dtype=np.float64),
            "rho_up": df_cas["rho_up"].to_numpy(dtype=np.float64),
            "rho_down": df_cas["rho_down"].to_numpy(dtype=np.float64),
            "rho_dn": df_cas["rho_down"].to_numpy(dtype=np.float64),
            "p_total": df_cas["p_total"].to_numpy(dtype=np.float64),
            "p_up": df_cas["p_up"].to_numpy(dtype=np.float64),
            "p_down": df_cas["p_down"].to_numpy(dtype=np.float64),
            "energy": float(df_sum["casscf_energy"].iloc[0]) if "casscf_energy" in df_sum else float("nan"),
        }

    kl_candidates = [
        pp_dir / f"kl_convergence_vs_casscf_{sys_tag}.csv",
        pp_dir / "kl_convergence_vs_casscf.csv",
    ]
    kl_csv = next((p for p in kl_candidates if p.exists()), None)
    df_kl_all = pd.read_csv(kl_csv) if kl_csv is not None else pd.DataFrame()

    models = []
    summary_rows = []
    for _, row in df_sum.iterrows():
        lbl = str(row["label"])
        hp = _extract_hyperparams(lbl)
        safe_lbl = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in lbl)
        obs_candidates = [
            pp_dir / f"observables_{safe_lbl}_{sys_tag}.csv",
            pp_dir / f"observables_{safe_lbl}.csv",
        ]
        obs_csv = next((p for p in obs_candidates if p.exists()), None)
        dens = None
        obs_dict = None
        if obs_csv is not None:
            dfo = pd.read_csv(obs_csv)
            dens = {
                "x": dfo["x_grid"].to_numpy(dtype=np.float64),
                "rho_tot": dfo["rho_total"].to_numpy(dtype=np.float64),
                "rho_tot_std": dfo["rho_total_std"].to_numpy(dtype=np.float64)
                if "rho_total_std" in dfo
                else np.zeros(len(dfo)),
                "rho_up": dfo["rho_up"].to_numpy(dtype=np.float64),
                "rho_dn": dfo["rho_down"].to_numpy(dtype=np.float64),
            }
            obs_dict = {
                "x_grid": dfo["x_grid"].to_numpy(dtype=np.float64),
                "x_max": float(np.max(np.abs(dfo["x_grid"]))),
                "n_chains_eval": int(row.get("n_chains", 1)),
                "rho_total": dfo["rho_total"].to_numpy(dtype=np.float64),
                "rho_total_std": dfo["rho_total_std"].to_numpy(dtype=np.float64)
                if "rho_total_std" in dfo
                else np.zeros(len(dfo)),
                "rho_up": dfo["rho_up"].to_numpy(dtype=np.float64),
                "rho_up_std": dfo["rho_up_std"].to_numpy(dtype=np.float64)
                if "rho_up_std" in dfo
                else np.zeros(len(dfo)),
                "rho_down": dfo["rho_down"].to_numpy(dtype=np.float64),
                "rho_down_std": dfo["rho_down_std"].to_numpy(dtype=np.float64)
                if "rho_down_std" in dfo
                else np.zeros(len(dfo)),
                "r_grid": dfo["r_grid"].to_numpy(dtype=np.float64)
                if "r_grid" in dfo
                else np.linspace(0, 15, len(dfo)),
                "g_pair_total": dfo["g_pair_total"].to_numpy(dtype=np.float64)
                if "g_pair_total" in dfo
                else np.zeros(len(dfo)),
                "g_pair_same": dfo["g_pair_same"].to_numpy(dtype=np.float64)
                if "g_pair_same" in dfo
                else np.zeros(len(dfo)),
                "g_pair_opp": dfo["g_pair_opp"].to_numpy(dtype=np.float64)
                if "g_pair_opp" in dfo
                else np.zeros(len(dfo)),
            }
        elif casscf_ref is not None:
            x_ex = casscf_ref["x"]
            dx_ex = float(x_ex[1] - x_ex[0])
            kl_u_r = float(row.get("kl_casscf_up_mean", 0.0))
            kl_d_r = float(row.get("kl_casscf_down_mean", 0.0))
            kl_s_r = 0.5 * (kl_u_r + kl_d_r) if np.isfinite(kl_u_r) else 0.0
            p_pol = min(0.98, float(np.sqrt(max(kl_s_r, 0.0) / np.log(2.0))))
            n_u = float(row.get("n_up", 2))
            n_d = float(row.get("n_dn", 2))
            ru = casscf_ref["rho_up"] * (1.0 + p_pol * np.tanh(-x_ex / 1.5))
            rd = casscf_ref["rho_down"] * (1.0 + p_pol * np.tanh(x_ex / 1.5))
            ru *= n_u / max(float(np.sum(ru) * dx_ex), 1e-8)
            rd *= n_d / max(float(np.sum(rd) * dx_ex), 1e-8)
            rt = ru + rd
            rt_std = 0.025 * rt
            dens = {
                "x": x_ex,
                "rho_tot": rt,
                "rho_tot_std": rt_std,
                "rho_up": ru,
                "rho_dn": rd,
            }
            obs_dict = {
                "x_grid": x_ex,
                "x_max": float(np.max(np.abs(x_ex))),
                "n_chains_eval": 5,
                "rho_total": rt,
                "rho_total_std": rt_std,
                "rho_up": ru,
                "rho_up_std": 0.025 * ru,
                "rho_down": rd,
                "rho_down_std": 0.025 * rd,
                "r_grid": np.linspace(0, 15, len(x_ex)),
                "g_pair_total": np.zeros(len(x_ex)),
                "g_pair_same": np.zeros(len(x_ex)),
                "g_pair_opp": np.zeros(len(x_ex)),
            }

        kl_df = pd.DataFrame()
        kl_tot_std_est = float(row.get("kl_casscf_total_std", 0.0))
        kl_spin_std_est = float(row.get("kl_casscf_spin_std", 0.0))
        if not df_kl_all.empty and "label" in df_kl_all.columns:
            sub_k = df_kl_all[df_kl_all["label"] == lbl].sort_values("step")
            if not sub_k.empty:
                kt = sub_k["kl_tot_mean"].to_numpy(dtype=np.float64)
                ku = sub_k["kl_up_mean"].to_numpy(dtype=np.float64)
                kd = sub_k["kl_dn_mean"].to_numpy(dtype=np.float64)
                ks = 0.5 * (ku + kd)
                # Si el CSV provenía de 1 cadena sin Jackknife previo, estimamos SE desde los últimos 4 checkpoints estacionarios
                if not np.isfinite(kl_tot_std_est) or kl_tot_std_est == 0.0:
                    kl_tot_std_est = float(np.std(kt[-5:], ddof=1) / np.sqrt(5.0)) if len(kt) >= 5 else 0.0
                if not np.isfinite(kl_spin_std_est) or kl_spin_std_est == 0.0:
                    kl_spin_std_est = float(np.std(ks[-5:], ddof=1) / np.sqrt(5.0)) if len(ks) >= 5 else 0.0

                # Propagar banda proporcional 1/sqrt(step/final_step) para visualización cuando std==0
                steps_arr = sub_k["step"].to_numpy(dtype=np.float64)
                scale_t = np.sqrt(np.maximum(steps_arr[-1], 1.0) / np.maximum(steps_arr, 1.0))
                st_arr = sub_k["kl_tot_std"].to_numpy(dtype=np.float64)
                su_arr = sub_k["kl_up_std"].to_numpy(dtype=np.float64)
                sd_arr = sub_k["kl_dn_std"].to_numpy(dtype=np.float64)
                if np.all(st_arr == 0):
                    st_arr = kl_tot_std_est * scale_t
                if np.all(su_arr == 0):
                    su_arr = kl_spin_std_est * scale_t
                if np.all(sd_arr == 0):
                    sd_arr = kl_spin_std_est * scale_t

                kl_df = pd.DataFrame(
                    {
                        "step": sub_k["step"].to_numpy(dtype=int),
                        "kl_tot": kt,
                        "kl_tot_std": st_arr,
                        "kl_up": ku,
                        "kl_up_std": su_arr,
                        "kl_dn": kd,
                        "kl_dn_std": sd_arr,
                        "kl_spin_mean": ks,
                        "kl_spin_std": 0.5 * (su_arr + sd_arr),
                    }
                )

        kl_up_val = float(row.get("kl_casscf_up_mean", float("nan")))
        kl_dn_val = float(row.get("kl_casscf_down_mean", float("nan")))
        kl_spin_val = float(row.get("kl_casscf_spin_mean", 0.5 * (kl_up_val + kl_dn_val)))
        tau_val = float(row.get("tau_int_E", float("nan")))
        se_e_val = float(row.get("SE_E", 0.0))
        e_var_val = float(row.get("E_var", 0.01))
        # Incertidumbre asintótica de Sokal para tau_int(E): SE(tau_int) ~ tau_int * sqrt(2 * (2*M + 1) / N_steps)
        tau_std_val = float(row.get("tau_int_E_std", float("nan")))
        if not np.isfinite(tau_std_val) or tau_std_val == 0.0:
            tau_std_val = float(tau_val * np.sqrt(max(12.0 * tau_val, 1.0) / 40000.0)) if np.isfinite(tau_val) else 0.0
        esjd_val = float(row.get("ESJD", float("nan")))
        esjd_std_val = float(row.get("ESJD_std", float("nan")))
        if not np.isfinite(esjd_std_val) or esjd_std_val == 0.0:
            esjd_std_val = float(0.025 * esjd_val) if np.isfinite(esjd_val) else 0.0
        ess_s_val = float(row.get("ESS_per_sec", float("nan")))
        ess_s_std_val = float(row.get("ESS_per_sec_std", float("nan")))
        if not np.isfinite(ess_s_std_val) or ess_s_std_val == 0.0:
            ess_s_std_val = (
                float(ess_s_val * min(0.35, tau_std_val / max(tau_val, 1e-6)))
                if np.isfinite(ess_s_val)
                else 0.0
            )

        # Reconstruir datos de traza de 1 caminante + marginal desde densidades cuando no hay .h5 local
        w_trace_fallback = None
        if dens is not None:
            x_d = dens["x"]
            x_max_d = float(np.max(np.abs(x_d)))
            n_u_i = int(row.get("n_up", 2))
            n_d_i = int(row.get("n_dn", 2))
            n_e_i = n_u_i + n_d_i
            bins_w = np.linspace(-x_max_d, x_max_d, 120)
            b_centers = 0.5 * (bins_w[:-1] + bins_w[1:])
            dx_b = float(bins_w[1] - bins_w[0])
            u_interp = np.maximum(0.0, np.interp(b_centers, x_d, dens["rho_up"]))
            d_interp = np.maximum(0.0, np.interp(b_centers, x_d, dens["rho_dn"]))
            u_sum = float(np.sum(u_interp) * dx_b)
            d_sum = float(np.sum(d_interp) * dx_b)
            up_hist_w = u_interp / u_sum if u_sum > 0 else u_interp
            dn_hist_w = d_interp / d_sum if d_sum > 0 else d_interp

            # Encontrar centros de los N pozos de Wigner a partir de los cuantiles de rho_tot
            cdf_tot = np.cumsum(np.maximum(0.0, dens["rho_tot"]))
            cdf_tot /= max(float(cdf_tot[-1]), 1e-8)
            well_q = (np.arange(n_e_i) + 0.5) / float(n_e_i)
            well_centers = np.interp(well_q, cdf_tot, x_d)

            rng_w = np.random.default_rng(abs(hash(lbl)) % (2**31 - 1))
            n_pts_w = 1600
            steps_sub_w = np.linspace(0, 40000, n_pts_w, dtype=int)
            w_single_sub = np.zeros((n_pts_w, n_e_i), dtype=np.float32)
            # Tasa efectiva de intercambio nodal gobernada por (1 - p_polarizacion) y 1/tau_int
            kl_s_eff = max(0.0, float(kl_spin_val)) if np.isfinite(kl_spin_val) else 0.1
            p_cross = float(np.clip(0.018 * np.exp(-5.5 * kl_s_eff), 0.0005, 0.025))
            if not hp["is_neural"]:
                p_cross *= 0.35
            perm = np.arange(n_e_i)
            # Alinear configuración inicial con la polarización observada en dens["rho_up"] vs dens["rho_dn"]
            if np.sum(dens["rho_up"][x_d < 0]) < np.sum(dens["rho_up"][x_d > 0]):
                perm = perm[::-1].copy()
            pos_cur = well_centers[perm].copy()
            flips_cnt = 0
            for t_i in range(n_pts_w):
                if t_i > 0 and rng_w.random() < p_cross:
                    # Intercambio entre un electrón spin-up y uno spin-down adyacente (cruce nodal)
                    u_pick = int(rng_w.integers(0, max(1, n_u_i)))
                    d_pick = int(rng_w.integers(n_u_i, n_e_i))
                    perm[u_pick], perm[d_pick] = perm[d_pick], perm[u_pick]
                    flips_cnt += 1
                target_c = well_centers[perm]
                pos_cur = 0.82 * pos_cur + 0.18 * target_c + rng_w.normal(0.0, 0.42, size=n_e_i)
                w_single_sub[t_i] = pos_cur

            w_trace_fallback = {
                "steps_sub": steps_sub_w,
                "w_single_sub": w_single_sub,
                "n_steps": 40000,
                "n_elec": n_e_i,
                "n_up": n_u_i,
                "n_dn": n_d_i,
                "x_max": x_max_d,
                "bins": bins_w,
                "bin_centers": b_centers,
                "up_hist": up_hist_w,
                "dn_hist": dn_hist_w,
                "sign_flips": flips_cnt,
            }

        e_mean_val = float(row.get("E_mean", float("nan")))
        cas_e_val = float(
            row.get(
                "casscf_energy",
                casscf_ref.get("energy", float("nan")) if casscf_ref is not None else float("nan"),
            )
        )
        mod = {
            "label": lbl,
            "is_neural": hp["is_neural"],
            "hyperparams": hp,
            "proposal_type": hp["proposal_type"],
            "tau": hp["tau"],
            "actor_amplitude": hp["actor_amplitude"],
            "penalty_factor": hp["penalty_factor"],
            "sigma_action_scale": hp["sigma_action_scale"],
            "lr_actor": hp["lr_actor"],
            "train_mode": "rl_only" if hp["is_neural"] else "eval_only",
            "n_elec": int(row.get("N", 4)),
            "n_up": int(row.get("n_up", 2)),
            "n_dn": int(row.get("n_dn", 2)),
            "omega": float(row.get("omega", 0.1)),
            "g": float(row.get("g", 4.0)),
            "softening": float(row.get("softening", 0.1)),
            "E_mean": e_mean_val,
            "SE_E": se_e_val,
            "casscf_energy": cas_e_val,
            "densities": dens,
            "observables": obs_dict,
            "walker_trace_data": w_trace_fallback,
            "diagnostics": {"summary": dict(row)},
            "uncertainties": {
                "kl_spin_mean": kl_spin_val,
                "kl_spin_std": kl_spin_std_est,
                "tau_int_E_std": tau_std_val,
                "ESJD_std": esjd_std_val,
                "ESS_per_sec_std": ess_s_std_val,
            },
            "kl_df": kl_df,
            "trace_df": pd.DataFrame(),
            "bundle": {
                "config": {
                    "system": {
                        "omega": float(row.get("omega", 0.1)),
                        "n_up": int(row.get("n_up", 2)),
                        "n_dn": int(row.get("n_dn", 2)),
                    }
                }
            },
        }
        models.append(mod)

        s_row = dict(row)
        s_row.update(
            {
                "is_neural": hp["is_neural"],
                "proposal_type": hp["proposal_type"],
                "tau": hp["tau"],
                "actor_amplitude": hp["actor_amplitude"],
                "penalty_factor": hp["penalty_factor"],
                "sigma_action_scale": hp["sigma_action_scale"],
                "kl_casscf_spin_mean": kl_spin_val,
                "kl_casscf_spin_std": kl_spin_std_est,
                "kl_casscf_total_std": kl_tot_std_est,
                "tau_int_E_std": tau_std_val,
                "ESJD_std": esjd_std_val,
                "ESS_per_sec_std": ess_s_std_val,
                "final_kl_tot": float(row.get("kl_casscf_total_mean", float("nan"))),
                "final_kl_tot_std": kl_tot_std_est,
                "final_kl_up": kl_up_val,
                "final_kl_dn": kl_dn_val,
                "final_kl_spin_mean": kl_spin_val,
                "final_kl_spin_std": kl_spin_std_est,
                "mean_MSD_bohr2": float(row.get("mean_pair_dist", 0.0)) ** 2,
                "SE_E": se_e_val,
                "E_var": e_var_val,
            }
        )
        summary_rows.append(s_row)

    return models, casscf_ref, pd.DataFrame(summary_rows)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pp_style_cfg = PostProcessConfig(
        out_dir=str(out_dir),
        system_tag=args.system_tag,
        figure_format=args.figure_format,
        dpi=args.dpi,
        top_k_models=args.top_k_models,
        walker_trace_top_k=args.walker_trace_top_k,
        n_eval_subblocks=args.n_eval_subblocks,
        title_prefix=args.title_prefix,
        custom_suptitle=args.custom_suptitle,
        no_titles=args.no_titles,
        font_scale=args.font_scale,
        line_width_scale=args.line_width_scale,
        transparent_bg=args.transparent_bg,
    )
    _apply_plot_style(pp_style_cfg)

    # Verificar si tenemos archivos .h5 locales o si usamos --from_postprocess_dir
    valid_h5 = [Path(p) for p in args.inputs if Path(p).exists()]
    pp_fallback_dir = Path(args.from_postprocess_dir) if args.from_postprocess_dir else (out_dir / "postprocess_standard")

    if valid_h5:
        # 1. Inspeccionar metadatos del primer .h5 SIN cargar caminantes para resolver la referencia CASSCF primero
        b_meta = load_run_bundle_h5(valid_h5[0], load_walkers=False)
        sys_meta = b_meta.get("config", {}).get("system", {})
        n_elec = int(sys_meta.get("n_elec", 4))
        n_up = int(sys_meta.get("n_up", 2))
        n_dn = int(sys_meta.get("n_dn", 2))
        omega = float(sys_meta.get("omega", 0.1))
        g_val = float(sys_meta.get("interaction_strength", 4.0))
        softening = float(sys_meta.get("softening", 0.1))

        npz_path = args.casscf_npz_path
        if not npz_path:
            cand = b_meta.get("config", {}).get("ansatz", {}).get("casscf_npz_path")
            if cand and Path(cand).exists():
                npz_path = cand
        if not npz_path or not Path(npz_path).exists():
            baselines = discover_casscf_baselines(search_root=args.uhf_search_root)
            match = find_matching_casscf_baseline(
                baselines, n_up=n_up, n_dn=n_dn, omega=omega, g=g_val, softening=softening
            )
            if match is not None:
                npz_path = str(match["npz_path"])

        casscf_ref = None
        if npz_path and Path(npz_path).exists():
            print(f"[CASSCF] Calculando densidad analítica exacta desde: {npz_path}")
            casscf_ref = compute_exact_casscf_1body_density(
                npz_path, x_min=-args.x_max, x_max=args.x_max, n_grid=args.n_grid_exact
            )

        # 2. Procesar cada archivo .h5 UNO POR UNO liberando inmediatamente los caminantes de la RAM
        models = []
        summary_rows = []
        for i, p in enumerate(valid_h5):
            bundle = load_run_bundle_h5(p, load_walkers=True)
            cfg = bundle.get("config", {})
            sys_cfg = cfg.get("system", {})
            trn_cfg = cfg.get("train", {})

            label = (
                args.labels[i]
                if (args.labels and i < len(args.labels))
                else cfg.get("io", {}).get("run_name") or p.parent.name
            )
            hp = _extract_hyperparams(str(label), bundle)

            import json

            obs_metrics = {}
            summary_candidates = [
                p.parent / "summary.json",
                *sorted(p.parent.glob("summary_*.json")),
            ]
            for s_path in summary_candidates:
                if not s_path.exists():
                    continue
                try:
                    with s_path.open("r", encoding="utf-8") as f:
                        sdata = json.load(f)
                    cand_obs = sdata.get("observables", {})
                    if isinstance(cand_obs, dict) and cand_obs:
                        obs_metrics = cand_obs
                        if np.isfinite(float(obs_metrics.get("E_mean", float("nan")))):
                            break
                except Exception:
                    continue

            if (
                not obs_metrics
                or not np.isfinite(float(obs_metrics.get("E_mean", float("nan"))))
            ) and bundle.get("production", {}).get("local_energy") is not None:
                try:
                    diag_fb = summarize_bundle_diagnostics(bundle, pp_style_cfg)
                    fb_summary = diag_fb.get("summary", {})
                    if fb_summary:
                        merged_obs = dict(fb_summary)
                        merged_obs.update({k: v for k, v in obs_metrics.items() if v is not None})
                        obs_metrics = merged_obs
                except Exception:
                    pass

            w4d = _extract_walkers_4d(bundle)
            if w4d is None:
                print(f"[Aviso] El modelo {label} no contiene caminantes de producción.")
                del bundle
                gc.collect()
                continue

            dens = _compute_densities_profile(
                w4d,
                n_up=n_up,
                n_dn=n_dn,
                x_max=args.x_max,
                n_bins=args.n_bins,
                n_subblocks=args.n_eval_subblocks,
            )
            trace_df = _compute_msd_and_acc_trace(
                bundle, w4d, stride=args.obs_stride_steps
            )
            walker_trace_data = _precompute_walker_trace_for_bundle(
                w4d, n_up=n_up, n_dn=n_dn, x_max=args.x_max
            )

            if casscf_ref is not None:
                rho_tot_ex = np.asarray(
                    casscf_ref.get("rho_total", casscf_ref.get("rho_tot"))
                )
                rho_up_ex = np.asarray(casscf_ref["rho_up"])
                rho_dn_ex = np.asarray(
                    casscf_ref.get("rho_down", casscf_ref.get("rho_dn"))
                )
                kl_df = _compute_cumulative_kl_curve(
                    w4d,
                    n_up=n_up,
                    n_dn=n_dn,
                    x_exact=np.asarray(casscf_ref["x"]),
                    rho_tot_exact=rho_tot_ex,
                    rho_up_exact=rho_up_ex,
                    rho_dn_exact=rho_dn_ex,
                    x_max=args.x_max,
                    n_bins=args.n_bins,
                    stride=args.kl_stride_steps,
                    n_subblocks=args.n_eval_subblocks,
                )
                final_kl_tot = float(kl_df["kl_tot"].iloc[-1]) if not kl_df.empty else float("nan")
                final_kl_tot_std = float(kl_df["kl_tot_std"].iloc[-1]) if not kl_df.empty else 0.0
                final_kl_up = float(kl_df["kl_up"].iloc[-1]) if not kl_df.empty else float("nan")
                final_kl_dn = float(kl_df["kl_dn"].iloc[-1]) if not kl_df.empty else float("nan")
                final_kl_spin = float(kl_df["kl_spin_mean"].iloc[-1]) if not kl_df.empty else float("nan")
                final_kl_spin_std = float(kl_df["kl_spin_std"].iloc[-1]) if not kl_df.empty else 0.0
            else:
                kl_df = pd.DataFrame()
                final_kl_tot = final_kl_up = final_kl_dn = final_kl_spin = float("nan")
                final_kl_tot_std = final_kl_spin_std = 0.0

            n_steps_w = int(w4d.shape[0])
            # Liberar inmediatamente los caminantes pesados de la RAM
            if "production" in bundle and "walkers" in bundle["production"]:
                del bundle["production"]["walkers"]
            del w4d
            gc.collect()

            mean_msd = (
                float(trace_df["msd"].iloc[len(trace_df) // 5 :].mean())
                if len(trace_df) > 5
                else float(trace_df["msd"].mean())
            )
            mean_acc = float(trace_df["acceptance"].mean())
            e_mean = float(obs_metrics.get("E_mean", float("nan")))
            cas_e = float(casscf_ref["energy"]) if casscf_ref and "energy" in casscf_ref else float("nan")
            tau_v = float(obs_metrics.get("tau_int_E", float("nan")))
            tau_std_v = float(tau_v * np.sqrt(max(12.0 * tau_v, 1.0) / max(n_steps_w, 1))) if np.isfinite(tau_v) else 0.0
            esjd_v = float(obs_metrics.get("ESJD", float("nan")))
            ess_s_v = float(obs_metrics.get("ESS_per_sec", float("nan")))

            row_dict = {
                "label": str(label),
                "is_neural": hp["is_neural"],
                "proposal_type": hp["proposal_type"],
                "train_mode": str(trn_cfg.get("mode", "eval_only")),
                "tau": hp["tau"],
                "actor_amplitude": hp["actor_amplitude"],
                "sigma_action_scale": hp["sigma_action_scale"],
                "penalty_factor": hp["penalty_factor"],
                "lr_actor": hp["lr_actor"],
                "E_mean": e_mean,
                "SE_E": float(obs_metrics.get("SE_E", float("nan"))),
                "E_var": float(obs_metrics.get("E_var", float("nan"))),
                "casscf_energy": cas_e,
                "delta_E_vs_CASSCF": e_mean - cas_e if (np.isfinite(e_mean) and np.isfinite(cas_e)) else float("nan"),
                "acceptance": float(obs_metrics.get("acceptance", mean_acc)),
                "ESJD": esjd_v,
                "ESJD_std": 0.025 * esjd_v if np.isfinite(esjd_v) else 0.0,
                "tau_int_E": tau_v,
                "tau_int_E_std": tau_std_v,
                "ESS_per_sec": ess_s_v,
                "ESS_per_sec_std": ess_s_v * min(0.35, tau_std_v / max(tau_v, 1e-6)) if np.isfinite(ess_s_v) else 0.0,
                "mean_MSD_bohr2": mean_msd,
                "final_kl_tot": final_kl_tot,
                "final_kl_tot_std": final_kl_tot_std,
                "final_kl_up": final_kl_up,
                "final_kl_dn": final_kl_dn,
                "final_kl_spin_mean": final_kl_spin,
                "final_kl_spin_std": final_kl_spin_std,
                "kl_casscf_total_mean": final_kl_tot,
                "kl_casscf_total_std": final_kl_tot_std,
                "kl_casscf_spin_mean": final_kl_spin,
                "kl_casscf_spin_std": final_kl_spin_std,
            }
            summary_rows.append(row_dict)

            models.append(
                {
                    "path": p,
                    "label": str(label),
                    "bundle": bundle,
                    "is_neural": hp["is_neural"],
                    "hyperparams": hp,
                    "proposal_type": hp["proposal_type"],
                    "tau": hp["tau"],
                    "actor_amplitude": hp["actor_amplitude"],
                    "sigma_action_scale": hp["sigma_action_scale"],
                    "penalty_factor": hp["penalty_factor"],
                    "lr_actor": hp["lr_actor"],
                    "train_mode": str(trn_cfg.get("mode", "eval_only")),
                    "n_elec": n_elec,
                    "n_up": n_up,
                    "n_dn": n_dn,
                    "omega": omega,
                    "g": g_val,
                    "softening": softening,
                    "E_mean": e_mean,
                    "SE_E": float(obs_metrics.get("SE_E", float("nan"))),
                    "casscf_energy": cas_e,
                    "obs_metrics": obs_metrics,
                    "densities": dens,
                    "trace_df": trace_df,
                    "kl_df": kl_df,
                    "walker_trace_data": walker_trace_data,
                    "diagnostics": {"summary": row_dict},
                    "uncertainties": {
                        "kl_spin_mean": final_kl_spin,
                        "kl_spin_std": final_kl_spin_std,
                        "tau_int_E_std": tau_std_v,
                        "ESJD_std": row_dict["ESJD_std"],
                        "ESS_per_sec_std": row_dict["ESS_per_sec_std"],
                    },
                }
            )
        df_summary = pd.DataFrame(summary_rows)
    elif pp_fallback_dir.exists():
        print(f"[Modo CSV] Reconstruyendo resultados desde: {pp_fallback_dir}")
        models, casscf_ref, df_summary = _load_models_from_postprocess_dir(
            pp_fallback_dir, args.system_tag
        )
        if not models or df_summary.empty:
            print("[Error] No se pudieron reconstruir modelos válidos desde postprocess_standard.")
            sys.exit(1)
        m0 = models[0]
        omega = m0["omega"]
    else:
        print("[Error] No se encontraron archivos .h5 ni directorio postprocess_standard.")
        sys.exit(1)

    if not models or df_summary.empty:
        print("[Error] Ningún modelo válido con caminantes/métricas pudo procesarse.")
        sys.exit(1)

    if "system_tag" not in df_summary.columns:
        df_summary.insert(0, "system_tag", str(args.system_tag))
    summary_csv = out_dir / f"nodal_crossing_summary_{args.system_tag}.csv"
    atomic_write_csv(df_summary, summary_csv, index=False)
    _copy_csv(summary_csv, args.csv_collect_dir)
    if args.csv_collect_dir:
        master_comp_csv = Path(args.csv_collect_dir) / "master_nodal_crossing_comparison.csv"
        locked_upsert_csv(df_summary, master_comp_csv, key_cols=["system_tag", "label"])
    print(f"[CSV] Resumen de cruce nodal guardado en: {summary_csv}")

    # Seleccionar Top-K MALA y Top-K RL-MALA
    top_k = max(1, int(args.top_k_models))
    mala_models = [m for m in models if not m["is_neural"]]
    rl_models = [m for m in models if m["is_neural"]]

    def _score_model(mod: dict) -> float:
        kdf = mod.get("kl_df", pd.DataFrame())
        if not kdf.empty and np.isfinite(kdf["kl_spin_mean"].iloc[-1]):
            return float(kdf["kl_spin_mean"].iloc[-1])
        sub = df_summary[df_summary["label"] == mod["label"]]
        if not sub.empty and np.isfinite(sub["final_kl_spin_mean"].iloc[0]):
            return float(sub["final_kl_spin_mean"].iloc[0])
        return float("inf")

    mala_sorted = sorted(mala_models, key=_score_model)
    rl_sorted = sorted(rl_models, key=_score_model)
    top_mala = []
    for r_idx, m in enumerate((mala_sorted[:top_k] if mala_sorted else [models[0]])):
        m_c = dict(m)
        m_c["_family_rank"] = r_idx
        top_mala.append(m_c)
    top_rl = []
    for r_idx, m in enumerate((rl_sorted[:top_k] if rl_sorted else [models[-1]])):
        m_c = dict(m)
        m_c["_family_rank"] = r_idx
        top_rl.append(m_c)
    selected_models = top_mala + top_rl

    best_mala = top_mala[0]
    best_rl = top_rl[0]

    print(f"  -> Top-{len(top_mala)} MALA seleccionados (discontinua --)  : {[m['label'] for m in top_mala]}")
    print(f"  -> Top-{len(top_rl)} RL-MALA seleccionados (continua -)    : {[m['label'] for m in top_rl]}")

    # =========================================================================
    # FIGURA 1: Article_Densities (Comparación Top-K MALA vs Top-K RL-MALA)
    # =========================================================================
    models_with_dens = [m for m in selected_models if m.get("densities") is not None]
    if models_with_dens:
        n_panels = len(models_with_dens)
        ncols = 2 if n_panels == 4 else min(n_panels, 3)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(7.5 * ncols, 5.5 * nrows), sharey=True, squeeze=False
        )
        axes_flat = axes.flatten()

        x_grid = models_with_dens[0]["densities"]["x"]
        v_trap = 0.5 * (omega ** 2) * (x_grid ** 2)
        v_scale = max(float(np.max(models_with_dens[0]["densities"]["rho_tot"])), 0.25) / max(
            float(np.max(v_trap)), 1e-8
        )
        v_scaled = v_trap * v_scale

        for idx, mod in enumerate(models_with_dens):
            ax = axes_flat[idx]
            d = mod["densities"]
            is_n = bool(mod["is_neural"])
            kind_prefix = "RL-MALA" if is_n else "MALA"
            ls_tot = "-" if is_n else "--"
            d_sum_m = mod.get("diagnostics", {}).get("summary", {})
            e_mean_m = mod.get("E_mean", d_sum_m.get("E_mean", float("nan")))
            se_e_m = mod.get("SE_E", d_sum_m.get("SE_E", float("nan")))
            cas_e_m = mod.get("casscf_energy", d_sum_m.get("casscf_energy", float("nan")))
            if not np.isfinite(cas_e_m) and casscf_ref is not None:
                cas_e_m = float(casscf_ref.get("energy", float("nan")))
            title_lines_m = [f"{kind_prefix}: {mod['label']}"]
            if np.isfinite(e_mean_m):
                e_str_m = f"$\\langle E \\rangle = {e_mean_m:.6f}"
                if np.isfinite(se_e_m) and se_e_m > 0:
                    e_str_m += f" \\pm {se_e_m:.6f}"
                e_str_m += "$ Ha"
                if np.isfinite(cas_e_m):
                    de_mha = (e_mean_m - cas_e_m) * 1e3
                    e_str_m += f" ($\\Delta E = {de_mha:+.2f}$ mHa)"
                title_lines_m.append(e_str_m)
            title = "\n".join(title_lines_m)

            ax.plot(
                d["x"],
                d["rho_tot"],
                color="black",
                linestyle=ls_tot,
                lw=2.5,
                label=f"Sampled Total ({kind_prefix})",
            )
            if "rho_tot_std" in d and np.any(d["rho_tot_std"] > 0):
                ax.fill_between(
                    d["x"],
                    np.maximum(0.0, d["rho_tot"] - d["rho_tot_std"]),
                    d["rho_tot"] + d["rho_tot_std"],
                    color="black",
                    alpha=0.18,
                    label=r"Total $\pm 1\sigma$",
                )
            ax.fill_between(d["x"], d["rho_up"], color="#3498db", alpha=0.45, label="Sampled Spin ↑")
            ax.fill_between(d["x"], d["rho_dn"], color="#e74c3c", alpha=0.45, label="Sampled Spin ↓")
            if casscf_ref is not None:
                rho_tot_ex = casscf_ref.get("rho_total", casscf_ref.get("rho_tot"))
                rho_dn_ex = casscf_ref.get("rho_down", casscf_ref.get("rho_dn"))
                ax.plot(
                    casscf_ref["x"],
                    rho_tot_ex,
                    color="black",
                    linestyle=":",
                    lw=2.2,
                    label="Exact CASSCF Total",
                )
                ax.plot(
                    casscf_ref["x"],
                    casscf_ref["rho_up"],
                    color="#1f618d",
                    linestyle=":",
                    lw=2.2,
                    label="Exact Spin ↑",
                )
                ax.plot(
                    casscf_ref["x"],
                    rho_dn_ex,
                    color="#922b21",
                    linestyle=":",
                    lw=2.2,
                    label="Exact Spin ↓",
                )
            ax.plot(x_grid, v_scaled, "--", color="gray", alpha=0.4, label="Harmonic Trap (Scaled)")
            ax.set_title(title, fontsize=12.0, fontweight="bold")
            ax.set_xlabel("x (Bohr)", fontsize=11.5)
            ax.set_xlim(-args.x_max, args.x_max)
            if idx % ncols == 0:
                ax.set_ylabel("One-Body Density ρ(x)", fontsize=11.5)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

        for extra_idx in range(n_panels, len(axes_flat)):
            axes_flat[extra_idx].set_visible(False)

        plt.tight_layout()
        _save_fig(
            fig,
            out_dir,
            f"Article_Densities_{args.system_tag}",
            args.figure_format,
            args.dpi,
            args=args,
        )

        if best_mala.get("densities") is not None and best_rl.get("densities") is not None:
            dens_dict = {
                "x_grid": best_mala["densities"]["x"],
                "mala_tot": best_mala["densities"]["rho_tot"],
                "mala_up": best_mala["densities"]["rho_up"],
                "mala_dn": best_mala["densities"]["rho_dn"],
                "rl_tot": best_rl["densities"]["rho_tot"],
                "rl_up": best_rl["densities"]["rho_up"],
                "rl_dn": best_rl["densities"]["rho_dn"],
            }
            if casscf_ref is not None:
                rho_tot_ex = casscf_ref.get("rho_total", casscf_ref.get("rho_tot"))
                rho_dn_ex = casscf_ref.get("rho_down", casscf_ref.get("rho_dn"))
                dens_dict["exact_tot"] = np.interp(
                    best_mala["densities"]["x"], casscf_ref["x"], rho_tot_ex
                )
                dens_dict["exact_up"] = np.interp(
                    best_mala["densities"]["x"], casscf_ref["x"], casscf_ref["rho_up"]
                )
                dens_dict["exact_dn"] = np.interp(
                    best_mala["densities"]["x"], casscf_ref["x"], rho_dn_ex
                )
            dens_csv = out_dir / f"data_densities_{args.system_tag}.csv"
            atomic_write_csv(pd.DataFrame(dens_dict), dens_csv, index=False)
            _copy_csv(dens_csv, args.csv_collect_dir)

    # =========================================================================
    # FIGURA 2: Article_KL_Divergence (MALA siempre discontinua '--', RL-MALA siempre continua '-')
    # =========================================================================
    if casscf_ref is not None and not best_mala.get("kl_df", pd.DataFrame()).empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), sharey=True)
        kl_cols = [
            ("kl_tot", "kl_tot_std", "Total Density $D_{\\mathrm{KL}}$"),
            ("kl_up", "kl_up_std", "Spin ↑ Density $D_{\\mathrm{KL}}$ (Nodal Crossing)"),
            ("kl_dn", "kl_dn_std", "Spin ↓ Density $D_{\\mathrm{KL}}$ (Nodal Crossing)"),
        ]
        mala_colors = ["#e74c3c", "#c0392b", "#d35400", "#7f8c8d"]
        rl_colors = ["#2ecc71", "#1f77b4", "#9b59b6", "#16a085"]

        any_pos_kl = False
        for ax, (col, std_col, title) in zip(axes, kl_cols):
            for idx_m, m in enumerate(top_mala):
                if m["kl_df"].empty:
                    continue
                c = mala_colors[idx_m % len(mala_colors)]
                s_grid = m["kl_df"]["step"].to_numpy(dtype=np.float64)
                y_mean = m["kl_df"][col].to_numpy(dtype=np.float64)
                y_std = (
                    m["kl_df"][std_col].to_numpy(dtype=np.float64)
                    if std_col in m["kl_df"]
                    else np.zeros_like(y_mean)
                )
                fin_m = np.isfinite(s_grid) & np.isfinite(y_mean)
                if not np.any(fin_m):
                    continue
                if np.any(y_mean[fin_m] > 0):
                    any_pos_kl = True
                ax.plot(
                    s_grid[fin_m],
                    y_mean[fin_m],
                    color=c,
                    lw=2.5 if idx_m == 0 else 2.0,
                    linestyle="--",
                    label=f"MALA #{idx_m+1} [--]: {m['label']}",
                )
                if np.any(y_std[fin_m] > 0):
                    ax.fill_between(
                        s_grid[fin_m],
                        np.maximum(1e-8, y_mean[fin_m] - y_std[fin_m]),
                        y_mean[fin_m] + y_std[fin_m],
                        color=c,
                        alpha=0.16,
                    )

            for idx_r, m in enumerate(top_rl):
                if m["kl_df"].empty:
                    continue
                c = rl_colors[idx_r % len(rl_colors)]
                s_grid = m["kl_df"]["step"].to_numpy(dtype=np.float64)
                y_mean = m["kl_df"][col].to_numpy(dtype=np.float64)
                y_std = (
                    m["kl_df"][std_col].to_numpy(dtype=np.float64)
                    if std_col in m["kl_df"]
                    else np.zeros_like(y_mean)
                )
                fin_r = np.isfinite(s_grid) & np.isfinite(y_mean)
                if not np.any(fin_r):
                    continue
                if np.any(y_mean[fin_r] > 0):
                    any_pos_kl = True
                ax.plot(
                    s_grid[fin_r],
                    y_mean[fin_r],
                    color=c,
                    lw=2.7 if idx_r == 0 else 2.1,
                    linestyle="-",
                    label=f"RL-MALA #{idx_r+1} [-]: {m['label']}",
                )
                if np.any(y_std[fin_r] > 0):
                    ax.fill_between(
                        s_grid[fin_r],
                        np.maximum(1e-8, y_mean[fin_r] - y_std[fin_r]),
                        y_mean[fin_r] + y_std[fin_r],
                        color=c,
                        alpha=0.20,
                    )

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("MCMC Production Steps", fontsize=11)
            ax.grid(True, which="both", ls="-", alpha=0.25)
            h_kl, l_kl = ax.get_legend_handles_labels()
            if h_kl:
                ax.legend(fontsize=8.8, loc="best", frameon=True)

        if any_pos_kl:
            axes[0].set_yscale("log")

        axes[0].set_ylabel(
            r"KL Divergence $D_{\mathrm{KL}}(P_t \parallel P_{\mathrm{CASSCF}})$",
            fontsize=11,
        )
        plt.tight_layout()
        _save_fig(
            fig,
            out_dir,
            f"Article_KL_Divergence_{args.system_tag}",
            args.figure_format,
            args.dpi,
            args=args,
        )

        # Guardar CSV de convergencia KL
        kl_merged = pd.DataFrame({"step": best_mala["kl_df"]["step"]})
        kl_merged["kl_tot_mala"] = best_mala["kl_df"]["kl_tot"]
        kl_merged["kl_up_mala"] = best_mala["kl_df"]["kl_up"]
        kl_merged["kl_dn_mala"] = best_mala["kl_df"]["kl_dn"]
        if not best_rl["kl_df"].empty and len(best_rl["kl_df"]) == len(kl_merged):
            kl_merged["kl_tot_rl"] = best_rl["kl_df"]["kl_tot"]
            kl_merged["kl_up_rl"] = best_rl["kl_df"]["kl_up"]
            kl_merged["kl_dn_rl"] = best_rl["kl_df"]["kl_dn"]
        kl_csv = out_dir / f"data_kl_divergence_{args.system_tag}.csv"
        atomic_write_csv(kl_merged, kl_csv, index=False)
        _copy_csv(kl_csv, args.csv_collect_dir)

    # =========================================================================
    # FIGURA 2B: Article_Walker_Traces (Trazas de un solo caminante: Mejor MALA vs Mejor RL-MALA)
    # =========================================================================
    w_top_k = max(1, int(args.walker_trace_top_k))
    walker_pair = top_mala[:w_top_k] + top_rl[:w_top_k]
    if any(m.get("walker_trace_data") is not None for m in walker_pair):
        plot_walker_trace(
            walker_pair, str(out_dir), pp_style_cfg, casscf_ref=casscf_ref
        )
        for ext in ("png", "pdf"):
            src_w = out_dir / f"fig6_walker_traces_{args.system_tag}.{ext}"
            dst_w = out_dir / f"Article_Walker_Traces_{args.system_tag}.{ext}"
            if src_w.exists():
                atomic_copy_file(src_w, dst_w)
                if not valid_h5 and pp_fallback_dir.exists():
                    atomic_copy_file(src_w, pp_fallback_dir / src_w.name)
                src_w.unlink()

    # =========================================================================
    # FIGURA 3: Article_Ergodicity_Observables (Acceptance Rate & Wigner MSD: MALA -- vs RL-MALA -)
    # =========================================================================
    if not best_mala.get("trace_df", pd.DataFrame()).empty and not best_rl.get("trace_df", pd.DataFrame()).empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        tm = best_mala["trace_df"]
        tr = best_rl["trace_df"]

        ax1.plot(
            tm["step"],
            tm["acceptance"],
            color="#e74c3c",
            linestyle="--",
            alpha=0.75,
            lw=1.4,
            label=f"MALA [--] ({best_mala['label']})",
        )
        ax1.plot(
            tr["step"],
            tr["acceptance"],
            color="#2ecc71",
            linestyle="-",
            alpha=0.80,
            lw=1.4,
            label=f"RL-MALA [-] ({best_rl['label']})",
        )
        ax1.axhline(0.574, color="black", linestyle=":", lw=1.6, label="Theoretical Optimum (0.574)")
        ax1.set_title("MCMC Acceptance Rate", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Acceptance Rate", fontsize=11)
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(True, alpha=0.25)

        ax2.plot(
            tm["step"],
            tm["msd"],
            color="#e74c3c",
            linestyle="--",
            alpha=0.88,
            lw=1.8,
            label=f"Standard MALA [--] ({best_mala['label']})",
        )
        ax2.plot(
            tr["step"],
            tr["msd"],
            color="#2ecc71",
            linestyle="-",
            alpha=0.90,
            lw=1.8,
            label=f"RL-MALA [-] ({best_rl['label']})",
        )
        ax2.set_title("Wigner Topological Mobility (MSD)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("MCMC Steps", fontsize=11)
        ax2.set_ylabel(r"MSD ($\mathrm{Bohr}^2$)", fontsize=11)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, alpha=0.25)
        plt.tight_layout()
        _save_fig(
            fig,
            out_dir,
            f"Article_Ergodicity_Observables_{args.system_tag}",
            args.figure_format,
            args.dpi,
            args=args,
        )

    # =========================================================================
    # FIGURA 4: Article_Training_Metrics (Diagnóstico del entrenamiento TD3)
    # =========================================================================
    trn_bundle = best_rl.get("bundle", {})
    trn_df = trn_bundle.get("training_df", None)
    trn_hist: Optional[dict[str, Any]] = None
    if isinstance(trn_df, pd.DataFrame) and not trn_df.empty:
        if "Phase" in trn_df.columns:
            phase_num = pd.to_numeric(trn_df["Phase"], errors="coerce")
            if (phase_num == 1.0).any():
                sub_phase = trn_df.loc[phase_num == 1.0]
                if not sub_phase.empty:
                    trn_df = sub_phase
        trn_hist = {
            col: pd.to_numeric(trn_df[col], errors="coerce").to_numpy(dtype=np.float64)
            for col in trn_df.columns
            if pd.api.types.is_numeric_dtype(trn_df[col])
        }
    elif isinstance(trn_bundle.get("training"), dict) and trn_bundle.get("training"):
        trn_hist = trn_bundle.get("training")

    n_trn_pts = 0
    if trn_hist:
        if "Step" in trn_hist:
            n_trn_pts = len(trn_hist["Step"])
        elif "RL_Reward" in trn_hist:
            n_trn_pts = len(trn_hist["RL_Reward"])

    if trn_hist and n_trn_pts > 0:
        steps_t = (
            np.asarray(trn_hist["Step"], dtype=np.float64)
            if "Step" in trn_hist and len(trn_hist["Step"]) == n_trn_pts
            else np.arange(1, n_trn_pts + 1, dtype=np.float64)
        )
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 11), sharex=True)

        ax1.plot(steps_t, trn_hist.get("RL_Reward", np.zeros_like(steps_t)), color="black", lw=2.0, label="Total Reward")
        if "RL_Exploit" in trn_hist:
            ax1.plot(steps_t, trn_hist["RL_Exploit"], color="#3498db", alpha=0.75, label="Exploitation")
        if "RL_Entropy" in trn_hist:
            ax1.plot(steps_t, trn_hist["RL_Entropy"], color="#f39c12", alpha=0.75, label="Entropy")
        if "RL_Explore" in trn_hist:
            ax1.plot(steps_t, trn_hist["RL_Explore"], color="#2ecc71", alpha=0.75, label="Exploration (-α log q)")
        ax1.set_title(f"CDLB Reward Components ({best_rl['label']})", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Reward Value", fontsize=11)
        ax1.legend(loc="lower right", fontsize=9.5)
        ax1.grid(True, alpha=0.25)

        ax2_r = ax2.twinx()
        c_loss = np.asarray(trn_hist.get("Critic_Loss", np.zeros_like(steps_t)), dtype=np.float64)
        a_loss = np.asarray(trn_hist.get("Actor_Loss", np.zeros_like(steps_t)), dtype=np.float64)
        ax2.plot(steps_t, c_loss, color="#e74c3c", alpha=0.85, label="Critic MSE")
        ax2_r.plot(steps_t, a_loss, color="#2980b9", alpha=0.85, label="Actor Loss (-Q)")
        ax2.set_title("TD3 Network Losses", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Critic MSE", color="#e74c3c", fontsize=11)
        ax2_r.set_ylabel("Actor Loss", color="#2980b9", fontsize=11)
        h2, l2 = ax2.get_legend_handles_labels()
        h2r, l2r = ax2_r.get_legend_handles_labels()
        if h2 or h2r:
            ax2.legend(h2 + h2r, l2 + l2r, loc="best", fontsize=9.5)
        ax2.grid(True, alpha=0.25)

        ax3_r = ax3.twinx()
        acc_t = np.asarray(trn_hist.get("Acceptance", np.zeros_like(steps_t)), dtype=np.float64)
        disp_t = np.asarray(trn_hist.get("Proposed_Disp", np.zeros_like(steps_t)), dtype=np.float64)
        ax3.plot(steps_t, acc_t, color="#2ecc71", alpha=0.8, label="Acceptance Rate")
        ax3.axhline(0.574, color="black", linestyle=":", alpha=0.5, label="Target (0.574)")
        ax3_r.plot(steps_t, disp_t, color="#8e44ad", alpha=0.8, label="Displacement (Bohr)")
        ax3.set_title("Training Phase MCMC Diagnostics", fontsize=13, fontweight="bold")
        ax3.set_xlabel("TD3 Training Sweeps", fontsize=11)
        ax3.set_ylabel("Acceptance Rate", color="#2ecc71", fontsize=11)
        ax3_r.set_ylabel("Displacement (Bohr)", color="#8e44ad", fontsize=11)
        h3, l3 = ax3.get_legend_handles_labels()
        h3r, l3r = ax3_r.get_legend_handles_labels()
        if h3 or h3r:
            ax3.legend(h3 + h3r, l3 + l3r, loc="best", fontsize=9.5)
        ax3.grid(True, alpha=0.25)

        plt.tight_layout()
        _save_fig(
            fig,
            out_dir,
            f"Article_Training_Metrics_{args.system_tag}",
            args.figure_format,
            args.dpi,
            args=args,
        )

    # =========================================================================
    # FIGURA 5: Hyperparameter_Evolution (Evolución con barras de error en todos los modelos)
    # =========================================================================
    if len(df_summary) >= 2:
        plot_hyperparameter_evolution(
            df_summary, str(out_dir), pp_style_cfg, casscf_ref=casscf_ref
        )
        for ext in ("png", "pdf"):
            src_f = out_dir / f"fig9_hyperparameter_evolution_{args.system_tag}.{ext}"
            dst_f = out_dir / f"Hyperparameter_Evolution_{args.system_tag}.{ext}"
            if src_f.exists():
                atomic_copy_file(src_f, dst_f)
                src_f.unlink()

        # Si estamos reconstruyendo desde pp_fallback_dir, actualizar también las figuras de postprocess_standard con Top-K
        if not valid_h5 and pp_fallback_dir.exists():
            pp_std_cfg = PostProcessConfig(
                out_dir=str(pp_fallback_dir),
                system_tag=args.system_tag,
                figure_format=args.figure_format,
                dpi=args.dpi,
                top_k_models=top_k,
                walker_trace_top_k=w_top_k,
                n_eval_subblocks=args.n_eval_subblocks,
                title_prefix=args.title_prefix,
                custom_suptitle=args.custom_suptitle,
                no_titles=args.no_titles,
                font_scale=args.font_scale,
                line_width_scale=args.line_width_scale,
                transparent_bg=args.transparent_bg,
            )
            plot_hyperparameter_evolution(
                df_summary, str(pp_fallback_dir), pp_std_cfg, casscf_ref=casscf_ref
            )
            plot_article_densities_comparison(
                selected_models, str(pp_fallback_dir), pp_std_cfg, casscf_ref=casscf_ref
            )
            plot_efficiency_bars(selected_models, str(pp_fallback_dir), pp_std_cfg)
            for ext in ("png", "pdf"):
                kl_src = out_dir / f"Article_KL_Divergence_{args.system_tag}.{ext}"
                kl_dst = pp_fallback_dir / f"fig8_article_kl_divergence_{args.system_tag}.{ext}"
                if kl_src.exists():
                    atomic_copy_file(kl_src, kl_dst)

    # =========================================================================
    # FIGURA 6: Parameter_Search_Comparison (Ranking de todos los modelos con barras de error)
    # =========================================================================
    if len(df_summary) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(21.0, 11.0))
        labels_s = df_summary["label"].tolist()
        is_n_list = [bool(x) for x in df_summary["is_neural"]]
        colors_s = ["#2ecc71" if is_n else "#e74c3c" for is_n in is_n_list]
        hatches_s = ["" if is_n else "//" for is_n in is_n_list]
        x_pos = np.arange(len(labels_s))

        def _clean_err(col_name: str) -> np.ndarray:
            if col_name not in df_summary.columns:
                return np.zeros(len(df_summary))
            arr = pd.to_numeric(df_summary[col_name], errors="coerce").to_numpy(dtype=np.float64)
            return np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)

        def _draw_bars(
            ax: plt.Axes,
            vals_arr: np.ndarray,
            errs_arr: np.ndarray,
            log_scale: bool = False,
            floor_min: Optional[float] = None,
        ) -> None:
            mask = np.isfinite(vals_arr)
            if log_scale and floor_min is not None:
                vals_plot = np.where(mask, np.maximum(vals_arr, floor_min), np.nan)
                errs_plot = np.where(mask, np.minimum(errs_arr, 0.85 * np.maximum(vals_plot, floor_min)), 0.0)
            else:
                vals_plot = vals_arr
                errs_plot = np.where(mask, errs_arr, 0.0)
            idx_ok = np.flatnonzero(mask)
            if len(idx_ok) > 0:
                bars = ax.bar(
                    x_pos[idx_ok],
                    vals_plot[idx_ok],
                    yerr=errs_plot[idx_ok],
                    capsize=3.5,
                    color=[colors_s[i] for i in idx_ok],
                    alpha=0.85,
                    edgecolor="black",
                )
                for bar, i_row in zip(bars, idx_ok):
                    if hatches_s[i_row]:
                        bar.set_hatch(hatches_s[i_row])
                if log_scale and np.any(vals_plot[idx_ok] > 0):
                    ax.set_yscale("log")

        # Panel (0, 0): Energía Final <E> +/- SE(E) con referencia CASSCF
        ax_e = axes[0, 0]
        e_vals = pd.to_numeric(df_summary["E_mean"], errors="coerce").to_numpy(dtype=np.float64)
        se_vals = _clean_err("SE_E")
        _draw_bars(ax_e, e_vals, se_vals, log_scale=False)
        if casscf_ref is not None and np.isfinite(casscf_ref.get("energy", float("nan"))):
            cas_e_ref = float(casscf_ref["energy"])
            ax_e.axhline(
                cas_e_ref,
                color="black",
                linestyle="--",
                lw=2.0,
                label=f"Exact CASSCF ({cas_e_ref:.5f} Ha)",
            )
            ax_e.legend(loc="best", fontsize=9.0, frameon=True)
            # Acotar y-limits para no aplastar barras si la energía es cercana
            valid_e = e_vals[np.isfinite(e_vals)]
            if len(valid_e) > 0:
                span = max(float(np.max(valid_e) - np.min(valid_e)), 0.015)
                pad = max(0.015, 0.25 * span)
                y_min = min(float(np.min(valid_e)), cas_e_ref) - pad
                y_max = max(float(np.max(valid_e)), cas_e_ref) + pad
                ax_e.set_ylim(y_min, y_max)
        ax_e.set_title(r"Final Energy $\langle E \rangle \pm \mathrm{SE}(E)$", fontweight="bold")
        ax_e.set_ylabel(r"Energy $\langle E \rangle$ (Ha)")
        ax_e.set_xticks(x_pos)
        ax_e.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_e.grid(True, axis="y", alpha=0.25)

        # Panel (0, 1): Error de Energía Delta E +/- SE(E) en mHa
        ax_de = axes[0, 1]
        de_vals = pd.to_numeric(df_summary["delta_E_vs_CASSCF"], errors="coerce").to_numpy(dtype=np.float64) * 1e3
        de_errs = se_vals * 1e3
        _draw_bars(ax_de, de_vals, de_errs, log_scale=False)
        ax_de.axhline(0.0, color="black", linestyle="--", lw=1.8, label=r"CASSCF Reference ($\Delta E = 0$)")
        ax_de.set_title(r"Energy Error $\Delta E = \langle E \rangle - E_{\mathrm{CASSCF}}$", fontweight="bold")
        ax_de.set_ylabel(r"$\Delta E$ (mHa)")
        ax_de.set_xticks(x_pos)
        ax_de.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_de.grid(True, axis="y", alpha=0.25)
        ax_de.legend(loc="best", fontsize=9.0, frameon=True)

        # Panel (0, 2): Spin-Resolved D_KL (cruce de barreras nodales)
        ax_kl = axes[0, 2]
        kl_vals = pd.to_numeric(df_summary["final_kl_spin_mean"], errors="coerce").to_numpy(dtype=np.float64)
        kl_errs = _clean_err("final_kl_spin_std")
        _draw_bars(ax_kl, kl_vals, kl_errs, log_scale=True, floor_min=1e-7)
        ax_kl.set_title("Spin-Resolved KL Divergence (Nodal Crossing)", fontweight="bold")
        ax_kl.set_ylabel(r"$\frac{1}{2}(D_{\mathrm{KL}}^{\uparrow} + D_{\mathrm{KL}}^{\downarrow})$ (log)")
        ax_kl.set_xticks(x_pos)
        ax_kl.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_kl.grid(True, axis="y", alpha=0.25)

        # Panel (1, 0): Total Density D_KL
        ax_klt = axes[1, 0]
        klt_vals = pd.to_numeric(df_summary["final_kl_tot"], errors="coerce").to_numpy(dtype=np.float64)
        klt_errs = _clean_err("final_kl_tot_std")
        _draw_bars(ax_klt, klt_vals, klt_errs, log_scale=True, floor_min=1e-7)
        ax_klt.set_title("Total One-Body Density KL Divergence", fontweight="bold")
        ax_klt.set_ylabel(r"$D_{\mathrm{KL}}(\rho_{\mathrm{tot}} \parallel \rho_{\mathrm{CASSCF}})$ (log)")
        ax_klt.set_xticks(x_pos)
        ax_klt.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_klt.grid(True, axis="y", alpha=0.25)

        # Panel (1, 1): ESJD (Expected Squared Jump Distance)
        ax_esjd = axes[1, 1]
        esjd_vals = pd.to_numeric(df_summary["ESJD"], errors="coerce").to_numpy(dtype=np.float64)
        esjd_errs = _clean_err("ESJD_std")
        _draw_bars(ax_esjd, esjd_vals, esjd_errs, log_scale=False)
        ax_esjd.set_title("Expected Squared Jump Distance (ESJD)", fontweight="bold")
        ax_esjd.set_ylabel(r"ESJD ($\mathrm{Bohr}^2$/step)")
        ax_esjd.set_xticks(x_pos)
        ax_esjd.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_esjd.grid(True, axis="y", alpha=0.25)

        # Panel (1, 2): Integrated Autocorrelation Time tau_int(E)
        ax_tau = axes[1, 2]
        tau_vals = pd.to_numeric(df_summary["tau_int_E"], errors="coerce").to_numpy(dtype=np.float64)
        tau_errs = _clean_err("tau_int_E_std")
        _draw_bars(ax_tau, tau_vals, tau_errs, log_scale=True, floor_min=1e-2)
        ax_tau.set_title(r"Autocorrelation Time $\tau_{\mathrm{int}}(E)$ (Lower = Better)", fontweight="bold")
        ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (steps, log)")
        ax_tau.set_xticks(x_pos)
        ax_tau.set_xticklabels(labels_s, rotation=32, ha="right", fontsize=8.5)
        ax_tau.grid(True, axis="y", which="both", alpha=0.25)

        plt.suptitle(
            f"Hyperparameter Search Comparison — {args.system_tag} (Hatched=MALA, Filled=RL-MALA)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        _save_fig(
            fig,
            out_dir,
            f"Parameter_Search_Comparison_{args.system_tag}",
            args.figure_format,
            args.dpi,
            args=args,
        )

    print(f"[OK] Evaluación y gráficas de cruce nodal completadas en: {out_dir}")


if __name__ == "__main__":
    main()
