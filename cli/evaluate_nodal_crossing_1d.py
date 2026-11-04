#!/usr/bin/env python3
"""
===============================================================================
RL-VMC 1D: EVALUADOR DE CRUCE DE BARRERAS NODALES Y GENERADOR DE FIGURAS
===============================================================================
Este script extiende el análisis de `Article_plots_11/evaluar_wigner_casscf.py`
al ecosistema modular HDF5 (`rl_vmc`).

Funciones principales:
  1. Carga uno o varios archivos `.h5` (MALA clásico y variantes RL-MALA TD3).
  2. Calcula (o carga) la densidad exacta analítica de 1 cuerpo del estado
     CASSCF de referencia (`rho_tot(x)`, `rho_up(x)`, `rho_dn(x)`).
  3. Evalúa la evolución acumulada paso a paso de la Divergencia de
     Kullback-Leibler resuelta por espín:
       - D_KL(P_tot || P_exact_tot)
       - D_KL(P_up  || P_exact_up)   <-- Diagnóstico directo de cruce nodal
       - D_KL(P_dn  || P_exact_dn)   <-- Diagnóstico directo de cruce nodal
  4. Calcula la Movilidad Topológica de Wigner (MSD respecto a la configuración
     inicial) y la tasa de aceptación respecto al óptimo de Langevin (0.574).
  5. Genera las 5 figuras de calidad de artículo (en PNG y PDF) y exporta
     todos los datos tabulares a CSV en `--out_dir` y `--csv_collect_dir`:
       - Article_Densities.{png,pdf}
       - Article_KL_Divergence.{png,pdf}
       - Article_Ergodicity_Observables.{png,pdf}
       - Article_Training_Metrics.{png,pdf}
       - Parameter_Search_Comparison.{png,pdf}
===============================================================================
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vmc.io.h5_bundle import load_run_bundle_h5
from rl_vmc.postprocess.casscf_baseline import (
    compute_exact_casscf_1body_density,
    discover_casscf_baselines,
    find_matching_casscf_baseline,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluador de Cruce de Barreras Nodales, Divergencia KL Resuelta por Espín "
            "y Movilidad Topológica de Wigner (compatible con Article_plots_11)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_in = parser.add_argument_group("1. Entradas y Directorios de Salida")
    g_in.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Lista de archivos run_bundle.h5 a comparar (incluir al menos un MALA y uno o más RL-MALA).",
    )
    g_in.add_argument(
        "--labels",
        nargs="+",
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
        help="Prefijo identificador del sistema físico (ej. N4_g4.0_w0.1).",
    )

    g_grid = parser.add_argument_group("2. Parámetros de Mallado Espacial y Estadística")
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
        type=str,
        default="both",
        choices=["png", "pdf", "both"],
        help="Formato de exportación gráfica.",
    )
    g_grid.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolución en puntos por pulgada (DPI) para archivos PNG.",
    )
    return parser


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str, fmt: str, dpi: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if fmt in ("png", "both"):
        p_png = out_dir / f"{stem}.png"
        fig.savefig(p_png, dpi=dpi, bbox_inches="tight")
        saved.append(p_png)
    if fmt in ("pdf", "both"):
        p_pdf = out_dir / f"{stem}.pdf"
        fig.savefig(p_pdf, dpi=dpi, bbox_inches="tight")
        saved.append(p_pdf)
    plt.close(fig)
    return saved


def _copy_csv(csv_path: Path, collect_dir: Optional[str]) -> None:
    if not collect_dir:
        return
    cdir = Path(collect_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_path, cdir / csv_path.name)


def _extract_walkers_4d(bundle: dict) -> Optional[np.ndarray]:
    """
    Devuelve las posiciones de producción con shape (n_steps, n_walkers, n_elec, 1).
    Si hay múltiples cadenas (n_chains, n_steps, n_walkers, n_elec, 1), concatena
    a lo largo del eje de caminantes para preservar la línea temporal paso a paso.
    """
    prod = bundle.get("production", {})
    w = prod.get("walkers", None)
    if w is None:
        return None
    w_arr = np.asarray(w, dtype=np.float64)
    if w_arr.size == 0:
        return None
    if w_arr.ndim == 5:
        # (n_chains, n_steps, n_walkers, n_elec, 1) -> (n_steps, n_chains*n_walkers, n_elec, 1)
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
) -> pd.DataFrame:
    """
    Calcula la evolución acumulada D_KL(P_t || P_exact) para la densidad total,
    espín arriba y espín abajo sin desbordar la memoria RAM (acumulación incremental de histogramas).
    """
    n_steps = walkers_4d.shape[0]
    n_elec = walkers_4d.shape[2]
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

    counts_tot = np.zeros(n_bins, dtype=np.float64)
    counts_up = np.zeros(n_bins, dtype=np.float64)
    counts_dn = np.zeros(n_bins, dtype=np.float64)

    stride = max(1, int(stride))
    eval_indices = list(range(stride, n_steps + 1, stride))
    if not eval_indices or eval_indices[-1] != n_steps:
        eval_indices.append(n_steps)

    rows = []
    prev_idx = 0
    for idx in eval_indices:
        chunk = walkers_4d[prev_idx:idx, :, :, 0]
        prev_idx = idx

        c_tot, _ = np.histogram(chunk.reshape(-1), bins=bins)
        c_up, _ = np.histogram(chunk[:, :, :n_up].reshape(-1), bins=bins)
        c_dn, _ = np.histogram(chunk[:, :, n_up:].reshape(-1), bins=bins)

        counts_tot += c_tot
        counts_up += c_up
        counts_dn += c_dn

        tot_sum = np.sum(counts_tot) * dx
        up_sum = np.sum(counts_up) * dx
        dn_sum = np.sum(counts_dn) * dx
        if tot_sum <= 0 or up_sum <= 0 or dn_sum <= 0:
            continue

        q_tot = np.clip(counts_tot / tot_sum, eps, None)
        q_up = np.clip(counts_up / up_sum, eps, None)
        q_dn = np.clip(counts_dn / dn_sum, eps, None)

        kl_tot = float(np.sum(p_tot_ex * np.log(p_tot_ex / q_tot) * dx))
        kl_up = float(np.sum(p_up_ex * np.log(p_up_ex / q_up) * dx))
        kl_dn = float(np.sum(p_dn_ex * np.log(p_dn_ex / q_dn) * dx))

        rows.append(
            {
                "step": int(idx),
                "kl_tot": max(0.0, kl_tot),
                "kl_up": max(0.0, kl_up),
                "kl_dn": max(0.0, kl_dn),
                "kl_spin_mean": max(0.0, 0.5 * (kl_up + kl_dn)),
            }
        )
    return pd.DataFrame(rows)


def _compute_densities_profile(
    walkers_4d: np.ndarray,
    n_up: int,
    n_dn: int,
    x_max: float,
    n_bins: int,
) -> dict[str, np.ndarray]:
    n_elec = n_up + n_dn
    bins = np.linspace(-x_max, x_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    # Descartamos el primer 10% si no hubo burn-in previo para representar el estado estacionario
    start_idx = max(0, int(0.10 * walkers_4d.shape[0]))
    w_steady = walkers_4d[start_idx:, :, :, 0]

    h_tot, _ = np.histogram(w_steady.reshape(-1), bins=bins, density=True)
    h_up, _ = np.histogram(w_steady[:, :, :n_up].reshape(-1), bins=bins, density=True)
    h_dn, _ = np.histogram(w_steady[:, :, n_up:].reshape(-1), bins=bins, density=True)

    return {
        "x": centers,
        "rho_tot": h_tot * n_elec,
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
        acc_sub = acc_1d[indices] if acc_1d.shape[0] >= n_steps else np.full_like(msd, float(np.mean(acc_arr)))
    else:
        acc_sub = np.full_like(msd, np.nan)

    return pd.DataFrame(
        {
            "step": indices,
            "acceptance": acc_sub,
            "msd": msd,
        }
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar todos los bundles .h5
    models = []
    for i, h5_str in enumerate(args.inputs):
        p = Path(h5_str)
        if not p.exists():
            print(f"[Aviso] No existe el archivo HDF5: {p}. Saltando.")
            continue
        bundle = load_run_bundle_h5(p, load_walkers=True)
        cfg = bundle.get("config", {})
        sys_cfg = cfg.get("system", {})
        prop_cfg = cfg.get("proposal", {})
        rl_cfg = cfg.get("rl", {})
        trn_cfg = cfg.get("train", {})

        label = (
            args.labels[i]
            if (args.labels and i < len(args.labels))
            else cfg.get("io", {}).get("run_name") or p.parent.name
        )
        ptype = str(prop_cfg.get("proposal_type", "mala"))
        is_neural = ptype in ("neural_deepset", "neural_flat", "deepset", "flat_mlp")

        # Cargar summary.json si existe junto al .h5 para métricas de energía y tau_int
        summary_json_path = p.parent / "summary.json"
        obs_metrics = {}
        if summary_json_path.exists():
            import json
            try:
                with summary_json_path.open("r", encoding="utf-8") as f:
                    sdata = json.load(f)
                obs_metrics = sdata.get("observables", {})
            except Exception:
                obs_metrics = {}

        models.append(
            {
                "path": p,
                "label": str(label),
                "bundle": bundle,
                "is_neural": is_neural,
                "proposal_type": ptype,
                "tau": float(prop_cfg.get("tau", 0.0)),
                "actor_amplitude": float(prop_cfg.get("actor_amplitude", 0.0)) if is_neural else 0.0,
                "sigma_action_scale": float(prop_cfg.get("sigma_action_scale", 0.0)) if is_neural else 0.0,
                "penalty_factor": float(rl_cfg.get("penalty_factor", 0.0)) if is_neural else 0.0,
                "lr_actor": float(rl_cfg.get("lr_actor", 0.0)) if is_neural else 0.0,
                "train_mode": str(trn_cfg.get("mode", "eval_only")),
                "n_elec": int(sys_cfg.get("n_elec", 4)),
                "n_up": int(sys_cfg.get("n_up", 2)),
                "n_dn": int(sys_cfg.get("n_dn", 2)),
                "omega": float(sys_cfg.get("omega", 0.1)),
                "g": float(sys_cfg.get("interaction_strength", 4.0)),
                "softening": float(sys_cfg.get("softening", 0.1)),
                "obs_metrics": obs_metrics,
            }
        )

    if not models:
        print("[Error] Ningún archivo .h5 válido fue cargado.")
        return

    m0 = models[0]
    n_elec, n_up, n_dn = m0["n_elec"], m0["n_up"], m0["n_dn"]
    omega, g_val, softening = m0["omega"], m0["g"], m0["softening"]

    # 2. Resolver y calcular la densidad exacta analítica del estado CASSCF
    npz_path = args.casscf_npz_path
    if not npz_path:
        for m in models:
            cand = m["bundle"].get("config", {}).get("ansatz", {}).get("casscf_npz_path")
            if cand and Path(cand).exists():
                npz_path = cand
                break
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

    # 3. Procesar cada modelo: Densidades, D_KL(t), MSD(t) y Resumen
    summary_rows = []
    for m in models:
        w4d = _extract_walkers_4d(m["bundle"])
        if w4d is None:
            print(f"[Aviso] El modelo {m['label']} no contiene caminantes de producción.")
            continue

        m["densities"] = _compute_densities_profile(
            w4d, n_up=n_up, n_dn=n_dn, x_max=args.x_max, n_bins=args.n_bins
        )
        m["trace_df"] = _compute_msd_and_acc_trace(
            m["bundle"], w4d, stride=args.obs_stride_steps
        )

        if casscf_ref is not None:
            kl_df = _compute_cumulative_kl_curve(
                w4d,
                n_up=n_up,
                n_dn=n_dn,
                x_exact=np.asarray(casscf_ref["x"]),
                rho_tot_exact=np.asarray(casscf_ref["rho_tot"]),
                rho_up_exact=np.asarray(casscf_ref["rho_up"]),
                rho_dn_exact=np.asarray(casscf_ref["rho_dn"]),
                x_max=args.x_max,
                n_bins=args.n_bins,
                stride=args.kl_stride_steps,
            )
            m["kl_df"] = kl_df
            final_kl_tot = float(kl_df["kl_tot"].iloc[-1]) if not kl_df.empty else float("nan")
            final_kl_up = float(kl_df["kl_up"].iloc[-1]) if not kl_df.empty else float("nan")
            final_kl_dn = float(kl_df["kl_dn"].iloc[-1]) if not kl_df.empty else float("nan")
            final_kl_spin = float(kl_df["kl_spin_mean"].iloc[-1]) if not kl_df.empty else float("nan")
        else:
            m["kl_df"] = pd.DataFrame()
            final_kl_tot = final_kl_up = final_kl_dn = final_kl_spin = float("nan")

        tdf = m["trace_df"]
        mean_msd = float(tdf["msd"].iloc[len(tdf) // 5 :].mean()) if len(tdf) > 5 else float(tdf["msd"].mean())
        mean_acc = float(tdf["acceptance"].mean())
        obs_m = m["obs_metrics"]

        summary_rows.append(
            {
                "label": m["label"],
                "is_neural": m["is_neural"],
                "proposal_type": m["proposal_type"],
                "train_mode": m["train_mode"],
                "tau": m["tau"],
                "actor_amplitude": m["actor_amplitude"],
                "sigma_action_scale": m["sigma_action_scale"],
                "penalty_factor": m["penalty_factor"],
                "lr_actor": m["lr_actor"],
                "E_mean": float(obs_m.get("E_mean", float("nan"))),
                "SE_E": float(obs_m.get("SE_E", float("nan"))),
                "E_var": float(obs_m.get("E_var", float("nan"))),
                "acceptance": float(obs_m.get("acceptance", mean_acc)),
                "ESJD": float(obs_m.get("ESJD", float("nan"))),
                "tau_int_E": float(obs_m.get("tau_int_E", float("nan"))),
                "ESS_per_sec": float(obs_m.get("ESS_per_sec", float("nan"))),
                "mean_MSD_bohr2": mean_msd,
                "final_kl_tot": final_kl_tot,
                "final_kl_up": final_kl_up,
                "final_kl_dn": final_kl_dn,
                "final_kl_spin_mean": final_kl_spin,
            }
        )

    df_summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / f"nodal_crossing_summary_{args.system_tag}.csv"
    df_summary.to_csv(summary_csv, index=False)
    _copy_csv(summary_csv, args.csv_collect_dir)
    print(f"[CSV] Resumen de cruce nodal guardado en: {summary_csv}")

    # Seleccionar el mejor MALA y el mejor RL-MALA (menor final_kl_spin_mean o menor tau_int_E)
    mala_models = [m for m in models if not m["is_neural"] and "densities" in m]
    rl_models = [m for m in models if m["is_neural"] and "densities" in m]

    def _score_model(mod: dict) -> float:
        kdf = mod.get("kl_df", pd.DataFrame())
        if not kdf.empty and np.isfinite(kdf["kl_spin_mean"].iloc[-1]):
            return float(kdf["kl_spin_mean"].iloc[-1])
        t_int = float(mod.get("obs_metrics", {}).get("tau_int_E", float("inf")))
        return t_int if np.isfinite(t_int) else float("inf")

    best_mala = min(mala_models, key=_score_model) if mala_models else models[0]
    best_rl = min(rl_models, key=_score_model) if rl_models else models[-1]

    print(f"  -> Mejor MALA seleccionado   : {best_mala['label']} (score={_score_model(best_mala):.6f})")
    print(f"  -> Mejor RL-MALA seleccionado: {best_rl['label']} (score={_score_model(best_rl):.6f})")

    # =========================================================================
    # FIGURA 1: Article_Densities (Comparación lado a lado MALA vs Mejor RL-MALA)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x_grid = best_mala["densities"]["x"]
    v_trap = 0.5 * (omega ** 2) * (x_grid ** 2)
    v_scale = max(float(np.max(best_mala["densities"]["rho_tot"])), 0.25) / max(float(np.max(v_trap)), 1e-8)
    v_scaled = v_trap * v_scale

    pairs = [
        (axes[0], best_mala, f"Standard MALA ({best_mala['label']})"),
        (axes[1], best_rl, f"Adaptive Policy RL-MALA ({best_rl['label']})"),
    ]
    for ax, mod, title in pairs:
        d = mod["densities"]
        ax.plot(d["x"], d["rho_tot"], color="black", lw=2.5, label="Sampled Total Density")
        ax.fill_between(d["x"], d["rho_up"], color="#3498db", alpha=0.45, label="Sampled Spin ↑")
        ax.fill_between(d["x"], d["rho_dn"], color="#e74c3c", alpha=0.45, label="Sampled Spin ↓")
        if casscf_ref is not None:
            ax.plot(casscf_ref["x"], casscf_ref["rho_tot"], color="black", linestyle=":", lw=2.0, label="Exact CASSCF Total")
            ax.plot(casscf_ref["x"], casscf_ref["rho_up"], color="#1f618d", linestyle=":", lw=2.0, label="Exact Spin ↑")
            ax.plot(casscf_ref["x"], casscf_ref["rho_dn"], color="#922b21", linestyle=":", lw=2.0, label="Exact Spin ↓")
        ax.plot(x_grid, v_scaled, "--", color="gray", alpha=0.4, label="Harmonic Trap (Scaled)")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("x (Bohr)", fontsize=12)
        ax.set_xlim(-args.x_max, args.x_max)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9.5, framealpha=0.9)
    axes[0].set_ylabel("One-Body Density ρ(x)", fontsize=12)
    plt.tight_layout()
    _save_fig(fig, out_dir, f"Article_Densities_{args.system_tag}", args.figure_format, args.dpi)

    # Exportar CSV de densidades (compatible con Article_plots_11/data_densities.csv)
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
        dens_dict["exact_tot"] = np.interp(best_mala["densities"]["x"], casscf_ref["x"], casscf_ref["rho_tot"])
        dens_dict["exact_up"] = np.interp(best_mala["densities"]["x"], casscf_ref["x"], casscf_ref["rho_up"])
        dens_dict["exact_dn"] = np.interp(best_mala["densities"]["x"], casscf_ref["x"], casscf_ref["rho_dn"])
    dens_csv = out_dir / f"data_densities_{args.system_tag}.csv"
    pd.DataFrame(dens_dict).to_csv(dens_csv, index=False)
    _copy_csv(dens_csv, args.csv_collect_dir)

    # =========================================================================
    # FIGURA 2: Article_KL_Divergence (Total, Spin Up, Spin Down)
    # =========================================================================
    if casscf_ref is not None and not best_mala.get("kl_df", pd.DataFrame()).empty:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
        kl_cols = [
            ("kl_tot", "Total Density $D_{KL}$"),
            ("kl_up", "Spin ↑ Density $D_{KL}$ (Nodal Crossing)"),
            ("kl_dn", "Spin ↓ Density $D_{KL}$ (Nodal Crossing)"),
        ]
        for ax, (col, title) in zip(axes, kl_cols):
            # Graficar todos los modelos en gris/azul tenue y remarcar best_mala y best_rl
            for m in mala_models:
                if m is not best_mala and not m["kl_df"].empty:
                    ax.plot(m["kl_df"]["step"], m["kl_df"][col], color="#e74c3c", alpha=0.30, lw=1.2, ls="--")
            for m in rl_models:
                if m is not best_rl and not m["kl_df"].empty:
                    ax.plot(m["kl_df"]["step"], m["kl_df"][col], color="#2ecc71", alpha=0.30, lw=1.2)

            ax.plot(
                best_mala["kl_df"]["step"],
                best_mala["kl_df"][col],
                color="#e74c3c",
                lw=2.6,
                label=f"Best MALA ({best_mala['label']})",
            )
            if not best_rl["kl_df"].empty:
                ax.plot(
                    best_rl["kl_df"]["step"],
                    best_rl["kl_df"][col],
                    color="#2ecc71",
                    lw=2.8,
                    label=f"Best RL-MALA ({best_rl['label']})",
                )
            ax.set_yscale("log")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("MCMC Samples Evaluated", fontsize=11)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize=9.5)
        axes[0].set_ylabel(r"KL Divergence $D_{KL}(P_t \parallel P_{\mathrm{exact}})$", fontsize=11)
        plt.tight_layout()
        _save_fig(fig, out_dir, f"Article_KL_Divergence_{args.system_tag}", args.figure_format, args.dpi)

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
        kl_merged.to_csv(kl_csv, index=False)
        _copy_csv(kl_csv, args.csv_collect_dir)

    # =========================================================================
    # FIGURA 3: Article_Ergodicity_Observables (Acceptance Rate & Wigner MSD)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    tm = best_mala["trace_df"]
    tr = best_rl["trace_df"]

    ax1.plot(tm["step"], tm["acceptance"], color="#e74c3c", alpha=0.65, lw=1.2, label=f"MALA ({best_mala['label']})")
    ax1.plot(tr["step"], tr["acceptance"], color="#2ecc71", alpha=0.65, lw=1.2, label=f"RL-MALA ({best_rl['label']})")
    ax1.axhline(0.574, color="black", linestyle="--", lw=1.5, label="Theoretical Optimum (0.574)")
    ax1.set_title("MCMC Acceptance Rate", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Acceptance Rate", fontsize=11)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.25)

    ax2.plot(tm["step"], tm["msd"], color="#e74c3c", alpha=0.85, lw=1.6, label=f"Standard MALA ({best_mala['label']})")
    ax2.plot(tr["step"], tr["msd"], color="#2ecc71", alpha=0.85, lw=1.6, label=f"RL-MALA ({best_rl['label']})")
    ax2.set_title("Wigner Topological Mobility (MSD)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("MCMC Steps", fontsize=11)
    ax2.set_ylabel(r"MSD ($\mathrm{Bohr}^2$)", fontsize=11)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_fig(fig, out_dir, f"Article_Ergodicity_Observables_{args.system_tag}", args.figure_format, args.dpi)

    obs_csv = out_dir / f"data_observables_{args.system_tag}.csv"
    obs_df = pd.DataFrame(
        {
            "step": tm["step"],
            "acc_mala": tm["acceptance"],
            "msd_mala": tm["msd"],
        }
    )
    if len(tr) == len(tm):
        obs_df["acc_rl"] = tr["acceptance"]
        obs_df["msd_rl"] = tr["msd"]
    obs_df.to_csv(obs_csv, index=False)
    _copy_csv(obs_csv, args.csv_collect_dir)

    # =========================================================================
    # FIGURA 4: Article_Training_Metrics (Diagnóstico del entrenamiento TD3)
    # =========================================================================
    trn_hist = best_rl["bundle"].get("training", {})
    if trn_hist and "Step" in trn_hist and len(trn_hist["Step"]) > 0:
        steps_t = np.asarray(trn_hist["Step"])
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
        c_loss = np.asarray(trn_hist.get("Critic_Loss", np.zeros_like(steps_t)))
        a_loss = np.asarray(trn_hist.get("Actor_Loss", np.zeros_like(steps_t)))
        ax2.plot(steps_t, c_loss, color="#e74c3c", alpha=0.85, label="Critic MSE")
        ax2_r.plot(steps_t, a_loss, color="#2980b9", alpha=0.85, label="Actor Loss (-Q)")
        ax2.set_title("TD3 Network Losses", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Critic MSE", color="#e74c3c", fontsize=11)
        ax2_r.set_ylabel("Actor Loss", color="#2980b9", fontsize=11)
        ax2.grid(True, alpha=0.25)

        ax3_r = ax3.twinx()
        acc_t = np.asarray(trn_hist.get("Acceptance", np.zeros_like(steps_t)))
        disp_t = np.asarray(trn_hist.get("Proposed_Disp", np.zeros_like(steps_t)))
        ax3.plot(steps_t, acc_t, color="#2ecc71", alpha=0.8, label="Acceptance Rate")
        ax3.axhline(0.574, color="black", linestyle="--", alpha=0.5)
        ax3_r.plot(steps_t, disp_t, color="#8e44ad", alpha=0.8, label="Displacement (Bohr)")
        ax3.set_title("Training Phase MCMC Diagnostics", fontsize=13, fontweight="bold")
        ax3.set_xlabel("TD3 Training Sweeps", fontsize=11)
        ax3.set_ylabel("Acceptance Rate", color="#2ecc71", fontsize=11)
        ax3_r.set_ylabel("Displacement (Bohr)", color="#8e44ad", fontsize=11)
        ax3.grid(True, alpha=0.25)

        plt.tight_layout()
        _save_fig(fig, out_dir, f"Article_Training_Metrics_{args.system_tag}", args.figure_format, args.dpi)

    # =========================================================================
    # FIGURA 5: Parameter_Search_Comparison (Ranking de todos los modelos)
    # =========================================================================
    if len(df_summary) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        labels_s = df_summary["label"].tolist()
        colors_s = ["#2ecc71" if is_n else "#e74c3c" for is_n in df_summary["is_neural"]]
        x_pos = np.arange(len(labels_s))

        # Panel 1: Spin-Resolved D_KL (cruce de barreras nodales)
        ax = axes[0, 0]
        kl_vals = df_summary["final_kl_spin_mean"].fillna(0.0).to_numpy()
        ax.bar(x_pos, np.maximum(kl_vals, 1e-7), color=colors_s, alpha=0.85, edgecolor="black")
        ax.set_yscale("log")
        ax.set_title("Spin-Resolved KL Divergence (Lower = Better Nodal Crossing)", fontweight="bold")
        ax.set_ylabel(r"$\frac{1}{2}(D_{KL}^{\uparrow} + D_{KL}^{\downarrow})$ (log)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25)

        # Panel 2: Wigner Topological Mobility (MSD)
        ax = axes[0, 1]
        msd_vals = df_summary["mean_MSD_bohr2"].fillna(0.0).to_numpy()
        ax.bar(x_pos, msd_vals, color=colors_s, alpha=0.85, edgecolor="black")
        ax.set_title("Wigner Topological Mobility MSD (Higher = Better Exploration)", fontweight="bold")
        ax.set_ylabel(r"Mean MSD ($\mathrm{Bohr}^2$)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25)

        # Panel 3: ESJD (Expected Squared Jump Distance)
        ax = axes[1, 0]
        esjd_vals = df_summary["ESJD"].fillna(0.0).to_numpy()
        ax.bar(x_pos, esjd_vals, color=colors_s, alpha=0.85, edgecolor="black")
        ax.set_title("Expected Squared Jump Distance (ESJD)", fontweight="bold")
        ax.set_ylabel(r"ESJD ($\mathrm{Bohr}^2$/step)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25)

        # Panel 4: Integrated Autocorrelation Time tau_int(E)
        ax = axes[1, 1]
        tau_vals = df_summary["tau_int_E"].fillna(0.0).to_numpy()
        ax.bar(x_pos, tau_vals, color=colors_s, alpha=0.85, edgecolor="black")
        ax.set_title(r"Integrated Autocorrelation Time $\tau_{\mathrm{int}}(E)$ (Lower = Better)", fontweight="bold")
        ax.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (steps)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_s, rotation=35, ha="right", fontsize=8.5)
        ax.grid(True, axis="y", alpha=0.25)

        plt.suptitle(f"Hyperparameter Search Comparison — {args.system_tag} (Red=MALA, Green=RL-MALA)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, out_dir, f"Parameter_Search_Comparison_{args.system_tag}", args.figure_format, args.dpi)

    print(f"[OK] Evaluación y gráficas de cruce nodal completadas en: {out_dir}")


if __name__ == "__main__":
    main()
