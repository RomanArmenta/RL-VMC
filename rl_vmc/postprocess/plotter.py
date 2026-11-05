from __future__ import annotations

import gc
import json
import os
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rl_vmc.config import PostProcessConfig, Experiment1DConfig
from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.postprocess.diagnostics import (
    summarize_bundle_diagnostics,
    compute_acf_1d,
    compute_iat_sokal,
)
from rl_vmc.postprocess.observables import (
    compute_1d_observables,
    compute_kl_divergence_2e,
)
from rl_vmc.postprocess.casscf_baseline import (
    discover_casscf_baselines,
    find_matching_casscf_baseline,
    compute_exact_casscf_1body_density,
    compute_spin_kl_divergences,
    compute_kl_convergence_curve,
)
from rl_vmc.ansatze.ansatz_1d import build_ansatz_1d


PALETTE = [
    "#e74c3c",  # Rojo (MALA 1)
    "#c0392b",  # Rojo oscuro (MALA 2)
    "#2ecc71",  # Verde (RL-MALA 1)
    "#2980b9",  # Azul (RL-MALA 2)
    "#9b59b6",
    "#e67e22",
    "#1abc9c",
    "#34495e",
]

MALA_PALETTE = [
    "#e74c3c",  # Rojo intenso
    "#c0392b",  # Carmesí
    "#d35400",  # Naranja oscuro
    "#7f8c8d",  # Gris pizarra
    "#8e44ad",  # Púrpura
]

RL_PALETTE = [
    "#2ecc71",  # Verde esmeralda (#1 RL-MALA)
    "#2980b9",  # Azul cobalto (#2 RL-MALA)
    "#16a085",  # Turquesa oscuro
    "#8e44ad",  # Violeta
    "#f39c12",  # Ámbar
]


def _apply_plot_style(cfg: PostProcessConfig | None) -> None:
    """Aplica escalamiento de fuentes y grosores de línea para figuras de artículo o reporte."""
    f_scale = float(getattr(cfg, "font_scale", 1.0)) if cfg is not None else 1.0
    lw_scale = float(getattr(cfg, "line_width_scale", 1.0)) if cfg is not None else 1.0
    plt.rcParams.update(
        {
            "font.size": 10.5 * f_scale,
            "axes.titlesize": 12.0 * f_scale,
            "axes.labelsize": 10.5 * f_scale,
            "legend.fontsize": 8.8 * f_scale,
            "xtick.labelsize": 9.5 * f_scale,
            "ytick.labelsize": 9.5 * f_scale,
            "lines.linewidth": 1.8 * lw_scale,
        }
    )


def _get_model_style(binfo: dict[str, Any], idx: int) -> tuple[str, str]:
    """
    Retorna `(color, linestyle)` garantizando que:
    - MALA clásico se grafique SIEMPRE con línea discontinua (`"--"`).
    - RL-MALA (sampler neuronal) se grafique SIEMPRE con línea continua (`"-"`).
    """
    hp = binfo.get("hyperparams", {})
    if not hp:
        hp = _extract_hyperparams(str(binfo.get("label", "")), binfo.get("bundle"))
    is_neural = bool(hp.get("is_neural", False))
    slot = int(binfo.get("_family_rank", idx))
    if is_neural:
        return RL_PALETTE[slot % len(RL_PALETTE)], "-"
    return MALA_PALETTE[slot % len(MALA_PALETTE)], "--"


def _save_fig(
    fig: plt.Figure,
    out_dir: str,
    stem: str,
    fmt: str = "png",
    dpi: int = 300,
    sys_tag: str = "",
    cfg: PostProcessConfig | None = None,
) -> None:
    """Guarda la figura una única vez con nombre limpio (por defecto en PNG; soporta PDF, PPI y títulos personalizados)."""
    if cfg is not None:
        if getattr(cfg, "no_titles", False):
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_text("")
            for ax in fig.axes:
                ax.set_title("")
        else:
            custom_sup = str(getattr(cfg, "custom_suptitle", "") or "").strip()
            t_prefix = str(getattr(cfg, "title_prefix", "") or "").strip()
            f_scale = float(getattr(cfg, "font_scale", 1.0))
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

    transparent = bool(getattr(cfg, "transparent_bg", False)) if cfg is not None else False
    final_stem = (
        f"{stem}_{sys_tag}"
        if (sys_tag and not stem.endswith(f"_{sys_tag}"))
        else stem
    )
    if fmt in ("png", "both"):
        fig.savefig(
            os.path.join(out_dir, f"{final_stem}.png"),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
    if fmt in ("pdf", "both"):
        fig.savefig(
            os.path.join(out_dir, f"{final_stem}.pdf"),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
    plt.close(fig)


def _extract_hyperparams(label: str, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extrae hiperparámetros del sampler/RL desde config del bundle o mediante regex sobre label."""
    cfg = bundle.get("config", {}) if bundle else {}
    prop_cfg = cfg.get("proposal", {})
    rl_cfg = cfg.get("rl", {})

    ptype = str(prop_cfg.get("proposal_type", "")).lower()
    is_neural = ptype in ("neural_deepset", "neural_flat", "deepset", "flat_mlp") or (
        "rl_mala" in label.lower() or "td3" in label.lower()
    )

    def _regex_float(pattern: str, text: str, default: float = float("nan")) -> float:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return default
        return default

    tau_val = float(prop_cfg.get("tau", _regex_float(r"tau([0-9.]+)", label, 0.08)))
    if not is_neural:
        amp_val = 0.0
        pen_val = 0.0
        sig_val = 0.0
        lr_act = 0.0
    else:
        amp_val = float(
            prop_cfg.get("actor_amplitude", _regex_float(r"amp([0-9.]+)", label, 0.15))
        )
        pen_val = float(
            rl_cfg.get("penalty_factor", _regex_float(r"pen([0-9.]+)", label, 0.25))
        )
        sig_val = float(
            prop_cfg.get("sigma_action_scale", _regex_float(r"sig([0-9.]+)", label, 1.1))
        )
        lr_act = float(rl_cfg.get("lr_actor", 1e-5))

    return {
        "is_neural": bool(is_neural),
        "proposal_type": ptype or ("neural_deepset" if is_neural else "mala"),
        "tau": tau_val,
        "actor_amplitude": amp_val,
        "penalty_factor": pen_val,
        "sigma_action_scale": sig_val,
        "lr_actor": lr_act,
    }


def _compute_multi_eval_uncertainties(
    w_all: np.ndarray | None,
    bundle: dict[str, Any],
    summary: dict[str, Any],
    casscf_ref: dict[str, Any] | None,
    cfg: PostProcessConfig,
    x_max: float,
) -> dict[str, float]:
    """
    Calcula barras de error (error estándar) sobre múltiples evaluaciones independientes:
    - Si existen múltiples cadenas (`n_chains > 1`), evalúa cada cadena y calcula SE entre cadenas.
    - Si existe 1 cadena con múltiples caminantes/bloques, aplica remuestreo Jackknife
      por bloques independientes de caminantes y sub-series temporales.
    """
    n_sub = max(2, int(getattr(cfg, "n_eval_subblocks", 5)))
    res = {
        "kl_tot_mean": float("nan"),
        "kl_tot_std": float("nan"),
        "kl_up_mean": float("nan"),
        "kl_up_std": float("nan"),
        "kl_dn_mean": float("nan"),
        "kl_dn_std": float("nan"),
        "kl_spin_mean": float("nan"),
        "kl_spin_std": float("nan"),
        "tau_int_E_std": float("nan"),
        "ESJD_std": float("nan"),
        "ESS_per_sec_std": float("nan"),
    }

    # 1. Estadística de D_KL resuelta por espín
    if casscf_ref is not None and w_all is not None and w_all.size > 0:
        w_5d = w_all[None, ...] if w_all.ndim == 4 else w_all
        n_chains, n_steps, n_batch, n_elec, _ = w_5d.shape

        # Media global usando todo el ensamble disponible
        w_merged = w_5d.reshape(-1, n_batch, n_elec, 1)
        kl_full = compute_spin_kl_divergences(
            w_merged, casscf_ref, n_bins=cfg.n_density_bins, x_max=x_max
        )
        res["kl_tot_mean"] = float(kl_full["kl_total"])
        res["kl_up_mean"] = float(kl_full["kl_up"])
        res["kl_dn_mean"] = float(kl_full["kl_down"])
        res["kl_spin_mean"] = 0.5 * (res["kl_up_mean"] + res["kl_dn_mean"])

        if n_chains > 1:
            kls_t, kls_u, kls_d, kls_s = [], [], [], []
            for c_i in range(n_chains):
                kd = compute_spin_kl_divergences(
                    w_5d[c_i], casscf_ref, n_bins=cfg.n_density_bins, x_max=x_max
                )
                kls_t.append(kd["kl_total"])
                kls_u.append(kd["kl_up"])
                kls_d.append(kd["kl_down"])
                kls_s.append(0.5 * (kd["kl_up"] + kd["kl_down"]))
            scale = 1.0 / np.sqrt(float(n_chains))
            res["kl_tot_std"] = float(np.std(kls_t, ddof=1) * scale)
            res["kl_up_std"] = float(np.std(kls_u, ddof=1) * scale)
            res["kl_dn_std"] = float(np.std(kls_d, ddof=1) * scale)
            res["kl_spin_std"] = float(np.std(kls_s, ddof=1) * scale)
        elif n_batch >= n_sub:
            # Remuestreo Jackknife dejando fuera 1 bloque de caminantes a la vez
            block_indices = np.array_split(np.arange(n_batch), n_sub)
            jk_t, jk_u, jk_d, jk_s = [], [], [], []
            for b_idx in block_indices:
                keep_mask = np.ones(n_batch, dtype=bool)
                keep_mask[b_idx] = False
                w_jk = w_5d[0, :, keep_mask, :, :]
                # w_jk tras indexar booleano en eje 1 puede ser (n_keep, n_steps, n_elec, 1)
                if w_jk.shape[0] != n_steps:
                    w_jk = np.swapaxes(w_jk, 0, 1)
                kd = compute_spin_kl_divergences(
                    w_jk, casscf_ref, n_bins=cfg.n_density_bins, x_max=x_max
                )
                jk_t.append(kd["kl_total"])
                jk_u.append(kd["kl_up"])
                jk_d.append(kd["kl_down"])
                jk_s.append(0.5 * (kd["kl_up"] + kd["kl_down"]))
            k_f = float(len(jk_t))
            jk_factor = np.sqrt((k_f - 1.0) / k_f)
            res["kl_tot_std"] = float(jk_factor * np.std(jk_t, ddof=0))
            res["kl_up_std"] = float(jk_factor * np.std(jk_u, ddof=0))
            res["kl_dn_std"] = float(jk_factor * np.std(jk_d, ddof=0))
            res["kl_spin_std"] = float(jk_factor * np.std(jk_s, ddof=0))

    # 2. Estadística por bloques para tau_int_E, ESJD y ESS/s
    e_series = summary.get("e_step_series", None)
    if e_series is not None and len(e_series) >= n_sub * 100:
        e_arr = np.asarray(e_series, dtype=np.float64)
        chunks = np.array_split(e_arr, n_sub)
        taus_sub = []
        for ch in chunks:
            acf_c = compute_acf_1d(ch, max_lag=min(cfg.max_acf_lag, len(ch) // 2))
            taus_sub.append(compute_iat_sokal(acf_c, c_sokal=cfg.sokal_window_c))
        tau_se = float(np.std(taus_sub, ddof=1) / np.sqrt(float(len(taus_sub))))
        res["tau_int_E_std"] = tau_se
        tau_mean = max(float(summary.get("tau_int_E", 1.0)), 1e-6)
        ess_s_mean = float(summary.get("ESS_per_sec", 0.0))
        res["ESS_per_sec_std"] = float(ess_s_mean * min(0.5, tau_se / tau_mean))

    prod = bundle.get("production", {})
    s_stats = prod.get("sampler_stats", {})
    if "esjd" in s_stats and np.asarray(s_stats["esjd"]).size >= n_sub * 10:
        esjd_arr = np.asarray(s_stats["esjd"], dtype=np.float64).reshape(-1)
        esjd_chunks = [float(np.mean(c)) for c in np.array_split(esjd_arr, n_sub)]
        res["ESJD_std"] = float(np.std(esjd_chunks, ddof=1) / np.sqrt(float(n_sub)))

    return res


def _select_top_k_bundles(
    bundles_info: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    top_k: int,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Selecciona los `top_k` mejores modelos clásicos (MALA) y los `top_k` mejores
    modelos neuronales (RL-MALA) para que las gráficas comparativas sean nítidas.
    """
    if top_k <= 0:
        for i, b in enumerate(bundles_info):
            b["_family_rank"] = i
        return bundles_info

    mala_pairs = []
    rl_pairs = []
    for binfo, row in zip(bundles_info, summary_rows):
        kl_spin = float(row.get("kl_casscf_spin_mean", float("nan")))
        tau_e = float(row.get("tau_int_E", float("inf")))
        if not np.isfinite(tau_e) or tau_e <= 0:
            tau_e = 1e9
        # Score compuesto: prioriza cruce nodal (D_KL por espín) y desempata con tau_int_E
        score = kl_spin + 1e-4 * tau_e if np.isfinite(kl_spin) else tau_e
        if row.get("is_neural", False):
            rl_pairs.append((score, binfo, row))
        else:
            mala_pairs.append((score, binfo, row))

    mala_pairs.sort(key=lambda item: item[0])
    rl_pairs.sort(key=lambda item: item[0])

    selected_mala = []
    for r_idx, (_, b, _) in enumerate(mala_pairs[:top_k]):
        b_copy = dict(b)
        b_copy["_family_rank"] = r_idx
        selected_mala.append(b_copy)

    selected_rl = []
    for r_idx, (_, b, _) in enumerate(rl_pairs[:top_k]):
        b_copy = dict(b)
        b_copy["_family_rank"] = r_idx
        selected_rl.append(b_copy)

    selected = selected_mala + selected_rl

    if not selected:
        return bundles_info[: 2 * top_k]

    if verbose and len(selected) < len(bundles_info):
        print(
            f"\n[PostProcess] Filtrando Top-{top_k} modelos por familia "
            f"(de {len(bundles_info)} totales):"
        )
        for b in selected_mala:
            print(f"  * [Top MALA --]    {b['label']}")
        for b in selected_rl:
            print(f"  * [Top RL-MALA -]  {b['label']}")
    return selected


def plot_training_dashboard(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    valid = [b for b in bundles_info if b["bundle"].get("training_df") is not None]
    if not valid:
        return

    _apply_plot_style(cfg)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for idx, binfo in enumerate(valid):
        df_t = binfo["bundle"]["training_df"]
        label = binfo["label"]
        color, ls = _get_model_style(binfo, idx)
        steps = df_t["Step"] if "Step" in df_t.columns else np.arange(len(df_t))

        if "Energy_Mean" in df_t.columns:
            axes[0, 0].plot(steps, df_t["Energy_Mean"], color=color, linestyle=ls, lw=1.7, label=label)
        if "Energy_Var" in df_t.columns:
            axes[0, 1].plot(steps, df_t["Energy_Var"], color=color, linestyle=ls, lw=1.7, label=label)
        if "Acceptance" in df_t.columns:
            axes[1, 0].plot(steps, df_t["Acceptance"], color=color, linestyle=ls, lw=1.7, label=label)
        if "ESJD" in df_t.columns:
            axes[1, 1].plot(steps, df_t["ESJD"], color=color, linestyle=ls, lw=1.7, label=label)

    if casscf_ref is not None and np.isfinite(casscf_ref.get("energy", float("nan"))):
        axes[0, 0].axhline(
            casscf_ref["energy"],
            color="black",
            linestyle=":",
            lw=2.2,
            label=f"Exact {casscf_ref['label']} ({casscf_ref['energy']:.5f} Ha)",
        )

    axes[0, 0].set_title("Convergencia de Energía Local $\\langle E_L \\rangle$")
    axes[0, 0].set_xlabel("Paso de Entrenamiento")
    axes[0, 0].set_ylabel("Energía (Ha)")

    axes[0, 1].set_title("Varianza de Energía Local $\\mathrm{Var}(E_L)$")
    axes[0, 1].set_xlabel("Paso de Entrenamiento")
    axes[0, 1].set_ylabel("Varianza ($\\mathrm{Ha}^2$)")

    axes[1, 0].set_title("Tasa de Aceptación MCMC")
    axes[1, 0].set_xlabel("Paso de Entrenamiento")
    axes[1, 0].set_ylabel("Aceptación")

    axes[1, 1].set_title("Expected Squared Jump Distance (ESJD)")
    axes[1, 1].set_xlabel("Paso de Entrenamiento")
    axes[1, 1].set_ylabel("ESJD ($a_0^2$)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig1_training_dashboard",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def plot_production_summary(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    valid = [
        b
        for b in bundles_info
        if "e_step_series" in b["diagnostics"].get("summary", {})
    ]
    if not valid:
        return

    _apply_plot_style(cfg)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for idx, binfo in enumerate(valid):
        summary = binfo["diagnostics"]["summary"]
        label = binfo["label"]
        color, ls = _get_model_style(binfo, idx)
        e_series = summary["e_step_series"]
        steps = np.arange(len(e_series))

        # 1. Traza y media acumulada
        axes[0, 0].plot(steps, e_series, color=color, linestyle=ls, alpha=0.22, lw=0.8)
        cum_mean = np.cumsum(e_series) / (steps + 1)
        axes[0, 0].plot(
            steps,
            cum_mean,
            color=color,
            linestyle=ls,
            lw=2.2,
            label=f"{label} ($\\langle E\\rangle={summary['E_mean']:.5f}\\pm{summary['SE_E']:.5f}$)",
        )

        # 2. Función de autocorrelación ACF
        acf_e = summary["acf_energy"]
        lags = np.arange(len(acf_e))
        axes[0, 1].plot(
            lags,
            acf_e,
            color=color,
            linestyle=ls,
            lw=2.2,
            label=f"{label} ($\\tau_{{\\mathrm{{int}},E}}={summary['tau_int_E']:.2f}$)",
        )

        # 3. Histograma de energía por paso con filtrado robusto de divergencias
        try:
            e_finite = e_series[np.isfinite(e_series)]
            if len(e_finite) > 1:
                q01, q99 = np.percentile(e_finite, [1.0, 99.0])
                if not (np.isfinite(q01) and np.isfinite(q99)) or q01 >= q99:
                    q01, q99 = float(np.min(e_finite)), float(np.max(e_finite))
                if q01 < q99 and (q99 - q01) < 1e10:
                    e_clip = e_finite[(e_finite >= q01) & (e_finite <= q99)]
                    if len(e_clip) > 5:
                        n_bins = min(35, max(5, int(np.sqrt(len(e_clip)))))
                        axes[1, 0].hist(
                            e_clip,
                            bins=n_bins,
                            range=(q01, q99),
                            density=True,
                            color=color,
                            alpha=0.40,
                            label=label,
                        )
        except Exception:
            pass

        # 4. Convergencia del Error Estándar SE(E) vs paso
        tau_raw = summary.get("tau_int_E", 1.0)
        tau_e = max(1.0, float(tau_raw)) if np.isfinite(tau_raw) else 1.0
        std_val = float(np.std(e_series)) if np.isfinite(np.std(e_series)) else 0.0
        se_curve = std_val * np.sqrt(tau_e / (steps + 1.0))
        valid_se = np.isfinite(se_curve) & (se_curve > 0)
        if np.any(valid_se):
            axes[1, 1].plot(
                steps[valid_se] + 1,
                se_curve[valid_se],
                color=color,
                linestyle=ls,
                lw=2.2,
                label=label,
            )

    if casscf_ref is not None and np.isfinite(casscf_ref.get("energy", float("nan"))):
        e_cas = casscf_ref["energy"]
        axes[0, 0].axhline(
            e_cas,
            color="black",
            linestyle=":",
            lw=2.4,
            label=f"Exact {casscf_ref['label']} ({e_cas:.5f} Ha)",
        )
        axes[1, 0].axvline(
            e_cas,
            color="black",
            linestyle=":",
            lw=2.4,
            label=f"Exact {casscf_ref['label']}",
        )

    axes[0, 0].set_title("Traza de Producción y Media Acumulada de Energía")
    axes[0, 0].set_xlabel("Paso de Producción")
    axes[0, 0].set_ylabel("$\\langle E_L \\rangle_w$ (Ha)")

    axes[0, 1].axhline(0.0, color="gray", linestyle=":", lw=1.0)
    axes[0, 1].set_title("Autocorrelación de Energía Local $\\rho_E(k)$")
    axes[0, 1].set_xlabel("Lag $k$")
    axes[0, 1].set_ylabel("$\\rho_E(k)$")

    axes[1, 0].set_title("Distribución Empírica de Energía por Paso")
    axes[1, 0].set_xlabel("$\\langle E_L \\rangle_w$ (Ha)")
    axes[1, 0].set_ylabel("Densidad")

    try:
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xscale("log")
    except Exception:
        pass
    axes[1, 1].set_title("Decaimiento del Error Estándar Correlacionado $\\mathrm{SE}(E)$")
    axes[1, 1].set_xlabel("Pasos de Producción $M$")
    axes[1, 1].set_ylabel("$\\mathrm{SE}(E)$ (Ha)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=8.5)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig2_production_summary",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def plot_observables_comparison(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    valid = [b for b in bundles_info if b.get("observables") is not None]
    if not valid:
        return

    _apply_plot_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for idx, binfo in enumerate(valid):
        obs = binfo["observables"]
        label = binfo["label"]
        color, ls = _get_model_style(binfo, idx)
        x = obs["x_grid"]
        r = obs["r_grid"]

        axes[0].plot(
            x,
            obs["rho_total"],
            color=color,
            linestyle=ls,
            lw=2.2,
            label=f"{label} (picos={obs['n_wigner_peaks']})",
        )
        if obs.get("n_chains_eval", 1) > 1 and "rho_total_std" in obs:
            axes[0].fill_between(
                x,
                np.maximum(0.0, obs["rho_total"] - obs["rho_total_std"]),
                obs["rho_total"] + obs["rho_total_std"],
                color=color,
                alpha=0.16,
            )

        axes[1].plot(
            x, obs["rho_up"], color=color, lw=2.0, linestyle=ls, label=f"{label} $\\uparrow$"
        )
        axes[1].plot(
            x,
            obs["rho_down"],
            color=color,
            lw=1.5,
            linestyle=ls,
            alpha=0.65,
            label=f"{label} $\\downarrow$",
        )
        axes[2].plot(
            r,
            obs["g_pair_total"],
            color=color,
            linestyle=ls,
            lw=2.2,
            label=f"{label} ($\\langle r_{{ij}}\\rangle={obs['mean_pair_dist']:.2f}$)",
        )

    # Si existe baseline CASSCF pre-entrenada para este sistema, superponerla
    if casscf_ref is not None:
        x_ex = casscf_ref["x"]
        axes[0].plot(
            x_ex,
            casscf_ref["rho_total"],
            "k:",
            lw=2.6,
            alpha=0.95,
            label=f"Exact {casscf_ref['label']} Total",
        )
        axes[1].plot(
            x_ex,
            casscf_ref["rho_up"],
            color="#2980b9",
            linestyle=":",
            lw=2.5,
            alpha=0.95,
            label=r"Exact CASSCF $\uparrow$",
        )
        axes[1].plot(
            x_ex,
            casscf_ref["rho_down"],
            color="#c0392b",
            linestyle=":",
            lw=2.5,
            alpha=0.95,
            label=r"Exact CASSCF $\downarrow$",
        )

    axes[0].set_title("Densidad de 1 Cuerpo $\\rho_{\\mathrm{tot}}(x)$")
    axes[0].set_xlabel("Posición $x$ ($a_0$)")
    axes[0].set_ylabel("$\\rho_{\\mathrm{tot}}(x)$")

    axes[1].set_title("Densidades Resueltas por Espín $\\rho_\\uparrow, \\rho_\\downarrow$")
    axes[1].set_xlabel("Posición $x$ ($a_0$)")
    axes[1].set_ylabel("$\\rho_\\sigma(x)$")

    axes[2].set_title("Distribución de Distancia de Pares $P(|x_i - x_j|)$")
    axes[2].set_xlabel("Distancia $|x_i - x_j|$ ($a_0$)")
    axes[2].set_ylabel("$P(|x_i - x_j|)$")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True, fontsize=8.0)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig3_observables_comparison",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def plot_article_densities_comparison(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Genera la gráfica comparativa de densidades estilo Artículo (`Article_Densities` de
    `scripts/evaluar_wigner_casscf_2.py`), incluyendo bandas de incertidumbre +/- 1 sigma
    entre corridas independientes y superponiendo la baseline CASSCF exacta si existe.
    Si no existe baseline CASSCF, grafica limpiamente las densidades muestreadas sin baseline.
    """
    valid = [b for b in bundles_info if b.get("observables") is not None]
    if not valid:
        return

    _apply_plot_style(cfg)
    n_panels = len(valid)
    ncols = 2 if n_panels == 4 else min(n_panels, 3)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.2 * ncols, 5.4 * nrows), sharey=True, squeeze=False
    )
    axes_flat = axes.flatten()

    for idx, binfo in enumerate(valid):
        ax = axes_flat[idx]
        obs = binfo["observables"]
        sys_cfg = binfo["bundle"].get("config", {}).get("system", {})
        omega = float(sys_cfg.get("omega", 0.1))
        x = obs["x_grid"]
        x_max = float(obs.get("x_max", 15.0))
        n_chains = int(obs.get("n_chains_eval", 1))
        _, ls_model = _get_model_style(binfo, idx)
        fam_tag = "RL-MALA [solid]" if ls_model == "-" else "MALA [dashed]"

        # Densidad total muestreada (discontinua para MALA, continua para RL-MALA)
        ax.plot(
            x,
            obs["rho_total"],
            color="black",
            linestyle=ls_model,
            lw=2.5,
            label=f"Sampled Total ({fam_tag}, $N_{{\\mathrm{{eval}}}}={max(n_chains, int(getattr(cfg, 'n_eval_subblocks', 5)))}$)",
        )
        if "rho_total_std" in obs and np.any(obs["rho_total_std"] > 0):
            ax.fill_between(
                x,
                np.maximum(0.0, obs["rho_total"] - obs["rho_total_std"]),
                obs["rho_total"] + obs["rho_total_std"],
                color="black",
                alpha=0.18,
                label=r"Total $\pm 1\sigma$ (evals)",
            )

        # Densidades resueltas por espín
        ax.fill_between(
            x,
            obs["rho_up"],
            color="#3498db",
            alpha=0.45,
            label=r"Sampled Spin $\uparrow$",
        )
        ax.fill_between(
            x,
            obs["rho_down"],
            color="#e74c3c",
            alpha=0.45,
            label=r"Sampled Spin $\downarrow$",
        )

        # Si existe baseline CASSCF pre-entrenada, superponer las 3 curvas punteadas exactas
        if casscf_ref is not None:
            x_ex = casscf_ref["x"]
            mask_ex = (x_ex >= -x_max) & (x_ex <= x_max)
            ax.plot(
                x_ex[mask_ex],
                casscf_ref["rho_total"][mask_ex],
                "k:",
                lw=2.5,
                alpha=0.95,
                label="Exact CASSCF Total",
            )
            ax.plot(
                x_ex[mask_ex],
                casscf_ref["rho_up"][mask_ex],
                color="#2980b9",
                linestyle=":",
                lw=2.5,
                alpha=0.95,
                label=r"Exact Spin $\uparrow$",
            )
            ax.plot(
                x_ex[mask_ex],
                casscf_ref["rho_down"][mask_ex],
                color="#c0392b",
                linestyle=":",
                lw=2.5,
                alpha=0.95,
                label=r"Exact Spin $\downarrow$",
            )

        # Potencial armónico confinante V(x) = 0.5 * omega^2 * x^2 en eje secundario
        espacio = np.linspace(-x_max, x_max, 400)
        potencial = 0.5 * (omega ** 2) * (espacio ** 2)
        ax_pot = ax.twinx()
        ax_pot.plot(
            espacio, potencial, "--", color="gray", lw=1.5, alpha=0.5, label="V(x)"
        )
        ax_pot.set_yticks([])

        ax.set_xlim(-x_max, x_max)
        ax.set_title(binfo["label"], fontsize=11.5, weight="bold")
        ax.set_xlabel("x (Bohr)")
        if idx % ncols == 0:
            ax.set_ylabel(r"One-Body Density $\rho(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.2, frameon=True)

    for extra_idx in range(n_panels, len(axes_flat)):
        axes_flat[extra_idx].set_visible(False)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig7_article_densities_comparison",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def _precompute_kl_curve_for_bundle(
    w_all: np.ndarray | None,
    casscf_ref: dict[str, Any] | None,
    cfg: PostProcessConfig,
    x_max: float,
) -> dict[str, np.ndarray] | None:
    """Precalcula la curva de convergencia D_KL(t) (20 puntos) para poder liberar w_all de RAM."""
    if casscf_ref is None or w_all is None or w_all.size == 0:
        return None
    n_sub = max(2, int(getattr(cfg, "n_eval_subblocks", 5)))
    w_5d = np.asarray(w_all, dtype=np.float64)
    if w_5d.ndim == 4:
        w_5d = w_5d[None, ...]
    n_chains, n_steps, n_batch, _, _ = w_5d.shape

    if n_chains > 1:
        tot_curves, up_curves, dn_curves = [], [], []
        steps_grid = None
        for c_idx in range(n_chains):
            curve = compute_kl_convergence_curve(
                w_5d[c_idx],
                casscf_ref,
                n_checkpoints=20,
                n_bins=cfg.n_density_bins,
                x_max=x_max,
            )
            steps_grid = curve["steps"]
            tot_curves.append(curve["kl_total"])
            up_curves.append(curve["kl_up"])
            dn_curves.append(curve["kl_down"])

        arr_tot = np.stack(tot_curves, axis=0)
        arr_up = np.stack(up_curves, axis=0)
        arr_dn = np.stack(dn_curves, axis=0)

        mean_tot, std_tot = np.mean(arr_tot, axis=0), np.std(arr_tot, axis=0)
        mean_up, std_up = np.mean(arr_up, axis=0), np.std(arr_up, axis=0)
        mean_dn, std_dn = np.mean(arr_dn, axis=0), np.std(arr_dn, axis=0)
    else:
        full_curve = compute_kl_convergence_curve(
            w_5d[0],
            casscf_ref,
            n_checkpoints=20,
            n_bins=cfg.n_density_bins,
            x_max=x_max,
        )
        steps_grid = full_curve["steps"]
        mean_tot = full_curve["kl_total"]
        mean_up = full_curve["kl_up"]
        mean_dn = full_curve["kl_down"]

        if n_batch >= n_sub:
            block_indices = np.array_split(np.arange(n_batch), n_sub)
            jk_t, jk_u, jk_d = [], [], []
            for b_idx in block_indices:
                keep_mask = np.ones(n_batch, dtype=bool)
                keep_mask[b_idx] = False
                w_jk = w_5d[0, :, keep_mask, :, :]
                if w_jk.shape[0] != n_steps:
                    w_jk = np.swapaxes(w_jk, 0, 1)
                c_jk = compute_kl_convergence_curve(
                    w_jk,
                    casscf_ref,
                    n_checkpoints=20,
                    n_bins=cfg.n_density_bins,
                    x_max=x_max,
                )
                jk_t.append(c_jk["kl_total"])
                jk_u.append(c_jk["kl_up"])
                jk_d.append(c_jk["kl_down"])
            k_f = float(len(jk_t))
            jk_factor = np.sqrt((k_f - 1.0) / k_f)
            std_tot = jk_factor * np.std(np.stack(jk_t, axis=0), axis=0)
            std_up = jk_factor * np.std(np.stack(jk_u, axis=0), axis=0)
            std_dn = jk_factor * np.std(np.stack(jk_d, axis=0), axis=0)
        else:
            std_tot = np.zeros_like(mean_tot)
            std_up = np.zeros_like(mean_up)
            std_dn = np.zeros_like(mean_dn)

    return {
        "steps": np.asarray(steps_grid),
        "mean_tot": np.asarray(mean_tot),
        "std_tot": np.asarray(std_tot),
        "mean_up": np.asarray(mean_up),
        "std_up": np.asarray(std_up),
        "mean_dn": np.asarray(mean_dn),
        "std_dn": np.asarray(std_dn),
    }


def _precompute_walker_trace_for_bundle(
    w_all: np.ndarray | None,
    n_up: int,
    n_dn: int,
    x_max: float,
) -> dict[str, Any] | None:
    """
    Extrae la trayectoria de un solo caminante (caminante 0 de cadena 0) y los histogramas
    marginales por espín, permitiendo liberar el arreglo completo de caminantes de la RAM.
    """
    if w_all is None or w_all.size == 0:
        return None
    w_arr = np.asarray(w_all)
    if w_arr.ndim == 4:
        w_arr = w_arr[None, ...]
    w_single = w_arr[0, :, 0, :, 0]  # (n_steps, n_elec)
    n_steps, n_elec = w_single.shape
    stride = max(1, n_steps // 2500)
    steps_sub = np.arange(0, n_steps, stride)
    w_single_sub = np.array(w_single[::stride, :], dtype=np.float32, copy=True)

    # Calcular saltos inter-pozo del caminante 0 (cambios de signo o cruces entre electrones)
    sign_flips = int(np.sum(np.diff(np.sign(w_single), axis=0) != 0))

    bins = np.linspace(-x_max, x_max, 120)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    up_hist = np.zeros(len(bin_centers), dtype=np.float64)
    dn_hist = np.zeros(len(bin_centers), dtype=np.float64)
    try:
        w_flat = w_arr.squeeze(-1).reshape(-1, n_elec)
        if n_up > 0:
            up_vals = w_flat[:, :n_up].reshape(-1)
            up_vals = up_vals[np.isfinite(up_vals)]
            up_vals = up_vals[(up_vals >= -x_max) & (up_vals <= x_max)]
            if len(up_vals) > 0:
                up_hist, _ = np.histogram(up_vals, bins=bins, density=True)
        if n_dn > 0:
            dn_vals = w_flat[:, n_up : n_up + n_dn].reshape(-1)
            dn_vals = dn_vals[np.isfinite(dn_vals)]
            dn_vals = dn_vals[(dn_vals >= -x_max) & (dn_vals <= x_max)]
            if len(dn_vals) > 0:
                dn_hist, _ = np.histogram(dn_vals, bins=bins, density=True)
    except Exception:
        pass

    return {
        "steps_sub": steps_sub,
        "w_single_sub": w_single_sub,
        "n_steps": int(n_steps),
        "n_elec": int(n_elec),
        "n_up": int(n_up),
        "n_dn": int(n_dn),
        "x_max": float(x_max),
        "bins": bins,
        "bin_centers": bin_centers,
        "up_hist": up_hist,
        "dn_hist": dn_hist,
        "sign_flips": sign_flips,
    }


def plot_article_kl_divergence(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Si existe una baseline CASSCF pre-entrenada (`casscf_ref is not None`), genera la
    evolución de la Divergencia KL resuelta por espín (`fig8_article_kl_divergence`),
    graficando MALA con líneas discontinuas (`--`) y RL-MALA con líneas continuas (`-`).
    """
    if casscf_ref is None:
        return

    valid = [
        b
        for b in bundles_info
        if b.get("kl_curve_data") is not None
        or (
            "walkers" in b["bundle"].get("production", {})
            and np.asarray(b["bundle"]["production"]["walkers"]).size > 0
        )
    ]
    if not valid:
        return

    _apply_plot_style(cfg)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    kl_records = []

    for idx, binfo in enumerate(valid):
        color, ls = _get_model_style(binfo, idx)
        label = binfo["label"]
        kl_data = binfo.get("kl_curve_data")
        if kl_data is None:
            x_max = float(
                binfo.get("observables", {}).get("x_max", cfg.density_x_max)
                if binfo.get("observables")
                else cfg.density_x_max
            )
            kl_data = _precompute_kl_curve_for_bundle(
                binfo["bundle"]["production"]["walkers"], casscf_ref, cfg, x_max=x_max
            )
        if kl_data is None:
            continue

        steps_grid = kl_data["steps"]
        mean_tot, std_tot = kl_data["mean_tot"], kl_data["std_tot"]
        mean_up, std_up = kl_data["mean_up"], kl_data["std_up"]
        mean_dn, std_dn = kl_data["mean_dn"], kl_data["std_dn"]

        for ax, m_arr, s_arr in zip(
            axes, (mean_tot, mean_up, mean_dn), (std_tot, std_up, std_dn)
        ):
            ax.plot(steps_grid, m_arr, color=color, linestyle=ls, lw=2.4, label=label)
            if np.any(s_arr > 0):
                ax.fill_between(
                    steps_grid,
                    np.maximum(1e-8, m_arr - s_arr),
                    m_arr + s_arr,
                    color=color,
                    alpha=0.18,
                )

        for s_step, mt, st, mu, su, md, sd in zip(
            steps_grid, mean_tot, std_tot, mean_up, std_up, mean_dn, std_dn
        ):
            kl_records.append(
                {
                    "label": label,
                    "step": int(s_step),
                    "kl_tot_mean": float(mt),
                    "kl_tot_std": float(st),
                    "kl_up_mean": float(mu),
                    "kl_up_std": float(su),
                    "kl_dn_mean": float(md),
                    "kl_dn_std": float(sd),
                }
            )

    titles = [
        r"Total Density $D_{\mathrm{KL}}$",
        r"Spin $\uparrow$ Density $D_{\mathrm{KL}}$",
        r"Spin $\downarrow$ Density $D_{\mathrm{KL}}$",
    ]
    for ax, title in zip(axes, titles):
        ax.set_yscale("log")
        ax.set_title(title, weight="bold", fontsize=13)
        ax.set_xlabel("MCMC Production Steps", fontsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=9.0, frameon=True)

    axes[0].set_ylabel(
        f"KL Divergence $D_{{\\mathrm{{KL}}}}(P_t \\parallel P_{{\\mathrm{{CASSCF}}}})$",
        fontsize=11,
    )
    fig.tight_layout()
    sys_tag = getattr(cfg, "system_tag", "")
    _save_fig(
        fig,
        out_dir,
        "fig8_article_kl_divergence",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=sys_tag,
        cfg=cfg,
    )

    if kl_records and cfg.save_observables_csv:
        df_kl = pd.DataFrame(kl_records)
        kl_fname = (
            f"kl_convergence_vs_casscf_{sys_tag}.csv"
            if sys_tag
            else "kl_convergence_vs_casscf.csv"
        )
        kl_csv_path = os.path.join(out_dir, kl_fname)
        df_kl.to_csv(kl_csv_path, index=False)
        if getattr(cfg, "csv_collect_dir", ""):
            import shutil

            os.makedirs(cfg.csv_collect_dir, exist_ok=True)
            shutil.copy2(kl_csv_path, os.path.join(cfg.csv_collect_dir, kl_fname))


def plot_efficiency_bars(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
) -> None:
    valid = [b for b in bundles_info if b["diagnostics"].get("summary")]
    if not valid:
        return

    _apply_plot_style(cfg)
    labels = [b["label"] for b in valid]
    taus = [b["diagnostics"]["summary"].get("tau_int_E", 1.0) for b in valid]
    taus_err = [b.get("uncertainties", {}).get("tau_int_E_std", 0.0) for b in valid]
    ess_s = [b["diagnostics"]["summary"].get("ESS_per_sec", 0.0) for b in valid]
    ess_s_err = [b.get("uncertainties", {}).get("ESS_per_sec_std", 0.0) for b in valid]
    esjds = [b["diagnostics"]["summary"].get("ESJD", 0.0) for b in valid]
    esjds_err = [b.get("uncertainties", {}).get("ESJD_std", 0.0) for b in valid]
    styles = [_get_model_style(b, i) for i, b in enumerate(valid)]
    colors = [s[0] for s in styles]
    hatches = ["//" if s[1] == "--" else "" for s in styles]

    taus_err = [0.0 if not np.isfinite(v) else v for v in taus_err]
    ess_s_err = [0.0 if not np.isfinite(v) else v for v in ess_s_err]
    esjds_err = [0.0 if not np.isfinite(v) else v for v in esjds_err]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    bars0 = axes[0].bar(labels, taus, yerr=taus_err, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[0].set_title("Tiempo de Autocorrelación $\\tau_{\\mathrm{int}, E}$ (menor es mejor)")
    axes[0].set_ylabel("$\\tau_{\\mathrm{int}, E}$ (pasos)")

    bars1 = axes[1].bar(labels, ess_s, yerr=ess_s_err, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[1].set_title("Muestras Efectivas por Segundo (ESS/s)")
    axes[1].set_ylabel("ESS / s")

    bars2 = axes[2].bar(labels, esjds, yerr=esjds_err, capsize=4, color=colors, alpha=0.85, edgecolor="black")
    axes[2].set_title("Expected Squared Jump Distance (ESJD)")
    axes[2].set_ylabel("ESJD ($a_0^2$)")

    for b_group in (bars0, bars1, bars2):
        for bar, ht in zip(b_group, hatches):
            if ht:
                bar.set_hatch(ht)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=22)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig4_efficiency_bars",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def plot_hyperparameter_evolution(
    df: pd.DataFrame,
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Grafica la evolución de la precisión y ergodicidad respecto a los hiperparámetros
    (`actor_amplitude`, `tau`, `penalty_factor`) a través de TODOS los modelos evaluados,
    mostrando barras de error estadísticas (`errorbar`) obtenidas de múltiples evaluaciones
    / sub-bloques y remarcando los mejores modelos.
    MALA se grafica con líneas discontinuas (`--`) y RL-MALA con líneas continuas (`-`).
    """
    if df is None or len(df) < 2:
        return

    _apply_plot_style(cfg)
    df_plot = df.copy()
    if "is_neural" not in df_plot.columns:
        hp_list = [_extract_hyperparams(str(lbl)) for lbl in df_plot["label"]]
        for k in ("is_neural", "tau", "actor_amplitude", "penalty_factor", "sigma_action_scale"):
            df_plot[k] = [h[k] for h in hp_list]

    df_mala = df_plot[~df_plot["is_neural"]].sort_values("tau")
    df_rl = df_plot[df_plot["is_neural"]].sort_values(
        ["tau", "penalty_factor", "actor_amplitude"]
    )

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5))
    ax_kl_spin, ax_kl_tot, ax_de = axes[0]
    ax_tau, ax_esjd, ax_ess = axes[1]

    # Estilos por combinación (tau, penalty_factor) para RL-MALA (SIEMPRE línea continua '-')
    rl_groups = list(df_rl.groupby(["tau", "penalty_factor"]))
    group_colors = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
        "#e377c2",
        "#8c564b",
        "#bcbd22",
        "#0066cc",
        "#00994d",
        "#cc3300",
    ]
    group_markers = ["o", "s", "^", "D", "v", "P", "p", "*", "h", "<", ">", "d"]

    def _safe_err(series: pd.Series) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float64)
        return np.where(np.isfinite(arr) & (arr >= 0), arr, 0.0)

    # Paneles 1 a 5: Evolución vs Actor Amplitude A_mu (con MALA en A_mu = 0 y líneas horizontales discontinuas '--')
    mala_taus = sorted(df_mala["tau"].unique())
    mala_palette = ["#333333", "#7f7f7f", "#8c564b", "#c0392b", "#d35400", "#555555", "#990000"]
    mala_colors = {t: mala_palette[i % len(mala_palette)] for i, t in enumerate(mala_taus)}

    # 1. D_KL Spin Promedio vs Actor Amplitude
    if "kl_casscf_spin_mean" in df_plot.columns and df_plot["kl_casscf_spin_mean"].notna().any():
        for _, m_row in df_mala.iterrows():
            t_v = float(m_row["tau"])
            y_v = float(m_row["kl_casscf_spin_mean"])
            y_e = float(m_row.get("kl_casscf_spin_std", 0.0))
            if np.isfinite(y_v):
                ax_kl_spin.axhline(
                    y_v, color=mala_colors.get(t_v, "gray"), linestyle="--", alpha=0.60, lw=1.4
                )
                ax_kl_spin.errorbar(
                    [0.0],
                    [y_v],
                    yerr=[y_e if np.isfinite(y_e) else 0.0],
                    fmt="X",
                    color=mala_colors.get(t_v, "gray"),
                    markersize=8,
                    capsize=4,
                    label=f"MALA -- ($\\tau={t_v:.2f}$)",
                )

        for g_idx, ((t_val, p_val), g_df) in enumerate(rl_groups):
            c = group_colors[g_idx % len(group_colors)]
            mk = group_markers[g_idx % len(group_markers)]
            x_v = g_df["actor_amplitude"].to_numpy(dtype=np.float64)
            y_v = g_df["kl_casscf_spin_mean"].to_numpy(dtype=np.float64)
            y_e = _safe_err(g_df.get("kl_casscf_spin_std", pd.Series(np.zeros(len(g_df)))))
            ax_kl_spin.errorbar(
                x_v,
                y_v,
                yerr=y_e,
                fmt=f"-{mk}",
                color=c,
                lw=2.0,
                markersize=6.5,
                capsize=4,
                label=f"RL-MALA ($\\tau={t_val:.2f}, \\lambda={p_val:.2f}$)",
            )

        best_spin_idx = df_plot["kl_casscf_spin_mean"].idxmin()
        if pd.notna(best_spin_idx):
            b_r = df_plot.loc[best_spin_idx]
            ax_kl_spin.scatter(
                [b_r["actor_amplitude"]],
                [b_r["kl_casscf_spin_mean"]],
                s=220,
                facecolors="none",
                edgecolors="gold",
                linewidths=3.0,
                zorder=10,
                label=f"★ Mejor: {b_r['label']}",
            )
        ax_kl_spin.set_yscale("log")
        ax_kl_spin.set_title(
            r"Cruce Nodal: $D_{\mathrm{KL}}^{\mathrm{spin}} = \frac{1}{2}(D_{\mathrm{KL}}^\uparrow + D_{\mathrm{KL}}^\downarrow)$ vs $A_\mu$",
            weight="bold",
        )
        ax_kl_spin.set_xlabel(r"Actor Amplitude $A_\mu$ ($0 = \mathrm{MALA}$)")
        ax_kl_spin.set_ylabel(r"Spin-Averaged $D_{\mathrm{KL}}$ (log)")
        ax_kl_spin.grid(True, which="both", alpha=0.3)
        ax_kl_spin.legend(fontsize=7.2, loc="best", frameon=True, ncol=1 if len(rl_groups) <= 6 else 2)

    # 2. D_KL Total vs Actor Amplitude
    if "kl_casscf_total_mean" in df_plot.columns and df_plot["kl_casscf_total_mean"].notna().any():
        for _, m_row in df_mala.iterrows():
            t_v = float(m_row["tau"])
            y_v = float(m_row["kl_casscf_total_mean"])
            y_e = float(m_row.get("kl_casscf_total_std", 0.0))
            if np.isfinite(y_v):
                ax_kl_tot.axhline(
                    y_v, color=mala_colors.get(t_v, "gray"), linestyle="--", alpha=0.60, lw=1.4
                )
                ax_kl_tot.errorbar(
                    [0.0],
                    [y_v],
                    yerr=[y_e if np.isfinite(y_e) else 0.0],
                    fmt="X",
                    color=mala_colors.get(t_v, "gray"),
                    markersize=8,
                    capsize=4,
                    label=f"MALA -- ($\\tau={t_v:.2f}$)",
                )
        for g_idx, ((t_val, p_val), g_df) in enumerate(rl_groups):
            c = group_colors[g_idx % len(group_colors)]
            mk = group_markers[g_idx % len(group_markers)]
            x_v = g_df["actor_amplitude"].to_numpy(dtype=np.float64)
            y_v = g_df["kl_casscf_total_mean"].to_numpy(dtype=np.float64)
            y_e = _safe_err(g_df.get("kl_casscf_total_std", pd.Series(np.zeros(len(g_df)))))
            ax_kl_tot.errorbar(
                x_v,
                y_v,
                yerr=y_e,
                fmt=f"-{mk}",
                color=c,
                lw=2.0,
                markersize=6.5,
                capsize=4,
                label=f"RL-MALA ($\\tau={t_val:.2f}, \\lambda={p_val:.2f}$)",
            )
        ax_kl_tot.set_yscale("log")
        ax_kl_tot.set_title(
            r"Precisión de Densidad Total: $D_{\mathrm{KL}}(\rho_{\mathrm{tot}} \parallel \rho_{\mathrm{CASSCF}})$ vs $A_\mu$",
            weight="bold",
        )
        ax_kl_tot.set_xlabel(r"Actor Amplitude $A_\mu$ ($0 = \mathrm{MALA}$)")
        ax_kl_tot.set_ylabel(r"Total Density $D_{\mathrm{KL}}$ (log)")
        ax_kl_tot.grid(True, which="both", alpha=0.3)

    # 3. Precisión Energética: <E_L> - E_CASSCF +/- SE(E) vs Actor Amplitude
    ax_de.axhline(0.0, color="black", linestyle=":", lw=1.8, label="Exact CASSCF ($\\Delta E = 0$)")
    for _, m_row in df_mala.iterrows():
        t_v = float(m_row["tau"])
        de_v = float(m_row.get("delta_E_vs_CASSCF", m_row["E_mean"]))
        se_v = float(m_row.get("SE_E", 0.0))
        if np.isfinite(de_v) and abs(de_v) < 0.05:
            ax_de.errorbar(
                [0.0],
                [de_v * 1e3],
                yerr=[(se_v if np.isfinite(se_v) else 0.0) * 1e3],
                fmt="X",
                color=mala_colors.get(t_v, "gray"),
                markersize=8,
                capsize=4,
                label=f"MALA -- ($\\tau={t_v:.2f}$)",
            )
    for g_idx, ((t_val, p_val), g_df) in enumerate(rl_groups):
        c = group_colors[g_idx % len(group_colors)]
        mk = group_markers[g_idx % len(group_markers)]
        sub = g_df[g_df["delta_E_vs_CASSCF"].abs() < 0.05] if "delta_E_vs_CASSCF" in g_df else g_df
        if len(sub) == 0:
            continue
        x_v = sub["actor_amplitude"].to_numpy(dtype=np.float64)
        y_v = sub["delta_E_vs_CASSCF"].to_numpy(dtype=np.float64) * 1e3
        y_e = _safe_err(sub["SE_E"]) * 1e3
        ax_de.errorbar(
            x_v,
            y_v,
            yerr=y_e,
            fmt=f"-{mk}",
            color=c,
            lw=2.0,
            markersize=6.5,
            capsize=4,
            label=f"RL-MALA ($\\tau={t_val:.2f}, \\lambda={p_val:.2f}$)",
        )
    ax_de.set_title(
        r"Precisión Energética: $(\langle E_L \rangle - E_{\mathrm{CASSCF}}) \pm \mathrm{SE}(E)$ vs $A_\mu$",
        weight="bold",
    )
    ax_de.set_xlabel(r"Actor Amplitude $A_\mu$ ($0 = \mathrm{MALA}$)")
    ax_de.set_ylabel(r"$\Delta E = \langle E_L \rangle - E_{\mathrm{CASSCF}}$ (mHa)")
    ax_de.grid(True, alpha=0.3)

    # 4. Tiempo de Autocorrelación tau_int(E) vs Actor Amplitude
    for _, m_row in df_mala.iterrows():
        t_v = float(m_row["tau"])
        y_v = float(m_row["tau_int_E"])
        y_e = float(m_row.get("tau_int_E_std", 0.0))
        if np.isfinite(y_v):
            ax_tau.axhline(
                y_v, color=mala_colors.get(t_v, "gray"), linestyle="--", alpha=0.60, lw=1.4
            )
            ax_tau.errorbar(
                [0.0],
                [y_v],
                yerr=[y_e if np.isfinite(y_e) else 0.0],
                fmt="X",
                color=mala_colors.get(t_v, "gray"),
                markersize=8,
                capsize=4,
            )
    for g_idx, ((t_val, p_val), g_df) in enumerate(rl_groups):
        c = group_colors[g_idx % len(group_colors)]
        mk = group_markers[g_idx % len(group_markers)]
        x_v = g_df["actor_amplitude"].to_numpy(dtype=np.float64)
        y_v = g_df["tau_int_E"].to_numpy(dtype=np.float64)
        y_e = _safe_err(g_df.get("tau_int_E_std", pd.Series(np.zeros(len(g_df)))))
        ax_tau.errorbar(
            x_v,
            y_v,
            yerr=y_e,
            fmt=f"-{mk}",
            color=c,
            lw=2.0,
            markersize=6.5,
            capsize=4,
            label=f"RL-MALA ($\\tau={t_val:.2f}, \\lambda={p_val:.2f}$)",
        )
    best_tau_idx = df_plot["tau_int_E"].idxmin()
    if pd.notna(best_tau_idx):
        b_r = df_plot.loc[best_tau_idx]
        ax_tau.scatter(
            [b_r["actor_amplitude"]],
            [b_r["tau_int_E"]],
            s=220,
            facecolors="none",
            edgecolors="gold",
            linewidths=3.0,
            zorder=10,
            label=f"★ Mín $\\tau_{{\\mathrm{{int}}}}$: {b_r['label']}",
        )
    ax_tau.set_yscale("log")
    ax_tau.set_title(
        r"Tiempo de Autocorrelación Integrado $\tau_{\mathrm{int}}(E)$ vs $A_\mu$ (menor es mejor)",
        weight="bold",
    )
    ax_tau.set_xlabel(r"Actor Amplitude $A_\mu$ ($0 = \mathrm{MALA}$)")
    ax_tau.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (pasos, log)")
    ax_tau.grid(True, which="both", alpha=0.3)
    ax_tau.legend(fontsize=7.2, loc="best", frameon=True, ncol=1 if len(rl_groups) <= 6 else 2)

    # 5. ESJD vs Actor Amplitude
    for _, m_row in df_mala.iterrows():
        t_v = float(m_row["tau"])
        y_v = float(m_row["ESJD"])
        y_e = float(m_row.get("ESJD_std", 0.0))
        if np.isfinite(y_v):
            ax_esjd.axhline(
                y_v, color=mala_colors.get(t_v, "gray"), linestyle="--", alpha=0.60, lw=1.4
            )
            ax_esjd.errorbar(
                [0.0],
                [y_v],
                yerr=[y_e if np.isfinite(y_e) else 0.0],
                fmt="X",
                color=mala_colors.get(t_v, "gray"),
                markersize=8,
                capsize=4,
            )
    for g_idx, ((t_val, p_val), g_df) in enumerate(rl_groups):
        c = group_colors[g_idx % len(group_colors)]
        mk = group_markers[g_idx % len(group_markers)]
        x_v = g_df["actor_amplitude"].to_numpy(dtype=np.float64)
        y_v = g_df["ESJD"].to_numpy(dtype=np.float64)
        y_e = _safe_err(g_df.get("ESJD_std", pd.Series(np.zeros(len(g_df)))))
        ax_esjd.errorbar(
            x_v,
            y_v,
            yerr=y_e,
            fmt=f"-{mk}",
            color=c,
            lw=2.0,
            markersize=6.5,
            capsize=4,
        )
    ax_esjd.set_title(
        r"Movilidad en Fase: $\mathrm{ESJD}$ ($\mathrm{Bohr}^2$) vs $A_\mu$ (mayor es mejor)",
        weight="bold",
    )
    ax_esjd.set_xlabel(r"Actor Amplitude $A_\mu$ ($0 = \mathrm{MALA}$)")
    ax_esjd.set_ylabel(r"$\mathrm{ESJD}$ ($\mathrm{Bohr}^2$)")
    ax_esjd.grid(True, alpha=0.3)

    # 6. Comparativa Directa MALA (discontinua '--') vs RL-MALA (continua '-') en función de tau (ESS/s)
    if len(df_mala) > 0:
        ax_ess.errorbar(
            df_mala["tau"].to_numpy(dtype=np.float64),
            df_mala["ESS_per_sec"].to_numpy(dtype=np.float64),
            yerr=_safe_err(df_mala.get("ESS_per_sec_std", pd.Series(np.zeros(len(df_mala))))),
            fmt="--X",
            color="#e74c3c",
            lw=2.5,
            markersize=9,
            capsize=5,
            label="MALA Clásico -- (barrido en $\\tau$)",
        )
    for g_idx, ((amp_val, p_val), g_df) in enumerate(
        df_rl.groupby(["actor_amplitude", "penalty_factor"])
    ):
        c = group_colors[g_idx % len(group_colors)]
        mk = group_markers[g_idx % len(group_markers)]
        g_sorted = g_df.sort_values("tau")
        ax_ess.errorbar(
            g_sorted["tau"].to_numpy(dtype=np.float64),
            g_sorted["ESS_per_sec"].to_numpy(dtype=np.float64),
            yerr=_safe_err(g_sorted.get("ESS_per_sec_std", pd.Series(np.zeros(len(g_sorted))))),
            fmt=f"-{mk}",
            color=c,
            lw=2.0,
            markersize=6.5,
            capsize=4,
            label=f"RL-MALA - ($A_\\mu={amp_val:.2f}, \\lambda={p_val:.2f}$)",
        )
    best_ess_idx = df_plot["ESS_per_sec"].idxmax()
    if pd.notna(best_ess_idx):
        b_r = df_plot.loc[best_ess_idx]
        ax_ess.scatter(
            [b_r["tau"]],
            [b_r["ESS_per_sec"]],
            s=220,
            facecolors="none",
            edgecolors="gold",
            linewidths=3.0,
            zorder=10,
            label=f"★ Máx ESS/s: {b_r['label']}",
        )
    ax_ess.set_title(
        r"Velocidad Efectiva Real: $\mathrm{ESS/s}$ vs Paso Base $\tau$ (MALA -- vs RL-MALA -)",
        weight="bold",
    )
    ax_ess.set_xlabel(r"Paso Base de Propuesta $\tau$")
    ax_ess.set_ylabel(r"$\mathrm{ESS} / \mathrm{s}$ ($\mathrm{s}^{-1}$)")
    ax_ess.grid(True, alpha=0.3)
    ax_ess.legend(fontsize=7.2, loc="best", frameon=True, ncol=1 if len(rl_groups) <= 6 else 2)

    sys_tag = getattr(cfg, "system_tag", "")
    fig.suptitle(
        f"Evolución de Precisión, Cruce Nodal y Eficiencia vs Hiperparámetros ({sys_tag or '1D Wigner'})",
        fontsize=15,
        weight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig9_hyperparameter_evolution",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=sys_tag,
        cfg=cfg,
    )


def plot_time_to_solution(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
) -> None:
    valid = [
        b
        for b in bundles_info
        if "e_step_series" in b["diagnostics"].get("summary", {})
    ]
    if not valid:
        return

    _apply_plot_style(cfg)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for idx, binfo in enumerate(valid):
        summary = binfo["diagnostics"]["summary"]
        e_series = summary["e_step_series"]
        wtime = max(1e-6, summary.get("wall_time_sec", 1.0))
        tau_e = max(1.0, summary.get("tau_int_E", 1.0))
        t_grid = np.linspace(wtime / len(e_series), wtime, len(e_series))
        se_t = np.std(e_series) * np.sqrt(tau_e / np.arange(1, len(e_series) + 1))
        color, ls = _get_model_style(binfo, idx)
        ax.plot(
            t_grid,
            se_t,
            color=color,
            linestyle=ls,
            lw=2.2,
            label=f"{binfo['label']} ($T_\\epsilon={summary['time_to_solution_sec']:.2f}$ s)",
        )

    ax.axhline(
        cfg.target_se_epsilon,
        color="black",
        linestyle=":",
        lw=1.4,
        label=f"Umbral $\\epsilon = {cfg.target_se_epsilon:.1e}$ Ha",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tiempo de Reloj en Pared (s)")
    ax.set_ylabel("Error Estándar $\\mathrm{SE}(E)$ (Ha)")
    ax.set_title("Time-to-Solution: Error Estándar vs Tiempo de Cómputo")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)
    _save_fig(
        fig,
        out_dir,
        "fig5_time_to_solution",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def plot_walker_trace(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Grafica la trayectoria de un caminante individual (Best MALA vs Best RL-MALA por defecto)
    junto con su histograma marginal de densidad por espín, superponiendo la densidad analítica
    CASSCF si existe para el sistema.
    """
    valid = [
        b
        for b in bundles_info
        if b.get("walker_trace_data") is not None
        or (
            "walkers" in b["bundle"].get("production", {})
            and np.asarray(b["bundle"]["production"]["walkers"]).size > 0
        )
    ]
    if not valid:
        return

    _apply_plot_style(cfg)
    n_models = len(valid)
    fig = plt.figure(figsize=(14.0, 4.8 * n_models))
    outer_gs = gridspec.GridSpec(n_models, 1, hspace=0.34)

    for idx, binfo in enumerate(valid):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[idx], width_ratios=[4, 1], wspace=0.05
        )
        ax_main = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1], sharey=ax_main)

        w_data = binfo.get("walker_trace_data")
        if w_data is None:
            sys_cfg = binfo["bundle"].get("config", {}).get("system", {})
            n_up_tmp = int(sys_cfg.get("n_up", 2))
            n_dn_tmp = int(sys_cfg.get("n_dn", sys_cfg.get("n_down", 2)))
            x_max_tmp = float(
                binfo.get("observables", {}).get("x_max", 15.0)
                if binfo.get("observables")
                else 15.0
            )
            w_data = _precompute_walker_trace_for_bundle(
                binfo["bundle"]["production"]["walkers"],
                n_up=n_up_tmp,
                n_dn=n_dn_tmp,
                x_max=x_max_tmp,
            )
        if w_data is None:
            continue

        steps_sub = w_data["steps_sub"]
        w_single_sub = w_data["w_single_sub"]
        n_elec = w_data["n_elec"]
        n_up = w_data["n_up"]
        n_dn = w_data["n_dn"]
        x_max = w_data["x_max"]
        _, ls_model = _get_model_style(binfo, idx)
        is_neural = ls_model == "-"
        role_tag = "★ BEST RL-MALA (Solid)" if is_neural else "★ BEST MALA (Dashed)"

        for e_i in range(n_elec):
            c_spin = "#2980b9" if e_i < n_up else "#e74c3c"
            lbl = (
                r"Spin $\uparrow$ (Single Walker #0)"
                if e_i == 0
                else (r"Spin $\downarrow$ (Single Walker #0)" if e_i == n_up else None)
            )
            ax_main.plot(
                steps_sub,
                w_single_sub[:, e_i],
                color=c_spin,
                linestyle=ls_model,
                alpha=0.78 if is_neural else 0.72,
                lw=0.95,
                label=lbl,
            )

        summary = binfo.get("diagnostics", {}).get("summary", {})
        unc = binfo.get("uncertainties", {})
        tau_val = summary.get("tau_int_E", float("nan"))
        kl_spin_val = unc.get("kl_spin_mean", float("nan"))
        metrics_str = f"$\\tau_{{\\mathrm{{int}}}}(E)={tau_val:.1f}$"
        if np.isfinite(kl_spin_val):
            metrics_str += f", $D_{{\\mathrm{{KL}}}}^{{\\mathrm{{spin}}}}={kl_spin_val:.4f}$"

        ax_main.set_title(
            f"{role_tag}: Single-Walker Trace — {binfo['label']} ({metrics_str})",
            loc="left",
            fontsize=11.0,
            weight="bold",
        )
        ax_main.set_xlabel("MCMC Production Steps")
        ax_main.set_ylabel("Electron Position $x_i(t)$ (Bohr)")
        ax_main.set_ylim(-x_max, x_max)
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc="upper right", fontsize=8.5, frameon=True)

        # Panel derecho: histograma marginal precalculado
        bin_centers = w_data["bin_centers"]
        bin_h = float(bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.2
        if n_up > 0 and np.any(w_data["up_hist"] > 0):
            ax_hist.barh(
                bin_centers,
                w_data["up_hist"],
                height=bin_h,
                color="#3498db",
                alpha=0.45,
                label=r"Sampled $\uparrow$",
            )
        if n_dn > 0 and np.any(w_data["dn_hist"] > 0):
            ax_hist.barh(
                bin_centers,
                w_data["dn_hist"],
                height=bin_h,
                color="#e74c3c",
                alpha=0.45,
                label=r"Sampled $\downarrow$",
            )

        if casscf_ref is not None:
            x_ex = casscf_ref["x"]
            mask_ex = (x_ex >= -x_max) & (x_ex <= x_max)
            if n_up > 0:
                ax_hist.plot(
                    casscf_ref["p_up"][mask_ex],
                    x_ex[mask_ex],
                    color="#2980b9",
                    linestyle=":",
                    lw=2.2,
                    label=r"Exact $\uparrow$",
                )
            if n_dn > 0:
                ax_hist.plot(
                    casscf_ref["p_down"][mask_ex],
                    x_ex[mask_ex],
                    color="#c0392b",
                    linestyle=":",
                    lw=2.2,
                    label=r"Exact $\downarrow$",
                )

        ax_hist.set_title("Marginal Density", weight="bold")
        ax_hist.set_xlabel(r"Normalized $p_\sigma(x)$")
        ax_hist.tick_params(axis="y", labelleft=False)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=8.0, frameon=True)

    _save_fig(
        fig,
        out_dir,
        "fig6_walker_traces",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
        cfg=cfg,
    )


def _resolve_casscf_reference_for_bundle(
    sys_cfg: dict[str, Any],
    cfg: PostProcessConfig,
    x_max: float = 15.0,
) -> dict[str, Any] | None:
    """
    Busca y carga una baseline CASSCF analítica para el sistema físico `sys_cfg`.
    Si no existe ninguna baseline para (N_up, N_dn, omega, g), retorna `None` sin lanzar error.
    """
    if cfg.casscf_npz_path and os.path.exists(cfg.casscf_npz_path):
        try:
            return compute_exact_casscf_1body_density(
                cfg.casscf_npz_path, x_min=-x_max, x_max=x_max, n_grid=500
            )
        except Exception as exc:
            print(f"  [Aviso] No se pudo cargar casscf_npz_path={cfg.casscf_npz_path}: {exc}")
            return None

    if not cfg.auto_casscf_baseline:
        return None

    search_root = cfg.uhf_search_root or "."
    baselines = discover_casscf_baselines(search_root=search_root)
    if not baselines:
        return None

    n_up = int(sys_cfg.get("n_up", 2))
    n_dn = int(sys_cfg.get("n_dn", sys_cfg.get("n_down", 2)))
    omega = float(sys_cfg.get("omega", 0.1))
    g = float(sys_cfg.get("interaction_strength", sys_cfg.get("g", 1.0)))
    softening = float(sys_cfg.get("softening", 0.1))

    match = find_matching_casscf_baseline(
        baselines, n_up=n_up, n_dn=n_dn, omega=omega, g=g, softening=softening
    )
    if match is None:
        return None

    try:
        return compute_exact_casscf_1body_density(
            match["npz_path"], x_min=-x_max, x_max=x_max, n_grid=500
        )
    except Exception as exc:
        print(f"  [Aviso] Falló cálculo analítico de baseline CASSCF ({match['npz_path']}): {exc}")
        return None


def run_postprocessing_pipeline(cfg: PostProcessConfig) -> pd.DataFrame:
    """
    Pipeline Modular de Post-Procesamiento:
    1. Recibe uno o varios archivos unificados `.h5`.
    2. Calcula diagnósticos MCMC (ACF, IAT de Sokal, ESS, ESS/s, G_RL, ESJD, Time-to-Solution).
    3. Calcula observables físicos 1D con incertidumbre entre evaluaciones independientes.
    4. Si el sistema físico tiene una baseline CASSCF pre-entrenada en `uhf*`, la carga
       automáticamente, calcula D_KL resuelta por espín y la superpone en las gráficas;
       si no existe baseline CASSCF, grafica limpiamente sólo los resultados evaluados.
    5. Exporta `summary_metrics.csv`, `summary_metrics.json` y perfiles `observables_<label>.csv`.
    """
    if not cfg.inputs:
        raise ValueError("Se requiere al menos un archivo `.h5` en `--inputs`.")

    os.makedirs(cfg.out_dir, exist_ok=True)
    labels = list(cfg.labels) if cfg.labels else []
    while len(labels) < len(cfg.inputs):
        idx = len(labels)
        p = cfg.inputs[idx]
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem == "run_bundle":
            parent = os.path.basename(os.path.dirname(os.path.abspath(p)))
            if parent:
                stem = parent
        labels.append(stem)

    bundles_info = []
    summary_rows = []
    baseline_tau_e = None
    casscf_ref: dict[str, Any] | None = None
    casscf_checked = False

    for h5_path, label in zip(cfg.inputs, labels):
        print(f"[PostProcess] Procesando '{label}' desde {h5_path}...")
        bundle = load_run_bundle_h5(h5_path, load_walkers=True)
        sys_cfg = bundle.get("config", {}).get("system", {})
        n_up = int(sys_cfg.get("n_up", 2))
        n_dn = int(sys_cfg.get("n_dn", sys_cfg.get("n_down", 2)))
        omega = float(sys_cfg.get("omega", 0.1))
        g_val = float(sys_cfg.get("interaction_strength", sys_cfg.get("g", 1.0)))
        if not getattr(cfg, "system_tag", ""):
            prefix = ""
            if "pipeline2" in cfg.out_dir:
                prefix = "P2_"
            elif "pipeline3" in cfg.out_dir:
                prefix = "P3_"
            cfg.system_tag = f"{prefix}N{n_up + n_dn}_w{omega}_g{g_val}"
        sys_tag = cfg.system_tag

        diag = (
            summarize_bundle_diagnostics(bundle, cfg)
            if cfg.compute_diagnostics
            else {"per_chain": [], "summary": {}}
        )
        summary = diag.get("summary", {})

        obs = None
        kl_div_2e = float("nan")
        prod = bundle.get("production", {})
        w_all = None
        if cfg.compute_observables and "walkers" in prod and np.asarray(prod["walkers"]).size > 0:
            w_all = np.asarray(prod["walkers"])
            obs = compute_1d_observables(
                w_all,
                n_up=n_up,
                n_down=n_dn,
                n_bins=cfg.n_density_bins,
                x_max=cfg.density_x_max,
                omega=omega,
            )
            if cfg.save_observables_csv:
                safe_lbl = "".join(
                    c if c.isalnum() or c in ("_", "-") else "_" for c in label
                )
                df_obs = pd.DataFrame(
                    {
                        "x_grid": obs["x_grid"],
                        "rho_total": obs["rho_total"],
                        "rho_total_std": obs["rho_total_std"],
                        "rho_up": obs["rho_up"],
                        "rho_up_std": obs["rho_up_std"],
                        "rho_down": obs["rho_down"],
                        "rho_down_std": obs["rho_down_std"],
                        "r_grid": obs["r_grid"],
                        "g_pair_total": obs["g_pair_total"],
                        "g_pair_same": obs["g_pair_same"],
                        "g_pair_opp": obs["g_pair_opp"],
                    }
                )
                obs_fname = (
                    f"observables_{safe_lbl}_{sys_tag}.csv"
                    if (sys_tag and not safe_lbl.endswith(sys_tag))
                    else f"observables_{safe_lbl}.csv"
                )
                obs_csv_path = os.path.join(cfg.out_dir, obs_fname)
                df_obs.to_csv(obs_csv_path, index=False)
                if getattr(cfg, "csv_collect_dir", ""):
                    import shutil

                    os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                    shutil.copy2(obs_csv_path, os.path.join(cfg.csv_collect_dir, obs_fname))

            if cfg.plot_kl_divergence and (n_up + n_dn) == 2:
                ckpts = bundle.get("checkpoints", {})
                ans_p = ckpts.get("ansatz_best", ckpts.get("ansatz_final", None))
                if ans_p is not None:
                    try:
                        exp_cfg = Experiment1DConfig.from_dict(bundle["config"])
                        ansatz = build_ansatz_1d(exp_cfg)
                        kl_div_2e = compute_kl_divergence_2e(
                            w_all, ansatz.apply, ans_p, x_max=obs["x_max"]
                        )
                    except Exception as exc:
                        print(f"  [Aviso] Divergencia KL 2e omitida para {label}: {exc}")

        # Resolver baseline CASSCF pre-entrenada una sola vez para el sistema físico
        if not casscf_checked:
            eff_xmax = float(obs["x_max"]) if obs is not None else float(cfg.density_x_max)
            casscf_ref = _resolve_casscf_reference_for_bundle(sys_cfg, cfg, x_max=eff_xmax)
            casscf_checked = True
            if casscf_ref is not None:
                print(
                    f"[PostProcess] Baseline CASSCF encontrada: {casscf_ref['label']} "
                    f"(E={casscf_ref['energy']:.6f} Ha) -> {casscf_ref['npz_path']}"
                )
                if cfg.save_observables_csv:
                    df_cas_ex = pd.DataFrame(
                        {
                            "x": casscf_ref["x"],
                            "rho_total": casscf_ref["rho_total"],
                            "rho_up": casscf_ref["rho_up"],
                            "rho_down": casscf_ref["rho_down"],
                            "p_total": casscf_ref["p_total"],
                            "p_up": casscf_ref["p_up"],
                            "p_down": casscf_ref["p_down"],
                        }
                    )
                    cas_fname = (
                        f"casscf_exact_density_{sys_tag}.csv"
                        if sys_tag
                        else "casscf_exact_density.csv"
                    )
                    cas_csv_path = os.path.join(cfg.out_dir, cas_fname)
                    df_cas_ex.to_csv(cas_csv_path, index=False)
                    if getattr(cfg, "csv_collect_dir", ""):
                        import shutil

                        os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                        shutil.copy2(cas_csv_path, os.path.join(cfg.csv_collect_dir, cas_fname))
            else:
                print(
                    f"[PostProcess] Sin baseline CASSCF pre-entrenada para "
                    f"(N_up={n_up}, N_dn={n_dn}, omega={omega}, g={sys_cfg.get('interaction_strength', 1.0)}). "
                    f"Graficando resultados sin baseline CASSCF."
                )

        # Extraer hiperparámetros e incertidumbres multi-evaluación / Jackknife
        hp_info = _extract_hyperparams(label, bundle)
        eff_xmax = float(obs["x_max"]) if obs is not None else float(cfg.density_x_max)
        unc_info = _compute_multi_eval_uncertainties(
            w_all=w_all,
            bundle=bundle,
            summary=summary,
            casscf_ref=casscf_ref,
            cfg=cfg,
            x_max=eff_xmax,
        )

        # Precalcular resúmenes compactos de curva KL y traza de 1 caminante y liberar w_all de RAM
        kl_curve_data = (
            _precompute_kl_curve_for_bundle(w_all, casscf_ref, cfg, x_max=eff_xmax)
            if (cfg.plot_observables_comparison and casscf_ref is not None)
            else None
        )
        walker_trace_data = (
            _precompute_walker_trace_for_bundle(
                w_all, n_up=n_up, n_dn=n_dn, x_max=eff_xmax
            )
            if cfg.plot_walker_trace
            else None
        )
        if "production" in bundle and "walkers" in bundle["production"]:
            del bundle["production"]["walkers"]
        if w_all is not None:
            del w_all
        gc.collect()

        tau_e = summary.get("tau_int_E", float("nan"))
        if baseline_tau_e is None and not np.isnan(tau_e) and tau_e > 0:
            baseline_tau_e = tau_e
        g_rl = (
            float(baseline_tau_e / tau_e)
            if (baseline_tau_e is not None and not np.isnan(tau_e) and tau_e > 0)
            else 1.0
        )

        e_mean_val = summary.get("E_mean", float("nan"))
        cas_e_val = casscf_ref["energy"] if casscf_ref is not None else float("nan")
        delta_e_cas = (
            float(e_mean_val - cas_e_val)
            if (np.isfinite(e_mean_val) and np.isfinite(cas_e_val))
            else float("nan")
        )

        row = {
            "label": label,
            "h5_path": str(h5_path),
            "is_neural": hp_info["is_neural"],
            "proposal_type": hp_info["proposal_type"],
            "tau": hp_info["tau"],
            "actor_amplitude": hp_info["actor_amplitude"],
            "penalty_factor": hp_info["penalty_factor"],
            "sigma_action_scale": hp_info["sigma_action_scale"],
            "N": int(n_up + n_dn),
            "n_up": int(n_up),
            "n_dn": int(n_dn),
            "omega": float(omega),
            "g": float(sys_cfg.get("interaction_strength", sys_cfg.get("g", 1.0))),
            "softening": float(sys_cfg.get("softening", 0.1)),
            "n_chains": summary.get("n_chains", 0),
            "E_mean": e_mean_val,
            "SE_E": summary.get("SE_E", float("nan")),
            "E_var": summary.get("E_var", float("nan")),
            "T_mean": summary.get("T_mean", float("nan")),
            "V_mean": summary.get("V_mean", float("nan")),
            "acceptance": summary.get("acceptance", float("nan")),
            "ESJD": summary.get("ESJD", float("nan")),
            "ESJD_std": unc_info["ESJD_std"],
            "accepted_disp": summary.get("accepted_disp", float("nan")),
            "tau_int_E": tau_e,
            "tau_int_E_std": unc_info["tau_int_E_std"],
            "tau_int_pos": summary.get("tau_int_pos", float("nan")),
            "G_RL_speedup": g_rl,
            "ESS": summary.get("ESS", float("nan")),
            "ESS_per_sec": summary.get("ESS_per_sec", float("nan")),
            "ESS_per_sec_std": unc_info["ESS_per_sec_std"],
            "wall_time_sec": summary.get("wall_time_sec", float("nan")),
            "time_to_solution_sec": summary.get("time_to_solution_sec", float("nan")),
            "n_wigner_peaks": obs["n_wigner_peaks"] if obs else -1,
            "rms_radius": obs["rms_radius"] if obs else float("nan"),
            "mean_pair_dist": obs["mean_pair_dist"] if obs else float("nan"),
            "has_casscf_baseline": bool(casscf_ref is not None),
            "casscf_energy": cas_e_val,
            "delta_E_vs_CASSCF": delta_e_cas,
            "kl_casscf_total_mean": unc_info["kl_tot_mean"],
            "kl_casscf_total_std": unc_info["kl_tot_std"],
            "kl_casscf_up_mean": unc_info["kl_up_mean"],
            "kl_casscf_up_std": unc_info["kl_up_std"],
            "kl_casscf_down_mean": unc_info["kl_dn_mean"],
            "kl_casscf_down_std": unc_info["kl_dn_std"],
            "kl_casscf_spin_mean": unc_info["kl_spin_mean"],
            "kl_casscf_spin_std": unc_info["kl_spin_std"],
            "kl_divergence_2e": kl_div_2e,
        }
        summary_rows.append(row)
        bundles_info.append(
            {
                "label": label,
                "h5_path": str(h5_path),
                "bundle": bundle,
                "diagnostics": diag,
                "observables": obs,
                "hyperparams": hp_info,
                "uncertainties": unc_info,
                "kl_curve_data": kl_curve_data,
                "walker_trace_data": walker_trace_data,
            }
        )

    # Enriquecer cada summary.json individual de los modelos evaluados con métricas comparativas
    baseline_ess_s = summary_rows[0]["ESS_per_sec"] if summary_rows else float("nan")
    baseline_e = summary_rows[0]["E_mean"] if summary_rows else float("nan")
    for r in summary_rows:
        ess_s_val = r.get("ESS_per_sec", float("nan"))
        r["ESS_speedup_vs_baseline"] = (
            float(ess_s_val / baseline_ess_s)
            if (np.isfinite(ess_s_val) and np.isfinite(baseline_ess_s) and baseline_ess_s > 0)
            else 1.0
        )
        r["delta_E_vs_baseline"] = (
            float(r["E_mean"] - baseline_e)
            if (np.isfinite(r["E_mean"]) and np.isfinite(baseline_e))
            else 0.0
        )
        try:
            bundle_dir = os.path.dirname(os.path.abspath(r["h5_path"]))
            indiv_summary_path = os.path.join(bundle_dir, "summary.json")
            if os.path.exists(indiv_summary_path):
                with open(indiv_summary_path, "r", encoding="utf-8") as f_in:
                    indiv_data = json.load(f_in)
                indiv_data["comparative_postprocess"] = {
                    "G_RL_speedup": r["G_RL_speedup"],
                    "ESS_speedup_vs_baseline": r["ESS_speedup_vs_baseline"],
                    "delta_E_vs_baseline": r["delta_E_vs_baseline"],
                    "postprocess_out_dir": str(cfg.out_dir),
                }
                with open(indiv_summary_path, "w", encoding="utf-8") as f_out:
                    json.dump(indiv_data, f_out, indent=2)
        except Exception:
            pass

    df = pd.DataFrame(summary_rows)
    sys_tag = getattr(cfg, "system_tag", "")
    if cfg.save_summary_csv:
        csv_fname = f"summary_metrics_{sys_tag}.csv" if sys_tag else "summary_metrics.csv"
        json_fname = f"summary_{sys_tag}.json" if sys_tag else "summary.json"
        csv_path = os.path.join(cfg.out_dir, csv_fname)
        json_path = os.path.join(cfg.out_dir, json_fname)
        df.to_csv(csv_path, index=False)
        summary_dict_payload = {
            "system_tag": sys_tag,
            "out_dir": str(cfg.out_dir),
            "casscf_reference": {
                "has_casscf_baseline": bool(casscf_ref is not None),
                "npz_path": str(casscf_ref["npz_path"]) if casscf_ref is not None else None,
                "energy": float(casscf_ref["energy"]) if casscf_ref is not None else None,
            },
            "models": summary_rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict_payload, f, indent=2)
        if getattr(cfg, "csv_collect_dir", ""):
            import shutil

            os.makedirs(cfg.csv_collect_dir, exist_ok=True)
            shutil.copy2(csv_path, os.path.join(cfg.csv_collect_dir, csv_fname))
        print(f"[PostProcess] Tabla resumen guardada en: {csv_path}")

    if getattr(cfg, "master_comparison_csv", None):
        try:
            import fcntl

            m_csv = os.path.abspath(cfg.master_comparison_csv)
            os.makedirs(os.path.dirname(m_csv) or ".", exist_ok=True)
            lock_file = m_csv + ".lock"
            with open(lock_file, "w") as f_lock:
                fcntl.flock(f_lock, fcntl.LOCK_EX)
                try:
                    write_hdr = not os.path.exists(m_csv) or os.path.getsize(m_csv) == 0
                    df.to_csv(m_csv, mode="a", header=write_hdr, index=False)
                    if getattr(cfg, "csv_collect_dir", ""):
                        import shutil

                        os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                        shutil.copy2(m_csv, os.path.join(cfg.csv_collect_dir, os.path.basename(m_csv)))
                finally:
                    fcntl.flock(f_lock, fcntl.LOCK_UN)
            print(f"[PostProcess] Comparativa agregada en vivo a: {m_csv}")
        except Exception as exc:
            print(f"  [Aviso] No se pudo escribir en master_comparison_csv: {exc}")

    def _safe_run_plot(fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"  [PostProcess Advertencia] No se pudo generar {fn.__name__}: {exc}")

    # 1. Graficar evolución respecto a hiperparámetros usando TODOS los modelos + barras de error
    if getattr(cfg, "plot_hyperparameter_evolution", True) and len(df) >= 2:
        _safe_run_plot(
            plot_hyperparameter_evolution, df, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )

    # 2. Filtrar los Top-K mejores modelos MALA y Top-K mejores RL-MALA para las figuras comparativas
    top_k = int(getattr(cfg, "top_k_models", 2))
    plot_bundles = _select_top_k_bundles(bundles_info, summary_rows, top_k=top_k)
    if len(plot_bundles) < len(bundles_info):
        sel_names = [b["label"] for b in plot_bundles]
        print(
            f"[PostProcess] Filtrando Top-{top_k} MALA vs Top-{top_k} RL-MALA "
            f"({len(plot_bundles)}/{len(bundles_info)} modelos) para figuras comparativas: {sel_names}"
        )

    if cfg.plot_training_metrics:
        _safe_run_plot(plot_training_dashboard, plot_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref)
    if cfg.plot_production_summary:
        _safe_run_plot(plot_production_summary, plot_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref)
    if cfg.plot_observables_comparison:
        _safe_run_plot(plot_observables_comparison, plot_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref)
        _safe_run_plot(
            plot_article_densities_comparison, plot_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )
        _safe_run_plot(
            plot_article_kl_divergence, plot_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )
    if cfg.plot_efficiency_bars:
        _safe_run_plot(plot_efficiency_bars, plot_bundles, cfg.out_dir, cfg)
    if cfg.plot_time_to_solution:
        _safe_run_plot(plot_time_to_solution, plot_bundles, cfg.out_dir, cfg)
    if cfg.plot_walker_trace:
        w_top_k = int(getattr(cfg, "walker_trace_top_k", 1))
        walker_bundles = _select_top_k_bundles(
            bundles_info, summary_rows, top_k=w_top_k, verbose=False
        )
        _safe_run_plot(
            plot_walker_trace, walker_bundles, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )

    print("\n" + "=" * 76)
    print("[PostProcess] Resumen Comparativo de Corridas:")
    cols = [
        "label",
        "E_mean",
        "SE_E",
        "acceptance",
        "ESJD",
        "tau_int_E",
        "G_RL_speedup",
        "ESS_per_sec",
    ]
    if casscf_ref is not None:
        cols.extend(
            [
                "casscf_energy",
                "delta_E_vs_CASSCF",
                "kl_casscf_total_mean",
                "kl_casscf_spin_mean",
            ]
        )
    print(df[cols].to_string(index=False))
    print("=" * 76)
    return df
