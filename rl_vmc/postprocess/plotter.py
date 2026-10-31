from __future__ import annotations

import json
import os
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
from rl_vmc.postprocess.diagnostics import summarize_bundle_diagnostics
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
    "#e74c3c",  # Rojo (estilo MALA en scripts legacy)
    "#2ecc71",  # Verde (estilo RL-MALA en scripts legacy)
    "#1f77b4",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
]


def _save_fig(
    fig: plt.Figure,
    out_dir: str,
    stem: str,
    fmt: str = "pdf",
    dpi: int = 300,
    sys_tag: str = "",
) -> None:
    stems = [stem]
    if sys_tag and not stem.endswith(f"_{sys_tag}"):
        stems.append(f"{stem}_{sys_tag}")
    for st in stems:
        if fmt in ("png", "both"):
            fig.savefig(os.path.join(out_dir, f"{st}.png"), dpi=dpi, bbox_inches="tight")
        if fmt in ("pdf", "both"):
            fig.savefig(os.path.join(out_dir, f"{st}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training_dashboard(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    valid = [b for b in bundles_info if b["bundle"].get("training_df") is not None]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for idx, binfo in enumerate(valid):
        df_t = binfo["bundle"]["training_df"]
        label = binfo["label"]
        color = PALETTE[idx % len(PALETTE)]
        steps = df_t["Step"] if "Step" in df_t.columns else np.arange(len(df_t))

        if "Energy_Mean" in df_t.columns:
            axes[0, 0].plot(steps, df_t["Energy_Mean"], color=color, lw=1.6, label=label)
        if "Energy_Var" in df_t.columns:
            axes[0, 1].plot(steps, df_t["Energy_Var"], color=color, lw=1.6, label=label)
        if "Acceptance" in df_t.columns:
            axes[1, 0].plot(steps, df_t["Acceptance"], color=color, lw=1.6, label=label)
        if "ESJD" in df_t.columns:
            axes[1, 1].plot(steps, df_t["ESJD"], color=color, lw=1.6, label=label)

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

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for idx, binfo in enumerate(valid):
        summary = binfo["diagnostics"]["summary"]
        label = binfo["label"]
        color = PALETTE[idx % len(PALETTE)]
        e_series = summary["e_step_series"]
        steps = np.arange(len(e_series))

        # 1. Traza y media acumulada
        axes[0, 0].plot(steps, e_series, color=color, alpha=0.25, lw=0.8)
        cum_mean = np.cumsum(e_series) / (steps + 1)
        axes[0, 0].plot(
            steps,
            cum_mean,
            color=color,
            lw=2.0,
            label=f"{label} ($\\langle E\\rangle={summary['E_mean']:.5f}\\pm{summary['SE_E']:.5f}$)",
        )

        # 2. Función de autocorrelación ACF
        acf_e = summary["acf_energy"]
        lags = np.arange(len(acf_e))
        axes[0, 1].plot(
            lags,
            acf_e,
            color=color,
            lw=2.0,
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
                            alpha=0.45,
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
            axes[1, 1].plot(steps[valid_se] + 1, se_curve[valid_se], color=color, lw=2.0, label=label)

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

    axes[0, 1].axhline(0.0, color="gray", linestyle="--", lw=1.0)
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

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for idx, binfo in enumerate(valid):
        obs = binfo["observables"]
        label = binfo["label"]
        color = PALETTE[idx % len(PALETTE)]
        x = obs["x_grid"]
        r = obs["r_grid"]

        axes[0].plot(
            x,
            obs["rho_total"],
            color=color,
            lw=2.0,
            label=f"{label} (picos={obs['n_wigner_peaks']})",
        )
        if obs.get("n_chains_eval", 1) > 1 and "rho_total_std" in obs:
            axes[0].fill_between(
                x,
                np.maximum(0.0, obs["rho_total"] - obs["rho_total_std"]),
                obs["rho_total"] + obs["rho_total_std"],
                color=color,
                alpha=0.18,
            )

        axes[1].plot(
            x, obs["rho_up"], color=color, lw=1.8, linestyle="-", label=f"{label} $\\uparrow$"
        )
        axes[1].plot(
            x, obs["rho_down"], color=color, lw=1.8, linestyle="--", label=f"{label} $\\downarrow$"
        )
        axes[2].plot(
            r,
            obs["g_pair_total"],
            color=color,
            lw=2.0,
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
    )


def plot_article_densities_comparison(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Genera la gráfica comparativa de densidades estilo Artículo (`Article_Densities.pdf` de
    `scripts/evaluar_wigner_casscf_2.py`), incluyendo bandas de incertidumbre +/- 1 sigma
    entre corridas independientes y superponiendo la baseline CASSCF exacta si existe.
    Si no existe baseline CASSCF, grafica limpiamente las densidades muestreadas sin baseline.
    """
    valid = [b for b in bundles_info if b.get("observables") is not None]
    if not valid:
        return

    n_panels = len(valid)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(7.0 * n_panels, 5.8), sharey=True, squeeze=False
    )

    for idx, binfo in enumerate(valid):
        ax = axes[0, idx]
        obs = binfo["observables"]
        sys_cfg = binfo["bundle"].get("config", {}).get("system", {})
        omega = float(sys_cfg.get("omega", 0.1))
        x = obs["x_grid"]
        x_max = float(obs.get("x_max", 15.0))
        n_chains = int(obs.get("n_chains_eval", 1))

        # Densidad total muestreada + banda +/- 1 sigma entre evaluaciones
        ax.plot(
            x,
            obs["rho_total"],
            "k-",
            lw=2.5,
            label=f"Sampled Total ($N_{{\\mathrm{{eval}}}}={n_chains}$)",
        )
        if n_chains > 1 and "rho_total_std" in obs:
            ax.fill_between(
                x,
                np.maximum(0.0, obs["rho_total"] - obs["rho_total_std"]),
                obs["rho_total"] + obs["rho_total_std"],
                color="black",
                alpha=0.18,
                label=r"Total $\pm 1\sigma$ (evals)",
            )

        # Densidades resueltas por espín (rellenas como en evaluar_wigner_casscf_2.py)
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
            espacio, potencial, "--", color="gray", lw=1.5, alpha=0.6, label="V(x)"
        )
        ax_pot.set_yticks([])

        ax.set_xlim(-x_max, x_max)
        ax.set_title(binfo["label"], fontsize=13, weight="bold")
        ax.set_xlabel("x (Bohr)")
        if idx == 0:
            ax.set_ylabel(r"One-Body Density $\rho(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig7_article_densities_comparison",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
    )


def plot_article_kl_divergence(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Si existe una baseline CASSCF pre-entrenada (`casscf_ref is not None`), genera la
    evolución de la Divergencia KL resuelta por espín (`fig8_article_kl_divergence`,
    equivalente a `Article_KL_Divergence.pdf` de `scripts/evaluar_wigner_casscf_2.py`),
    promediando entre todas las corridas MCMC independientes y sombreando +/- 1 sigma.
    Si NO existe baseline CASSCF, no hace nada.
    """
    if casscf_ref is None:
        return

    valid = [
        b
        for b in bundles_info
        if "walkers" in b["bundle"].get("production", {})
        and np.asarray(b["bundle"]["production"]["walkers"]).size > 0
    ]
    if not valid:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), sharey=True)
    kl_records = []

    for idx, binfo in enumerate(valid):
        color = PALETTE[idx % len(PALETTE)]
        label = binfo["label"]
        w_all = np.asarray(binfo["bundle"]["production"]["walkers"], dtype=np.float64)
        if w_all.ndim == 4:
            w_all = w_all[None, ...]  # (1, n_steps, batch_size, n_elec, 1)

        n_chains = w_all.shape[0]
        x_max = float(
            binfo.get("observables", {}).get("x_max", cfg.density_x_max)
            if binfo.get("observables")
            else cfg.density_x_max
        )

        tot_curves, up_curves, dn_curves = [], [], []
        steps_grid = None
        for c_idx in range(n_chains):
            curve = compute_kl_convergence_curve(
                w_all[c_idx],
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

        for ax, m_arr, s_arr in zip(
            axes, (mean_tot, mean_up, mean_dn), (std_tot, std_up, std_dn)
        ):
            ax.plot(steps_grid, m_arr, color=color, lw=2.4, label=label)
            if n_chains > 1:
                ax.fill_between(
                    steps_grid,
                    np.maximum(1e-8, m_arr - s_arr),
                    m_arr + s_arr,
                    color=color,
                    alpha=0.20,
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
        ax.legend(fontsize=9.5, frameon=True)

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
    )

    if kl_records and cfg.save_observables_csv:
        df_kl = pd.DataFrame(kl_records)
        kl_csv_std = os.path.join(out_dir, "kl_convergence_vs_casscf.csv")
        df_kl.to_csv(kl_csv_std, index=False)
        if sys_tag:
            kl_csv_sys = os.path.join(out_dir, f"kl_convergence_vs_casscf_{sys_tag}.csv")
            df_kl.to_csv(kl_csv_sys, index=False)
            if getattr(cfg, "csv_collect_dir", ""):
                import shutil
                os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                shutil.copy2(kl_csv_sys, os.path.join(cfg.csv_collect_dir, os.path.basename(kl_csv_sys)))


def plot_efficiency_bars(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
) -> None:
    valid = [b for b in bundles_info if b["diagnostics"].get("summary")]
    if not valid:
        return

    labels = [b["label"] for b in valid]
    taus = [b["diagnostics"]["summary"].get("tau_int_E", 1.0) for b in valid]
    ess_s = [b["diagnostics"]["summary"].get("ESS_per_sec", 0.0) for b in valid]
    esjds = [b["diagnostics"]["summary"].get("ESJD", 0.0) for b in valid]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(valid))]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].bar(labels, taus, color=colors, alpha=0.85)
    axes[0].set_title("Tiempo de Autocorrelación $\\tau_{\\mathrm{int}, E}$ (menor es mejor)")
    axes[0].set_ylabel("$\\tau_{\\mathrm{int}, E}$ (pasos)")

    axes[1].bar(labels, ess_s, color=colors, alpha=0.85)
    axes[1].set_title("Muestras Efectivas por Segundo (ESS/s)")
    axes[1].set_ylabel("ESS / s")

    axes[2].bar(labels, esjds, color=colors, alpha=0.85)
    axes[2].set_title("Expected Squared Jump Distance (ESJD)")
    axes[2].set_ylabel("ESJD ($a_0^2$)")

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    _save_fig(
        fig,
        out_dir,
        "fig4_efficiency_bars",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
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

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for idx, binfo in enumerate(valid):
        summary = binfo["diagnostics"]["summary"]
        e_series = summary["e_step_series"]
        wtime = max(1e-6, summary.get("wall_time_sec", 1.0))
        tau_e = max(1.0, summary.get("tau_int_E", 1.0))
        t_grid = np.linspace(wtime / len(e_series), wtime, len(e_series))
        se_t = np.std(e_series) * np.sqrt(tau_e / np.arange(1, len(e_series) + 1))
        color = PALETTE[idx % len(PALETTE)]
        ax.plot(
            t_grid,
            se_t,
            color=color,
            lw=2.0,
            label=f"{binfo['label']} ($T_\\epsilon={summary['time_to_solution_sec']:.2f}$ s)",
        )

    ax.axhline(
        cfg.target_se_epsilon,
        color="black",
        linestyle="--",
        lw=1.2,
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
    )


def plot_walker_trace(
    bundles_info: list[dict],
    out_dir: str,
    cfg: PostProcessConfig,
    casscf_ref: dict[str, Any] | None = None,
) -> None:
    """
    Grafica la trayectoria de un caminante individual junto con su histograma marginal de
    densidad por espín (siguiendo el diseño `plot_walker_ergodicity` de `evaluar_wigner_casscf_2.py`),
    superponiendo la densidad analítica CASSCF si existe para el sistema.
    """
    valid = [
        b
        for b in bundles_info
        if "walkers" in b["bundle"].get("production", {})
        and np.asarray(b["bundle"]["production"]["walkers"]).size > 0
    ]
    if not valid:
        return

    n_models = len(valid)
    fig = plt.figure(figsize=(13.5, 4.6 * n_models))
    outer_gs = gridspec.GridSpec(n_models, 1, hspace=0.32)

    for idx, binfo in enumerate(valid):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs[idx], width_ratios=[4, 1], wspace=0.05
        )
        ax_main = fig.add_subplot(inner_gs[0])
        ax_hist = fig.add_subplot(inner_gs[1], sharey=ax_main)

        sys_cfg = binfo["bundle"].get("config", {}).get("system", {})
        n_up = int(sys_cfg.get("n_up", 2))
        n_dn = int(sys_cfg.get("n_dn", sys_cfg.get("n_down", 2)))
        w_arr = np.asarray(binfo["bundle"]["production"]["walkers"])
        if w_arr.ndim == 4:
            w_arr = w_arr[None, ...]
        # Tomar el caminante 0 de la cadena 0 para la traza temporal
        w_single = w_arr[0, :, 0, :, 0]  # (n_steps, n_elec)
        n_steps, n_elec = w_single.shape
        stride = max(1, n_steps // 2000)
        steps_sub = np.arange(0, n_steps, stride)

        for e_i in range(n_elec):
            c_spin = "#3498db" if e_i < n_up else "#e74c3c"
            lbl = (
                r"Spin $\uparrow$"
                if e_i == 0
                else (r"Spin $\downarrow$" if e_i == n_up else None)
            )
            ax_main.plot(
                steps_sub,
                w_single[::stride, e_i],
                color=c_spin,
                alpha=0.7,
                lw=0.8,
                label=lbl,
            )

        ax_main.set_title(
            f"Walker Trajectory & Nodal Tunneling - {binfo['label']}", weight="bold"
        )
        ax_main.set_xlabel("MCMC Production Steps")
        ax_main.set_ylabel("Electron Position (Bohr)")
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc="upper right", fontsize=8.5)

        # Panel derecho: histograma marginal usando todos los caminantes de la corrida
        x_max = float(
            binfo.get("observables", {}).get("x_max", 15.0)
            if binfo.get("observables")
            else 15.0
        )
        bins = np.linspace(-x_max, x_max, 120)
        try:
            w_flat = w_arr.squeeze(-1).reshape(-1, n_elec)
            if n_up > 0:
                up_vals = w_flat[:, :n_up].reshape(-1)
                up_vals = up_vals[np.isfinite(up_vals)]
                up_vals = up_vals[(up_vals >= -x_max) & (up_vals <= x_max)]
                if len(up_vals) > 0:
                    ax_hist.hist(
                        up_vals,
                        bins=bins,
                        orientation="horizontal",
                        density=True,
                        color="#3498db",
                        alpha=0.45,
                        label=r"Sampled $\uparrow$",
                    )
            if n_dn > 0:
                dn_vals = w_flat[:, n_up : n_up + n_dn].reshape(-1)
                dn_vals = dn_vals[np.isfinite(dn_vals)]
                dn_vals = dn_vals[(dn_vals >= -x_max) & (dn_vals <= x_max)]
                if len(dn_vals) > 0:
                    ax_hist.hist(
                        dn_vals,
                        bins=bins,
                        orientation="horizontal",
                        density=True,
                        color="#e74c3c",
                        alpha=0.45,
                        label=r"Sampled $\downarrow$",
                    )
        except Exception:
            pass

        if casscf_ref is not None:
            x_ex = casscf_ref["x"]
            mask_ex = (x_ex >= -x_max) & (x_ex <= x_max)
            if n_up > 0:
                ax_hist.plot(
                    casscf_ref["p_up"][mask_ex],
                    x_ex[mask_ex],
                    color="#2980b9",
                    linestyle="--",
                    lw=2.0,
                    label=r"Exact $\uparrow$",
                )
            if n_dn > 0:
                ax_hist.plot(
                    casscf_ref["p_down"][mask_ex],
                    x_ex[mask_ex],
                    color="#c0392b",
                    linestyle="--",
                    lw=2.0,
                    label=r"Exact $\downarrow$",
                )

        ax_hist.set_title("Marginal Density", weight="bold")
        ax_hist.set_xlabel(r"Normalized $p_\sigma(x)$")
        ax_hist.tick_params(axis="y", labelleft=False)
        ax_hist.grid(True, alpha=0.3)
        ax_hist.legend(loc="upper right", fontsize=8.0)

    _save_fig(
        fig,
        out_dir,
        "fig6_walker_traces",
        fmt=cfg.figure_format,
        dpi=cfg.dpi,
        sys_tag=getattr(cfg, "system_tag", ""),
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
                obs_csv_std = os.path.join(cfg.out_dir, f"observables_{safe_lbl}.csv")
                df_obs.to_csv(obs_csv_std, index=False)
                if sys_tag and not safe_lbl.endswith(sys_tag):
                    obs_csv_sys = os.path.join(cfg.out_dir, f"observables_{safe_lbl}_{sys_tag}.csv")
                    df_obs.to_csv(obs_csv_sys, index=False)
                else:
                    obs_csv_sys = obs_csv_std
                if getattr(cfg, "csv_collect_dir", ""):
                    import shutil
                    os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                    shutil.copy2(obs_csv_sys, os.path.join(cfg.csv_collect_dir, os.path.basename(obs_csv_sys)))

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
                    cas_csv_std = os.path.join(cfg.out_dir, "casscf_exact_density.csv")
                    df_cas_ex.to_csv(cas_csv_std, index=False)
                    if sys_tag:
                        cas_csv_sys = os.path.join(cfg.out_dir, f"casscf_exact_density_{sys_tag}.csv")
                        df_cas_ex.to_csv(cas_csv_sys, index=False)
                        if getattr(cfg, "csv_collect_dir", ""):
                            import shutil
                            os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                            shutil.copy2(cas_csv_sys, os.path.join(cfg.csv_collect_dir, os.path.basename(cas_csv_sys)))
            else:
                print(
                    f"[PostProcess] Sin baseline CASSCF pre-entrenada para "
                    f"(N_up={n_up}, N_dn={n_dn}, omega={omega}, g={sys_cfg.get('interaction_strength', 1.0)}). "
                    f"Graficando resultados sin baseline CASSCF."
                )

        # Calcular estadística de D_KL contra CASSCF a través de las corridas de evaluación
        kl_cas_tot_mean = float("nan")
        kl_cas_tot_std = float("nan")
        kl_cas_up_mean = float("nan")
        kl_cas_dn_mean = float("nan")
        if casscf_ref is not None and w_all is not None and w_all.size > 0:
            w_5d = w_all[None, ...] if w_all.ndim == 4 else w_all
            kls_tot, kls_up, kls_dn = [], [], []
            eff_xmax = float(obs["x_max"]) if obs is not None else float(cfg.density_x_max)
            for c_i in range(w_5d.shape[0]):
                kl_dict = compute_spin_kl_divergences(
                    w_5d[c_i], casscf_ref, n_bins=cfg.n_density_bins, x_max=eff_xmax
                )
                kls_tot.append(kl_dict["kl_total"])
                kls_up.append(kl_dict["kl_up"])
                kls_dn.append(kl_dict["kl_down"])
            kl_cas_tot_mean = float(np.mean(kls_tot))
            kl_cas_tot_std = float(np.std(kls_tot))
            kl_cas_up_mean = float(np.mean(kls_up))
            kl_cas_dn_mean = float(np.mean(kls_dn))

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
            "accepted_disp": summary.get("accepted_disp", float("nan")),
            "tau_int_E": tau_e,
            "tau_int_pos": summary.get("tau_int_pos", float("nan")),
            "G_RL_speedup": g_rl,
            "ESS": summary.get("ESS", float("nan")),
            "ESS_per_sec": summary.get("ESS_per_sec", float("nan")),
            "wall_time_sec": summary.get("wall_time_sec", float("nan")),
            "time_to_solution_sec": summary.get("time_to_solution_sec", float("nan")),
            "n_wigner_peaks": obs["n_wigner_peaks"] if obs else -1,
            "rms_radius": obs["rms_radius"] if obs else float("nan"),
            "mean_pair_dist": obs["mean_pair_dist"] if obs else float("nan"),
            "has_casscf_baseline": bool(casscf_ref is not None),
            "casscf_energy": cas_e_val,
            "delta_E_vs_CASSCF": delta_e_cas,
            "kl_casscf_total_mean": kl_cas_tot_mean,
            "kl_casscf_total_std": kl_cas_tot_std,
            "kl_casscf_up_mean": kl_cas_up_mean,
            "kl_casscf_down_mean": kl_cas_dn_mean,
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
        csv_path = os.path.join(cfg.out_dir, "summary_metrics.csv")
        json_path = os.path.join(cfg.out_dir, "summary_metrics.json")
        std_summary_json = os.path.join(cfg.out_dir, "summary.json")
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
            json.dump(summary_rows, f, indent=2)
        with open(std_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_dict_payload, f, indent=2)
        if sys_tag:
            csv_sys_path = os.path.join(cfg.out_dir, f"summary_metrics_{sys_tag}.csv")
            json_sys_path = os.path.join(cfg.out_dir, f"summary_{sys_tag}.json")
            df.to_csv(csv_sys_path, index=False)
            with open(json_sys_path, "w", encoding="utf-8") as f:
                json.dump(summary_dict_payload, f, indent=2)
            if getattr(cfg, "csv_collect_dir", ""):
                import shutil
                os.makedirs(cfg.csv_collect_dir, exist_ok=True)
                shutil.copy2(csv_sys_path, os.path.join(cfg.csv_collect_dir, os.path.basename(csv_sys_path)))
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

    if cfg.plot_training_metrics:
        _safe_run_plot(plot_training_dashboard, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref)
    if cfg.plot_production_summary:
        _safe_run_plot(plot_production_summary, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref)
    if cfg.plot_observables_comparison:
        _safe_run_plot(plot_observables_comparison, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref)
        _safe_run_plot(
            plot_article_densities_comparison, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )
        _safe_run_plot(
            plot_article_kl_divergence, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref
        )
    if cfg.plot_efficiency_bars:
        _safe_run_plot(plot_efficiency_bars, bundles_info, cfg.out_dir, cfg)
    if cfg.plot_time_to_solution:
        _safe_run_plot(plot_time_to_solution, bundles_info, cfg.out_dir, cfg)
    if cfg.plot_walker_trace:
        _safe_run_plot(plot_walker_trace, bundles_info, cfg.out_dir, cfg, casscf_ref=casscf_ref)

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
        cols.extend(["casscf_energy", "delta_E_vs_CASSCF", "kl_casscf_total_mean"])
    print(df[cols].to_string(index=False))
    print("=" * 76)
    return df
