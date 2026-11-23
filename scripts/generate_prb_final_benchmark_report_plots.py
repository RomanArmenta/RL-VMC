#!/usr/bin/env python3
"""
Generate publication-quality diagnostic figures for the exhaustive analysis report
on `results_prb_final_benchmark` (PRB Final Benchmark — Confinement-Regularized RL-MALA).

Outputs (both .pdf and 600 DPI .png) to `figures_prb_final_report/`:
  1. report_fig1_scaling_matrix        : Flips/1k, ESJD, and D_KL^spin across N in {4,6,10}, g in {0.1..5.0}
  2. report_fig2_training_dynamics     : Training acceptance, reward, ESJD curves + anatomy of late-stage collapse
  3. report_fig3_spatial_expressivity  : Spatial probe overlays mu_theta(x) & sigma_theta(x): Case 1 vs Case 2 vs Legacy
  4. report_fig4_rl_vs_rwmh_verdict    : Direct 1:1 RL/RWMH ratio heatmap + drift-sign scorecard
  5. report_fig5_n10_diagnostics       : N=10 auto-CASSCF quadrature offset & MALA frozen-walker anomaly
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_ROOT / "results_prb_final_benchmark"
OUT_DIR = PROJECT_ROOT / "figures_prb_final_report"

PALETTE = {
    "exact": "#252932",
    "grid": "#E1E5EB",
    "spine": "#3D424D",
    "rl": "#4B86BD",          # RL Case 2 (Confined) — Pastel ocean-steel blue
    "rl_dark": "#255B8E",
    "rl_alt": "#68C490",      # RL Case 1 (Unconstrained eq+unclip) — Pastel mint-emerald
    "rl_alt_dark": "#287D52",
    "mala": "#DF737B",        # MALA base — Pastel rose-red
    "mala_dark": "#A83E48",
    "mala_alt": "#8E6CB8",    # MALA tau=0.74 — Pastel plum-purple
    "mala_alt_dark": "#5A3D82",
    "rwmh": "#DEB273",        # RWMH 2.71 — Pastel warm sand-amber
    "rwmh_dark": "#996B2B",
    "rwmh_soft": "#B79C78",   # RWMH 1.22
    "dmu": "#E69635",
    "dmu_dark": "#A85C08",
    "jump_band": "#DDEFF8",
    "gray_fill": "#EEF1F5",
}


def setup_helvetica_style() -> None:
    ttc_path = Path("/System/Library/Fonts/Helvetica.ttc")
    if ttc_path.is_file():
        try:
            from fontTools.ttLib import TTCollection
            cache_dir = Path("/tmp/mpl_helvetica_fonts")
            cache_dir.mkdir(parents=True, exist_ok=True)
            ttc = TTCollection(str(ttc_path))
            for idx, name in [(0, "Helvetica_Regular.ttf"), (1, "Helvetica_Bold.ttf"),
                              (2, "Helvetica_Oblique.ttf"), (3, "Helvetica_BoldOblique.ttf")]:
                if idx < len(ttc.fonts):
                    out_ttf = cache_dir / name
                    if not out_ttf.is_file():
                        ttc.fonts[idx].save(str(out_ttf))
                    fm.fontManager.addfont(str(out_ttf))
        except Exception:
            pass

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.edgecolor": PALETTE["spine"],
        "axes.linewidth": 0.85,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 7.2,
        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D0D5DE",
        "legend.framealpha": 0.96,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save_fig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[+] Saved {OUT_DIR / name}.pdf/.png")


# ------------------------------------------------------------------------------
# FIGURE 1: FULL SCALING MATRIX (N=4, 6, 10 x g=0.1..5.0)
# ------------------------------------------------------------------------------
def make_report_fig1(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(8.8, 6.9), sharex=True)
    plt.subplots_adjust(wspace=0.28, hspace=0.22, top=0.87)

    ns = [4, 6, 10]
    models = [
        ("B2_RL_MALA_confined_amp0.80_sig1.8", "RL-MALA Confined (Case 2)", PALETTE["rl"], "s", "-", 1.8),
        ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "RL-MALA Unconstrained (Case 1)", PALETTE["rl_alt_dark"], "^", "--", 1.5),
        ("B1_RWMH_matched_sig2.7055", r"RWMH ($\sigma=2.71$)", PALETTE["rwmh_dark"], "D", "-.", 1.4),
        ("B1_RWMH_matched_sig1.2156", r"RWMH ($\sigma=1.22$)", PALETTE["rwmh"], "v", ":", 1.3),
        ("B1_MALA_base_tau0.10", r"MALA base ($\tau=0.10$)", PALETTE["mala"], "o", "-", 1.4),
        ("B1_MALA_matched_tau0.7389", r"MALA ($\tau=0.74$)", PALETTE["mala_alt"], "P", "--", 1.25),
    ]

    for col_i, n_val in enumerate(ns):
        sub_n = df[df["N"] == n_val]
        ax_fl, ax_es, ax_kl = axes[0, col_i], axes[1, col_i], axes[2, col_i]
        ax_fl.set_title(rf"$N = {n_val}$ electrons ($\omega=0.1$)", fontweight="bold", pad=6.0)

        for mk, lbl, col, mkr, ls, lw in models:
            s = sub_n[sub_n["model_key"] == mk].sort_values("g")
            # Filter out collapsed runs (acc < 0.001) so log scales remain truthful, mark collapses with X
            ok = s[s["acceptance"] >= 0.001]
            bad = s[s["acceptance"] < 0.001]
            if len(ok):
                g_ok = ok["g"].to_numpy()
                ax_fl.plot(g_ok, np.maximum(ok["hyst_flips_total_1k"].to_numpy(), 1e-3),
                           color=col, marker=mkr, ls=ls, lw=lw, ms=4.2, label=lbl, zorder=5 if "B2_" in mk else 3)
                ax_es.plot(g_ok, ok["ESJD"].to_numpy(),
                           color=col, marker=mkr, ls=ls, lw=lw, ms=4.2, zorder=5 if "B2_" in mk else 3)
                ax_kl.plot(g_ok, np.maximum(ok["kl_spin"].to_numpy(), 1e-5),
                           color=col, marker=mkr, ls=ls, lw=lw, ms=4.2, zorder=5 if "B2_" in mk else 3)
            if len(bad):
                # Mark collapsed points along bottom edge with red/colored X
                g_bad = bad["g"].to_numpy()
                ax_fl.scatter(g_bad, np.full_like(g_bad, 2e-3), marker="x", s=38, color=col, lw=1.4, zorder=8)
                ax_es.scatter(g_bad, np.full_like(g_bad, 0.12), marker="x", s=38, color=col, lw=1.4, zorder=8)
                ax_kl.scatter(g_bad, np.minimum(bad["kl_spin"].to_numpy(), 3.2), marker="x", s=38, color=col, lw=1.4, zorder=8)

        ax_fl.set_yscale("log")
        ax_fl.set_ylim(1e-3, 800)
        ax_es.set_ylim(0.0, 8.2)
        ax_kl.set_yscale("log")
        ax_kl.set_ylim(1e-5, 5.0)
        ax_kl.axhline(1e-3, color="#8C919B", ls=":", lw=0.75)

        for ax in (ax_fl, ax_es, ax_kl):
            ax.set_xscale("log")
            ax.set_xticks([0.1, 0.5, 1.0, 3.0, 5.0])
            ax.set_xticklabels(["0.1", "0.5", "1.0", "3.0", "5.0"])
            ax.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
            ax.tick_params(top=True, right=True)
        ax_kl.set_xlabel(r"Interaction strength $g$")

    axes[0, 0].set_ylabel(r"Flips / $10^3$ steps")
    axes[1, 0].set_ylabel(r"ESJD ($a_0^2$/step)")
    axes[2, 0].set_ylabel(r"Spin error $D_{\mathrm{KL}}^{\mathrm{spin}}$")

    # External legend above (RL first, RWMH second, MALA third + collapse X marker)
    handles = [
        Line2D([0], [0], color=PALETTE["rl"], marker="s", ls="-", lw=1.8, ms=4.5, label="RL-MALA Confined (Case 2)"),
        Line2D([0], [0], color=PALETTE["rl_alt_dark"], marker="^", ls="--", lw=1.5, ms=4.5, label="RL-MALA Unconstrained (Case 1)"),
        Line2D([0], [0], color=PALETTE["rwmh_dark"], marker="D", ls="-.", lw=1.4, ms=4.3, label=r"RWMH ($\sigma=2.71$)"),
        Line2D([0], [0], color=PALETTE["rwmh"], marker="v", ls=":", lw=1.3, ms=4.3, label=r"RWMH ($\sigma=1.22$)"),
        Line2D([0], [0], color=PALETTE["mala"], marker="o", ls="-", lw=1.4, ms=4.3, label=r"MALA base ($\tau=0.10$)"),
        Line2D([0], [0], color=PALETTE["mala_alt"], marker="P", ls="--", lw=1.25, ms=4.3, label=r"MALA ($\tau=0.74$)"),
        Line2D([0], [0], color="#444444", marker="x", ls="none", ms=5.5, mew=1.4, label=r"Variance collapse ($\alpha=0\%$)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.90), ncol=4, fontsize=6.9, columnspacing=1.0)
    save_fig(fig, "report_fig1_scaling_matrix")


# ------------------------------------------------------------------------------
# FIGURE 2: TRAINING DYNAMICS & ANATOMY OF LATE-STAGE COLLAPSE
# ------------------------------------------------------------------------------
def make_report_fig2() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    plt.subplots_adjust(wspace=0.28, hspace=0.32, top=0.86)

    def load_th(sys_tag: str, case_dir: str) -> pd.DataFrame:
        p = next((BENCH_DIR / sys_tag / case_dir).glob("training_history_*.csv"))
        return pd.read_csv(p)

    # (a) Healthy convergence: N=4 g=1.0 (Confined rescues sign -> higher ESJD & Acc)
    t_n4_c1 = load_th("N4_w0.1_g1.0", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")
    t_n4_c2 = load_th("N4_w0.1_g1.0", "B2_RL_MALA_confined_amp0.80_sig1.8")
    ax = axes[0, 0]
    ax.plot(t_n4_c2["Step"], t_n4_c2["ESJD"].rolling(7, min_periods=1).mean(), color=PALETTE["rl"], lw=1.7, label="RL-MALA Confined (Case 2)")
    ax.plot(t_n4_c1["Step"], t_n4_c1["ESJD"].rolling(7, min_periods=1).mean(), color=PALETTE["rl_alt_dark"], ls="--", lw=1.45, label="RL-MALA Unconstrained (Case 1)")
    ax.axhline(3.76, color=PALETTE["rwmh_dark"], ls=":", lw=1.2, label=r"RWMH ($\sigma=2.71$) eval")
    ax.set_title(r"$N=4, g=1.0$: Confinement rescues ESJD", fontweight="bold")
    ax.set_ylabel(r"Training ESJD ($a_0^2$/step)")
    ax.set_ylim(0, 5.2)

    # (b) Collapse comparison on N=6, g=0.5 (Case 1 collapses @5200, Case 2 stays healthy!)
    t_n6g05_c1 = load_th("N6_w0.1_g0.5", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")
    t_n6g05_c2 = load_th("N6_w0.1_g0.5", "B2_RL_MALA_confined_amp0.80_sig1.8")
    ax = axes[0, 1]
    ax.plot(t_n6g05_c2["Step"], t_n6g05_c2["Acceptance"].rolling(5, min_periods=1).mean()*100, color=PALETTE["rl"], lw=1.7)
    ax.plot(t_n6g05_c1["Step"], t_n6g05_c1["Acceptance"].rolling(5, min_periods=1).mean()*100, color=PALETTE["rl_alt_dark"], ls="--", lw=1.45)
    ax.annotate("Case 1 collapse\n(step ~5200)", xy=(5250, 1.5), xytext=(3300, 26),
                fontsize=6.8, fontweight="bold", color=PALETTE["rl_alt_dark"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["rl_alt_dark"], lw=0.9))
    ax.set_title(r"$N=6, g=0.5$: Case 2 stable vs Case 1 collapse", fontweight="bold")
    ax.set_ylabel(r"Acceptance rate $\bar{\alpha}$ (%)")
    ax.set_ylim(-1, 43)

    # (c) Anatomy of step-5975 collapse on N=6, g=1.0 Case 2!
    t_n6g1_c1 = load_th("N6_w0.1_g1.0", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")
    t_n6g1_c2 = load_th("N6_w0.1_g1.0", "B2_RL_MALA_confined_amp0.80_sig1.8")
    ax = axes[1, 0]
    ax.plot(t_n6g1_c2["Step"], t_n6g1_c2["Acceptance"]*100, color=PALETTE["rl"], lw=1.5, alpha=0.92)
    ax.plot(t_n6g1_c1["Step"], t_n6g1_c1["Acceptance"]*100, color=PALETTE["rl_alt_dark"], ls="--", lw=1.3, alpha=0.88)
    ax.annotate("Last-step collapse\n(step 5975 -> 6000!)", xy=(5990, 0.5), xytext=(2850, 26),
                fontsize=6.8, fontweight="bold", color=PALETTE["rl_dark"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.95))
    ax.set_title(r"$N=6, g=1.0$: Healthy for 5900 steps, last-step drop", fontweight="bold")
    ax.set_xlabel("TD3 training step")
    ax.set_ylabel(r"Acceptance rate $\bar{\alpha}$ (%)")
    ax.set_ylim(-1, 45)

    # (d) Negative reward trap on N=10, g=3.0 & N=6, g=5.0 (Confinement penalty dominates when alpha -> 0)
    t_n10g3_c2 = load_th("N10_w0.1_g3.0", "B2_RL_MALA_confined_amp0.80_sig1.8")
    t_n6g5_c2 = load_th("N6_w0.1_g5.0", "B2_RL_MALA_confined_amp0.80_sig1.8")
    t_n6g5_c1 = load_th("N6_w0.1_g5.0", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")
    ax = axes[1, 1]
    ax.axhline(0.0, color=PALETTE["exact"], ls=":", lw=0.85)
    ax.plot(t_n6g5_c1["Step"], t_n6g5_c1["RL_Reward"].rolling(7, min_periods=1).mean(), color=PALETTE["rl_alt_dark"], ls="--", lw=1.4, label=r"$N=6, g=5.0$ Case 1")
    ax.plot(t_n6g5_c2["Step"], t_n6g5_c2["RL_Reward"].rolling(7, min_periods=1).mean(), color=PALETTE["rl"], lw=1.6, label=r"$N=6, g=5.0$ Case 2")
    ax.plot(t_n10g3_c2["Step"], t_n10g3_c2["RL_Reward"].rolling(7, min_periods=1).mean(), color=PALETTE["mala"], lw=1.6, label=r"$N=10, g=3.0$ Case 2")
    ax.annotate(r"$R<0 \Rightarrow \sigma \to \sigma_{\min}$", xy=(800, -0.15), xytext=(2200, -0.22),
                fontsize=6.8, fontweight="bold", color=PALETTE["mala_dark"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.9))
    ax.set_title(r"High-$N$/$g$ negative-reward trap under $(1-\alpha)\Delta \bar{V}_{\mathrm{trap}}$", fontweight="bold")
    ax.set_xlabel("TD3 training step")
    ax.set_ylabel("Mean RL reward")
    ax.set_ylim(-0.28, 0.55)
    ax.legend(loc="upper right", fontsize=6.4, frameon=True)

    for ax in axes.flat:
        ax.set_xlim(0, 6050)
        ax.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
        ax.tick_params(top=True, right=True)

    handles = [
        Line2D([0], [0], color=PALETTE["rl"], lw=1.8, ls="-", label="RL-MALA Confined (Case 2)"),
        Line2D([0], [0], color=PALETTE["rl_alt_dark"], lw=1.5, ls="--", label="RL-MALA Unconstrained (Case 1)"),
        Line2D([0], [0], color=PALETTE["rwmh_dark"], lw=1.2, ls=":", label=r"RWMH ($\sigma=2.71$) eval benchmark"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.90), ncol=3, fontsize=7.2)
    save_fig(fig, "report_fig2_training_dynamics")


# ------------------------------------------------------------------------------
# FIGURE 3: SPATIAL EXPRESSIVITY (mu_theta & sigma_theta) CASE 2 vs CASE 1
# ------------------------------------------------------------------------------
def make_report_fig3() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6))
    plt.subplots_adjust(wspace=0.34, hspace=0.38, top=0.86)

    cases_to_plot = [
        ("N4_w0.1_g1.0", r"$N=4, g=1.0$: Drift-sign correction ($\mu_\theta$)", "mu"),
        ("N6_w0.1_g3.0", r"$N=6, g=3.0$: Drift-sign correction ($\mu_\theta$)", "mu"),
        ("N6_w0.1_g3.0", r"$N=6, g=3.0$: State-dependent width ($\sigma_\theta$)", "sigma"),
        ("N4_w0.1_g5.0", r"$N=4, g=5.0$: State-dependent width ($\sigma_\theta$)", "sigma"),
    ]

    for ax, (sys_tag, title, mode) in zip(axes.flat, cases_to_plot):
        sp_dir = BENCH_DIR / sys_tag / "figures_ablation" / "spatial_probes"
        d_c1 = pd.read_csv(sp_dir / "data_fig_actor_spatial_probe_B2_RL_MALA_eq_unclip_amp0.80_sig1.8_spin_down.csv")
        d_c2 = pd.read_csv(sp_dir / "data_fig_actor_spatial_probe_B2_RL_MALA_confined_amp0.80_sig1.8_spin_down.csv")
        x = d_c2["x_probe"].to_numpy()
        p_cond = d_c2["cond_prob_norm"].to_numpy()

        ax_bg = ax.twinx()
        ax_bg.fill_between(x, 0.0, p_cond, color=PALETTE["gray_fill"], alpha=0.85, zorder=0)
        ax_bg.set_ylim(0.0, 1.35)
        ax_bg.set_yticks([])
        for sp in ("top", "right", "bottom", "left"):
            ax_bg.spines[sp].set_visible(False)
        ax.set_zorder(ax_bg.get_zorder() + 1)
        ax.patch.set_visible(False)

        if mode == "mu":
            ax.axhline(0.0, color=PALETTE["exact"], ls=":", lw=0.75, alpha=0.7)
            ax.axvline(0.0, color=PALETTE["exact"], ls=":", lw=0.75, alpha=0.7)
            ax.plot(x, d_c2["mu_actor"].to_numpy(), color=PALETTE["rl"], lw=1.75, label="RL-MALA Confined (Case 2)", zorder=6)
            ax.plot(x, d_c1["mu_actor"].to_numpy(), color=PALETTE["mala"], ls="--", lw=1.45, label="RL-MALA Unconstrained (Case 1)", zorder=5)
            ax.set_ylabel(r"Residual drift $\mu_\theta(x)$ ($a_0$)")
            ax.set_ylim(-1.02, 1.16)
            ax.text(0.04, 0.96, r"Inward ($\mu_\theta x < 0$)", transform=ax.transAxes, va="top", fontsize=6.5, fontweight="bold", color=PALETTE["rl_dark"])
            ax.text(0.96, 0.96, r"Outward ($\mu_\theta x > 0$)", transform=ax.transAxes, ha="right", va="top", fontsize=6.5, fontweight="bold", color=PALETTE["mala_dark"])
        else:
            ax.plot(x, d_c2["sigma_applied"].to_numpy(), color=PALETTE["rl"], lw=1.75, zorder=6)
            ax.plot(x, d_c1["sigma_applied"].to_numpy(), color=PALETTE["mala"], ls="--", lw=1.45, zorder=5)
            ax.axhline(1.2156, color=PALETTE["exact"], ls=":", lw=0.95, alpha=0.8)
            ax.set_ylabel(r"Proposal width $\sigma_\theta(x)$ ($a_0$)")
            ax.set_ylim(1.0, 2.90)
            s2 = float(np.std(d_c2["sigma_applied"]))
            s1 = float(np.std(d_c1["sigma_applied"]))
            ax.text(0.04, 0.10, rf"$\mathrm{{std}}(\sigma)$: Case 2 = {s2:.2f} vs Case 1 = {s1:.2f} $a_0$",
                    transform=ax.transAxes, fontsize=6.4, fontweight="bold", color=PALETTE["rl_dark"],
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D0D5DE", lw=0.5, alpha=0.92))

        ax.set_title(title, fontweight="bold", pad=5.0)
        ax.set_xlim(x.min(), x.max())
        ax.set_xlabel(r"Probe electron position $x$ ($a_0$)")
        ax.tick_params(top=True, right=True)
        ax.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])

    handles = [
        Line2D([0], [0], color=PALETTE["rl"], lw=1.8, ls="-", label=r"RL-MALA Confined (Case 2, $r_{\mathrm{CDLB}}+r_{\mathrm{reg}}$)"),
        Line2D([0], [0], color=PALETTE["mala"], lw=1.5, ls="--", label=r"RL-MALA Unconstrained (Case 1, $r_{\mathrm{CDLB}}$ only)"),
        Line2D([0], [0], color=PALETTE["exact"], lw=1.0, ls=":", label=r"Legacy ceiling $\sigma=1.22\,a_0$ / zero line"),
        Patch(facecolor=PALETTE["gray_fill"], edgecolor="#C5CBD6", lw=0.5, label=r"Conditional $|\Psi(x)|^2$"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.90), ncol=2, fontsize=7.0)
    save_fig(fig, "report_fig3_spatial_expressivity")


# ------------------------------------------------------------------------------
# FIGURE 4: RL-MALA vs MATCHED-SIGMA RWMH (2.71) VERDICT + DRIFT SIGN MATRIX
# ------------------------------------------------------------------------------
def make_report_fig4(df: pd.DataFrame) -> None:
    ns = [4, 6, 10]
    gs_vals = [0.1, 0.5, 1.0, 3.0, 5.0]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.45))
    plt.subplots_adjust(wspace=0.32, top=0.84)

    for ax, (case_key, title) in zip(
        axes,
        [
            ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", r"Case 1: Unconstrained $r_{\mathrm{CDLB}}$ vs RWMH ($\sigma=2.71$)"),
            ("B2_RL_MALA_confined_amp0.80_sig1.8", r"Case 2: Confined $r_{\mathrm{CDLB}}+r_{\mathrm{reg}}$ vs RWMH ($\sigma=2.71$)"),
        ],
    ):
        mat = np.full((len(ns), len(gs_vals)), np.nan)
        labels = [["" for _ in gs_vals] for _ in ns]
        for i, n_val in enumerate(ns):
            for j, g_val in enumerate(gs_vals):
                s_rl = df[(df["N"] == n_val) & (df["g"] == g_val) & (df["model_key"] == case_key)]
                s_rw = df[(df["N"] == n_val) & (df["g"] == g_val) & (df["model_key"] == "B1_RWMH_matched_sig2.7055")]
                if len(s_rl) == 0:
                    labels[i][j] = "CUDA\nerr"
                    continue
                row_rl = s_rl.iloc[0]
                row_rw = s_rw.iloc[0]
                # Check spatial probe sign
                sp_dir = BENCH_DIR / f"N{n_val}_w0.1_g{g_val}" / "figures_ablation" / "spatial_probes"
                case_short = "confined" if "confined" in case_key else "eq_unclip"
                sp_csv = sp_dir / f"data_fig_actor_spatial_probe_B2_RL_MALA_{case_short}_amp0.80_sig1.8_spin_down.csv"
                sgn_tag = ""
                if sp_csv.is_file():
                    spd = pd.read_csv(sp_csv)
                    w = np.maximum(spd["cond_prob_norm"].to_numpy(), 0.0)
                    w /= np.maximum(w.sum(), 1e-12)
                    al = float(np.sum(w * spd["mu_actor"].to_numpy() * np.sign(spd["x_probe"].to_numpy())))
                    sgn_tag = "IN" if al < 0 else "OUT"
                if row_rl["acceptance"] < 0.001:
                    mat[i, j] = -60.0
                    labels[i][j] = f"COLLAPSE\n({sgn_tag})"
                else:
                    pct = (row_rl["ESJD"] / row_rw["ESJD"] - 1.0) * 100.0
                    mat[i, j] = pct
                    labels[i][j] = f"{pct:+.0f}%\n({sgn_tag})"

        cmap = plt.cm.RdBu.copy()
        cmap.set_bad("#EAECEF")
        im = ax.imshow(mat, cmap=cmap, vmin=-55, vmax=55, aspect="auto")
        ax.set_xticks(range(len(gs_vals)))
        ax.set_xticklabels([f"{g}" for g in gs_vals])
        ax.set_yticks(range(len(ns)))
        ax.set_yticklabels([f"$N={n}$" for n in ns])
        ax.set_xlabel(r"Interaction strength $g$")
        ax.set_title(title, fontweight="bold", pad=8.0, fontsize=8.1)

        for i in range(len(ns)):
            for j in range(len(gs_vals)):
                val = mat[i, j]
                txt_col = "white" if (not np.isnan(val) and abs(val) > 33) else "#1E222B"
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6.8, fontweight="bold", color=txt_col)

    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.03)
    cbar.set_label(r"$\Delta$ESJD vs matched RWMH (%)", fontsize=7.6)
    save_fig(fig, "report_fig4_rl_vs_rwmh_verdict")


# ------------------------------------------------------------------------------
# FIGURE 5: N=10 CASSCF QUADRATURE OFFSET & MALA FROZEN-WALKER DIAGNOSTIC
# ------------------------------------------------------------------------------
def make_report_fig5(df: pd.DataFrame) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.2, 3.35))
    plt.subplots_adjust(wspace=0.34, top=0.84)

    # (a) N=10 g=0.1 and g=0.5: Solver quadrature energy vs consistent MCMC real-space energies
    groups = [r"$N=10, g=0.1$" + "\n(auto-CASSCF\nquad=100, b=48)", r"$N=10, g=0.5$" + "\n(auto-CASSCF\nquad=100, b=48)"]
    e_solver = [4.52155, 10.65720]
    e_mala = [
        float(df[(df["N"] == 10) & (df["g"] == 0.1) & (df["model_key"] == "B1_MALA_base_tau0.10")]["E_mean"].iloc[0]),
        float(df[(df["N"] == 10) & (df["g"] == 0.5) & (df["model_key"] == "B1_MALA_base_tau0.10")]["E_mean"].iloc[0]),
    ]
    e_rwmh = [
        float(df[(df["N"] == 10) & (df["g"] == 0.1) & (df["model_key"] == "B1_RWMH_matched_sig2.7055")]["E_mean"].iloc[0]),
        float(df[(df["N"] == 10) & (df["g"] == 0.5) & (df["model_key"] == "B1_RWMH_matched_sig2.7055")]["E_mean"].iloc[0]),
    ]
    e_rl = [
        float(df[(df["N"] == 10) & (df["g"] == 0.1) & (df["model_key"] == "B2_RL_MALA_confined_amp0.80_sig1.8")]["E_mean"].iloc[0]),
        float(df[(df["N"] == 10) & (df["g"] == 0.5) & (df["model_key"] == "B2_RL_MALA_confined_amp0.80_sig1.8")]["E_mean"].iloc[0]),
    ]

    x_pos = np.arange(2)
    w = 0.19
    ax_a.bar(x_pos - 1.5*w, e_solver, width=w, color="#7D8594", edgecolor="#3D424D", lw=0.6, label="CASSCF solver (quad=100)")
    ax_a.bar(x_pos - 0.5*w, e_rl, width=w, color=PALETTE["rl"], edgecolor=PALETTE["rl_dark"], lw=0.6, label="RL-MALA Confined (MCMC)")
    ax_a.bar(x_pos + 0.5*w, e_rwmh, width=w, color=PALETTE["rwmh"], edgecolor=PALETTE["rwmh_dark"], lw=0.6, label="RWMH 2.71 (MCMC)")
    ax_a.bar(x_pos + 1.5*w, e_mala, width=w, color=PALETTE["mala"], edgecolor=PALETTE["mala_dark"], lw=0.6, label="MALA base (MCMC)")

    for i, diff in enumerate([e_solver[0] - e_rl[0], e_solver[1] - e_rl[1]]):
        ax_a.annotate(f"+{diff:.2f} Ha\nquad bias", xy=(x_pos[i] - 1.5*w, e_solver[i] + 0.15),
                      xytext=(x_pos[i] + 0.05, e_solver[i] + 1.55), ha="center",
                      fontsize=6.6, fontweight="bold", color="#3D424D",
                      arrowprops=dict(arrowstyle="->", color="#3D424D", lw=0.85))

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(groups)
    ax_a.set_xlim(-0.55, 1.55)
    ax_a.set_ylabel("Energy (Ha)")
    ax_a.set_ylim(0, 13.8)
    ax_a.set_title(r"Auto-generated $N=10$ CASSCF quadrature bias", fontweight="bold", fontsize=8.2, pad=34.0)
    ax_a.grid(True, axis="y", alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.50, 1.02), ncol=2, fontsize=6.0, frameon=True, columnspacing=0.8)

    # (b) N=10, g=5.0: Per-step chain energy traces comparing RWMH 2.71 vs MALA base (trapped walker!)
    with h5py.File(BENCH_DIR / "N10_w0.1_g5.0/B1_RWMH_matched_sig2.7055/run_bundle.h5", "r") as f:
        e_rw = np.asarray(f["production"]["local_energy"])[:, ::100, 0]
    with h5py.File(BENCH_DIR / "N10_w0.1_g5.0/B1_MALA_base_tau0.10/run_bundle.h5", "r") as f:
        e_ma = np.asarray(f["production"]["local_energy"])[:, ::100, 0]

    steps = np.arange(e_rw.shape[1]) * 100
    ax_b.axhline(31.7034, color=PALETTE["exact"], ls=":", lw=1.1, label=r"Exact CASSCF ($31.70$ Ha)")
    ax_b.plot(steps, e_rw.mean(axis=0), color=PALETTE["rwmh_dark"], lw=1.6, label=r"RWMH ($\sigma=2.71$, mean)")
    ax_b.plot(steps, e_ma[1], color=PALETTE["mala"], lw=1.25, ls="--", label="MALA base (Chain 1)")
    ax_b.plot(steps, e_ma[0], color=PALETTE["mala_alt"], lw=1.25, ls="-.", label="MALA base (Chain 0)")
    ax_b.set_ylim(-110, 58)
    ax_b.set_xlim(0, 40000)
    ax_b.set_xlabel("Production MCMC step")
    ax_b.set_ylabel(r"Step-mean local energy $\bar{E}_L$ (Ha)")
    ax_b.set_title(r"$N=10, g=5.0$: MALA trapped-walker instability", fontweight="bold", fontsize=8.2, pad=34.0)
    ax_b.text(0.04, 0.56, "MALA Chain 2 trapped at\n$-13{,}913$ Ha (off-scale)!",
              transform=ax_b.transAxes, fontsize=6.5, fontweight="bold", color=PALETTE["mala_dark"],
              bbox=dict(boxstyle="round,pad=0.20", fc="#FFF5F5", ec=PALETTE["mala"], lw=0.6))
    ax_b.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.50, 1.02), ncol=2, fontsize=6.0, frameon=True, columnspacing=0.8)

    save_fig(fig, "report_fig5_n10_diagnostics")


def main() -> None:
    setup_helvetica_style()
    df = pd.read_csv(BENCH_DIR / "prb_final_benchmark_master.csv")
    make_report_fig1(df)
    make_report_fig2()
    make_report_fig3()
    make_report_fig4(df)
    make_report_fig5(df)
    print("[DONE] All 5 benchmark report figures generated.")


if __name__ == "__main__":
    main()
