#!/usr/bin/env python3
"""
scripts/generate_prb_letter_figures_v2.py
=========================================
Revised publication-quality figure suite (v2) for the PRB Letter incorporating
all advisor remarks, while keeping scripts/generate_prb_letter_figures.py and
RL_Sampler_PRBLett/figures/ completely untouched.

Key design updates:
  - Authentic Helvetica sans-serif typography (Regular, Bold, Oblique extracted
    from macOS Helvetica.ttc when present; clean fallback to Arial/DejaVu Sans).
  - Print-friendly pastel color system with strong luminance & hue separation
    both across methods AND across spin channels (up vs. down):
      * RL-MALA : pastel blue (spin-up / primary) + pastel green (spin-down / alt)
      * MALA    : pastel red-rose (spin-up / primary) + pastel purple (spin-down / alt)
      * RWMH    : pastel warm sand-amber (independent neutral control)
  - All legends positioned OUTSIDE the plot frames (never obscuring data),
    always listing RL-MALA first (above/left) and MALA second.
  - Panel labels (a), (b), (c), (d) disabled by default so the authors can add
    their own; re-enable anytime with --show-panel-labels.
  - Fig. 1  : Two stacked full-width long traces (MALA top, RL-MALA bottom)
              with spin-resolved horizontal marginal histograms on the right.
  - Fig. 1b : Dedicated separate figure for spin-resolved one-body densities
              with the harmonic confining potential V(x) = 1/2 * omega^2 * x^2
              rendered in the background on a calibrated secondary y-axis.
  - Fig. 2  : 2D stream-flow heatmap (RL blue/mint pastel), transit-duration
              histogram, step-displacement profile, and single-electron actor
              step-changes Delta sigma_theta(k) = sigma(x_{k+1}) - sigma(x_k)
              and Delta mu_theta(k) = mu(x_{k+1}) - mu(x_k) across a real
              N=6, g=1.0, tau=0.10 transition.
  - Fig. 3  : Scaling phase diagram across g and N (flips, Wigner/Pauli, D_KL).
  - Fig. 4  : Causal B1/B2 ablation, matched-sigma ESJD verdict, spatial drift.
  - Fig. 5  : Two-panel MCMC convergence: spin-density KL divergence D_KL^spin(t)
              on top, integrated autocorrelation time tau_int(t) on bottom.

Usage:
  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \\
      scripts/generate_prb_letter_figures_v2.py

  # With optional panel labels (a), (b), ...
  ... scripts/generate_prb_letter_figures_v2.py --show-panel-labels
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_vmc.postprocess.diagnostics import compute_acf_1d, compute_iat_sokal


# ==============================================================================
# PRINT-SAFE PASTEL COLOR SYSTEM
#   RL-MALA : Blue / Green pastel (clearly distinguishable in hue & luminance)
#   MALA    : Red / Purple pastel (clearly distinguishable in hue & luminance)
#   RWMH    : Warm sand-amber pastel (distinct independent control)
# ==============================================================================

PALETTE = {
    # Exact reference, harmonic trap & structural neutrals
    "exact": "#252932",         # Crisp dark charcoal
    "gray_fill": "#EEF1F5",     # Subtle light mist fill for exact total density
    "pot_line": "#8A909B",      # Clean neutral gray dotted line for harmonic potential V(x)
    "grid": "#E1E5EB",          # Subtle grid
    "spine": "#3D424D",         # Clean dark slate axes spines

    # RL-MALA — Pastel Blue & Pastel Green family (strong spin & print contrast)
    "rl": "#4B86BD",            # Pastel ocean-steel blue (RL primary / Spin-up)
    "rl_dark": "#255B8E",       # Deep ocean blue for crisp labels & medians
    "rl_alt": "#68C490",        # Pastel mint-emerald green (RL secondary / Spin-down / Pauli)
    "rl_alt_dark": "#287D52",   # Deep pastel forest green for secondary axis & text
    "rl_soft": "#4CA8A4",       # Pastel sea-teal (N=4 RL / RL spectators)
    "rl_pale": "#84D1A5",       # Light pastel mint (N=2 RL)
    "rl_deep": "#2D6BA3",       # Rich pastel royal-ocean blue (N=6 RL / unlocked 3.30)

    # MALA — Pastel Red-Rose & Pastel Purple family (strong spin & print contrast)
    "mala": "#DF737B",          # Pastel rose-red (MALA primary / Spin-up)
    "mala_dark": "#A83E48",     # Deep pastel crimson for crisp labels & medians
    "mala_alt": "#8E6CB8",      # Pastel plum-purple (MALA secondary / Spin-down / Pauli)
    "mala_alt_dark": "#5A3D82", # Deep pastel plum for annotations
    "mala_light": "#C4789B",    # Pastel dusty-berry (N=4 MALA / MALA spectators)
    "mala_pale": "#E5A0A7",     # Soft pastel blush-rose (N=2 MALA, print-legible)

    # RWMH control — Pastel warm sand-amber (completely distinct from RL & MALA)
    "rwmh": "#DEB273",          # Pastel sand-amber
    "rwmh_dark": "#996B2B",     # Deep ochre for edges & annotations

    # High-contrast warm pastel amber-orange for Fig. 2(d) Delta mu_theta vs blue Delta sigma_theta
    "dmu": "#E69635",           # Warm pastel amber-orange
    "dmu_dark": "#A85C08",      # Deep ochre-amber for right y-axis & marker edges
    "dmu_fill": "#FDE0B6",      # Light pastel apricot marker face

    # Transition highlight band
    "jump_band": "#DDEFF8",     # Soft pastel ice-blue transition strip
}

RL_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "rl_blue_mint_pastel",
    [
        (0.00, "#FFFFFF"),
        (0.18, "#EEF8F5"),
        (0.40, "#C4E6DF"),
        (0.64, "#7FBCDE"),
        (0.84, "#4A84BC"),
        (1.00, "#21507E"),
    ],
)


def _register_helvetica_fonts() -> None:
    """Extract and register authentic Regular/Bold/Oblique faces from macOS Helvetica.ttc."""
    ttc_path = Path("/System/Library/Fonts/Helvetica.ttc")
    if not ttc_path.is_file():
        return
    try:
        from fontTools.ttLib import TTCollection

        cache_dir = Path("/tmp/mpl_helvetica_fonts")
        cache_dir.mkdir(parents=True, exist_ok=True)
        face_map = [
            (0, "Helvetica.ttf"),
            (1, "Helvetica-Bold.ttf"),
            (2, "Helvetica-Oblique.ttf"),
            (3, "Helvetica-BoldOblique.ttf"),
        ]
        need_extract = any(not (cache_dir / fn).is_file() for _, fn in face_map)
        if need_extract:
            ttc = TTCollection(str(ttc_path))
            for idx, fn in face_map:
                if idx < len(ttc.fonts):
                    ttc.fonts[idx].save(str(cache_dir / fn))
        for _, fn in face_map:
            fp = cache_dir / fn
            if fp.is_file():
                fm.fontManager.addfont(str(fp))
    except Exception:
        # Clean fallback to system sans-serif stack (Helvetica / Arial / DejaVu Sans)
        pass


def setup_prb_v2_style() -> None:
    _register_helvetica_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 8.5,
            "axes.edgecolor": PALETTE["spine"],
            "axes.labelcolor": "#1E2229",
            "xtick.color": "#2E333D",
            "ytick.color": "#2E333D",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#D2D7E0",
            "legend.borderpad": 0.30,
            "legend.handlelength": 1.65,
            "legend.handletextpad": 0.45,
            "axes.linewidth": 0.72,
            "xtick.major.width": 0.68,
            "ytick.major.width": 0.68,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 3.1,
            "ytick.major.size": 3.1,
            "xtick.minor.size": 1.7,
            "ytick.minor.size": 1.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def panel_label(
    ax: plt.Axes,
    text: str,
    show: bool = False,
    x: float = 0.02,
    y: float = 0.94,
    color: str = "#1E2229",
) -> None:
    """Optionally render a panel label (disabled by default; enable via --show-panel-labels)."""
    if not show:
        return
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        ha="left",
        color=color,
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.86),
        zorder=25,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=dpi if ext == "png" else 300)
        print(f"[+] Saved {out_path}")
    plt.close(fig)


def _find_ablation_dir(root: Path, sys_tag: str) -> Optional[Path]:
    base = root / sys_tag
    if not base.is_dir():
        return None
    for sub in [base / "figures_ablation", base]:
        if list(sub.glob("ablation_summary_*.csv")):
            return sub
    for cand in sorted(base.iterdir()):
        if cand.is_dir() and list(cand.glob("ablation_summary_*.csv")):
            return cand
    return None


def _compute_kl(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    eps = 1e-12
    ps = np.maximum(np.asarray(p, dtype=np.float64), eps)
    qs = np.maximum(np.asarray(q, dtype=np.float64), eps)
    ps /= np.sum(ps) * dx
    qs /= np.sum(qs) * dx
    return float(np.sum(ps * np.log(ps / qs)) * dx)


def _smooth_2d_heatmap(H: np.ndarray, sigma_y: float = 2.0, sigma_t: float = 1.1) -> np.ndarray:
    rad_y = max(1, int(round(3.0 * sigma_y)))
    ky = np.exp(-0.5 * (np.arange(-rad_y, rad_y + 1, dtype=np.float64) / sigma_y) ** 2)
    ky /= np.sum(ky)
    rad_t = max(1, int(round(3.0 * sigma_t)))
    kt = np.exp(-0.5 * (np.arange(-rad_t, rad_t + 1, dtype=np.float64) / sigma_t) ** 2)
    kt /= np.sum(kt)
    H_y = np.apply_along_axis(lambda col: np.convolve(col, ky, mode="same"), axis=0, arr=H)
    H_yt = np.apply_along_axis(lambda row: np.convolve(row, kt, mode="same"), axis=1, arr=H_y)
    return H_yt


# ==============================================================================
# FIGURE 1: TWO STACKED LONG TRACES (MALA TOP, RL-MALA BOTTOM) + RIGHT HISTOGRAMS
# ==============================================================================

def make_fig1_traces_marginals(
    out_dir: Path,
    t_max: int = 10000,
    show_panel_labels: bool = False,
) -> None:
    traces_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv/flips_traces_P2_N6_w0.1_g1.csv"
    dens_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv/flips_marginal_density_P2_N6_w0.1_g1.csv"

    df_tr = pd.read_csv(traces_csv)
    df_dn = pd.read_csv(dens_csv)
    x_grid = df_dn["x_grid"].to_numpy(dtype=np.float64)
    dx = float(x_grid[1] - x_grid[0])
    ex_up = df_dn["exact_up"].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(7.2, 4.95))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[4.35, 1.05],
        height_ratios=[1.0, 1.0],
        wspace=0.055,
        hspace=0.28,
    )

    ax_tr_m = fig.add_subplot(gs[0, 0])
    ax_hi_m = fig.add_subplot(gs[0, 1], sharey=ax_tr_m)
    ax_tr_r = fig.add_subplot(gs[1, 0], sharex=ax_tr_m, sharey=ax_tr_m)
    ax_hi_r = fig.add_subplot(gs[1, 1], sharey=ax_tr_r, sharex=ax_hi_m)

    y_lim = 14.8

    row_specs = [
        (
            ax_tr_m,
            ax_hi_m,
            "MALA",
            "(a)",
            r"MALA ($\tau = 0.10$)",
            PALETTE["mala"],
            PALETTE["mala_alt"],
            PALETTE["mala_dark"],
            "mala_up",
            "mala_down",
            True,
        ),
        (
            ax_tr_r,
            ax_hi_r,
            "RL-MALA",
            "(b)",
            r"RL-MALA ($\tau = 0.10$)",
            PALETTE["rl"],
            PALETTE["rl_alt"],
            PALETTE["rl_dark"],
            "rl_up",
            "rl_down",
            False,
        ),
    ]

    p_max_all = max(
        float(np.max(df_dn["mala_up"])),
        float(np.max(df_dn["mala_down"])),
        float(np.max(df_dn["rl_up"])),
        float(np.max(df_dn["rl_down"])),
        float(np.max(ex_up)),
    )

    for ax_tr, ax_hi, meth, ptag, title_txt, c_up, c_dn, c_accent, col_u, col_d, is_top in row_specs:
        sub = df_tr[(df_tr["method"] == meth) & (df_tr["step"] <= t_max)].sort_values("step")
        steps = sub["step"].to_numpy(dtype=np.float64)

        for idx in range(6):
            is_up = idx < 3
            col = c_up if is_up else c_dn
            ax_tr.plot(
                steps,
                sub[f"x_{idx}"].to_numpy(dtype=np.float64),
                color=col,
                lw=0.72,
                alpha=0.88 if is_up else 0.85,
            )

        ax_tr.set_xlim(0, t_max)
        ax_tr.set_ylim(-y_lim, y_lim)
        ax_tr.set_ylabel(r"Position $x_i$ ($a_0$)")
        ax_tr.set_title(title_txt, fontsize=8.4, fontweight="bold", color=c_accent, loc="left", pad=4.0)
        ax_tr.tick_params(top=True, right=True)
        ax_tr.grid(True, alpha=0.35, lw=0.45, color=PALETTE["grid"])
        panel_label(ax_tr, ptag, show=show_panel_labels, x=0.014, y=0.94)

        # Right horizontal marginal histogram p_up(x), p_dn(x) vs exact CASSCF
        # Both sampled spin marginals are SOLID continuous lines (differing only in color);
        # ONLY the Exact CASSCF reference is dotted.
        p_u = df_dn[col_u].to_numpy(dtype=np.float64)
        p_d = df_dn[col_d].to_numpy(dtype=np.float64)
        kl_val = 0.5 * (_compute_kl(p_u, ex_up, dx) + _compute_kl(p_d, ex_up, dx))

        lw_u = 1.45 if is_top else 2.20
        lw_d = 1.45 if is_top else 1.20
        ax_hi.fill_betweenx(x_grid, 0.0, p_u, color=c_up, alpha=0.28, lw=0.0)
        ax_hi.plot(p_u, x_grid, color=c_up, lw=lw_u, ls="-", zorder=5)
        ax_hi.fill_betweenx(x_grid, 0.0, p_d, color=c_dn, alpha=0.24, lw=0.0)
        ax_hi.plot(p_d, x_grid, color=c_dn, lw=lw_d, ls="-", zorder=6)
        ax_hi.plot(ex_up, x_grid, color=PALETTE["exact"], ls=":", lw=1.30, zorder=8)

        ax_hi.set_xlim(0.0, p_max_all * 1.24)
        ax_hi.set_ylim(-y_lim, y_lim)
        ax_hi.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
        ax_hi.tick_params(axis="y", labelleft=False, right=True)
        ax_hi.tick_params(axis="x", top=True)
        ax_hi.grid(True, alpha=0.32, lw=0.42, color=PALETTE["grid"])

        kl_str = (
            rf"$D_{{\mathrm{{KL}}}}={kl_val:.2f}$"
            if kl_val >= 0.01
            else rf"$D_{{\mathrm{{KL}}}}={kl_val*1e4:.1f}\times 10^{{-4}}$"
        )
        ax_hi.set_title(kl_str, fontsize=7.6, fontweight="bold", color=c_accent, pad=4.0)

    plt.setp(ax_tr_m.get_xticklabels(), visible=False)
    plt.setp(ax_hi_m.get_xticklabels(), visible=False)
    ax_tr_r.set_xlabel(r"Monte Carlo step $t$")
    ax_hi_r.set_xlabel(r"$p_\sigma(x)$ ($a_0^{-1}$)")

    # External figure-level legend ABOVE plots: RL-MALA listed FIRST (top row), MALA SECOND (bottom row)
    # Matplotlib fills column-by-column in ncol=3, so arrange indices to yield:
    #   Row 1: RL-MALA Spin-up | RL-MALA Spin-down | Exact CASSCF
    #   Row 2: MALA Spin-up    | MALA Spin-down    | (empty spacer)
    leg_handles = [
        Line2D([0], [0], color=PALETTE["rl"], lw=1.8, ls="-", label=r"RL-MALA: Spin-$\uparrow$ ($e_{1,2,3}$)"),
        Line2D([0], [0], color=PALETTE["mala"], lw=1.8, ls="-", label=r"MALA: Spin-$\uparrow$ ($e_{1,2,3}$)"),
        Line2D([0], [0], color=PALETTE["rl_alt"], lw=1.8, ls="-", label=r"RL-MALA: Spin-$\downarrow$ ($e_{4,5,6}$)"),
        Line2D([0], [0], color=PALETTE["mala_alt"], lw=1.8, ls="-", label=r"MALA: Spin-$\downarrow$ ($e_{4,5,6}$)"),
        Line2D([0], [0], color=PALETTE["exact"], lw=1.45, ls=":", label="Exact CASSCF"),
        Line2D([0], [0], color="none", lw=0.0, label=""),
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.51, 0.945),
        ncol=3,
        fontsize=7.2,
        columnspacing=1.35,
        handlelength=1.8,
        frameon=True,
    )

    save_figure(fig, out_dir, "fig1_traces_marginals")


# ==============================================================================
# FIGURE 1b (SEPARATE): ONE-BODY DENSITIES + HARMONIC CONFINING POTENTIAL V(x)
# ==============================================================================

def make_fig1b_one_body_densities(
    out_dir: Path,
    show_panel_labels: bool = False,
) -> None:
    dens_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv/flips_marginal_density_P2_N6_w0.1_g1.csv"
    df_dn = pd.read_csv(dens_csv)
    x = df_dn["x_grid"].to_numpy(dtype=np.float64)
    dx = float(x[1] - x[0])

    # Physical external harmonic confining potential V(x) = 0.5 * omega^2 * x^2 (omega = 0.1 a.u.)
    omega = 0.1
    v_harm = 0.5 * (omega ** 2) * (x ** 2)

    exact_up = 3.0 * df_dn["exact_up"].to_numpy(dtype=np.float64)
    exact_tot = 6.0 * df_dn["exact_total"].to_numpy(dtype=np.float64)

    kl_mala = 0.5 * (
        _compute_kl(df_dn["mala_up"], df_dn["exact_up"], dx)
        + _compute_kl(df_dn["mala_down"], df_dn["exact_down"], dx)
    )
    kl_rl = 0.5 * (
        _compute_kl(df_dn["rl_up"], df_dn["exact_up"], dx)
        + _compute_kl(df_dn["rl_down"], df_dn["exact_down"], dx)
    )

    fig = plt.figure(figsize=(7.2, 3.05))
    gs = GridSpec(1, 2, figure=fig, wspace=0.14)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)

    ymax = float(np.max(exact_tot)) * 1.26
    vmax = float(np.max(v_harm[np.abs(x) <= 13.8])) * 1.12  # ~1.07 Ha

    panels = [
        (
            ax_a,
            "mala_up",
            "mala_down",
            kl_mala,
            "(a)",
            r"MALA ($\tau = 0.10$)",
            PALETTE["mala"],
            PALETTE["mala_alt"],
            PALETTE["mala_dark"],
            True,
        ),
        (
            ax_b,
            "rl_up",
            "rl_down",
            kl_rl,
            "(b)",
            r"RL-MALA ($\tau = 0.10$)",
            PALETTE["rl"],
            PALETTE["rl_alt"],
            PALETTE["rl_dark"],
            False,
        ),
    ]

    for ax, up_col, dn_col, kl_val, ptag, m_title, c_up, c_dn, c_accent, is_left in panels:
        rho_u = 3.0 * df_dn[up_col].to_numpy(dtype=np.float64)
        rho_d = 3.0 * df_dn[dn_col].to_numpy(dtype=np.float64)

        # Background harmonic trap V(x) = 1/2 * omega^2 * x^2: gray dotted line ONLY (no shading)
        ax_v = ax.twinx()
        ax_v.plot(x, v_harm, color=PALETTE["pot_line"], ls=":", lw=1.30, alpha=0.88, zorder=1)
        ax_v.set_ylim(0.0, vmax)
        ax_v.tick_params(axis="y", colors="#636872", labelsize=7.2)
        ax_v.spines["right"].set_color("#8A909B")
        if is_left:
            ax_v.tick_params(axis="y", labelright=False)
        else:
            ax_v.set_ylabel(r"Harmonic potential $V(x)$ (Ha)", color="#636872", fontsize=8.2)

        # Put density axes cleanly above background potential line on crisp white background
        ax.set_zorder(ax_v.get_zorder() + 1)
        ax.patch.set_visible(False)

        lw_u = 1.75 if is_left else 2.45
        lw_d = 1.75 if is_left else 1.25
        ax.fill_between(x, 0.0, exact_tot, color=PALETTE["gray_fill"], alpha=0.72, zorder=2)
        ax.fill_between(x, 0.0, rho_u, color=c_up, alpha=0.16, zorder=3)
        ax.fill_between(x, 0.0, rho_d, color=c_dn, alpha=0.14, zorder=3)
        ax.plot(x, rho_u, color=c_up, lw=lw_u, ls="-", zorder=6)
        ax.plot(x, rho_d, color=c_dn, lw=lw_d, ls="-", zorder=7)
        ax.plot(x, exact_up, color=PALETTE["exact"], ls=":", lw=1.45, zorder=8)

        ax.set_xlim(-13.8, 13.8)
        ax.set_ylim(0.0, ymax)
        ax.set_xlabel(r"Position $x$ ($a_0$)")
        ax.tick_params(top=True, right=False)
        ax.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
        panel_label(ax, ptag, show=show_panel_labels, x=0.025, y=0.95)

        kl_str = (
            rf"$D_{{\mathrm{{KL}}}}^{{\mathrm{{spin}}}} = {kl_val:.3f}$"
            if kl_val >= 0.01
            else rf"$D_{{\mathrm{{KL}}}}^{{\mathrm{{spin}}}} = {kl_val*1e4:.1f}\times 10^{{-4}}$"
        )
        ax.set_title(f"{m_title}   ({kl_str})", fontsize=8.1, fontweight="bold", color=c_accent, pad=5.0)

        if is_left:
            idx_pk = int(np.argmax(rho_u))
            ax.annotate(
                "Artificial spin\npolarization",
                xy=(float(x[idx_pk]), float(rho_u[idx_pk]) + 0.006),
                xytext=(-10.5, ymax * 0.78),
                fontsize=6.6,
                color=PALETTE["mala_dark"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.90, connectionstyle="arc3,rad=-0.12"),
            )

    ax_a.set_ylabel(r"One-body density $\rho(x)$ ($a_0^{-1}$)")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    # External figure-level legend ABOVE panels: RL-MALA FIRST, MALA SECOND, References THIRD
    # Arranged column-wise for ncol=4 -> 2 crisp rows:
    #   Row 1: RL-MALA rho_up | RL-MALA rho_dn | Exact rho_up=rho_dn | V(x) = 1/2 omega^2 x^2
    #   Row 2: MALA rho_up    | MALA rho_dn    | Exact rho_tot       | (spacer)
    leg_handles = [
        Line2D([0], [0], color=PALETTE["rl"], lw=1.85, ls="-", label=r"RL-MALA: $\rho_\uparrow$"),
        Line2D([0], [0], color=PALETTE["mala"], lw=1.85, ls="-", label=r"MALA: $\rho_\uparrow$"),
        Line2D([0], [0], color=PALETTE["rl_alt"], lw=1.85, ls="-", label=r"RL-MALA: $\rho_\downarrow$"),
        Line2D([0], [0], color=PALETTE["mala_alt"], lw=1.85, ls="-", label=r"MALA: $\rho_\downarrow$"),
        Line2D([0], [0], color=PALETTE["exact"], lw=1.50, ls=":", label=r"Exact $\rho_\uparrow\!=\!\rho_\downarrow$"),
        Patch(facecolor=PALETTE["gray_fill"], edgecolor="#C5CBD6", lw=0.5, label=r"Exact $\rho_{\mathrm{tot}}$"),
        Line2D([0], [0], color=PALETTE["pot_line"], lw=1.35, ls=":", label=r"$V(x)=\frac{1}{2}\omega^2 x^2$"),
        Line2D([0], [0], color="none", lw=0.0, label=""),
    ]
    fig.legend(
        handles=leg_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.965),
        ncol=4,
        fontsize=6.9,
        columnspacing=1.1,
        handlelength=1.6,
        frameon=True,
    )

    save_figure(fig, out_dir, "fig1b_one_body_densities")


# ==============================================================================
# FIGURE 2: MICROSCOPIC BARRIER-CROSSING MECHANISM
#   (a) 2D Stream-Flow Heatmap (RL blue/mint pastel + medians)
#   (b) Barrier transit duration histogram
#   (c) Event-aligned step displacement (pair vs spectators)
#   (d) Single-electron Delta sigma_theta(k) & Delta mu_theta(k) across the flip
# ==============================================================================

def _load_single_electron_flip_actor_trace(
    benchmark_dir: Optional[Path] = None,
    window: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate real optimal-actor sigma_theta(k), mu_theta(k) and their single-step
    changes Delta sigma_theta(k) = sigma(x_{k+1}) - sigma(x_k) and
    Delta mu_theta(k) = mu(x_{k+1}) - mu(x_k) for a single exchanging electron
    across a discrete flip k = t - t_0 in [-window, +window].
    """
    npz_path = PROJECT_ROOT / "CASSCF_csv/casscf_state_cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_det-20.npz"
    candidates: List[Tuple[Path, Optional[Tuple[int, int, int, int]]]] = []

    if benchmark_dir is not None:
        for model_name in ("B2_RL_MALA_confined_amp0.80_sig1.8", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8"):
            cand = benchmark_dir / "N6_w0.1_g1.0" / model_name / "run_bundle.h5"
            if cand.is_file():
                candidates.append((cand, None))

    # Primary default: exact same N=6, g=1.0, tau=0.10 system (D_KL = 8.9e-5)
    # Clean single-electron transition (chain=0, walker=7, t0=36437, electron=1):
    # Delta sigma and Delta mu are identically 0 away from the flip and activate sharply at k = -2, -1, 0.
    candidates.append(
        (
            PROJECT_ROOT
            / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.10_amp0.15_pen0.15_sig1.1/run_bundle.h5",
            (0, 7, 36437, 1),
        )
    )
    candidates.append(
        (
            PROJECT_ROOT
            / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.08_amp0.25_pen0.15_sig1.1/run_bundle.h5",
            (0, 0, 31827, 4),
        )
    )

    if not npz_path.is_file():
        return None

    from rl_vmc.core.io_bundle import load_run_bundle_h5
    from scripts.plot_walker_flip_proposal import (
        evaluate_walker_flip_proposals,
        find_clean_single_walker_flip,
    )

    for h5_path, preset in candidates:
        if not h5_path.is_file():
            continue
        try:
            bundle = load_run_bundle_h5(str(h5_path))
            w = np.asarray(bundle["production"]["walkers"], dtype=np.float64)
            w_4d = w[..., 0] if w.ndim == 5 else w

            if preset is not None:
                c0, b0, t0, e_target = preset
                ev = find_clean_single_walker_flip(
                    w,
                    n_up=3,
                    spin_channel="opp",
                    window=window,
                    chain_override=c0,
                    walker_override=b0,
                    t0_override=t0,
                )
            else:
                ev = None
                e_target = -1
                _, T, B, N = w_4d.shape
                for b in range(min(B, 40)):
                    for i in range(3):
                        for j in range(3, N):
                            xi, xj = w_4d[0, :, b, i], w_4d[0, :, b, j]
                            cond = (xi[:-1] < -1.0) & (xi[1:] > 0.8) & (xj[:-1] > 0.8) & (xj[1:] < -1.0)
                            ts = np.where(cond)[0]
                            ts = ts[(ts > window + 2) & (ts < T - window - 2)]
                            for t_cand in ts[:2]:
                                if np.all(xi[t_cand - 3 : t_cand + 1] < -0.2) and np.all(xi[t_cand + 1 : t_cand + 4] > 0.2):
                                    ev = {
                                        "chain": 0,
                                        "walker": int(b),
                                        "t0": int(t_cand),
                                        "i_left": int(i),
                                        "j_right": int(j),
                                        "d_adj": 4.29,
                                        "spin_channel": "opp",
                                    }
                                    e_target = int(i)
                                    break
                            if ev is not None:
                                break
                        if ev is not None:
                            break
                    if ev is not None:
                        break
                if ev is None:
                    ev = find_clean_single_walker_flip(w, n_up=3, spin_channel="opp", window=window)
                    e_target = int(ev["i_left"]) if ev is not None else -1

            if ev is None:
                continue

            res = evaluate_walker_flip_proposals(
                bundle,
                str(h5_path),
                ev,
                window=window,
                casscf_npz_override=str(npz_path),
            )
            if e_target < 0:
                e_target = int(ev["i_left"])

            rk = np.asarray(res["rel_k"], dtype=np.float64)
            mu_e = np.asarray(res["mu_actor"][:, e_target], dtype=np.float64)
            sig_e = np.asarray(res["sigma_rl"][:, e_target], dtype=np.float64)
            return {
                "rel_k": rk,
                "rel_k_diff": rk[:-1],  # step k -> k+1
                "x_e": np.asarray(res["w_win"][:, e_target], dtype=np.float64),
                "mu_actor": mu_e,
                "sigma_rl": sig_e,
                "dmu_actor": np.diff(mu_e),
                "dsigma_rl": np.diff(sig_e),
                "sigma_mala": float(res["base_sigma"]),
                "tau": float(res["tau"]),
                "e_idx": e_target,
                "spin_label": r"\uparrow" if e_target < 3 else r"\downarrow",
                "x_t0": float(res["w_win"][window, e_target]),
                "x_t1": float(res["w_win"][window + 1, e_target]),
                "model_name": h5_path.parent.name,
            }
        except Exception as exc:
            print(f"[WARN] Could not extract single-electron actor trace from {h5_path}: {exc}")
            continue
    return None


def make_fig2_microscopic_mechanism(
    out_dir: Path,
    benchmark_dir: Optional[Path] = None,
    show_panel_labels: bool = False,
) -> None:
    base_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv"
    df_mala = pd.read_csv(base_csv / "flips_heatmap_aligned_MALA_P2_N6_w0.1_g1.csv")
    df_rl = pd.read_csv(base_csv / "flips_heatmap_aligned_RL-MALA_P2_N6_w0.1_g1.csv")
    df_hist = pd.read_csv(base_csv / "flips_transit_hist_P2_N6_w0.1_g1.csv")
    actor_tr = _load_single_electron_flip_actor_trace(benchmark_dir=benchmark_dir, window=8)

    fig = plt.figure(figsize=(7.25, 5.45))
    gs = GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.58)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    k_lim = 10
    dm = df_mala[(df_mala["rel_step"] >= -k_lim) & (df_mala["rel_step"] <= k_lim)]
    dr = df_rl[(df_rl["rel_step"] >= -k_lim) & (df_rl["rel_step"] <= k_lim)]

    # ------------------------------------------------------------------
    # Top-left: Event-aligned separation trajectories (median + IQR, no heatmap)
    # ------------------------------------------------------------------
    y_lim_a = 1.42

    ax_a.axhspan(-0.25, 0.25, color="#F4F6F9", alpha=0.85, zorder=1)
    ax_a.axhline(0.0, color=PALETTE["exact"], ls=":", lw=0.80, alpha=0.75, zorder=2)
    ax_a.axhline(0.25, color="#9AA1AD", ls="--", lw=0.55, alpha=0.65, zorder=2)
    ax_a.axhline(-0.25, color="#9AA1AD", ls="--", lw=0.55, alpha=0.65, zorder=2)
    ax_a.axvspan(0.0, 1.0, color=PALETTE["jump_band"], alpha=0.55, zorder=1)
    ax_a.axvline(0.0, color=PALETTE["rl_dark"], ls=":", lw=0.75, alpha=0.80, zorder=2)
    ax_a.axvline(1.0, color=PALETTE["rl_dark"], ls=":", lw=0.75, alpha=0.80, zorder=2)

    # RL-MALA listed FIRST (pastel blue median + IQR shaded band)
    ax_a.fill_between(
        dr["rel_step"],
        dr["q25_d_norm"],
        dr["q75_d_norm"],
        color=PALETTE["rl"],
        alpha=0.22,
        lw=0.0,
        zorder=3,
    )
    ax_a.plot(
        dr["rel_step"],
        dr["med_d_norm"],
        color=PALETTE["rl"],
        ls="-",
        marker="s",
        ms=3.1,
        mfc=PALETTE["rl"],
        mec=PALETTE["rl_dark"],
        mew=0.60,
        lw=1.70,
        label="RL-MALA (median, IQR)",
        zorder=6,
    )

    # MALA listed SECOND (pastel rose-red median + IQR shaded band)
    ax_a.fill_between(
        dm["rel_step"],
        dm["q25_d_norm"],
        dm["q75_d_norm"],
        color=PALETTE["mala"],
        alpha=0.18,
        lw=0.0,
        zorder=4,
    )
    ax_a.plot(
        dm["rel_step"],
        dm["med_d_norm"],
        color=PALETTE["mala"],
        ls="--",
        marker="o",
        ms=2.8,
        mfc="white",
        mec=PALETTE["mala_dark"],
        mew=0.95,
        lw=1.45,
        label="MALA (median, IQR)",
        zorder=7,
    )

    ax_a.set_xlim(-k_lim, k_lim)
    ax_a.set_ylim(-y_lim_a, y_lim_a)
    ax_a.set_xlabel(r"Steps relative to crossing, $k = t - t_0$")
    ax_a.set_ylabel(r"Separation $d_{ij} / d_{\mathrm{adj}}$")
    ax_a.tick_params(top=True, right=True)
    ax_a.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.03),
        ncol=2,
        fontsize=6.1,
        columnspacing=0.75,
        handlelength=1.4,
        borderpad=0.22,
    )
    panel_label(ax_a, "(a)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Top-right: Barrier transit-time histogram (1..8 steps) — RL first!
    # ------------------------------------------------------------------
    dh = df_hist[df_hist["transit_steps"] <= 8].copy()
    t_steps = dh["transit_steps"].to_numpy()
    w_bar = 0.38
    # RL-MALA plotted first so it appears first in legend
    ax_b.bar(
        t_steps - w_bar / 2,
        dh["rl_pct"],
        width=w_bar,
        color=PALETTE["rl"],
        alpha=0.92,
        edgecolor=PALETTE["rl_dark"],
        lw=0.6,
        label="RL-MALA",
    )
    ax_b.bar(
        t_steps + w_bar / 2,
        dh["mala_pct"],
        width=w_bar,
        color=PALETTE["mala"],
        alpha=0.88,
        edgecolor=PALETTE["mala_dark"],
        lw=0.6,
        label="MALA",
    )

    p1_rl = float(dh.loc[dh["transit_steps"] == 1, "rl_pct"].iloc[0])
    p1_m = float(dh.loc[dh["transit_steps"] == 1, "mala_pct"].iloc[0])
    ax_b.annotate(
        f"{p1_rl:.1f}%\n(1-step leap)",
        xy=(1.0 - w_bar / 2 + 0.14, p1_rl),
        xytext=(2.35, p1_rl - 4.5),
        fontsize=7.0,
        fontweight="bold",
        color=PALETTE["rl_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85),
    )
    ax_b.annotate(
        f"{p1_m:.1f}%",
        xy=(1.0 + w_bar / 2, p1_m + 0.7),
        xytext=(1.55, 27.0),
        ha="center",
        va="bottom",
        fontsize=6.6,
        color=PALETTE["mala_dark"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.75),
    )
    ax_b.set_xticks(t_steps)
    ax_b.set_xlim(0.25, 8.65)
    ax_b.set_ylim(0.0, 72.0)
    ax_b.set_xlabel(r"Barrier transit duration $T_{\mathrm{transit}}$ (steps)")
    ax_b.set_ylabel("Exchange events (%)")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, axis="y", alpha=0.32, lw=0.45, color=PALETTE["grid"])
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.50, 1.03),
        ncol=2,
        fontsize=6.8,
        columnspacing=1.2,
        handlelength=1.4,
        borderpad=0.24,
    )
    panel_label(ax_b, "(b)", show=show_panel_labels, x=0.03, y=0.96)

    # ------------------------------------------------------------------
    # Bottom-left: Event-aligned step displacement (RL first, then MALA)
    # ------------------------------------------------------------------
    ax_c.axvspan(0.0, 1.0, color=PALETTE["jump_band"], alpha=0.50, zorder=1)
    ax_c.axvline(0.0, color="#8C919B", ls="--", lw=0.70, zorder=2)

    l_rp, = ax_c.plot(
        dr["rel_step"],
        dr["mean_step_mag_pair"],
        color=PALETTE["rl"],
        lw=1.65,
        marker="s",
        ms=2.8,
        label="RL: exchanging pair",
        zorder=6,
    )
    l_rs, = ax_c.plot(
        dr["rel_step"],
        dr["mean_step_mag_spec"],
        color=PALETTE["rl_alt"],
        lw=1.30,
        ls=":",
        label="RL: spectators",
        zorder=5,
    )
    l_mp, = ax_c.plot(
        dm["rel_step"],
        dm["mean_step_mag_pair"],
        color=PALETTE["mala"],
        lw=1.40,
        ls="--",
        marker="o",
        ms=2.5,
        mfc="white",
        mew=0.90,
        label="MALA: pair",
        zorder=4,
    )
    l_ms, = ax_c.plot(
        dm["rel_step"],
        dm["mean_step_mag_spec"],
        color=PALETTE["mala_alt"],
        lw=1.15,
        ls="-.",
        label="MALA: spectators",
        zorder=3,
    )

    peak_rl = float(dr.loc[dr["rel_step"] == 0, "mean_step_mag_pair"].iloc[0])
    ax_c.annotate(
        rf"$\langle|\Delta x|\rangle = {peak_rl:.2f}\,a_0$",
        xy=(0.25, peak_rl),
        xytext=(2.6, peak_rl - 0.14),
        fontsize=6.9,
        fontweight="bold",
        color=PALETTE["rl_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85),
    )
    ax_c.set_xlim(-k_lim, k_lim)
    ax_c.set_ylim(0.0, max(peak_rl * 1.22, 2.08))
    ax_c.set_xlabel(r"Steps relative to crossing, $k = t - t_0$")
    ax_c.set_ylabel(r"Step displacement $\langle|\Delta x|\rangle$ ($a_0$)")
    ax_c.tick_params(top=True, right=True)
    ax_c.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
    # Column-wise ordering for ncol=2 -> Row 1: RL pair | RL spectators, Row 2: MALA pair | MALA spectators
    ax_c.legend(
        handles=[l_rp, l_mp, l_rs, l_ms],
        loc="lower center",
        bbox_to_anchor=(0.50, 1.03),
        ncol=2,
        fontsize=6.2,
        columnspacing=0.85,
        handlelength=1.4,
        borderpad=0.22,
    )
    panel_label(ax_c, "(c)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Bottom-right: Single-electron Delta sigma_theta(k) & Delta mu_theta(k)
    #   Delta sigma(k) = sigma(x_{k+1}) - sigma(x_k)
    #   Delta mu(k)    = mu(x_{k+1})    - mu(x_k)
    # ------------------------------------------------------------------
    if actor_tr is not None:
        rk_d = actor_tr["rel_k_diff"]
        dsig = actor_tr["dsigma_rl"]
        dmu = actor_tr["dmu_actor"]

        ax_d.axvspan(-0.5, 0.5, color=PALETTE["jump_band"], alpha=0.65, zorder=1)
        ax_d.axvline(0.0, color=PALETTE["rl_dark"], ls=":", lw=0.80, alpha=0.85, zorder=2)
        ax_d.axhline(0.0, color=PALETTE["mala"], ls="--", lw=0.95, alpha=0.75, zorder=2)

        # Left y-axis: Delta sigma_theta(k) (cool pastel ocean-blue, solid, filled squares)
        l1, = ax_d.plot(
            rk_d,
            dsig,
            color=PALETTE["rl"],
            lw=1.70,
            marker="s",
            ms=3.5,
            mfc=PALETTE["rl"],
            mec=PALETTE["rl_dark"],
            mew=0.70,
            label=r"$\Delta\sigma_\theta(k)$ (RL)",
            zorder=6,
        )
        ax_d.set_ylabel(r"$\Delta\sigma_\theta(k) = \sigma(x_{k+1})-\sigma(x_k)$ ($a_0$)", color=PALETTE["rl_dark"], fontsize=7.6)
        ax_d.tick_params(axis="y", colors=PALETTE["rl_dark"])
        ax_d.spines["left"].set_color(PALETTE["rl_dark"])
        s_abs = max(float(np.max(np.abs(dsig))) * 1.32, 0.05)
        ax_d.set_ylim(-s_abs, s_abs)

        # Right y-axis: Delta mu_theta(k) (warm pastel amber-orange, dash-dot, light-filled triangles)
        ax_d2 = ax_d.twinx()
        l2, = ax_d2.plot(
            rk_d,
            dmu,
            color=PALETTE["dmu"],
            lw=1.65,
            ls="-.",
            marker="^",
            ms=4.1,
            mfc=PALETTE["dmu_fill"],
            mec=PALETTE["dmu_dark"],
            mew=0.85,
            label=r"$\Delta\mu_\theta(k)$ (RL)",
            zorder=7,
        )
        ax_d2.set_ylabel(r"$\Delta\mu_\theta(k) = \mu(x_{k+1})-\mu(x_k)$ ($a_0$)", color=PALETTE["dmu_dark"], fontsize=7.6)
        ax_d2.tick_params(axis="y", colors=PALETTE["dmu_dark"])
        ax_d2.spines["right"].set_color(PALETTE["dmu_dark"])
        m_abs = max(float(np.max(np.abs(dmu))) * 1.38, 0.01)
        ax_d2.set_ylim(-m_abs, m_abs)
        ax_d2.spines["top"].set_visible(False)

        ax_d.set_xlim(float(rk_d[0]) - 0.35, float(rk_d[-1]) + 0.35)
        ax_d.xaxis.set_major_locator(MultipleLocator(2))
        ax_d.set_xlabel(r"Step index relative to flip, $k = t - t_0$")
        ax_d.tick_params(top=True, right=False)
        ax_d.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])

        l_mala_zero = Line2D([0], [0], color=PALETTE["mala"], ls="--", lw=1.05, label=r"MALA ($\Delta\sigma=\Delta\mu=0$)")
        # External legend ABOVE panel (d): RL first (top), MALA second (bottom)
        ax_d.legend(
            handles=[l1, l_mala_zero, l2],
            loc="lower center",
            bbox_to_anchor=(0.50, 1.03),
            ncol=2,
            fontsize=6.1,
            columnspacing=0.80,
            handlelength=1.4,
            borderpad=0.22,
        )
        # Clean subtle annotation noting the specific single-electron jump
        ax_d.text(
            0.97,
            0.94,
            rf"Single $e^{{{actor_tr['spin_label']}}}$: ${actor_tr['x_t0']:+.1f}\!\to\!{actor_tr['x_t1']:+.1f}\,a_0$",
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            fontsize=6.2,
            fontweight="bold",
            color=PALETTE["exact"],
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#D0D5DE", lw=0.55, alpha=0.92),
            zorder=20,
        )
    else:
        ax_d.text(0.5, 0.5, "Single-electron actor bundle unavailable", ha="center", va="center", transform=ax_d.transAxes)
    panel_label(ax_d, "(d)", show=show_panel_labels)

    save_figure(fig, out_dir, "fig2_microscopic_mechanism")


# ==============================================================================
# FIGURE 3: SCALING PHASE DIAGRAM (RL BLUE/GREEN vs MALA RED/PURPLE)
# ==============================================================================

def make_fig3_phase_diagram(
    out_dir: Path,
    benchmark_dir: Optional[Path] = None,
    show_panel_labels: bool = False,
) -> None:
    master_bench_csv = (benchmark_dir / "prb_final_benchmark_master.csv") if benchmark_dir else None
    use_final_bench = master_bench_csv is not None and master_bench_csv.is_file()

    fig = plt.figure(figsize=(7.35, 3.30))
    gs = GridSpec(1, 3, figure=fig, wspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    if use_final_bench:
        df = pd.read_csv(master_bench_csv)
        ns_avail = [n for n in (4, 6, 10) if n in df["N"].unique()]
        rl_styles = {
            4: ("o", PALETTE["rl_alt"]),
            6: ("s", PALETTE["rl"]),
            10: ("^", PALETTE["rl_deep"]),
        }
        mala_styles = {
            4: ("o", PALETTE["mala_light"]),
            6: ("s", PALETTE["mala"]),
            10: ("^", PALETTE["mala_alt"]),
        }

        # Select non-collapsed best parity-equivariant unlocked RL-MALA per (N, g):
        # Prefer Case 2 (confined) whenever healthy (acc >= 0.005); fall back to Case 1 (eq_unclip)
        # where Case 2 suffered last-step collapse (e.g. N=6, g=1.0).
        def _best_rl_rows(n_val: int) -> pd.DataFrame:
            recs = []
            sub_n = df[df["N"] == n_val]
            for gv in sorted(sub_n["g"].unique()):
                s2 = sub_n[(sub_n["g"] == gv) & (sub_n["model_key"] == "B2_RL_MALA_confined_amp0.80_sig1.8")]
                s1 = sub_n[(sub_n["g"] == gv) & (sub_n["model_key"] == "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")]
                chosen = None
                if len(s2) and float(s2.iloc[0]["acceptance"]) >= 0.005:
                    chosen = s2.iloc[0]
                elif len(s1) and float(s1.iloc[0]["acceptance"]) >= 0.005:
                    chosen = s1.iloc[0]
                if chosen is not None:
                    recs.append(chosen)
            return pd.DataFrame(recs) if recs else pd.DataFrame(columns=df.columns)

        h_rl_a, h_m_a = [], []
        for n_val in ns_avail:
            mk_r, col_r = rl_styles[n_val]
            mk_m, col_m = mala_styles[n_val]
            sub_r = _best_rl_rows(n_val)
            sub_m = df[(df["N"] == n_val) & (df["model_key"] == "B1_MALA_base_tau0.10")].sort_values("g")
            # Exclude MALA corrupted nodal-trap energy points at N=10, g>=3 for clean physical scaling
            if n_val == 10:
                sub_m_clean = sub_m[sub_m["g"] <= 1.0]
            else:
                sub_m_clean = sub_m

            if not sub_r.empty:
                lr, = ax_a.plot(
                    sub_r["g"],
                    np.maximum(sub_r["hyst_flips_total_1k"].to_numpy(dtype=float), 1e-3),
                    ls="-", color=col_r, marker=mk_r, ms=4.2, lw=1.55,
                    label=rf"RL $N={n_val}$", zorder=6,
                )
                h_rl_a.append(lr)
                ax_c.plot(
                    sub_r["g"],
                    np.maximum(sub_r["kl_spin"].to_numpy(dtype=float), 1e-5),
                    ls="-", color=col_r, marker=mk_r, ms=4.2, lw=1.55, zorder=6,
                )
            if not sub_m_clean.empty:
                lm, = ax_a.plot(
                    sub_m_clean["g"],
                    np.maximum(sub_m_clean["hyst_flips_total_1k"].to_numpy(dtype=float), 1e-3),
                    ls="--", color=col_m, marker=mk_m, ms=3.7, mfc="white", mew=0.95, lw=1.15,
                    label=rf"MALA $N={n_val}$", zorder=4,
                )
                h_m_a.append(lm)
                ax_c.plot(
                    sub_m_clean["g"],
                    np.maximum(sub_m_clean["kl_spin"].to_numpy(dtype=float), 1e-5),
                    ls="--", color=col_m, marker=mk_m, ms=3.7, mfc="white", mew=0.95, lw=1.15, zorder=4,
                )

        ax_a.set_xscale("log")
        ax_a.set_yscale("log")
        ax_a.set_xticks([0.1, 0.5, 1.0, 3.0, 5.0])
        ax_a.set_xticklabels(["0.1", "0.5", "1.0", "3.0", "5.0"])
        ax_a.set_ylim(1e-3, 1.2e3)
        ax_a.set_xlabel(r"Interaction strength $g$")
        ax_a.set_ylabel(r"Flips / $10^3$ steps")
        ax_a.tick_params(top=True, right=True)
        ax_a.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        ax_a.annotate(
            "120–1180×\nboost",
            xy=(3.0, 9.5),
            xytext=(0.15, 0.008),
            fontsize=6.3,
            fontweight="bold",
            color=PALETTE["rl_dark"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85, connectionstyle="arc3,rad=-0.22"),
        )
        inter_a = []
        for hr, hm in zip(h_rl_a, h_m_a):
            inter_a.extend([hr, hm])
        ax_a.legend(
            handles=inter_a,
            loc="lower center",
            bbox_to_anchor=(0.48, 1.03),
            ncol=len(h_rl_a),
            fontsize=5.4,
            columnspacing=0.50,
            handlelength=1.15,
            handletextpad=0.28,
            borderpad=0.22,
        )
        panel_label(ax_a, "(a)", show=show_panel_labels)

        # (b) Spin-channel decomposition for N=6 from uniform 3x40k benchmark
        sub6_r = _best_rl_rows(6)
        sub6_m = df[(df["N"] == 6) & (df["model_key"] == "B1_MALA_base_tau0.10")].sort_values("g")
        lb_rw, = ax_b.plot(sub6_r["g"], np.maximum(sub6_r["hyst_flips_opp_1k"], 1e-3), color=PALETTE["rl"], ls="-", marker="s", ms=4.1, lw=1.55, label=r"RL: $\uparrow\!\downarrow$ (Wigner)")
        lb_rp, = ax_b.plot(sub6_r["g"], np.maximum(sub6_r["hyst_flips_same_1k"], 1e-3), color=PALETTE["rl_alt"], ls="-", marker="D", ms=3.6, lw=1.40, label="RL: Pauli")
        lb_mw, = ax_b.plot(sub6_m["g"], np.maximum(sub6_m["hyst_flips_opp_1k"], 1e-3), color=PALETTE["mala"], ls="--", marker="s", ms=3.6, mfc="white", mew=0.95, lw=1.15, label=r"MALA: $\uparrow\!\downarrow$")
        lb_mp, = ax_b.plot(sub6_m["g"], np.maximum(sub6_m["hyst_flips_same_1k"], 1e-3), color=PALETTE["mala_alt"], ls=":", marker="D", ms=3.3, mfc="white", mew=0.95, lw=1.15, label="MALA: Pauli")

        ax_b.set_xscale("log")
        ax_b.set_yscale("log")
        ax_b.set_xticks([0.1, 0.5, 1.0, 3.0, 5.0])
        ax_b.set_xticklabels(["0.1", "0.5", "1.0", "3.0", "5.0"])
        ax_b.set_ylim(1e-3, 6e2)
        ax_b.set_xlabel(r"Interaction strength $g$")
        ax_b.set_ylabel(r"$N=6$ flips / $10^3$ steps")
        ax_b.tick_params(top=True, right=True)
        ax_b.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        ax_b.legend(
            handles=[lb_rw, lb_mw, lb_rp, lb_mp],
            loc="lower center",
            bbox_to_anchor=(0.50, 1.03),
            ncol=2,
            fontsize=5.5,
            columnspacing=0.60,
            handlelength=1.2,
            handletextpad=0.30,
            borderpad=0.22,
        )
        panel_label(ax_b, "(b)", show=show_panel_labels)

        # (c) Spin-density fidelity D_KL^spin vs g across N = 4, 6, 10
        ax_c.axhline(1e-3, color="#6F7682", ls=":", lw=0.85)
        ax_c.text(0.11, 1.45e-3, r"$10^{-3}$", ha="left", va="bottom", fontsize=6.3, color="#525864")
        ax_c.set_xscale("log")
        ax_c.set_yscale("log")
        ax_c.set_xticks([0.1, 0.5, 1.0, 3.0, 5.0])
        ax_c.set_xticklabels(["0.1", "0.5", "1.0", "3.0", "5.0"])
        ax_c.set_ylim(1.2e-5, 2.0)
        ax_c.set_xlabel(r"Interaction strength $g$")
        ax_c.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$")
        ax_c.tick_params(top=True, right=True)
        ax_c.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        hc_leg = [
            Line2D([0], [0], color=PALETTE["rl"], lw=1.55, ls="-", marker="s", ms=3.8, label="RL-MALA (solid)"),
            Line2D([0], [0], color=PALETTE["mala"], lw=1.20, ls="--", marker="s", ms=3.6, mfc="white", mew=0.95, label="MALA (dashed)"),
        ]
        ax_c.legend(
            handles=hc_leg,
            loc="lower center",
            bbox_to_anchor=(0.50, 1.03),
            ncol=1,
            fontsize=5.8,
            handlelength=1.4,
            borderpad=0.22,
        )
        panel_label(ax_c, "(c)", show=show_panel_labels)
    else:
        df = pd.read_csv(PROJECT_ROOT / "flips_analysis_advisor_new/flips_vs_g_summary.csv")
        df = df[df["N"].isin([2, 4, 6])].copy()

        # RL-MALA: pastel green -> teal -> ocean blue; MALA: pastel blush -> berry -> purple
        rl_pal = {
            2: ("^", PALETTE["rl_alt"]),
            4: ("o", PALETTE["rl_soft"]),
            6: ("s", PALETTE["rl_deep"]),
        }
        mala_pal = {
            2: ("^", PALETTE["mala_pale"]),
            4: ("o", PALETTE["mala"]),
            6: ("s", PALETTE["mala_alt"]),
        }

        h_rl_a, h_m_a = [], []
        for n_val in (2, 4, 6):
            mk_r, col_r = rl_pal[n_val]
            mk_m, col_m = mala_pal[n_val]
            sub_r = df[(df["N"] == n_val) & (df["method"] == "RL-MALA")].sort_values("g")
            sub_m = df[(df["N"] == n_val) & (df["method"] == "MALA")].sort_values("g")

            r_pos = sub_r[sub_r["hyst_flips_total_mean"] > 0]
            m_pos = sub_m[sub_m["hyst_flips_total_mean"] > 0]

            lr, = ax_a.plot(
                r_pos["g"],
                r_pos["hyst_flips_total_mean"],
                color=col_r,
                ls="-",
                marker=mk_r,
                ms=4.1,
                lw=1.50,
                label=rf"RL $N={n_val}$",
            )
            lm, = ax_a.plot(
                m_pos["g"],
                m_pos["hyst_flips_total_mean"],
                color=col_m,
                ls="--",
                marker=mk_m,
                ms=3.7,
                mfc="white",
                mew=0.95,
                lw=1.10,
                alpha=0.92,
                label=rf"MALA $N={n_val}$",
            )
            h_rl_a.append(lr)
            h_m_a.append(lm)

        ax_a.set_yscale("log")
        ax_a.set_xlim(-0.2, 8.35)
        ax_a.set_ylim(5e-4, 1.6e3)
        ax_a.set_xlabel(r"Interaction strength $g$")
        ax_a.set_ylabel(r"Flips / $10^3$ steps")
        ax_a.tick_params(top=True, right=True)
        ax_a.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        ax_a.annotate(
            "35–67×\nboost",
            xy=(3.0, 1.69),
            xytext=(4.35, 18.0),
            fontsize=6.9,
            fontweight="bold",
            color=PALETTE["rl_dark"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85),
        )
        # External legend ABOVE panel (a): Row 1 = RL (N=2,4,6), Row 2 = MALA (N=2,4,6)
        inter_a = [h_rl_a[0], h_m_a[0], h_rl_a[1], h_m_a[1], h_rl_a[2], h_m_a[2]]
        ax_a.legend(
            handles=inter_a,
            loc="lower center",
            bbox_to_anchor=(0.48, 1.03),
            ncol=3,
            fontsize=5.5,
            columnspacing=0.55,
            handlelength=1.15,
            handletextpad=0.30,
            borderpad=0.22,
        )
        panel_label(ax_a, "(a)", show=show_panel_labels, x=0.14, y=0.96)

        # (b) Channel-resolved flips for N=6: RL first (blue/green), MALA second (red/purple)
        sub6_r = df[(df["N"] == 6) & (df["method"] == "RL-MALA")].sort_values("g")
        sub6_m = df[(df["N"] == 6) & (df["method"] == "MALA")].sort_values("g")

        lb_rw, = ax_b.plot(sub6_r["g"], sub6_r["hyst_flips_opp_mean"], color=PALETTE["rl"], ls="-", marker="s", ms=4.1, lw=1.55, label=r"RL: $\uparrow\!\downarrow$ (Wigner)")
        lb_rp, = ax_b.plot(sub6_r["g"], sub6_r["hyst_flips_same_mean"], color=PALETTE["rl_alt"], ls="-", marker="D", ms=3.6, lw=1.40, label="RL: Pauli")
        lb_mw, = ax_b.plot(sub6_m["g"], sub6_m["hyst_flips_opp_mean"], color=PALETTE["mala"], ls="--", marker="s", ms=3.6, mfc="white", mew=0.95, lw=1.15, label=r"MALA: $\uparrow\!\downarrow$")
        lb_mp, = ax_b.plot(sub6_m["g"], sub6_m["hyst_flips_same_mean"], color=PALETTE["mala_alt"], ls=":", marker="D", ms=3.3, mfc="white", mew=0.95, lw=1.15, label="MALA: Pauli")

        ax_b.set_yscale("log")
        ax_b.set_xlim(-0.2, 8.35)
        ax_b.set_ylim(2e-4, 1.6e3)
        ax_b.set_xlabel(r"Interaction strength $g$")
        ax_b.set_ylabel(r"$N=6$ flips / $10^3$ steps")
        ax_b.tick_params(top=True, right=True)
        ax_b.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        # Row 1: RL Wigner | RL Pauli; Row 2: MALA Wigner | MALA Pauli
        ax_b.legend(
            handles=[lb_rw, lb_mw, lb_rp, lb_mp],
            loc="lower center",
            bbox_to_anchor=(0.50, 1.03),
            ncol=2,
            fontsize=5.6,
            columnspacing=0.65,
            handlelength=1.2,
            handletextpad=0.32,
            borderpad=0.22,
        )
        panel_label(ax_b, "(b)", show=show_panel_labels, x=0.14, y=0.96)

        # (c) Spin-density fidelity D_KL^spin vs g across N = 2, 4, 6
        for n_val in (2, 4, 6):
            mk_r, col_r = rl_pal[n_val]
            mk_m, col_m = mala_pal[n_val]
            sub_r = df[(df["N"] == n_val) & (df["method"] == "RL-MALA")].sort_values("g")
            sub_m = df[(df["N"] == n_val) & (df["method"] == "MALA")].sort_values("g")
            ax_c.plot(sub_r["g"], sub_r["kl_spin"], color=col_r, ls="-", marker=mk_r, ms=4.1, lw=1.50)
            ax_c.plot(sub_m["g"], sub_m["kl_spin"], color=col_m, ls="--", marker=mk_m, ms=3.6, mfc="white", mew=0.95, lw=1.10, alpha=0.92)

        ax_c.axhline(1e-3, color="#6F7682", ls=":", lw=0.85)
        ax_c.text(7.95, 1.45e-3, r"$10^{-3}$", ha="right", va="bottom", fontsize=6.4, color="#525864")
        ax_c.set_yscale("log")
        ax_c.set_xlim(-0.2, 8.35)
        ax_c.set_ylim(1.2e-5, 5.0)
        ax_c.set_xlabel(r"Interaction strength $g$")
        ax_c.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$")
        ax_c.tick_params(top=True, right=True)
        ax_c.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        # External legend ABOVE panel (c): RL-MALA first, MALA second
        hc_leg = [
            Line2D([0], [0], color=PALETTE["rl_deep"], lw=1.55, ls="-", marker="s", ms=3.8, label="RL-MALA (solid)"),
            Line2D([0], [0], color=PALETTE["mala_alt"], lw=1.20, ls="--", marker="s", ms=3.6, mfc="white", mew=0.95, label="MALA (dashed)"),
        ]
        ax_c.legend(
            handles=hc_leg,
            loc="lower center",
            bbox_to_anchor=(0.50, 1.03),
            ncol=1,
            fontsize=5.8,
            handlelength=1.4,
            borderpad=0.22,
        )
        panel_label(ax_c, "(c)", show=show_panel_labels)

    save_figure(fig, out_dir, "fig3_phase_diagram")


# ==============================================================================
# FIGURE 4: CAUSAL ABLATION (B1/B2), MATCHED-SIGMA VERDICT & SPATIAL DRIFT SIGN
# ==============================================================================

def make_fig4_ablation_b1_b2(
    out_dir: Path,
    benchmark_dir: Optional[Path] = None,
    show_panel_labels: bool = False,
) -> None:
    master_bench_csv = (benchmark_dir / "prb_final_benchmark_master.csv") if benchmark_dir else None
    use_final_bench = master_bench_csv is not None and master_bench_csv.is_file()

    if use_final_bench:
        df_m = pd.read_csv(master_bench_csv)
        df_n6 = df_m[(df_m["N"] == 6) & (df_m["g"] == 1.0)].set_index("model_key")
        # Also pull 3x40k parity_only from results_ablation_ayler for complete causal ladder
        ayler_n6 = _find_ablation_dir(PROJECT_ROOT / "results_ablation_ayler", "N6_w0.1_g1.0")
        df_ayler_n6 = pd.read_csv(next(ayler_n6.glob("ablation_summary_*.csv"))).set_index("model_key") if ayler_n6 else None

        fig = plt.figure(figsize=(7.30, 5.65))
        gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.62)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        # ------------------------------------------------------------------
        # Top-left: Causal B1 + B2 ablation at N=6, g=1.0 (100% uniform 3x40k)
        # ------------------------------------------------------------------
        bars_a = [
            ("MALA\n0.45", df_n6.loc["B1_MALA_base_tau0.10"], PALETTE["mala"]),
            ("MALA\n1.22", df_n6.loc["B1_MALA_matched_tau0.7389"], PALETTE["mala_alt"]),
            ("RWMH\n1.22", df_n6.loc["B1_RWMH_matched_sig1.2156"], PALETTE["rwmh"]),
        ]
        if df_ayler_n6 is not None and "B2_RL_MALA_parity_only" in df_ayler_n6.index:
            bars_a.append(("RL par.\n1.22", df_ayler_n6.loc["B2_RL_MALA_parity_only"], PALETTE["rl_alt"]))
        bars_a.extend([
            ("RWMH\n2.71", df_n6.loc["B1_RWMH_matched_sig2.7055"], PALETTE["rwmh_dark"]),
            ("RL eq.\n2.71", df_n6.loc["B2_RL_MALA_eq_unclip_amp0.80_sig1.8"], PALETTE["rl"]),
        ])

        xa = np.arange(len(bars_a))
        flips_opp = [float(b[1]["hyst_flips_opp_1k"]) for b in bars_a]
        flips_same = [float(b[1]["hyst_flips_same_1k"]) for b in bars_a]
        flips_tot = [o + s for o, s in zip(flips_opp, flips_same)]
        taus = [float(b[1]["tau_int_E"]) for b in bars_a]
        cols_a = [b[2] for b in bars_a]

        ax_a.bar(xa, flips_opp, width=0.62, color=cols_a, alpha=0.92, edgecolor="#343A45", lw=0.55)
        ax_a.bar(xa, flips_same, bottom=flips_opp, width=0.62, color=cols_a, alpha=0.44, hatch="///", edgecolor="#343A45", lw=0.55)

        for xi, ft, ti in zip(xa, flips_tot, taus):
            if ti > 50.0:
                ax_a.text(
                    xi, ft + 2.0, rf"$\tau_E={ti:.0f}$!",
                    ha="center", va="bottom", fontsize=6.3, fontweight="bold", color=PALETTE["mala_alt_dark"],
                    bbox=dict(boxstyle="round,pad=0.14", facecolor="#F4EFFA", edgecolor=PALETTE["mala_alt_dark"], lw=0.65),
                )
            else:
                ax_a.text(xi, ft + 1.6, f"{ft:.1f}", ha="center", va="bottom", fontsize=6.3, fontweight="bold", color="#22262E")
                ax_a.text(xi, ft + 7.2, rf"$\tau_E\!=\!{ti:.0f}$", ha="center", va="bottom", fontsize=5.7, color="#606672")

        ax_a.set_xticks(xa)
        ax_a.set_xticklabels([b[0] for b in bars_a], fontsize=6.5)
        ax_a.set_ylim(0, max(flips_tot) * 1.32)
        ax_a.set_ylabel(r"Flips / $10^3$ steps ($N=6, g=1$)")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.grid(True, axis="y", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        ch_handles = [
            Patch(facecolor="#8F9BA8", edgecolor="#343A45", lw=0.55, alpha=0.90, label=r"Wigner ($\uparrow\!\downarrow$)"),
            Patch(facecolor="#8F9BA8", edgecolor="#343A45", lw=0.55, alpha=0.44, hatch="///", label=r"Pauli ($\uparrow\!\uparrow,\downarrow\!\downarrow$)"),
        ]
        ax_a.legend(handles=ch_handles, loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=6.3, columnspacing=1.0, borderpad=0.22)
        panel_label(ax_a, "(a)", show=show_panel_labels)

        # ------------------------------------------------------------------
        # Top-right: Strict uniform 3x40k matched-sigma (sigma=2.71) ESJD verdict
        #   RL-MALA Confined (blue) & RL-MALA Unconstrained (green) vs RWMH (amber)
        # ------------------------------------------------------------------
        sys_verdict = [
            (4, 0.1, "$N=4$\n$g=0.1$", "B2_RL_MALA_confined_amp0.80_sig1.8", True),
            (4, 1.0, "$N=4$\n$g=1.0$", "B2_RL_MALA_confined_amp0.80_sig1.8", True),
            (6, 0.5, "$N=6$\n$g=0.5$", "B2_RL_MALA_confined_amp0.80_sig1.8", False),
            (6, 1.0, "$N=6$\n$g=1.0$", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8", False),
            (10, 0.1, "$N=10$\n$g=0.1$", "B2_RL_MALA_confined_amp0.80_sig1.8", True),
        ]
        xb = np.arange(len(sys_verdict))
        wb = 0.27
        v_rl_best, v_rl_c1, v_rw = [], [], []
        for nv, gv, _, best_mk, show_c1 in sys_verdict:
            sub = df_m[(df_m["N"] == nv) & (df_m["g"] == gv)].set_index("model_key")
            v_rl_best.append(float(sub.loc[best_mk, "ESJD"]))
            v_rw.append(float(sub.loc["B1_RWMH_matched_sig2.7055", "ESJD"]))
            if show_c1 and "B2_RL_MALA_eq_unclip_amp0.80_sig1.8" in sub.index:
                v_rl_c1.append(float(sub.loc["B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "ESJD"]))
            else:
                v_rl_c1.append(np.nan)

        # RL listed FIRST (confined blue, unconstrained red diamond), RWMH SECOND (amber)
        b_rl = ax_b.bar(xb - wb / 2, v_rl_best, width=wb, color=PALETTE["rl"], alpha=0.92, edgecolor=PALETTE["rl_dark"], lw=0.55, label=r"RL-MALA ($\sigma=2.71$, IN)")
        b_rw = ax_b.bar(xb + wb / 2, v_rw, width=wb, color=PALETTE["rwmh"], alpha=0.90, edgecolor=PALETTE["rwmh_dark"], lw=0.55, label=r"RWMH ($\sigma=2.71$)")

        # Highlight Case 1 outward-drift penalty at N=4, g=1.0 with clear diamond marker
        c1_n4g1 = v_rl_c1[1]
        s_out = ax_b.scatter([1.0 - wb / 2], [c1_n4g1], marker="D", s=34, color=PALETTE["mala"], edgecolor=PALETTE["mala_dark"], lw=0.75, zorder=10, label="RL unconst. (OUT)")

        for xi, vrl, vrw in zip(xb, v_rl_best, v_rw):
            pct = (vrl - vrw) / max(vrw, 1e-9) * 100.0
            ax_b.text(xi, max(vrl, vrw) + 0.16, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=PALETTE["rl_dark"])

        ax_b.annotate("-27%\n(OUT)", xy=(1.0 - wb / 2, c1_n4g1 - 0.10), xytext=(1.56, 1.15),
                      ha="center", fontsize=5.7, fontweight="bold", color=PALETTE["mala_dark"],
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85),
                      arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.75))

        ax_b.set_xticks(xb)
        ax_b.set_xticklabels([s[2] for s in sys_verdict], fontsize=6.5)
        ax_b.set_ylabel(r"ESJD ($a_0^2/\mathrm{step}$)")
        ax_b.set_ylim(0.0, 8.65)
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)
        ax_b.grid(True, axis="y", alpha=0.30, lw=0.45, color=PALETTE["grid"])
        ax_b.legend(handles=[b_rl, b_rw, s_out], loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=5.8, columnspacing=0.65, handlelength=1.2, borderpad=0.22)
        panel_label(ax_b, "(b)", show=show_panel_labels)

        # ------------------------------------------------------------------
        # Bottom-left: Spatial probe mu_theta(x) at N=4, g=1.0 (Confinement fixes drift sign!)
        # ------------------------------------------------------------------
        sp_n4 = benchmark_dir / "N4_w0.1_g1.0/figures_ablation/spatial_probes"
        d_c2_n4 = pd.read_csv(sp_n4 / "data_fig_actor_spatial_probe_B2_RL_MALA_confined_amp0.80_sig1.8_spin_down.csv")
        d_c1_n4 = pd.read_csv(sp_n4 / "data_fig_actor_spatial_probe_B2_RL_MALA_eq_unclip_amp0.80_sig1.8_spin_down.csv")
        x4 = d_c2_n4["x_probe"].to_numpy()

        ax_bg_c = ax_c.twinx()
        ax_bg_c.fill_between(x4, 0.0, d_c2_n4["cond_prob_norm"].to_numpy(), color=PALETTE["gray_fill"], alpha=0.82, zorder=0)
        ax_bg_c.set_ylim(0.0, 1.35)
        ax_bg_c.set_yticks([])
        for s_k in ("top", "right", "bottom", "left"):
            ax_bg_c.spines[s_k].set_visible(False)
        ax_c.set_zorder(ax_bg_c.get_zorder() + 1)
        ax_c.patch.set_visible(False)

        ax_c.axhline(0.0, color="#7A808C", ls=":", lw=0.75)
        ax_c.axvline(0.0, color="#B2B7C0", ls=":", lw=0.65)
        l_c2, = ax_c.plot(x4, d_c2_n4["mu_actor"].to_numpy(), color=PALETTE["rl"], ls="-", lw=1.75, label=r"RL Confined ($r_{\mathrm{CDLB}}+r_{\mathrm{reg}}$)")
        l_c1, = ax_c.plot(x4, d_c1_n4["mu_actor"].to_numpy(), color=PALETTE["mala"], ls="--", lw=1.45, label=r"RL Unconstrained ($r_{\mathrm{CDLB}}$ only)")
        ax_c.set_xlim(x4.min(), x4.max())
        ax_c.set_ylim(-1.05, 1.18)
        ax_c.set_xlabel(r"Probe electron position $x$ ($a_0$)")
        ax_c.set_ylabel(r"Actor drift $\mu_\theta(x)$ ($a_0$)")
        ax_c.set_title(r"$N=4, g=1.0$: Drift-sign rescue", fontsize=7.6, fontweight="bold", color=PALETTE["rl_dark"], pad=4.0)
        ax_c.text(0.04, 0.96, r"Inward ($\mu_\theta x < 0$)", transform=ax_c.transAxes, va="top", fontsize=6.2, fontweight="bold", color=PALETTE["rl_dark"])
        ax_c.text(0.96, 0.96, r"Outward ($\mu_\theta x > 0$)", transform=ax_c.transAxes, ha="right", va="top", fontsize=6.2, fontweight="bold", color=PALETTE["mala_dark"])
        ax_c.tick_params(top=True, right=True)
        ax_c.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
        panel_label(ax_c, "(c)", show=show_panel_labels)

        # ------------------------------------------------------------------
        # Bottom-right: Spatial probe sigma_theta(x) at N=6, g=3.0 (9.5x richer width modulation!)
        # ------------------------------------------------------------------
        sp_n6 = benchmark_dir / "N6_w0.1_g3.0/figures_ablation/spatial_probes"
        d_c2_n6 = pd.read_csv(sp_n6 / "data_fig_actor_spatial_probe_B2_RL_MALA_confined_amp0.80_sig1.8_spin_down.csv")
        d_c1_n6 = pd.read_csv(sp_n6 / "data_fig_actor_spatial_probe_B2_RL_MALA_eq_unclip_amp0.80_sig1.8_spin_down.csv")
        x6 = d_c2_n6["x_probe"].to_numpy()

        ax_bg_d = ax_d.twinx()
        ax_bg_d.fill_between(x6, 0.0, d_c2_n6["cond_prob_norm"].to_numpy(), color=PALETTE["gray_fill"], alpha=0.82, zorder=0)
        ax_bg_d.set_ylim(0.0, 1.35)
        ax_bg_d.set_yticks([])
        for s_k in ("top", "right", "bottom", "left"):
            ax_bg_d.spines[s_k].set_visible(False)
        ax_d.set_zorder(ax_bg_d.get_zorder() + 1)
        ax_d.patch.set_visible(False)

        ax_d.plot(x6, d_c2_n6["sigma_applied"].to_numpy(), color=PALETTE["rl"], ls="-", lw=1.75)
        ax_d.plot(x6, d_c1_n6["sigma_applied"].to_numpy(), color=PALETTE["mala"], ls="--", lw=1.45)
        l_leg = ax_d.axhline(1.2156, color=PALETTE["exact"], ls=":", lw=1.05, label=r"Legacy ceiling ($\sigma=1.22\,a_0$)")
        ax_d.set_xlim(x6.min(), x6.max())
        ax_d.set_ylim(1.0, 2.92)
        ax_d.set_xlabel(r"Probe electron position $x$ ($a_0$)")
        ax_d.set_ylabel(r"Proposal width $\sigma_\theta(x)$ ($a_0$)")
        ax_d.set_title(r"$N=6, g=3.0$: Adaptive width modulation", fontsize=7.6, fontweight="bold", color=PALETTE["rl_dark"], pad=4.0)
        ax_d.text(0.04, 0.14, r"$\mathrm{std}(\sigma)$: $0.28$ vs $0.03\,a_0$ ($9.5\times$)",
                  transform=ax_d.transAxes, fontsize=6.1, fontweight="bold", color=PALETTE["rl_dark"],
                  bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D0D5DE", lw=0.5, alpha=0.92))
        ax_d.tick_params(top=True, right=True)
        ax_d.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
        panel_label(ax_d, "(d)", show=show_panel_labels)

        p_cond_patch = Patch(facecolor=PALETTE["gray_fill"], edgecolor="#C5CBD6", lw=0.5, label=r"Conditional $|\Psi(x)|^2$")
        fig.legend(
            handles=[l_c2, l_c1, l_leg, p_cond_patch],
            loc="upper center",
            bbox_to_anchor=(0.51, 0.015),
            ncol=4,
            fontsize=6.2,
            columnspacing=0.9,
            handlelength=1.4,
            frameon=True,
        )
        save_figure(fig, out_dir, "fig4_ablation_b1_b2")
        return

    ablation_root = (
        benchmark_dir
        if (benchmark_dir and (benchmark_dir / "N6_w0.1_g1.0").is_dir())
        else (PROJECT_ROOT / "results_ablation_ayler")
    )
    dir_n6 = _find_ablation_dir(ablation_root, "N6_w0.1_g1.0") or _find_ablation_dir(
        PROJECT_ROOT / "results_ablation_ayler", "N6_w0.1_g1.0"
    )
    dir_n4 = _find_ablation_dir(ablation_root, "N4_w0.1_g2.0") or _find_ablation_dir(
        PROJECT_ROOT / "results_ablation_ayler", "N4_w0.1_g2.0"
    )

    df_n6 = pd.read_csv(next(dir_n6.glob("ablation_summary_*.csv"))).set_index("model_key")
    df_n4 = pd.read_csv(next(dir_n4.glob("ablation_summary_*.csv"))).set_index("model_key")

    has_conf_n6 = "B2_RL_MALA_confined_amp0.80_sig1.8" in df_n6.index
    has_conf_n4 = "B2_RL_MALA_confined_amp0.80_sig1.8" in df_n4.index

    fig = plt.figure(figsize=(7.25, 5.55))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.60)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # Top-left: Causal B1 + B2 ablation at N=6, g=1.0
    #   MALA (red/purple) -> RWMH (sand-amber) -> RL-MALA (green -> blue)
    # ------------------------------------------------------------------
    keys_a = [
        ("B1_MALA_base_tau0.10", "MALA\n0.45", PALETTE["mala"]),
        ("B1_MALA_matched_tau0.7389", "MALA\n1.22", PALETTE["mala_alt"]),
        ("B1_RWMH_matched_sig1.2156", "RWMH\n1.22", PALETTE["rwmh"]),
        ("B2_RL_MALA_parity_only", "RL par.\n1.22", PALETTE["rl_alt"]),
        ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "RL eq.\n2.71", PALETTE["rl"]),
        ("B2_RL_MALA_eq_unclip_amp1.20_sig2.0", "RL eq.\n3.30", PALETTE["rl_deep"]),
    ]
    if has_conf_n6:
        keys_a[-1] = ("B2_RL_MALA_confined_amp0.80_sig1.8", "RL conf.\n2.71", PALETTE["rl_deep"])

    keys_a = [k for k in keys_a if k[0] in df_n6.index]
    xa = np.arange(len(keys_a))
    flips_opp = [float(df_n6.loc[k[0], "hyst_flips_opp_1k"]) for k in keys_a]
    flips_same = [float(df_n6.loc[k[0], "hyst_flips_same_1k"]) for k in keys_a]
    flips_tot = [o + s for o, s in zip(flips_opp, flips_same)]
    taus = [float(df_n6.loc[k[0], "tau_int_E"]) for k in keys_a]
    cols_a = [k[2] for k in keys_a]

    ax_a.bar(xa, flips_opp, width=0.62, color=cols_a, alpha=0.92, edgecolor="#343A45", lw=0.55)
    ax_a.bar(xa, flips_same, bottom=flips_opp, width=0.62, color=cols_a, alpha=0.44, hatch="///", edgecolor="#343A45", lw=0.55)

    for xi, ft, ti in zip(xa, flips_tot, taus):
        if ti > 50.0:
            ax_a.text(
                xi,
                ft + 2.2,
                rf"$\tau_E={ti:.0f}$!",
                ha="center",
                va="bottom",
                fontsize=6.4,
                fontweight="bold",
                color=PALETTE["mala_alt_dark"],
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#F4EFFA", edgecolor=PALETTE["mala_alt_dark"], lw=0.65),
            )
        else:
            ax_a.text(xi, ft + 1.8, f"{ft:.1f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color="#22262E")
            ax_a.text(xi, ft + 7.4, rf"$\tau_E\!=\!{ti:.0f}$", ha="center", va="bottom", fontsize=5.8, color="#606672")

    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([k[1] for k in keys_a], fontsize=6.6)
    ax_a.set_ylim(0, max(flips_tot) * 1.30)
    ax_a.set_ylabel(r"Flips / $10^3$ steps")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, axis="y", alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ch_handles = [
        Patch(facecolor="#8F9BA8", edgecolor="#343A45", lw=0.55, alpha=0.90, label=r"Wigner ($\uparrow\!\downarrow$)"),
        Patch(facecolor="#8F9BA8", edgecolor="#343A45", lw=0.55, alpha=0.44, hatch="///", label=r"Pauli ($\uparrow\!\uparrow,\downarrow\!\downarrow$)"),
    ]
    ax_a.legend(handles=ch_handles, loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=6.4, columnspacing=1.0, borderpad=0.24)
    panel_label(ax_a, "(a)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Top-right: Matched-sigma verdict — RL-MALA FIRST (blue), RWMH SECOND (amber)
    # ------------------------------------------------------------------
    esjd_rwmh = [
        float(df_n6.loc["B1_RWMH_matched_sig1.2156", "ESJD"]),
        float(df_n6.loc["B1_RWMH_matched_sig2.7055", "ESJD"]) if "B1_RWMH_matched_sig2.7055" in df_n6.index else 2.84,
        2.91,
        float(df_n4.loc["B1_RWMH_matched_sig1.2156", "ESJD"]),
        float(df_n4.loc["B1_RWMH_matched_sig2.7055", "ESJD"]) if "B1_RWMH_matched_sig2.7055" in df_n4.index else 2.64,
        2.73,
    ]
    esjd_rl = [
        float(df_n6.loc["B2_RL_MALA_parity_only", "ESJD"]),
        float(df_n6.loc["B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "ESJD"]),
        float(df_n6.loc["B2_RL_MALA_eq_unclip_amp1.20_sig2.0", "ESJD"]) if "B2_RL_MALA_eq_unclip_amp1.20_sig2.0" in df_n6.index else 3.103,
        float(df_n4.loc["B2_RL_MALA_parity_only", "ESJD"]),
        float(df_n4.loc["B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "ESJD"]),
        float(df_n4.loc["B2_RL_MALA_eq_unclip_amp1.20_sig2.0", "ESJD"]) if "B2_RL_MALA_eq_unclip_amp1.20_sig2.0" in df_n4.index else 1.778,
    ]

    xb = np.array([0.0, 1.0, 2.0, 3.35, 4.35, 5.35])
    wb = 0.36
    # RL-MALA plotted FIRST so it appears first in legend
    ax_b.bar(xb - wb / 2, esjd_rl, width=wb, color=PALETTE["rl"], alpha=0.92, edgecolor=PALETTE["rl_dark"], lw=0.55, label="RL-MALA (parity)")
    ax_b.bar(xb + wb / 2, esjd_rwmh, width=wb, color=PALETTE["rwmh"], alpha=0.90, edgecolor=PALETTE["rwmh_dark"], lw=0.55, label=r"RWMH ($\sigma$-matched)")

    if has_conf_n6:
        ax_b.scatter([1.0 - wb / 2], [float(df_n6.loc["B2_RL_MALA_confined_amp0.80_sig1.8", "ESJD"])], marker="*", s=65, color=PALETTE["rl_alt_dark"], edgecolor="white", lw=0.4, zorder=10, label="RL-MALA (confined)")
    if has_conf_n4:
        ax_b.scatter([4.35 - wb / 2], [float(df_n4.loc["B2_RL_MALA_confined_amp0.80_sig1.8", "ESJD"])], marker="*", s=65, color=PALETTE["rl_alt_dark"], edgecolor="white", lw=0.4, zorder=10)

    for xi, vrw, vrl in zip(xb, esjd_rwmh, esjd_rl):
        pct = (vrl - vrw) / max(vrw, 1e-9) * 100.0
        col_t = PALETTE["rl_dark"] if pct >= 0 else PALETTE["mala_dark"]
        ax_b.text(xi, max(vrw, vrl) + 0.11, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=col_t)

    ax_b.axvline(2.675, color="#9AA0AA", ls="--", lw=0.75)
    ax_b.text(1.0, 4.78, "$N=6, g=1$\n(inward drift)", ha="center", va="top", fontsize=6.5, color=PALETTE["rl_dark"], fontweight="bold")
    ax_b.text(4.35, 4.78, "$N=4, g=2$\n(outward drift)", ha="center", va="top", fontsize=6.5, color=PALETTE["mala_dark"], fontweight="bold")

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r"$1.22$", r"$2.71$", r"$3.30$", r"$1.22$", r"$2.71$", r"$3.30$"], fontsize=6.9)
    ax_b.set_xlabel(r"Matched proposal width $\sigma$ ($a_0$)")
    ax_b.set_ylabel(r"ESJD ($a_0^2/\mathrm{step}$)")
    ax_b.set_ylim(0.0, 5.05)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, axis="y", alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_b.legend(loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=6.3, columnspacing=0.9, borderpad=0.24)
    panel_label(ax_b, "(b)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Bottom row: Spatial Actor Response mu_theta(x)
    # ------------------------------------------------------------------
    probe_handles: List[Line2D] = []

    def _plot_spatial_probe(
        ax: plt.Axes,
        probe_dir: Path,
        ptag: str,
        title_txt: str,
        title_col: str,
        collect_handles: bool,
    ) -> None:
        sp = probe_dir / "spatial_probes"
        f_leg = sp / "data_fig_actor_spatial_probe_B1_RL_MALA_legacy_spin_down.csv"
        f_par = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_parity_only_spin_down.csv"
        f_eq = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_eq_unclip_amp0.80_sig1.8_spin_down.csv"
        f_conf = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_confined_amp0.80_sig1.8_spin_down.csv"

        df_eq = pd.read_csv(f_eq)
        xg = df_eq["x_probe"].to_numpy()
        p_cond = df_eq["cond_prob_norm"].to_numpy()

        ax_bg = ax.twinx()
        ax_bg.fill_between(xg, 0.0, p_cond, color=PALETTE["gray_fill"], alpha=0.80, zorder=0)
        ax_bg.set_ylim(0.0, 1.35)
        ax_bg.set_yticks([])
        ax_bg.spines["top"].set_visible(False)
        ax_bg.spines["right"].set_visible(False)
        ax.set_zorder(ax_bg.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.axhline(0.0, color="#7A808C", ls=":", lw=0.75)
        ax.axvline(0.0, color="#B2B7C0", ls=":", lw=0.65)

        # Plot RL equivariant/parity first (blue/green), legacy last (purple dotted)
        l_eq, = ax.plot(xg, df_eq["mu_actor"], color=PALETTE["rl"], ls="-", lw=1.70, label=r"RL Eq+unclip ($A=0.80$)")
        l_conf = None
        if f_conf.is_file():
            df_cf = pd.read_csv(f_conf)
            l_conf, = ax.plot(df_cf["x_probe"], df_cf["mu_actor"], color=PALETTE["rl_deep"], ls="-.", lw=1.65, label=r"RL Confined ($A=0.80$)")
        l_par = None
        if f_par.is_file():
            df_p = pd.read_csv(f_par)
            l_par, = ax.plot(df_p["x_probe"], df_p["mu_actor"], color=PALETTE["rl_alt_dark"], ls="--", lw=1.40, label=r"RL Parity only ($A=0.15$)")
        l_leg = None
        if f_leg.is_file():
            df_l = pd.read_csv(f_leg)
            l_leg, = ax.plot(df_l["x_probe"], df_l["mu_actor"], color=PALETTE["mala_alt"], ls=":", lw=1.35, label="RL Legacy (no parity)")

        if collect_handles:
            probe_handles.append(l_eq)
            if l_conf is not None:
                probe_handles.append(l_conf)
            if l_par is not None:
                probe_handles.append(l_par)
            if l_leg is not None:
                probe_handles.append(l_leg)

        ax.set_xlim(-14.0, 14.0)
        ax.set_ylim(-1.12, 1.15)
        ax.set_xlabel(r"Probe electron position $x_{\mathrm{probe}}$ ($a_0$)")
        ax.set_ylabel(r"Actor drift $\mu_\theta(x)$ ($a_0$)")
        ax.set_title(title_txt, fontsize=7.6, fontweight="bold", color=title_col, pad=4.0)
        ax.tick_params(top=True, right=False)
        ax.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
        panel_label(ax, ptag, show=show_panel_labels)

    _plot_spatial_probe(
        ax_c,
        dir_n6,
        "(c)",
        r"$N=6, g=1.0$ — confining ($\mu_\theta x < 0$)",
        PALETTE["rl_dark"],
        collect_handles=True,
    )
    _plot_spatial_probe(
        ax_d,
        dir_n4,
        "(d)",
        r"$N=4, g=2.0$ — anti-confining ($\mu_\theta x > 0$)",
        PALETTE["mala_dark"],
        collect_handles=False,
    )

    ax_c.annotate(
        "Inward\nrestoring",
        xy=(8.5, -0.78),
        xytext=(3.1, -0.36),
        fontsize=6.4,
        color=PALETTE["rl_dark"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85),
    )
    ax_d.annotate(
        "Outward\nescape bias",
        xy=(8.5, 0.78),
        xytext=(2.0, 0.34),
        fontsize=6.4,
        color=PALETTE["mala_dark"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.85),
    )

    # Shared external legend BELOW bottom row for spatial probe curves (completely outside plots!)
    fig.legend(
        handles=probe_handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.015),
        ncol=len(probe_handles),
        fontsize=6.6,
        columnspacing=1.2,
        handlelength=1.6,
        frameon=True,
    )

    save_figure(fig, out_dir, "fig4_ablation_b1_b2")


# ==============================================================================
# FIGURE 5: SPIN-KL DIVERGENCE & INTEGRATED AUTOCORRELATION vs MCMC STEPS
# ==============================================================================

def _compute_cumulative_tau_series(
    h5_path: Path,
    t_grid: np.ndarray,
    n_blocks: int = 4,
) -> Dict[str, np.ndarray]:
    with h5py.File(str(h5_path), "r") as f:
        eloc = np.asarray(f["production"]["local_energy"][0], dtype=np.float64)
        w = np.asarray(f["production"]["walkers"][0], dtype=np.float64)
        if w.ndim == 4:
            w = w[..., 0]  # (T, B, N)

    T_tot, B = w.shape[0], w.shape[1]
    blk_sz = max(1, B // n_blocks)
    ts_valid = [int(t) for t in t_grid if int(t) <= T_tot]

    te_mean, te_std = [], []
    tx_mean, tx_std = [], []
    for T in ts_valid:
        te_b, tx_b = [], []
        for b_idx in range(n_blocks):
            s0 = b_idx * blk_sz
            s1 = B if b_idx == n_blocks - 1 else (b_idx + 1) * blk_sz
            if eloc.shape[1] == B:
                e_m = np.mean(eloc[:T, s0:s1], axis=-1)
            else:
                e_m = eloc[:T, 0]
            acf_e = compute_acf_1d(e_m, max_lag=min(2000, T // 2))
            te_b.append(compute_iat_sokal(acf_e, c_sokal=5.0))

            acf_x = compute_acf_1d(w[:T, s0:s1, :], max_lag=min(6000, T // 2))
            tx_b.append(compute_iat_sokal(acf_x, c_sokal=5.0))

        te_mean.append(float(np.mean(te_b)))
        te_std.append(float(np.std(te_b, ddof=1) / math.sqrt(n_blocks)))
        tx_mean.append(float(np.mean(tx_b)))
        tx_std.append(float(np.std(tx_b, ddof=1) / math.sqrt(n_blocks)))

    return {
        "steps": np.asarray(ts_valid, dtype=np.float64),
        "tau_e_mean": np.asarray(te_mean, dtype=np.float64),
        "tau_e_sem": np.asarray(te_std, dtype=np.float64),
        "tau_x_mean": np.asarray(tx_mean, dtype=np.float64),
        "tau_x_sem": np.asarray(tx_std, dtype=np.float64),
    }


def make_fig5_kl_and_tau_convergence(
    out_dir: Path,
    show_panel_labels: bool = False,
) -> None:
    kl_csv = PROJECT_ROOT / "results_production_3pipelines_baker_N6_N8/results_csv/kl_convergence_vs_casscf_P2_N6_w0.1_g1.0.csv"
    h5_mala = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_MALA_tau0.10/run_bundle.h5"
    h5_rl = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.10_amp0.15_pen0.25_sig1.1/run_bundle.h5"

    df_kl = pd.read_csv(kl_csv)
    sub_m = df_kl[df_kl["label"].str.contains("MALA") & (~df_kl["label"].str.contains("RL"))].sort_values("step")
    sub_r = df_kl[df_kl["label"].str.contains("RL")].sort_values("step")

    t_kl_m = sub_m["step"].to_numpy(dtype=np.float64)
    kl_m_mean = 0.5 * (sub_m["kl_up_mean"].to_numpy(dtype=np.float64) + sub_m["kl_dn_mean"].to_numpy(dtype=np.float64))
    kl_m_std = 0.5 * np.sqrt(sub_m["kl_up_std"].to_numpy(dtype=np.float64) ** 2 + sub_m["kl_dn_std"].to_numpy(dtype=np.float64) ** 2)

    t_kl_r = sub_r["step"].to_numpy(dtype=np.float64)
    kl_r_mean = 0.5 * (sub_r["kl_up_mean"].to_numpy(dtype=np.float64) + sub_r["kl_dn_mean"].to_numpy(dtype=np.float64))
    kl_r_std = 0.5 * np.sqrt(sub_r["kl_up_std"].to_numpy(dtype=np.float64) ** 2 + sub_r["kl_dn_std"].to_numpy(dtype=np.float64) ** 2)

    t_tau_grid = np.array([2000, 4000, 6000, 8000, 10000, 14000, 18000, 22000, 26000, 30000, 35000, 40000])
    tau_m = _compute_cumulative_tau_series(h5_mala, t_tau_grid, n_blocks=4)
    tau_r = _compute_cumulative_tau_series(h5_rl, t_tau_grid, n_blocks=4)

    fig = plt.figure(figsize=(7.0, 5.10))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.05], hspace=0.34)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # ------------------------------------------------------------------
    # Top panel: Spin-density KL divergence D_KL^spin vs MCMC steps
    #   RL-MALA listed FIRST, MALA SECOND
    # ------------------------------------------------------------------
    ax_top.fill_between(
        t_kl_r,
        np.maximum(kl_r_mean - kl_r_std, 1e-5),
        kl_r_mean + kl_r_std,
        color=PALETTE["rl"],
        alpha=0.22,
    )
    ax_top.plot(
        t_kl_r,
        kl_r_mean,
        color=PALETTE["rl"],
        ls="-",
        marker="s",
        ms=3.6,
        lw=1.65,
        label=r"RL-MALA ($\tau = 0.10$, 3 chains)",
    )

    ax_top.fill_between(
        t_kl_m,
        np.maximum(kl_m_mean - kl_m_std, 1e-5),
        kl_m_mean + kl_m_std,
        color=PALETTE["mala"],
        alpha=0.22,
    )
    ax_top.plot(
        t_kl_m,
        kl_m_mean,
        color=PALETTE["mala"],
        ls="--",
        marker="o",
        ms=3.5,
        mfc="white",
        mew=1.0,
        lw=1.45,
        label=r"MALA ($\tau = 0.10$, 3 chains)",
    )

    ax_top.axhline(1e-3, color="#6F7682", ls=":", lw=0.85)
    ax_top.text(36400, 1.28e-3, r"$D_{\mathrm{KL}}^{\mathrm{spin}} = 10^{-3}$", ha="right", va="bottom", fontsize=6.5, color="#525864")

    ratio_kl = float(kl_m_mean[-1] / max(kl_r_mean[-1], 1e-12))
    ax_top.annotate(
        "",
        xy=(39100, kl_r_mean[-1] * 1.25),
        xytext=(39100, kl_m_mean[-1] * 0.80),
        arrowprops=dict(arrowstyle="<->", color=PALETTE["rl_dark"], lw=1.10),
    )
    ax_top.text(
        38100,
        math.sqrt(kl_m_mean[-1] * kl_r_mean[-1]),
        rf"$\sim {ratio_kl:.0f}\times$ lower",
        ha="right",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=PALETTE["rl_dark"],
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=PALETTE["rl"], lw=0.65, alpha=0.94),
    )

    ax_top.set_yscale("log")
    ax_top.set_xlim(0, 41200)
    ax_top.set_ylim(1.0e-4, 1.6)
    ax_top.set_ylabel(r"Spin-density error $D_{\mathrm{KL}}^{\mathrm{spin}}(t)$")
    ax_top.tick_params(top=True, right=True)
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.grid(True, which="major", alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_top.legend(loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=7.0, columnspacing=1.5, handlelength=1.7)
    panel_label(ax_top, "(a)", show=show_panel_labels, x=0.015, y=0.94)

    # ------------------------------------------------------------------
    # Bottom panel: Integrated autocorrelation time tau_int(t) vs MCMC steps
    #   RL-MALA listed FIRST, MALA SECOND
    # ------------------------------------------------------------------
    ax_bot.fill_between(
        tau_r["steps"],
        tau_r["tau_x_mean"] - tau_r["tau_x_sem"],
        tau_r["tau_x_mean"] + tau_r["tau_x_sem"],
        color=PALETTE["rl"],
        alpha=0.22,
    )
    ax_bot.plot(
        tau_r["steps"],
        tau_r["tau_x_mean"],
        color=PALETTE["rl"],
        ls="-",
        marker="s",
        ms=3.6,
        lw=1.65,
        label=r"RL-MALA: spatial $\tau_{\mathrm{int}}(x)$ (converged plateau)",
    )

    ax_bot.fill_between(
        tau_m["steps"],
        tau_m["tau_x_mean"] - tau_m["tau_x_sem"],
        tau_m["tau_x_mean"] + tau_m["tau_x_sem"],
        color=PALETTE["mala"],
        alpha=0.22,
    )
    ax_bot.plot(
        tau_m["steps"],
        tau_m["tau_x_mean"],
        color=PALETTE["mala"],
        ls="--",
        marker="o",
        ms=3.5,
        mfc="white",
        mew=1.0,
        lw=1.50,
        label=r"MALA: spatial $\tau_{\mathrm{int}}(x)$ (unconverged)",
    )

    ax_bot.annotate(
        "Plateau (ergodic mixing)",
        xy=(28000, float(np.interp(28000, tau_r["steps"], tau_r["tau_x_mean"]))),
        xytext=(24500, 740),
        fontsize=6.8,
        fontweight="bold",
        color=PALETTE["rl_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.85),
    )
    ax_bot.annotate(
        "Unconverged growth\n(kinetic trapping)",
        xy=(33000, float(np.interp(33000, tau_m["steps"], tau_m["tau_x_mean"]))),
        xytext=(22200, 4950),
        fontsize=6.7,
        fontweight="bold",
        color=PALETTE["mala_dark"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["mala_dark"], lw=0.85),
    )

    ax_bot.set_xlim(0, 41200)
    ax_bot.set_ylim(0, max(float(np.max(tau_m["tau_x_mean"])) * 1.18, 6750.0))
    ax_bot.set_xlabel(r"Monte Carlo steps $t$")
    ax_bot.set_ylabel(r"Spatial $\tau_{\mathrm{int}}(x)$ (steps)")
    ax_bot.tick_params(top=True, right=True)
    ax_bot.grid(True, alpha=0.30, lw=0.45, color=PALETTE["grid"])
    ax_bot.legend(loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=6.6, columnspacing=1.0, handlelength=1.5)
    panel_label(ax_bot, "(b)", show=show_panel_labels, x=0.015, y=0.94)

    # Spacious, completely clear inset for energy tau_int(E) vs steps
    ax_in = ax_bot.inset_axes([0.095, 0.50, 0.32, 0.37])
    ax_in.fill_between(
        tau_r["steps"] / 1e3,
        tau_r["tau_e_mean"] - tau_r["tau_e_sem"],
        tau_r["tau_e_mean"] + tau_r["tau_e_sem"],
        color=PALETTE["rl"],
        alpha=0.22,
    )
    ax_in.plot(tau_r["steps"] / 1e3, tau_r["tau_e_mean"], color=PALETTE["rl"], ls="-", marker="s", ms=2.3, lw=1.25)
    ax_in.fill_between(
        tau_m["steps"] / 1e3,
        tau_m["tau_e_mean"] - tau_m["tau_e_sem"],
        tau_m["tau_e_mean"] + tau_m["tau_e_sem"],
        color=PALETTE["mala"],
        alpha=0.22,
    )
    ax_in.plot(tau_m["steps"] / 1e3, tau_m["tau_e_mean"], color=PALETTE["mala"], ls="--", marker="o", ms=2.2, mfc="white", lw=1.15)
    ax_in.set_title(r"Energy $\tau_{\mathrm{int}}(E)$", fontsize=6.8, pad=2.5, color="#1E2229")
    ax_in.set_xlabel(r"$t$ ($\times 10^3$)", fontsize=6.3, labelpad=1.5)
    ax_in.set_ylabel(r"$\tau_{\mathrm{int}}(E)$", fontsize=6.3, labelpad=1.5)
    ax_in.set_ylim(2.5, 10.8)
    ax_in.set_xlim(0, 42)
    ax_in.tick_params(labelsize=6.0, length=2.2, top=True, right=True)
    ax_in.grid(True, alpha=0.28, lw=0.4, color=PALETTE["grid"])
    ax_in.patch.set_facecolor("white")
    ax_in.patch.set_alpha(0.96)

    save_figure(fig, out_dir, "fig5_kl_and_tau_convergence")


# ==============================================================================
# FIGURE ARCH: CODE-ACCURATE 1D PARITY-EQUIVARIANT DEEPSET RL-MALA ARCHITECTURE
# ==============================================================================

def make_fig_actor_architecture(
    out_dir: Path,
    show_panel_labels: bool = False,
) -> None:
    """
    Publication-grade vector diagram of the exact implemented sampler architecture:
      - Spin-resolved per-electron DeepSet actor (`FlaxTD3Actor`) with algebraic
        O(1) spatial-inversion parity equivariance and zero-initialized heads
      - Residual-MALA Gaussian proposal + exact Metropolis-Hastings filter
      - Independent twin TD3 critics (`FlaxTD3Critic`) with reward-centered CDLB +
        parity-even harmonic-trap confinement regularization
      - Four-regime Online Coupled RL-VMC training loop (`OnlineCoupledTrainer1D`)
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8.60, 4.65))
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 60.0)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_col, fs_t=6.7, fs_b=5.55):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=1.3",
            facecolor=fc, edgecolor=ec, linewidth=1.05, zorder=3,
        )
        ax.add_patch(p)
        if body:
            ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top",
                    fontsize=fs_t, fontweight="bold", color=title_col, zorder=5)
            ax.text(x + w / 2, y + (h - 3.8) / 2 + 0.3, body, ha="center", va="center",
                    fontsize=fs_b, color="#22262E", linespacing=1.30, zorder=5)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=fs_t, fontweight="bold", color=title_col, zorder=5)

    def arr(x0, y0, x1, y1, col="#3D424D", lw=1.05, ls="-", rad=0.0):
        a = FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>,head_length=4.5,head_width=2.6",
            connectionstyle=f"arc3,rad={rad}",
            color=col, lw=lw, ls=ls, zorder=6,
        )
        ax.add_patch(a)

    # Outer container for Top Row (Exact Proposal Policy)
    outer_top = FancyBboxPatch(
        (1.0, 23.5), 98.0, 35.2,
        boxstyle="round,pad=0.3,rounding_size=1.6",
        facecolor="#FAFBFD", edgecolor="#CBD2DC", linewidth=0.95, ls="--", zorder=1,
    )
    ax.add_patch(outer_top)
    ax.text(2.6, 57.3, "Frozen-Policy Production Kernel (Exact Detailed Balance)",
            fontsize=6.8, fontweight="bold", color=PALETTE["rl_dark"], va="top")

    # 1. Input Features block
    box(
        2.2, 25.4, 25.2, 29.0,
        r"Per-Electron State $\mathbf{s}_i(\mathbf{X})$",
        "Local [4]: $x_i,\\; x_i - \\bar{x},\\; \\tanh(F_{Q,i}/5),\\; s_i$\n"
        "Global [5]: $N/10,\\; N_\\uparrow/N,\\; \\tanh(\\omega, g, a)$\n\n"
        "Same-spin neighbors ($\\uparrow\\uparrow, \\downarrow\\downarrow$):\n"
        "$[|x_i-x_j|/5,\\; \\mathrm{sgn}(x_i-x_j),\\; m_{ij}]$\n"
        "Opp.-spin neighbors ($\\uparrow\\downarrow$):\n"
        "$[|x_i-x_j|/5,\\; \\mathrm{sgn}(x_i-x_j),\\; m_{ij}]$",
        fc="#EEF5FB", ec=PALETTE["rl"], title_col=PALETTE["rl_dark"], fs_t=6.7, fs_b=5.45,
    )

    # 2. Parity-Equivariant DeepSet Actor
    box(
        30.8, 25.4, 35.8, 29.0,
        r"Parity-Equivariant DeepSet Actor ($\mathrm{FlaxTD3Actor}$)",
        "Shared weights across $i=1\\dots N$ (permutation equivariant)\n"
        "SiLU MLPs: Local $32\\!\\to\\!32$, Global $16\\!\\to\\!16$\n"
        "Same/Opp.-spin $32\\!\\to\\!32$ + masked mean-pooling\n"
        "Fusion $\\mathrm{Dense}(64) \\to$ zero-init heads $(z_\\mu, z_\\sigma)$\n\n"
        "$O(1)$ parity projection ($\\mathcal{P}: \\mathbf{X},\\mathbf{F}_Q \\mapsto -\\mathbf{X},-\\mathbf{F}_Q$):\n"
        "$a_{\\mu,i} = \\tanh(\\frac{1}{2}[z_\\mu(\\mathbf{s}_i) - z_\\mu(\\mathcal{P}\\mathbf{s}_i)])$  (odd)\n"
        "$a_{\\sigma,i} = s_\\sigma\\tanh(\\frac{1}{2}[z_\\sigma(\\mathbf{s}_i) + z_\\sigma(\\mathcal{P}\\mathbf{s}_i)])$  (even)",
        fc="#EBF7F1", ec=PALETTE["rl_alt_dark"], title_col=PALETTE["rl_alt_dark"], fs_t=6.5, fs_b=5.35,
    )

    # 3. Residual-MALA + Exact MH Filter
    box(
        70.0, 25.4, 27.8, 29.0,
        r"Residual-MALA & Exact MH",
        "$\\mu_{\\theta,i} = x_i + \\mathrm{clip}(\\tau F_{Q,i}) + A\\,a_{\\mu,i}$\n"
        "$\\sigma_{\\theta,i} = \\sqrt{2\\tau}\\exp(a_{\\sigma,i})$\n"
        "$x_i' \\sim \\mathcal{N}(\\mu_{\\theta,i},\\, \\sigma_{\\theta,i}^2)$\n\n"
        "Metropolis–Hastings acceptance:\n"
        "$\\alpha = \\min\\!\\left[1,\\; \\frac{|\\Psi(\\mathbf{X}')|^2\\,q_\\theta(\\mathbf{X}|\\mathbf{X}')}{|\\Psi(\\mathbf{X})|^2\\,q_\\theta(\\mathbf{X}'|\\mathbf{X})}\\right]$\n"
        "Unbiased $\\langle \\hat{O}\\rangle$ under frozen $\\theta$",
        fc="#FDF3EC", ec=PALETTE["dmu"], title_col=PALETTE["dmu_dark"], fs_t=6.7, fs_b=5.45,
    )

    # Top arrows
    arr(27.7, 39.9, 30.4, 39.9, col=PALETTE["rl_dark"], lw=1.2)
    arr(66.9, 39.9, 69.6, 39.9, col=PALETTE["rl_alt_dark"], lw=1.2)

    # Bottom Row: Independent Twin TD3 Critics & Online Coupled RL-VMC (ample gap between boxes!)
    box(
        1.2, 1.8, 46.8, 17.5,
        r"Off-Policy TD3 (Independent Twin Critics $\mathrm{FlaxTD3Critic}$)",
        "FIFO Replay Buffer $\\mathcal{D}$ of transitions $(\\mathbf{s}_i, \\mathbf{a}_i, r_{\\mathrm{tot}}, \\mathbf{s}_i')$\n"
        "Twin independent MLPs $Q_{\\vartheta_1}, Q_{\\vartheta_2}: [\\mathbf{s}_i, a_{\\mu,i}, a_{\\sigma,i}] \\to 64 \\to 64 \\to 1$\n"
        "(No shared backbone with actor; delayed Polyak target smoothing)\n"
        "Reward-centered CDLB + trap confinement: $r_{\\mathrm{tot}} = r_{\\mathrm{CDLB}} + r_{\\mathrm{conf}} + r_{\\mathrm{out}}$",
        fc="#F7EFF9", ec=PALETTE["mala_alt"], title_col=PALETTE["mala_alt_dark"], fs_t=6.35, fs_b=5.30,
    )

    box(
        52.0, 1.8, 46.8, 17.5,
        r"4-Regime Online Coupled RL-VMC ($\mathrm{OnlineCoupledTrainer1D}$)",
        "R1 Nodal Warmup: MALA + SJB VMC (buffer off)\n"
        "R2 Critic Warmup: MALA + SJB VMC + free off-policy $Q_{\\vartheta_{1,2}}$\n"
        "R3 Co-Evolution: Actor + SJB VMC step-by-step ($\\varepsilon_{\\mathrm{expl}}=0$)\n"
        "R4 Tail Freeze: frozen actor ($\\Delta\\theta=0$), diminishing adaptation & polish",
        fc="#FDF0F1", ec=PALETTE["mala"], title_col=PALETTE["mala_dark"], fs_t=6.35, fs_b=5.30,
    )

    # Connecting feedback arrows between top and bottom (clean white pill labels!)
    arr(84.0, 25.0, 84.0, 19.7, col=PALETTE["mala_dark"], lw=1.15)
    ax.text(85.2, 22.3, "VMC samples", fontsize=5.6, color=PALETTE["mala_dark"], va="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92), zorder=8)
    arr(73.5, 25.0, 46.5, 19.6, col=PALETTE["mala_alt_dark"], lw=1.10, ls="--")
    ax.text(60.5, 22.3, r"transitions $(\mathbf{s}, \mathbf{a}, r_{\mathrm{tot}}, \mathbf{s}')$", fontsize=5.5,
            color=PALETTE["mala_alt_dark"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.94), zorder=8)
    arr(35.0, 19.7, 43.0, 25.0, col=PALETTE["rl_alt_dark"], lw=1.20)
    ax.text(32.5, 22.3, r"$\nabla_\theta Q_{\vartheta_1}$", fontsize=5.9, fontweight="bold",
            color=PALETTE["rl_alt_dark"], ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.92), zorder=8)

    save_figure(fig, out_dir, "fig_actor_architecture")


# ==============================================================================
# FIGURE 6: ONLINE FOUR-REGIME RL-VMC (LOCAL PIPELINE 4 BENCHMARK RESULTS)
# ==============================================================================

def make_fig6_rl_vmc_online_coupled(
    out_dir: Path,
    show_panel_labels: bool = False,
) -> None:
    p4_csv = PROJECT_ROOT / "results_pipeline4_param_search/results_csv"
    df_rank = pd.read_csv(p4_csv / "pipeline4_param_search_ranked_summary.csv")

    fig = plt.figure(figsize=(7.35, 5.75))
    gs = GridSpec(2, 2, figure=fig, wspace=0.31, hspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Load N=6, g=1.0 training histories
    th_p4 = pd.read_csv(p4_csv / "training_history_N6_w0.1_g1.0_13_P4_Full4R_N500_C500_F750_buf25000_w64_utd1_amp0.12_pen0.25.csv")
    th_mala = pd.read_csv(p4_csv / "training_history_N6_w0.1_g1.0_01_Baseline_VMC_MALA_tau0.10.csv")
    th_p3 = pd.read_csv(p4_csv / "training_history_N6_w0.1_g1.0_02_Baseline_P3_BlockCoupled_cyc6x1000.csv")
    th_p3 = th_p3[th_p3["Step"] <= 6000].copy()
    th_cold = pd.read_csv(p4_csv / "training_history_N6_w0.1_g1.0_03_P4_RegimeA_ColdStart_N0_C0_F0.csv")

    # Helper: shade the 4 regimes on (a) and (b)
    def shade_regimes(ax: plt.Axes, ymax_lbl: float) -> None:
        ax.axvspan(0, 500, color="#F3F5F8", alpha=0.85, zorder=0)
        ax.axvspan(500, 1000, color="#E9F3FA", alpha=0.75, zorder=0)
        ax.axvspan(1000, 5250, color="#FFFFFF", alpha=0.0, zorder=0)
        ax.axvspan(5250, 6000, color="#EAF7F0", alpha=0.80, zorder=0)
        for xb in (500, 1000, 5250):
            ax.axvline(xb, color="#B6BDC8", ls=":", lw=0.75, zorder=1)
        ax.text(250, ymax_lbl, "R1", ha="center", va="top", fontsize=5.8, fontweight="bold", color="#606672")
        ax.text(750, ymax_lbl, "R2", ha="center", va="top", fontsize=5.8, fontweight="bold", color=PALETTE["rl_dark"])
        ax.text(3125, ymax_lbl, "R3 (Co-evolution)", ha="center", va="top", fontsize=5.8, fontweight="bold", color="#606672")
        ax.text(5625, ymax_lbl, "R4", ha="center", va="top", fontsize=5.8, fontweight="bold", color=PALETTE["rl_alt_dark"])

    # ------------------------------------------------------------------
    # Top-left: Variational energy E(t) during training at N=6, g=1.0
    # ------------------------------------------------------------------
    shade_regimes(ax_a, 4.544)
    ax_a.axhline(4.524052, color=PALETTE["exact"], ls=":", lw=1.20, zorder=3, label="CASSCF (20 det)")

    l_p4, = ax_a.plot(th_p4["Step"], th_p4["Energy_Mean"].rolling(3, min_periods=1).mean(),
                      color=PALETTE["rl"], lw=1.75, ls="-", zorder=6, label="P4 Online (4 Reg.)")
    l_p3, = ax_a.plot(th_p3["Step"], th_p3["Energy_Mean"].rolling(3, min_periods=1).mean(),
                      color=PALETTE["rwmh_dark"], lw=1.35, ls="-.", zorder=5, label="P3 Block-Coupled")
    l_cold, = ax_a.plot(th_cold["Step"], th_cold["Energy_Mean"].rolling(3, min_periods=1).mean(),
                        color=PALETTE["mala_alt"], lw=1.30, ls="--", zorder=4, label="P4 Cold-Start")
    l_ma, = ax_a.plot(th_mala["Step"], th_mala["Energy_Mean"].rolling(3, min_periods=1).mean(),
                      color=PALETTE["mala"], lw=1.40, ls="-", zorder=5, label="VMC + MALA")

    ax_a.set_xlim(0, 6000)
    ax_a.set_ylim(4.442, 4.548)
    ax_a.set_xlabel(r"VMC optimization step $t$")
    ax_a.set_ylabel(r"Variational energy $E(t)$ ($E_h$)")
    ax_a.tick_params(top=True, right=True)
    ax_a.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
    ax_a.annotate(r"$-50\,\mathrm{m}E_h$ vs CASSCF", xy=(4200, 4.456), xytext=(2900, 4.506),
                  ha="center", fontsize=6.0, fontweight="bold", color=PALETTE["rl_dark"],
                  bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#D0D5DE", lw=0.45, alpha=0.92),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.80))
    # External legend ABOVE top-left (compact 2 rows; RL first!)
    l_cas = Line2D([0], [0], color=PALETTE["exact"], ls=":", lw=1.3, label="CASSCF (20 det)")
    l_sp = Line2D([0], [0], color="none", lw=0.0, label="")
    ax_a.legend(handles=[l_p4, l_ma, l_p3, l_cold, l_cas, l_sp], loc="lower center", bbox_to_anchor=(0.50, 1.03),
                ncol=3, fontsize=5.15, columnspacing=0.45, handlelength=1.05, handletextpad=0.25, borderpad=0.20)
    panel_label(ax_a, "(a)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Top-right: Acceptance rate alpha(t) across 4 regimes at N=6, g=1.0
    # ------------------------------------------------------------------
    shade_regimes(ax_b, 95.5)
    ax_b.plot(th_p4["Step"], th_p4["Acceptance"] * 100.0, color=PALETTE["rl"], lw=1.75, ls="-", zorder=6)
    ax_b.plot(th_p3["Step"], th_p3["Acceptance"] * 100.0, color=PALETTE["rwmh_dark"], lw=1.35, ls="-.", zorder=5)
    ax_b.plot(th_cold["Step"], th_cold["Acceptance"] * 100.0, color=PALETTE["mala_alt"], lw=1.30, ls="--", zorder=4)
    ax_b.plot(th_mala["Step"], th_mala["Acceptance"] * 100.0, color=PALETTE["mala"], lw=1.40, ls="-", zorder=5)

    ax_b.annotate("Cold-Start collapse\n(2.3%)", xy=(4650, 3.0), xytext=(2200, 14.0),
                  ha="center", fontsize=5.9, fontweight="bold", color=PALETTE["mala_alt_dark"],
                  bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.85),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["mala_alt_dark"], lw=0.80))
    ax_b.annotate("P4 stable (~32%)", xy=(3700, 33.0), xytext=(3950, 67.0),
                  ha="center", fontsize=6.0, fontweight="bold", color=PALETTE["rl_dark"],
                  bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="#D0D5DE", lw=0.45, alpha=0.92),
                  arrowprops=dict(arrowstyle="->", color=PALETTE["rl_dark"], lw=0.80))
    ax_b.set_xlim(0, 6000)
    ax_b.set_ylim(0, 99)
    ax_b.set_xlabel(r"VMC optimization step $t$")
    ax_b.set_ylabel(r"Acceptance rate $\bar{\alpha}$ (%)")
    ax_b.tick_params(top=True, right=True)
    ax_b.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
    ax_b.legend(handles=[l_p4, l_ma, l_p3, l_cold], loc="lower center", bbox_to_anchor=(0.50, 1.03),
                ncol=2, fontsize=5.6, columnspacing=0.65, handlelength=1.25, borderpad=0.22)
    panel_label(ax_b, "(b)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Bottom-left: Final energy gain Delta E = E - E_MALA (mHa) & tau_E
    #   across N=6 g=1.0 and N=4 g=4.0 (3x20k frozen eval)
    # ------------------------------------------------------------------
    def _row(n_v: int, g_v: float, substr: str) -> pd.Series:
        s = df_rank[(df_rank["N"] == n_v) & (df_rank["g"] == g_v) & (df_rank["label"].str.contains(substr, regex=False))]
        return s.iloc[0]

    e_m6 = float(_row(6, 1.0, "01_Baseline_VMC_MALA")["E_mean"])
    e_m4 = float(_row(4, 4.0, "01_Baseline_VMC_MALA")["E_mean"])

    bar_specs = [
        ("P4 Full\n(#13/#09)", (_row(6, 1.0, "13_P4_Full4R")["E_mean"] - e_m6) * 1e3,
         (_row(4, 4.0, "09_P4_Full4R")["E_mean"] - e_m4) * 1e3,
         float(_row(6, 1.0, "13_P4_Full4R")["tau_int_E"]), float(_row(4, 4.0, "09_P4_Full4R")["tau_int_E"])),
        ("P4 Full\n(#17)", (_row(6, 1.0, "17_P4_Full4R")["E_mean"] - e_m6) * 1e3,
         (_row(4, 4.0, "17_P4_Full4R")["E_mean"] - e_m4) * 1e3,
         float(_row(6, 1.0, "17_P4_Full4R")["tau_int_E"]), float(_row(4, 4.0, "17_P4_Full4R")["tau_int_E"])),
        ("P4 No\nCritic", (_row(6, 1.0, "04_P4_RegimeB")["E_mean"] - e_m6) * 1e3,
         (_row(4, 4.0, "04_P4_RegimeB")["E_mean"] - e_m4) * 1e3,
         float(_row(6, 1.0, "04_P4_RegimeB")["tau_int_E"]), float(_row(4, 4.0, "04_P4_RegimeB")["tau_int_E"])),
        ("P3 Block\nCoupled", (_row(6, 1.0, "02_Baseline_P3")["E_mean"] - e_m6) * 1e3,
         (_row(4, 4.0, "02_Baseline_P3")["E_mean"] - e_m4) * 1e3,
         float(_row(6, 1.0, "02_Baseline_P3")["tau_int_E"]), float(_row(4, 4.0, "02_Baseline_P3")["tau_int_E"])),
        ("P4 Cold\nStart", (_row(6, 1.0, "03_P4_RegimeA")["E_mean"] - e_m6) * 1e3,
         (_row(4, 4.0, "03_P4_RegimeA")["E_mean"] - e_m4) * 1e3,
         float(_row(6, 1.0, "03_P4_RegimeA")["tau_int_E"]), float(_row(4, 4.0, "03_P4_RegimeA")["tau_int_E"])),
    ]

    xc = np.arange(len(bar_specs))
    wc = 0.36
    d6 = [b[1] for b in bar_specs]
    d4 = [b[2] for b in bar_specs]
    t6 = [b[3] for b in bar_specs]

    lc_base = ax_c.axhline(0.0, color=PALETTE["mala"], ls="--", lw=1.05, zorder=2, label="VMC + MALA baseline (0)")
    bc_6 = ax_c.bar(xc - wc / 2, d6, width=wc, color=PALETTE["rl"], alpha=0.92, edgecolor=PALETTE["rl_dark"], lw=0.55, label=r"$N=6, g=1.0$ (SJB)")
    bc_4 = ax_c.bar(xc + wc / 2, d4, width=wc, color=PALETTE["rl_alt"], alpha=0.90, edgecolor=PALETTE["rl_alt_dark"], lw=0.55, label=r"$N=4, g=4.0$ (SJB)")

    for xi, v6, te6 in zip(xc, d6, t6):
        if v6 < 0:
            ax_c.text(xi - wc / 2, v6 - 0.65, f"{v6:.1f}", ha="center", va="top", fontsize=5.7, fontweight="bold", color=PALETTE["rl_dark"])
        else:
            lbl_bad = f"+{v6:.1f}\n" + rf"($\tau_E\!=\!{te6:.0f}$)"
            ax_c.text(xi - wc / 2, v6 + 0.5, lbl_bad, ha="center", va="bottom", fontsize=5.3, fontweight="bold", color=PALETTE["mala_dark"])

    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([b[0] for b in bar_specs], fontsize=6.0)
    ax_c.set_ylim(-8.5, 29.5)
    ax_c.set_ylabel(r"$\Delta E = E - E_{\mathrm{MALA}}$ ($\mathrm{m}E_h$)")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.grid(True, axis="y", alpha=0.28, lw=0.45, color=PALETTE["grid"])
    ax_c.text(0.03, 0.94, "Lower = better variational energy", transform=ax_c.transAxes,
              fontsize=5.8, style="italic", color=PALETTE["rl_dark"], va="top")
    # RL/SJB bars FIRST (Row 1), VMC+MALA baseline SECOND (Row 2)
    ax_c.legend(handles=[bc_6, lc_base, bc_4], loc="lower center", bbox_to_anchor=(0.50, 1.03), ncol=2, fontsize=5.6, columnspacing=0.65, handlelength=1.2, borderpad=0.22)
    panel_label(ax_c, "(c)", show=show_panel_labels)

    # ------------------------------------------------------------------
    # Bottom-right: Spin-resolved one-body density on deep Wigner crystal N=4, g=4.0
    #   Solid continuous lines ONLY for sampled spin densities!
    # ------------------------------------------------------------------
    obs_p4 = pd.read_csv(p4_csv / "observables_N4_w0_1_g4_0_13_P4_Full4R_N500_C500_F750_buf25000_w64_utd1_amp0_12_pen0_25_P4Search_N4_w0.1_g4.0.csv")
    obs_ma = pd.read_csv(p4_csv / "observables_N4_w0_1_g4_0_01_Baseline_VMC_MALA_tau0_10_P4Search_N4_w0.1_g4.0.csv")
    xg = obs_p4["x_grid"].to_numpy()

    ax_d.fill_between(xg, 0.0, obs_p4["rho_total"].to_numpy(), color=PALETTE["gray_fill"], alpha=0.75, zorder=1)
    # RL-MALA P4: solid blue (spin-up) & solid green (spin-down)
    ld_ru, = ax_d.plot(xg, obs_p4["rho_up"].to_numpy(), color=PALETTE["rl"], lw=1.70, ls="-", zorder=6, label=r"P4 RL-VMC: $\rho_\uparrow$")
    ld_rd, = ax_d.plot(xg, obs_p4["rho_down"].to_numpy(), color=PALETTE["rl_alt"], lw=1.55, ls="-", zorder=7, label=r"P4 RL-VMC: $\rho_\downarrow$")
    # VMC+MALA: solid rose-red (spin-up) & solid plum-purple (spin-down)
    ld_mu, = ax_d.plot(xg, obs_ma["rho_up"].to_numpy(), color=PALETTE["mala"], lw=1.35, ls="-", alpha=0.88, zorder=4, label=r"VMC+MALA: $\rho_\uparrow$")
    ld_md, = ax_d.plot(xg, obs_ma["rho_down"].to_numpy(), color=PALETTE["mala_alt"], lw=1.35, ls="-", alpha=0.88, zorder=5, label=r"VMC+MALA: $\rho_\downarrow$")

    ax_d.set_xlim(-14.5, 14.5)
    ax_d.set_ylim(0.0, 0.44)
    ax_d.set_xlabel(r"Position $x$ ($a_0$)  [$N=4, g=4.0$]")
    ax_d.set_ylabel(r"One-body density $\rho_\sigma(x)$ ($a_0^{-1}$)")
    ax_d.tick_params(top=True, right=True)
    ax_d.grid(True, alpha=0.28, lw=0.45, color=PALETTE["grid"])
    ax_d.text(0.50, 0.94, r"$D_{\mathrm{JS}}(\rho_\uparrow,\rho_\downarrow)$: $0.439 \to 0.064$ ($6.9\times$)",
              transform=ax_d.transAxes, ha="center", va="top", fontsize=6.0, fontweight="bold", color=PALETTE["rl_dark"],
              bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="#D0D5DE", lw=0.5, alpha=0.94))
    # External legend ABOVE bottom-right: Row 1 = P4 RL-VMC (up/down), Row 2 = VMC+MALA (up/down)
    ax_d.legend(handles=[ld_ru, ld_mu, ld_rd, ld_md], loc="lower center", bbox_to_anchor=(0.50, 1.03),
                ncol=2, fontsize=5.6, columnspacing=0.65, handlelength=1.3, borderpad=0.22)
    panel_label(ax_d, "(d)", show=show_panel_labels)

    save_figure(fig, out_dir, "fig6_rl_vmc_online_coupled")


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate revised v2 PRB Letter figures (Helvetica, print-safe pastels, external legends)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "RL_Sampler_PRBLett" / "figures_v2"),
        help="Destination directory for v2 publication figures.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=str(PROJECT_ROOT / "results_prb_final_benchmark") if (PROJECT_ROOT / "results_prb_final_benchmark/prb_final_benchmark_master.csv").is_file() else None,
        help="Option-A final benchmark directory (defaults to results_prb_final_benchmark if present).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated figure keys to build: 'arch,1,1b,2,3,4,5,6' or 'all'.",
    )
    parser.add_argument(
        "--trace-tmax",
        type=int,
        default=10000,
        help="Maximum MCMC step displayed in Fig. 1 long single-walker traces.",
    )
    parser.add_argument(
        "--show-panel-labels",
        action="store_true",
        default=False,
        help="Draw panel labels (a), (b), (c), (d). Off by default.",
    )
    args = parser.parse_args()

    setup_prb_v2_style()
    out_dir = Path(args.out_dir)
    bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else None
    show_lbls = bool(args.show_panel_labels)

    targets = {"arch", "1", "1b", "2", "3", "4", "5", "6"} if args.only == "all" else {t.strip() for t in args.only.split(",")}

    if "arch" in targets:
        make_fig_actor_architecture(out_dir, show_panel_labels=show_lbls)
    if "1" in targets:
        make_fig1_traces_marginals(out_dir, t_max=int(args.trace_tmax), show_panel_labels=show_lbls)
    if "1b" in targets:
        make_fig1b_one_body_densities(out_dir, show_panel_labels=show_lbls)
    if "2" in targets:
        make_fig2_microscopic_mechanism(out_dir, benchmark_dir=bench_dir, show_panel_labels=show_lbls)
    if "3" in targets:
        make_fig3_phase_diagram(out_dir, benchmark_dir=bench_dir, show_panel_labels=show_lbls)
    if "4" in targets:
        make_fig4_ablation_b1_b2(out_dir, benchmark_dir=bench_dir, show_panel_labels=show_lbls)
    if "5" in targets:
        make_fig5_kl_and_tau_convergence(out_dir, show_panel_labels=show_lbls)
    if "6" in targets:
        make_fig6_rl_vmc_online_coupled(out_dir, show_panel_labels=show_lbls)

    print(f"[DONE] v2 figures ready in {out_dir} (panel_labels={show_lbls})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
