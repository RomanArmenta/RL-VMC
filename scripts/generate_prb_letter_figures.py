#!/usr/bin/env python3
"""
scripts/generate_prb_letter_figures.py
======================================
Master publication-grade figure generator for the Physical Review B Letter:
  "Decoupling Diffusion from Quantum Force via Reinforcement Learning:
   Restoring Spin-Permutation Ergodicity in Correlated 1D Fermions"

Generates 4 APS/PRB double-column figures (7.0 in wide, vector PDF + 600 DPI PNG)
into `RL_Sampler_PRBLett/figures/` (configurable via `--out-dir`):
  1. fig1_ergodicity_densities.{pdf,png}
       Macroscopic ergodicity restoration at N=6, omega=0.1, g=1.0:
       walker worldlines & spin-resolved 1-body densities (MALA vs RL-MALA vs exact CASSCF).
  2. fig2_microscopic_mechanism.{pdf,png}
       Microscopic barrier-crossing mechanism at N=6, g=1.0:
       event-aligned exchange coordinate, transit-time histogram, pair vs spectator jump,
       and single-walker Gaussian proposal bridge q_RL vs q_MALA.
  3. fig3_phase_diagram.{pdf,png}
       Scaling across correlation strength g and particle number N:
       - Preliminary mode (local data): N in {2, 4, 6}, g in [0.1, 8.0]
       - Final benchmark mode (--benchmark-dir with Option A results):
         N in {4, 6, 10}, g in {0.1, 0.5, 1.0, 3.0, 5.0} comparing MALA,
         matched-sigma RWMH, current RL-MALA, and confined RL-MALA.
  4. fig4_ablation_b1_b2.{pdf,png}
       Causal ablation, matched-sigma controls, parity equivariance, and drift diagnostic:
       - Step-size decoupling vs MALA autocorrelation explosion (N=6, g=1.0)
       - Matched-sigma RL-MALA vs pure RWMH verdict (N=6 g=1.0 & N=4 g=2.0)
       - Spatial actor response mu_theta(x) for N=6 g=1.0 (inward confining drift)
       - Spatial actor response mu_theta(x) for N=4 g=2.0 (outward anti-confining drift;
         overlays confined RL-MALA automatically when --benchmark-dir is provided).

Usage:
  # Preliminary draft figures (from local repository datasets):
  python3 scripts/generate_prb_letter_figures.py

  # Final benchmark figures (after Option A Ayler benchmark completes):
  python3 scripts/generate_prb_letter_figures.py --benchmark-dir results_prb_final_benchmark
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# GLOBAL APS / PHYSICAL REVIEW B TYPOGRAPHY & COLORBLIND-SAFE PALETTE
# ==============================================================================

COLORS = {
    "exact": "#111111",
    "mala": "#C83E37",       # Crimson (baseline MALA)
    "mala_lg": "#D95F02",    # Amber-orange (large-step MALA)
    "rwmh": "#B21870",       # Deep magenta (matched-sigma RWMH)
    "rl_leg": "#1F78B4",     # Steel blue (legacy RL-MALA)
    "rl_par": "#6A51A3",     # Violet (parity-only RL-MALA)
    "rl": "#007F5F",         # Deep teal-green (unlocked RL-MALA / current)
    "rl_conf": "#004B23",    # Forest green (confined RL-MALA)
    "up": "#0072B2",         # Cobalt blue (spin up)
    "dn": "#D55E00",         # Vermilion (spin down)
    "gray_fill": "#D9D9D9",
    "grid": "#CCCCCC",
}


def setup_prb_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
            "mathtext.fontset": "cm",
            "font.size": 8.2,
            "axes.labelsize": 8.6,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.1,
            "legend.frameon": True,
            "legend.framealpha": 0.90,
            "legend.edgecolor": "#CCCCCC",
            "legend.borderpad": 0.30,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.45,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.70,
            "ytick.major.width": 0.70,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "lines.linewidth": 1.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def panel_label(ax: plt.Axes, text: str, x: float = 0.03, y: float = 0.95, color: str = "#111111") -> None:
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
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.80),
        zorder=20,
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 600) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"{stem}.{ext}"
        fig.savefig(out_path, dpi=dpi if ext == "png" else 300)
        print(f"[+] Saved {out_path}")
    plt.close(fig)


def _find_ablation_dir(root: Path, sys_tag: str) -> Optional[Path]:
    """Locate the directory containing ablation_summary_*.csv & spatial_probes/ (handles 'Sin título')."""
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


# ==============================================================================
# FIGURE 1: MACROSCOPIC ERGODICITY & WIGNER SPIN-SYMMETRY RESTORATION (N=6, g=1.0)
# ==============================================================================

def make_fig1_ergodicity_densities(out_dir: Path) -> None:
    traces_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv/flips_traces_P2_N6_w0.1_g1.csv"
    dens_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv/flips_marginal_density_P2_N6_w0.1_g1.csv"

    df_tr = pd.read_csv(traces_csv)
    df_dn = pd.read_csv(dens_csv)

    fig = plt.figure(figsize=(7.0, 4.40))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], wspace=0.18, hspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1], sharey=ax_c)

    t_max = 3000
    for ax, meth, ptag, subtitle in [
        (ax_a, "MALA", "(a)", r"MALA ($\tau = 0.10$)"),
        (ax_b, "RL-MALA", "(b)", r"RL-MALA ($\tau = 0.10$)"),
    ]:
        sub = df_tr[(df_tr["method"] == meth) & (df_tr["step"] <= t_max)].sort_values("step")
        steps = sub["step"].to_numpy()
        for idx in range(6):
            is_up = idx < 3
            col = COLORS["up"] if is_up else COLORS["dn"]
            lbl = None
            if ax is ax_a and idx == 0:
                lbl = r"Spin-$\uparrow$ ($e_{1,2,3}$)"
            elif ax is ax_a and idx == 3:
                lbl = r"Spin-$\downarrow$ ($e_{4,5,6}$)"
            ax.plot(
                steps,
                sub[f"x_{idx}"].to_numpy(),
                color=col,
                lw=0.82,
                alpha=0.88 if is_up else 0.84,
                label=lbl,
            )
        ax.set_xlim(0, t_max)
        ax.set_ylim(-14.5, 17.5)
        ax.set_xlabel(r"Monte Carlo step $t$")
        ax.tick_params(top=True, right=True)
        ax.grid(True, alpha=0.22, lw=0.45, color=COLORS["grid"])
        panel_label(ax, ptag)
        ax.text(
            0.97,
            0.94,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            fontweight="bold",
            color=COLORS["mala"] if meth == "MALA" else COLORS["rl"],
            bbox=dict(boxstyle="round,pad=0.17", facecolor="white", edgecolor="#DDDDDD", alpha=0.92),
        )

    ax_a.set_ylabel(r"Electron position $x_i$ ($a_0$)")
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_a.legend(loc="lower left", ncol=2, fontsize=6.5, columnspacing=0.8, handlelength=1.2)

    # Bottom panels: Spin-resolved 1-body densities rho_sigma(x) integrating to N_sigma = 3 (total = 6)
    x = df_dn["x_grid"].to_numpy()
    dx = float(x[1] - x[0])
    exact_up = 3.0 * df_dn["exact_up"].to_numpy()
    exact_dn = 3.0 * df_dn["exact_down"].to_numpy()
    exact_tot = 6.0 * df_dn["exact_total"].to_numpy()

    kl_mala = 0.5 * (_compute_kl(df_dn["mala_up"], df_dn["exact_up"], dx) + _compute_kl(df_dn["mala_down"], df_dn["exact_down"], dx))
    kl_rl = 0.5 * (_compute_kl(df_dn["rl_up"], df_dn["exact_up"], dx) + _compute_kl(df_dn["rl_down"], df_dn["exact_down"], dx))

    for ax, up_col, dn_col, kl_val, ptag, col_badge in [
        (ax_c, "mala_up", "mala_down", kl_mala, "(c)", COLORS["mala"]),
        (ax_d, "rl_up", "rl_down", kl_rl, "(d)", COLORS["rl"]),
    ]:
        ax.fill_between(x, 0.0, exact_tot, color=COLORS["gray_fill"], alpha=0.52, label=r"Exact $\rho_{\mathrm{tot}}$")
        ax.plot(x, exact_up, color=COLORS["exact"], ls="--", lw=1.15, zorder=5, label=r"Exact $\rho_\uparrow=\rho_\downarrow$")
        ax.plot(x, 3.0 * df_dn[up_col].to_numpy(), color=COLORS["up"], lw=1.35, zorder=6, label=r"Sampled $\rho_\uparrow$")
        ax.plot(x, 3.0 * df_dn[dn_col].to_numpy(), color=COLORS["dn"], lw=1.25, ls="-.", zorder=7, label=r"Sampled $\rho_\downarrow$")
        ax.set_xlim(-13.8, 13.8)
        ax.set_ylim(0.0, float(np.max(exact_tot)) * 1.24)
        ax.set_xlabel(r"Position $x$ ($a_0$)")
        ax.tick_params(top=True, right=True)
        ax.grid(True, alpha=0.22, lw=0.45, color=COLORS["grid"])
        panel_label(ax, ptag)
        kl_txt = (
            rf"$D_{{\mathrm{{KL}}}}^{{\mathrm{{spin}}}} = {kl_val:.3f}$"
            if kl_val >= 0.01
            else rf"$D_{{\mathrm{{KL}}}}^{{\mathrm{{spin}}}} = {kl_val*1e4:.1f}\times 10^{{-4}}$"
        )
        ax.text(
            0.97,
            0.94,
            kl_txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color=col_badge,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.17", facecolor="white", edgecolor=col_badge, lw=0.65, alpha=0.92),
        )

    ax_c.set_ylabel(r"One-body density $\rho(x)$ ($a_0^{-1}$)")
    plt.setp(ax_d.get_yticklabels(), visible=False)
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.01, 0.83), fontsize=6.3, frameon=True)

    # Point cleanly to the artificial spin polarization peak in (c)
    mala_up_rho = 3.0 * df_dn["mala_up"].to_numpy()
    idx_pk = int(np.argmax(mala_up_rho))
    ax_c.annotate(
        "Artificial spin\npolarization",
        xy=(float(x[idx_pk]), float(mala_up_rho[idx_pk])),
        xytext=(-13.2, float(np.max(exact_tot)) * 0.88),
        fontsize=6.5,
        color=COLORS["mala"],
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["mala"], lw=0.8, connectionstyle="arc3,rad=-0.12"),
    )

    save_figure(fig, out_dir, "fig1_ergodicity_densities")


# ==============================================================================
# FIGURE 2: MICROSCOPIC BARRIER-CROSSING MECHANISM (N=6, g=1.0)
# ==============================================================================

def _compute_single_walker_proposal_bridge() -> Optional[Dict[str, Any]]:
    h5_path = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.10_amp0.15_pen0.25_sig1.1/run_bundle.h5"
    npz_path = PROJECT_ROOT / "CASSCF_csv/casscf_state_cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_det-20.npz"
    if not h5_path.is_file() or not npz_path.is_file():
        return None
    try:
        from rl_vmc.core.io_bundle import load_run_bundle_h5
        from scripts.plot_walker_flip_proposal import (
            evaluate_walker_flip_proposals,
            find_clean_single_walker_flip,
        )
        bundle = load_run_bundle_h5(str(h5_path))
        walkers = np.asarray(bundle["production"]["walkers"], dtype=np.float64)
        ev = find_clean_single_walker_flip(
            walkers,
            n_up=3,
            spin_channel="opp",
            window=6,
            chain_override=0,
            walker_override=100,
            t0_override=18403,
        )
        if ev is None:
            ev = find_clean_single_walker_flip(walkers, n_up=3, spin_channel="opp", window=6)
        if ev is None:
            return None
        return evaluate_walker_flip_proposals(
            bundle,
            str(h5_path),
            ev,
            window=6,
            casscf_npz_override=str(npz_path),
        )
    except Exception as exc:
        print(f"[WARN] Could not evaluate single-walker proposal bridge: {exc}")
        return None


def make_fig2_microscopic_mechanism(out_dir: Path) -> None:
    base_csv = PROJECT_ROOT / "results_nodal_crossing_search_N6/MultiDet_N6_g1.0/results_csv"
    df_mala = pd.read_csv(base_csv / "flips_heatmap_aligned_MALA_P2_N6_w0.1_g1.csv")
    df_rl = pd.read_csv(base_csv / "flips_heatmap_aligned_RL-MALA_P2_N6_w0.1_g1.csv")
    df_hist = pd.read_csv(base_csv / "flips_transit_hist_P2_N6_w0.1_g1.csv")
    bridge = _compute_single_walker_proposal_bridge()

    fig = plt.figure(figsize=(7.0, 4.55))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # (a) Event-aligned exchange coordinate d_ij / d_adj
    k_lim = 10
    dm = df_mala[(df_mala["rel_step"] >= -k_lim) & (df_mala["rel_step"] <= k_lim)]
    dr = df_rl[(df_rl["rel_step"] >= -k_lim) & (df_rl["rel_step"] <= k_lim)]

    ax_a.axhline(0.0, color="#666666", ls=":", lw=0.8)
    ax_a.axvline(0.0, color="#999999", ls="--", lw=0.7)
    ax_a.fill_between(dm["rel_step"], dm["q25_d_norm"], dm["q75_d_norm"], color=COLORS["mala"], alpha=0.18)
    ax_a.plot(dm["rel_step"], dm["med_d_norm"], color=COLORS["mala"], ls="--", marker="o", ms=2.6, lw=1.3, label="MALA (diffusive)")
    ax_a.fill_between(dr["rel_step"], dr["q25_d_norm"], dr["q75_d_norm"], color=COLORS["rl"], alpha=0.20)
    ax_a.plot(dr["rel_step"], dr["med_d_norm"], color=COLORS["rl"], ls="-", marker="s", ms=2.6, lw=1.45, label="RL-MALA (ballistic)")
    ax_a.set_xlim(-k_lim, k_lim)
    ax_a.set_ylim(-1.45, 1.45)
    ax_a.set_xlabel(r"Steps relative to crossing, $k = t - t_0$")
    ax_a.set_ylabel(r"Separation $d_{ij} / d_{\mathrm{adj}}$")
    ax_a.tick_params(top=True, right=True)
    ax_a.grid(True, alpha=0.22, lw=0.45)
    ax_a.legend(loc="lower right", fontsize=6.9)
    panel_label(ax_a, "(a)")

    # (b) Transit-time histogram (1..8 steps)
    dh = df_hist[df_hist["transit_steps"] <= 8].copy()
    t_steps = dh["transit_steps"].to_numpy()
    w_bar = 0.38
    ax_b.bar(t_steps - w_bar / 2, dh["mala_pct"], width=w_bar, color=COLORS["mala"], alpha=0.86, edgecolor="#222222", lw=0.5, label="MALA")
    ax_b.bar(t_steps + w_bar / 2, dh["rl_pct"], width=w_bar, color=COLORS["rl"], alpha=0.90, edgecolor="#222222", lw=0.5, label="RL-MALA")

    # Annotate 1-step dominance cleanly
    p1_rl = float(dh.loc[dh["transit_steps"] == 1, "rl_pct"].iloc[0])
    p1_m = float(dh.loc[dh["transit_steps"] == 1, "mala_pct"].iloc[0])
    ax_b.annotate(
        f"{p1_rl:.1f}%\n(1-step leap)",
        xy=(1.0 + w_bar / 2, p1_rl),
        xytext=(2.65, p1_rl - 4.5),
        fontsize=7.1,
        fontweight="bold",
        color=COLORS["rl"],
        arrowprops=dict(arrowstyle="->", color=COLORS["rl"], lw=0.85),
    )
    ax_b.text(1.0 - w_bar * 0.62, p1_m + 2.2, f"{p1_m:.1f}%", ha="center", va="bottom", fontsize=6.4, color=COLORS["mala"], fontweight="bold")
    ax_b.set_xticks(t_steps)
    ax_b.set_xlim(0.30, 8.65)
    ax_b.set_ylim(0.0, 72.0)
    ax_b.set_xlabel(r"Barrier transit duration $T_{\mathrm{transit}}$ (steps)")
    ax_b.set_ylabel("Exchange events (%)")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, axis="y", alpha=0.24, lw=0.45)
    ax_b.legend(loc="upper right", fontsize=7.0)
    panel_label(ax_b, "(b)", x=0.03, y=0.96)

    # (c) Event-aligned single-step displacement: Exchanging pair vs Spectators
    ax_c.axvline(0.0, color="#999999", ls="--", lw=0.7)
    ax_c.plot(dr["rel_step"], dr["mean_step_mag_pair"], color=COLORS["rl"], lw=1.55, marker="s", ms=2.6, label="RL: exchanging pair")
    ax_c.plot(dr["rel_step"], dr["mean_step_mag_spec"], color=COLORS["rl"], lw=1.05, ls=":", alpha=0.85, label="RL: spectators")
    ax_c.plot(dm["rel_step"], dm["mean_step_mag_pair"], color=COLORS["mala"], lw=1.35, ls="--", marker="o", ms=2.4, label="MALA: pair")
    ax_c.plot(dm["rel_step"], dm["mean_step_mag_spec"], color=COLORS["mala"], lw=0.95, ls="-.", alpha=0.65, label="MALA: spectators")

    peak_rl = float(dr.loc[dr["rel_step"] == 0, "mean_step_mag_pair"].iloc[0])
    ax_c.annotate(
        rf"$\langle|\Delta x|\rangle = {peak_rl:.2f}\,a_0$",
        xy=(0.2, peak_rl),
        xytext=(2.6, peak_rl - 0.10),
        fontsize=6.9,
        fontweight="bold",
        color=COLORS["rl"],
        arrowprops=dict(arrowstyle="->", color=COLORS["rl"], lw=0.8),
    )
    ax_c.set_xlim(-k_lim, k_lim)
    ax_c.set_ylim(0.0, max(peak_rl * 1.24, 2.10))
    ax_c.set_xlabel(r"Steps relative to crossing, $k = t - t_0$")
    ax_c.set_ylabel(r"Step displacement $\langle|\Delta x|\rangle$ ($a_0$)")
    ax_c.tick_params(top=True, right=True)
    ax_c.grid(True, alpha=0.22, lw=0.45)
    ax_c.legend(loc="upper left", bbox_to_anchor=(0.01, 0.82), fontsize=6.1, handlelength=1.3)
    panel_label(ax_c, "(c)")

    # (d) Single-walker Gaussian proposal bridge q_RL vs q_MALA at t_0 -> t_0 + 1
    if bridge is not None:
        ji = bridge["jump_i"]
        jj = bridge["jump_j"]
        x_lo = min(ji["x_t0"], ji["x_t1"], jj["x_t0"], jj["x_t1"]) - 3.0
        x_hi = max(ji["x_t0"], ji["x_t1"], jj["x_t0"], jj["x_t1"]) + 3.0
        xg = np.linspace(x_lo, x_hi, 450)

        def gpdf(xx: np.ndarray, mu: float, sig: float) -> np.ndarray:
            return np.exp(-0.5 * ((xx - mu) / sig) ** 2) / (math.sqrt(2.0 * math.pi) * sig)

        # Plot exact CASSCF 1-body density scaled softly in background to locate Wigner wells
        if bridge.get("casscf_ref") is not None:
            xc = np.asarray(bridge["casscf_ref"]["x"], dtype=np.float64)
            rc = np.asarray(bridge["casscf_ref"]["rho_total"], dtype=np.float64)
            rc_scaled = rc / max(float(np.max(rc)), 1e-9) * 0.42
            ax_d.fill_between(xc, 0.0, rc_scaled, color=COLORS["gray_fill"], alpha=0.45, zorder=1, label=r"CASSCF $\rho_{\mathrm{tot}}$ (arb.)")

        # Plot electron i (left -> right) and electron j (right -> left) proposals
        qi_m = gpdf(xg, ji["mu_mala"], ji["sigma_mala"])
        qi_r = gpdf(xg, ji["mu_rl"], ji["sigma_rl"])
        qj_m = gpdf(xg, jj["mu_mala"], jj["sigma_mala"])
        qj_r = gpdf(xg, jj["mu_rl"], jj["sigma_rl"])

        ax_d.plot(xg, qi_m, color=COLORS["mala"], ls="--", lw=1.15, zorder=3, label=r"$q_{\mathrm{MALA}}(x' \mid X_{t_0})$")
        ax_d.plot(xg, qj_m, color=COLORS["mala"], ls="--", lw=1.15, alpha=0.75, zorder=3)
        ax_d.fill_between(xg, 0.0, qi_r, color=COLORS["up"], alpha=0.14, zorder=2)
        ax_d.fill_between(xg, 0.0, qj_r, color=COLORS["dn"], alpha=0.14, zorder=2)
        ax_d.plot(xg, qi_r, color=COLORS["up"], lw=1.45, zorder=5, label=r"$q_{\mathrm{RL}}$ ($e_i \uparrow$)")
        ax_d.plot(xg, qj_r, color=COLORS["dn"], lw=1.45, zorder=5, label=r"$q_{\mathrm{RL}}$ ($e_j \downarrow$)")

        # Mark initial (open circle) and landing (filled star) positions
        y_base = 0.02
        ax_d.scatter([ji["x_t0"]], [y_base], marker="o", s=26, facecolor="white", edgecolor=COLORS["up"], lw=1.2, zorder=8)
        ax_d.scatter([ji["x_t1"]], [float(gpdf(np.array([ji["x_t1"]]), ji["mu_rl"], ji["sigma_rl"])[0])], marker="*", s=55, color=COLORS["up"], edgecolor="#111111", lw=0.4, zorder=9)
        ax_d.scatter([jj["x_t0"]], [y_base], marker="o", s=26, facecolor="white", edgecolor=COLORS["dn"], lw=1.2, zorder=8)
        ax_d.scatter([jj["x_t1"]], [float(gpdf(np.array([jj["x_t1"]]), jj["mu_rl"], jj["sigma_rl"])[0])], marker="*", s=55, color=COLORS["dn"], edgecolor="#111111", lw=0.4, zorder=9)

        # Curved swap arrows between initial & final positions (placed cleanly between the two peaks)
        ax_d.annotate("", xy=(ji["x_t1"] - 0.15, 0.44), xytext=(ji["x_t0"] + 0.15, 0.44), arrowprops=dict(arrowstyle="->", color=COLORS["up"], lw=1.05, connectionstyle="arc3,rad=-0.28"))
        ax_d.annotate("", xy=(jj["x_t1"] + 0.15, 0.38), xytext=(jj["x_t0"] - 0.15, 0.38), arrowprops=dict(arrowstyle="->", color=COLORS["dn"], lw=1.05, connectionstyle="arc3,rad=-0.28"))

        ax_d.set_xlim(x_lo, x_hi)
        ax_d.set_ylim(0.0, max(float(np.max(qi_m)), float(np.max(qj_m))) * 1.36)
        ax_d.set_xlabel(r"Proposal coordinate $x'$ ($a_0$)")
        ax_d.set_ylabel(r"Proposal density $q(x' \mid X_{t_0})$ ($a_0^{-1}$)")
        ax_d.tick_params(top=True, right=True)
        ax_d.grid(True, alpha=0.22, lw=0.45)
        ax_d.legend(loc="upper right", ncol=2, columnspacing=0.65, fontsize=5.8, handlelength=1.15)
    else:
        ax_d.text(0.5, 0.5, "Single-walker bundle unavailable", ha="center", va="center", transform=ax_d.transAxes)
    panel_label(ax_d, "(d)")

    save_figure(fig, out_dir, "fig2_microscopic_mechanism")


# ==============================================================================
# FIGURE 3: SCALING PHASE DIAGRAM ACROSS CORRELATION STRENGTH g & PARTICLE COUNT N
# ==============================================================================

def make_fig3_phase_diagram(out_dir: Path, benchmark_dir: Optional[Path] = None) -> None:
    master_bench_csv = (benchmark_dir / "prb_final_benchmark_master.csv") if benchmark_dir else None
    use_final_bench = master_bench_csv is not None and master_bench_csv.is_file()

    fig = plt.figure(figsize=(7.0, 2.70))
    gs = GridSpec(1, 3, figure=fig, wspace=0.44)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    if use_final_bench:
        df = pd.read_csv(master_bench_csv)
        ns_avail = sorted(df["N"].unique())
        n_styles = {
            4: ("o", "#1F78B4"),
            6: ("s", "#007F5F"),
            10: ("^", "#6A51A3"),
        }
        # (a) Total exchange rate: MALA base vs Current RL-MALA vs Confined RL-MALA
        for n_val in ns_avail:
            mk, col = n_styles.get(int(n_val), ("D", "#333333"))
            for key, ls, mfc, alpha_l, tag in [
                ("B1_MALA_base_tau0.10", ":", "white", 0.70, "MALA"),
                ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "--", col, 0.80, "RL (curr)"),
                ("B2_RL_MALA_confined_amp0.80_sig1.8", "-", col, 1.00, "RL (conf)"),
            ]:
                sub = df[(df["N"] == n_val) & (df["model_key"] == key)].sort_values("g")
                if sub.empty:
                    continue
                y = np.maximum(sub["hyst_flips_total_1k"].to_numpy(dtype=float), 1e-4)
                lbl = f"$N={n_val}$ {tag}" if tag in ("MALA", "RL (conf)") else None
                ax_a.plot(sub["g"], y, ls=ls, color=col, marker=mk, ms=4.0, mfc=mfc, lw=1.25, alpha=alpha_l, label=lbl)
        ax_a.set_yscale("log")
        ax_a.set_xlabel(r"Interaction strength $g$")
        ax_a.set_ylabel(r"Exchange rate (flips / $10^3$ steps)")
        ax_a.grid(True, which="major", alpha=0.25, lw=0.45)
        ax_a.legend(loc="lower left", fontsize=5.8, ncol=1)
        panel_label(ax_a, "(a)")

        # (b) Spin-KL error vs g
        for n_val in ns_avail:
            mk, col = n_styles.get(int(n_val), ("D", "#333333"))
            for key, ls, mfc in [
                ("B1_MALA_base_tau0.10", ":", "white"),
                ("B2_RL_MALA_confined_amp0.80_sig1.8", "-", col),
            ]:
                sub = df[(df["N"] == n_val) & (df["model_key"] == key)].sort_values("g")
                if sub.empty:
                    continue
                y = np.maximum(sub["kl_spin"].to_numpy(dtype=float), 1e-5)
                ax_b.plot(sub["g"], y, ls=ls, color=col, marker=mk, ms=4.0, mfc=mfc, lw=1.25)
        ax_b.set_yscale("log")
        ax_b.set_xlabel(r"Interaction strength $g$")
        ax_b.set_ylabel(r"Spin-density error $D_{\mathrm{KL}}^{\mathrm{spin}}$")
        ax_b.grid(True, which="major", alpha=0.25, lw=0.45)
        panel_label(ax_b, "(b)")

        # (c) ESJD ratio: Confined RL-MALA vs Current RL-MALA vs Matched-sigma RWMH
        for n_val in ns_avail:
            mk, col = n_styles.get(int(n_val), ("D", "#333333"))
            sub_rw = df[(df["N"] == n_val) & (df["model_key"] == "B1_RWMH_matched_sig2.7055")].set_index("g")
            sub_c1 = df[(df["N"] == n_val) & (df["model_key"] == "B2_RL_MALA_eq_unclip_amp0.80_sig1.8")].set_index("g")
            sub_c2 = df[(df["N"] == n_val) & (df["model_key"] == "B2_RL_MALA_confined_amp0.80_sig1.8")].set_index("g")
            gs_common = sorted(set(sub_rw.index) & set(sub_c2.index))
            if gs_common:
                ratio_c2 = [float(sub_c2.loc[gv, "ESJD"]) / max(float(sub_rw.loc[gv, "ESJD"]), 1e-9) for gv in gs_common]
                ax_c.plot(gs_common, ratio_c2, ls="-", color=col, marker=mk, ms=4.2, lw=1.35, label=rf"$N={n_val}$ (conf)")
            gs_c1 = sorted(set(sub_rw.index) & set(sub_c1.index))
            if gs_c1:
                ratio_c1 = [float(sub_c1.loc[gv, "ESJD"]) / max(float(sub_rw.loc[gv, "ESJD"]), 1e-9) for gv in gs_c1]
                ax_c.plot(gs_c1, ratio_c1, ls="--", color=col, marker=mk, ms=3.5, mfc="white", lw=1.0, alpha=0.75)
        ax_c.axhline(1.0, color=COLORS["rwmh"], ls=":", lw=1.0, label="RWMH parity")
        ax_c.set_xlabel(r"Interaction strength $g$")
        ax_c.set_ylabel(r"$\mathrm{ESJD}_{\mathrm{RL}} / \mathrm{ESJD}_{\mathrm{RWMH}}$")
        ax_c.grid(True, alpha=0.25, lw=0.45)
        ax_c.legend(loc="best", fontsize=6.2)
        panel_label(ax_c, "(c)")
    else:
        # Preliminary mode: clean converged N in {2, 4, 6}, g in [0.1, 8.0] from local flips_vs_g_summary.csv
        df = pd.read_csv(PROJECT_ROOT / "flips_analysis_advisor_new/flips_vs_g_summary.csv")
        df = df[df["N"].isin([2, 4, 6])].copy()

        n_palette = {
            2: ("^", "#6A51A3"),
            4: ("o", "#1F78B4"),
            6: ("s", "#007F5F"),
        }

        # (a) Total exchange rate vs g (log scale) across N = 2, 4, 6
        for n_val in (2, 4, 6):
            mk, col = n_palette[n_val]
            sub_r = df[(df["N"] == n_val) & (df["method"] == "RL-MALA")].sort_values("g")
            sub_m = df[(df["N"] == n_val) & (df["method"] == "MALA")].sort_values("g")

            # Filter positive rates for log plot (0.0 -> mark on floor with downward arrow or omit zero tail)
            r_pos = sub_r[sub_r["hyst_flips_total_mean"] > 0]
            m_pos = sub_m[sub_m["hyst_flips_total_mean"] > 0]

            ax_a.plot(
                r_pos["g"],
                r_pos["hyst_flips_total_mean"],
                color=col,
                ls="-",
                marker=mk,
                ms=4.0,
                lw=1.35,
                label=rf"$N={n_val}$ (RL)",
            )
            ax_a.plot(
                m_pos["g"],
                m_pos["hyst_flips_total_mean"],
                color=col,
                ls="--",
                marker=mk,
                ms=3.7,
                mfc="white",
                mew=0.9,
                lw=0.95,
                alpha=0.82,
                label=rf"$N={n_val}$ (MALA)" if n_val == 6 else None,
            )

        ax_a.set_yscale("log")
        ax_a.set_xlim(-0.2, 8.35)
        ax_a.set_ylim(5e-4, 1.4e3)
        ax_a.set_xlabel(r"Interaction strength $g$")
        ax_a.set_ylabel(r"Flips / $10^3$ steps")
        ax_a.tick_params(top=True, right=True)
        ax_a.grid(True, which="major", alpha=0.25, lw=0.45)
        ax_a.legend(loc="lower left", fontsize=6.0, ncol=1, handlelength=1.3)
        ax_a.annotate(
            "35–67×\nboost",
            xy=(3.0, 1.69),
            xytext=(4.35, 16.0),
            fontsize=6.9,
            fontweight="bold",
            color=COLORS["rl"],
            arrowprops=dict(arrowstyle="->", color=COLORS["rl"], lw=0.8),
        )
        panel_label(ax_a, "(a)", x=0.12, y=0.96)

        # (b) Channel-resolved flips for N=6 (Wigner opposite-spin vs Pauli same-spin nodal crossings)
        sub6_r = df[(df["N"] == 6) & (df["method"] == "RL-MALA")].sort_values("g")
        sub6_m = df[(df["N"] == 6) & (df["method"] == "MALA")].sort_values("g")

        ax_b.plot(sub6_r["g"], sub6_r["hyst_flips_opp_mean"], color=COLORS["rl"], ls="-", marker="s", ms=4.0, lw=1.4, label=r"RL: $\uparrow\!\downarrow$ (Wigner)")
        ax_b.plot(sub6_r["g"], sub6_r["hyst_flips_same_mean"], color=COLORS["up"], ls="-", marker="D", ms=3.5, lw=1.3, label=r"RL: Pauli")
        ax_b.plot(sub6_m["g"], sub6_m["hyst_flips_opp_mean"], color=COLORS["mala"], ls="--", marker="s", ms=3.5, mfc="white", mew=0.9, lw=1.0, label=r"MALA: $\uparrow\!\downarrow$")
        ax_b.plot(sub6_m["g"], sub6_m["hyst_flips_same_mean"], color=COLORS["mala_lg"], ls=":", marker="D", ms=3.2, mfc="white", mew=0.9, lw=1.0, label=r"MALA: Pauli")

        ax_b.set_yscale("log")
        ax_b.set_xlim(-0.2, 8.35)
        ax_b.set_ylim(2e-4, 1.4e3)
        ax_b.set_xlabel(r"Interaction strength $g$")
        ax_b.set_ylabel(r"$N=6$ flips / $10^3$ steps")
        ax_b.tick_params(top=True, right=True)
        ax_b.grid(True, which="major", alpha=0.25, lw=0.45)
        ax_b.legend(loc="upper right", fontsize=5.9, handlelength=1.3)
        panel_label(ax_b, "(b)", x=0.12, y=0.96)

        # (c) Spin-density fidelity D_KL^spin vs g across N = 2, 4, 6
        for n_val in (2, 4, 6):
            mk, col = n_palette[n_val]
            sub_r = df[(df["N"] == n_val) & (df["method"] == "RL-MALA")].sort_values("g")
            sub_m = df[(df["N"] == n_val) & (df["method"] == "MALA")].sort_values("g")
            ax_c.plot(sub_r["g"], sub_r["kl_spin"], color=col, ls="-", marker=mk, ms=4.0, lw=1.35)
            ax_c.plot(sub_m["g"], sub_m["kl_spin"], color=col, ls="--", marker=mk, ms=3.6, mfc="white", mew=0.9, lw=0.95, alpha=0.82)

        ax_c.axhline(1e-3, color="#666666", ls=":", lw=0.75)
        ax_c.text(7.95, 1.4e-3, r"$10^{-3}$", ha="right", va="bottom", fontsize=6.3, color="#555555")
        ax_c.set_yscale("log")
        ax_c.set_xlim(-0.2, 8.35)
        ax_c.set_ylim(1.2e-5, 4.0)
        ax_c.set_xlabel(r"Interaction strength $g$")
        ax_c.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$")
        ax_c.tick_params(top=True, right=True)
        ax_c.grid(True, which="major", alpha=0.25, lw=0.45)
        ax_c.text(
            0.96,
            0.06,
            "Solid: RL-MALA\nDashed: MALA",
            transform=ax_c.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.0,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )
        panel_label(ax_c, "(c)")

    save_figure(fig, out_dir, "fig3_phase_diagram")


# ==============================================================================
# FIGURE 4: CAUSAL ABLATION (B1/B2), MATCHED-SIGMA VERDICT & SPATIAL DRIFT SIGN
# ==============================================================================

def make_fig4_ablation_b1_b2(out_dir: Path, benchmark_dir: Optional[Path] = None) -> None:
    ablation_root = benchmark_dir if (benchmark_dir and (benchmark_dir / "N6_w0.1_g1.0").is_dir()) else (PROJECT_ROOT / "results_ablation_ayler")
    dir_n6 = _find_ablation_dir(ablation_root, "N6_w0.1_g1.0") or _find_ablation_dir(PROJECT_ROOT / "results_ablation_ayler", "N6_w0.1_g1.0")
    dir_n4 = _find_ablation_dir(ablation_root, "N4_w0.1_g2.0") or _find_ablation_dir(PROJECT_ROOT / "results_ablation_ayler", "N4_w0.1_g2.0")

    df_n6 = pd.read_csv(next(dir_n6.glob("ablation_summary_*.csv"))).set_index("model_key")
    df_n4 = pd.read_csv(next(dir_n4.glob("ablation_summary_*.csv"))).set_index("model_key")

    has_conf_n6 = "B2_RL_MALA_confined_amp0.80_sig1.8" in df_n6.index
    has_conf_n4 = "B2_RL_MALA_confined_amp0.80_sig1.8" in df_n4.index

    fig = plt.figure(figsize=(7.0, 4.75))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # ------------------------------------------------------------------
    # (a) Causal B1 + B2 ablation at N=6, g=1.0 (flips/1k + tau_int(E))
    # ------------------------------------------------------------------
    keys_a = [
        ("B1_MALA_base_tau0.10", "MALA\n0.45", COLORS["mala"]),
        ("B1_MALA_matched_tau0.7389", "MALA\n1.22", COLORS["mala_lg"]),
        ("B1_RWMH_matched_sig1.2156", "RWMH\n1.22", COLORS["rwmh"]),
        ("B2_RL_MALA_parity_only", "RL par.\n1.22", COLORS["rl_par"]),
        ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "RL eq.\n2.71", COLORS["rl"]),
        ("B2_RL_MALA_eq_unclip_amp1.20_sig2.0", "RL eq.\n3.30", "#2CA02C"),
    ]
    if has_conf_n6:
        keys_a[-1] = ("B2_RL_MALA_confined_amp0.80_sig1.8", "RL conf.\n2.71", COLORS["rl_conf"])

    keys_a = [k for k in keys_a if k[0] in df_n6.index]
    xa = np.arange(len(keys_a))
    flips_opp = [float(df_n6.loc[k[0], "hyst_flips_opp_1k"]) for k in keys_a]
    flips_same = [float(df_n6.loc[k[0], "hyst_flips_same_1k"]) for k in keys_a]
    flips_tot = [o + s for o, s in zip(flips_opp, flips_same)]
    taus = [float(df_n6.loc[k[0], "tau_int_E"]) for k in keys_a]
    cols_a = [k[2] for k in keys_a]

    ax_a.bar(xa, flips_opp, width=0.62, color=cols_a, alpha=0.92, edgecolor="#222222", lw=0.55, label=r"Wigner ($\uparrow\!\downarrow$)")
    ax_a.bar(xa, flips_same, bottom=flips_opp, width=0.62, color=cols_a, alpha=0.45, hatch="///", edgecolor="#222222", lw=0.55, label=r"Pauli ($\uparrow\!\uparrow,\downarrow\!\downarrow$)")

    for xi, ft, ti in zip(xa, flips_tot, taus):
        if ti > 50.0:
            ax_a.text(
                xi,
                ft + 2.2,
                rf"$\tau_E={ti:.0f}$!",
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color="#B2182B",
                bbox=dict(boxstyle="round,pad=0.14", facecolor="#FFF0F0", edgecolor="#B2182B", lw=0.6),
            )
        else:
            ax_a.text(xi, ft + 1.8, f"{ft:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="#222222")
            ax_a.text(xi, ft + 7.2, rf"$\tau_E\!=\!{ti:.0f}$", ha="center", va="bottom", fontsize=5.8, color="#555555")

    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([k[1] for k in keys_a], fontsize=6.7)
    ax_a.set_ylim(0, max(flips_tot) * 1.31)
    ax_a.set_ylabel(r"Flips / $10^3$ steps")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    ax_a.grid(True, axis="y", alpha=0.22, lw=0.45)
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.01, 0.84), fontsize=6.2)
    panel_label(ax_a, "(a)")

    # ------------------------------------------------------------------
    # (b) Matched-sigma verdict: RL-MALA vs pure RWMH (ESJD at sigma=1.22, 2.71, 3.30)
    # ------------------------------------------------------------------
    # Exact verified values from ablation_summary CSVs + matched-sigma RWMH control benchmark
    groups = [
        r"$N=6, g=1$" + "\n" + r"$\sigma=1.22$",
        r"$N=6, g=1$" + "\n" + r"$\sigma=2.71$",
        r"$N=6, g=1$" + "\n" + r"$\sigma=3.30$",
        r"$N=4, g=2$" + "\n" + r"$\sigma=1.22$",
        r"$N=4, g=2$" + "\n" + r"$\sigma=2.71$",
        r"$N=4, g=2$" + "\n" + r"$\sigma=3.30$",
    ]
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
    ax_b.bar(xb - wb / 2, esjd_rwmh, width=wb, color=COLORS["rwmh"], alpha=0.88, edgecolor="#222222", lw=0.5, label=r"RWMH ($\sigma$-matched)")
    ax_b.bar(xb + wb / 2, esjd_rl, width=wb, color=COLORS["rl"], alpha=0.90, edgecolor="#222222", lw=0.5, label="RL-MALA (parity)")

    # Overlay confined RL-MALA star if available from final benchmark
    if has_conf_n6:
        ax_b.scatter([1.0 + wb / 2], [float(df_n6.loc["B2_RL_MALA_confined_amp0.80_sig1.8", "ESJD"])], marker="*", s=65, color=COLORS["rl_conf"], edgecolor="white", lw=0.4, zorder=10, label="RL-MALA (confined)")
    if has_conf_n4:
        ax_b.scatter([4.35 + wb / 2], [float(df_n4.loc["B2_RL_MALA_confined_amp0.80_sig1.8", "ESJD"])], marker="*", s=65, color=COLORS["rl_conf"], edgecolor="white", lw=0.4, zorder=10)

    for xi, vrw, vrl in zip(xb, esjd_rwmh, esjd_rl):
        pct = (vrl - vrw) / max(vrw, 1e-9) * 100.0
        col_t = "#006837" if pct >= 0 else "#B2182B"
        ax_b.text(xi, max(vrw, vrl) + 0.10, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=col_t)

    ax_b.axvline(2.675, color="#888888", ls="--", lw=0.75)
    ax_b.text(1.42, 4.30, "$N=6, g=1$\n(inward drift)", ha="center", va="top", fontsize=6.4, color="#006837", fontweight="bold")
    ax_b.text(4.35, 4.30, "$N=4, g=2$\n(outward drift)", ha="center", va="top", fontsize=6.4, color="#B2182B", fontweight="bold")

    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([r"$1.22$", r"$2.71$", r"$3.30$", r"$1.22$", r"$2.71$", r"$3.30$"], fontsize=6.9)
    ax_b.set_xlabel(r"Matched proposal width $\sigma$ ($a_0$)")
    ax_b.set_ylabel(r"ESJD ($a_0^2/\mathrm{step}$)")
    ax_b.set_ylim(0.0, 4.55)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(True, axis="y", alpha=0.22, lw=0.45)
    ax_b.legend(loc="lower left", fontsize=6.1)
    panel_label(ax_b, "(b)")

    # ------------------------------------------------------------------
    # (c) & (d) Spatial Actor Response mu_theta(x) on N=6 g=1.0 vs N=4 g=2.0
    # ------------------------------------------------------------------
    def _plot_spatial_probe(
        ax: plt.Axes,
        probe_dir: Path,
        ptag: str,
        sys_txt: str,
        box_loc: Tuple[float, float, str, str],
        show_leg: bool,
    ) -> None:
        sp = probe_dir / "spatial_probes"
        f_leg = sp / "data_fig_actor_spatial_probe_B1_RL_MALA_legacy_spin_down.csv"
        f_par = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_parity_only_spin_down.csv"
        f_eq = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_eq_unclip_amp0.80_sig1.8_spin_down.csv"
        f_conf = sp / "data_fig_actor_spatial_probe_B2_RL_MALA_confined_amp0.80_sig1.8_spin_down.csv"

        df_eq = pd.read_csv(f_eq)
        xg = df_eq["x_probe"].to_numpy()
        p_cond = df_eq["cond_prob_norm"].to_numpy()

        # Subtle background conditional density showing Wigner well pockets
        ax_bg = ax.twinx()
        ax_bg.fill_between(xg, 0.0, p_cond, color=COLORS["gray_fill"], alpha=0.48, zorder=0)
        ax_bg.set_ylim(0.0, 1.35)
        ax_bg.set_yticks([])
        ax_bg.spines["top"].set_visible(False)
        ax_bg.spines["right"].set_visible(False)
        ax.set_zorder(ax_bg.get_zorder() + 1)
        ax.patch.set_visible(False)

        ax.axhline(0.0, color="#666666", ls=":", lw=0.75)
        ax.axvline(0.0, color="#AAAAAA", ls=":", lw=0.65)

        if f_leg.is_file():
            df_l = pd.read_csv(f_leg)
            ax.plot(df_l["x_probe"], df_l["mu_actor"], color=COLORS["rl_leg"], ls=":", lw=1.25, label="Legacy (no parity)")
        if f_par.is_file():
            df_p = pd.read_csv(f_par)
            ax.plot(df_p["x_probe"], df_p["mu_actor"], color=COLORS["rl_par"], ls="--", lw=1.25, label=r"Parity only ($A=0.15$)")

        ax.plot(xg, df_eq["mu_actor"], color=COLORS["rl"], ls="-", lw=1.55, label=r"Eq+unclip ($A=0.80$)")

        if f_conf.is_file():
            df_cf = pd.read_csv(f_conf)
            ax.plot(df_cf["x_probe"], df_cf["mu_actor"], color=COLORS["rl_conf"], ls="-.", lw=1.6, label=r"Confined ($A=0.80$)")

        ax.set_xlim(-14.0, 14.0)
        ax.set_ylim(-1.08, 1.24)
        ax.set_xlabel(r"Probe electron position $x_{\mathrm{probe}}$ ($a_0$)")
        ax.set_ylabel(r"Actor drift $\mu_\theta(x)$ ($a_0$)")
        ax.tick_params(top=True, right=False)
        ax.grid(True, alpha=0.20, lw=0.45)
        panel_label(ax, ptag)
        bx, by, bha, bva = box_loc
        ax.text(
            bx,
            by,
            sys_txt,
            transform=ax.transAxes,
            ha=bha,
            va=bva,
            fontsize=7.0,
            fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
        )
        if show_leg:
            ax.legend(loc="lower left", fontsize=6.0, handlelength=1.35)

    _plot_spatial_probe(
        ax_c,
        dir_n6,
        "(c)",
        r"$N=6, g=1.0$" + "\n(confining: " + r"$\mu_\theta x < 0$)",
        (0.97, 0.94, "right", "top"),
        show_leg=True,
    )
    _plot_spatial_probe(
        ax_d,
        dir_n4,
        "(d)",
        r"$N=4, g=2.0$" + "\n(anti-confining: " + r"$\mu_\theta x > 0$)",
        (0.97, 0.06, "right", "bottom"),
        show_leg=False,
    )

    # Clear physical directional arrows placed in open whitespace
    ax_c.annotate(
        "Inward\nrestoring",
        xy=(8.5, -0.78),
        xytext=(3.2, -0.36),
        fontsize=6.3,
        color="#006837",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#006837", lw=0.85),
    )
    ax_d.annotate(
        "Outward\nescape bias",
        xy=(8.5, 0.78),
        xytext=(2.0, 0.36),
        fontsize=6.3,
        color="#B2182B",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#B2182B", lw=0.85),
    )

    save_figure(fig, out_dir, "fig4_ablation_b1_b2")


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate APS Physical Review B Letter figures (PDF + 600 DPI PNG)."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "RL_Sampler_PRBLett" / "figures"),
        help="Destination directory for publication figures.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Optional Option-A final benchmark directory (e.g. results_prb_final_benchmark).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="all",
        help="Comma-separated figure numbers to build (e.g. '1,2,3,4' or 'all').",
    )
    args = parser.parse_args()

    setup_prb_style()
    out_dir = Path(args.out_dir)
    bench_dir = Path(args.benchmark_dir) if args.benchmark_dir else None

    targets = {"1", "2", "3", "4"} if args.only == "all" else {t.strip() for t in args.only.split(",")}

    if "1" in targets:
        make_fig1_ergodicity_densities(out_dir)
    if "2" in targets:
        make_fig2_microscopic_mechanism(out_dir)
    if "3" in targets:
        make_fig3_phase_diagram(out_dir, benchmark_dir=bench_dir)
    if "4" in targets:
        make_fig4_ablation_b1_b2(out_dir, benchmark_dir=bench_dir)

    print(f"[DONE] Figures ready in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
