#!/usr/bin/env python3
"""
scripts/plot_actor_ablation_comparison.py
=========================================
Analiza y grafica de manera conjunta los resultados de los dos benchmarks de ablación
del Actor RL-MALA:
  - Benchmark 1 (Control Justo vs MALA Grande y RWMH de Ancho Emparejado):
      B1_MALA_base_tau0.10, B1_MALA_large_tau0.50, B1_MALA_matched_tau0.7389,
      B1_RWMH_matched_sig1.2156, B1_RL_MALA_legacy
  - Benchmark 2 (Equivarianza Exacta de Paridad + Sigma Desbloqueada + Amplitud Física):
      B1_RL_MALA_legacy, B2_RL_MALA_parity_only,
      B2_RL_MALA_eq_unclip_amp0.80_sig1.8, B2_RL_MALA_eq_unclip_amp1.20_sig2.0

Produce:
  1. `ablation_summary_<SYS_TAG>.csv`: tabla cuantitativa completa (energía, SE,
     aceptación, ESJD, tau_int(E), D_KL^spin, flips opp/same con histéresis, y
     métricas de sonda espacial del actor: rango de mu, sigma aplicada, % pegado
     al techo, asimetría de paridad y sensibilidad a F_Q).
  2. `fig_ablation_b1_b2_metrics_<SYS_TAG>.{png,pdf}`: comparativa multi-panel (2x3)
     de todos los modelos de Benchmark 1 y Benchmark 2.
  3. `fig_ablation_b2_spatial_overlay_<SYS_TAG>.{png,pdf}`: superposición directa
     de la respuesta espacial mu_actor(x) y sigma_RL(x) de todos los actores neuronales
     (Legado vs Paridad-Only vs Equivariante+Desbloqueado) sobre el mismo barrido.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_vmc.config import Experiment1DConfig, PostProcessConfig
from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.postprocess.diagnostics import summarize_bundle_diagnostics
from scripts.analyze_flips_vs_g import compute_flip_statistics, load_bundle_walkers_and_meta
from scripts.plot_flips_system_detailed import (
    _compute_exact_casscf_density_np,
    _find_casscf_npz_for_system,
)
from scripts.probe_actor_spatial_response import (
    find_wigner_well_centers,
    plot_spatial_probe_figure,
    run_spatial_actor_probe,
)


# Orden canónico y estilos de presentación para B1 + B2
MODEL_ORDER = [
    ("B1_MALA_base_tau0.10", "MALA base\n($\\tau=0.10, \\sigma=0.45$)", "#636363", "B1"),
    ("B1_MALA_large_tau0.50", "MALA grande\n($\\tau=0.50, \\sigma=1.00$)", "#969696", "B1"),
    ("B1_MALA_matched_tau0.7389", "MALA $\\sigma$-match\n($\\tau=0.74, \\sigma=1.22$)", "#d95f02", "B1"),
    ("B1_RWMH_matched_sig1.2156", "RWMH $\\sigma$-match\n(deriva$=0, \\sigma=1.22$)", "#e7298a", "B1"),
    ("B1_RWMH_matched_sig2.7055", "RWMH $\\sigma$-match\n(deriva$=0, \\sigma=2.71$)", "#ce1256", "B1"),
    ("B1_RL_MALA_legacy", "RL-MALA legado\n(par=OFF, clip=1.0)", "#1f77b4", "B1/B2"),
    ("B2_RL_MALA_parity_only", "RL-MALA paridad\n(par=ON, clip=1.0)", "#7570b3", "B2"),
    ("B2_RL_MALA_eq_unclip_amp0.80_sig1.8", "RL-MALA eq+unclip\n($A=0.80, s_\\sigma=1.8$)", "#1b9e77", "B2"),
    ("B2_RL_MALA_confined_amp0.80_sig1.8", "RL-MALA confined\n($A=0.80, s_\\sigma=1.8$)", "#006837", "B2"),
    ("B2_RL_MALA_eq_unclip_amp1.20_sig2.0", "RL-MALA eq+unclip\n($A=1.20, s_\\sigma=2.0$)", "#2ca02c", "B2"),
]


def _kl_divergence(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    eps = 1e-10
    p_s = np.maximum(np.asarray(p, dtype=np.float64), eps)
    q_s = np.maximum(np.asarray(q, dtype=np.float64), eps)
    p_s = p_s / (np.sum(p_s) * dx)
    q_s = q_s / (np.sum(q_s) * dx)
    return float(np.sum(p_s * np.log(p_s / q_s)) * dx)


def _compute_spin_kl_from_walkers(
    walkers: np.ndarray,
    n_up: int,
    casscf_ref: Optional[Dict[str, np.ndarray]],
) -> Tuple[float, float, float]:
    if casscf_ref is None:
        return float("nan"), float("nan"), float("nan")
    xg = np.asarray(casscf_ref["x"], dtype=np.float64)
    p_up_ref = np.asarray(casscf_ref["p_up"], dtype=np.float64)
    p_dn_ref = np.asarray(casscf_ref["p_down"], dtype=np.float64)
    dx = float(xg[1] - xg[0])
    edges = np.concatenate([[xg[0] - 0.5 * dx], 0.5 * (xg[:-1] + xg[1:]), [xg[-1] + 0.5 * dx]])

    w = np.asarray(walkers, dtype=np.float64)
    if w.ndim == 5:
        w = w[..., 0]
    w_up = w[..., :n_up].reshape(-1)
    w_dn = w[..., n_up:].reshape(-1)

    h_up, _ = np.histogram(w_up, bins=edges, density=True)
    h_dn, _ = np.histogram(w_dn, bins=edges, density=True)
    kl_up = _kl_divergence(h_up, p_up_ref, dx)
    kl_dn = _kl_divergence(h_dn, p_dn_ref, dx)
    return 0.5 * (kl_up + kl_dn), kl_up, kl_dn


def _get_casscf_energy(npz_path: Optional[str]) -> float:
    if not npz_path or not os.path.isfile(npz_path):
        return float("nan")
    try:
        data = np.load(npz_path, allow_pickle=True)
        for k in ("selected_energy", "energy", "E_total", "total_energy", "E_casscf", "e_tot"):
            if k in data:
                val = np.asarray(data[k]).reshape(-1)[0]
                return float(val)
        if "metadata_json" in data:
            import json
            meta = json.loads(str(data["metadata_json"].item()))
            for k in ("selected_energy", "energy"):
                if k in meta:
                    return float(meta[k])
    except Exception:
        pass
    return float("nan")


def analyze_system_ablation(
    sys_dir: str,
    casscf_npz: Optional[str] = None,
    probe_spin: str = "down",
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    probes: Dict[str, Dict[str, Any]] = {}
    pp_cfg = PostProcessConfig()

    # Descubrir subcarpetas con run_bundle.h5 respetando MODEL_ORDER y añadiendo extras al final
    known_dirs = {m[0] for m in MODEL_ORDER}
    ordered_specs: List[Tuple[str, str, str, str]] = []
    for dname, disp, col, grp in MODEL_ORDER:
        h5p = os.path.join(sys_dir, dname, "run_bundle.h5")
        if os.path.isfile(h5p):
            ordered_specs.append((dname, disp, col, grp))
    for entry in sorted(os.listdir(sys_dir)):
        if entry in known_dirs:
            continue
        h5p = os.path.join(sys_dir, entry, "run_bundle.h5")
        if os.path.isfile(h5p):
            ordered_specs.append((entry, entry, "#377eb8", "Extra"))

    if not ordered_specs:
        raise FileNotFoundError(f"No se encontraron bundles run_bundle.h5 en {sys_dir}")

    # Referencia CASSCF
    first_h5 = os.path.join(sys_dir, ordered_specs[0][0], "run_bundle.h5")
    first_bundle = load_run_bundle_h5(first_h5)
    first_cfg = Experiment1DConfig.from_dict(first_bundle["config"])
    if casscf_npz is None:
        casscf_npz = first_cfg.ansatz.casscf_npz_path or first_cfg.io.reference_casscf_npz
    e_ref = _get_casscf_energy(casscf_npz)
    casscf_ref = _compute_exact_casscf_density_np(casscf_npz, x_max=20.0, n_grid=400) if (casscf_npz and os.path.isfile(casscf_npz)) else None

    sys_meta = {
        "N": int(first_cfg.system.n_elec),
        "n_up": int(first_cfg.system.n_up),
        "n_dn": int(first_cfg.system.n_dn),
        "omega": float(first_cfg.system.omega),
        "g": float(first_cfg.system.interaction_strength),
        "E_casscf": e_ref,
        "sys_tag": f"N{first_cfg.system.n_elec}_w{first_cfg.system.omega:g}_g{first_cfg.system.interaction_strength:g}",
    }

    for dname, disp, col, grp in ordered_specs:
        h5p = os.path.join(sys_dir, dname, "run_bundle.h5")
        print(f"  [*] Analizando {dname}...")
        bundle = load_run_bundle_h5(h5p)
        cfg = Experiment1DConfig.from_dict(bundle["config"])
        diag = summarize_bundle_diagnostics(bundle, pp_cfg)["summary"]

        binfo = load_bundle_walkers_and_meta(h5p)
        is_neural = cfg.proposal.proposal_type in ("neural_deepset", "neural_flat")
        flips = compute_flip_statistics(
            binfo["walkers"],
            n_up=binfo["n_up"],
            omega=binfo["omega"],
            g_val=binfo["g"],
            tau=binfo["tau"],
            actor_amp=binfo["actor_amplitude"],
            sigma_scale=binfo["sigma_action_scale"],
            actor_params=binfo.get("actor_params"),
            is_neural=is_neural,
            npz_path=casscf_npz,
            parity_equivariant=bool(binfo.get("parity_equivariant", False)),
            unclip_sigma_action=bool(binfo.get("unclip_sigma_action", False)),
            scale_clip_min=float(binfo.get("scale_clip_min", -5.0)),
            scale_clip_max=float(binfo.get("scale_clip_max", 1.0)),
        )
        kl_spin, kl_up, kl_dn = _compute_spin_kl_from_walkers(binfo["walkers"], binfo["n_up"], casscf_ref)

        # Determinar sigma efectiva de propuesta
        if cfg.proposal.proposal_type == "metropolis":
            sigma_eff = float(cfg.proposal.rw_sigma)
        elif not is_neural:
            sigma_eff = math.sqrt(2.0 * float(cfg.proposal.tau))
        else:
            s_stats = bundle.get("production", {}).get("sampler_stats", {})
            if "actor_sigma" in s_stats:
                sigma_eff = float(np.mean(np.asarray(s_stats["actor_sigma"], dtype=np.float64)))
            else:
                sigma_eff = float("nan")

        row: Dict[str, Any] = {
            "model_key": dname,
            "label": disp.replace("\n", " "),
            "label_multiline": disp,
            "color": col,
            "group": grp,
            "proposal_type": cfg.proposal.proposal_type,
            "tau": float(cfg.proposal.tau),
            "rw_sigma": float(cfg.proposal.rw_sigma),
            "actor_amplitude": float(cfg.proposal.actor_amplitude),
            "sigma_action_scale": float(cfg.proposal.sigma_action_scale),
            "scale_clip_max": float(cfg.proposal.scale_clip_max),
            "parity_equivariant": bool(cfg.proposal.parity_equivariant),
            "unclip_sigma_action": bool(cfg.proposal.unclip_sigma_action),
            "sigma_eff_mean": sigma_eff,
            "E_mean": float(diag.get("E_mean", float("nan"))),
            "SE_E": float(diag.get("SE_E", float("nan"))),
            "E_err_abs": abs(float(diag.get("E_mean", float("nan"))) - e_ref) if np.isfinite(e_ref) else float("nan"),
            "acceptance": float(diag.get("acceptance", float("nan"))),
            "ESJD": float(diag.get("ESJD", float("nan"))),
            "tau_int_E": float(diag.get("tau_int_E", float("nan"))),
            "tau_int_pos": float(diag.get("tau_int_pos", float("nan"))),
            "kl_spin": kl_spin,
            "kl_up": kl_up,
            "kl_dn": kl_dn,
            "hyst_flips_total_1k": float(flips.get("hyst_flips_total_mean", 0.0)),
            "hyst_flips_opp_1k": float(flips.get("hyst_flips_opp_mean", 0.0)),
            "hyst_flips_same_1k": float(flips.get("hyst_flips_same_mean", 0.0)),
            "one_step_fraction": float(flips.get("one_step_fraction", 0.0)),
            # Campos de sonda espacial (solo para actores neuronales)
            "a_mu_std": float("nan"),
            "mu_actor_range_bohr": float("nan"),
            "sigma_probe_min": float("nan"),
            "sigma_probe_max": float("nan"),
            "sigma_probe_std": float("nan"),
            "frac_sigma_clipped_pct": float("nan"),
            "mean_parity_asymmetry_mu": float("nan"),
            "mean_parity_asymmetry_sig": float("nan"),
            "max_fq_sensitivity_mu": float("nan"),
            "max_fq_sensitivity_sig": float("nan"),
        }

        if is_neural and bundle.get("checkpoints", {}).get("actor_final") is not None:
            pr = run_spatial_actor_probe(
                bundle,
                h5p,
                probe_spin=probe_spin,
                n_grid=501,
                casscf_npz_override=casscf_npz,
            )
            probes[dname] = {"probe": pr, "label": disp.replace("\n", " "), "color": col}
            sm = pr["summary"]
            row["a_mu_std"] = float(sm["a_mu_std"])
            row["mu_actor_range_bohr"] = float(sm["mu_actor_range_bohr"])
            row["sigma_probe_min"] = float(sm["sigma_applied_min"])
            row["sigma_probe_max"] = float(sm["sigma_applied_max"])
            row["sigma_probe_std"] = float(sm["sigma_applied_std"])
            row["frac_sigma_clipped_pct"] = float(sm["frac_sigma_clipped_pct"])
            row["mean_parity_asymmetry_mu"] = float(sm["mean_parity_asymmetry_mu"])
            row["mean_parity_asymmetry_sig"] = float(sm["mean_parity_asymmetry_sig"])
            row["max_fq_sensitivity_mu"] = float(sm["max_fq_sensitivity_mu"])
            row["max_fq_sensitivity_sig"] = float(sm["max_fq_sensitivity_sig"])

        rows.append(row)

    return rows, probes, sys_meta


def plot_metrics_comparison(
    rows: List[Dict[str, Any]],
    sys_meta: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    plt.rcParams.update({"font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 11.2), dpi=220)
    axes = axes.ravel()

    x = np.arange(len(rows))
    labels = [r["label_multiline"] for r in rows]
    colors = [r["color"] for r in rows]

    def _annotate_bars(ax, bars, fmt="{:.2f}", log_y=False):
        for b in bars:
            h = b.get_height()
            if not np.isfinite(h):
                continue
            y_pos = h * (1.12 if log_y else 1.0)
            va = "bottom"
            ax.annotate(
                fmt.format(h),
                xy=(b.get_x() + b.get_width() / 2.0, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7.6,
                weight="bold",
                color="#222222",
            )

    # (a) ESJD y Tasa de Aceptación (doble eje)
    ax = axes[0]
    esjd_vals = [r["ESJD"] for r in rows]
    acc_vals = [r["acceptance"] * 100.0 for r in rows]
    bars = ax.bar(x, esjd_vals, color=colors, alpha=0.86, edgecolor="#222222", lw=0.7)
    _annotate_bars(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("ESJD [Bohr$^2$/paso]", weight="bold", color="#222222")
    ax.set_title("(a) ESJD (barras) y Aceptación MH (puntos)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    ax_r = ax.twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.plot(x, acc_vals, "o--", color="#111111", lw=1.5, ms=5.5, label="Aceptación [%]")
    ax_r.set_ylabel("Aceptación MH [%]", weight="bold", color="#111111")
    ax_r.set_ylim(0, 105)
    for xi, av in zip(x, acc_vals):
        if np.isfinite(av):
            ax_r.annotate(f"{av:.0f}%", (xi, av), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.2, color="#111111")

    # (b) Opposite-Spin Flips / 1k pasos
    ax = axes[1]
    opp_vals = [r["hyst_flips_opp_1k"] for r in rows]
    bars = ax.bar(x, opp_vals, color=colors, alpha=0.86, edgecolor="#222222", lw=0.7)
    _annotate_bars(ax, bars, fmt="{:.2f}")
    ax.set_ylabel("Flips $\\uparrow\\downarrow$ / 1k pasos (histéresis)", weight="bold")
    ax.set_title("(b) Intercambios de Espín Opuesto ($\\uparrow\\downarrow$)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    # (c) Same-Spin (Pauli) Flips / 1k pasos
    ax = axes[2]
    same_vals = [r["hyst_flips_same_1k"] for r in rows]
    bars = ax.bar(x, same_vals, color=colors, alpha=0.86, edgecolor="#222222", lw=0.7)
    _annotate_bars(ax, bars, fmt="{:.3f}")
    ax.set_ylabel("Flips $\\uparrow\\uparrow,\\downarrow\\downarrow$ / 1k pasos (Pauli)", weight="bold")
    ax.set_title("(c) Cruces Nodales de Pauli ($\\uparrow\\uparrow / \\downarrow\\downarrow$)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    # (d) D_KL^spin vs CASSCF (menor es mejor)
    ax = axes[3]
    kl_vals = [max(r["kl_spin"], 1e-6) if np.isfinite(r["kl_spin"]) else np.nan for r in rows]
    bars = ax.bar(x, kl_vals, color=colors, alpha=0.86, edgecolor="#222222", lw=0.7)
    ax.set_yscale("log")
    _annotate_bars(ax, bars, fmt="{:.3f}", log_y=True)
    ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ vs CASSCF (escala log)", weight="bold")
    ax.set_title("(d) Ergodicidad de Espín $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ (menor = mejor)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    # (e) Autocorrelación integrada tau_int(E)
    ax = axes[4]
    tau_vals = [r["tau_int_E"] for r in rows]
    bars = ax.bar(x, tau_vals, color=colors, alpha=0.86, edgecolor="#222222", lw=0.7)
    _annotate_bars(ax, bars, fmt="{:.1f}")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ [pasos]", weight="bold")
    ax.set_title("(e) Tiempo de Autocorrelación $\\tau_{\\mathrm{int}}(E)$ (menor = mejor)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")

    # (f) Energía media ± SE vs E_CASSCF exacto
    ax = axes[5]
    e_vals = [r["E_mean"] for r in rows]
    se_vals = [r["SE_E"] for r in rows]
    e_ref = sys_meta["E_casscf"]
    if np.isfinite(e_ref):
        ax.axhline(e_ref, color="#d62728", lw=1.6, ls="--", label=f"$E_{{\\mathrm{{CASSCF}}}} = {e_ref:.6f}\\,E_h$")
    for xi, ev, sv, col in zip(x, e_vals, se_vals, colors):
        ax.errorbar([xi], [ev], yerr=[sv], fmt="o", color=col, ecolor=col, capsize=4, ms=7, lw=1.8)
    ax.set_ylabel(r"Energía Media $\langle E_L \rangle \pm \mathrm{SE}$ [$E_h$]", weight="bold")
    ax.set_title("(f) Insesgamiento de Energía (Balance Detallado)", weight="bold")
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    if np.isfinite(e_ref):
        ax.legend(loc="best", fontsize=8.2)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=7.6)
        # Separador visual entre B1 clásicos y actores RL
        n_classic = sum(1 for r in rows if r["proposal_type"] in ("mala", "metropolis"))
        if 0 < n_classic < len(rows):
            ax.axvline(n_classic - 0.5, color="#333333", ls=":", lw=1.2, alpha=0.65)

    fig.suptitle(
        f"Benchmarks B1 (Control Justo vs MALA/RWMH) y B2 (Actor Equivariante Desbloqueado) — "
        f"$N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$",
        fontsize=12.6,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955], h_pad=2.8, w_pad=1.6)

    saved = []
    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_ablation_b1_b2_metrics_{sys_meta['sys_tag']}"
    for ext in ("png", "pdf"):
        out_p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
        print(f"  [+] Figura guardada: {out_p}")
    plt.close(fig)
    return saved


def plot_spatial_overlay(
    probes: Dict[str, Dict[str, Any]],
    sys_meta: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    if not probes:
        return []
    plt.rcParams.update({"font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(14.6, 14.8), dpi=240, gridspec_kw={"height_ratios": [1.15, 1.15, 0.95]})
    ax1, ax2, ax3 = axes

    first_pr = next(iter(probes.values()))["probe"]
    xg = first_pr["x_grid"]
    same_fx = first_pr["same_fixed_x"]
    opp_fx = first_pr["opp_fixed_x"]
    empty_x = first_pr["empty_well_x"]

    for ax in (ax1, ax2):
        for idx_s, xs in enumerate(same_fx):
            ax.axvline(xs, color="#d62728", ls="--", lw=1.2, alpha=0.55, label=("Fijo mismo espín (nodo Pauli)" if (ax is ax1 and idx_s == 0) else None))
        for idx_o, xo in enumerate(opp_fx):
            ax.axvline(xo, color="#1f77b4", ls=":", lw=1.1, alpha=0.45, label=("Fijo espín opuesto" if (ax is ax1 and idx_o == 0) else None))
        ax.axvspan(empty_x - 0.5, empty_x + 0.5, color="#2ca02c", alpha=0.10, label=("Pozo Wigner vacío" if ax is ax1 else None))

    ax1.axhline(0.0, color="#222222", lw=0.9, alpha=0.6)
    ax2.axhline(first_pr["summary"]["base_sigma_mala"], color="#636363", ls=":", lw=1.4, label=f"MALA base ($\\sigma={first_pr['summary']['base_sigma_mala']:.3f}$ Bohr)")
    ax2.axhline(first_pr["summary"]["base_sigma_mala"] * math.e, color="#d95f02", ls="-.", lw=1.3, label=f"Techo legado ($\\sigma={first_pr['summary']['base_sigma_mala']*math.e:.3f}$ Bohr)")

    names_bar, par_mu_vals, sig_std_vals, fq_sens_vals, cols_bar = [], [], [], [], []
    for dname, item in probes.items():
        pr = item["probe"]
        sm = pr["summary"]
        col = item["color"]
        lbl = item["label"]
        ls = "--" if not sm.get("parity_equivariant", False) else "-"
        lw = 2.4 if sm.get("unclip_sigma_action", False) else 1.8

        ax1.plot(pr["x_grid"], pr["mu_actor"], color=col, ls=ls, lw=lw, label=f"{lbl} (par={sm['mean_parity_asymmetry_mu']:.3f})")
        ax2.plot(pr["x_grid"], pr["sigma_applied"], color=col, ls=ls, lw=lw, label=f"{lbl} ($\\sigma \\in [{sm['sigma_applied_min']:.2f}, {sm['sigma_applied_max']:.2f}]$, clip={sm['frac_sigma_clipped_pct']:.0f}%)")

        names_bar.append(lbl)
        par_mu_vals.append(sm["mean_parity_asymmetry_mu"])
        sig_std_vals.append(sm["sigma_applied_std"])
        fq_sens_vals.append(sm["max_fq_sensitivity_mu"])
        cols_bar.append(col)

    ax1.set_xlim(xg[0], xg[-1])
    ax1.set_ylabel(r"$\mu_{\mathrm{actor}}(x)$ [Bohr]", weight="bold")
    ax1.set_title("(a) Deriva Aprendida $\\mu_{\\mathrm{actor}}(x)$: Eliminación del Sesgo de Paridad y Escalado Físico de Amplitud", weight="bold", pad=46)
    ax1.grid(True, alpha=0.25, ls="--")
    ax1.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=6.9,
        ncol=4,
        framealpha=0.72,
        borderpad=0.3,
        columnspacing=0.85,
        handlelength=1.8,
    )

    ax2.set_xlim(xg[0], xg[-1])
    ax2.set_xlabel(r"Posición del electrón de prueba $x_{\mathrm{probe}}$ [Bohr]", weight="bold")
    ax2.set_ylabel(r"$\sigma_{\mathrm{RL}}(x)$ [Bohr]", weight="bold")
    ax2.set_title("(b) Modulación Espacial del Ancho $\\sigma_{\\mathrm{RL}}(x)$: Desbloqueo del Techo Legado (clip=1.0)", weight="bold", pad=46)
    ax2.grid(True, alpha=0.25, ls="--")
    ax2.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=6.9,
        ncol=3,
        framealpha=0.72,
        borderpad=0.3,
        columnspacing=0.85,
        handlelength=1.8,
    )

    # Panel (c): Barras agrupadas de diagnósticos de calidad estructural del actor
    idx_b = np.arange(len(names_bar))
    w_b = 0.25
    b1 = ax3.bar(idx_b - w_b, par_mu_vals, width=w_b, color="#d62728", alpha=0.85, label=r"Asimetría Paridad $\langle|a_\mu(x)+a_\mu(-x)|\rangle$ (ideal = 0)")
    b2 = ax3.bar(idx_b, sig_std_vals, width=w_b, color="#1b9e77", alpha=0.85, label=r"Modulación Espacial $\mathrm{std}(\sigma_{\mathrm{RL}})$ [Bohr]")
    b3 = ax3.bar(idx_b + w_b, fq_sens_vals, width=w_b, color="#7570b3", alpha=0.85, label=r"Sensibilidad a $F_Q$: $\max|\Delta a_\mu|$")
    for bars_g in (b1, b2, b3):
        for b in bars_g:
            h = b.get_height()
            ax3.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2.0, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.6, weight="bold")

    ax3.set_xticks(idx_b)
    ax3.set_xticklabels(names_bar, rotation=10, ha="right", fontsize=8.2)
    ax3.set_ylabel("Magnitud Diagnóstica", weight="bold")
    ax3.set_ylim(0, max(max(par_mu_vals + sig_std_vals + fq_sens_vals), 0.1) * 1.18)
    ax3.set_title("(c) Diagnósticos Estructurales: Paridad Exacta, Modulación de $\\sigma(x)$ y Sensibilidad a la Fuerza Cuántica $F_Q$", weight="bold", pad=28)
    ax3.grid(True, axis="y", alpha=0.25, ls="--")
    ax3.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=7.4,
        ncol=3,
        framealpha=0.72,
        borderpad=0.3,
        columnspacing=1.0,
    )

    fig.suptitle(
        f"Benchmark 2 — Comparativa Directa de Respuesta Espacial del Actor ($N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$)",
        fontsize=12.4,
        weight="bold",
        y=0.988,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.958], h_pad=3.2)

    saved = []
    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_ablation_b2_spatial_overlay_{sys_meta['sys_tag']}"
    for ext in ("png", "pdf"):
        out_p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
        print(f"  [+] Figura guardada: {out_p}")
    plt.close(fig)
    return saved


def _extract_fixed_positions_from_dmin(x_grid: np.ndarray, d_min: np.ndarray) -> np.ndarray:
    """Reconstruye con precisión de máquina las posiciones fijas desde los mínimos locales de d_min(x)."""
    if len(x_grid) < 3 or np.all(np.isnan(d_min)):
        return np.array([], dtype=np.float64)
    dx = float(np.median(np.diff(x_grid)))
    found: List[float] = []
    for k in range(1, len(x_grid) - 1):
        if np.isfinite(d_min[k]) and d_min[k] <= d_min[k - 1] and d_min[k] <= d_min[k + 1] and d_min[k] < 1.5 * dx:
            # En el lado izquierdo de la V (k-1), x_f = x[k-1] + d_min[k-1]
            # En el lado derecho de la V (k+1), x_f = x[k+1] - d_min[k+1]
            xf = 0.5 * ((x_grid[k - 1] + d_min[k - 1]) + (x_grid[k + 1] - d_min[k + 1]))
            if not found or abs(xf - found[-1]) > 0.25:
                found.append(float(xf))
    return np.asarray(sorted(found), dtype=np.float64)


def replot_spatial_from_csv_dir(
    csv_dir: str,
    out_dir: Optional[str] = None,
    casscf_npz: Optional[str] = None,
    probe_spin: str = "down",
    replot_individual_probes: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Re-grafica `fig_ablation_b2_spatial_overlay_*.{png,pdf}` (y opcionalmente cada
    `fig_actor_spatial_probe_*.{png,pdf}`) directamente desde `ablation_summary_*.csv`
    y `spatial_probes/data_fig_actor_spatial_probe_*.csv` sin necesitar los `.h5`.
    """
    import glob
    import re

    summary_csvs = sorted(glob.glob(os.path.join(csv_dir, "ablation_summary_*.csv")))
    if not summary_csvs:
        raise FileNotFoundError(f"No se encontró ablation_summary_*.csv en {csv_dir}")
    summary_csv = summary_csvs[0]
    m_tag = re.match(r"ablation_summary_N(\d+)_w([0-9.]+)_g([0-9.]+)\.csv$", os.path.basename(summary_csv))
    if not m_tag:
        raise ValueError(f"No se pudo parsear N/omega/g de {summary_csv}")
    N = int(m_tag.group(1))
    omega = float(m_tag.group(2))
    g_val = float(m_tag.group(3))
    n_up = N // 2
    n_dn = N - n_up
    sys_tag = f"N{N}_w{omega:g}_g{g_val:g}"

    if casscf_npz is None:
        casscf_npz = _find_casscf_npz_for_system([PROJECT_ROOT], N, omega, g_val)
    well_centers_ref, casscf_ref = find_wigner_well_centers({}, N, omega, g_val, npz_path=casscf_npz)

    with open(summary_csv, "r", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    row_by_key = {r["model_key"]: r for r in summary_rows}

    sp_dir = os.path.join(csv_dir, "spatial_probes")
    probes: Dict[str, Dict[str, Any]] = {}
    target_out = out_dir or csv_dir
    probe_out_dir = os.path.join(target_out, "spatial_probes") if replot_individual_probes else target_out

    plot_args = argparse.Namespace(
        figure_format="both",
        dpi=300,
        font_scale=1.0,
        no_titles=False,
        no_legends=False,
    )

    saved_probes: List[str] = []
    for dname, disp, col, _grp in MODEL_ORDER:
        if dname not in row_by_key:
            continue
        r = row_by_key[dname]
        if r.get("proposal_type") not in ("neural_deepset", "neural_flat"):
            continue
        data_csv = os.path.join(sp_dir, f"data_fig_actor_spatial_probe_{dname}_spin_{probe_spin}.csv")
        if not os.path.isfile(data_csv):
            continue

        cols_data: Dict[str, List[float]] = {}
        with open(data_csv, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for c_name in (reader.fieldnames or []):
                cols_data[c_name] = []
            for rec in reader:
                for c_name, val in rec.items():
                    cols_data[c_name].append(float(val))
        arr = {k: np.asarray(v, dtype=np.float64) for k, v in cols_data.items()}

        xg = arr["x_probe"]
        same_fx = _extract_fixed_positions_from_dmin(xg, arr["d_min_same"])
        opp_fx = _extract_fixed_positions_from_dmin(xg, arr["d_min_opp"])
        all_fx = np.sort(np.concatenate([same_fx, opp_fx]))

        # Pozo vacío: el único centro de `well_centers_ref` (o el reflejo de `all_fx`) no presente en `all_fx`
        if len(well_centers_ref) == N and len(all_fx) == N - 1:
            dists = [float(np.min(np.abs(wc - all_fx))) for wc in well_centers_ref]
            empty_idx = int(np.argmax(dists))
            empty_x = float(well_centers_ref[empty_idx])
            well_centers = np.sort(np.append(all_fx, empty_x))
        else:
            # Simetría del potencial armónico 1D
            mirrors = -all_fx
            dists = [float(np.min(np.abs(m - all_fx))) for m in mirrors]
            empty_x = float(mirrors[int(np.argmax(dists))]) if len(mirrors) > 0 else 0.0
            well_centers = np.sort(np.append(all_fx, empty_x))

        tau = float(r["tau"])
        amp = float(r["actor_amplitude"])
        sig_sc = float(r["sigma_action_scale"])
        clip_max = float(r["scale_clip_max"])
        parity_eq = str(r["parity_equivariant"]).strip().lower() == "true"
        unclip_sig = str(r["unclip_sigma_action"]).strip().lower() == "true"
        eff_sig_hi = clip_max if unclip_sig else min(clip_max, 1.0)
        eff_sig_lo = -5.0 if unclip_sig else -1.0
        base_sigma = math.sqrt(2.0 * tau)

        a_mu_clip = arr["a_mu_clip"]
        a_sig_raw = arr["a_sig_raw"]
        mu_actor = arr["mu_actor"]
        sigma_applied = arr["sigma_applied"]

        if "mu_actor_parity_expected" in arr:
            mu_par_exp = arr["mu_actor_parity_expected"]
        elif parity_eq:
            mu_par_exp = mu_actor.copy()
        else:
            mu_par_exp = -mu_actor[::-1]

        sm = {
            "h5_path": os.path.join(dname, "run_bundle.h5"),
            "N": N,
            "n_up": n_up,
            "n_dn": n_dn,
            "omega": omega,
            "g": g_val,
            "tau": tau,
            "amp": amp,
            "sigma_action_scale": sig_sc,
            "scale_clip_max": clip_max,
            "parity_equivariant": parity_eq,
            "unclip_sigma_action": unclip_sig,
            "eff_sig_hi": eff_sig_hi,
            "eff_sig_lo": eff_sig_lo,
            "base_sigma_mala": base_sigma,
            "sigma_ceiling": base_sigma * math.exp(eff_sig_hi),
            "probe_spin": probe_spin,
            "empty_well_x": empty_x,
            "a_mu_min": float(np.min(a_mu_clip)),
            "a_mu_max": float(np.max(a_mu_clip)),
            "a_mu_mean": float(np.mean(a_mu_clip)),
            "a_mu_std": float(np.std(a_mu_clip)),
            "mu_actor_range_bohr": float(np.max(mu_actor) - np.min(mu_actor)),
            "frac_mu_saturated_pct": float(np.mean(np.abs(a_mu_clip) >= 0.99) * 100.0),
            "a_sig_raw_min": float(np.min(a_sig_raw)),
            "a_sig_raw_max": float(np.max(a_sig_raw)),
            "a_sig_raw_std": float(np.std(a_sig_raw)),
            "sigma_applied_min": float(np.min(sigma_applied)),
            "sigma_applied_max": float(np.max(sigma_applied)),
            "sigma_applied_std": float(np.std(sigma_applied)),
            "frac_sigma_clipped_pct": float(r["frac_sigma_clipped_pct"]),
            "max_fq_sensitivity_mu": float(r["max_fq_sensitivity_mu"]),
            "max_fq_sensitivity_sig": float(r["max_fq_sensitivity_sig"]),
            "mean_parity_asymmetry_mu": float(r["mean_parity_asymmetry_mu"]),
            "mean_parity_asymmetry_sig": float(r["mean_parity_asymmetry_sig"]),
        }

        pr = {
            "summary": sm,
            "x_grid": xg,
            "well_centers": well_centers,
            "same_fixed_x": same_fx,
            "opp_fixed_x": opp_fx,
            "empty_well_x": empty_x,
            "cond_prob": arr["cond_prob_norm"],
            "casscf_ref": casscf_ref,
            "fq_probe": arr["fq_probe"],
            "langevin_drift": arr["langevin_drift"],
            "a_mu_raw": arr["a_mu_raw"],
            "a_mu_clip": a_mu_clip,
            "a_sig_raw": a_sig_raw,
            "a_sig_clip": arr["a_sig_clip"],
            "mu_actor": mu_actor,
            "mu_actor_nofq": arr["mu_actor_nofq"],
            "mu_actor_parity_expected": mu_par_exp,
            "mu_total_disp": arr["mu_total_disp"],
            "sigma_applied": sigma_applied,
            "sigma_unclipped": arr["sigma_unclipped"],
            "sigma_nofq": arr["sigma_nofq"],
            "d_min_same": arr["d_min_same"],
            "d_min_opp": arr["d_min_opp"],
        }
        probes[dname] = {"probe": pr, "label": disp.replace("\n", " "), "color": col}

        if replot_individual_probes:
            stem = f"fig_actor_spatial_probe_{dname}_spin_{probe_spin}"
            saved_probes.extend(plot_spatial_probe_figure(pr, probe_out_dir, stem, plot_args))

    sys_meta = {
        "N": N,
        "n_up": n_up,
        "n_dn": n_dn,
        "omega": omega,
        "g": g_val,
        "sys_tag": sys_tag,
    }
    saved_overlay = plot_spatial_overlay(probes, sys_meta, target_out)
    return saved_overlay, saved_probes


def main() -> int:
    parser = argparse.ArgumentParser(description="Análisis y graficación comparativa de Benchmarks B1 y B2.")
    parser.add_argument("--sys-dir", default=None, help="Directorio del sistema con subcarpetas B1_*/B2_*.")
    parser.add_argument(
        "--replot-from-csv-dir",
        default=None,
        help="Re-grafica spatial overlay y spatial probes desde una carpeta figures_ablation con CSVs (sin .h5).",
    )
    parser.add_argument("--casscf-npz", default=None, help="Ruta opcional al archivo CASSCF .npz de referencia.")
    parser.add_argument("--out-dir", default=None, help="Directorio de salida para CSV y figuras.")
    parser.add_argument("--probe-spin", default="down", choices=["up", "down"])
    args = parser.parse_args()

    if args.replot_from_csv_dir:
        replot_spatial_from_csv_dir(
            args.replot_from_csv_dir,
            out_dir=args.out_dir,
            casscf_npz=args.casscf_npz,
            probe_spin=args.probe_spin,
            replot_individual_probes=True,
        )
        return 0

    if not args.sys_dir:
        parser.error("Debe especificar --sys-dir o --replot-from-csv-dir.")

    out_dir = args.out_dir or os.path.join(args.sys_dir, "figures_ablation")
    os.makedirs(out_dir, exist_ok=True)

    rows, probes, sys_meta = analyze_system_ablation(args.sys_dir, casscf_npz=args.casscf_npz, probe_spin=args.probe_spin)

    csv_out = os.path.join(out_dir, f"ablation_summary_{sys_meta['sys_tag']}.csv")
    fieldnames = [k for k in rows[0].keys() if k not in ("label_multiline", "color")]
    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"  [+] Tabla resumen guardada: {csv_out}")

    plot_metrics_comparison(rows, sys_meta, out_dir)
    plot_spatial_overlay(probes, sys_meta, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
