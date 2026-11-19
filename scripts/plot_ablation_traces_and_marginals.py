#!/usr/bin/env python3
"""
scripts/plot_ablation_traces_and_marginals.py
=============================================
Genera dos figuras comparativas de trazas de caminantes y distribuciones
marginales resueltas por espín para todos los muestreadores de los Benchmarks
B1 y B2 (MALA base/grande/matched, RWMH matched, RL-MALA legado/paridad/eq+unclip):

  1. `fig_ablation_traces_and_marginals_<SYS_TAG>.{png,pdf}`:
     Una fila por sampler con:
       - Izquierda: traza discreta x_1(t), ..., x_N(t) de 1 caminante real
         (espín up en azules, espín down en rojos) mostrando el mezclado de
         pozos de Wigner a lo largo de los primeros `--trace-steps` pasos.
       - Derecha (eje vertical compartido en x [Bohr]): distribuciones
         marginales completas rho_up(x) y rho_down(x) calculadas sobre TODA la
         producción (todas las cadenas x pasos x caminantes), superpuestas
         contra la densidad exacta CASSCF.
  2. `fig_ablation_marginals_spin_symmetry_<SYS_TAG>.{png,pdf}`:
     Comparación directa de rho_up(x), rho_down(x) y la polarización residual
     de espín Delta_rho(x) = rho_up(x) - rho_down(x) (idealmente 0 por
     ergodicidad exacta) entre todos los muestreadores.

Uso en Ayler o local:
  python3 scripts/plot_ablation_traces_and_marginals.py \
      --sys-dir results_actor_ablation_ayler/N4_w0.1_g2.0 \
      --casscf-npz CASSCF_csv/casscf_statecas_n4_g2.0_omega0.1_soft0.1_b256_c0-a4_det6.npz \
      --trace-steps 4000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_vmc.config import Experiment1DConfig, PostProcessConfig
from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.postprocess.diagnostics import summarize_bundle_diagnostics
from scripts.analyze_flips_vs_g import compute_flip_statistics, load_bundle_walkers_and_meta
from scripts.plot_flips_system_detailed import _compute_exact_casscf_density_np
from scripts.plot_actor_ablation_comparison import MODEL_ORDER, _compute_spin_kl_from_walkers


def _select_representative_walker(w_chain: np.ndarray, n_up: int) -> int:
    """
    Elige un caminante representativo (cercano a la mediana de flips totales en la
    cadena) evitando caminantes fugados o atípicos.
    w_chain: shape (T, B, N)
    """
    T, B, N = w_chain.shape
    # Filtrar caminantes sanos (sin coordenadas extremas)
    max_abs = np.max(np.abs(w_chain), axis=(0, 2))
    med_max = float(np.median(max_abs))
    healthy = np.where(max_abs <= max(med_max * 1.6, 25.0))[0]
    if len(healthy) == 0:
        return 0

    # Contar cambios de orden entre electrones up y down por caminante
    flips_per_w = np.zeros(len(healthy), dtype=np.int64)
    for k, b in enumerate(healthy):
        wb = w_chain[:, b, :]
        # Número de cambios de signo en x_i - x_j para todos los pares
        cnt = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = wb[:, i] - wb[:, j]
                cnt += int(np.sum(d[:-1] * d[1:] < 0.0))
        flips_per_w[k] = cnt

    med_flips = float(np.median(flips_per_w))
    best_local = int(np.argmin(np.abs(flips_per_w - med_flips)))
    return int(healthy[best_local])


def load_all_sampler_traces(
    sys_dir: str,
    casscf_npz: Optional[str] = None,
    chain_idx: int = 0,
    walker_idx: Optional[int] = None,
    trace_steps: int = 4000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
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
        raise FileNotFoundError(f"No se encontraron archivos run_bundle.h5 en {sys_dir}")

    first_bundle = load_run_bundle_h5(os.path.join(sys_dir, ordered_specs[0][0], "run_bundle.h5"))
    first_cfg = Experiment1DConfig.from_dict(first_bundle["config"])
    if casscf_npz is None:
        casscf_npz = first_cfg.ansatz.casscf_npz_path or first_cfg.io.reference_casscf_npz

    pp_cfg = PostProcessConfig()
    items: List[Dict[str, Any]] = []
    global_x_max = 14.0

    # Primera pasada para x_max común
    raw_cache = []
    for dname, disp, col, grp in ordered_specs:
        h5p = os.path.join(sys_dir, dname, "run_bundle.h5")
        print(f"  [*] Cargando trazas y marginales de {dname}...")
        bundle = load_run_bundle_h5(h5p)
        cfg = Experiment1DConfig.from_dict(bundle["config"])
        binfo = load_bundle_walkers_and_meta(h5p)
        w = binfo["walkers"]  # (C, T, B, N)
        p998 = float(np.percentile(np.abs(w), 99.8))
        if np.isfinite(p998) and p998 < 50.0:
            global_x_max = max(global_x_max, math.ceil(p998 + 2.5))
        raw_cache.append((dname, disp, col, grp, h5p, bundle, cfg, binfo))

    casscf_ref = (
        _compute_exact_casscf_density_np(casscf_npz, x_max=global_x_max + 2.0, n_grid=400)
        if (casscf_npz and os.path.isfile(casscf_npz))
        else None
    )

    n_bins = 180
    edges = np.linspace(-global_x_max, global_x_max, n_bins + 1)
    x_centers = 0.5 * (edges[:-1] + edges[1:])

    for dname, disp, col, grp, h5p, bundle, cfg, binfo in raw_cache:
        w = binfo["walkers"]  # (C, T, B, N)
        C, T, B, N = w.shape
        n_up = int(binfo["n_up"])
        c_sel = min(max(0, chain_idx), C - 1)
        w_chain = w[c_sel]  # (T, B, N)
        b_sel = (
            min(max(0, walker_idx), B - 1)
            if walker_idx is not None
            else _select_representative_walker(w_chain, n_up)
        )
        t_view = min(T, max(100, int(trace_steps)))
        trace = w_chain[:t_view, b_sel, :].copy()  # (t_view, N)

        # Marginales exactas sobre TODA la producción (C x T x B)
        w_up = w[..., :n_up].reshape(-1)
        w_dn = w[..., n_up:].reshape(-1)
        rho_up, _ = np.histogram(w_up, bins=edges, density=True)
        rho_dn, _ = np.histogram(w_dn, bins=edges, density=True)

        diag = summarize_bundle_diagnostics(bundle, pp_cfg)["summary"]
        is_neural = cfg.proposal.proposal_type in ("neural_deepset", "neural_flat")
        flips = compute_flip_statistics(
            w,
            n_up=n_up,
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
        kl_spin, _, _ = _compute_spin_kl_from_walkers(w, n_up, casscf_ref)

        if cfg.proposal.proposal_type == "metropolis":
            sig_eff = float(cfg.proposal.rw_sigma)
        elif not is_neural:
            sig_eff = math.sqrt(2.0 * float(cfg.proposal.tau))
        else:
            s_stats = bundle.get("production", {}).get("sampler_stats", {})
            sig_eff = (
                float(np.mean(np.asarray(s_stats["actor_sigma"], dtype=np.float64)))
                if "actor_sigma" in s_stats
                else float("nan")
            )

        items.append(
            {
                "model_key": dname,
                "label": disp.replace("\n", " "),
                "color": col,
                "group": grp,
                "trace": trace,
                "walker_idx": b_sel,
                "chain_idx": c_sel,
                "total_steps": T,
                "n_chains": C,
                "n_batch_saved": B,
                "x_centers": x_centers,
                "rho_up": rho_up,
                "rho_dn": rho_dn,
                "sigma_eff": sig_eff,
                "acceptance": float(diag.get("acceptance", float("nan"))),
                "esjd": float(diag.get("ESJD", float("nan"))),
                "tau_int_E": float(diag.get("tau_int_E", float("nan"))),
                "kl_spin": kl_spin,
                "flips_total_1k": float(flips.get("hyst_flips_total_mean", 0.0)),
                "flips_opp_1k": float(flips.get("hyst_flips_opp_mean", 0.0)),
                "flips_same_1k": float(flips.get("hyst_flips_same_mean", 0.0)),
            }
        )

    sys_meta = {
        "N": int(first_cfg.system.n_elec),
        "n_up": int(first_cfg.system.n_up),
        "n_dn": int(first_cfg.system.n_dn),
        "omega": float(first_cfg.system.omega),
        "g": float(first_cfg.system.interaction_strength),
        "x_max": global_x_max,
        "sys_tag": f"N{first_cfg.system.n_elec}_w{first_cfg.system.omega:g}_g{first_cfg.system.interaction_strength:g}",
    }
    return items, sys_meta, casscf_ref


def plot_traces_and_marginals_grid(
    items: List[Dict[str, Any]],
    sys_meta: Dict[str, Any],
    casscf_ref: Optional[Dict[str, np.ndarray]],
    out_dir: str,
    fmt: str = "both",
    dpi: int = 300,
) -> List[str]:
    n_models = len(items)
    N = sys_meta["N"]
    n_up = sys_meta["n_up"]
    x_max = sys_meta["x_max"]

    plt.rcParams.update({"font.size": 8.6, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15.2, max(2.25 * n_models + 1.5, 8.0)), dpi=220)
    gs = GridSpec(
        n_models,
        2,
        figure=fig,
        width_ratios=[3.1, 1.05],
        wspace=0.06,
        hspace=0.42,
    )

    up_colors = ["#08519c", "#3182bd", "#6baed6", "#2171b5", "#08306b"]
    dn_colors = ["#a50f15", "#de2d26", "#fb6a4a", "#cb181d", "#67000d"]

    # Calcular límite horizontal común de densidad para todas las marginales
    max_rho = 0.15
    for it in items:
        max_rho = max(max_rho, float(np.max(it["rho_up"])), float(np.max(it["rho_dn"])))
    if casscf_ref is not None:
        max_rho = max(max_rho, float(np.max(casscf_ref["p_up"])))
    max_rho *= 1.15

    # Picos de Wigner de referencia desde CASSCF (para líneas guía suaves)
    well_peaks: List[float] = []
    if casscf_ref is not None:
        rho_c = casscf_ref["rho_total"]
        xg_c = casscf_ref["x"]
        for k in range(1, len(rho_c) - 1):
            if rho_c[k] > rho_c[k - 1] and rho_c[k] >= rho_c[k + 1] and rho_c[k] > 0.15 * np.max(rho_c):
                well_peaks.append(float(xg_c[k]))

    for idx, it in enumerate(items):
        ax_tr = fig.add_subplot(gs[idx, 0])
        ax_mg = fig.add_subplot(gs[idx, 1], sharey=ax_tr)

        tr = it["trace"]
        t_axis = np.arange(tr.shape[0])

        # Guías horizontales tenues de los pozos de Wigner
        for wp in well_peaks:
            ax_tr.axhline(wp, color="#888888", ls=":", lw=0.65, alpha=0.35, zorder=1)
            ax_mg.axhline(wp, color="#888888", ls=":", lw=0.65, alpha=0.35, zorder=1)

        # Trazas de electrones (up en azul, down en rojo)
        for e in range(n_up):
            ax_tr.plot(t_axis, tr[:, e], color=up_colors[e % len(up_colors)], lw=0.85, alpha=0.88, zorder=3)
        for e in range(n_up, N):
            ax_tr.plot(t_axis, tr[:, e], color=dn_colors[(e - n_up) % len(dn_colors)], lw=0.85, alpha=0.88, zorder=3)

        ax_tr.set_xlim(0, max(1, tr.shape[0] - 1))
        ax_tr.set_ylim(-x_max, x_max)
        ax_tr.set_ylabel(r"$x$ [Bohr]", fontsize=8.5, weight="bold")
        ax_tr.grid(True, axis="x", alpha=0.22, ls="--")
        if idx < n_models - 1:
            ax_tr.tick_params(labelbottom=False)
        else:
            ax_tr.set_xlabel("Paso MCMC $t$ (caminante individual representativo)", fontsize=9.2, weight="bold")

        title_str = (
            f"{it['label']}  |  "
            f"$\\sigma={it['sigma_eff']:.2f}$, Acc$={it['acceptance']*100:.1f}\\%$, "
            f"Flips$={it['flips_total_1k']:.2f}$/1k ($\\uparrow\\downarrow={it['flips_opp_1k']:.2f}$, Pauli$={it['flips_same_1k']:.2f}$), "
            f"$D_{{\\mathrm{{KL}}}}^{{\\mathrm{{spin}}}}={it['kl_spin']:.4f}$, $\\tau_{{\\mathrm{{int}}}}(E)={it['tau_int_E']:.1f}$"
        )
        ax_tr.set_title(title_str, loc="left", fontsize=8.8, weight="bold", color="#111111", pad=4)

        # Panel derecho: distribuciones marginales de espín (toda la producción)
        xc = it["x_centers"]
        ax_mg.fill_betweenx(xc, 0.0, it["rho_up"], color="#1f77b4", alpha=0.25)
        ax_mg.plot(it["rho_up"], xc, color="#1f77b4", lw=1.6)
        ax_mg.fill_betweenx(xc, 0.0, it["rho_dn"], color="#d62728", alpha=0.22)
        ax_mg.plot(it["rho_dn"], xc, color="#d62728", lw=1.6, ls="-.")
        if casscf_ref is not None:
            ax_mg.plot(casscf_ref["p_up"], casscf_ref["x"], color="#111111", lw=1.3, ls="--", alpha=0.90)

        ax_mg.set_xlim(0.0, max_rho)
        ax_mg.tick_params(labelleft=False)
        ax_mg.grid(True, axis="x", alpha=0.22, ls="--")
        if idx == 0:
            ax_mg.set_title(
                f"Marginal espín ({it['n_chains']}×{it['total_steps']:,} pasos)",
                fontsize=8.6,
                weight="bold",
                pad=4,
            )
        if idx < n_models - 1:
            ax_mg.tick_params(labelbottom=False)
        else:
            ax_mg.set_xlabel(r"$\rho_\sigma(x)$", fontsize=9.2, weight="bold")

    # Leyenda única limpia AFUERA arriba de toda la figura
    legend_handles = [
        Line2D([0], [0], color="#1f77b4", lw=2.0, label=r"Electrones espín $\uparrow$ (traza y $\rho_\uparrow$)"),
        Line2D([0], [0], color="#d62728", lw=2.0, ls="-.", label=r"Electrones espín $\downarrow$ (traza y $\rho_\downarrow$)"),
        Line2D([0], [0], color="#111111", lw=1.5, ls="--", label=r"Densidad exacta CASSCF $\frac{1}{N}\rho_{\mathrm{CASSCF}}(x)$"),
        Line2D([0], [0], color="#888888", lw=1.0, ls=":", label="Centros de pozos de Wigner"),
    ]
    fig.suptitle(
        f"Comparativa de Trazas de 1 Caminante y Marginales de Espín por Sampler — "
        f"$N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$ (inicio polarizado `spin_ordered_offset`)",
        fontsize=11.8,
        weight="bold",
        y=0.992,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.968),
        ncol=4,
        fontsize=8.2,
        framealpha=0.85,
        borderpad=0.35,
    )
    fig.subplots_adjust(top=0.925, bottom=0.045, left=0.06, right=0.985)

    saved: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_ablation_traces_and_marginals_{sys_meta['sys_tag']}"
    exts = ["png", "pdf"] if fmt == "both" else [fmt]
    for ext in exts:
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(p)
        print(f"  [+] Figura guardada: {p}")
    plt.close(fig)
    return saved


def plot_marginals_spin_symmetry(
    items: List[Dict[str, Any]],
    sys_meta: Dict[str, Any],
    casscf_ref: Optional[Dict[str, np.ndarray]],
    out_dir: str,
    fmt: str = "both",
    dpi: int = 300,
) -> List[str]:
    plt.rcParams.update({"font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 11.4), dpi=220, sharex=True)
    ax_u, ax_d, ax_diff = axes

    if casscf_ref is not None:
        ax_u.plot(casscf_ref["x"], casscf_ref["p_up"], color="#111111", lw=2.2, ls="--", label="Exacto CASSCF", zorder=10)
        ax_d.plot(casscf_ref["x"], casscf_ref["p_down"], color="#111111", lw=2.2, ls="--", zorder=10)
    ax_diff.axhline(0.0, color="#111111", lw=1.4, ls="--", alpha=0.8)

    for it in items:
        xc = it["x_centers"]
        col = it["color"]
        lbl = f"{it['label']} ($D_{{\\mathrm{{KL}}}}={it['kl_spin']:.4f}$)"
        lw = 2.0 if it["model_key"] in ("B1_RWMH_matched_sig1.2156", "B2_RL_MALA_eq_unclip_amp0.80_sig1.8") else 1.4
        ax_u.plot(xc, it["rho_up"], color=col, lw=lw, alpha=0.88, label=lbl)
        ax_d.plot(xc, it["rho_dn"], color=col, lw=lw, alpha=0.88)
        ax_diff.plot(xc, it["rho_up"] - it["rho_dn"], color=col, lw=lw, alpha=0.88)

    ax_u.set_xlim(-sys_meta["x_max"], sys_meta["x_max"])
    ax_u.set_ylabel(r"$\rho_\uparrow(x)$", weight="bold")
    ax_u.set_title(r"(a) Densidad Marginal de Espín Arriba $\rho_\uparrow(x)$ vs CASSCF Exacto", weight="bold", pad=6)
    ax_u.grid(True, alpha=0.25, ls="--")

    ax_d.set_ylabel(r"$\rho_\downarrow(x)$", weight="bold")
    ax_d.set_title(r"(b) Densidad Marginal de Espín Abajo $\rho_\downarrow(x)$ vs CASSCF Exacto", weight="bold", pad=6)
    ax_d.grid(True, alpha=0.25, ls="--")

    ax_diff.set_ylabel(r"$\rho_\uparrow(x) - \rho_\downarrow(x)$", weight="bold")
    ax_diff.set_xlabel(r"Posición $x$ [Bohr]", weight="bold")
    ax_diff.set_title(r"(c) Polarización Residual de Espín $\Delta\rho(x) = \rho_\uparrow(x) - \rho_\downarrow(x)$ (Ergodicidad Exacta $\equiv 0$)", weight="bold", pad=6)
    ax_diff.grid(True, alpha=0.25, ls="--")

    h, l = ax_u.get_legend_handles_labels()
    fig.suptitle(
        f"Convergencia de Densidades Marginales y Simetría de Espín — $N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$",
        fontsize=12.0,
        weight="bold",
        y=0.992,
    )
    fig.legend(
        h,
        l,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        fontsize=7.6,
        framealpha=0.82,
        borderpad=0.35,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.885], h_pad=1.6)

    saved: List[str] = []
    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_ablation_marginals_spin_symmetry_{sys_meta['sys_tag']}"
    exts = ["png", "pdf"] if fmt == "both" else [fmt]
    for ext in exts:
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(p)
        print(f"  [+] Figura guardada: {p}")
    plt.close(fig)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara trazas y marginales de espín entre todos los samplers de B1 y B2.")
    parser.add_argument("--sys-dir", required=True, help="Directorio del sistema (con subcarpetas B1_*/B2_*).")
    parser.add_argument("--casscf-npz", default=None, help="Ruta opcional al archivo .npz de CASSCF.")
    parser.add_argument("--out-dir", default=None, help="Directorio de salida (por defecto <sys-dir>/figures_ablation).")
    parser.add_argument("--trace-steps", type=int, default=4000, help="Número de pasos a mostrar en la traza izquierda.")
    parser.add_argument("--chain-idx", type=int, default=0, help="Índice de cadena para la traza.")
    parser.add_argument("--walker-idx", type=int, default=None, help="Índice de caminante (None = representativo mediano).")
    parser.add_argument("--format", default="both", choices=["png", "pdf", "both"])
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.sys_dir, "figures_ablation")
    items, sys_meta, casscf_ref = load_all_sampler_traces(
        args.sys_dir,
        casscf_npz=args.casscf_npz,
        chain_idx=args.chain_idx,
        walker_idx=args.walker_idx,
        trace_steps=args.trace_steps,
    )
    plot_traces_and_marginals_grid(items, sys_meta, casscf_ref, out_dir, fmt=args.format, dpi=args.dpi)
    plot_marginals_spin_symmetry(items, sys_meta, casscf_ref, out_dir, fmt=args.format, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
