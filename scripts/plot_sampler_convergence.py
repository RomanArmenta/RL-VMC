#!/usr/bin/env python3
"""
scripts/plot_sampler_convergence.py
===================================
Cuantifica y grafica qué tan rápido converge cada sampler (Benchmarks B1 y B2)
tanto a la energía de equilibrio como al muestreo real del estado base |Psi_0|^2.

Soporta dos análisis complementarios (ejecutables juntos con `--mode both`):

1. **Convergencia desde estado frío (`--mode cold_start`, burn_in = 0):**
   Carga los checkpoints congelados (`ansatz_best` y `actor_final`) de cada
   `run_bundle.h5`, inicializa TODOS los samplers exactamente en el MISMO ensamble
   fuera de equilibrio X_0 (por defecto `spin_ordered_offset`, donde E(0) >> E_0
   y existe polarización máxima de espín) y corre M caminantes paralelos desde t=0
   SIN descartar burn-in. Mide:
     (a) Relajación instantánea de energía <E_L(X_t)> -> E_CASSCF.
     (b) Error instantáneo |<E_L(X_t)> - E_CASSCF| (escala log-log).
     (c) Relajación espacial del centro de masa |<x_bar(t)>| -> 0 (llegada a pozos Wigner).
     (d) Relajación del dipolo de espín |<x_up(t) - x_dn(t)>| -> 0 (convergencia real
         al sector antisimétrico del estado base, revelando si la energía ya equilibró
         mientras los espines siguen congelados).

2. **Convergencia acumulada en producción (`--mode production`, directo de los .h5):**
   Sin re-simular nada, lee `production/local_energy` y `production/walkers` ya guardados
   en cada `run_bundle.h5` y grafica como función del número de pasos T:
     (a) Energía media acumulada E_bar(T) vs E_CASSCF (con banda ±1 mHa).
     (b) Error acumulado |E_bar(T) - E_CASSCF| en escala log-log con guía 1/sqrt(T).
     (c) Divergencia KL acumulada de espín D_KL^spin(T) vs T (log-log): muestra
         directamente el cruce entre régimen dominado por sesgo (T corto) y régimen
         dominado por varianza/aceptación (T largo).
     (d) Divergencia KL acumulada de densidad total D_KL^tot(T) vs T (log-log).

Ejemplo de uso (Ayler o local):
  python3 scripts/plot_sampler_convergence.py \\
    --sys-dir results_actor_ablation_ayler/N4_w0.1_g2.0 \\
    --casscf-npz CASSCF_csv/casscf_statecas_n4_g2.0_omega0.1_soft0.1_b256_c0-a4_det6.npz \\
    --mode both --cold-walkers 128 --cold-steps 2500 --init-mode spin_ordered_offset
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl_vmc.config import Experiment1DConfig
from rl_vmc.core.io_bundle import load_run_bundle_h5
from scripts.plot_actor_ablation_comparison import (
    MODEL_ORDER,
    _get_casscf_energy,
    _kl_divergence,
)
from scripts.plot_flips_system_detailed import (
    _compute_exact_casscf_density_np,
    _find_casscf_npz_for_system,
)


def _discover_models(sys_dir: str) -> List[Tuple[str, str, str, str]]:
    known = {m[0] for m in MODEL_ORDER}
    specs: List[Tuple[str, str, str, str]] = []
    for dname, disp, col, grp in MODEL_ORDER:
        h5p = os.path.join(sys_dir, dname, "run_bundle.h5")
        if os.path.isfile(h5p):
            specs.append((dname, disp.replace("\n", " "), col, grp))
    for entry in sorted(os.listdir(sys_dir)):
        if entry in known:
            continue
        h5p = os.path.join(sys_dir, entry, "run_bundle.h5")
        if os.path.isfile(h5p):
            specs.append((entry, entry, "#377eb8", "Extra"))
    if not specs:
        raise FileNotFoundError(f"No se encontraron bundles run_bundle.h5 en {sys_dir}")
    return specs


def _resolve_casscf_npz(
    first_cfg: Experiment1DConfig,
    casscf_npz_override: Optional[str],
) -> Optional[str]:
    if casscf_npz_override and os.path.isfile(casscf_npz_override):
        return casscf_npz_override
    for cand in (first_cfg.ansatz.casscf_npz_path, first_cfg.io.reference_casscf_npz):
        if cand and os.path.isfile(cand):
            return cand
        if cand and os.path.isfile(os.path.join(PROJECT_ROOT, cand)):
            return os.path.join(PROJECT_ROOT, cand)
    return _find_casscf_npz_for_system(
        [PROJECT_ROOT],
        int(first_cfg.system.n_elec),
        float(first_cfg.system.omega),
        float(first_cfg.system.interaction_strength),
    )


def _build_energy_eval_steps(n_steps: int) -> np.ndarray:
    """Malla densa al inicio (t=0..60 paso a paso) y log-espaciada/regular hasta n_steps."""
    pts = list(range(0, min(61, n_steps + 1)))
    if n_steps > 60:
        pts.extend(range(65, min(201, n_steps + 1), 5))
    if n_steps > 200:
        pts.extend(range(220, min(601, n_steps + 1), 20))
    if n_steps > 600:
        pts.extend(range(650, n_steps + 1, max(50, n_steps // 50)))
    if pts[-1] != n_steps:
        pts.append(n_steps)
    return np.asarray(sorted(set(pts)), dtype=np.int64)


def _first_sustained_crossing(
    steps: np.ndarray,
    values: np.ndarray,
    threshold: float,
    sustain_count: int = 3,
) -> float:
    """Primer paso donde `values <= threshold` y se mantiene así al menos `sustain_count` muestras."""
    n = len(values)
    for i in range(n):
        end = min(n, i + sustain_count)
        if np.all(np.isfinite(values[i:end])) and np.all(values[i:end] <= threshold):
            return float(steps[i])
    return float("nan")


# ==============================================================================
# 1. ANÁLISIS DESDE ESTADO FRÍO (t = 0, burn_in = 0)
# ==============================================================================

def run_cold_start_benchmark(
    sys_dir: str,
    specs: List[Tuple[str, str, str, str]],
    casscf_npz: Optional[str],
    n_walkers: int = 128,
    n_steps: int = 2000,
    init_mode: str = "spin_ordered_offset",
    seed: int = 2026,
    eloc_chunk_size: int = 512,
    extra_rwmh_sigmas: Optional[List[float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import jax
    import jax.numpy as jnp
    from rl_vmc.ansatze.ansatz_1d import build_ansatz_1d
    from rl_vmc.physics.harmonic_1d import Harmonic1DHamiltonian
    from rl_vmc.proposals.proposals_1d import build_proposal_1d
    from rl_vmc.samplers.mh_sampler_1d import MHSampler1D

    first_bundle = load_run_bundle_h5(os.path.join(sys_dir, specs[0][0], "run_bundle.h5"))
    first_cfg = Experiment1DConfig.from_dict(first_bundle["config"])
    if casscf_npz:
        first_cfg.ansatz.casscf_npz_path = casscf_npz
    ansatz = build_ansatz_1d(first_cfg)
    hamiltonian = Harmonic1DHamiltonian(first_cfg.system)
    params_psi = first_bundle["checkpoints"].get("ansatz_best") or first_bundle["checkpoints"].get("ansatz_final")
    if params_psi is None:
        params_psi = ansatz.init(jax.random.PRNGKey(0), jnp.zeros((1, first_cfg.system.n_elec, 1), dtype=jnp.float64))

    N = int(first_cfg.system.n_elec)
    n_up = int(first_cfg.system.n_up)
    e_ref = _get_casscf_energy(casscf_npz)
    casscf_ref = _compute_exact_casscf_density_np(casscf_npz, x_max=20.0, n_grid=400) if casscf_npz else None

    # Crear estado inicial COMÚN idéntico para todos los samplers (comparación 100% justa)
    ref_prop = build_proposal_1d(first_cfg)
    ref_sampler = MHSampler1D(first_cfg, ansatz, ref_prop)
    init_state_shared = ref_sampler.init_state(
        jax.random.PRNGKey(seed),
        params_psi,
        batch_size=n_walkers,
        init_mode=init_mode,
    )

    # Evaluar observables exactos en t = 0
    @jax.jit
    def eval_eloc_batch(w_batch: jax.Array) -> jax.Array:
        e_l, _, _ = hamiltonian.local_energy_and_components(
            ansatz.apply, params_psi, w_batch, chunk_size=eloc_chunk_size
        )
        return e_l

    e0_arr = np.asarray(eval_eloc_batch(init_state_shared.walkers), dtype=np.float64)
    w0_np = np.asarray(init_state_shared.walkers[..., 0], dtype=np.float64)
    com0 = float(np.mean(w0_np))
    dip0 = float(np.mean(np.mean(w0_np[:, :n_up], axis=1) - np.mean(w0_np[:, n_up:], axis=1)))
    print(
        f"  [Cold-Start t=0 | {init_mode} | M={n_walkers}] "
        f"<E_L(0)> = {np.mean(e0_arr):.4f} Ha (E_CASSCF = {e_ref:.4f} Ha) | "
        f"<x_bar(0)> = {com0:+.3f} Bohr | dipolo_spin(0) = {dip0:+.3f} Bohr"
    )

    eval_steps = _build_energy_eval_steps(n_steps)
    # Intervalos entre snapshots consecutivos donde guardamos los caminantes para E_L y KL
    seg_lengths = np.diff(eval_steps)

    # Construir lista de trabajos: bundles descubiertos + controles RWMH opcionales a sigmas extras
    run_items: List[Tuple[str, str, str, str, Optional[str], Optional[float]]] = [
        (dname, disp, col, grp, os.path.join(sys_dir, dname, "run_bundle.h5"), None)
        for dname, disp, col, grp in specs
    ]
    extra_cols = ["#dd1c77", "#980043", "#7a0177", "#ae017e"]
    for idx_ex, ex_sig in enumerate(extra_rwmh_sigmas or []):
        run_items.append((
            f"CTRL_RWMH_sig{ex_sig:.3f}",
            f"RWMH control (deriva$=0, \\sigma={ex_sig:.2f}$)",
            extra_cols[idx_ex % len(extra_cols)],
            "Control",
            None,
            float(ex_sig),
        ))

    results: List[Dict[str, Any]] = []
    for dname, disp, col, grp, h5p, override_rw_sigma in run_items:
        print(f"  [*] Cold-start ({n_steps} pasos, M={n_walkers}) para {dname}...")
        if h5p is not None:
            bundle = load_run_bundle_h5(h5p)
            cfg = Experiment1DConfig.from_dict(bundle["config"])
            actor_params = bundle.get("checkpoints", {}).get("actor_final")
        else:
            cfg = Experiment1DConfig.from_dict(first_bundle["config"])
            cfg.proposal.proposal_type = "metropolis"
            cfg.proposal.rw_sigma = float(override_rw_sigma)
            actor_params = None

        if casscf_npz:
            cfg.ansatz.casscf_npz_path = casscf_npz
        proposal = build_proposal_1d(cfg)
        sampler = MHSampler1D(cfg, ansatz, proposal)

        use_actor = bool(proposal.is_neural and actor_params is not None)
        if actor_params is None:
            actor_params = {}

        # Función JIT que avanza exactamente L pasos devolviendo series por paso (com, dip, acc)
        # Compilamos por longitud de segmento única (son solo ~4 longitudes distintas: 1, 5, 20, 50)
        seg_cache: Dict[int, Any] = {}

        def _get_seg_fn(length: int):
            if length not in seg_cache:
                @jax.jit
                def _run_seg(k_in, st_in):
                    def body(carry, _):
                        k_c, st = carry
                        k_nx, k_st = jax.random.split(k_c)
                        st_nx, stats, _ = sampler.step(
                            k_st, st, params_psi, actor_params, use_actor=use_actor, exploration_std=0.0
                        )
                        xw = st_nx.walkers[..., 0]
                        com_t = jnp.mean(xw)
                        dip_t = jnp.mean(jnp.mean(xw[:, :n_up], axis=1) - jnp.mean(xw[:, n_up:], axis=1))
                        std_t = jnp.std(xw)
                        acc_t = jnp.mean(stats.acceptance)
                        return (k_nx, st_nx), (com_t, dip_t, std_t, acc_t)
                    return jax.lax.scan(body, (k_in, st_in), None, length=length)
                seg_cache[length] = _run_seg
            return seg_cache[length]

        curr_key = jax.random.PRNGKey(seed + 777)
        curr_state = init_state_shared

        com_series = [com0]
        dip_series = [dip0]
        std_series = [float(np.std(w0_np))]
        acc_series = [np.nan]

        e_mean_pts = [float(np.mean(e0_arr))]
        e_se_pts = [float(np.std(e0_arr) / math.sqrt(n_walkers))]
        kl_spin_inst_pts = []
        kl_tot_inst_pts = []

        if casscf_ref is not None:
            xg = np.asarray(casscf_ref["x"], dtype=np.float64)
            dx = float(xg[1] - xg[0])
            edges = np.concatenate([[xg[0] - 0.5 * dx], 0.5 * (xg[:-1] + xg[1:]), [xg[-1] + 0.5 * dx]])
            p_up_ref = np.asarray(casscf_ref["p_up"], dtype=np.float64)
            p_dn_ref = np.asarray(casscf_ref["p_down"], dtype=np.float64)
            p_tot_ref = 0.5 * (p_up_ref + p_dn_ref)

            def _inst_kls(w_arr_2d: np.ndarray) -> Tuple[float, float]:
                h_up, _ = np.histogram(w_arr_2d[:, :n_up].reshape(-1), bins=edges, density=True)
                h_dn, _ = np.histogram(w_arr_2d[:, n_up:].reshape(-1), bins=edges, density=True)
                h_tot, _ = np.histogram(w_arr_2d.reshape(-1), bins=edges, density=True)
                kl_sp = 0.5 * (_kl_divergence(h_up, p_up_ref, dx) + _kl_divergence(h_dn, p_dn_ref, dx))
                kl_tt = _kl_divergence(h_tot, p_tot_ref, dx)
                return kl_sp, kl_tt

            k_sp0, k_tt0 = _inst_kls(w0_np)
            kl_spin_inst_pts.append(k_sp0)
            kl_tot_inst_pts.append(k_tt0)

        for seg_len in seg_lengths:
            fn = _get_seg_fn(int(seg_len))
            (curr_key, curr_state), (c_s, d_s, s_s, a_s) = fn(curr_key, curr_state)
            com_series.extend(np.asarray(c_s, dtype=np.float64).tolist())
            dip_series.extend(np.asarray(d_s, dtype=np.float64).tolist())
            std_series.extend(np.asarray(s_s, dtype=np.float64).tolist())
            acc_series.extend(np.asarray(a_s, dtype=np.float64).tolist())

            el_now = np.asarray(eval_eloc_batch(curr_state.walkers), dtype=np.float64)
            e_mean_pts.append(float(np.mean(el_now)))
            e_se_pts.append(float(np.std(el_now) / math.sqrt(n_walkers)))
            if casscf_ref is not None:
                k_sp, k_tt = _inst_kls(np.asarray(curr_state.walkers[..., 0], dtype=np.float64))
                kl_spin_inst_pts.append(k_sp)
                kl_tot_inst_pts.append(k_tt)

        steps_full = np.arange(n_steps + 1, dtype=np.int64)
        com_arr = np.asarray(com_series, dtype=np.float64)
        dip_arr = np.asarray(dip_series, dtype=np.float64)
        std_arr = np.asarray(std_series, dtype=np.float64)
        acc_arr = np.asarray(acc_series, dtype=np.float64)
        e_mean_arr = np.asarray(e_mean_pts, dtype=np.float64)
        e_se_arr = np.asarray(e_se_pts, dtype=np.float64)
        e_err_arr = np.abs(e_mean_arr - e_ref)

        # Métricas de tiempo de equilibrio (pasos hasta cruzar umbrales sostenidos)
        t_eq_e_50mha = _first_sustained_crossing(eval_steps, e_err_arr, 0.050, sustain_count=3)
        t_eq_e_20mha = _first_sustained_crossing(eval_steps, e_err_arr, 0.020, sustain_count=3)
        t_eq_com_0p5 = _first_sustained_crossing(steps_full, np.abs(com_arr), 0.50, sustain_count=10)
        # El dipolo de espín primero se expande cuando los pozos se separan (en Wigner |dip_peak| > |dip0|)
        dip_abs = np.abs(dip_arr)
        dip_peak = float(np.max(dip_abs[: min(len(dip_abs), 300)]))
        t_peak_idx = int(np.argmax(dip_abs[: min(len(dip_abs), 300)]))
        t_spin_half = _first_sustained_crossing(steps_full[t_peak_idx:], dip_abs[t_peak_idx:], 0.50 * dip_peak, sustain_count=15)
        t_spin_10pct = _first_sustained_crossing(steps_full[t_peak_idx:], dip_abs[t_peak_idx:], 0.10 * dip_peak, sustain_count=15)

        results.append({
            "model_key": dname,
            "label": disp,
            "color": col,
            "group": grp,
            "is_neural": use_actor,
            "steps_full": steps_full,
            "eval_steps": eval_steps,
            "e_mean": e_mean_arr,
            "e_se": e_se_arr,
            "e_err": e_err_arr,
            "com": com_arr,
            "dip": dip_arr,
            "std_x": std_arr,
            "acc": acc_arr,
            "kl_spin_inst": np.asarray(kl_spin_inst_pts, dtype=np.float64) if kl_spin_inst_pts else None,
            "kl_tot_inst": np.asarray(kl_tot_inst_pts, dtype=np.float64) if kl_tot_inst_pts else None,
            "acc_mean_tail": float(np.nanmean(acc_arr[max(1, n_steps // 2):])),
            "t_eq_e_50mha": t_eq_e_50mha,
            "t_eq_e_20mha": t_eq_e_20mha,
            "t_eq_com_0p5": t_eq_com_0p5,
            "dip_peak": dip_peak,
            "dip_final_abs": float(np.mean(dip_abs[max(1, int(0.85 * n_steps)):])),
            "t_spin_half": t_spin_half,
            "t_spin_10pct": t_spin_10pct,
        })

    sys_meta = {
        "N": N,
        "n_up": n_up,
        "omega": float(first_cfg.system.omega),
        "g": float(first_cfg.system.interaction_strength),
        "E_casscf": e_ref,
        "sys_tag": f"N{N}_w{first_cfg.system.omega:g}_g{first_cfg.system.interaction_strength:g}",
        "init_mode": init_mode,
        "n_walkers": n_walkers,
        "n_steps": n_steps,
    }
    return results, sys_meta


def plot_cold_start_figure(
    results: List[Dict[str, Any]],
    sys_meta: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    plt.rcParams.update({"font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.6), dpi=240)
    ax_e, ax_err, ax_com, ax_dip = axes.ravel()
    e_ref = sys_meta["E_casscf"]

    for r in results:
        col = r["color"]
        lw = 2.2 if ("RWMH" in r["model_key"] or "eq_unclip" in r["model_key"]) else 1.5
        if r["model_key"].startswith("CTRL_RWMH"):
            ls = "-."
        elif "MALA_base" in r["model_key"] or "MALA_large" in r["model_key"] or "legacy" in r["model_key"]:
            ls = "--"
        else:
            ls = "-"
        t_ev = r["eval_steps"]
        t_all = r["steps_full"]

        # (a) Energía instantánea <E_L(t)>
        ax_e.plot(t_ev + 1, r["e_mean"], color=col, lw=lw, ls=ls, label=r["label"])
        ax_e.fill_between(t_ev + 1, r["e_mean"] - r["e_se"], r["e_mean"] + r["e_se"], color=col, alpha=0.09)

        # (b) Error instantáneo |<E_L(t)> - E_CASSCF|
        ax_err.plot(t_ev + 1, np.maximum(r["e_err"], 1e-4), color=col, lw=lw, ls=ls)

        # (c) Relajación del centro de masa |<x_bar(t)>|
        ax_com.plot(t_all + 1, np.abs(r["com"]), color=col, lw=lw, ls=ls)

        # (d) Relajación del dipolo de espín |<x_up(t) - x_dn(t)>|
        ax_dip.plot(t_all + 1, np.abs(r["dip"]), color=col, lw=lw, ls=ls)

    # Decoración (a)
    if np.isfinite(e_ref):
        ax_e.axhline(e_ref, color="#111111", ls=":", lw=1.5, label=f"$E_{{\\mathrm{{CASSCF}}}} = {e_ref:.4f}\\,E_h$")
        ax_e.axhspan(e_ref - 0.05, e_ref + 0.05, color="#2ca02c", alpha=0.10)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("Paso MCMC $t + 1$ (escala log, burn-in $= 0$)", weight="bold")
    ax_e.set_ylabel(r"Energía Instantánea $\langle E_L(X_t) \rangle_{\mathrm{ens}}$ [$E_h$]", weight="bold")
    ax_e.set_title("(a) Termalización de Energía desde Estado Frío ($t=0$)", weight="bold")
    ax_e.grid(True, which="both", alpha=0.22, ls="--")

    # Decoración (b)
    ax_err.axhline(0.05, color="#2ca02c", ls=":", lw=1.2, alpha=0.8)
    ax_err.text(1.3, 0.058, r"$50\ \mathrm{mHa}$", fontsize=7.6, color="#1b7837", weight="bold")
    ax_err.axhline(0.01, color="#555555", ls="--", lw=1.0, alpha=0.7)
    ax_err.text(1.3, 0.0115, r"$10\ \mathrm{mHa}$", fontsize=7.6, color="#444444")
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Paso MCMC $t + 1$ (escala log)", weight="bold")
    ax_err.set_ylabel(r"$|\langle E_L(X_t) \rangle_{\mathrm{ens}} - E_{\mathrm{CASSCF}}|$ [$E_h$]", weight="bold")
    ax_err.set_title("(b) Decaimiento del Error Instantáneo de Energía (Log-Log)", weight="bold")
    ax_err.grid(True, which="both", alpha=0.22, ls="--")

    # Decoración (c)
    ax_com.axhline(0.5, color="#555555", ls=":", lw=1.1, alpha=0.75)
    ax_com.set_xscale("log")
    ax_com.set_xlabel("Paso MCMC $t + 1$ (escala log)", weight="bold")
    ax_com.set_ylabel(r"Centro de Masa $|\langle \bar{x}(t) \rangle|$ [Bohr]", weight="bold")
    ax_com.set_title("(c) Equilibrado Espacial de Carga (Migración a Pozos Wigner)", weight="bold")
    ax_com.grid(True, which="both", alpha=0.22, ls="--")

    # Decoración (d)
    ax_dip.axhline(0.0, color="#111111", ls=":", lw=1.2)
    ax_dip.set_xscale("log")
    ax_dip.set_xlabel("Paso MCMC $t + 1$ (escala log)", weight="bold")
    ax_dip.set_ylabel(r"Dipolo de Espín $|\langle \bar{x}_\uparrow(t) - \bar{x}_\downarrow(t) \rangle|$ [Bohr]", weight="bold")
    ax_dip.set_title("(d) Relajación de Simetría de Espín (Muestreo Real de $|\\Psi_0|^2$)", weight="bold")
    ax_dip.grid(True, which="both", alpha=0.22, ls="--")

    # Leyenda compartida arriba (cero solapamiento)
    handles, labels = ax_e.get_legend_handles_labels()
    leg_ncol = 4 if len(handles) > 9 else 3
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=leg_ncol,
        fontsize=7.6 if len(handles) > 9 else 8.0,
        framealpha=0.80,
        borderpad=0.35,
        columnspacing=0.95,
    )
    fig.suptitle(
        f"Velocidad de Termalización desde Estado Frío ({sys_meta['init_mode']}, burn-in$=0$, $M={sys_meta['n_walkers']}$ caminantes) — "
        f"$N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$",
        fontsize=12.0,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.845], h_pad=2.2, w_pad=2.0)

    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_convergence_cold_start_{sys_meta['sys_tag']}_{sys_meta['init_mode']}"
    saved = []
    for ext in ("png", "pdf"):
        out_p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
        print(f"  [+] Figura cold-start guardada: {out_p}")
    plt.close(fig)
    return saved


# ==============================================================================
# 2. ANÁLISIS DE CONVERGENCIA ACUMULADA EN PRODUCCIÓN (directo de run_bundle.h5)
# ==============================================================================

def analyze_production_convergence(
    sys_dir: str,
    specs: List[Tuple[str, str, str, str]],
    casscf_npz: Optional[str],
    n_kl_checkpoints: int = 18,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    first_bundle = load_run_bundle_h5(os.path.join(sys_dir, specs[0][0], "run_bundle.h5"))
    first_cfg = Experiment1DConfig.from_dict(first_bundle["config"])
    N = int(first_cfg.system.n_elec)
    n_up = int(first_cfg.system.n_up)
    e_ref = _get_casscf_energy(casscf_npz)
    casscf_ref = _compute_exact_casscf_density_np(casscf_npz, x_max=20.0, n_grid=400) if casscf_npz else None

    if casscf_ref is not None:
        xg = np.asarray(casscf_ref["x"], dtype=np.float64)
        dx = float(xg[1] - xg[0])
        edges = np.concatenate([[xg[0] - 0.5 * dx], 0.5 * (xg[:-1] + xg[1:]), [xg[-1] + 0.5 * dx]])
        p_up_ref = np.asarray(casscf_ref["p_up"], dtype=np.float64)
        p_dn_ref = np.asarray(casscf_ref["p_down"], dtype=np.float64)
        p_tot_ref = 0.5 * (p_up_ref + p_dn_ref)

    results: List[Dict[str, Any]] = []
    for dname, disp, col, grp in specs:
        h5p = os.path.join(sys_dir, dname, "run_bundle.h5")
        print(f"  [*] Convergencia de producción (desde .h5) para {dname}...")
        b = load_run_bundle_h5(h5p)
        e_arr = np.asarray(b["production"]["local_energy"], dtype=np.float64)  # (C, T, ...)
        if e_arr.ndim >= 3:
            e_per_chain = np.mean(e_arr, axis=tuple(range(2, e_arr.ndim)))  # (C, T)
        else:
            e_per_chain = e_arr
        n_chains, T_total = e_per_chain.shape
        t_steps = np.arange(1, T_total + 1, dtype=np.int64)

        # Media acumulada por cadena y global
        cum_per_chain = np.cumsum(e_per_chain, axis=1) / t_steps[None, :]
        cum_mean = np.mean(cum_per_chain, axis=0)
        cum_err_abs = np.abs(cum_mean - e_ref)
        # RMSE entre cadenas respecto a E_CASSCF (más suave que |mean - E_ref| de una sola realización)
        cum_rmse = np.sqrt(np.mean((cum_per_chain - e_ref) ** 2, axis=0))

        # KL acumulada de espín y total en checkpoints log-espaciados
        kl_steps = np.unique(
            np.round(np.geomspace(max(20, T_total // 200), T_total, n_kl_checkpoints)).astype(np.int64)
        )
        if kl_steps[-1] != T_total:
            kl_steps = np.append(kl_steps, T_total)

        kl_spin_vals = np.full(len(kl_steps), np.nan, dtype=np.float64)
        kl_tot_vals = np.full(len(kl_steps), np.nan, dtype=np.float64)
        if casscf_ref is not None and "walkers" in b["production"]:
            w = np.asarray(b["production"]["walkers"], dtype=np.float64)
            if w.ndim == 5:
                w = w[..., 0]  # (C, T, B, N)
            # Construir histogramas incrementales en O(T) para máxima velocidad
            cum_hist_up = np.zeros(len(edges) - 1, dtype=np.float64)
            cum_hist_dn = np.zeros(len(edges) - 1, dtype=np.float64)
            prev_t = 0
            for idx_k, tk in enumerate(kl_steps):
                chunk_w = w[:, prev_t:int(tk), :, :]
                h_u, _ = np.histogram(chunk_w[..., :n_up].reshape(-1), bins=edges, density=False)
                h_d, _ = np.histogram(chunk_w[..., n_up:].reshape(-1), bins=edges, density=False)
                cum_hist_up += h_u
                cum_hist_dn += h_d
                prev_t = int(tk)
                p_u = cum_hist_up / max(float(np.sum(cum_hist_up) * dx), 1e-12)
                p_d = cum_hist_dn / max(float(np.sum(cum_hist_dn) * dx), 1e-12)
                p_t = 0.5 * (p_u + p_d)
                kl_spin_vals[idx_k] = 0.5 * (_kl_divergence(p_u, p_up_ref, dx) + _kl_divergence(p_d, p_dn_ref, dx))
                kl_tot_vals[idx_k] = _kl_divergence(p_t, p_tot_ref, dx)

        # Tiempo para alcanzar precisión química (1 mHa) sostenida en RMSE acumulado
        t_rmse_1mha = _first_sustained_crossing(t_steps, cum_rmse, 1e-3, sustain_count=50)
        t_kl_1e2 = _first_sustained_crossing(kl_steps, kl_spin_vals, 1e-2, sustain_count=2)
        t_kl_1e3 = _first_sustained_crossing(kl_steps, kl_spin_vals, 1e-3, sustain_count=2)

        results.append({
            "model_key": dname,
            "label": disp,
            "color": col,
            "group": grp,
            "t_steps": t_steps,
            "cum_mean": cum_mean,
            "cum_err_abs": cum_err_abs,
            "cum_rmse": cum_rmse,
            "kl_steps": kl_steps,
            "kl_spin": kl_spin_vals,
            "kl_tot": kl_tot_vals,
            "t_rmse_1mha": t_rmse_1mha,
            "t_kl_1e2": t_kl_1e2,
            "t_kl_1e3": t_kl_1e3,
            "final_rmse_E": float(cum_rmse[-1]),
            "final_kl_spin": float(kl_spin_vals[-1]),
            "final_kl_tot": float(kl_tot_vals[-1]),
            "n_chains": n_chains,
            "T_total": T_total,
            "eval_burn_in": int(b["config"].get("eval", {}).get("eval_burn_in", 0)),
        })

    sys_meta = {
        "N": N,
        "n_up": n_up,
        "omega": float(first_cfg.system.omega),
        "g": float(first_cfg.system.interaction_strength),
        "E_casscf": e_ref,
        "sys_tag": f"N{N}_w{first_cfg.system.omega:g}_g{first_cfg.system.interaction_strength:g}",
        "n_chains": results[0]["n_chains"],
        "T_total": results[0]["T_total"],
        "eval_burn_in": results[0]["eval_burn_in"],
    }
    return results, sys_meta


def plot_production_convergence_figure(
    results: List[Dict[str, Any]],
    sys_meta: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    plt.rcParams.update({"font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 10.6), dpi=240)
    ax_e, ax_err, ax_klsp, ax_kltot = axes.ravel()
    e_ref = sys_meta["E_casscf"]

    for r in results:
        col = r["color"]
        lw = 2.2 if ("RWMH" in r["model_key"] or "eq_unclip" in r["model_key"]) else 1.5
        if r["model_key"].startswith("CTRL_RWMH"):
            ls = "-."
        elif "MALA_base" in r["model_key"] or "MALA_large" in r["model_key"] or "legacy" in r["model_key"]:
            ls = "--"
        else:
            ls = "-"
        t_s = r["t_steps"]
        # Submuestrear ligeramente para PDF limpio si T_total > 4000
        stride = max(1, len(t_s) // 1500)
        ax_e.plot(t_s[::stride], r["cum_mean"][::stride], color=col, lw=lw, ls=ls, label=r["label"])
        ax_err.plot(t_s[::stride], np.maximum(r["cum_rmse"][::stride], 1e-6), color=col, lw=lw, ls=ls)
        ax_klsp.plot(r["kl_steps"], np.maximum(r["kl_spin"], 1e-6), marker="o", ms=3.5, color=col, lw=lw, ls=ls)
        ax_kltot.plot(r["kl_steps"], np.maximum(r["kl_tot"], 1e-6), marker="s", ms=3.2, color=col, lw=lw, ls=ls)

    # Panel (a): Energía media acumulada (zoom robusto alrededor de E_CASSCF para que outliers no aplasten el panel)
    if np.isfinite(e_ref):
        ax_e.axhline(e_ref, color="#111111", ls=":", lw=1.6, label=f"$E_{{\\mathrm{{CASSCF}}}} = {e_ref:.5f}\\,E_h$")
        ax_e.axhspan(e_ref - 1e-3, e_ref + 1e-3, color="#2ca02c", alpha=0.14, label=r"Precisión química ($\pm 1\ \mathrm{mHa}$)")
        # Calcular rango robusto usando percentiles de los modelos sanos (ignorando >0.2 Ha de sesgo)
        healthy_devs = [
            float(np.percentile(np.abs(r["cum_mean"][min(10, len(r["cum_mean"]) - 1):] - e_ref), 95))
            for r in results
            if float(np.abs(r["cum_mean"][-1] - e_ref)) < 0.15
        ]
        if healthy_devs:
            y_pad = max(0.012, min(0.08, 1.35 * max(healthy_devs)))
            ax_e.set_ylim(e_ref - y_pad, e_ref + y_pad)
    ax_e.set_xscale("log")
    ax_e.set_xlabel("Pasos de producción acumulados $T$ (post burn-in)", weight="bold")
    ax_e.set_ylabel(r"Energía Acumulada $\bar{E}(T)$ [$E_h$]", weight="bold")
    ax_e.set_title("(a) Convergencia de la Media Acumulada de Energía $\\bar{E}(T)$", weight="bold")
    ax_e.grid(True, which="both", alpha=0.22, ls="--")

    # Panel (b): RMSE acumulado de energía + pendiente 1/sqrt(T)
    t_guide = np.geomspace(max(10, sys_meta["T_total"] / 400.0), sys_meta["T_total"], 50)
    y0_guide = float(np.median([r["cum_rmse"][min(len(r["cum_rmse"]) - 1, int(t_guide[0]))] for r in results]))
    ax_err.plot(t_guide, y0_guide * np.sqrt(t_guide[0] / t_guide), color="#222222", ls=":", lw=1.3, alpha=0.75)
    ax_err.text(t_guide[-1] * 0.45, y0_guide * np.sqrt(t_guide[0] / t_guide[-1]) * 1.45, r"$\propto T^{-1/2}$", fontsize=8.2, color="#222222", weight="bold")
    ax_err.axhline(1e-3, color="#2ca02c", ls="--", lw=1.1, alpha=0.8)
    ax_err.text(t_guide[0] * 1.1, 1.18e-3, r"$1\ \mathrm{mHa}$", fontsize=7.6, color="#1b7837", weight="bold")
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("Pasos de producción acumulados $T$", weight="bold")
    ax_err.set_ylabel(r"$\mathrm{RMSE}(\bar{E}(T) - E_{\mathrm{CASSCF}})$ [$E_h$]", weight="bold")
    ax_err.set_title("(b) Velocidad de Convergencia de Energía (Precisión Estadística)", weight="bold")
    ax_err.grid(True, which="both", alpha=0.22, ls="--")

    # Panel (c): KL acumulada de espín
    ax_klsp.set_xscale("log")
    ax_klsp.set_yscale("log")
    ax_klsp.set_xlabel("Pasos de producción acumulados $T$", weight="bold")
    ax_klsp.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}(T)$ vs CASSCF", weight="bold")
    ax_klsp.set_title("(c) Convergencia al Estado Base con Espín: $D_{\\mathrm{KL}}^{\\mathrm{spin}}(T)$ (Sesgo vs Varianza)", weight="bold")
    ax_klsp.grid(True, which="both", alpha=0.22, ls="--")

    # Panel (d): KL acumulada de densidad total
    ax_kltot.set_xscale("log")
    ax_kltot.set_yscale("log")
    ax_kltot.set_xlabel("Pasos de producción acumulados $T$", weight="bold")
    ax_kltot.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{tot}}(T)$ vs CASSCF", weight="bold")
    ax_kltot.set_title("(d) Convergencia de Densidad de Carga Total $D_{\\mathrm{KL}}^{\\mathrm{tot}}(T)$", weight="bold")
    ax_kltot.grid(True, which="both", alpha=0.22, ls="--")

    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        fontsize=7.9,
        framealpha=0.80,
        borderpad=0.35,
        columnspacing=1.1,
    )
    fig.suptitle(
        f"Convergencia Acumulada en Producción ({sys_meta['n_chains']}$\\times${sys_meta['T_total']:,} pasos, burn-in previo={sys_meta['eval_burn_in']}) — "
        f"$N={sys_meta['N']}, \\omega={sys_meta['omega']:g}, g={sys_meta['g']:g}$",
        fontsize=12.0,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.848], h_pad=2.2, w_pad=2.0)

    os.makedirs(out_dir, exist_ok=True)
    stem = f"fig_convergence_production_{sys_meta['sys_tag']}"
    saved = []
    for ext in ("png", "pdf"):
        out_p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        saved.append(out_p)
        print(f"  [+] Figura de producción guardada: {out_p}")
    plt.close(fig)
    return saved


def _save_summary_csv(rows: List[Dict[str, Any]], keys: List[str], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"  [+] Tabla resumen guardada: {out_csv}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark de velocidad de convergencia (cold-start + producción) por sampler.")
    parser.add_argument("--sys-dir", required=True, help="Carpeta del sistema con subcarpetas B1_*/B2_*.")
    parser.add_argument("--casscf-npz", default=None, help="Ruta opcional al archivo CASSCF .npz de referencia.")
    parser.add_argument("--out-dir", default=None, help="Directorio de salida (por defecto <sys-dir>/figures_ablation).")
    parser.add_argument("--mode", choices=["both", "cold_start", "production"], default="both")
    parser.add_argument("--init-mode", default="spin_ordered_offset", choices=["spin_ordered_offset", "symmetric_gaussian", "uniform", "randomized_wigner"])
    parser.add_argument("--cold-walkers", type=int, default=128, help="Número de caminantes paralelos para el benchmark cold-start.")
    parser.add_argument("--cold-steps", type=int, default=2000, help="Número de pasos desde t=0 (burn-in=0) para cold-start.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eloc-chunk-size", type=int, default=512)
    parser.add_argument(
        "--extra-rwmh-sigmas",
        type=float,
        nargs="*",
        default=None,
        help="Sigmas adicionales de RWMH puro (ej. 2.705 3.304) para superponer en la gráfica cold-start.",
    )
    args = parser.parse_args()

    specs = _discover_models(args.sys_dir)
    first_bundle = load_run_bundle_h5(os.path.join(args.sys_dir, specs[0][0], "run_bundle.h5"))
    first_cfg = Experiment1DConfig.from_dict(first_bundle["config"])
    casscf_npz = _resolve_casscf_npz(first_cfg, args.casscf_npz)
    out_dir = args.out_dir or os.path.join(args.sys_dir, "figures_ablation")
    os.makedirs(out_dir, exist_ok=True)

    if args.mode in ("both", "cold_start"):
        cs_res, cs_meta = run_cold_start_benchmark(
            args.sys_dir,
            specs,
            casscf_npz=casscf_npz,
            n_walkers=args.cold_walkers,
            n_steps=args.cold_steps,
            init_mode=args.init_mode,
            seed=args.seed,
            eloc_chunk_size=args.eloc_chunk_size,
            extra_rwmh_sigmas=args.extra_rwmh_sigmas,
        )
        plot_cold_start_figure(cs_res, cs_meta, out_dir)
        _save_summary_csv(
            cs_res,
            ["model_key", "label", "acc_mean_tail", "t_eq_e_50mha", "t_eq_e_20mha", "t_eq_com_0p5", "dip_peak", "dip_final_abs", "t_spin_half", "t_spin_10pct"],
            os.path.join(out_dir, f"convergence_cold_start_{cs_meta['sys_tag']}_{args.init_mode}.csv"),
        )
        print("\n=== Resumen Cold-Start (pasos hasta equilibrio desde t=0, burn-in=0) ===")
        for r in cs_res:
            print(
                f"  {r['model_key']:38s} | Acc={r['acc_mean_tail']*100:5.1f}% | "
                f"t_E(<50mHa)={r['t_eq_e_50mha']:6.0f} | t_E(<20mHa)={r['t_eq_e_20mha']:6.0f} | "
                f"t_COM(<0.5)={r['t_eq_com_0p5']:5.0f} | t_spin(50%)={r['t_spin_half']:6.0f} | "
                f"t_spin(10%)={r['t_spin_10pct']:6.0f} | |dip_fin|={r['dip_final_abs']:.2f}"
            )

    if args.mode in ("both", "production"):
        pr_res, pr_meta = analyze_production_convergence(args.sys_dir, specs, casscf_npz=casscf_npz)
        plot_production_convergence_figure(pr_res, pr_meta, out_dir)
        _save_summary_csv(
            pr_res,
            ["model_key", "label", "t_rmse_1mha", "t_kl_1e2", "t_kl_1e3", "final_rmse_E", "final_kl_spin", "final_kl_tot"],
            os.path.join(out_dir, f"convergence_production_{pr_meta['sys_tag']}.csv"),
        )
        print("\n=== Resumen Producción Acumulada (desde .h5 guardados) ===")
        for r in pr_res:
            print(
                f"  {r['model_key']:38s} | t_RMSE(<1mHa)={r['t_rmse_1mha']:6.0f} | "
                f"t_KL(<1e-2)={r['t_kl_1e2']:6.0f} | t_KL(<1e-3)={r['t_kl_1e3']:6.0f} | "
                f"RMSE_fin={r['final_rmse_E']:.5f} | KL_spin_fin={r['final_kl_spin']:.5f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
