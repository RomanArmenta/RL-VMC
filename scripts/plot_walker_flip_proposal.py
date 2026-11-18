#!/usr/bin/env python3
"""
Visualización Microscópica de 1 Caminante Durante un Flip + Gaussiana de Propuesta del Actor.
=============================================================================================

Sigue a UN SOLO caminante real extraído de un archivo `run_bundle.h5` de RL-MALA
alrededor de un evento de intercambio discreto real (t_0 -> t_0 + 1), tanto en
el canal de espín opuesto (barrera de Wigner) como en el canal del mismo espín
(nodo exacto de Pauli Psi = 0), y dibuja explícitamente:

  - Panel (a): En el paso exacto t_0 (justo antes del brinco), la distribución
    Gaussiana exacta de propuesta q_RL(x' | x(t_0)) = N(mu_RL, sigma_RL^2) para
    ambos electrones que intercambian, comparada directamente con la Gaussiana
    contrafactual de MALA q_MALA(x' | x(t_0)) = N(mu_MALA, 2*tau) evaluada en
    la MISMA configuración x(t_0), marcando dónde cayeron realmente en t_0 + 1
    y el cociente de densidad de propuesta q_RL / q_MALA.
  - Panel (b): Los 4 pasos enteros consecutivos (t_0-1, t_0, t_0+1, t_0+2) del
    caminante con las curvas Gaussianas q_RL(x' | x(t)) dibujadas en cada paso.
  - Panel (c): Trayectoria paso a paso x_m(t_0 + k) en k in [-8, +8] con bandas
    sombreadas mu_RL(t) +/- 1*sigma_RL(t).
  - Panel (d): Acciones exactas CON SIGNO del Actor (a_mu para e_i y e_j, deriva
    MALA tau*F_Q, y sigma_RL aplicada vs sin recorte) a lo largo de k in [-8, +8].

Ejemplo de uso:
  python3 scripts/plot_walker_flip_proposal.py \
    --h5 results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.08_amp0.15_pen0.15_sig1.1/run_bundle.h5 \
    --spin_channel both --format both --dpi 300
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jax.numpy as jnp

from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.proposals.proposals_1d import FlaxTD3Actor, build_td3_state
from scripts.plot_flips_system_detailed import (
    _atomic_savefig,
    _compute_exact_casscf_density_np,
)
from scripts.probe_actor_spatial_response import (
    _expand_h5_inputs,
    build_exact_logpsi_and_grad_fn,
)


# ==============================================================================
# 1. DETECCIÓN DE EVENTOS REALES DE FLIP EN UN CAMINANTE INDIVIDUAL
# ==============================================================================

def find_clean_single_walker_flip(
    walkers: np.ndarray,
    n_up: int,
    spin_channel: str = "opp",
    window: int = 8,
    chain_override: Optional[int] = None,
    walker_override: Optional[int] = None,
    t0_override: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Encuentra un evento real, limpio y preferentemente de 1 solo paso donde dos
    electrones adyacentes de un mismo caminante intercambian posiciones entre
    el paso entero t_0 y t_0 + 1.
    """
    w = np.asarray(walkers, dtype=np.float64)
    if w.ndim == 5:
        w = w[..., 0]
    C, T, B, N = w.shape
    spins = np.array([1 if i < n_up else -1 for i in range(N)], dtype=np.int32)

    # Estimar separación característica entre vecinos adyacentes d_adj
    w_sorted = np.sort(w, axis=-1)
    d_adj = float(np.nanmedian(np.diff(w_sorted, axis=-1)))
    if not np.isfinite(d_adj) or d_adj <= 1e-6:
        d_adj = 2.0
    delta = 0.25 * d_adj

    pair_i, pair_j = np.triu_indices(N, k=1)
    is_opp = spins[pair_i] != spins[pair_j]
    target_pairs = np.where(is_opp if spin_channel == "opp" else (~is_opp))[0]
    if len(target_pairs) == 0:
        return None

    # Si el usuario fijó (walker_override, t0_override), buscar el par que cruzó ahí
    if t0_override is not None:
        c0 = int(chain_override or 0)
        b0 = int(walker_override or 0)
        t0 = int(t0_override)
        if t0 - window < 0 or t0 + window + 1 >= T:
            raise ValueError(f"t0={t0} fuera de rango para ventana={window} y T={T}")
        best_p = None
        best_jump = -1.0
        for p_idx in target_pairs:
            a, b = int(pair_i[p_idx]), int(pair_j[p_idx])
            d0 = w[c0, t0, b0, a] - w[c0, t0, b0, b]
            d1 = w[c0, t0 + 1, b0, a] - w[c0, t0 + 1, b0, b]
            if d0 * d1 < 0.0:
                jp = abs(d1 - d0)
                if jp > best_jump:
                    best_jump = jp
                    best_p = p_idx
        if best_p is None:
            return None
        a, b = int(pair_i[best_p]), int(pair_j[best_p])
        if w[c0, t0, b0, a] < w[c0, t0, b0, b]:
            i_left, j_right = a, b
        else:
            i_left, j_right = b, a
        return {
            "chain": c0,
            "walker": b0,
            "t0": t0,
            "i_left": i_left,
            "j_right": j_right,
            "d_adj": d_adj,
            "jump_dij": float(abs((w[c0, t0 + 1, b0, i_left] - w[c0, t0 + 1, b0, j_right]) - (w[c0, t0, b0, i_left] - w[c0, t0, b0, j_right]))),
            "spin_channel": spin_channel,
        }

    # Filtrar caminantes físicamente confinados
    max_bound = max(28.0, 4.0 * float(N + 1))
    walker_max = np.nanmax(np.abs(w), axis=(1, 3))
    healthy = walker_max <= max_bound
    if not np.any(healthy):
        healthy = np.ones((C, B), dtype=bool)

    d_pairs = w[..., pair_i[target_pairs]] - w[..., pair_j[target_pairs]]  # (C, T, B, P_t)
    # Cruces directos de un solo paso que saltan de |d| > delta a |d| > delta con signo opuesto
    one_step_mask = (
        (d_pairs[:, :-1, :, :] * d_pairs[:, 1:, :, :] < 0.0)
        & (np.abs(d_pairs[:, :-1, :, :]) >= delta)
        & (np.abs(d_pairs[:, 1:, :, :]) >= delta)
        & healthy[:, None, :, None]
    )
    cc, tt, bb, pp = np.where(one_step_mask)
    valid = np.where((tt >= window + 2) & (tt + window + 2 < T))[0]

    # Si no hubo cruces de 1 paso con |d|>=delta, permitir cualquier cruce de signo
    if len(valid) == 0:
        any_mask = (d_pairs[:, :-1, :, :] * d_pairs[:, 1:, :, :] < 0.0) & healthy[:, None, :, None]
        cc, tt, bb, pp = np.where(any_mask)
        valid = np.where((tt >= window + 2) & (tt + window + 2 < T))[0]
    if len(valid) == 0:
        return None

    # Evaluar hasta 400 candidatos y escoger el intercambio más limpio entre vecinos adyacentes
    if len(valid) > 400:
        sel = np.linspace(0, len(valid) - 1, 400, dtype=int)
        cand_indices = valid[sel]
    else:
        cand_indices = valid

    best_score = -1e9
    best_ev: Optional[Dict[str, Any]] = None

    for idx in cand_indices:
        c0, t0, b0, p_loc = int(cc[idx]), int(tt[idx]), int(bb[idx]), int(pp[idx])
        if chain_override is not None and c0 != chain_override:
            continue
        if walker_override is not None and b0 != walker_override:
            continue
        p_glob = int(target_pairs[p_loc])
        a, b = int(pair_i[p_glob]), int(pair_j[p_glob])
        if w[c0, t0, b0, a] < w[c0, t0, b0, b]:
            i_left, j_right = a, b
        else:
            i_left, j_right = b, a

        w_win = w[c0, t0 - window : t0 + window + 1, b0, :]
        dij_win = w_win[:, i_left] - w_win[:, j_right]
        # Preferir eventos estables antes (k in [-3..0]: d < 0) y después (k in [1..4]: d > 0)
        pre_ok = bool(np.all(dij_win[window - 3 : window + 1] < 0.0))
        post_ok = bool(np.all(dij_win[window + 1 : window + 5] > 0.0))

        # Verificar si i_left y j_right eran vecinos adyacentes en t0
        order_t0 = np.argsort(w[c0, t0, b0, :])
        inv_t0 = np.argsort(order_t0)
        is_adjacent = abs(int(inv_t0[i_left]) - int(inv_t0[j_right])) == 1

        # Preferir saltos de magnitud cercana a 2*d_adj (intercambio genuino pozo-a-pozo)
        d0 = float(dij_win[window])
        d1 = float(dij_win[window + 1])
        jump_dij = abs(d1 - d0)
        jump_i = abs(float(w_win[window + 1, i_left] - w_win[window, i_left]))
        jump_j = abs(float(w_win[window + 1, j_right] - w_win[window, j_right]))
        # Premiar que AMBOS electrones salten significativamente y que no sean vecinos lejanos
        score = (
            (10.0 if is_adjacent else 0.0)
            + (5.0 if (pre_ok and post_ok) else 0.0)
            + min(jump_i, jump_j)
            + 0.35 * min(jump_dij, 2.5 * d_adj)
            - 0.25 * abs(abs(d0) - d_adj)
        )
        if score > best_score:
            best_score = score
            best_ev = {
                "chain": c0,
                "walker": b0,
                "t0": t0,
                "i_left": i_left,
                "j_right": j_right,
                "d_adj": d_adj,
                "jump_dij": float(jump_dij),
                "spin_channel": spin_channel,
            }

    return best_ev


# ==============================================================================
# 2. EVALUACIÓN EXACTA DEL ACTOR Y PROPUESTAS GAUSSIANAS EN LA VENTANA DEL FLIP
# ==============================================================================

def evaluate_walker_flip_proposals(
    bundle: Dict[str, Any],
    h5_path: str,
    event: Dict[str, Any],
    window: int = 8,
    casscf_npz_override: Optional[str] = None,
) -> Dict[str, Any]:
    eval_logpsi_grad, cfg, npz_path = build_exact_logpsi_and_grad_fn(
        bundle, h5_path, casscf_npz_override=casscf_npz_override
    )
    N = int(cfg.system.n_elec)
    n_up = int(cfg.system.n_up)
    n_dn = int(cfg.system.n_dn)
    omega = float(cfg.system.omega)
    g_val = float(cfg.system.interaction_strength)
    soft = float(cfg.system.softening)
    tau = float(cfg.proposal.tau)
    amp = float(cfg.proposal.actor_amplitude)
    sig_sc = float(cfg.proposal.sigma_action_scale)
    clip_min = float(cfg.proposal.scale_clip_min)
    clip_max = float(cfg.proposal.scale_clip_max)
    parity_eq = bool(getattr(cfg.proposal, "parity_equivariant", False))
    unclip_sig = bool(getattr(cfg.proposal, "unclip_sigma_action", False))
    max_drift_clip = float(cfg.proposal.max_drift_clip)
    drift_pref = 1.0 if cfg.proposal.legacy_drift_factor else 2.0
    base_sigma = math.sqrt(2.0 * tau)

    actor_params = bundle.get("checkpoints", {}).get("actor_final")
    if actor_params is None:
        raise ValueError(f"El bundle no contiene checkpoints/actor_final: {h5_path}")
    p_in = actor_params.get("params", actor_params)

    w = np.asarray(bundle["production"]["walkers"], dtype=np.float64)
    if w.ndim == 5:
        w = w[..., 0]
    c0, b0, t0 = event["chain"], event["walker"], event["t0"]
    i_l, j_r = event["i_left"], event["j_right"]

    w_win = w[c0, t0 - window : t0 + window + 1, b0, :].copy()  # (2*W+1, N)
    rel_k = np.arange(-window, window + 1, dtype=int)
    _, g_win = eval_logpsi_grad(w_win)

    actor_mod = FlaxTD3Actor(
        n_up=n_up,
        n_dn=n_dn,
        actor_amplitude=amp,
        sigma_action_scale=sig_sc,
        hidden_dim_local=int(cfg.proposal.hidden_dim_local),
        hidden_dim_global=int(cfg.proposal.hidden_dim_global),
        hidden_dim_messages=int(cfg.proposal.hidden_dim_messages),
        hidden_dim_actor=int(cfg.proposal.hidden_dim_actor),
        parity_equivariant=parity_eq,
    )

    T_w = len(rel_k)
    a_mu_raw = np.zeros((T_w, N), dtype=np.float64)
    a_sig_raw = np.zeros((T_w, N), dtype=np.float64)
    a_mu_clip = np.zeros((T_w, N), dtype=np.float64)
    a_sig_clip = np.zeros((T_w, N), dtype=np.float64)

    for e_idx in range(N):
        st = build_td3_state(
            jnp.asarray(w_win),
            jnp.asarray(g_win),
            e_idx,
            N,
            n_up,
            n_dn,
            quantum_force_norm=float(cfg.proposal.force_scale),
            particle_factor=float(cfg.proposal.n_scale),
            dist_factor=float(cfg.proposal.r_scale),
            omega=omega,
            gs=g_val,
            softening=soft,
        )
        out = np.asarray(actor_mod.apply({"params": p_in}, st), dtype=np.float64)
        a_mu_raw[:, e_idx] = out[:, 0]
        a_sig_raw[:, e_idx] = out[:, 1]
        a_mu_clip[:, e_idx] = np.clip(out[:, 0], -1.0, 1.0)
        if unclip_sig:
            a_sig_clip[:, e_idx] = np.clip(out[:, 1], clip_min, clip_max)
        else:
            a_sig_clip[:, e_idx] = np.clip(np.clip(out[:, 1], -1.0, 1.0), clip_min, clip_max)

    langevin_drift = np.clip(drift_pref * tau * g_win, -max_drift_clip, max_drift_clip)
    mu_actor = amp * a_mu_clip
    mu_mala = w_win + langevin_drift
    mu_rl = w_win + langevin_drift + mu_actor
    sigma_mala = np.full((T_w, N), base_sigma, dtype=np.float64)
    sigma_rl = np.maximum(base_sigma * np.exp(a_sig_clip), 1e-4)
    sigma_raw = np.maximum(base_sigma * np.exp(a_sig_raw), 1e-4)

    # Estadísticas exactas en el salto t0 (idx=window) -> t0+1 (idx=window+1)
    def _jump_prob_stats(e_idx: int) -> Dict[str, float]:
        x_next = float(w_win[window + 1, e_idx])
        m_rl = float(mu_rl[window, e_idx])
        s_rl = float(sigma_rl[window, e_idx])
        m_m = float(mu_mala[window, e_idx])
        s_m = float(sigma_mala[window, e_idx])
        z_rl = abs(x_next - m_rl) / s_rl
        z_m = abs(x_next - m_m) / s_m
        log_q_rl = -0.5 * math.log(2.0 * math.pi) - math.log(s_rl) - 0.5 * (z_rl ** 2)
        log_q_m = -0.5 * math.log(2.0 * math.pi) - math.log(s_m) - 0.5 * (z_m ** 2)
        log10_ratio = (log_q_rl - log_q_m) / math.log(10.0)
        return {
            "x_t0": float(w_win[window, e_idx]),
            "x_t1": x_next,
            "mu_mala": m_m,
            "mu_rl": m_rl,
            "mu_actor": float(mu_actor[window, e_idx]),
            "langevin": float(langevin_drift[window, e_idx]),
            "sigma_mala": s_m,
            "sigma_rl": s_rl,
            "z_rl": z_rl,
            "z_mala": z_m,
            "log10_q_ratio": log10_ratio,
        }

    x_extent = float(np.ceil(max(float(np.max(np.abs(w_win))) + 4.5, 11.0)))
    casscf_ref = _compute_exact_casscf_density_np(npz_path, x_max=x_extent + 2.0, n_grid=600) if (npz_path and os.path.isfile(npz_path)) else None

    spins = np.array([1 if i < n_up else -1 for i in range(N)], dtype=np.int32)
    return {
        "h5_path": h5_path,
        "N": N,
        "n_up": n_up,
        "n_dn": n_dn,
        "omega": omega,
        "g": g_val,
        "tau": tau,
        "amp": amp,
        "sigma_action_scale": sig_sc,
        "base_sigma": base_sigma,
        "spins": spins,
        "event": event,
        "window": window,
        "rel_k": rel_k,
        "w_win": w_win,
        "g_win": g_win,
        "langevin_drift": langevin_drift,
        "a_mu_clip": a_mu_clip,
        "a_sig_raw": a_sig_raw,
        "a_sig_clip": a_sig_clip,
        "mu_actor": mu_actor,
        "mu_mala": mu_mala,
        "mu_rl": mu_rl,
        "sigma_mala": sigma_mala,
        "sigma_rl": sigma_rl,
        "sigma_raw": sigma_raw,
        "jump_i": _jump_prob_stats(i_l),
        "jump_j": _jump_prob_stats(j_r),
        "x_extent": x_extent,
        "casscf_ref": casscf_ref,
    }


# ==============================================================================
# 3. GRAFICACIÓN DEL CAMINANTE + GAUSSIANAS DE PROPUESTA
# ==============================================================================

def _gauss_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (math.sqrt(2.0 * math.pi) * sigma)


def plot_walker_flip_with_gaussians(
    data: Dict[str, Any],
    out_dir: str,
    stem: str,
    args: argparse.Namespace,
) -> List[str]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N = data["N"]
    spins = data["spins"]
    ev = data["event"]
    W = data["window"]
    rel_k = data["rel_k"]
    w_win = data["w_win"]
    i_l, j_r = ev["i_left"], ev["j_right"]
    x_max = data["x_extent"]
    xg = np.linspace(-x_max, x_max, 800)
    f_scale = float(getattr(args, "font_scale", 1.0))

    # Ordenar electrones en t0 para etiquetas limpias e_1..e_N
    ord_t0 = np.argsort(w_win[W])
    rank_of = {int(el): rk + 1 for rk, el in enumerate(ord_t0)}
    lbl_i = f"e_{{{rank_of[i_l]}}}{'\\uparrow' if spins[i_l] > 0 else '\\downarrow'}"
    lbl_j = f"e_{{{rank_of[j_r]}}}{'\\uparrow' if spins[j_r] > 0 else '\\downarrow'}"

    # Colores distinguibles incluso cuando ambos electrones son del mismo espín
    if spins[i_l] != spins[j_r]:
        col_i = "#1f77b4" if spins[i_l] > 0 else "#d62728"
        col_j = "#1f77b4" if spins[j_r] > 0 else "#d62728"
    else:
        col_i = "#1f77b4" if spins[i_l] > 0 else "#d62728"
        col_j = "#00a6a6" if spins[j_r] > 0 else "#ff7f0e"

    fig = plt.figure(figsize=(16.0, 14.5))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.15, 1.05], hspace=0.35, wspace=0.22)
    ax_top = fig.add_subplot(gs[0, :])
    ax_mid = fig.add_subplot(gs[1, :], sharex=ax_top)
    ax_bl = fig.add_subplot(gs[2, 0])
    ax_br = fig.add_subplot(gs[2, 1])

    # --------------------------------------------------------------------------
    # Panel (a): Paso exacto t0 — Propuesta Gaussiana RL-MALA vs MALA contrafactual
    # --------------------------------------------------------------------------
    if data["casscf_ref"] is not None:
        cr = data["casscf_ref"]
        rho_scaled = cr["rho_total"] / max(float(np.max(cr["rho_total"])), 1e-8) * 0.45
        ax_top.fill_between(cr["x"], 0.0, rho_scaled, color="#888888", alpha=0.14, label=r"Pozos de Wigner $\rho_{\mathrm{CASSCF}}(x)$ (escala ref.)")

    ji, jj = data["jump_i"], data["jump_j"]
    q_rl_i = _gauss_pdf(xg, ji["mu_rl"], ji["sigma_rl"])
    q_m_i = _gauss_pdf(xg, ji["mu_mala"], ji["sigma_mala"])
    q_rl_j = _gauss_pdf(xg, jj["mu_rl"], jj["sigma_rl"])
    q_m_j = _gauss_pdf(xg, jj["mu_mala"], jj["sigma_mala"])

    # Dibujar Gaussianas de RL-MALA (rellenas) y MALA contrafactual (punteadas)
    ax_top.fill_between(xg, 0.0, q_rl_i, color=col_i, alpha=0.24)
    ax_top.plot(xg, q_rl_i, color=col_i, lw=2.4, label=f"Propuesta RL-MALA ${lbl_i}$: $\\mathcal{{N}}({ji['mu_rl']:+.2f},\\ {ji['sigma_rl']:.2f}^2)$")
    ax_top.plot(xg, q_m_i, color=col_i, lw=1.4, ls="--", alpha=0.80, label=f"MALA contrafactual ${lbl_i}$: $\\mathcal{{N}}({ji['mu_mala']:+.2f},\\ {ji['sigma_mala']:.2f}^2)$")

    ax_top.fill_between(xg, 0.0, q_rl_j, color=col_j, alpha=0.24)
    ax_top.plot(xg, q_rl_j, color=col_j, lw=2.4, label=f"Propuesta RL-MALA ${lbl_j}$: $\\mathcal{{N}}({jj['mu_rl']:+.2f},\\ {jj['sigma_rl']:.2f}^2)$")
    ax_top.plot(xg, q_m_j, color=col_j, lw=1.4, ls="--", alpha=0.80, label=f"MALA contrafactual ${lbl_j}$: $\\mathcal{{N}}({jj['mu_mala']:+.2f},\\ {jj['sigma_mala']:.2f}^2)$")

    y_peak = max(float(np.max(q_m_i)), float(np.max(q_m_j)), float(np.max(q_rl_i)), float(np.max(q_rl_j)))

    # Marcar posiciones actuales en t0 (en y=0) y aterrizaje real en t0+1 (estrellas!)
    for e_k in range(N):
        xk = float(w_win[W, e_k])
        if e_k in (i_l, j_r):
            continue
        c_sp = "#1f77b4" if spins[e_k] > 0 else "#d62728"
        ax_top.scatter([xk], [0.02 * y_peak], color=c_sp, marker="o" if spins[e_k] > 0 else "s", s=65, alpha=0.55, edgecolors="black", zorder=5)
        ax_top.text(xk, 0.07 * y_peak, f"$e_{{{rank_of[e_k]}}}$", ha="center", va="bottom", fontsize=7.5 * f_scale, color="#444444")

    for e_k, jp, col_k, lbl_k, arc_rad in [
        (i_l, ji, col_i, lbl_i, -0.28),
        (j_r, jj, col_j, lbl_j, +0.28),
    ]:
        mk = "o" if spins[e_k] > 0 else "s"
        # Posición inicial en t0
        ax_top.scatter([jp["x_t0"]], [0.025 * y_peak], color=col_k, marker=mk, s=140, edgecolors="#ffd700", linewidths=2.2, zorder=8)
        # Posición final en t0+1 (estrella sobre la curva q_RL)
        q_at_land = float(_gauss_pdf(np.array([jp["x_t1"]]), jp["mu_rl"], jp["sigma_rl"])[0])
        ax_top.scatter(
            [jp["x_t1"]],
            [max(q_at_land, 0.03 * y_peak)],
            color=col_k,
            marker="*",
            s=260,
            edgecolors="black",
            linewidths=1.1,
            zorder=9,
            label=f"Aterrizaje real ${lbl_k}$ en $t_0+1$ ($x'={jp['x_t1']:+.2f}$)",
        )
        # Flecha curva mostrando el brinco exacto t0 -> t0+1
        ax_top.annotate(
            "",
            xy=(jp["x_t1"], max(q_at_land, 0.04 * y_peak) + 0.02 * y_peak),
            xytext=(jp["x_t0"], 0.04 * y_peak),
            arrowprops=dict(arrowstyle="-|>", color=col_k, lw=2.0, connectionstyle=f"arc3,rad={arc_rad}", alpha=0.88),
            zorder=7,
        )
        # Línea vertical punteada marcando la media propuesta mu_RL
        ax_top.axvline(jp["mu_rl"], color=col_k, ls=":", lw=1.1, alpha=0.65)

    # Cuadro cuantitativo comparando z-score y probabilidad de propuesta q_RL / q_MALA
    tot_log10_ratio = ji["log10_q_ratio"] + jj["log10_q_ratio"]
    info_box = (
        f"Caminante #{ev['walker']} en $t_0={ev['t0']}$ (salto $|\\Delta d_{{ij}}|={ev['jump_dij']:.2f}$ Bohr):\n"
        f"  ${lbl_i}$: $x={ji['x_t0']:+.2f} \\to {ji['x_t1']:+.2f}$ | "
        f"$\\mu_{{\\mathrm{{actor}}}}={ji['mu_actor']:+.3f}$, $\\tau F_Q={ji['langevin']:+.3f}$ | "
        f"$z_{{\\mathrm{{RL}}}}={ji['z_rl']:.2f}\\sigma$ vs $z_{{\\mathrm{{MALA}}}}={ji['z_mala']:.2f}\\sigma$ "
        f"($q_{{\\mathrm{{RL}}}}/q_{{\\mathrm{{MALA}}}} = 10^{{{ji['log10_q_ratio']:+.1f}}}$)\n"
        f"  ${lbl_j}$: $x={jj['x_t0']:+.2f} \\to {jj['x_t1']:+.2f}$ | "
        f"$\\mu_{{\\mathrm{{actor}}}}={jj['mu_actor']:+.3f}$, $\\tau F_Q={jj['langevin']:+.3f}$ | "
        f"$z_{{\\mathrm{{RL}}}}={jj['z_rl']:.2f}\\sigma$ vs $z_{{\\mathrm{{MALA}}}}={jj['z_mala']:.2f}\\sigma$ "
        f"($q_{{\\mathrm{{RL}}}}/q_{{\\mathrm{{MALA}}}} = 10^{{{jj['log10_q_ratio']:+.1f}}}$)\n"
        f"  Cociente conjunto de propuesta del par: $q_{{\\mathrm{{RL}}}}(x_{{t_0+1}}\\mid x_{{t_0}}) / q_{{\\mathrm{{MALA}}}} = 10^{{{tot_log10_ratio:+.1f}}}$"
    )
    ax_top.text(
        0.012,
        0.96,
        info_box,
        transform=ax_top.transAxes,
        va="top",
        ha="left",
        fontsize=8.0 * f_scale,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.93, edgecolor="#333333"),
        zorder=10,
    )

    ax_top.set_xlim(-x_max, x_max)
    ax_top.set_ylim(0.0, y_peak * 1.24)
    ax_top.set_ylabel(r"Densidad de propuesta $q(x' \mid x(t_0))$", fontsize=9.8 * f_scale, weight="bold")
    ax_top.grid(True, alpha=0.24, ls="--")
    ch_name = r"Espín Opuesto ($\uparrow\downarrow$, Barrera Wigner)" if ev["spin_channel"] == "opp" else r"Mismo Espín ($\uparrow\uparrow/\downarrow\downarrow$, Nodo Pauli $\Psi=0$)"
    if not getattr(args, "no_titles", False):
        ax_top.set_title(
            f"(a) Propuesta Gaussiana del Actor $q_{{\\mathrm{{RL}}}}(x'\\mid x(t_0))$ vs MALA Contrafactual Justo Antes del Flip — {ch_name}",
            fontsize=11.0 * f_scale,
            weight="bold",
        )
    if not getattr(args, "no_legends", False):
        ax_top.legend(loc="upper right", fontsize=7.6 * f_scale, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel (b): Secuencia de 4 pasos discretos (t0-1, t0, t0+1, t0+2) + Gaussianas
    # --------------------------------------------------------------------------
    steps_show = [W - 1, W, W + 1, W + 2]
    y_levels = [3.0, 2.0, 1.0, 0.0]
    y_labels = [
        f"Paso $t_0 - 1$\n($t={ev['t0']-1}$)",
        f"Justo ANTES ($t_0$)\n($t={ev['t0']}$)",
        f"Justo DESPUÉS ($t_0+1$)\n($t={ev['t0']+1}$, ¡ya brincó!)",
        f"Paso $t_0 + 2$\n($t={ev['t0']+2}$)",
    ]
    ax_mid.axhspan(1.0, 2.0, color="#2ca02c", alpha=0.09, zorder=0)

    for lv_i, (st_idx, y_lv) in enumerate(zip(steps_show, y_levels)):
        ax_mid.axhline(y_lv, color="#777777", lw=0.75, alpha=0.35, zorder=1)
        if data["casscf_ref"] is not None:
            cr = data["casscf_ref"]
            rn = cr["rho_total"] / max(float(np.max(cr["rho_total"])), 1e-8) * 0.25
            ax_mid.fill_between(cr["x"], y_lv, y_lv + rn, color="#888888", alpha=0.10, zorder=1)

        # Dibujar mini-Gaussianas q_RL(x' | x(t)) sobre los 2 electrones en flip en cada paso
        for e_k, col_k in [(i_l, col_i), (j_r, col_j)]:
            q_rl_k = _gauss_pdf(xg, float(data["mu_rl"][st_idx, e_k]), float(data["sigma_rl"][st_idx, e_k]))
            q_scaled = q_rl_k / max(float(np.max(q_rl_k)), 1e-8) * 0.42
            ax_mid.fill_between(xg, y_lv, y_lv + q_scaled, color=col_k, alpha=0.22, zorder=2)
            ax_mid.plot(xg, y_lv + q_scaled, color=col_k, lw=1.4, alpha=0.88, zorder=3)

    # Conectar trayectorias entre los 4 niveles
    for e_k in range(N):
        xs_4 = [float(w_win[s_i, e_k]) for s_i in steps_show]
        if e_k in (i_l, j_r):
            c_k = col_i if e_k == i_l else col_j
            ax_mid.plot(xs_4, y_levels, color=c_k, lw=2.3, zorder=5)
            ax_mid.annotate("", xy=(xs_4[2], 1.07), xytext=(xs_4[1], 1.93), arrowprops=dict(arrowstyle="-|>", color=c_k, lw=2.2), zorder=6)
        else:
            ax_mid.plot(xs_4, y_levels, color="#666666", lw=1.0, ls="--", alpha=0.45, zorder=2)

    for lv_i, (st_idx, y_lv) in enumerate(zip(steps_show, y_levels)):
        for e_k in range(N):
            xv = float(w_win[st_idx, e_k])
            is_f = e_k in (i_l, j_r)
            c_k = (col_i if e_k == i_l else col_j) if is_f else ("#1f77b4" if spins[e_k] > 0 else "#d62728")
            mk = "o" if spins[e_k] > 0 else "s"
            ax_mid.scatter([xv], [y_lv], color=c_k, marker=mk, s=115 if is_f else 60, edgecolors="#ffd700" if is_f else "black", linewidths=2.0 if is_f else 0.7, zorder=7 if is_f else 4)
            if is_f and lv_i in (1, 2):
                lbl_k = lbl_i if e_k == i_l else lbl_j
                ax_mid.text(xv, y_lv - 0.16, f"${lbl_k}={xv:+.1f}$", ha="center", va="top", fontsize=7.6 * f_scale, weight="bold", color=c_k, bbox=dict(boxstyle="round,pad=0.10", facecolor="white", alpha=0.85, edgecolor="none"), zorder=8)

    ax_mid.set_yticks(y_levels)
    ax_mid.set_yticklabels(y_labels, fontsize=8.4 * f_scale, weight="bold")
    ax_mid.set_ylim(-0.45, 3.65)
    ax_mid.set_xlabel("Posición exacta de los electrones $x$ [Bohr]", fontsize=9.8 * f_scale, weight="bold")
    ax_mid.grid(True, alpha=0.22, ls="--", axis="x")
    if not getattr(args, "no_titles", False):
        ax_mid.set_title(
            "(b) Evolución Discreta Paso a Paso del Caminante ($t_0-1 \\to t_0+2$) con la Gaussiana de Propuesta $q_{\\mathrm{RL}}(x'\\mid x(t))$ en Cada Paso",
            fontsize=11.0 * f_scale,
            weight="bold",
        )

    # --------------------------------------------------------------------------
    # Panel (c): Trayectoria temporal x_m(t0+k) con banda mu_RL +/- 1*sigma_RL
    # --------------------------------------------------------------------------
    ax_bl.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.12, label="Salto discreto ($t_0 \\to t_0+1$)")
    for e_k in range(N):
        if e_k in (i_l, j_r):
            c_k = col_i if e_k == i_l else col_j
            lbl_k = lbl_i if e_k == i_l else lbl_j
            mk = "o" if spins[e_k] > 0 else "s"
            ax_bl.plot(rel_k, w_win[:, e_k], color=c_k, lw=2.3, marker=mk, markersize=4.5, label=f"${lbl_k}$ (brinca)", zorder=5)
            ax_bl.fill_between(
                rel_k,
                data["mu_rl"][:, e_k] - data["sigma_rl"][:, e_k],
                data["mu_rl"][:, e_k] + data["sigma_rl"][:, e_k],
                color=c_k,
                alpha=0.16,
                label=f"Banda $\\mu_{{\\mathrm{{RL}}}} \\pm 1\\sigma_{{\\mathrm{{RL}}}}$ (${lbl_k}$)",
            )
        else:
            c_sp = "#1f77b4" if spins[e_k] > 0 else "#d62728"
            ax_bl.plot(rel_k, w_win[:, e_k], color=c_sp, lw=0.95, ls="--", alpha=0.45)

    ax_bl.set_xlim(-W, W)
    ax_bl.set_ylim(-x_max * 0.95, x_max * 0.95)
    ax_bl.set_xlabel("Paso entero relativo $k = t - t_0$", fontsize=9.6 * f_scale, weight="bold")
    ax_bl.set_ylabel("Posición $x_m(t_0+k)$ [Bohr]", fontsize=9.6 * f_scale, weight="bold")
    ax_bl.grid(True, alpha=0.25, ls="--")
    if not getattr(args, "no_titles", False):
        ax_bl.set_title("(c) Trayectoria del Caminante y Envolvente de Propuesta ($\\pm 1\\sigma_{\\mathrm{RL}}$)", fontsize=10.4 * f_scale, weight="bold")
    if not getattr(args, "no_legends", False):
        ax_bl.legend(loc="best", fontsize=7.4 * f_scale, ncol=2, framealpha=0.90)

    # --------------------------------------------------------------------------
    # Panel (d): Acción EXACTA CON SIGNO del Actor en este caminante (k in [-W, +W])
    # --------------------------------------------------------------------------
    ax_br.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.12)
    ax_br.axhline(0.0, color="#333333", lw=0.8, ls="-", alpha=0.5)
    ax_br.plot(rel_k, data["mu_actor"][:, i_l], color=col_i, lw=2.2, marker="o", markersize=4.2, label=f"$\\mu_{{\\mathrm{{actor}}}}$ (${lbl_i}$, con signo)")
    ax_br.plot(rel_k, data["mu_actor"][:, j_r], color=col_j, lw=2.2, marker="s", markersize=4.2, label=f"$\\mu_{{\\mathrm{{actor}}}}$ (${lbl_j}$, con signo)")
    ax_br.plot(rel_k, data["langevin_drift"][:, i_l], color=col_i, lw=1.3, ls=":", alpha=0.80, label=f"Deriva MALA $\\tau F_Q$ (${lbl_i}$)")
    ax_br.plot(rel_k, data["langevin_drift"][:, j_r], color=col_j, lw=1.3, ls=":", alpha=0.80, label=f"Deriva MALA $\\tau F_Q$ (${lbl_j}$)")

    ax_br_r = ax_br.twinx()
    ax_br_r.plot(rel_k, data["sigma_rl"][:, i_l], color="#c51b8a", lw=2.1, ls="-", marker="d", markersize=3.8, label=f"$\\sigma_{{\\mathrm{{RL}}}}$ aplicada (${lbl_i}$)")
    ax_br_r.plot(rel_k, data["sigma_raw"][:, i_l], color="#ff7f0e", lw=1.5, ls="--", alpha=0.85, label=f"$\\sigma_{{\\mathrm{{raw}}}}$ sin clip (${lbl_i}$)")
    ax_br_r.axhline(data["base_sigma"], color="#d62728", lw=1.2, ls=":", label=f"$\\sigma_{{\\mathrm{{MALA}}}}={data['base_sigma']:.2f}$")
    ax_br_r.set_ylabel(r"Ancho $\sigma(t_0+k)$ [Bohr]", color="#c51b8a", fontsize=9.4 * f_scale, weight="bold")
    ax_br_r.tick_params(axis="y", colors="#c51b8a")

    ax_br.set_xlim(-W, W)
    ax_br.set_xlabel("Paso entero relativo $k = t - t_0$", fontsize=9.6 * f_scale, weight="bold")
    ax_br.set_ylabel(r"Deriva $\mu_{\mathrm{actor}}, \tau F_Q$ [Bohr]", fontsize=9.4 * f_scale, weight="bold")
    ax_br.grid(True, alpha=0.25, ls="--")
    if not getattr(args, "no_titles", False):
        ax_br.set_title("(d) Acción Exacta del Actor (Con Signo) y $\\sigma$ en Este Caminante", fontsize=10.4 * f_scale, weight="bold")
    if not getattr(args, "no_legends", False):
        h1, l1 = ax_br.get_legend_handles_labels()
        h2, l2 = ax_br_r.get_legend_handles_labels()
        ax_br.legend(h1 + h2, l1 + l2, loc="best", fontsize=7.0 * f_scale, ncol=2, framealpha=0.90)

    if not getattr(args, "no_titles", False):
        run_label = os.path.basename(os.path.dirname(os.path.abspath(data["h5_path"])))
        fig.suptitle(
            f"Anatomía de Propuesta Gaussiana de 1 Caminante en un Flip (`{run_label}`) — "
            f"$N={N}, \\omega={data['omega']:g}, g={data['g']:g}, \\tau={data['tau']:g}, A={data['amp']:g}, s_\\sigma={data['sigma_action_scale']:g}$",
            fontsize=12.2 * f_scale,
            weight="bold",
            y=0.995,
        )

    saved: List[str] = []
    fmt = str(getattr(args, "figure_format", "both")).lower()
    dpi = int(getattr(args, "dpi", 300))
    os.makedirs(out_dir, exist_ok=True)
    if fmt in ("png", "both"):
        p_png = os.path.join(out_dir, f"{stem}.png")
        _atomic_savefig(fig, p_png, "png", dpi=dpi, transparent=False)
        saved.append(p_png)
        print(f"  [+] Figura guardada: {p_png}")
    if fmt in ("pdf", "both"):
        p_pdf = os.path.join(out_dir, f"{stem}.pdf")
        _atomic_savefig(fig, p_pdf, "pdf", dpi=dpi, transparent=False)
        saved.append(p_pdf)
        print(f"  [+] Figura guardada: {p_pdf}")
    plt.close(fig)
    return saved


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sigue a 1 caminante real durante un flip discreto y grafica la Gaussiana exacta de propuesta q_RL(x'|x(t0)) vs MALA."
    )
    p.add_argument("--h5", required=True, nargs="+", help="Uno o más archivos run_bundle.h5 de RL-MALA entrenado (o carpeta).")
    p.add_argument("--casscf_npz", default=None, help="Ruta opcional a casscf_state.npz.")
    p.add_argument("--spin_channel", choices=["opp", "same", "both"], default="both", help="Canal de espín del flip a visualizar.")
    p.add_argument("--chain", type=int, default=None, help="Índice opcional de cadena.")
    p.add_argument("--walker", type=int, default=None, help="Índice opcional de caminante.")
    p.add_argument("--t0", type=int, default=None, help="Paso entero exacto t_0 opcional donde ocurre el flip.")
    p.add_argument("--window", type=int, default=8, help="Ventana de pasos enteros [-W, +W] alrededor de t0.")
    p.add_argument("--out_dir", default="figures/walker_flip_proposals", help="Carpeta de salida.")
    p.add_argument("--figure_format", "--format", dest="figure_format", choices=["png", "pdf", "both"], default="both")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--font_scale", type=float, default=1.0)
    p.add_argument("--no_titles", action="store_true")
    p.add_argument("--no_legends", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    args = build_arg_parser().parse_args(argv)
    h5_files = _expand_h5_inputs(args.h5)
    if not h5_files:
        print("[ERROR] No se encontraron archivos .h5 de RL-MALA válidos.", file=sys.stderr)
        return 1

    channels = ["opp", "same"] if args.spin_channel == "both" else [args.spin_channel]

    print("==============================================================================")
    print(" VISUALIZADOR DE 1 CAMINANTE EN FLIP + GAUSSIANA DE PROPUESTA DEL ACTOR")
    print(f" Modelos a analizar : {len(h5_files)}")
    print(f" Carpeta de salida  : {os.path.abspath(args.out_dir)}")
    print("==============================================================================")

    for h5p in h5_files:
        try:
            bundle = load_run_bundle_h5(h5p)
            if bundle.get("checkpoints", {}).get("actor_final") is None:
                print(f"  [OMITIDO] Sin checkpoints/actor_final: {h5p}")
                continue
        except Exception as exc:
            print(f"  [WARN] No se pudo abrir {h5p}: {exc}", file=sys.stderr)
            continue

        cfg = bundle["config"]
        n_up = int(cfg["system"]["n_up"])
        walkers = bundle["production"]["walkers"]
        run_dir_name = os.path.basename(os.path.dirname(os.path.abspath(h5p)))
        file_stem = os.path.splitext(os.path.basename(h5p))[0]
        model_tag = run_dir_name if file_stem == "run_bundle" else f"{run_dir_name}_{file_stem}"

        for ch in channels:
            ev = find_clean_single_walker_flip(
                walkers,
                n_up=n_up,
                spin_channel=ch,
                window=args.window,
                chain_override=args.chain,
                walker_override=args.walker,
                t0_override=args.t0,
            )
            if ev is None:
                print(f"  [AVISO] {model_tag} ({ch}): no se encontró evento de flip en este canal.")
                continue
            data = evaluate_walker_flip_proposals(
                bundle,
                h5_path=h5p,
                event=ev,
                window=args.window,
                casscf_npz_override=args.casscf_npz,
            )
            ji, jj = data["jump_i"], data["jump_j"]
            print(
                f"\n[{model_tag} | canal={ch}] Caminante #{ev['walker']}, t0={ev['t0']}, |Delta d_ij|={ev['jump_dij']:.2f} Bohr\n"
                f"  e_i: {ji['x_t0']:+.2f}->{ji['x_t1']:+.2f} | z_RL={ji['z_rl']:.2f} vs z_MALA={ji['z_mala']:.2f} | q_RL/q_MALA = 10^{ji['log10_q_ratio']:+.1f}\n"
                f"  e_j: {jj['x_t0']:+.2f}->{jj['x_t1']:+.2f} | z_RL={jj['z_rl']:.2f} vs z_MALA={jj['z_mala']:.2f} | q_RL/q_MALA = 10^{jj['log10_q_ratio']:+.1f}"
            )
            stem = f"fig_walker_flip_proposal_{model_tag}_{ch}_w{ev['walker']}_t{ev['t0']}"
            plot_walker_flip_with_gaussians(data, args.out_dir, stem, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
