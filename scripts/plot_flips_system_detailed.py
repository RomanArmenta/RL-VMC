#!/usr/bin/env python3
"""
scripts/plot_flips_system_detailed.py
==============================================================================
Generador modular y 100% basado en trayectorias reales de Monte Carlo (sin
interpolación fraccionaria artificial en t=0 ni datos sintéticos) para el
análisis detallado de flips por sistema (N, omega, g).

Implementa todas las indicaciones del asesor (incluyendo los 5 audios):
  1. Figura Principal de 4 Paneles Limpios (`fig_flips_system_<pipe>_N<N>_w<w>_g<g>`):
     - Panel (a) [Arriba Izquierda]: Trazas de 1 caminante (MALA arriba, RL-MALA
       abajo) CON su histograma marginal de densidad por espín p_up(x), p_dn(x)
       y la densidad exacta CASSCF a un costado.
     - Panel (b) [Arriba Derecha]: Histograma de pasos de Markov necesarios para
       completar el flip (Delta t_transit en [-delta, +delta]).
     - Panel (c) [Abajo Izquierda]: Mapa de calor 2D del Stream-Flow normalizado
       (x_i - x_j) / d_adj para MALA en pasos enteros discretos k = t - t0
       (donde el cruce ocurre entre el paso entero k = 0 y k = +1), con rango
       vertical ampliado [-2.5, +2.5].
     - Panel (d) [Abajo Derecha]: Mapa de calor 2D del Stream-Flow normalizado
       para RL-MALA en pasos enteros discretos k = t - t0 con rango vertical
       ampliado [-2.5, +2.5], limpio (sin encimar curvas de mu y sigma encima
       del mapa de calor).

  2. Figura de Dinámica del Actor (mu, sigma) Fuera del Heatmap y Magnitud del
     Paso Promedio < |Delta x(k)| > (`fig_flips_actor_and_step_size_<pipe>_N<N>_w<w>_g<g>`):
     - Panel (a) [Arriba]: Evolución de < sigma(k) > y < |mu(k)| > del Actor
       RL-MALA fuera del mapa de calor en función del paso relativo k = t - t0,
       comparada con sigma_MALA = sqrt(2*tau).
     - Panel (b) [Abajo]: Magnitud del paso promedio real aceptado < |Delta x(k)| >
       como función del tiempo centrado en la transición k = t - t0 in [-15, +15],
       comparando directamente MALA (casi constante) contra RL-MALA (pico
       pronunciado en el paso exacto del cruce k = 0 -> +1).

  3. Figura del Paso Exacto de Transición de UN Caminante Individual — Espín
     Opuesto (`fig_flips_before_after_<pipe>_N<N>_w<w>_g<g>`):
     - Toma UN SOLO CAMINANTE REAL y grafica sus posiciones exactas en pasos
       enteros consecutivos t0 - 1, t0 (justo antes del brinco), t0 + 1 (justo
       después del brinco, donde ya se brincaron uno al otro sin encimarse en
       el mismo punto) y t0 + 2, además de las trayectorias paso a paso de los
       N electrones en k in [-10, +10].

  4. Figura Comparativa Espín Opuesto (up-down) vs Mismo Espín (up-up / down-down,
     Nodo de Pauli) en un Caminante (`fig_flips_opp_vs_same_spin_<pipe>_N<N>_w<w>_g<g>`):
     - Muestra paso por paso entero cómo transiciona un caminante cuando los
       electrones tienen espines opuestos frente a cuando tienen el mismo espín.

  5. Paneles individuales en `<out_dir>/panels/panel_*.{png,pdf}`.
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import pickle
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

METHOD_COLORS = {
    "MALA": "#d62728",          # Rojo ladrillo
    "Adaptive-MALA": "#ff7f0e", # Naranja
    "RL-MALA": "#1f77b4",       # Azul profundo
}


# ==============================================================================
# 1. UTILIDADES NUMÉRICAS Y EVALUACIÓN EXACTA DEL ACTOR Y GRADIENTE CASSCF
# ==============================================================================

def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or val == "":
        return default
    try:
        return int(round(float(val)))
    except (ValueError, TypeError):
        return default


def _atomic_write_csv(rows: List[Dict[str, Any]], fieldnames: List[str], out_csv: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(out_csv)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{stem}.tmp.", suffix=".csv", dir=out_dir)
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        os.replace(tmp_path, out_csv)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _atomic_savefig(fig: Any, out_path: str, fmt: str, dpi: int, transparent: bool) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(out_path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{stem}.tmp.", suffix=f".{fmt}", dir=out_dir)
    os.close(fd)
    try:
        fig.savefig(tmp_name, format=fmt, dpi=dpi, bbox_inches="tight", transparent=transparent)
        os.replace(tmp_name, out_path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _dense_np(x: np.ndarray, layer_dict: Dict[str, Any]) -> np.ndarray:
    kernel = np.asarray(layer_dict["kernel"], dtype=np.float64)
    bias = np.asarray(layer_dict["bias"], dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        out = x @ kernel + bias
    return np.nan_to_num(out, nan=0.0, posinf=20.0, neginf=-20.0)


def build_td3_state_np(
    x: np.ndarray,
    g: np.ndarray,
    elec_idx: int,
    n_elec: int,
    n_up: int,
    n_dn: int,
    quantum_force_norm: float = 5.0,
    particle_factor: float = 10.0,
    dist_factor: float = 5.0,
    omega: float = 0.1,
    gs: float = 1.0,
    softening: float = 0.1,
) -> np.ndarray:
    """Construye exactamente el vector de estado de `FlaxTD3Actor` en NumPy."""
    B = x.shape[0]
    i = int(elec_idx)
    x_i = x[:, i]
    g_i = g[:, i]
    spins = np.array([1.0 if k < n_up else -1.0 for k in range(n_elec)], dtype=np.float64)
    spin_i = spins[i]

    local_feat = np.stack(
        [
            x_i,
            x_i - np.mean(x, axis=1),
            np.tanh(g_i / max(quantum_force_norm, 1e-6)),
            np.full_like(x_i, spin_i),
        ],
        axis=-1,
    )
    gf = np.array(
        [
            n_elec / max(particle_factor, 1e-6),
            n_up / max(n_elec, 1),
            math.tanh(omega),
            math.tanh(gs),
            math.tanh(softening),
        ],
        dtype=np.float64,
    )
    global_feat = np.broadcast_to(gf[None, :], (B, 5))

    dx = x[:, i, None] - x
    r = np.abs(dx)
    not_self = np.arange(n_elec) != i
    same_mask = (spins == spin_i) & not_self
    opp_mask = (spins != spin_i) & not_self

    def _get_slots(mask_val: np.ndarray, max_slots: int) -> np.ndarray:
        if max_slots == 0:
            return np.zeros((B, 0, 3), dtype=np.float64)
        mask_w = np.broadcast_to(mask_val[None, :], (B, n_elec))
        keys = np.where(mask_w, r, 1e12)
        order = np.argsort(keys, axis=1)[:, :max_slots]
        rr = np.take_along_axis(r, order, axis=1)
        dd = np.take_along_axis(dx, order, axis=1)
        mm = np.take_along_axis(mask_w, order, axis=1)
        return np.stack(
            [
                np.where(mm, rr / max(dist_factor, 1e-6), 0.0),
                np.where(mm, np.sign(dd), 0.0),
                mm.astype(np.float64),
            ],
            axis=-1,
        )

    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0
    same_slots = _get_slots(same_mask, max_same).reshape((B, -1))
    opp_slots = _get_slots(opp_mask, max_opp).reshape((B, -1))
    return np.concatenate([local_feat, global_feat, same_slots, opp_slots], axis=-1)


def forward_td3_actor_np(
    state: np.ndarray,
    params_dict: Dict[str, Any],
    n_up: int,
    n_dn: int,
    sigma_action_scale: float = 1.0,
    parity_equivariant: bool = False,
) -> np.ndarray:
    """Evalúa exactamente los pesos entrenados de `FlaxTD3Actor` en NumPy sin sesgos."""
    p_inner = params_dict.get("params", params_dict) if isinstance(params_dict, dict) else {}
    if "Dense_0" not in p_inner or "Dense_8" not in p_inner:
        return np.zeros((state.shape[0], 2), dtype=np.float64)

    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0

    if parity_equivariant:
        signs = np.asarray(
            [-1.0, -1.0, -1.0, 1.0]
            + [1.0, 1.0, 1.0, 1.0, 1.0]
            + [1.0, -1.0, 1.0] * (max_same + max_opp),
            dtype=np.float64,
        )
        inp = np.stack([state, state * signs], axis=0)
    else:
        inp = state

    p = 0
    local_feat = inp[..., p : p + 4]
    p += 4
    global_feat = inp[..., p : p + 5]
    p += 5
    same_slots = inp[..., p : p + 3 * max_same].reshape(inp.shape[:-1] + (max_same, 3))
    p += 3 * max_same
    opp_slots = inp[..., p : p + 3 * max_opp].reshape(inp.shape[:-1] + (max_opp, 3))

    d_loc_in, d_loc_out = (
        ("Dense_1", "Dense_0") if p_inner["Dense_1"]["kernel"].shape[0] == 4 else ("Dense_0", "Dense_1")
    )
    d_glb_in, d_glb_out = (
        ("Dense_3", "Dense_2") if p_inner["Dense_3"]["kernel"].shape[0] == 5 else ("Dense_2", "Dense_3")
    )
    d_sam_in, d_sam_out = (
        ("Dense_5", "Dense_4") if p_inner["Dense_5"]["kernel"].shape[0] == 3 else ("Dense_4", "Dense_5")
    )
    d_opp_in, d_opp_out = (
        ("Dense_7", "Dense_6") if p_inner["Dense_7"]["kernel"].shape[0] == 3 else ("Dense_6", "Dense_7")
    )

    h_local = _silu_np(_dense_np(_silu_np(_dense_np(local_feat, p_inner[d_loc_in])), p_inner[d_loc_out]))
    h_global = _silu_np(_dense_np(_silu_np(_dense_np(global_feat, p_inner[d_glb_in])), p_inner[d_glb_out]))

    msg_same = _silu_np(_dense_np(_silu_np(_dense_np(same_slots, p_inner[d_sam_in])), p_inner[d_sam_out]))
    mask_same = same_slots[..., 2:3]
    pool_same = np.sum(msg_same * mask_same, axis=-2) / np.maximum(np.sum(mask_same, axis=-2), 1.0)

    msg_opp = _silu_np(_dense_np(_silu_np(_dense_np(opp_slots, p_inner[d_opp_in])), p_inner[d_opp_out]))
    mask_opp = opp_slots[..., 2:3]
    pool_opp = np.sum(msg_opp * mask_opp, axis=-2) / np.maximum(np.sum(mask_opp, axis=-2), 1.0)

    fused = np.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1)
    h = _silu_np(_dense_np(fused, p_inner["Dense_8"]))
    raw_mean = _dense_np(h, p_inner["Dense_9"])
    raw_scale = _dense_np(h, p_inner["Dense_10"])
    if parity_equivariant:
        mean = np.tanh(0.5 * (raw_mean[0] - raw_mean[1]))
        scale = float(sigma_action_scale) * np.tanh(0.5 * (raw_scale[0] + raw_scale[1]))
    else:
        mean = np.tanh(raw_mean)
        scale = float(sigma_action_scale) * np.tanh(raw_scale)
    return np.concatenate([mean, scale], axis=-1)


_CASSCF_GRAD_FN_CACHE: Dict[Tuple[str, int, int, float], Any] = {}


def _compute_quantum_force_batch(
    x: np.ndarray,
    n_up: int,
    omega: float,
    g_val: float,
    npz_path: Optional[str] = None,
) -> np.ndarray:
    """
    Evalúa grad_x log|Psi(x)| para un lote de configuraciones `x` de forma `(B, N)`.
    Si JAX y el estado `casscf_state.npz` están disponibles, evalúa el gradiente
    exacto de `CASSCF1DAnsatz` (con caché JIT); de lo contrario usa diferencias finitas / derivada
    física exacta sin ningún término de sesgo artificial.
    """
    B, N = x.shape
    if npz_path and os.path.isfile(npz_path):
        try:
            import jax
            import jax.numpy as jnp
            from rl_vmc.ansatze.ansatz_1d import (
                _eval_factored_casscf_logabs,
                _unique_determinants_with_map,
            )

            cache_key = (os.path.abspath(npz_path), int(N), int(n_up), round(float(omega), 6))
            grad_fn = _CASSCF_GRAD_FN_CACHE.get(cache_key)
            if grad_fn is None:
                data = np.load(npz_path, allow_pickle=True)
                meta = json.loads(str(data["metadata_json"]))
                m_cfg = meta["config"]
                n_basis = int(m_cfg.get("n_basis", 256))
                n_core = int(m_cfg.get("n_core", m_cfg.get("n_core_orbitals", 0)))
                n_act = int(m_cfg.get("n_active_orbitals", m_cfg.get("n_active", N)))
                dets = data["selected_ci_determinants"]
                core_idx = list(range(n_core))
                occ_up = np.asarray(
                    [core_idx + [n_core + i for i in range(n_act) if int(d) & (1 << i)] for d in dets],
                    dtype=np.int32,
                )
                occ_dn = np.asarray(
                    [core_idx + [n_core + i for i in range(n_act) if int(d) & (1 << (n_act + i))] for d in dets],
                    dtype=np.int32,
                )
                u_up, up_m = _unique_determinants_with_map(occ_up)
                u_dn, dn_m = _unique_determinants_with_map(occ_dn)
                n_mo = int(max(u_up.max() if u_up.size else 0, u_dn.max() if u_dn.size else 0) + 1)
                orb_used = jnp.asarray(data["orbital_coefficients"][:, :n_mo])
                ci = jnp.asarray(data["selected_ci_coefficients"])
                ci_mat = jnp.zeros((len(u_up), len(u_dn)), dtype=ci.dtype).at[up_m, dn_m].add(ci)

                grad_fn = jax.jit(
                    jax.vmap(
                        jax.grad(
                            lambda x_s: _eval_factored_casscf_logabs(
                                x_s,
                                orb_used,
                                n_up,
                                N - n_up,
                                n_basis,
                                omega,
                                jnp.asarray(u_up),
                                jnp.asarray(u_dn),
                                jnp.asarray(up_m),
                                jnp.asarray(dn_m),
                                ci,
                                ci_mat,
                            )
                        )
                    )
                )
                _CASSCF_GRAD_FN_CACHE[cache_key] = grad_fn
            g_arr = np.asarray(grad_fn(jnp.asarray(x, dtype=jnp.float64)), dtype=np.float64)
            return np.clip(np.nan_to_num(g_arr, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        except Exception:
            pass

    # Respaldo analítico exacto de Slater-Jastrow 1D si no está el .npz
    grad = -float(omega) * x.copy()
    spins = np.array([1 if k < n_up else -1 for k in range(N)], dtype=np.int32)
    for a in range(N):
        dx = x[:, a][:, None] - x
        not_a = np.arange(N) != a
        dx_o = dx[:, not_a]
        same_s = (spins[not_a] == spins[a])[None, :]
        term = np.where(
            same_s,
            dx_o / (dx_o ** 2 + 1e-6),
            0.5 * float(g_val) * dx_o / np.sqrt(dx_o ** 2 + 0.01),
        )
        grad[:, a] += np.sum(term, axis=1)
    return np.clip(np.nan_to_num(grad, nan=0.0), -20.0, 20.0)


def evaluate_legitimate_actor_on_events(
    w_events: np.ndarray,
    i_left_arr: np.ndarray,
    j_right_arr: np.ndarray,
    n_up: int,
    omega: float,
    g_val: float,
    tau: float,
    actor_amp: float,
    sigma_scale: float,
    actor_params: Optional[Dict[str, Any]],
    is_neural: bool,
    npz_path: Optional[str] = None,
    parity_equivariant: bool = False,
    unclip_sigma_action: bool = False,
    scale_clip_min: float = -5.0,
    scale_clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evalúa de forma 100% legítima (sin términos artificiales ni prox_boost) sobre
    un conjunto de ventanas discretas `w_events` de forma `(M, 2*W + 1, N)`:
      - `abs_mu_actor`: magnitud de la deriva residual del Actor `|A_actor * a_mu|`
      - `abs_mu_total`: magnitud de la deriva total propuesta `|tau * grad_log_psi + A_actor * a_mu|`
      - `sigma_prop`: desviación estándar propuesta `sqrt(2*tau) * exp(a_sigma)`
    """
    M, T_w, N = w_events.shape
    n_dn = N - n_up
    tau_eff = max(float(tau), 1e-4)
    base_sigma = math.sqrt(2.0 * tau_eff)
    amp_eff = float(actor_amp) if (np.isfinite(actor_amp) and actor_amp > 0) else 0.10
    sig_sc_eff = float(sigma_scale) if (np.isfinite(sigma_scale) and sigma_scale > 0) else 0.90

    if M == 0:
        empty = np.empty((0, T_w), dtype=np.float64)
        return empty, empty, empty

    # Evaluar sobre hasta 120 eventos representativos para velocidad en CPU
    max_eval = min(M, 120)
    x_flat = w_events[:max_eval].reshape(max_eval * T_w, N)
    g_flat = _compute_quantum_force_batch(x_flat, n_up=n_up, omega=omega, g_val=g_val, npz_path=npz_path)
    g_ev = g_flat.reshape(max_eval, T_w, N)

    abs_mu_actor = np.zeros((max_eval, T_w), dtype=np.float64)
    abs_mu_total = np.zeros((max_eval, T_w), dtype=np.float64)
    sigma_prop = np.full((max_eval, T_w), base_sigma, dtype=np.float64)

    for ev_i in range(max_eval):
        w_e = w_events[ev_i]
        g_e = g_ev[ev_i]
        il = int(i_left_arr[ev_i])
        jr = int(j_right_arr[ev_i])
        lang_i = np.clip(tau_eff * g_e[:, il], -10.0, 10.0)
        lang_j = np.clip(tau_eff * g_e[:, jr], -10.0, 10.0)

        if is_neural and actor_params is not None:
            st_i = build_td3_state_np(w_e, g_e, il, N, n_up, n_dn, omega=omega, gs=g_val)
            st_j = build_td3_state_np(w_e, g_e, jr, N, n_up, n_dn, omega=omega, gs=g_val)
            act_i = forward_td3_actor_np(
                st_i, actor_params, n_up, n_dn,
                sigma_action_scale=sig_sc_eff,
                parity_equivariant=parity_equivariant,
            )
            act_j = forward_td3_actor_np(
                st_j, actor_params, n_up, n_dn,
                sigma_action_scale=sig_sc_eff,
                parity_equivariant=parity_equivariant,
            )
            res_i = amp_eff * np.clip(act_i[:, 0], -1.0, 1.0)
            res_j = amp_eff * np.clip(act_j[:, 0], -1.0, 1.0)
            if unclip_sigma_action:
                sc_i = np.clip(act_i[:, 1], scale_clip_min, scale_clip_max)
                sc_j = np.clip(act_j[:, 1], scale_clip_min, scale_clip_max)
            else:
                sc_i = np.clip(np.clip(act_i[:, 1], -1.0, 1.0), scale_clip_min, scale_clip_max)
                sc_j = np.clip(np.clip(act_j[:, 1], -1.0, 1.0), scale_clip_min, scale_clip_max)
            abs_mu_actor[ev_i] = 0.5 * (np.abs(res_i) + np.abs(res_j))
            abs_mu_total[ev_i] = 0.5 * (np.abs(lang_i + res_i) + np.abs(lang_j + res_j))
            sigma_prop[ev_i] = 0.5 * base_sigma * (np.exp(sc_i) + np.exp(sc_j))
        else:
            abs_mu_actor[ev_i] = 0.0
            abs_mu_total[ev_i] = 0.5 * (np.abs(lang_i) + np.abs(lang_j))
            sigma_prop[ev_i] = base_sigma

    return abs_mu_actor, abs_mu_total, sigma_prop


# ==============================================================================
# 2. DESCUBRIMIENTO DE ARCHIVOS CASSCF Y RUTAS DE PRODUCCIÓN
# ==============================================================================

def _find_casscf_npz_for_system(search_roots: List[str], n_elec: int, omega: float, g_val: float) -> Optional[str]:
    sys_tag = f"N{n_elec}_w{omega:g}_g{g_val:g}"
    sys_tag_f = f"N{n_elec}_w{omega:.1f}_g{g_val:.1f}"
    candidates: List[str] = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for tag in (sys_tag, sys_tag_f):
            p1 = os.path.join(root, "pipeline1_casscf", tag, "casscf_state.npz")
            if os.path.isfile(p1):
                return p1
        for pat in (
            os.path.join(root, "CASSCF_csv", f"*n{n_elec}_g{g_val:g}_*b256*.npz"),
            os.path.join(root, "CASSCF_csv", f"*n{n_elec}_g{g_val:.1f}_*b256*.npz"),
            os.path.join(PROJECT_ROOT, "CASSCF_csv", f"*n{n_elec}_g{g_val:g}_*b256*.npz"),
            os.path.join(PROJECT_ROOT, "CASSCF_csv", f"*n{n_elec}_g{g_val:.1f}_*b256*.npz"),
        ):
            candidates.extend(sorted(glob.glob(pat)))
    return candidates[0] if candidates else None


def _compute_exact_casscf_density_np(
    npz_path: str,
    x_max: float = 25.0,
    n_grid: int = 500,
) -> Optional[Dict[str, np.ndarray]]:
    """Evalúa la densidad exacta de 1 cuerpo de un estado CASSCF `.npz`."""
    if not npz_path or not os.path.isfile(npz_path):
        return None
    try:
        from rl_vmc.postprocess.casscf_baseline import compute_exact_casscf_1body_density
        return compute_exact_casscf_1body_density(npz_path, x_min=-x_max, x_max=x_max, n_grid=n_grid)
    except Exception:
        pass
    try:
        data = np.load(npz_path, allow_pickle=True)
        meta = json.loads(str(data["metadata_json"]))
        m_cfg = meta["config"]
        omega = float(m_cfg.get("omega", 0.1))
        n_up = int(m_cfg.get("n_up", 2))
        n_dn = int(m_cfg.get("n_down", 2))
        n_basis = int(m_cfg.get("n_basis", 256))
        n_core = int(m_cfg.get("n_core", m_cfg.get("n_core_orbitals", 0)))
        n_act = int(m_cfg.get("n_active_orbitals", m_cfg.get("n_active", 2)))
        orb = np.asarray(data["orbital_coefficients"], dtype=np.float64)
        ci = np.asarray(data["selected_ci_coefficients"], dtype=np.float64)
        dets = data["selected_ci_determinants"]

        n_mo = n_core + n_act
        gamma_up = np.zeros((n_mo, n_mo), dtype=np.float64)
        gamma_dn = np.zeros((n_mo, n_mo), dtype=np.float64)
        core_idx = list(range(n_core))
        for k, d_str in enumerate(dets):
            val = int(d_str)
            w_k = float(ci[k] ** 2)
            occ_u = core_idx + [n_core + i for i in range(n_act) if val & (1 << i)]
            occ_d = core_idx + [n_core + i for i in range(n_act) if val & (1 << (n_act + i))]
            for u_idx in occ_u:
                gamma_up[u_idx, u_idx] += w_k
            for d_idx in occ_d:
                gamma_dn[d_idx, d_idx] += w_k

        x_grid = np.linspace(-x_max, x_max, n_grid, dtype=np.float64)
        z = math.sqrt(omega) * x_grid
        pref = (omega / math.pi) ** 0.25 * np.exp(-0.5 * z ** 2)
        psi = [pref]
        if n_basis > 1:
            psi.append(math.sqrt(2.0) * z * pref)
        for n in range(2, n_basis):
            psi.append(math.sqrt(2.0 / n) * z * psi[n - 1] - math.sqrt((n - 1.0) / n) * psi[n - 2])
        basis_x = np.stack(psi[:n_basis], axis=-1)
        mo_x = basis_x @ orb[:, :n_mo]

        rho_up = np.einsum("xm,mn,xn->x", mo_x, gamma_up, mo_x)
        rho_dn = np.einsum("xm,mn,xn->x", mo_x, gamma_dn, mo_x)
        rho_tot = rho_up + rho_dn
        p_up = rho_up / max(float(n_up), 1.0)
        p_dn = rho_dn / max(float(n_dn), 1.0)
        p_tot = rho_tot / max(float(n_up + n_dn), 1.0)
        return {
            "x": x_grid,
            "rho_total": rho_tot,
            "rho_up": rho_up,
            "rho_down": rho_dn,
            "p_total": p_tot,
            "p_up": p_up,
            "p_down": p_dn,
        }
    except Exception:
        return None


def resolve_production_and_csv_dirs(input_path: str) -> Tuple[str, str]:
    abs_in = os.path.abspath(input_path)
    if os.path.isfile(abs_in):
        abs_in = os.path.dirname(abs_in)

    base = os.path.basename(abs_in.rstrip(os.sep))
    if base in ("results_csv", "flips_csv", "csv_data"):
        if base == "results_csv":
            return os.path.dirname(abs_in), abs_in
        parent = os.path.dirname(abs_in)
        if os.path.basename(parent) == "results_csv":
            return os.path.dirname(parent), parent
        return parent, abs_in

    cand_csv = os.path.join(abs_in, "results_csv")
    os.makedirs(cand_csv, exist_ok=True)
    return abs_in, cand_csv


# ==============================================================================
# 3. EXPORTACIÓN Y LECTURA DE TABLAS CSV REALES POR SISTEMA
# ==============================================================================

def export_system_flips_csvs(
    records_ng: List[Dict[str, Any]],
    csv_dir: str,
    prod_root: Optional[str] = None,
) -> Dict[str, str]:
    """
    Exporta a `<csv_dir>` las tablas CSV reales extraídas de los archivos HDF5
    para el sistema `(pipe, N, omega, g)`.
    """
    if not records_ng:
        return {}

    os.makedirs(csv_dir, exist_ok=True)
    rec0 = records_ng[0]
    pipe = str(rec0.get("pipeline", "P2"))
    N = int(rec0["N"])
    n_up = int(rec0.get("n_up", N // 2))
    n_dn = int(rec0.get("n_dn", N - n_up))
    omega = float(rec0.get("omega", 0.1))
    g_val = float(rec0["g"])
    sys_tag = f"{pipe}_N{N}_w{omega:g}_g{g_val:g}"

    order_map = {"MALA": 0, "Adaptive-MALA": 1, "RL-MALA": 2}
    records_sorted = sorted(records_ng, key=lambda r: order_map.get(str(r.get("method", "")), 99))
    saved_csvs: Dict[str, str] = {}

    # 1. Metadatos
    meta_rows: List[Dict[str, Any]] = []
    for r in records_sorted:
        st = r.get("stats", {})
        meta_rows.append(
            {
                "pipeline": pipe,
                "N": N,
                "n_up": n_up,
                "n_dn": n_dn,
                "omega": omega,
                "g": g_val,
                "method": r.get("method", "MALA"),
                "run_name": r.get("run_name", ""),
                "tau": _safe_float(r.get("tau"), 0.08),
                "actor_amplitude": _safe_float(r.get("actor_amplitude"), 0.10),
                "sigma_action_scale": _safe_float(r.get("sigma_action_scale"), 0.90),
                "d_adj": _safe_float(st.get("d_adj"), 4.5),
                "delta_hyst": _safe_float(st.get("delta_hyst"), 1.12),
                "hysteresis_frac": _safe_float(st.get("hysteresis_frac"), 0.25),
                "hyst_flips_total_mean": _safe_float(st.get("hyst_flips_total_mean"), 0.0),
                "hyst_flips_opp_mean": _safe_float(st.get("hyst_flips_opp_mean"), 0.0),
                "hyst_flips_same_mean": _safe_float(st.get("hyst_flips_same_mean"), 0.0),
                "raw_flips_total_mean": _safe_float(st.get("raw_flips_total_mean"), 0.0),
                "one_step_fraction": _safe_float(st.get("one_step_fraction"), 0.0),
                "two_step_fraction": _safe_float(st.get("two_step_fraction"), 0.0),
                "transit_steps_mean": _safe_float(st.get("transit_steps_mean"), float("nan")),
                "transit_steps_median": _safe_float(st.get("transit_steps_median"), float("nan")),
                "jump_at_flip_mean": _safe_float(st.get("jump_at_flip_mean"), float("nan")),
                "normal_pair_jump_mean": _safe_float(st.get("normal_pair_jump_mean"), float("nan")),
                "acceptance": _safe_float(r.get("acceptance"), float("nan")),
                "ESJD": _safe_float(r.get("ESJD"), float("nan")),
                "tau_int_E": _safe_float(r.get("tau_int_E"), float("nan")),
                "kl_spin": _safe_float(r.get("kl_spin"), float("nan")),
                "E_mean": _safe_float(r.get("E_mean"), float("nan")),
                "SE_E": _safe_float(r.get("SE_E"), float("nan")),
            }
        )
    meta_csv = os.path.join(csv_dir, f"flips_system_meta_{sys_tag}.csv")
    _atomic_write_csv(meta_rows, list(meta_rows[0].keys()), meta_csv)
    saved_csvs["meta"] = meta_csv

    # 2. Trazas de caminante
    trace_rows: List[Dict[str, Any]] = []
    trace_fields = ["method", "step", "is_flip_event"] + [f"x_{i}" for i in range(N)]
    for r in records_sorted:
        m_name = str(r.get("method", "MALA"))
        st = r.get("stats", {})
        trace = np.asarray(st.get("best_walker_trace", np.empty((0, N))), dtype=np.float64)
        flips_t = set(int(x) for x in np.asarray(st.get("best_walker_flips_t", []), dtype=int).tolist())
        if trace.shape[0] > 0:
            T_tr = trace.shape[0]
            max_pts = min(T_tr, 2500)
            stride = max(1, T_tr // max_pts)
            sampled_steps = sorted(set(range(0, T_tr, stride)).union(t for t in flips_t if 0 <= t < T_tr))
            for t_idx in sampled_steps:
                row_t: Dict[str, Any] = {
                    "method": m_name,
                    "step": int(t_idx),
                    "is_flip_event": 1 if int(t_idx) in flips_t else 0,
                }
                for i_el in range(min(N, trace.shape[1])):
                    row_t[f"x_{i_el}"] = float(trace[t_idx, i_el])
                trace_rows.append(row_t)
    if trace_rows:
        traces_csv = os.path.join(csv_dir, f"flips_traces_{sys_tag}.csv")
        _atomic_write_csv(trace_rows, trace_fields, traces_csv)
        saved_csvs["traces"] = traces_csv

    # 3. Densidades marginales
    x_max_candidate = 18.0
    for r in records_sorted:
        st = r.get("stats", {})
        tr = np.asarray(st.get("best_walker_trace", np.empty((0, N))), dtype=np.float64)
        if tr.size > 0:
            x_max_candidate = max(x_max_candidate, float(np.nanmax(np.abs(tr))) * 1.15)
        d_a = _safe_float(st.get("d_adj"), 4.0)
        x_max_candidate = max(x_max_candidate, 0.55 * N * d_a + 5.0)
    x_max_grid = float(np.ceil(min(max(x_max_candidate, 15.0), 42.0)))

    search_roots = [p for p in [prod_root, os.path.dirname(csv_dir), PROJECT_ROOT] if p]
    npz_path = _find_casscf_npz_for_system(search_roots, N, omega, g_val)
    casscf_ref = _compute_exact_casscf_density_np(npz_path, x_max=x_max_grid, n_grid=240) if npz_path else None

    n_bins_marg = 160
    edges = np.linspace(-x_max_grid, x_max_grid, n_bins_marg + 1)
    x_centers = 0.5 * (edges[:-1] + edges[1:])

    marg_by_method: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for r in records_sorted:
        m_name = str(r.get("method", "MALA"))
        st = r.get("stats", {})
        if "marginal_x_centers" in st and "marginal_up_hist" in st and "marginal_dn_hist" in st:
            x_c_m = np.asarray(st["marginal_x_centers"], dtype=np.float64)
            u_h_m = np.interp(x_centers, x_c_m, np.asarray(st["marginal_up_hist"], dtype=np.float64), left=0.0, right=0.0)
            d_h_m = np.interp(x_centers, x_c_m, np.asarray(st["marginal_dn_hist"], dtype=np.float64), left=0.0, right=0.0)
            marg_by_method[m_name] = (u_h_m, d_h_m)
        else:
            tr = np.asarray(st.get("best_walker_trace", np.empty((0, N))), dtype=np.float64)
            if tr.size > 0:
                u_h, _ = np.histogram(tr[:, :n_up].reshape(-1), bins=edges, density=True)
                d_h, _ = np.histogram(tr[:, n_up:].reshape(-1), bins=edges, density=True)
                marg_by_method[m_name] = (u_h, d_h)
            else:
                marg_by_method[m_name] = (np.zeros_like(x_centers), np.zeros_like(x_centers))

    if casscf_ref is not None:
        ex_up = np.interp(x_centers, casscf_ref["x"], casscf_ref["p_up"], left=0.0, right=0.0)
        ex_dn = np.interp(x_centers, casscf_ref["x"], casscf_ref["p_down"], left=0.0, right=0.0)
    else:
        ex_up = np.zeros_like(x_centers)
        ex_dn = np.zeros_like(x_centers)

    marg_rows: List[Dict[str, Any]] = []
    m_u, m_d = marg_by_method.get("MALA", (np.zeros_like(x_centers), np.zeros_like(x_centers)))
    r_u, r_d = marg_by_method.get("RL-MALA", (np.zeros_like(x_centers), np.zeros_like(x_centers)))
    for idx_x, xv in enumerate(x_centers):
        marg_rows.append(
            {
                "x_grid": float(xv),
                "mala_up": float(m_u[idx_x]),
                "mala_down": float(m_d[idx_x]),
                "mala_total": float(0.5 * (m_u[idx_x] + m_d[idx_x])),
                "rl_up": float(r_u[idx_x]),
                "rl_down": float(r_d[idx_x]),
                "rl_total": float(0.5 * (r_u[idx_x] + r_d[idx_x])),
                "exact_up": float(ex_up[idx_x]),
                "exact_down": float(ex_dn[idx_x]),
                "exact_total": float(0.5 * (ex_up[idx_x] + ex_dn[idx_x])),
            }
        )
    marg_csv = os.path.join(csv_dir, f"flips_marginal_density_{sys_tag}.csv")
    _atomic_write_csv(marg_rows, list(marg_rows[0].keys()), marg_csv)
    saved_csvs["marginal"] = marg_csv

    # 4. Histograma de pasos de tránsito
    bins_tr = np.arange(1, 12) - 0.5
    x_tr_centers = np.arange(1, 11)
    tr_pct_by_m: Dict[str, np.ndarray] = {}
    tr_cnt_by_m: Dict[str, np.ndarray] = {}
    for r in records_sorted:
        m_name = str(r.get("method", "MALA"))
        ts = np.asarray(r.get("stats", {}).get("transit_steps_samples", []), dtype=np.int32)
        if ts.size > 0:
            ts_c = np.clip(ts, 1, 10)
            cnts, _ = np.histogram(ts_c, bins=bins_tr, density=False)
            pcts = (cnts.astype(np.float64) / max(float(np.sum(cnts)), 1.0)) * 100.0
            tr_pct_by_m[m_name] = pcts
            tr_cnt_by_m[m_name] = cnts
        else:
            tr_pct_by_m[m_name] = np.zeros(10, dtype=np.float64)
            tr_cnt_by_m[m_name] = np.zeros(10, dtype=np.int64)

    transit_rows: List[Dict[str, Any]] = []
    for k_idx, step_k in enumerate(x_tr_centers):
        transit_rows.append(
            {
                "transit_steps": int(step_k),
                "mala_pct": float(tr_pct_by_m.get("MALA", np.zeros(10))[k_idx]),
                "rl_pct": float(tr_pct_by_m.get("RL-MALA", np.zeros(10))[k_idx]),
                "mala_count": int(tr_cnt_by_m.get("MALA", np.zeros(10, dtype=int))[k_idx]),
                "rl_count": int(tr_cnt_by_m.get("RL-MALA", np.zeros(10, dtype=int))[k_idx]),
            }
        )
    transit_csv = os.path.join(csv_dir, f"flips_transit_hist_{sys_tag}.csv")
    _atomic_write_csv(transit_rows, list(transit_rows[0].keys()), transit_csv)
    saved_csvs["transit"] = transit_csv

    # 5. Trayectorias discretas alineadas en pasos enteros k = t - t0, Actor (mu, sigma) y magnitud de paso |Delta x(k)|
    for r in records_sorted:
        m_name = str(r.get("method", "MALA"))
        st = r.get("stats", {})
        aligned_norm = np.asarray(st.get("aligned_rel_dist_norm", np.empty((0, 31))), dtype=np.float64)
        xi_norm = np.asarray(st.get("aligned_xi_norm", np.empty((0, 31))), dtype=np.float64)
        xj_norm = np.asarray(st.get("aligned_xj_norm", np.empty((0, 31))), dtype=np.float64)
        abs_mu_actor = np.asarray(st.get("aligned_abs_mu_actor", np.empty((0, 31))), dtype=np.float64)
        abs_mu_total = np.asarray(st.get("aligned_abs_mu", np.empty((0, 31))), dtype=np.float64)
        sigma_arr = np.asarray(st.get("aligned_sigma", np.empty((0, 31))), dtype=np.float64)
        step_mag_pair = np.asarray(st.get("aligned_step_mag_pair", np.empty((0, 31))), dtype=np.float64)
        step_mag_spec = np.asarray(st.get("aligned_step_mag_spec", np.empty((0, 31))), dtype=np.float64)

        if aligned_norm.shape[0] > 0:
            W = (aligned_norm.shape[1] - 1) // 2
            rel_t = np.arange(-W, W + 1, dtype=int)
            med_d = np.median(aligned_norm, axis=0)
            q25_d = np.percentile(aligned_norm, 25, axis=0)
            q75_d = np.percentile(aligned_norm, 75, axis=0)
            q10_d = np.percentile(aligned_norm, 10, axis=0)
            q90_d = np.percentile(aligned_norm, 90, axis=0)
            med_xi = np.median(xi_norm, axis=0) if xi_norm.shape == aligned_norm.shape else -0.5 * med_d
            med_xj = np.median(xj_norm, axis=0) if xj_norm.shape == aligned_norm.shape else 0.5 * med_d

            def _stat_3(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                if arr.shape[0] > 0 and arr.shape[1] == 2 * W + 1:
                    return np.mean(arr, axis=0), np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
                z = np.zeros(2 * W + 1, dtype=np.float64)
                return z, z, z

            mean_mu_act, q25_mu_act, q75_mu_act = _stat_3(abs_mu_actor)
            mean_mu_tot, q25_mu_tot, q75_mu_tot = _stat_3(abs_mu_total)
            mean_sig, q25_sig, q75_sig = _stat_3(sigma_arr)
            mean_dx_p, q25_dx_p, q75_dx_p = _stat_3(step_mag_pair)
            mean_dx_s, q25_dx_s, q75_dx_s = _stat_3(step_mag_spec)

            al_rows: List[Dict[str, Any]] = []
            for k_t, rt in enumerate(rel_t):
                al_rows.append(
                    {
                        "rel_step": int(rt),
                        "n_events": int(aligned_norm.shape[0]),
                        "med_d_norm": float(med_d[k_t]),
                        "q25_d_norm": float(q25_d[k_t]),
                        "q75_d_norm": float(q75_d[k_t]),
                        "q10_d_norm": float(q10_d[k_t]),
                        "q90_d_norm": float(q90_d[k_t]),
                        "med_xi_norm": float(med_xi[k_t]),
                        "med_xj_norm": float(med_xj[k_t]),
                        "mean_abs_mu_actor": float(mean_mu_act[k_t]),
                        "q25_abs_mu_actor": float(q25_mu_act[k_t]),
                        "q75_abs_mu_actor": float(q75_mu_act[k_t]),
                        "mean_abs_mu": float(mean_mu_tot[k_t]),
                        "q25_abs_mu": float(q25_mu_tot[k_t]),
                        "q75_abs_mu": float(q75_mu_tot[k_t]),
                        "mean_sigma": float(mean_sig[k_t]),
                        "q25_sigma": float(q25_sig[k_t]),
                        "q75_sigma": float(q75_sig[k_t]),
                        "mean_step_mag_pair": float(mean_dx_p[k_t]),
                        "q25_step_mag_pair": float(q25_dx_p[k_t]),
                        "q75_step_mag_pair": float(q75_dx_p[k_t]),
                        "mean_step_mag_spec": float(mean_dx_s[k_t]),
                    }
                )
            al_csv = os.path.join(csv_dir, f"flips_heatmap_aligned_{m_name}_{sys_tag}.csv")
            _atomic_write_csv(al_rows, list(al_rows[0].keys()), al_csv)
            saved_csvs[f"aligned_{m_name}"] = al_csv

            ev_rows: List[Dict[str, Any]] = []
            max_ev_save = min(aligned_norm.shape[0], 350)
            for ev_i in range(max_ev_save):
                for k_t, rt in enumerate(rel_t):
                    ev_rows.append(
                        {
                            "event_id": int(ev_i),
                            "rel_step": int(rt),
                            "d_ij_norm": float(aligned_norm[ev_i, k_t]),
                            "xi_norm": float(xi_norm[ev_i, k_t]) if xi_norm.shape == aligned_norm.shape else float(-0.5 * aligned_norm[ev_i, k_t]),
                            "xj_norm": float(xj_norm[ev_i, k_t]) if xj_norm.shape == aligned_norm.shape else float(0.5 * aligned_norm[ev_i, k_t]),
                            "abs_mu_actor": float(abs_mu_actor[min(ev_i, len(abs_mu_actor) - 1), k_t]) if len(abs_mu_actor) > 0 else float(mean_mu_act[k_t]),
                            "abs_mu": float(abs_mu_total[min(ev_i, len(abs_mu_total) - 1), k_t]) if len(abs_mu_total) > 0 else float(mean_mu_tot[k_t]),
                            "sigma": float(sigma_arr[min(ev_i, len(sigma_arr) - 1), k_t]) if len(sigma_arr) > 0 else float(mean_sig[k_t]),
                            "step_mag_pair": float(step_mag_pair[min(ev_i, len(step_mag_pair) - 1), k_t]) if len(step_mag_pair) > 0 else float(mean_dx_p[k_t]),
                            "step_mag_spec": float(step_mag_spec[min(ev_i, len(step_mag_spec) - 1), k_t]) if len(step_mag_spec) > 0 else float(mean_dx_s[k_t]),
                        }
                    )
            ev_csv = os.path.join(csv_dir, f"flips_heatmap_events_{m_name}_{sys_tag}.csv")
            _atomic_write_csv(ev_rows, list(ev_rows[0].keys()), ev_csv)
            saved_csvs[f"events_{m_name}"] = ev_csv

    # 6. Eventos reales de caminante individual (opp_spin, same_spin, y ensemble)
    ba_rows: List[Dict[str, Any]] = []
    ba_fields = (
        ["method", "mode", "rel_step", "well_left", "well_right", "x_cross", "d_adj", "t0_abs", "walker_idx", "transit_steps"]
        + [f"x_{i}" for i in range(N)]
        + [f"std_{i}" for i in range(N)]
        + [f"spin_{i}" for i in range(N)]
    )
    for r in records_sorted:
        m_name = str(r.get("method", "MALA"))
        st = r.get("stats", {})
        d_adj_m = _safe_float(st.get("d_adj"), 4.5)
        ba_info = st.get("before_after_info")
        if ba_info is not None:
            rel_steps_ba = np.asarray(ba_info["rel_steps"], dtype=int)
            for mode_key in ("opp_single", "same_single", "ensemble"):
                if f"{mode_key}_x" not in ba_info or ba_info[f"{mode_key}_x"] is None:
                    continue
                x_mat = np.asarray(ba_info[f"{mode_key}_x"], dtype=np.float64)
                s_mat = np.asarray(ba_info.get(f"{mode_key}_std", np.zeros_like(x_mat)), dtype=np.float64)
                sp_vec = np.asarray(ba_info[f"{mode_key}_spins"], dtype=int)
                w_l = int(ba_info.get(f"{mode_key}_well_left", ba_info.get("well_left", 0)))
                w_r = int(ba_info.get(f"{mode_key}_well_right", ba_info.get("well_right", 1)))
                x_cr = float(ba_info.get(f"{mode_key}_x_cross", ba_info.get("x_cross", 0.0)))
                t0_abs = int(ba_info.get(f"{mode_key}_t0", 0))
                w_idx = int(ba_info.get(f"{mode_key}_walker", 0))
                tr_st = int(ba_info.get(f"{mode_key}_transit_steps", 1))
                for idx_t, rt in enumerate(rel_steps_ba):
                    row_ba: Dict[str, Any] = {
                        "method": m_name,
                        "mode": mode_key,
                        "rel_step": int(rt),
                        "well_left": w_l,
                        "well_right": w_r,
                        "x_cross": x_cr,
                        "d_adj": d_adj_m,
                        "t0_abs": t0_abs,
                        "walker_idx": w_idx,
                        "transit_steps": tr_st,
                    }
                    for m_el in range(N):
                        row_ba[f"x_{m_el}"] = float(x_mat[idx_t, m_el])
                        row_ba[f"std_{m_el}"] = float(s_mat[idx_t, m_el])
                        row_ba[f"spin_{m_el}"] = int(sp_vec[m_el])
                    ba_rows.append(row_ba)
    if ba_rows:
        ba_csv = os.path.join(csv_dir, f"flips_before_after_state_{sys_tag}.csv")
        _atomic_write_csv(ba_rows, ba_fields, ba_csv)
        saved_csvs["before_after"] = ba_csv

    return saved_csvs


def load_system_records_from_csvs(
    csv_dir: str,
    prod_root: str,
    n_elec: int,
    omega: float,
    g_val: float,
    pipeline: str = "P2",
) -> Optional[List[Dict[str, Any]]]:
    """
    Lee los archivos CSV reales (`flips_*_<tag>.csv`) desde `csv_dir` y
    reconstruye `records_ng` para graficar.
    """
    tags_to_try = [
        f"{pipeline}_N{n_elec}_w{omega:g}_g{g_val:g}",
        f"{pipeline}_N{n_elec}_w{omega:.1f}_g{g_val:.1f}",
    ]
    chosen_tag: Optional[str] = None
    for tg in tags_to_try:
        if os.path.isfile(os.path.join(csv_dir, f"flips_system_meta_{tg}.csv")):
            chosen_tag = tg
            break
    if chosen_tag is None:
        return None

    meta_csv = os.path.join(csv_dir, f"flips_system_meta_{chosen_tag}.csv")
    traces_csv = os.path.join(csv_dir, f"flips_traces_{chosen_tag}.csv")
    marg_csv = os.path.join(csv_dir, f"flips_marginal_density_{chosen_tag}.csv")
    transit_csv = os.path.join(csv_dir, f"flips_transit_hist_{chosen_tag}.csv")
    ba_csv = os.path.join(csv_dir, f"flips_before_after_state_{chosen_tag}.csv")

    meta_by_m: Dict[str, Dict[str, Any]] = {}
    with open(meta_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta_by_m[str(row["method"])] = dict(row)

    traces_by_m: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if os.path.isfile(traces_csv):
        raw_tr: Dict[str, List[Tuple[int, int, List[float]]]] = {}
        with open(traces_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m_n = str(row["method"])
                st_i = _safe_int(row["step"])
                is_f = _safe_int(row.get("is_flip_event", 0))
                xs = [_safe_float(row.get(f"x_{i}"), 0.0) for i in range(n_elec)]
                raw_tr.setdefault(m_n, []).append((st_i, is_f, xs))
        for m_n, items in raw_tr.items():
            items.sort(key=lambda t: t[0])
            steps_a = np.array([it[0] for it in items], dtype=np.int64)
            flips_a = np.array([it[0] for it in items if it[1] > 0], dtype=np.int64)
            xs_a = np.array([it[2] for it in items], dtype=np.float64)
            traces_by_m[m_n] = (steps_a, xs_a, flips_a)

    x_marg = np.linspace(-20.0, 20.0, 160)
    marg_data: Dict[str, np.ndarray] = {}
    if os.path.isfile(marg_csv):
        cols_m: Dict[str, List[float]] = {}
        with open(marg_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                for k, v in row.items():
                    cols_m.setdefault(k, []).append(_safe_float(v, 0.0))
        if "x_grid" in cols_m:
            x_marg = np.asarray(cols_m["x_grid"], dtype=np.float64)
            for k, arr_l in cols_m.items():
                marg_data[k] = np.asarray(arr_l, dtype=np.float64)

    tr_pct: Dict[str, np.ndarray] = {}
    if os.path.isfile(transit_csv):
        m_p_list: List[float] = []
        r_p_list: List[float] = []
        with open(transit_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m_p_list.append(_safe_float(row.get("mala_pct"), 0.0))
                r_p_list.append(_safe_float(row.get("rl_pct"), 0.0))
        tr_pct["MALA"] = np.asarray(m_p_list, dtype=np.float64)
        tr_pct["RL-MALA"] = np.asarray(r_p_list, dtype=np.float64)

    ba_by_m: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(ba_csv):
        ba_rows_grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        with open(ba_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ba_rows_grouped.setdefault((str(row["method"]), str(row["mode"])), []).append(dict(row))
        for m_n in meta_by_m.keys():
            info_m: Dict[str, Any] = {}
            for mode_key in ("opp_single", "same_single", "ensemble", "example"):
                rows_mk = ba_rows_grouped.get((m_n, mode_key), [])
                if not rows_mk:
                    continue
                rows_mk.sort(key=lambda r: _safe_int(r["rel_step"]))
                target_prefix = "opp_single" if mode_key == "example" else mode_key
                info_m["rel_steps"] = np.array([_safe_int(r["rel_step"]) for r in rows_mk], dtype=int)
                info_m[f"{target_prefix}_x"] = np.array([[_safe_float(r[f"x_{i}"]) for i in range(n_elec)] for r in rows_mk], dtype=np.float64)
                info_m[f"{target_prefix}_std"] = np.array([[_safe_float(r.get(f"std_{i}"), 0.0) for i in range(n_elec)] for r in rows_mk], dtype=np.float64)
                info_m[f"{target_prefix}_spins"] = np.array([_safe_int(rows_mk[0][f"spin_{i}"], 1) for i in range(n_elec)], dtype=int)
                info_m[f"{target_prefix}_well_left"] = _safe_int(rows_mk[0].get("well_left"), 0)
                info_m[f"{target_prefix}_well_right"] = _safe_int(rows_mk[0].get("well_right"), 1)
                info_m[f"{target_prefix}_x_cross"] = _safe_float(rows_mk[0].get("x_cross"), 0.0)
                info_m[f"{target_prefix}_t0"] = _safe_int(rows_mk[0].get("t0_abs"), 0)
                info_m[f"{target_prefix}_walker"] = _safe_int(rows_mk[0].get("walker_idx"), 0)
                info_m[f"{target_prefix}_transit_steps"] = _safe_int(rows_mk[0].get("transit_steps"), 1)
            if info_m:
                ba_by_m[m_n] = info_m

    records_loaded: List[Dict[str, Any]] = []
    for m_n, m_row in meta_by_m.items():
        ev_csv = os.path.join(csv_dir, f"flips_heatmap_events_{m_n}_{chosen_tag}.csv")
        aligned_norm = np.empty((0, 31))
        xi_norm = np.empty((0, 31))
        xj_norm = np.empty((0, 31))
        abs_mu_actor = np.empty((0, 31))
        abs_mu_arr = np.empty((0, 31))
        sigma_arr = np.empty((0, 31))
        step_mag_p = np.empty((0, 31))
        step_mag_s = np.empty((0, 31))

        if os.path.isfile(ev_csv):
            by_ev: Dict[int, List[Tuple[int, float, float, float, float, float, float, float, float]]] = {}
            with open(ev_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    eid = _safe_int(row["event_id"])
                    rt = _safe_int(row["rel_step"])
                    by_ev.setdefault(eid, []).append(
                        (
                            rt,
                            _safe_float(row.get("d_ij_norm"), 0.0),
                            _safe_float(row.get("xi_norm"), 0.0),
                            _safe_float(row.get("xj_norm"), 0.0),
                            _safe_float(row.get("abs_mu_actor"), 0.0),
                            _safe_float(row.get("abs_mu"), 0.0),
                            _safe_float(row.get("sigma"), 0.0),
                            _safe_float(row.get("step_mag_pair"), 0.0),
                            _safe_float(row.get("step_mag_spec"), 0.0),
                        )
                    )
            if by_ev:
                d_l, xi_l, xj_l, mua_l, mu_l, sig_l, dxp_l, dxs_l = [], [], [], [], [], [], [], []
                for eid in sorted(by_ev.keys()):
                    pts = sorted(by_ev[eid], key=lambda t: t[0])
                    d_l.append([p[1] for p in pts])
                    xi_l.append([p[2] for p in pts])
                    xj_l.append([p[3] for p in pts])
                    mua_l.append([p[4] for p in pts])
                    mu_l.append([p[5] for p in pts])
                    sig_l.append([p[6] for p in pts])
                    dxp_l.append([p[7] for p in pts])
                    dxs_l.append([p[8] for p in pts])
                aligned_norm = np.asarray(d_l, dtype=np.float64)
                xi_norm = np.asarray(xi_l, dtype=np.float64)
                xj_norm = np.asarray(xj_l, dtype=np.float64)
                abs_mu_actor = np.asarray(mua_l, dtype=np.float64)
                abs_mu_arr = np.asarray(mu_l, dtype=np.float64)
                sigma_arr = np.asarray(sig_l, dtype=np.float64)
                step_mag_p = np.asarray(dxp_l, dtype=np.float64)
                step_mag_s = np.asarray(dxs_l, dtype=np.float64)

        steps_tr, xs_tr, flips_tr = traces_by_m.get(
            m_n,
            (np.arange(0), np.empty((0, n_elec)), np.arange(0)),
        )
        pcts = tr_pct.get(m_n, np.zeros(10))
        if np.sum(pcts) > 0:
            counts_rep = np.maximum(0, np.round(pcts * 10.0).astype(int))
            ts_samples = np.repeat(np.arange(1, 11, dtype=np.int32), counts_rep)
        else:
            ts_samples = np.array([], dtype=np.int32)

        pref = "rl" if m_n == "RL-MALA" else "mala"
        st_loaded = {
            "d_adj": _safe_float(m_row.get("d_adj"), 4.5),
            "delta_hyst": _safe_float(m_row.get("delta_hyst"), 1.12),
            "hysteresis_frac": _safe_float(m_row.get("hysteresis_frac"), 0.25),
            "hyst_flips_total_mean": _safe_float(m_row.get("hyst_flips_total_mean"), 0.0),
            "hyst_flips_opp_mean": _safe_float(m_row.get("hyst_flips_opp_mean"), 0.0),
            "hyst_flips_same_mean": _safe_float(m_row.get("hyst_flips_same_mean"), 0.0),
            "raw_flips_total_mean": _safe_float(m_row.get("raw_flips_total_mean"), 0.0),
            "one_step_fraction": _safe_float(m_row.get("one_step_fraction"), 0.0),
            "two_step_fraction": _safe_float(m_row.get("two_step_fraction"), 0.0),
            "transit_steps_mean": _safe_float(m_row.get("transit_steps_mean")),
            "transit_steps_median": _safe_float(m_row.get("transit_steps_median")),
            "jump_at_flip_mean": _safe_float(m_row.get("jump_at_flip_mean")),
            "normal_pair_jump_mean": _safe_float(m_row.get("normal_pair_jump_mean")),
            "transit_steps_samples": ts_samples,
            "transit_pct_direct": pcts,
            "aligned_rel_dist_norm": aligned_norm,
            "aligned_xi_norm": xi_norm,
            "aligned_xj_norm": xj_norm,
            "aligned_abs_mu_actor": abs_mu_actor,
            "aligned_abs_mu": abs_mu_arr,
            "aligned_sigma": sigma_arr,
            "aligned_step_mag_pair": step_mag_p,
            "aligned_step_mag_spec": step_mag_s,
            "best_walker_steps": steps_tr,
            "best_walker_trace": xs_tr,
            "best_walker_flips_t": flips_tr,
            "marginal_x_centers": x_marg,
            "marginal_up_hist": marg_data.get(f"{pref}_up", np.zeros_like(x_marg)),
            "marginal_dn_hist": marg_data.get(f"{pref}_down", np.zeros_like(x_marg)),
            "exact_up_hist": marg_data.get("exact_up", np.zeros_like(x_marg)),
            "exact_dn_hist": marg_data.get("exact_down", np.zeros_like(x_marg)),
            "before_after_info": ba_by_m.get(m_n),
        }

        records_loaded.append(
            {
                "pipeline": str(m_row.get("pipeline", pipeline)),
                "N": _safe_int(m_row.get("N"), n_elec),
                "n_up": _safe_int(m_row.get("n_up"), n_elec // 2),
                "n_dn": _safe_int(m_row.get("n_dn"), n_elec - n_elec // 2),
                "omega": _safe_float(m_row.get("omega"), omega),
                "g": _safe_float(m_row.get("g"), g_val),
                "method": m_n,
                "run_name": str(m_row.get("run_name", "")),
                "tau": _safe_float(m_row.get("tau"), 0.08),
                "actor_amplitude": _safe_float(m_row.get("actor_amplitude"), 0.10),
                "sigma_action_scale": _safe_float(m_row.get("sigma_action_scale"), 0.90),
                "acceptance": _safe_float(m_row.get("acceptance")),
                "ESJD": _safe_float(m_row.get("ESJD")),
                "tau_int_E": _safe_float(m_row.get("tau_int_E")),
                "kl_spin": _safe_float(m_row.get("kl_spin")),
                "E_mean": _safe_float(m_row.get("E_mean")),
                "SE_E": _safe_float(m_row.get("SE_E")),
                "stats": st_loaded,
            }
        )
    return records_loaded if records_loaded else None


# ==============================================================================
# 4. FUNCIONES DE GRAFICACIÓN (SIN SESGOS NI CURVAS ENCIMADAS EN EL HEATMAP)
# ==============================================================================

def _smooth_2d_heatmap(H: np.ndarray, sigma_y: float = 2.0, sigma_t: float = 1.2) -> np.ndarray:
    rad_y = max(1, int(round(3.0 * sigma_y)))
    ky = np.exp(-0.5 * (np.arange(-rad_y, rad_y + 1, dtype=np.float64) / sigma_y) ** 2)
    ky /= np.sum(ky)
    rad_t = max(1, int(round(3.0 * sigma_t)))
    kt = np.exp(-0.5 * (np.arange(-rad_t, rad_t + 1, dtype=np.float64) / sigma_t) ** 2)
    kt /= np.sum(kt)
    H_y = np.apply_along_axis(lambda col: np.convolve(col, ky, mode="same"), axis=0, arr=H)
    H_yt = np.apply_along_axis(lambda row: np.convolve(row, kt, mode="same"), axis=1, arr=H_y)
    return H_yt


def _method_display_label(m_name: str, args: argparse.Namespace) -> str:
    if m_name == "MALA":
        return str(getattr(args, "label_mala", "") or "MALA")
    if m_name == "RL-MALA":
        return str(getattr(args, "label_rl", "") or "RL-MALA")
    return m_name


def _panel_prefix(letter: str, args: argparse.Namespace) -> str:
    if not getattr(args, "show_panel_letters", True):
        return ""
    return f"({letter}) "


def _apply_legend(ax: Any, args: argparse.Namespace, default_loc: str = "best", fontsize: float = 8.0, ncol: int = 1, framealpha: float = 0.90) -> None:
    if getattr(args, "no_legends", False):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        return
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    loc_override = str(getattr(args, "legend_loc", "") or "").strip()
    loc_use = loc_override if loc_override else default_loc
    f_scale = float(getattr(args, "font_scale", 1.0))
    leg_fs = _safe_float(getattr(args, "legend_fontsize", None), fontsize * f_scale)
    ax.legend(handles, labels, loc=loc_use, fontsize=leg_fs, ncol=ncol, framealpha=framealpha)


def _draw_single_trace_and_marginal(
    ax_tr: Any,
    ax_hist: Any,
    rec: Dict[str, Any],
    casscf_ref: Optional[Dict[str, np.ndarray]],
    x_max: float,
    args: argparse.Namespace,
    is_top: bool = True,
    is_bottom: bool = True,
    custom_title: Optional[str] = None,
) -> None:
    from matplotlib.ticker import MaxNLocator

    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))

    m_raw = str(rec.get("method", "MALA"))
    m_disp = _method_display_label(m_raw, args)
    N = int(rec["N"])
    n_up = int(rec.get("n_up", N // 2))
    n_dn = int(rec.get("n_dn", N - n_up))
    st = rec.get("stats", {})

    trace = np.asarray(st.get("best_walker_trace", np.empty((0, N))), dtype=np.float64)
    steps_custom = st.get("best_walker_steps")
    flips_t = np.asarray(st.get("best_walker_flips_t", []), dtype=int)

    if trace.shape[0] > 0:
        if steps_custom is not None and len(steps_custom) == trace.shape[0]:
            t_grid = np.asarray(steps_custom, dtype=np.float64)
            trace_plot = trace
        else:
            T_tr = trace.shape[0]
            max_pts = min(T_tr, 5000)
            stride = max(1, T_tr // max_pts)
            t_grid = np.arange(0, T_tr, stride, dtype=np.float64)
            trace_plot = trace[::stride]

        for i_el in range(min(N, trace_plot.shape[1])):
            is_up = (i_el < n_up)
            c_el = "#1f77b4" if is_up else "#d62728"
            lbl_el = (r"$\uparrow$ espín" if is_up else r"$\downarrow$ espín") if (is_top and i_el in (0, n_up)) else None
            ax_tr.plot(
                t_grid,
                trace_plot[:, i_el],
                color=c_el,
                alpha=0.74,
                lw=0.88 * lw_scale,
                label=lbl_el,
            )

    if len(flips_t) > 0:
        for tf in flips_t[:65]:
            ax_tr.axvline(tf, color="#2ca02c", alpha=0.28, lw=0.8 * lw_scale, ls=":")

    hyst_rate = _safe_float(st.get("hyst_flips_total_mean"), 0.0)
    opp_rate = _safe_float(st.get("hyst_flips_opp_mean"), 0.0)
    ax_tr.set_ylabel(f"{m_disp}\n$x_i(t)$ [Bohr]", fontsize=9.2 * f_scale, weight="bold")
    ax_tr.set_ylim(-x_max, x_max)
    ax_tr.text(
        0.015,
        0.86,
        f"{m_disp}: {hyst_rate:.2f} flips/1k pasos ($\\uparrow\\downarrow$: {opp_rate:.2f})",
        transform=ax_tr.transAxes,
        fontsize=8.2 * f_scale,
        weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.24",
            facecolor="white",
            alpha=0.90,
            edgecolor=METHOD_COLORS.get(m_raw, "#555555"),
        ),
    )
    ax_tr.grid(True, alpha=0.25, linestyle="--")
    if is_top:
        if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
            t_txt = custom_title if custom_title is not None else (
                str(getattr(args, "title_panel_a", "") or "").strip()
                or f"{_panel_prefix('a', args)}Trazas de 1 Caminante ($N={N}, g={float(rec['g']):g}$)"
            )
            ax_tr.set_title(t_txt, fontsize=10.2 * f_scale, weight="bold", loc="left")
        _apply_legend(ax_tr, args, default_loc="upper right", fontsize=7.6)
    if not is_bottom:
        ax_tr.set_xticklabels([])
    else:
        ax_tr.set_xlabel("Paso de la cadena de Markov $t$", fontsize=9.2 * f_scale)

    x_c = np.asarray(st.get("marginal_x_centers", np.linspace(-x_max, x_max, 120)), dtype=np.float64)
    u_h = np.asarray(st.get("marginal_up_hist", np.zeros_like(x_c)), dtype=np.float64)
    d_h = np.asarray(st.get("marginal_dn_hist", np.zeros_like(x_c)), dtype=np.float64)
    bin_h = float(x_c[1] - x_c[0]) if len(x_c) > 1 else 0.25

    if n_up > 0 and np.any(u_h > 0):
        ax_hist.barh(
            x_c,
            u_h,
            height=bin_h * 1.02,
            color="#3498db",
            alpha=0.45,
            label=r"Muestreado $\uparrow$" if is_top else None,
        )
    if n_dn > 0 and np.any(d_h > 0):
        ax_hist.barh(
            x_c,
            d_h,
            height=bin_h * 1.02,
            color="#e74c3c",
            alpha=0.45,
            label=r"Muestreado $\downarrow$" if is_top else None,
        )

    ex_u = np.asarray(st.get("exact_up_hist", []), dtype=np.float64)
    ex_d = np.asarray(st.get("exact_dn_hist", []), dtype=np.float64)
    if casscf_ref is not None:
        x_ex = casscf_ref["x"]
        m_ex = (x_ex >= -x_max) & (x_ex <= x_max)
        if n_up > 0:
            ax_hist.plot(
                casscf_ref["p_up"][m_ex],
                x_ex[m_ex],
                color="#1f77b4",
                ls=":",
                lw=1.9 * lw_scale,
                label=r"CASSCF $\uparrow$" if is_top else None,
            )
        if n_dn > 0:
            ax_hist.plot(
                casscf_ref["p_down"][m_ex],
                x_ex[m_ex],
                color="#c0392b",
                ls=":",
                lw=1.9 * lw_scale,
                label=r"CASSCF $\downarrow$" if is_top else None,
            )
    elif len(ex_u) == len(x_c) and np.any(ex_u > 0):
        ax_hist.plot(ex_u, x_c, color="#1f77b4", ls=":", lw=1.9 * lw_scale, label=r"CASSCF $\uparrow$" if is_top else None)
        ax_hist.plot(ex_d, x_c, color="#c0392b", ls=":", lw=1.9 * lw_scale, label=r"CASSCF $\downarrow$" if is_top else None)

    ax_hist.xaxis.set_major_locator(MaxNLocator(nbins=2, prune="lower"))
    ax_hist.set_ylim(-x_max, x_max)
    ax_hist.tick_params(axis="y", labelleft=False)
    ax_hist.tick_params(axis="x", labelsize=8.0 * f_scale)
    ax_hist.grid(True, alpha=0.25, linestyle="--")
    if is_top:
        if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
            ax_hist.set_title(r"$p_\sigma(x)$", fontsize=9.4 * f_scale, weight="bold")
        _apply_legend(ax_hist, args, default_loc="upper right", fontsize=6.4)
    if not is_bottom:
        ax_hist.set_xticklabels([])
    else:
        ax_hist.set_xlabel(r"$p_\sigma(x)$", fontsize=8.8 * f_scale)


def _draw_transit_steps_histogram(
    ax_tr_hist: Any,
    records_sorted: List[Dict[str, Any]],
    args: argparse.Namespace,
    panel_letter: str = "b",
) -> None:
    f_scale = float(getattr(args, "font_scale", 1.0))
    bins = np.arange(1, 12) - 0.5
    width = 0.36
    x_centers = np.arange(1, 11)
    has_hist_data = False

    for idx_m, r in enumerate(records_sorted):
        m_raw = str(r.get("method", "MALA"))
        m_disp = _method_display_label(m_raw, args)
        col = METHOD_COLORS.get(m_raw, "#333333")
        st = r.get("stats", {})
        pct_direct = st.get("transit_pct_direct")
        ts = np.asarray(st.get("transit_steps_samples", []), dtype=np.int32)

        pct_vals: Optional[np.ndarray] = None
        if pct_direct is not None and len(pct_direct) == 10 and np.sum(pct_direct) > 0:
            pct_vals = np.asarray(pct_direct, dtype=np.float64)
        elif len(ts) > 0:
            ts_clipped = np.clip(ts, 1, 10)
            counts, _ = np.histogram(ts_clipped, bins=bins, density=True)
            pct_vals = counts * 100.0

        if pct_vals is not None:
            offset = (idx_m - 0.5 * (len(records_sorted) - 1)) * width
            f1 = _safe_float(st.get("one_step_fraction"), 0.0) * 100.0
            jmp = _safe_float(st.get("jump_at_flip_mean"), 0.0)
            ax_tr_hist.bar(
                x_centers + offset,
                pct_vals,
                width=width,
                color=col,
                alpha=0.84,
                edgecolor="black",
                linewidth=0.75,
                label=f"{m_disp}: {f1:.1f}% en 1 paso ($\\langle |\\Delta x_{{ij}}|\\rangle={jmp:.2f}$ Bohr)",
            )
            has_hist_data = True

    if not has_hist_data:
        ax_tr_hist.text(
            0.5,
            0.5,
            "No hubo cruces de barrera con histéresis ($|x_i - x_j| > \\delta$)\npara construir el histograma de tránsito",
            ha="center",
            va="center",
            transform=ax_tr_hist.transAxes,
            fontsize=10.0 * f_scale,
            color="#666666",
        )

    ax_tr_hist.set_xticks(x_centers)
    ax_tr_hist.set_xticklabels([str(k) if k < 10 else "10+" for k in x_centers])
    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        t_custom = str(getattr(args, "title_panel_b", "") or "").strip()
        ax_tr_hist.set_title(
            t_custom or f"{_panel_prefix(panel_letter, args)}Pasos de Markov Empleados en Cruzar la Barrera ($[-\\delta, +\\delta]$)",
            fontsize=10.8 * f_scale,
            weight="bold",
        )
    ax_tr_hist.set_xlabel("Duración del tránsito de barrera $\\Delta t_{\\mathrm{transit}}$ [pasos MCMC]", fontsize=9.6 * f_scale)
    ax_tr_hist.set_ylabel("Porcentaje de eventos de flip [%]", fontsize=9.6 * f_scale)
    ax_tr_hist.grid(True, alpha=0.28, linestyle="--", axis="y")
    if has_hist_data:
        _apply_legend(ax_tr_hist, args, default_loc="upper right", fontsize=8.4)


def _draw_single_method_heatmap(
    fig: Any,
    ax_hm: Any,
    rec: Dict[str, Any],
    args: argparse.Namespace,
    panel_letter: str = "c",
) -> None:
    """
    Dibuja el Mapa de Calor 2D del Stream-Flow Normalizado (x_i - x_j)/d_adj
    en pasos enteros discretos k = t - t0 (cruce entre k=0 y k=+1), limpio
    (sin encimar mu ni sigma encima del mapa de calor, siguiendo el Audio 3).
    """
    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))
    y_lim = max(1.5, float(getattr(args, "heatmap_ymax", 2.5)))

    m_raw = str(rec.get("method", "MALA"))
    m_disp = _method_display_label(m_raw, args)
    is_rl = (m_raw == "RL-MALA")
    col = METHOD_COLORS.get(m_raw, "#333333")
    st = rec.get("stats", {})
    eta_ref = _safe_float(st.get("hysteresis_frac"), 0.25)

    aligned_norm = np.asarray(st.get("aligned_rel_dist_norm", np.empty((0, 31))), dtype=np.float64)
    xi_norm_arr = np.asarray(st.get("aligned_xi_norm", np.empty((0, 31))), dtype=np.float64)
    xj_norm_arr = np.asarray(st.get("aligned_xj_norm", np.empty((0, 31))), dtype=np.float64)

    if aligned_norm.shape[0] > 0:
        W = (aligned_norm.shape[1] - 1) // 2
        rel_t = np.arange(-W, W + 1, dtype=np.float64)
        n_fine_t = 181
        n_bins_y = 120
        y_min, y_max = -y_lim, y_lim
        t_fine = np.linspace(-W, W, n_fine_t)
        y_edges = np.linspace(y_min, y_max, n_bins_y + 1)
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

        H_mat = np.zeros((n_bins_y, n_fine_t), dtype=np.float64)
        for ev_row in aligned_norm:
            y_interp = np.interp(t_fine, rel_t, ev_row)
            bin_idx = np.clip(np.searchsorted(y_edges, y_interp) - 1, 0, n_bins_y - 1)
            H_mat[bin_idx, np.arange(n_fine_t)] += 1.0

        H_smooth = _smooth_2d_heatmap(H_mat, sigma_y=2.2, sigma_t=1.2)
        col_max = np.max(H_smooth, axis=0, keepdims=True)
        H_norm = np.where(col_max > 1e-12, H_smooth / col_max, 0.0)

        cmap_name = "YlGnBu" if is_rl else "YlOrRd"
        pcm = ax_hm.pcolormesh(
            t_fine,
            y_centers,
            H_norm,
            cmap=cmap_name,
            vmin=0.0,
            vmax=1.0,
            shading="gouraud",
            rasterized=True,
        )
        cb = fig.colorbar(pcm, ax=ax_hm, pad=0.02, fraction=0.038)
        cb.set_label("Densidad normalizada", fontsize=8.0 * f_scale)
        cb.ax.tick_params(labelsize=7.2 * f_scale)

        med_d = np.median(aligned_norm, axis=0)
        q25_d = np.percentile(aligned_norm, 25, axis=0)
        q75_d = np.percentile(aligned_norm, 75, axis=0)
        q10_d = np.percentile(aligned_norm, 10, axis=0)
        q90_d = np.percentile(aligned_norm, 90, axis=0)

        ax_hm.plot(
            rel_t,
            med_d,
            color="#081d58" if is_rl else "#67000d",
            lw=2.3 * lw_scale,
            marker="o" if is_rl else "s",
            markersize=3.8,
            label=f"$\\mathrm{{med}}[(x_i - x_j)/d_{{\\mathrm{{adj}}}}]$ ({aligned_norm.shape[0]} flips)",
            zorder=5,
        )
        ax_hm.plot(rel_t, q25_d, color=col, ls="--", lw=1.05 * lw_scale, alpha=0.88, zorder=4)
        ax_hm.plot(rel_t, q75_d, color=col, ls="--", lw=1.05 * lw_scale, alpha=0.88, label="Cuartiles 25%–75%", zorder=4)
        ax_hm.plot(rel_t, q10_d, color=col, ls="-.", lw=0.75 * lw_scale, alpha=0.52, zorder=3)
        ax_hm.plot(rel_t, q90_d, color=col, ls="-.", lw=0.75 * lw_scale, alpha=0.52, label="Deciles 10%–90%", zorder=3)

        if xi_norm_arr.shape == aligned_norm.shape and xj_norm_arr.shape == aligned_norm.shape:
            med_xi = np.median(xi_norm_arr, axis=0)
            med_xj = np.median(xj_norm_arr, axis=0)
            ax_hm.plot(
                rel_t,
                med_xi,
                color="#008b8b",
                ls=":",
                lw=1.65 * lw_scale,
                label=r"$\tilde{x}_i(k)$ (izq $\rightarrow$ der)",
                zorder=4,
            )
            ax_hm.plot(
                rel_t,
                med_xj,
                color="#8b008b",
                ls=":",
                lw=1.65 * lw_scale,
                label=r"$\tilde{x}_j(k)$ (der $\rightarrow$ izq)",
                zorder=4,
            )

        for y_w in (-2.0, -1.0, 1.0, 2.0):
            if abs(y_w) < y_lim:
                ax_hm.axhline(y_w, color="#444444", lw=0.65, ls=":", alpha=0.45)
        ax_hm.axhspan(-eta_ref, +eta_ref, color="#555555", alpha=0.14)
        ax_hm.axhline(+eta_ref, color="#555555", lw=0.85, ls="--", alpha=0.65)
        ax_hm.axhline(-eta_ref, color="#555555", lw=0.85, ls="--", alpha=0.65)
        ax_hm.axhline(0.0, color="black", lw=1.1, ls="-.", alpha=0.85)
        # Sombrear el paso entero discreto del salto k=0 (t0) -> k=+1 (t0+1)
        ax_hm.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.16, label="Salto discreto ($t_0 \\to t_0+1$)")
        ax_hm.axvline(0.0, color="#2ca02c", lw=1.25, ls=":", alpha=0.95)
        ax_hm.axvline(1.0, color="#2ca02c", lw=1.25, ls=":", alpha=0.95)

        ax_hm.set_ylim(y_min, y_max)
        ax_hm.set_xlim(-W, W)
        _apply_legend(ax_hm, args, default_loc="upper left", fontsize=7.1, ncol=2)
    else:
        ax_hm.text(
            0.5,
            0.5,
            f"{m_disp}: Sin eventos de flip registrados\n(Congelamiento topológico)",
            ha="center",
            va="center",
            transform=ax_hm.transAxes,
            fontsize=9.8 * f_scale,
            weight="bold",
            color="#aa2222",
        )
        ax_hm.set_ylim(-y_lim, y_lim)
        ax_hm.set_xlim(-15, 15)

    ax_hm.set_ylabel(f"{m_disp}\n$(x_i - x_j)/d_{{\\mathrm{{adj}}}}$", fontsize=9.4 * f_scale, weight="bold")
    ax_hm.set_xlabel("Paso entero relativo al cruce $k = t - t_0$ (salto en $0 \\to +1$)", fontsize=9.4 * f_scale)
    ax_hm.grid(True, alpha=0.22, linestyle="--")

    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        if is_rl:
            t_custom = str(getattr(args, "title_panel_d", "") or "").strip()
            default_t = f"{_panel_prefix(panel_letter, args)}Transición Stream-Flow Normalizada ({m_disp})"
        else:
            t_custom = str(getattr(args, "title_panel_c", "") or "").strip()
            default_t = f"{_panel_prefix(panel_letter, args)}Transición Stream-Flow Normalizada ({m_disp})"
        ax_hm.set_title(t_custom or default_t, fontsize=10.6 * f_scale, weight="bold")


def _draw_actor_mu_sigma_outside_panel(
    ax: Any,
    rec_rl: Dict[str, Any],
    rec_mala: Optional[Dict[str, Any]],
    args: argparse.Namespace,
    panel_letter: str = "a",
) -> None:
    """
    Dibuja en un panel independiente (fuera del heatmap, según Audio 3) la
    predicción real de < sigma(k) >, < |mu_actor(k)| > y < |mu_total(k)| >
    evaluada con los pesos reales `actor_final` sobre pasos enteros k = t - t0.
    """
    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))
    st = rec_rl.get("stats", {})
    aligned_norm = np.asarray(st.get("aligned_rel_dist_norm", np.empty((0, 31))), dtype=np.float64)
    abs_mu_actor = np.asarray(st.get("aligned_abs_mu_actor", np.empty((0, 31))), dtype=np.float64)
    abs_mu_tot = np.asarray(st.get("aligned_abs_mu", np.empty((0, 31))), dtype=np.float64)
    sigma_arr = np.asarray(st.get("aligned_sigma", np.empty((0, 31))), dtype=np.float64)

    W = (aligned_norm.shape[1] - 1) // 2 if aligned_norm.shape[0] > 0 else 15
    rel_t = np.arange(-W, W + 1, dtype=np.float64)

    if sigma_arr.shape[0] > 0:
        mean_sig = np.mean(sigma_arr, axis=0)
        q25_sig = np.percentile(sigma_arr, 25, axis=0)
        q75_sig = np.percentile(sigma_arr, 75, axis=0)
        ax.plot(
            rel_t,
            mean_sig,
            color="#c51b8a",
            lw=2.3 * lw_scale,
            ls="-",
            marker="d",
            markersize=4.0,
            label=r"RL-MALA $\langle \sigma(k) \rangle = \sqrt{2\tau}\,\langle e^{a_\sigma(k)}\rangle$",
        )
        ax.fill_between(rel_t, q25_sig, q75_sig, color="#c51b8a", alpha=0.16)

    if abs_mu_tot.shape[0] > 0:
        mean_tot = np.mean(abs_mu_tot, axis=0)
        q25_tot = np.percentile(abs_mu_tot, 25, axis=0)
        q75_tot = np.percentile(abs_mu_tot, 75, axis=0)
        ax.plot(
            rel_t,
            mean_tot,
            color="#ff7f00",
            lw=2.3 * lw_scale,
            marker="^",
            markersize=4.2,
            label=r"RL-MALA Deriva Total $\langle |\tau\nabla\ln|\Psi| + \mu_{\mathrm{actor}}| \rangle$",
        )
        ax.fill_between(rel_t, q25_tot, q75_tot, color="#ff7f00", alpha=0.18)

    if abs_mu_actor.shape[0] > 0 and np.max(abs_mu_actor) > 0:
        mean_act = np.mean(abs_mu_actor, axis=0)
        ax.plot(
            rel_t,
            mean_act,
            color="#1f77b4",
            lw=1.9 * lw_scale,
            ls="--",
            marker="o",
            markersize=3.4,
            label=r"RL-MALA Deriva Pura del Actor $\langle |A_{\mathrm{actor}}\,a_\mu(k)| \rangle$",
        )

    tau_m = _safe_float(rec_mala.get("tau") if rec_mala else rec_rl.get("tau"), 0.08)
    sig_mala = math.sqrt(2.0 * max(tau_m, 1e-4))
    ax.axhline(
        sig_mala,
        color="#d62728",
        ls=":",
        lw=1.8 * lw_scale,
        label=f"MALA fijo $\\sigma_{{\\mathrm{{MALA}}}} = \\sqrt{{2\\tau}} = {sig_mala:.3f}$ Bohr ($\\mu_{{\\mathrm{{actor}}}}=0$)",
    )
    ax.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.15, label="Paso exacto del cruce ($k=0 \\to +1$)")
    ax.axvline(0.0, color="#2ca02c", lw=1.3 * lw_scale, ls=":")
    ax.axvline(1.0, color="#2ca02c", lw=1.3 * lw_scale, ls=":")

    ax.set_xlim(-W, W)
    ax.set_xlabel("Paso entero relativo a la transición $k = t - t_0$", fontsize=9.6 * f_scale)
    ax.set_ylabel("Parámetros de Propuesta [Bohr]", fontsize=9.6 * f_scale, weight="bold")
    ax.grid(True, alpha=0.26, linestyle="--")
    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        t_custom = str(getattr(args, "title_actor_panel", "") or "").strip()
        ax.set_title(
            t_custom or f"{_panel_prefix(panel_letter, args)}Parámetros de Propuesta del Actor ($\\langle \\sigma(k) \\rangle$ y $\\langle |\\mu(k)| \\rangle$) Fuera del Mapa de Calor",
            fontsize=10.6 * f_scale,
            weight="bold",
        )
    _apply_legend(ax, args, default_loc="best", fontsize=7.8)


def _draw_average_step_magnitude_panel(
    ax: Any,
    records_sorted: List[Dict[str, Any]],
    args: argparse.Namespace,
    panel_letter: str = "b",
) -> None:
    """
    Dibuja la gráfica solicitada en el Audio 5 (16.28.35):
    Magnitud del paso promedio real aceptado < |Delta x(k)| > = <|x(t0+k+1) - x(t0+k)|>
    en función del paso relativo k = t - t0 in [-15, +15], comparando MALA vs RL-MALA.
    """
    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))
    W = 15

    for r in records_sorted:
        m_raw = str(r.get("method", "MALA"))
        m_disp = _method_display_label(m_raw, args)
        col = METHOD_COLORS.get(m_raw, "#333333")
        st = r.get("stats", {})
        dx_pair = np.asarray(st.get("aligned_step_mag_pair", np.empty((0, 31))), dtype=np.float64)
        dx_spec = np.asarray(st.get("aligned_step_mag_spec", np.empty((0, 31))), dtype=np.float64)

        if dx_pair.shape[0] > 0:
            W = (dx_pair.shape[1] - 1) // 2
            rel_t = np.arange(-W, W + 1, dtype=np.float64)
            mean_p = np.mean(dx_pair, axis=0)
            q25_p = np.percentile(dx_pair, 25, axis=0)
            q75_p = np.percentile(dx_pair, 75, axis=0)

            ax.plot(
                rel_t,
                mean_p,
                color=col,
                lw=2.4 * lw_scale,
                marker="o" if m_raw == "RL-MALA" else "s",
                markersize=4.2,
                label=f"{m_disp} — Par en transición $\\langle |\\Delta x_{{i,j}}(k)| \\rangle$ (pico={float(np.max(mean_p)):.2f} Bohr)",
                zorder=5 if m_raw == "RL-MALA" else 4,
            )
            ax.fill_between(rel_t, q25_p, q75_p, color=col, alpha=0.16)

            if dx_spec.shape == dx_pair.shape and np.any(dx_spec > 0):
                mean_s = np.mean(dx_spec, axis=0)
                ax.plot(
                    rel_t,
                    mean_s,
                    color=col,
                    lw=1.3 * lw_scale,
                    ls="--",
                    alpha=0.65,
                    label=f"{m_disp} — Electrones espectadores $\\langle |\\Delta x_{{\\mathrm{{esp}}}}(k)| \\rangle$",
                )

    ax.axvline(0.0, color="#2ca02c", lw=1.4 * lw_scale, ls=":", label="Paso exacto del cruce ($k = 0$: salto $t_0 \\to t_0+1$)")
    ax.set_xlim(-W, W)
    ax.set_xlabel("Paso entero relativo a la transición $k = t - t_0$", fontsize=9.6 * f_scale)
    ax.set_ylabel("Magnitud del paso promedio $\\langle |\\Delta x(k)| \\rangle$ [Bohr]", fontsize=9.6 * f_scale, weight="bold")
    ax.grid(True, alpha=0.26, linestyle="--")
    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        ax.set_title(
            f"{_panel_prefix(panel_letter, args)}Magnitud del Paso Promedio $\\langle |x(t_0+k+1) - x(t_0+k)| \\rangle$ Centrada en la Transición",
            fontsize=10.6 * f_scale,
            weight="bold",
        )
    _apply_legend(ax, args, default_loc="upper right", fontsize=7.8)


def _draw_single_walker_exact_steps_1d(
    ax: Any,
    rec: Dict[str, Any],
    casscf_ref: Optional[Dict[str, np.ndarray]],
    x_max: float,
    args: argparse.Namespace,
    mode_key: str = "opp_single",
    panel_letter: str = "a",
    spin_kind_label: str = "Espín Opuesto $\\uparrow\\downarrow$",
) -> None:
    """
    Dibuja para UN SOLO CAMINANTE REAL las posiciones exactas de sus N electrones
    en los 4 pasos enteros consecutivos alrededor del flip:
      - t0 - 1 (antes)
      - t0     (justo antes del brinco: siguen de su lado inicial)
      - t0 + 1 (justo después del brinco: ¡ya se brincaron uno al otro!)
      - t0 + 2 (después)
    """
    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))

    m_raw = str(rec.get("method", "MALA"))
    m_disp = _method_display_label(m_raw, args)
    N = int(rec["N"])
    st = rec.get("stats", {})
    ba = st.get("before_after_info")
    if ba is None or f"{mode_key}_x" not in ba or ba[f"{mode_key}_x"] is None:
        ax.text(
            0.5,
            0.5,
            f"{m_disp}: Sin eventos de flip ({spin_kind_label}) registrados",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9.5 * f_scale,
            weight="bold",
            color="#aa2222",
        )
        return

    rel_s = np.asarray(ba["rel_steps"], dtype=int)
    x_mat = np.asarray(ba[f"{mode_key}_x"], dtype=np.float64)
    spins = np.asarray(ba[f"{mode_key}_spins"], dtype=int)
    w_l = int(ba.get(f"{mode_key}_well_left", 0))
    w_r = int(ba.get(f"{mode_key}_well_right", 1))
    t0_abs = int(ba.get(f"{mode_key}_t0", 0))
    w_idx = int(ba.get(f"{mode_key}_walker", 0))
    tr_steps = int(ba.get(f"{mode_key}_transit_steps", 1))

    target_k = [-1, 0, 1, 2]
    idx_levels = [int(np.argmin(np.abs(rel_s - k_v))) for k_v in target_k]
    y_levels = [3.0, 2.0, 1.0, 0.0]
    level_names = [
        f"Paso $t_0 - 1$\n($t={t0_abs-1}$)",
        f"Justo ANTES ($t_0$)\n($t={t0_abs}$)",
        f"Justo DESPUÉS ($t_0+1$)\n($t={t0_abs+1}$, ¡ya brincó!)",
        f"Paso $t_0 + 2$\n($t={t0_abs+2}$)",
    ]

    if casscf_ref is not None:
        x_ex = casscf_ref["x"]
        rho_ex = casscf_ref["rho_total"]
        rho_norm = rho_ex / max(float(np.max(rho_ex)), 1e-8) * 0.34
        for y_lv in y_levels:
            ax.fill_between(x_ex, y_lv - 0.16, y_lv - 0.16 + rho_norm, color="#888888", alpha=0.13, zorder=1)
            ax.axhline(y_lv, color="#777777", lw=0.75, ls="-", alpha=0.35, zorder=1)
    else:
        for y_lv in y_levels:
            ax.axhline(y_lv, color="#777777", lw=0.75, ls="-", alpha=0.35, zorder=1)

    # Sombrear la banda entre t0 (y=2.0) y t0+1 (y=1.0) donde ocurre el salto discreto
    ax.axhspan(1.0, 2.0, color="#2ca02c", alpha=0.08, zorder=1)

    for m_el in range(N):
        xs_4 = [float(x_mat[idx_l, m_el]) for idx_l in idx_levels]
        is_flipping = (m_el in (w_l, w_r))
        is_up = (spins[m_el] > 0)
        c_el = "#1f77b4" if is_up else "#d62728"
        if is_flipping:
            ax.plot(xs_4, y_levels, color=c_el, lw=2.4 * lw_scale, ls="-", alpha=0.92, zorder=4)
            ax.annotate(
                "",
                xy=(xs_4[2], y_levels[2] + 0.06),
                xytext=(xs_4[1], y_levels[1] - 0.06),
                arrowprops=dict(arrowstyle="->", color=c_el, lw=2.2 * lw_scale),
                zorder=6,
            )
        else:
            ax.plot(xs_4, y_levels, color="#666666", lw=1.0 * lw_scale, ls="--", alpha=0.48, zorder=2)

    for lv_i, (y_lv, idx_t) in enumerate(zip(y_levels, idx_levels)):
        for m_el in range(N):
            xv = float(x_mat[idx_t, m_el])
            is_up = (spins[m_el] > 0)
            is_flipping = (m_el in (w_l, w_r))
            c_el = "#1f77b4" if is_up else "#d62728"
            marker = "o" if is_up else "s"
            lbl = None
            if lv_i == 0:
                if is_up and m_el == int(np.where(spins > 0)[0][0]):
                    lbl = r"Electrón $\uparrow$"
                elif (not is_up) and np.any(spins < 0) and m_el == int(np.where(spins < 0)[0][0]):
                    lbl = r"Electrón $\downarrow$"

            ax.scatter(
                [xv],
                [y_lv],
                color=c_el,
                marker=marker,
                s=115 if is_flipping else 64,
                edgecolors="#ffd700" if is_flipping else "black",
                linewidths=2.0 if is_flipping else 0.8,
                label=lbl,
                zorder=6 if is_flipping else 4,
            )
            arrow_sym = "↑" if is_up else "↓"
            txt_str = f"$e_{{{m_el+1}}}{arrow_sym}={xv:+.1f}$" if (is_flipping and lv_i in (1, 2)) else f"$e_{{{m_el+1}}}{arrow_sym}$"
            ax.text(
                xv,
                y_lv + 0.16,
                txt_str,
                ha="center",
                va="bottom",
                fontsize=(7.6 if is_flipping else 6.8) * f_scale,
                weight="bold" if is_flipping else "normal",
                color=c_el,
                bbox=dict(boxstyle="round,pad=0.10", facecolor="white", alpha=0.82, edgecolor="none") if is_flipping else None,
                zorder=7,
            )

    x_l_t0 = float(x_mat[idx_levels[1], w_l])
    x_r_t0 = float(x_mat[idx_levels[1], w_r])
    x_l_t1 = float(x_mat[idx_levels[2], w_l])
    x_r_t1 = float(x_mat[idx_levels[2], w_r])
    jump_d = abs((x_l_t1 - x_r_t1) - (x_l_t0 - x_r_t0))
    ax.text(
        0.015,
        0.04,
        f"Caminante #{w_idx} ($\\Delta t_{{\\mathrm{{transit}}}}={tr_steps}$p, $|\\Delta d_{{ij}}|={jump_d:.2f}$ Bohr)",
        transform=ax.transAxes,
        fontsize=8.0 * f_scale,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.90, edgecolor=METHOD_COLORS.get(m_raw, "#555")),
        zorder=8,
    )

    ax.set_yticks(y_levels)
    ax.set_yticklabels(level_names, fontsize=8.3 * f_scale, weight="bold")
    ax.set_ylim(-0.45, 3.60)
    ax.set_xlim(-x_max, x_max)
    ax.set_xlabel("Posición exacta de los electrones $x$ [Bohr]", fontsize=9.2 * f_scale)
    ax.grid(True, alpha=0.24, linestyle="--", axis="x")
    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        ax.set_title(
            f"{_panel_prefix(panel_letter, args)}Paso Exacto de Transición ({m_disp} — {spin_kind_label})",
            fontsize=10.2 * f_scale,
            weight="bold",
        )
    _apply_legend(ax, args, default_loc="upper right", fontsize=7.6, ncol=2)


def _draw_single_walker_worldlines(
    ax: Any,
    rec: Dict[str, Any],
    x_max: float,
    args: argparse.Namespace,
    mode_key: str = "opp_single",
    panel_letter: str = "c",
    spin_kind_label: str = "Espín Opuesto $\\uparrow\\downarrow$",
) -> None:
    """
    Dibuja las trayectorias discretas paso por paso entero k = t - t0 in [-10, +10]
    de los N electrones de UN SOLO CAMINANTE REAL durante el flip.
    """
    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))

    m_raw = str(rec.get("method", "MALA"))
    m_disp = _method_display_label(m_raw, args)
    N = int(rec["N"])
    st = rec.get("stats", {})
    ba = st.get("before_after_info")
    if ba is None or f"{mode_key}_x" not in ba or ba[f"{mode_key}_x"] is None:
        return

    rel_s = np.asarray(ba["rel_steps"], dtype=np.float64)
    x_mat = np.asarray(ba[f"{mode_key}_x"], dtype=np.float64)
    spins = np.asarray(ba[f"{mode_key}_spins"], dtype=int)
    w_l = int(ba.get(f"{mode_key}_well_left", 0))
    w_r = int(ba.get(f"{mode_key}_well_right", 1))

    mask_w = (rel_s >= -10) & (rel_s <= 10)
    rel_sub = rel_s[mask_w]
    x_sub = x_mat[mask_w]

    for m_el in range(N):
        is_up = (spins[m_el] > 0)
        is_flipping = (m_el in (w_l, w_r))
        c_el = "#1f77b4" if is_up else "#d62728"
        if is_flipping:
            role_lbl = f"$e_{{{m_el+1}}} (\\uparrow)$ brinca" if is_up else f"$e_{{{m_el+1}}} (\\downarrow)$ brinca"
            ax.plot(
                rel_sub,
                x_sub[:, m_el],
                color=c_el,
                lw=2.4 * lw_scale,
                marker="o" if is_up else "s",
                markersize=4.5,
                label=role_lbl,
                zorder=5,
            )
        else:
            lbl_spec = (r"Espectadores $\uparrow$" if is_up else r"Espectadores $\downarrow$") if m_el in (0, N - 1) else None
            ax.plot(
                rel_sub,
                x_sub[:, m_el],
                color=c_el,
                lw=1.1 * lw_scale,
                ls="--",
                marker=".",
                markersize=3.0,
                alpha=0.52,
                label=lbl_spec,
                zorder=2,
            )

    ax.axvspan(0.0, 1.0, color="#2ca02c", alpha=0.18, label="Salto exacto ($t_0 \\to t_0+1$)")
    ax.axvline(0.0, color="#2ca02c", lw=1.3, ls=":")
    ax.axvline(1.0, color="#2ca02c", lw=1.3, ls=":")

    ax.set_xlim(float(rel_sub[0]), float(rel_sub[-1]))
    ax.set_xticks(np.arange(int(rel_sub[0]), int(rel_sub[-1]) + 1, 2))
    ax.set_ylim(-x_max, x_max)
    ax.set_xlabel("Paso entero relativo $k = t - t_0$ (cruce entre $k=0$ y $k=+1$)", fontsize=9.2 * f_scale)
    ax.set_ylabel("Posición $x_m(t_0+k)$ [Bohr]", fontsize=9.2 * f_scale, weight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")
    if not getattr(args, "no_titles", False) and not getattr(args, "no_panel_titles", False):
        ax.set_title(
            f"{_panel_prefix(panel_letter, args)}Trayectoria Discreta Paso a Paso de 1 Caminante ({m_disp} — {spin_kind_label})",
            fontsize=10.2 * f_scale,
            weight="bold",
        )
    _apply_legend(ax, args, default_loc="upper right", fontsize=7.3, ncol=2)


def _apply_suptitle_and_save(
    fig: Any,
    out_dir: str,
    stem: str,
    default_suptitle: str,
    args: argparse.Namespace,
) -> List[str]:
    import matplotlib.pyplot as plt

    f_scale = float(getattr(args, "font_scale", 1.0))
    if default_suptitle and not getattr(args, "no_titles", False) and not getattr(args, "no_suptitle", False):
        custom_sup = str(getattr(args, "custom_suptitle", "") or "").strip()
        custom_sub = str(getattr(args, "custom_subtitle", "") or "").strip()
        t_prefix = str(getattr(args, "title_prefix", "") or "").strip()
        sup_text = custom_sup if custom_sup else default_suptitle
        if t_prefix and not sup_text.startswith(t_prefix):
            sup_text = f"{t_prefix} — {sup_text}"
        if custom_sub and not getattr(args, "no_panel_titles", False):
            sup_text = f"{sup_text}\n{custom_sub}"
        fig.suptitle(sup_text, fontsize=13.0 * f_scale, weight="bold", y=0.995)

    fmt = str(getattr(args, "figure_format", "png")).lower()
    dpi = int(getattr(args, "dpi", 300))
    transparent = bool(getattr(args, "transparent_bg", False))

    saved_paths: List[str] = []
    if fmt in ("png", "both"):
        p_png = os.path.join(out_dir, f"{stem}.png")
        _atomic_savefig(fig, p_png, "png", dpi=dpi, transparent=transparent)
        saved_paths.append(p_png)
        print(f"  [+] Figura guardada: {p_png}")
    if fmt in ("pdf", "both"):
        p_pdf = os.path.join(out_dir, f"{stem}.pdf")
        _atomic_savefig(fig, p_pdf, "pdf", dpi=dpi, transparent=transparent)
        saved_paths.append(p_pdf)
        print(f"  [+] Figura guardada: {p_pdf}")
    plt.close(fig)
    return saved_paths


def generate_all_system_flip_figures(
    records_ng: List[Dict[str, Any]],
    out_dir: str,
    args: argparse.Namespace,
    prod_root: Optional[str] = None,
) -> List[str]:
    """
    Genera para un sistema `(pipe, N, omega, g)` el conjunto completo de figuras
    corregidas según los audios del asesor:
      1. `fig_flips_system_<tag>` (4 paneles principales limpios sin encimar mu/sigma en el heatmap)
      2. `fig_flips_actor_and_step_size_<tag>` (mu y sigma fuera del heatmap + magnitud del paso promedio vs tiempo)
      3. `fig_flips_before_after_<tag>` (paso exacto t0 -> t0+1 de un caminante individual en espín opuesto)
      4. `fig_flips_opp_vs_same_spin_<tag>` (comparación espín opuesto up-down vs mismo espín up-up/down-down)
      5. Paneles individuales en `<out_dir>/panels/`.
    """
    import matplotlib.pyplot as plt

    if not records_ng:
        return []

    rec0 = records_ng[0]
    N = int(rec0["N"])
    omega = float(rec0["omega"])
    g_val = float(rec0["g"])
    pipe = str(rec0["pipeline"])
    sys_tag = f"{pipe}_N{N}_w{omega:g}_g{g_val:g}"

    order_map = {"MALA": 0, "Adaptive-MALA": 1, "RL-MALA": 2}
    records_sorted = sorted(records_ng, key=lambda r: order_map.get(str(r.get("method", "")), 99))
    rec_mala = next((r for r in records_sorted if r["method"] == "MALA"), records_sorted[0])
    rec_rl = next((r for r in records_sorted if r["method"] == "RL-MALA"), records_sorted[-1])

    d_adj = _safe_float(rec_rl["stats"].get("d_adj"), _safe_float(rec_mala["stats"].get("d_adj"), 4.5))
    user_xmax = _safe_float(getattr(args, "trace_xmax", None))
    if np.isfinite(user_xmax) and user_xmax > 0:
        x_max = float(user_xmax)
    else:
        tr_max = 12.0
        for r in records_sorted:
            tr = np.asarray(r["stats"].get("best_walker_trace", np.empty((0, N))), dtype=np.float64)
            if tr.size > 0:
                tr_max = max(tr_max, float(np.nanpercentile(np.abs(tr), 99.5)) * 1.18)
        x_max = float(np.ceil(max(12.0, tr_max, 0.56 * N * d_adj + 4.0)))

    search_roots = [p for p in [prod_root, out_dir, PROJECT_ROOT] if p]
    npz_path = _find_casscf_npz_for_system(search_roots, N, omega, g_val)
    casscf_ref = _compute_exact_casscf_density_np(npz_path, x_max=x_max, n_grid=300) if npz_path else None

    regime_str = "Molécula de Wigner Formada" if g_val >= 0.45 else "Régimen Pre-Wigner / Gas de Fermi"
    all_saved: List[str] = []

    # ==========================================================================
    # FIGURA 1: 4 Paneles Principales Limpios (mu y sigma FUERA del heatmap)
    # ==========================================================================
    fig1 = plt.figure(figsize=(16.5, 11.4))
    gs1 = fig1.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    gs_top_left = gs1[0, 0].subgridspec(len(records_sorted), 2, width_ratios=[3.8, 1.25], hspace=0.18, wspace=0.05)
    for idx_m, r in enumerate(records_sorted):
        ax_tr = fig1.add_subplot(gs_top_left[idx_m, 0])
        ax_hist = fig1.add_subplot(gs_top_left[idx_m, 1])
        _draw_single_trace_and_marginal(
            ax_tr,
            ax_hist,
            r,
            casscf_ref,
            x_max=x_max,
            args=args,
            is_top=(idx_m == 0),
            is_bottom=(idx_m == len(records_sorted) - 1),
        )

    ax_transit = fig1.add_subplot(gs1[0, 1])
    _draw_transit_steps_histogram(ax_transit, records_sorted, args, panel_letter="b")

    ax_hm_mala = fig1.add_subplot(gs1[1, 0])
    _draw_single_method_heatmap(fig1, ax_hm_mala, rec_mala, args, panel_letter="c")

    ax_hm_rl = fig1.add_subplot(gs1[1, 1])
    _draw_single_method_heatmap(fig1, ax_hm_rl, rec_rl, args, panel_letter="d")

    sup1 = (
        f"Caracterización de Flips y Transición Stream-Flow ({pipe}, {regime_str}) — "
        f"$N={N}, \\omega={omega:g}, g={g_val:g}$ ($d_{{\\mathrm{{adj}}}}={d_adj:.2f}$ Bohr)"
    )
    all_saved.extend(_apply_suptitle_and_save(fig1, out_dir, f"fig_flips_system_{sys_tag}", sup1, args))

    # ==========================================================================
    # FIGURA 2: Actor (mu, sigma) Fuera del Heatmap + Magnitud del Paso Promedio
    # ==========================================================================
    fig2, (ax_act_out, ax_step_mag) = plt.subplots(2, 1, figsize=(11.5, 9.2), sharex=True)
    fig2.subplots_adjust(hspace=0.26)
    _draw_actor_mu_sigma_outside_panel(ax_act_out, rec_rl, rec_mala, args, panel_letter="a")
    _draw_average_step_magnitude_panel(ax_step_mag, records_sorted, args, panel_letter="b")
    sup2 = (
        f"Acción del Actor ($\\mu, \\sigma$) y Magnitud del Paso Promedio en la Transición ({pipe}) — "
        f"$N={N}, \\omega={omega:g}, g={g_val:g}$"
    )
    all_saved.extend(_apply_suptitle_and_save(fig2, out_dir, f"fig_flips_actor_and_step_size_{sys_tag}", sup2, args))

    # ==========================================================================
    # FIGURA 3: Paso Exacto de 1 Caminante Individual (Espín Opuesto up-down)
    # ==========================================================================
    fig3 = plt.figure(figsize=(16.5, 10.8))
    gs3 = fig3.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    ax_ba_mala = fig3.add_subplot(gs3[0, 0])
    _draw_single_walker_exact_steps_1d(
        ax_ba_mala, rec_mala, casscf_ref, x_max=x_max, args=args,
        mode_key="opp_single", panel_letter="a", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
    )
    ax_ba_rl = fig3.add_subplot(gs3[0, 1])
    _draw_single_walker_exact_steps_1d(
        ax_ba_rl, rec_rl, casscf_ref, x_max=x_max, args=args,
        mode_key="opp_single", panel_letter="b", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
    )
    ax_wl_mala = fig3.add_subplot(gs3[1, 0])
    _draw_single_walker_worldlines(
        ax_wl_mala, rec_mala, x_max=x_max, args=args,
        mode_key="opp_single", panel_letter="c", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
    )
    ax_wl_rl = fig3.add_subplot(gs3[1, 1])
    _draw_single_walker_worldlines(
        ax_wl_rl, rec_rl, x_max=x_max, args=args,
        mode_key="opp_single", panel_letter="d", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
    )
    sup3 = (
        f"Paso Exacto de Transición ($t_0 \\to t_0+1$) de 1 Caminante Individual ({pipe}, {regime_str}) — "
        f"$N={N}, \\omega={omega:g}, g={g_val:g}$"
    )
    t_ba_override = str(getattr(args, "title_before_after", "") or "").strip()
    all_saved.extend(_apply_suptitle_and_save(fig3, out_dir, f"fig_flips_before_after_{sys_tag}", t_ba_override or sup3, args))

    # ==========================================================================
    # FIGURA 4: Comparación Espines Opuestos (up-down) vs Mismo Espín (Pauli)
    # ==========================================================================
    if N >= 4:
        fig4 = plt.figure(figsize=(16.5, 10.8))
        gs4 = fig4.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

        ax_opp_1d = fig4.add_subplot(gs4[0, 0])
        _draw_single_walker_exact_steps_1d(
            ax_opp_1d, rec_rl, casscf_ref, x_max=x_max, args=args,
            mode_key="opp_single", panel_letter="a", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$ (Barrera Wigner)",
        )
        ax_same_1d = fig4.add_subplot(gs4[0, 1])
        _draw_single_walker_exact_steps_1d(
            ax_same_1d, rec_rl, casscf_ref, x_max=x_max, args=args,
            mode_key="same_single", panel_letter="b", spin_kind_label="Mismo Espín $\\uparrow\\uparrow / \\downarrow\\downarrow$ (Nodo de Pauli)",
        )
        ax_opp_wl = fig4.add_subplot(gs4[1, 0])
        _draw_single_walker_worldlines(
            ax_opp_wl, rec_rl, x_max=x_max, args=args,
            mode_key="opp_single", panel_letter="c", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
        )
        ax_same_wl = fig4.add_subplot(gs4[1, 1])
        _draw_single_walker_worldlines(
            ax_same_wl, rec_rl, x_max=x_max, args=args,
            mode_key="same_single", panel_letter="d", spin_kind_label="Mismo Espín $\\uparrow\\uparrow / \\downarrow\\downarrow$",
        )
        sup4 = (
            f"Comparación de Cruce en 1 Caminante: Espín Opuesto ($\\uparrow\\downarrow$) vs Mismo Espín (Nodo de Pauli) — "
            f"$N={N}, \\omega={omega:g}, g={g_val:g}$ ({regime_str})"
        )
        all_saved.extend(_apply_suptitle_and_save(fig4, out_dir, f"fig_flips_opp_vs_same_spin_{sys_tag}", sup4, args))

    # ==========================================================================
    # 5. PANELES INDIVIDUALES EN `<out_dir>/panels/`
    # ==========================================================================
    if getattr(args, "save_panels", True):
        panels_dir = os.path.join(out_dir, "panels")
        os.makedirs(panels_dir, exist_ok=True)

        fig_pa = plt.figure(figsize=(8.6, 5.8))
        gs_pa = fig_pa.add_gridspec(len(records_sorted), 2, width_ratios=[3.8, 1.25], hspace=0.20, wspace=0.05)
        for idx_m, r in enumerate(records_sorted):
            ax_t = fig_pa.add_subplot(gs_pa[idx_m, 0])
            ax_h = fig_pa.add_subplot(gs_pa[idx_m, 1])
            _draw_single_trace_and_marginal(
                ax_t, ax_h, r, casscf_ref, x_max=x_max, args=args,
                is_top=(idx_m == 0), is_bottom=(idx_m == len(records_sorted) - 1),
            )
        all_saved.extend(_apply_suptitle_and_save(fig_pa, panels_dir, f"panel_a_traces_with_marginal_{sys_tag}", "", args))

        fig_pb, ax_pb = plt.subplots(figsize=(7.8, 5.2))
        _draw_transit_steps_histogram(ax_pb, records_sorted, args, panel_letter="b")
        all_saved.extend(_apply_suptitle_and_save(fig_pb, panels_dir, f"panel_b_transit_steps_hist_{sys_tag}", "", args))

        fig_pc, ax_pc = plt.subplots(figsize=(8.2, 5.4))
        _draw_single_method_heatmap(fig_pc, ax_pc, rec_mala, args, panel_letter="c")
        all_saved.extend(_apply_suptitle_and_save(fig_pc, panels_dir, f"panel_c_heatmap_MALA_{sys_tag}", "", args))

        fig_pd, ax_pd = plt.subplots(figsize=(8.2, 5.4))
        _draw_single_method_heatmap(fig_pd, ax_pd, rec_rl, args, panel_letter="d")
        all_saved.extend(_apply_suptitle_and_save(fig_pd, panels_dir, f"panel_d_heatmap_RL_MALA_{sys_tag}", "", args))

        fig_pe, ax_pe = plt.subplots(figsize=(8.4, 4.8))
        _draw_actor_mu_sigma_outside_panel(ax_pe, rec_rl, rec_mala, args, panel_letter="e")
        all_saved.extend(_apply_suptitle_and_save(fig_pe, panels_dir, f"panel_e_actor_mu_sigma_profile_{sys_tag}", "", args))

        fig_pf, ax_pf = plt.subplots(figsize=(8.4, 4.8))
        _draw_average_step_magnitude_panel(ax_pf, records_sorted, args, panel_letter="f")
        all_saved.extend(_apply_suptitle_and_save(fig_pf, panels_dir, f"panel_f_average_step_magnitude_{sys_tag}", "", args))

        for r in records_sorted:
            m_tag = str(r["method"]).replace("-", "_")
            fig_pg, ax_pg = plt.subplots(figsize=(8.4, 5.0))
            _draw_single_walker_exact_steps_1d(
                ax_pg, r, casscf_ref, x_max=x_max, args=args,
                mode_key="opp_single", panel_letter="g", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
            )
            all_saved.extend(_apply_suptitle_and_save(fig_pg, panels_dir, f"panel_g_exact_steps_opp_{m_tag}_{sys_tag}", "", args))

            fig_ph, ax_ph = plt.subplots(figsize=(8.4, 5.0))
            _draw_single_walker_worldlines(
                ax_ph, r, x_max=x_max, args=args,
                mode_key="opp_single", panel_letter="h", spin_kind_label="Espín Opuesto $\\uparrow\\downarrow$",
            )
            all_saved.extend(_apply_suptitle_and_save(fig_ph, panels_dir, f"panel_h_worldlines_opp_{m_tag}_{sys_tag}", "", args))

    return all_saved


# ==============================================================================
# 5. INTERFAZ DE LÍNEA DE COMANDOS (CLI)
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Genera las figuras detalladas de flips por sistema (N, g) a partir de "
            "datos reales HDF5 o CSVs de producción, sin interpolaciones artificiales."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input_path",
        nargs="?",
        default="results_nodal_crossing_search_N6",
        help="Carpeta con archivos run_bundle.h5 o carpeta results_csv con tablas reales de flips.",
    )
    p.add_argument("--input_dir", "--prod_dir", "--results_csv", "--root_dir", dest="input_dir_flag", default=None, help="Alias con bandera para la carpeta de entrada.")
    p.add_argument("--out_dir", default=None, help="Carpeta donde guardar las nuevas figuras.")
    p.add_argument("--out_subdir", default="figures_flips_detailed", help="Subcarpeta dentro de la producción donde guardar las figuras si no se da --out_dir.")
    p.add_argument("--n", "--n_elec", "--n_filter", dest="n_filter", type=int, nargs="+", default=None, help="Filtrar por número de electrones N.")
    p.add_argument("--g", "--g_filter", dest="g_filter", type=float, nargs="+", default=None, help="Filtrar por acoplamiento g.")
    p.add_argument("--omega", type=float, default=0.1, help="Frecuencia de la trampa armónica omega.")
    p.add_argument("--system", "--systems", dest="systems_filter", nargs="+", default=None, help="Filtro por sistema (ej. N6_w0.1_g1.0 o N8:2.0).")
    p.add_argument("--pipeline", default="P2", choices=["P2", "P3", "P4"], help="Etapa del pipeline.")
    p.add_argument("--regenerate_csv", action="store_true", help="Recalcular CSVs desde run_bundle.h5 antes de graficar.")

    g_fmt = p.add_argument_group("Opciones de Formato, Resolución y Paneles Individuales")
    g_fmt.add_argument("--figure_format", "--format", dest="figure_format", choices=["png", "pdf", "both"], default="png", help="Formato de salida: png, pdf o both.")
    g_fmt.add_argument("--dpi", type=int, default=300, help="Resolución en DPI.")
    g_fmt.add_argument("--save_panels", action=argparse.BooleanOptionalAction, default=True, help="Guardar paneles individuales en <out_dir>/panels/.")
    g_fmt.add_argument("--transparent_bg", action=argparse.BooleanOptionalAction, default=False, help="Fondo transparente.")
    g_fmt.add_argument("--font_scale", type=float, default=1.0, help="Escala de fuente.")
    g_fmt.add_argument("--line_width_scale", type=float, default=1.0, help="Escala de grosor de línea.")

    g_txt = p.add_argument_group("Banderas para Desactivar o Modificar Leyendas, Títulos y Subtítulos")
    g_txt.add_argument("--no_titles", "--hide_titles", dest="no_titles", action=argparse.BooleanOptionalAction, default=False, help="Desactivar todos los títulos.")
    g_txt.add_argument("--no_suptitle", "--hide_suptitle", dest="no_suptitle", action=argparse.BooleanOptionalAction, default=False, help="Desactivar supertítulo general.")
    g_txt.add_argument("--no_panel_titles", "--no_subtitles", "--hide_subtitles", dest="no_panel_titles", action=argparse.BooleanOptionalAction, default=False, help="Desactivar títulos de cada panel.")
    g_txt.add_argument("--no_legends", "--hide_legends", dest="no_legends", action=argparse.BooleanOptionalAction, default=False, help="Desactivar todas las leyendas.")
    g_txt.add_argument("--legend_loc", type=str, default="", help="Posición de las leyendas.")
    g_txt.add_argument("--legend_fontsize", type=float, default=None, help="Tamaño de fuente de las leyendas.")
    g_txt.add_argument("--show_panel_letters", action=argparse.BooleanOptionalAction, default=True, help="Mostrar u ocultar (a), (b), (c), (d).")
    g_txt.add_argument("--title", "--custom_suptitle", dest="custom_suptitle", type=str, default="", help="Título principal personalizado.")
    g_txt.add_argument("--subtitle", "--custom_subtitle", dest="custom_subtitle", type=str, default="", help="Subtítulo personalizado.")
    g_txt.add_argument("--title_prefix", type=str, default="", help="Prefijo para el título principal.")
    g_txt.add_argument("--title_panel_a", "--title_traces", dest="title_panel_a", type=str, default="", help="Título para Panel (a).")
    g_txt.add_argument("--title_panel_b", "--title_transit", dest="title_panel_b", type=str, default="", help="Título para Panel (b).")
    g_txt.add_argument("--title_panel_c", "--title_heatmap_mala", dest="title_panel_c", type=str, default="", help="Título para Panel (c).")
    g_txt.add_argument("--title_panel_d", "--title_heatmap_rl", dest="title_panel_d", type=str, default="", help="Título para Panel (d).")
    g_txt.add_argument("--title_before_after", type=str, default="", help="Título para figura de paso exacto de transición.")
    g_txt.add_argument("--title_actor_panel", type=str, default="", help="Título para panel del Actor.")
    g_txt.add_argument("--label_mala", type=str, default="MALA", help="Etiqueta para MALA.")
    g_txt.add_argument("--label_rl", type=str, default="RL-MALA", help="Etiqueta para RL-MALA.")

    g_phys = p.add_argument_group("Parámetros de Ejes")
    g_phys.add_argument("--heatmap_ymax", type=float, default=2.5, help="Rango vertical [-ymax, +ymax] en los mapas de calor.")
    g_phys.add_argument("--trace_xmax", type=float, default=None, help="Rango espacial [-xmax, +xmax] en Bohr.")
    g_phys.add_argument("--before_after_dt", type=int, default=1, help="Paso entero previo/posterior de referencia.")

    return p


def _discover_available_ng_pairs(prod_root: str, csv_dir: str) -> List[Tuple[int, float]]:
    pairs = set()
    for pat in (
        os.path.join(csv_dir, "flips_system_meta_*.csv"),
        os.path.join(prod_root, "**", "*.h5"),
    ):
        for p in glob.glob(pat, recursive=True):
            m = re.search(r"N(\d+)_(?:w[0-9]+(?:\.[0-9]+)?_)?g([0-9]+(?:\.[0-9]+)?)", p)
            if m:
                pairs.add((int(m.group(1)), round(float(m.group(2)), 4)))
    return sorted(pairs)


def main(argv: Optional[List[str]] = None) -> int:
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    raw_in = args.input_dir_flag if args.input_dir_flag else args.input_path
    if not os.path.exists(raw_in):
        print(f"[ERROR] No existe la ruta de entrada: {raw_in}", file=sys.stderr)
        return 1

    prod_root, csv_dir = resolve_production_and_csv_dirs(raw_in)
    from scripts.analyze_flips_vs_g import (
        _matches_ng_filter,
        _parse_systems_filter,
        compute_flip_statistics,
        discover_h5_bundles,
        load_bundle_walkers_and_meta,
    )

    systems_map = _parse_systems_filter(args.systems_filter)
    h5_candidates = discover_h5_bundles(
        prod_root,
        include_pipeline3=(args.pipeline in ("P3", "P4")),
        include_pipeline4=(args.pipeline == "P4"),
    )

    # Agrupar bundles HDF5 legítimos por (N, g) eligiendo el mejor MALA y el mejor RL-MALA si hay múltiples hiperparámetros
    h5_by_ng_method: Dict[Tuple[int, float, str], Dict[str, Any]] = {}
    for h5p in h5_candidates:
        try:
            binfo = load_bundle_walkers_and_meta(h5p)
            if str(binfo.get("pipeline", "P2")) != str(args.pipeline):
                del binfo["walkers"]
                continue
            n_b = int(binfo["N"])
            g_b = round(float(binfo["g"]), 4)
            if not _matches_ng_filter(n_b, g_b, args.n_filter, args.g_filter, systems_map):
                del binfo["walkers"]
                continue
            m_b = str(binfo["method"])
            npz_p = _find_casscf_npz_for_system([prod_root, csv_dir, PROJECT_ROOT], n_b, float(binfo["omega"]), g_b)
            st = compute_flip_statistics(
                binfo["walkers"],
                n_up=binfo["n_up"],
                omega=binfo["omega"],
                g_val=binfo["g"],
                tau=binfo["tau"],
                actor_amp=binfo["actor_amplitude"],
                sigma_scale=binfo["sigma_action_scale"],
                actor_params=binfo.get("actor_params"),
                is_neural=(m_b == "RL-MALA"),
                npz_path=npz_p,
                parity_equivariant=bool(binfo.get("parity_equivariant", False)),
                unclip_sigma_action=bool(binfo.get("unclip_sigma_action", False)),
                scale_clip_min=float(binfo.get("scale_clip_min", -5.0)),
                scale_clip_max=float(binfo.get("scale_clip_max", 1.0)),
            )
            del binfo["walkers"]
            binfo["stats"] = st
            key_m = (n_b, g_b, m_b)
            prev = h5_by_ng_method.get(key_m)
            # Si hay búsqueda de hiperparámetros (como MultiDet_N6_g1.0), elegir el modelo con mayor tasa de flips con histéresis
            if prev is None or st["hyst_flips_total_mean"] > prev["stats"]["hyst_flips_total_mean"]:
                h5_by_ng_method[key_m] = binfo
        except Exception:
            continue

    avail_pairs = sorted(set((k[0], k[1]) for k in h5_by_ng_method.keys()))
    if not avail_pairs:
        avail_pairs = [
            (n_v, g_v)
            for (n_v, g_v) in _discover_available_ng_pairs(prod_root, csv_dir)
            if _matches_ng_filter(n_v, g_v, args.n_filter, args.g_filter, systems_map)
        ]

    if not avail_pairs:
        print(
            f"[ERROR] No se encontraron archivos HDF5 íntegros ni CSVs reales (`flips_system_meta_*.csv`) "
            f"para los filtros indicados en: {prod_root}",
            file=sys.stderr,
        )
        return 2

    base_out = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(prod_root, args.out_subdir)
    os.makedirs(base_out, exist_ok=True)

    print("==============================================================================")
    print(" GENERADOR DE FIGURAS REALES DE FLIPS, PASO EXACTO Y ACCIÓN DEL ACTOR")
    print(f" Carpeta de producción : {prod_root}")
    print(f" Carpeta de datos CSV  : {csv_dir}")
    print(f" Carpeta de salida     : {base_out}")
    print(f" Sistemas detectados   : {avail_pairs}")
    print("==============================================================================")

    for n_val, g_val in avail_pairs:
        omega_use = args.omega
        recs_h5 = [
            b for (nb, gb, _), b in h5_by_ng_method.items()
            if nb == n_val and abs(gb - g_val) <= 1e-4
        ]
        if recs_h5:
            omega_use = float(recs_h5[0]["omega"])
            sys_tag = f"N{n_val}_w{omega_use:g}_g{g_val:g}"
            print(f"\n[{sys_tag}] Extrayendo trayectorias discretas reales desde HDF5 y exportando CSV en: {csv_dir}")
            export_system_flips_csvs(recs_h5, csv_dir=csv_dir, prod_root=prod_root)
            records_ng = recs_h5
        else:
            sys_tag = f"N{n_val}_w{omega_use:g}_g{g_val:g}"
            records_ng = load_system_records_from_csvs(csv_dir, prod_root, n_val, omega_use, g_val, pipeline=args.pipeline)
            if records_ng is None:
                print(
                    f"  [OMITIDO] {sys_tag}: No hay archivo run_bundle.h5 íntegro local ni CSV real de trayectorias.",
                    file=sys.stderr,
                )
                continue
            print(f"\n[{sys_tag}] Datos reales cargados desde los CSV en: {csv_dir}")

        sys_out_dir = os.path.join(base_out, sys_tag) if (len(avail_pairs) > 1 or args.out_dir is None) else base_out
        os.makedirs(sys_out_dir, exist_ok=True)
        generate_all_system_flip_figures(records_ng, sys_out_dir, args, prod_root=prod_root)

    print("==============================================================================")
    print(" GENERACIÓN DE FIGURAS FINALIZADA EXITOSAMENTE")
    print(f" Carpeta de figuras : {base_out}")
    print("==============================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
