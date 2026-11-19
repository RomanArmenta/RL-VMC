#!/usr/bin/env python3
"""
Diagnóstico Espacial del Actor Entrenado (`FlaxTD3Actor`) en 1D.
================================================================

Toma uno o varios modelos RL-MALA entrenados (`run_bundle.h5`), fija N - 1
electrones en los pozos de equilibrio del cristal de Wigner (o en posiciones
definidas por el usuario) y barre la posición del N-ésimo electrón de prueba
a lo largo de toda la caja 1D `x_probe in [-x_max, +x_max]`.

En cada punto x_probe evalúa de forma 100% legítima (usando la función de onda
exacta del bundle / CASSCF y los pesos reales de `FlaxTD3Actor`):
  1. La densidad condicional exacta |Psi(x_1, ..., x_probe, ..., x_N)|^2 y la
     fuerza cuántica exacta F_Q(x) = d/dx ln|Psi|.
  2. La deriva pura del Actor mu_actor(x) = A_actor * a_mu(x) (CON SIGNO),
     más un análisis de ablación con F_Q = 0 para distinguir si el actor
     reacciona a las posiciones/distancias (DeepSet) o al gradiente F_Q, y
     una prueba de simetría de paridad (-mu_actor(-x)).
  3. El ancho de propuesta sigma_RL(x) = sqrt(2*tau)*exp(clip(a_sigma, -1, 1))
     junto con la salida SIN recorte sigma_raw(x) = sqrt(2*tau)*exp(a_sigma)
     para diagnosticar directamente si el actor está saturado en el techo
     `scale_clip_max = 1.0`.
  4. Las Gaussianas de propuesta q_RL(x' | x_0) vs q_MALA(x' | x_0) en 4
     posiciones físicas clave de la caja.

Ejemplo de uso:
  python3 scripts/probe_actor_spatial_response.py \
    --h5 results_nodal_crossing_search_N6/MultiDet_N6_g1.0/MultiDet_N6_g1.0_RL_MALA_tau0.08_amp0.15_pen0.15_sig1.1/run_bundle.h5 \
    --probe_spin both --format both --dpi 300
"""

from __future__ import annotations

import argparse
import csv
import glob
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

import jax
import jax.numpy as jnp

from rl_vmc.ansatze.ansatz_1d import build_ansatz_1d
from rl_vmc.config import Experiment1DConfig, parse_experiment_args
from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.proposals.proposals_1d import FlaxTD3Actor, build_td3_state
from scripts.plot_flips_system_detailed import (
    _atomic_savefig,
    _compute_exact_casscf_density_np,
    _find_casscf_npz_for_system,
)


# ==============================================================================
# 1. UTILIDADES DE RECONSTRUCCIÓN DE ANSATZ Y GRADIENTE EXACTO
# ==============================================================================

def _build_cfg_from_bundle_dict(cfg_dict: Dict[str, Any], casscf_npz_override: Optional[str] = None) -> Experiment1DConfig:
    """Reconstruye un `Experiment1DConfig` desde el diccionario `config` de `run_bundle.h5`."""
    cfg = parse_experiment_args([])
    for section_name in ("system", "ansatz", "proposal", "sampler", "rl", "train", "eval"):
        sub_dict = cfg_dict.get(section_name, {})
        if not isinstance(sub_dict, dict):
            continue
        sec_obj = getattr(cfg, section_name, None)
        if sec_obj is None:
            continue
        for k, v in sub_dict.items():
            if hasattr(sec_obj, k):
                setattr(sec_obj, k, v)

    if casscf_npz_override and os.path.isfile(casscf_npz_override):
        cfg.ansatz.casscf_npz_path = casscf_npz_override
    elif cfg.ansatz.ansatz_type in ("casscf_npz", "multi_sjb", "multi_slater_jastrow"):
        cur_p = cfg.ansatz.casscf_npz_path
        if not cur_p or not os.path.isfile(cur_p):
            # Intentar ruta relativa al proyecto o búsqueda en CASSCF_csv/
            if cur_p and os.path.isfile(os.path.join(PROJECT_ROOT, cur_p)):
                cfg.ansatz.casscf_npz_path = os.path.join(PROJECT_ROOT, cur_p)
            else:
                found = _find_casscf_npz_for_system(
                    [PROJECT_ROOT],
                    int(cfg.system.n_elec),
                    float(cfg.system.omega),
                    float(cfg.system.interaction_strength),
                )
                if found:
                    cfg.ansatz.casscf_npz_path = found
    return cfg


def build_exact_logpsi_and_grad_fn(
    bundle: Dict[str, Any],
    h5_path: str,
    casscf_npz_override: Optional[str] = None,
) -> Tuple[Any, Experiment1DConfig, Optional[str]]:
    """
    Construye una función `eval_logpsi_grad(x_batch)` exacta para el bundle:
      x_batch: np.ndarray de forma `(M, N)`
      retorna: `(log_psi (M,), grad_log_psi (M, N))`
    """
    cfg_dict = bundle.get("config", {})
    prod_dir = os.path.dirname(os.path.abspath(h5_path))
    parent_dir = os.path.dirname(prod_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    npz_path = casscf_npz_override
    if not npz_path:
        raw_p = cfg_dict.get("ansatz", {}).get("casscf_npz_path")
        if raw_p and os.path.isfile(raw_p):
            npz_path = raw_p
        elif raw_p and os.path.isfile(os.path.join(PROJECT_ROOT, raw_p)):
            npz_path = os.path.join(PROJECT_ROOT, raw_p)
        else:
            sys_c = cfg_dict.get("system", {})
            npz_path = _find_casscf_npz_for_system(
                [prod_dir, parent_dir, grandparent_dir, PROJECT_ROOT],
                int(sys_c.get("n_elec", 4)),
                float(sys_c.get("omega", 0.1)),
                float(sys_c.get("interaction_strength", 1.0)),
            )

    cfg = _build_cfg_from_bundle_dict(cfg_dict, casscf_npz_override=npz_path)
    ansatz = build_ansatz_1d(cfg)

    ckpts = bundle.get("checkpoints", {})
    ansatz_params = ckpts.get("ansatz_best") or ckpts.get("ansatz_final")
    if ansatz_params is None:
        dummy = jnp.zeros((1, cfg.system.n_elec), dtype=jnp.float64)
        ansatz_params = ansatz.init(jax.random.PRNGKey(0), dummy)

    min_c = float(cfg.sampler.min_clip_grads_ansatz)
    max_c = float(cfg.sampler.max_clip_grads_ansatz)

    @jax.jit
    def _eval_batch(x_jnp: jax.Array) -> Tuple[jax.Array, jax.Array]:
        lp, g = ansatz.value_and_grad(ansatz_params, x_jnp, min_clip=min_c, max_clip=max_c)
        return lp, g

    def eval_logpsi_grad(x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_in = jnp.asarray(np.asarray(x_np, dtype=np.float64))
        lp, g = _eval_batch(x_in)
        lp_np = np.asarray(lp, dtype=np.float64).reshape(-1)
        g_np = np.asarray(g, dtype=np.float64)
        if g_np.ndim == 3:
            g_np = g_np[..., 0]
        return lp_np, g_np

    return eval_logpsi_grad, cfg, npz_path


def find_wigner_well_centers(
    bundle: Dict[str, Any],
    N: int,
    omega: float,
    g_val: float,
    npz_path: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Determina las N posiciones de equilibrio de los pozos de Wigner:
      1. Primero intenta usar la mediana de las posiciones ordenadas de `production/walkers`
         (los pozos reales de equilibrio del sistema).
      2. También calcula la densidad 1-cuerpo exacta de CASSCF si el `.npz` existe.
    """
    walkers = bundle.get("production", {}).get("walkers")
    well_centers: Optional[np.ndarray] = None
    x_extent = 15.0

    if walkers is not None:
        w = np.asarray(walkers, dtype=np.float64)
        if w.ndim == 5:
            w = w[..., 0]
        w_flat = w.reshape(-1, N)
        # Descartar caminantes fugados extremos si existieran
        row_max = np.max(np.abs(w_flat), axis=1)
        p95 = float(np.percentile(row_max, 95))
        w_clean = w_flat[row_max <= max(p95 * 1.5, 25.0)]
        if len(w_clean) > 0:
            w_sorted = np.sort(w_clean, axis=-1)
            well_centers = np.median(w_sorted, axis=0)
            x_extent = float(np.ceil(max(12.0, float(np.percentile(np.abs(w_clean), 99.5)) + 3.5)))

    if well_centers is None:
        d_approx = 2.4 * ((1.0 + 1.8 * g_val) ** 0.28) * ((6.0 / max(N, 2)) ** 0.12)
        well_centers = (np.arange(N, dtype=np.float64) - 0.5 * (N - 1)) * d_approx
        x_extent = float(np.ceil(0.5 * N * d_approx + 5.0))

    casscf_ref = None
    if npz_path and os.path.isfile(npz_path):
        casscf_ref = _compute_exact_casscf_density_np(npz_path, x_max=max(x_extent + 3.0, 22.0), n_grid=600)
        # Refinar picos desde los máximos locales de rho_total si hay exactamente N picos claros
        if casscf_ref is not None:
            rho = casscf_ref["rho_total"]
            xg = casscf_ref["x"]
            peaks = []
            for k in range(1, len(rho) - 1):
                if rho[k] > rho[k - 1] and rho[k] >= rho[k + 1] and rho[k] > 0.15 * np.max(rho):
                    peaks.append(float(xg[k]))
            if len(peaks) == N:
                well_centers = np.asarray(peaks, dtype=np.float64)

    return well_centers, casscf_ref


# ==============================================================================
# 2. BARRIDO ESPACIAL DEL ELECTRÓN DE PRUEBA
# ==============================================================================

def run_spatial_actor_probe(
    bundle: Dict[str, Any],
    h5_path: str,
    probe_spin: str = "down",
    probe_well: Optional[int] = None,
    fixed_positions: Optional[List[float]] = None,
    x_max_override: Optional[float] = None,
    n_grid: int = 601,
    casscf_npz_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fija N - 1 electrones en los pozos de Wigner (alternando espines up/down
    de forma antiferromagnética natural) y barre el N-ésimo electrón de prueba
    a lo largo de x_probe in [-x_max, +x_max].
    """
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
        raise ValueError(f"El bundle HDF5 no contiene pesos de actor en checkpoints/actor_final: {h5_path}")
    p_in = actor_params.get("params", actor_params)

    well_centers, casscf_ref = find_wigner_well_centers(bundle, N, omega, g_val, npz_path=npz_path)

    # Elegir índice del electrón de prueba según espín solicitado
    if probe_spin == "up":
        probe_elec_idx = n_up - 1  # último electrón spin up
        probe_spin_sign = +1
    else:
        probe_elec_idx = N - 1     # último electrón spin down
        probe_spin_sign = -1

    # Asignar los N - 1 electrones fijos a pozos de Wigner en orden alternado de espín
    # (ej. para N=4: pozos [0, 1, 2, 3] -> [up, dn, up, dn]; quitando el pozo `w_empty`)
    if probe_well is None:
        # Por defecto dejar libre el penúltimo o último pozo interior de ese espín para ver ambos lados
        probe_well = N - 2 if (N >= 4 and probe_spin == "up") else (N - 1 if N <= 4 else N - 2)
    probe_well = int(np.clip(probe_well, 0, N - 1))

    if fixed_positions is not None:
        if len(fixed_positions) != N - 1:
            raise ValueError(f"--fixed_positions requiere exactamente N-1={N-1} valores, se recibieron {len(fixed_positions)}")
        rem_wells = np.sort(np.asarray(fixed_positions, dtype=np.float64))
    else:
        rem_wells = np.delete(well_centers, probe_well)

    # Construir la configuración base de los N - 1 electrones fijos asignando espines alternados
    other_up_indices = [i for i in range(n_up) if i != probe_elec_idx]
    other_dn_indices = [i for i in range(n_up, N) if i != probe_elec_idx]
    base_x = np.zeros(N, dtype=np.float64)
    # Asignar alternando up/dn a lo largo de rem_wells
    u_ptr, d_ptr = 0, 0
    for k_w, pos_w in enumerate(rem_wells):
        want_up = (k_w % 2 == 0)
        if want_up and u_ptr < len(other_up_indices):
            base_x[other_up_indices[u_ptr]] = pos_w
            u_ptr += 1
        elif (not want_up) and d_ptr < len(other_dn_indices):
            base_x[other_dn_indices[d_ptr]] = pos_w
            d_ptr += 1
        elif u_ptr < len(other_up_indices):
            base_x[other_up_indices[u_ptr]] = pos_w
            u_ptr += 1
        else:
            base_x[other_dn_indices[d_ptr]] = pos_w
            d_ptr += 1

    if x_max_override is not None and x_max_override > 0:
        x_max = float(x_max_override)
    else:
        x_max = float(np.ceil(max(float(np.max(np.abs(well_centers))) + 4.5, 11.0)))

    x_grid = np.linspace(-x_max, x_max, n_grid, dtype=np.float64)
    # Evitar coincidencia exacta a nivel máquina (0.0) con un electrón fijo del mismo espín para que grad_log_psi no sea NaN
    for e_k in range(N):
        if e_k == probe_elec_idx:
            continue
        near = np.abs(x_grid - base_x[e_k]) < 1e-4
        if np.any(near):
            x_grid[near] += 1.5e-4

    X_sweep = np.tile(base_x[None, :], (n_grid, 1))
    X_sweep[:, probe_elec_idx] = x_grid

    # 1. Evaluar ln|Psi| y grad_log_psi exactos
    log_psi, grad_log_psi = eval_logpsi_grad(X_sweep)
    fq_probe = grad_log_psi[:, probe_elec_idx]
    langevin_drift = np.clip(drift_pref * tau * fq_probe, -max_drift_clip, max_drift_clip)

    # Densidad condicional normalizada |Psi(x_probe)|^2
    lp_shift = log_psi - float(np.max(log_psi))
    cond_prob = np.exp(2.0 * np.clip(lp_shift, -50.0, 0.0))
    cond_prob = cond_prob / max(float(np.max(cond_prob)), 1e-12)

    # 2. Evaluar FlaxTD3Actor REAL (con F_Q real y en ablación con F_Q = 0)
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

    eff_sig_hi = clip_max if unclip_sig else min(clip_max, 1.0)
    eff_sig_lo = clip_min if unclip_sig else max(clip_min, -1.0)

    def _eval_actor_on(X_in: np.ndarray, G_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        st = build_td3_state(
            jnp.asarray(X_in),
            jnp.asarray(G_in),
            probe_elec_idx,
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
        raw_out = np.asarray(actor_mod.apply({"params": p_in}, st), dtype=np.float64)
        a_mu_raw = raw_out[:, 0]
        a_sig_raw = raw_out[:, 1]
        # Replicar exactamente el recorte de DeepSetNeuralProposal1D.forward_action + distribution_from_action
        a_mu_clip = np.clip(a_mu_raw, -1.0, 1.0)
        if unclip_sig:
            a_sig_clip = np.clip(a_sig_raw, clip_min, clip_max)
        else:
            a_sig_clip = np.clip(np.clip(a_sig_raw, -1.0, 1.0), clip_min, clip_max)
        return a_mu_raw, a_sig_raw, a_mu_clip, a_sig_clip

    a_mu_raw, a_sig_raw, a_mu_clip, a_sig_clip = _eval_actor_on(X_sweep, grad_log_psi)

    # Ablación 1: F_Q = 0 (mismas posiciones, sin fuerza cuántica)
    a_mu_nofq_raw, a_sig_nofq_raw, a_mu_nofq, a_sig_nofq = _eval_actor_on(X_sweep, np.zeros_like(grad_log_psi))

    # Ablación 2: Prueba de paridad espacial (reflejar TODA la configuración x -> -x, F_Q -> -F_Q)
    a_mu_par_raw, a_sig_par_raw, a_mu_par, a_sig_par = _eval_actor_on(-X_sweep, -grad_log_psi)
    # Bajo paridad exacta: a_mu(-x, -F_Q) debería ser igual a -a_mu(x, F_Q), y a_sig(-x, -F_Q) == a_sig(x, F_Q)
    parity_err_mu = np.abs(a_mu_clip + a_mu_par)
    parity_err_sig = np.abs(a_sig_raw - a_sig_par_raw)

    mu_actor = amp * a_mu_clip
    mu_actor_nofq = amp * a_mu_nofq
    mu_total_disp = langevin_drift + mu_actor
    sigma_applied = np.maximum(base_sigma * np.exp(a_sig_clip), 1e-4)
    sigma_unclipped = np.maximum(base_sigma * np.exp(a_sig_raw), 1e-4)
    sigma_nofq = np.maximum(base_sigma * np.exp(a_sig_nofq), 1e-4)

    # Separar posiciones fijas por mismo espín vs espín opuesto respecto al electrón de prueba
    same_fixed_x = np.sort([base_x[i] for i in (other_up_indices if probe_spin_sign > 0 else other_dn_indices)])
    opp_fixed_x = np.sort([base_x[i] for i in (other_dn_indices if probe_spin_sign > 0 else other_up_indices)])

    # Distancias al vecino más cercano del mismo espín y de espín opuesto
    d_min_same = np.min(np.abs(x_grid[:, None] - same_fixed_x[None, :]), axis=1) if len(same_fixed_x) > 0 else np.full_like(x_grid, np.nan)
    d_min_opp = np.min(np.abs(x_grid[:, None] - opp_fixed_x[None, :]), axis=1) if len(opp_fixed_x) > 0 else np.full_like(x_grid, np.nan)

    # Estadísticas cuantitativas diagnósticas
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "h5_path": h5_path,
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
        "probe_well": probe_well,
        "empty_well_x": float(well_centers[probe_well]),
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
        "frac_sigma_clipped_pct": float(np.mean(a_sig_raw >= eff_sig_hi - 1e-6) * 100.0),
        "max_fq_sensitivity_mu": float(np.max(np.abs(a_mu_clip - a_mu_nofq))),
        "max_fq_sensitivity_sig": float(np.max(np.abs(a_sig_raw - a_sig_nofq_raw))),
        "mean_parity_asymmetry_mu": float(np.mean(parity_err_mu)),
        "mean_parity_asymmetry_sig": float(np.mean(parity_err_sig)),
        "corr_amu_vs_x": _corr(a_mu_clip, x_grid),
        "corr_amu_vs_tanhFQ": _corr(a_mu_clip, np.tanh(fq_probe / float(cfg.proposal.force_scale))),
        "corr_asig_vs_absx": _corr(a_sig_raw, np.abs(x_grid)),
        "corr_asig_vs_absFQ": _corr(a_sig_raw, np.abs(np.tanh(fq_probe / float(cfg.proposal.force_scale)))),
    }

    return {
        "summary": summary,
        "x_grid": x_grid,
        "well_centers": well_centers,
        "same_fixed_x": same_fixed_x,
        "opp_fixed_x": opp_fixed_x,
        "empty_well_x": float(well_centers[probe_well]),
        "cond_prob": cond_prob,
        "casscf_ref": casscf_ref,
        "fq_probe": fq_probe,
        "langevin_drift": langevin_drift,
        "a_mu_raw": a_mu_raw,
        "a_mu_clip": a_mu_clip,
        "a_sig_raw": a_sig_raw,
        "a_sig_clip": a_sig_clip,
        "mu_actor": mu_actor,
        "mu_actor_nofq": mu_actor_nofq,
        "mu_actor_parity_expected": -amp * a_mu_par,
        "mu_total_disp": mu_total_disp,
        "sigma_applied": sigma_applied,
        "sigma_unclipped": sigma_unclipped,
        "sigma_nofq": sigma_nofq,
        "d_min_same": d_min_same,
        "d_min_opp": d_min_opp,
    }


# ==============================================================================
# 3. GRAFICACIÓN DE PUBLICACIÓN DEL DIAGNÓSTICO ESPACIAL
# ==============================================================================

def _draw_fixed_electron_markers(
    ax: Any,
    same_fixed_x: np.ndarray,
    opp_fixed_x: np.ndarray,
    empty_well_x: float,
    probe_spin: str,
    add_labels: bool = False,
) -> None:
    same_col = "#1f77b4" if probe_spin == "up" else "#d62728"
    opp_col = "#d62728" if probe_spin == "up" else "#1f77b4"
    same_sym = r"$\uparrow$" if probe_spin == "up" else r"$\downarrow$"
    opp_sym = r"$\downarrow$" if probe_spin == "up" else r"$\uparrow$"

    for idx, sx in enumerate(same_fixed_x):
        ax.axvline(
            sx,
            color=same_col,
            ls="-",
            lw=1.35,
            alpha=0.65,
            label=(f"Fijo mismo espín {same_sym} (Nodo Pauli $\\Psi=0$)" if (add_labels and idx == 0) else None),
            zorder=2,
        )
    for idx, ox in enumerate(opp_fixed_x):
        ax.axvline(
            ox,
            color=opp_col,
            ls="--",
            lw=1.25,
            alpha=0.60,
            label=(f"Fijo espín opuesto {opp_sym} (Barrera Wigner)" if (add_labels and idx == 0) else None),
            zorder=2,
        )
    ax.axvline(
        empty_well_x,
        color="#2ca02c",
        ls=":",
        lw=1.6,
        alpha=0.80,
        label=("Pozo vacío asignado al electrón de prueba" if add_labels else None),
        zorder=2,
    )


def plot_spatial_probe_figure(
    res: Dict[str, Any],
    out_dir: str,
    stem: str,
    args: argparse.Namespace,
) -> List[str]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    sm = res["summary"]
    xg = res["x_grid"]
    same_fx = res["same_fixed_x"]
    opp_fx = res["opp_fixed_x"]
    empty_x = res["empty_well_x"]
    p_spin = sm["probe_spin"]
    spin_sym = r"\uparrow" if p_spin == "up" else r"\downarrow"
    f_scale = float(getattr(args, "font_scale", 1.0))

    fig = plt.figure(figsize=(15.8, 17.8))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.0, 1.05, 1.05, 1.05], hspace=0.78, top=0.925, bottom=0.045)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)

    # --------------------------------------------------------------------------
    # Panel (a): Ambiente físico local — |Psi(x)|^2 condicional y deriva MALA
    # --------------------------------------------------------------------------
    _draw_fixed_electron_markers(ax1, same_fx, opp_fx, empty_x, p_spin, add_labels=True)
    ax1.fill_between(xg, 0.0, res["cond_prob"], color="#4c78a8", alpha=0.28, label=r"$|\Psi(x_{\mathrm{probe}} \mid x_{\mathrm{fijos}})|^2$ (norm.)")
    ax1.plot(xg, res["cond_prob"], color="#1f4e79", lw=1.8)
    if res["casscf_ref"] is not None:
        cr = res["casscf_ref"]
        rho_n = cr["rho_total"] / max(float(np.max(cr["rho_total"])), 1e-8)
        ax1.plot(cr["x"], rho_n, color="#555555", ls=":", lw=1.4, alpha=0.85, label=r"Densidad exacta 1-cuerpo $\rho_{\mathrm{CASSCF}}(x)$ (norm.)")

    ax1_r = ax1.twinx()
    ax1_r.plot(xg, res["langevin_drift"], color="#e45756", lw=1.35, alpha=0.82, label=r"Deriva Langevin MALA $\tau \partial_x \ln|\Psi|$")
    ax1_r.axhline(0.0, color="#e45756", ls="-.", lw=0.7, alpha=0.4)
    ax1_r.set_ylabel(r"Deriva MALA $\tau F_Q(x)$ [Bohr]", color="#c0392b", fontsize=9.6 * f_scale, weight="bold")
    ax1_r.tick_params(axis="y", colors="#c0392b")
    q_lim = min(float(np.percentile(np.abs(res["langevin_drift"]), 97)) * 1.4 + 0.3, 4.5)
    ax1_r.set_ylim(-q_lim, q_lim)

    ax1.set_ylim(0.0, 1.15)
    ax1.set_ylabel(r"Densidad norm. $|\Psi|^2$", fontsize=9.8 * f_scale, weight="bold")
    ax1.grid(True, alpha=0.25, ls="--")
    if not getattr(args, "no_titles", False):
        ax1.set_title(
            f"(a) Ambiente Físico Local al Barrer el Electrón ${spin_sym}$ ($N={sm['N']}, g={sm['g']:g}$, {sm['N']-1} electrones fijos en pozos de Wigner)",
            fontsize=10.8 * f_scale,
            weight="bold",
            pad=46,
        )
    if not getattr(args, "no_legends", False):
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1_r.get_legend_handles_labels()
        ax1.legend(
            h1 + h2,
            l1 + l2,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            fontsize=7.0 * f_scale,
            ncol=3,
            framealpha=0.72,
            borderpad=0.3,
            columnspacing=0.95,
        )

    # --------------------------------------------------------------------------
    # Panel (b): Deriva Residual del Actor mu_actor(x) = A * a_mu(x) CON SIGNO
    # --------------------------------------------------------------------------
    _draw_fixed_electron_markers(ax2, same_fx, opp_fx, empty_x, p_spin, add_labels=False)
    ax2.axhline(0.0, color="#222222", lw=0.9, ls="-", alpha=0.5)
    ax2.axhline(+sm["amp"], color="#1f77b4", lw=1.0, ls=":", alpha=0.65, label=f"Techo saturación $\\pm A_{{\\mathrm{{actor}}}} = \\pm {sm['amp']:.2f}$ Bohr")
    ax2.axhline(-sm["amp"], color="#1f77b4", lw=1.0, ls=":", alpha=0.65)

    ax2.plot(xg, res["mu_actor"], color="#1f77b4", lw=2.3, label=r"Deriva del Actor $\mu_{\mathrm{actor}}(x) = A_{\mathrm{actor}}\,a_\mu(x)$ (con $F_Q$ real)")
    ax2.plot(xg, res["mu_actor_nofq"], color="#9467bd", lw=1.8, ls="--", alpha=0.88, label=r"Ablación sin fuerza cuántica ($F_Q \equiv 0$, solo posiciones DeepSet)")
    ax2.plot(xg, res["mu_actor_parity_expected"], color="#7f7f7f", lw=1.3, ls="-.", alpha=0.72, label=r"Simetría esperada por paridad ($-\mu_{\mathrm{actor}}(-x, -F_Q)$)")

    amp_pad = max(sm["amp"] * 1.42, 0.04)
    ax2.set_ylim(-amp_pad, amp_pad)
    ax2.set_ylabel(r"Deriva Actor $\mu_{\mathrm{actor}}(x)$ [Bohr]", fontsize=9.8 * f_scale, weight="bold")
    ax2.grid(True, alpha=0.25, ls="--")
    ax2.text(
        0.50,
        0.03,
        f"$a_\\mu \\in [{sm['a_mu_min']:+.3f}, {sm['a_mu_max']:+.3f}]$ (std={sm['a_mu_std']:.3f}, sat={sm['frac_mu_saturated_pct']:.1f}%) | "
        f"Sensibilidad a $F_Q$: max $|\\Delta a_\\mu|={sm['max_fq_sensitivity_mu']:.3f}$ | "
        f"Asimetría paridad: $\\langle|a_\\mu(x)+a_\\mu(-x)|\\rangle={sm['mean_parity_asymmetry_mu']:.3f}$",
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.4 * f_scale,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.72, edgecolor="#1f77b4", lw=0.7),
    )
    if not getattr(args, "no_titles", False):
        ax2.set_title(
            "(b) Respuesta de Deriva del Actor $\\mu_{\\mathrm{actor}}(x)$ (Con Signo): ¿Empuja Izquierda/Derecha o Está Saturado?",
            fontsize=10.8 * f_scale,
            weight="bold",
            pad=46,
        )
    if not getattr(args, "no_legends", False):
        ax2.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            fontsize=7.1 * f_scale,
            ncol=2,
            framealpha=0.72,
            borderpad=0.3,
            columnspacing=1.0,
        )

    # --------------------------------------------------------------------------
    # Panel (c): Ancho de Propuesta sigma_RL(x) — Aplicado (clipped) vs Sin Clip
    # --------------------------------------------------------------------------
    _draw_fixed_electron_markers(ax3, same_fx, opp_fx, empty_x, p_spin, add_labels=False)
    ax3.axhline(sm["base_sigma_mala"], color="#d62728", lw=1.6, ls=":", label=f"MALA fijo $\\sigma_{{\\mathrm{{MALA}}}} = \\sqrt{{2\\tau}} = {sm['base_sigma_mala']:.3f}$ Bohr")
    clip_lbl = (
        f"Techo desbloqueado (clip={sm['scale_clip_max']:g})"
        if sm.get("unclip_sigma_action", False)
        else f"Techo duro legado (clip={sm.get('eff_sig_hi', 1.0):g})"
    )
    ax3.axhline(sm["sigma_ceiling"], color="#222222", lw=1.3, ls="-.", alpha=0.80, label=f"{clip_lbl}: $\\sigma_{{\\max}} = {sm['sigma_ceiling']:.3f}$ Bohr")

    ax3.plot(xg, res["sigma_unclipped"], color="#ff7f0e", lw=1.9, ls="--", alpha=0.90, label=r"Salida red sin recorte $\sqrt{2\tau}\,e^{a_\sigma^{\mathrm{raw}}}$")
    sig_clip_tex = (
        rf"Ancho aplicado $\sigma_{{\mathrm{{RL}}}}(x)$ (clip $\in [{sm.get('eff_sig_lo', -5.0):g}, {sm.get('eff_sig_hi', 1.0):g}]$)"
    )
    ax3.plot(xg, res["sigma_applied"], color="#c51b8a", lw=2.5, label=sig_clip_tex)
    ax3.plot(xg, res["sigma_nofq"], color="#17becf", lw=1.4, ls=":", alpha=0.85, label=r"Ablación sin fuerza cuántica ($F_Q \equiv 0$)")

    y_lo = max(0.0, min(sm["base_sigma_mala"], float(np.min(res["sigma_applied"])), float(np.min(res["sigma_unclipped"]))) * 0.55)
    y_hi = max(sm["sigma_ceiling"], float(np.max(res["sigma_unclipped"]))) * 1.12
    ax3.set_ylim(y_lo, y_hi)
    ax3.set_ylabel(r"Ancho $\sigma(x)$ [Bohr]", fontsize=9.8 * f_scale, weight="bold")
    ax3.grid(True, alpha=0.25, ls="--")

    # Ubicar la caja estadística en la banda blanca libre entre MALA base y sigma_RL
    sig_curves_min = min(
        float(np.min(res["sigma_applied"])),
        float(np.min(res["sigma_unclipped"])),
        float(np.min(res["sigma_nofq"])),
    )
    band_gap = sig_curves_min - sm["base_sigma_mala"]
    if band_gap > 0.20 * (y_hi - y_lo):
        y_box = 0.5 * (sm["base_sigma_mala"] + sig_curves_min)
        box_transform = ax3.get_yaxis_transform()
        box_va = "center"
    else:
        y_box = 0.03
        box_transform = ax3.transAxes
        box_va = "bottom"

    ax3.text(
        0.50,
        y_box,
        f"$a_\\sigma^{{\\mathrm{{raw}}}} \\in [{sm['a_sig_raw_min']:+.3f}, {sm['a_sig_raw_max']:+.3f}]$ (std={sm['a_sig_raw_std']:.3f}) | "
        f"$\\sigma_{{\\mathrm{{RL}}}} \\in [{sm['sigma_applied_min']:.3f}, {sm['sigma_applied_max']:.3f}]$ Bohr (std={sm['sigma_applied_std']:.4f}) | "
        f"Puntos pegados al techo ($a_\\sigma \\geq {sm.get('eff_sig_hi', 1.0):g}$): {sm['frac_sigma_clipped_pct']:.1f}%",
        transform=box_transform,
        ha="center",
        va=box_va,
        fontsize=7.4 * f_scale,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.72, edgecolor="#c51b8a", lw=0.7),
    )
    if not getattr(args, "no_titles", False):
        ax3.set_title(
            f"(c) Respuesta de Ancho de Propuesta $\\sigma_{{\\mathrm{{RL}}}}(x)$ vs Techo (eff_clip={sm.get('eff_sig_hi', 1.0):g}) y MALA Fijo",
            fontsize=10.8 * f_scale,
            weight="bold",
            pad=46,
        )
    if not getattr(args, "no_legends", False):
        ax3.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            fontsize=7.0 * f_scale,
            ncol=3,
            framealpha=0.72,
            borderpad=0.3,
            columnspacing=0.9,
        )

    # --------------------------------------------------------------------------
    # Panel (d): Propuestas Gaussianas q_RL(x'|x_0) vs q_MALA(x'|x_0) en 4 puntos
    # --------------------------------------------------------------------------
    _draw_fixed_electron_markers(ax4, same_fx, opp_fx, empty_x, p_spin, add_labels=False)
    # Seleccionar 4 posiciones representativas para visualizar la Gaussiana de propuesta
    probe_pts = [empty_x]
    if len(opp_fx) > 0:
        nearest_opp = opp_fx[int(np.argmin(np.abs(opp_fx - empty_x)))]
        probe_pts.append(0.5 * (empty_x + nearest_opp))
    if len(same_fx) > 0:
        nearest_same = same_fx[int(np.argmin(np.abs(same_fx - empty_x)))]
        # Un poco al lado del nodo de Pauli
        probe_pts.append(nearest_same + 0.55 * np.sign(empty_x - nearest_same + 1e-6))
    probe_pts.append(float(np.min(res["well_centers"])) - 1.8)
    # Garantizar exactamente 4 puntos distintos dentro de [-x_max+1, x_max-1]
    probe_pts = [float(np.clip(p, xg[15], xg[-16])) for p in probe_pts[:4]]
    pt_colors = ["#2ca02c", "#ff7f0e", "#9467bd", "#1f77b4"]
    pt_names = [
        "En el pozo vacío",
        "Sobre barrera Wigner",
        "Cerca de nodo Pauli",
        "Borde externo de la trampa",
    ]

    for idx_p, (x0, col_p, name_p) in enumerate(zip(probe_pts, pt_colors, pt_names)):
        gi = int(np.argmin(np.abs(xg - x0)))
        mu_m = xg[gi] + res["langevin_drift"][gi]
        sig_m = sm["base_sigma_mala"]
        mu_rl = xg[gi] + res["mu_total_disp"][gi]
        sig_rl = res["sigma_applied"][gi]

        q_m = np.exp(-0.5 * ((xg - mu_m) / sig_m) ** 2) / (math.sqrt(2.0 * math.pi) * sig_m)
        q_rl = np.exp(-0.5 * ((xg - mu_rl) / sig_rl) ** 2) / (math.sqrt(2.0 * math.pi) * sig_rl)

        ax4.fill_between(xg, 0.0, q_rl, color=col_p, alpha=0.22)
        ax4.plot(
            xg,
            q_rl,
            color=col_p,
            lw=2.1,
            label=f"RL en $x_0={xg[gi]:+.1f}$ ({name_p}, $\\sigma={sig_rl:.2f}$)",
        )
        ax4.plot(
            xg,
            q_m,
            color=col_p,
            lw=1.3,
            ls="--",
            alpha=0.80,
            label=(f"MALA ($\\sigma={sig_m:.2f}$)" if idx_p == 0 else None),
        )
        ax4.scatter([xg[gi]], [0.02], color=col_p, marker="^", s=70, edgecolors="black", zorder=6)

    ax4.set_xlim(xg[0], xg[-1])
    ax4.set_ylim(0.0, None)
    ax4.set_xlabel(r"Posición del electrón de prueba $x_{\mathrm{probe}}$ [Bohr]", fontsize=10.2 * f_scale, weight="bold")
    ax4.set_ylabel(r"Densidad de propuesta $q(x' \mid x_0)$", fontsize=9.8 * f_scale, weight="bold")
    ax4.grid(True, alpha=0.25, ls="--")
    if not getattr(args, "no_titles", False):
        ax4.set_title(
            "(d) Distribuciones Gaussianas de Propuesta $q_{\\mathrm{RL}}(x' \\mid x_0)$ (relleno) vs $q_{\\mathrm{MALA}}(x' \\mid x_0)$ (punteado) en 4 Puntos de la Caja",
            fontsize=10.8 * f_scale,
            weight="bold",
            pad=46,
        )
    if not getattr(args, "no_legends", False):
        ax4.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.015),
            fontsize=7.0 * f_scale,
            ncol=3,
            framealpha=0.72,
            borderpad=0.3,
            columnspacing=0.9,
        )

    if not getattr(args, "no_titles", False):
        run_label = os.path.basename(os.path.dirname(os.path.abspath(sm["h5_path"])))
        eq_tag = "parity=ON" if sm.get("parity_equivariant", False) else "parity=OFF (legacy)"
        uc_tag = f"unclip_sig=ON ({sm.get('eff_sig_hi', 1.0):g})" if sm.get("unclip_sigma_action", False) else "clip_sig=1.0 (legacy)"
        fig.suptitle(
            f"Diagnóstico Espacial del Actor ({run_label}) — Barrido de Electrón ${spin_sym}$ "
            f"($N={sm['N']}, \\omega={sm['omega']:g}, g={sm['g']:g}, \\tau={sm['tau']:g}, A={sm['amp']:g}, s_\\sigma={sm['sigma_action_scale']:g}$ | {eq_tag}, {uc_tag})",
            fontsize=11.4 * f_scale,
            weight="bold",
            y=0.985,
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


def save_probe_csv(res: Dict[str, Any], out_csv: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    xg = res["x_grid"]
    fieldnames = [
        "x_probe",
        "cond_prob_norm",
        "fq_probe",
        "langevin_drift",
        "a_mu_raw",
        "a_mu_clip",
        "mu_actor",
        "mu_actor_nofq",
        "mu_actor_parity_expected",
        "mu_total_disp",
        "a_sig_raw",
        "a_sig_clip",
        "sigma_applied",
        "sigma_unclipped",
        "sigma_nofq",
        "d_min_same",
        "d_min_opp",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k in range(len(xg)):
            w.writerow(
                {
                    "x_probe": float(xg[k]),
                    "cond_prob_norm": float(res["cond_prob"][k]),
                    "fq_probe": float(res["fq_probe"][k]),
                    "langevin_drift": float(res["langevin_drift"][k]),
                    "a_mu_raw": float(res["a_mu_raw"][k]),
                    "a_mu_clip": float(res["a_mu_clip"][k]),
                    "mu_actor": float(res["mu_actor"][k]),
                    "mu_actor_nofq": float(res["mu_actor_nofq"][k]),
                    "mu_actor_parity_expected": float(res["mu_actor_parity_expected"][k]),
                    "mu_total_disp": float(res["mu_total_disp"][k]),
                    "a_sig_raw": float(res["a_sig_raw"][k]),
                    "a_sig_clip": float(res["a_sig_clip"][k]),
                    "sigma_applied": float(res["sigma_applied"][k]),
                    "sigma_unclipped": float(res["sigma_unclipped"][k]),
                    "sigma_nofq": float(res["sigma_nofq"][k]),
                    "d_min_same": float(res["d_min_same"][k]),
                    "d_min_opp": float(res["d_min_opp"][k]),
                }
            )
    print(f"  [+] CSV de barrido espacial guardado: {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fija N-1 electrones en pozos de Wigner y barre el N-ésimo electrón por toda la caja para diagnosticar (mu, sigma) del Actor entrenado."
    )
    p.add_argument("--h5", required=True, nargs="+", help="Uno o más archivos run_bundle.h5 de RL-MALA entrenado (o carpeta que los contenga).")
    p.add_argument("--casscf_npz", default=None, help="Ruta opcional a casscf_state.npz (si no, se detecta automáticamente).")
    p.add_argument("--probe_spin", choices=["up", "down", "both"], default="both", help="Espín del electrón de prueba a barrer (default: both).")
    p.add_argument("--probe_well", type=int, default=None, help="Índice del pozo (0..N-1) que queda libre para el electrón de prueba.")
    p.add_argument("--fixed_positions", type=float, nargs="+", default=None, help="Posiciones explícitas en Bohr para los N-1 electrones fijos.")
    p.add_argument("--x_max", type=float, default=None, help="Límite espacial [-x_max, +x_max] en Bohr.")
    p.add_argument("--n_grid", type=int, default=601, help="Número de puntos del barrido espacial.")
    p.add_argument("--out_dir", default="figures/actor_spatial_probe", help="Carpeta de salida para las figuras y CSVs.")
    p.add_argument("--figure_format", "--format", dest="figure_format", choices=["png", "pdf", "both"], default="both")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--font_scale", type=float, default=1.0)
    p.add_argument("--no_titles", action="store_true")
    p.add_argument("--no_legends", action="store_true")
    return p


def _expand_h5_inputs(raw_list: List[str]) -> List[str]:
    out: List[str] = []
    for item in raw_list:
        if os.path.isfile(item) and item.endswith(".h5"):
            out.append(item)
        elif os.path.isdir(item):
            for cand in sorted(glob.glob(os.path.join(item, "**", "*.h5"), recursive=True)):
                low = cand.lower()
                if any(k in low for k in ("rl", "td3", "deepset", "coupled")):
                    out.append(cand)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")

    args = build_arg_parser().parse_args(argv)
    h5_files = _expand_h5_inputs(args.h5)
    if not h5_files:
        print("[ERROR] No se encontraron archivos .h5 de RL-MALA válidos.", file=sys.stderr)
        return 1

    spins_to_run = ["up", "down"] if args.probe_spin == "both" else [args.probe_spin]
    summary_rows: List[Dict[str, Any]] = []

    print("==============================================================================")
    print(" DIAGNÓSTICO ESPACIAL DEL ACTOR ENTRENADO (mu, sigma vs x_probe)")
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

        run_dir_name = os.path.basename(os.path.dirname(os.path.abspath(h5p)))
        file_stem = os.path.splitext(os.path.basename(h5p))[0]
        model_tag = run_dir_name if file_stem == "run_bundle" else f"{run_dir_name}_{file_stem}"

        for sp in spins_to_run:
            res = run_spatial_actor_probe(
                bundle,
                h5_path=h5p,
                probe_spin=sp,
                probe_well=args.probe_well,
                fixed_positions=args.fixed_positions,
                x_max_override=args.x_max,
                n_grid=args.n_grid,
                casscf_npz_override=args.casscf_npz,
            )
            sm = res["summary"]
            summary_rows.append(sm)
            print(
                f"\n[{model_tag} | probe={sp}] N={sm['N']}, g={sm['g']:g}, tau={sm['tau']:g}, amp={sm['amp']:g}, sig_sc={sm['sigma_action_scale']:g}\n"
                f"  a_mu   : [{sm['a_mu_min']:+.3f}, {sm['a_mu_max']:+.3f}] std={sm['a_mu_std']:.3f} | sat={sm['frac_mu_saturated_pct']:.1f}% | "
                f"corr(x)={sm['corr_amu_vs_x']:+.3f} corr(F_Q)={sm['corr_amu_vs_tanhFQ']:+.3f} | max|d_mu(F_Q=0)|={sm['max_fq_sensitivity_mu']:.3f}\n"
                f"  a_sigma: raw=[{sm['a_sig_raw_min']:+.3f}, {sm['a_sig_raw_max']:+.3f}] std={sm['a_sig_raw_std']:.3f} | "
                f"sigma_RL=[{sm['sigma_applied_min']:.3f}, {sm['sigma_applied_max']:.3f}] Bohr | clipped>=1: {sm['frac_sigma_clipped_pct']:.1f}%"
            )

            stem = f"fig_actor_spatial_probe_{model_tag}_spin_{sp}"
            plot_spatial_probe_figure(res, args.out_dir, stem, args)
            save_probe_csv(res, os.path.join(args.out_dir, f"data_{stem}.csv"))

    if summary_rows:
        sum_csv = os.path.join(args.out_dir, "actor_spatial_probe_summary.csv")
        with open(sum_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)
        print(f"\n[+] Resumen comparativo guardado en: {sum_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
