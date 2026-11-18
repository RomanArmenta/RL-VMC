#!/usr/bin/env python3
"""
scripts/analyze_flips_vs_g.py
==============================================================================
Caracterización física y estadística de intercambios topológicos ("flips" de
electrones entre pozos/modos gaussianos y cruces nodales) en función del
acoplamiento de interacción g y del número de electrones N, comparando MALA
clásico contra RL-MALA (Actor DeepSet + TD3).

Diseñado para operar directamente sobre la jerarquía de carpetas generada por
`run_production_benchmark_3pipelines.sh`:

    <BASE_OUT_DIR>/
      ├── pipeline1_casscf/N{N}_w{omega}_g{g}/casscf_state.npz
      ├── pipeline2_pure_sampling/N{N}_w{omega}_g{g}/
      │     ├── N{N}_w{omega}_g{g}_MALA_tau*/run_bundle.h5       (o MALA_FixedCASSCF/)
      │     └── N{N}_w{omega}_g{g}_RL_MALA_*/run_bundle.h5       (o RL_TD3_FixedCASSCF/)
      ├── pipeline3_variational_training/N{N}_w{omega}_g{g}/     (opcional)
      ├── figures/
      └── results_csv/

También soporta pasar cualquier directorio que contenga subcarpetas con archivos
`run_bundle*.h5` (por ejemplo `results_nodal_crossing_search_N6`).

Métricas calculadas por cada (N, g, método):
  1. Flips crudos por pares (R_raw):
     Cambios de signo de (x_i(t) - x_j(t)) entre pasos consecutivos t-1 y t,
     normalizados por cada 1,000 pasos de Markov y por caminante (walker).
  2. Flips con histéresis / filtro de zona muerta (R_hyst):
     Filtra las fluctuaciones térmicas microscópicas alrededor de x_i = x_j
     (dominantes en el régimen de Hartree g < 0.3) exigiendo que el par (i, j)
     supere un umbral físico delta = eta * d_adj(g), donde d_adj(g) es la
     separación media entre electrones adyacentes ordenados.
  3. Desglose por canal de espín:
     - Espín opuesto (up-down): cruce de la barrera de correlación de Wigner /
       cúspide de Coulomb (responsable de la ergodicidad de espín y D_KL^spin).
     - Mismo espín (up-up / down-down): cruce de un nodo exacto de Pauli (Psi = 0).
  4. Anatomía de 1 paso vs multi-paso (Delta t_transit y f_1step):
     Mide cuántos pasos consecutivos permanece un par dentro de la barrera
     |x_i - x_j| <= delta durante un intercambio verdadero, cuantificando qué
     fracción de los cruces del valle ocurre en un único paso colectivo de
     Markov (Delta t_transit = 1) frente a tránsitos de múltiples pasos.

Salidas generadas:
  - CSV detallado por modelo y sistema: `<out_dir>/flips_vs_g_summary.csv`
  - Para cada (N, g): `<out_dir>/per_system/fig_flips_system_N{N}_w{omega}_g{g}.{png,pdf}`
    (4 paneles: trazas de caminante con eventos de flip, curva acumulada N_flips(t),
    anatomía microscópica alineada en t - t_flip, e histograma de pasos de tránsito).
  - Para cada N vs g: `<out_dir>/per_N/fig_flips_vs_g_N{N}.{png,pdf}`
    (4 paneles: evolución de R_flip^up-down vs g desde Hartree g < 0.3 hasta
    Wigner g > 0.4, desglose espín opuesto vs mismo espín, correlación con
    D_KL^spin y factor de aceleración, y fracción de saltos en 1 paso vs g).
  - Síntesis global multi-N vs g: `<out_dir>/fig_flips_vs_g_all_N_master.{png,pdf}`
==============================================================================
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

N_COLORS = {
    2: "#1f77b4",   # Azul
    4: "#2ca02c",   # Verde
    6: "#ff7f0e",   # Naranja
    8: "#9467bd",   # Morado
    10: "#d62728",  # Rojo
}
N_MARKERS = {
    2: "o",
    4: "s",
    6: "^",
    8: "P",
    10: "D",
}
METHOD_COLORS = {
    "MALA": "#d62728",          # Rojo ladrillo
    "Adaptive-MALA": "#ff7f0e", # Naranja
    "RL-MALA": "#1f77b4",       # Azul profundo
}


# ==============================================================================
# 1. UTILIDADES DE CARGA DE BUNDLES HDF5 Y METADATOS DE SISTEMA
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


def _classify_method(label: str, proposal_kind: str = "", h5_path: str = "") -> str:
    joined = f"{label} {proposal_kind} {os.path.basename(os.path.dirname(h5_path))} {os.path.basename(h5_path)}".lower()
    if any(k in joined for k in ("rl_mala", "rl-mala", "rl_td3", "td3", "sac", "neural", "deepset", "coupled_rl")):
        return "RL-MALA"
    if "adaptive" in joined:
        return "Adaptive-MALA"
    return "MALA"


def _classify_pipeline(label: str, h5_path: str = "") -> str:
    joined = f"{label} {h5_path}".lower()
    if "pipeline4" in joined or "p4_" in joined or "online_coupled" in joined:
        return "P4"
    if "pipeline3" in joined or "p3_" in joined or "vmc_mala_sjb" in joined or "coupled_rl" in joined:
        return "P3"
    return "P2"


def _parse_n_omega_g_from_text(text: str) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    n_val: Optional[int] = None
    w_val: Optional[float] = None
    g_val: Optional[float] = None

    m_sys = re.search(r"N(\d+)_w([0-9]+(?:\.[0-9]+)?)_g([0-9]+(?:\.[0-9]+)?)", text)
    if m_sys:
        return int(m_sys.group(1)), float(m_sys.group(2)), float(m_sys.group(3))

    m_n = re.search(r"(?:^|[_\-/])N(\d+)(?:[_\-/]|$)", text)
    if m_n:
        n_val = int(m_n.group(1))
    m_w = re.search(r"(?:^|[_\-/])w([0-9]+(?:\.[0-9]+)?)(?:[_\-/]|$)", text)
    if m_w:
        w_val = float(m_w.group(1))
    m_g = re.search(r"(?:^|[_\-/])g([0-9]+(?:\.[0-9]+)?)(?:[_\-/]|$)", text)
    if m_g:
        g_val = float(m_g.group(1))
    return n_val, w_val, g_val


def _load_csv_lookup_tables(root_dir: str) -> Dict[Tuple[str, int, float, str], Dict[str, float]]:
    """
    Lee los archivos CSV de resumen en `<root_dir>` y `<root_dir>/results_csv*`
    para enriquecer métricas como `kl_casscf_up_mean`, `kl_casscf_down_mean`,
    `ESJD`, `acceptance` y `tau_int_E` si no estuvieran en el HDF5.
    """
    lookup: Dict[Tuple[str, int, float, str], Dict[str, float]] = {}
    candidate_csvs: List[str] = []
    for pat in (
        os.path.join(root_dir, "live_*.csv"),
        os.path.join(root_dir, "results_csv*", "summary_metrics_*.csv"),
        os.path.join(root_dir, "results_csv*", "live_*.csv"),
        os.path.join(root_dir, "pipeline2_pure_sampling", "*", "postprocess_comparison", "summary_metrics_*.csv"),
        os.path.join(root_dir, "pipeline3_variational_training", "*", "postprocess_comparison", "summary_metrics_*.csv"),
        os.path.join(root_dir, "pipeline4_online_coupled", "*", "postprocess_comparison", "summary_metrics_*.csv"),
    ):
        candidate_csvs.extend(sorted(glob.glob(pat)))

    for csv_path in candidate_csvs:
        if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lbl = str(row.get("label", ""))
                    h5p = str(row.get("h5_path", ""))
                    n_val = _safe_int(row.get("N"), 0)
                    g_val = _safe_float(row.get("g"), float("nan"))
                    if n_val <= 0 or not np.isfinite(g_val):
                        n_p, _, g_p = _parse_n_omega_g_from_text(f"{lbl} {h5p}")
                        if n_val <= 0 and n_p is not None:
                            n_val = n_p
                        if not np.isfinite(g_val) and g_p is not None:
                            g_val = g_p
                    if n_val <= 0 or not np.isfinite(g_val):
                        continue
                    method = _classify_method(lbl, "", h5p)
                    pipe = _classify_pipeline(lbl, h5p)
                    key = (pipe, int(n_val), round(float(g_val), 6), method)

                    kl_up = _safe_float(row.get("kl_casscf_up_mean"))
                    kl_dn = _safe_float(row.get("kl_casscf_down_mean"))
                    kl_tot = _safe_float(row.get("kl_casscf_total_mean"))
                    kl_spin = _safe_float(row.get("kl_casscf_spin_mean"))
                    if not np.isfinite(kl_spin):
                        if np.isfinite(kl_up) and np.isfinite(kl_dn):
                            kl_spin = 0.5 * (kl_up + kl_dn)
                        else:
                            kl_spin = kl_tot

                    lookup[key] = {
                        "acceptance": _safe_float(row.get("acceptance")),
                        "ESJD": _safe_float(row.get("ESJD")),
                        "tau_int_E": _safe_float(row.get("tau_int_E")),
                        "tau_int_pos": _safe_float(row.get("tau_int_pos")),
                        "E_mean": _safe_float(row.get("E_mean")),
                        "SE_E": _safe_float(row.get("SE_E")),
                        "kl_total": kl_tot,
                        "kl_up": kl_up,
                        "kl_down": kl_dn,
                        "kl_spin": kl_spin,
                        "G_RL_speedup": _safe_float(row.get("G_RL_speedup")),
                    }
        except Exception:
            continue
    return lookup


def discover_h5_bundles(root_dir: str, include_pipeline3: bool = False, include_pipeline4: bool = False) -> List[str]:
    """
    Descubre todos los archivos `run_bundle*.h5` siguiendo la organización de
    `run_production_benchmark_3pipelines.sh`, con respaldo automático a búsqueda
    recursiva si se apunta a una carpeta específica de experimentos.
    """
    root_abs = os.path.abspath(root_dir)
    if os.path.isfile(root_abs) and root_abs.endswith(".h5"):
        return [root_abs]

    discovered: List[str] = []
    seen_realpaths = set()

    def _add_candidate(path: str) -> None:
        if not os.path.isfile(path):
            return
        if ".tmp." in os.path.basename(path) or ".corrupt_" in os.path.basename(path):
            return
        rp = os.path.realpath(path)
        if rp not in seen_realpaths:
            seen_realpaths.add(rp)
            discovered.append(path)

    p2_dir = os.path.join(root_abs, "pipeline2_pure_sampling")
    p3_dir = os.path.join(root_abs, "pipeline3_variational_training")
    p4_dir = os.path.join(root_abs, "pipeline4_online_coupled")

    stage_dirs: List[str] = []
    if os.path.isdir(p2_dir):
        stage_dirs.append(p2_dir)
    if include_pipeline3 and os.path.isdir(p3_dir):
        stage_dirs.append(p3_dir)
    if (include_pipeline4 or include_pipeline3) and os.path.isdir(p4_dir):
        stage_dirs.append(p4_dir)

    if stage_dirs:
        for sdir in stage_dirs:
            sys_dirs = sorted(glob.glob(os.path.join(sdir, "N*_w*_g*")))
            for sys_dir in sys_dirs:
                if not os.path.isdir(sys_dir):
                    continue
                sys_tag = os.path.basename(sys_dir)
                # Priorizar las carpetas estándar de run_production_benchmark_3pipelines.sh
                # 1) Nombres nuevos: ${SYS_TAG}_MALA_tau* y ${SYS_TAG}_RL_MALA_*
                # 2) Nombres legacy: MALA_FixedCASSCF y RL_TD3_FixedCASSCF
                subdirs = sorted(os.listdir(sys_dir))
                mala_selected = False
                rl_selected = False

                # Primero buscar carpetas nuevas parametrizadas
                for sd in subdirs:
                    sd_path = os.path.join(sys_dir, sd)
                    if not os.path.isdir(sd_path) or sd == "postprocess_comparison":
                        continue
                    h5_main = os.path.join(sd_path, "run_bundle.h5")
                    if sd.startswith(f"{sys_tag}_MALA") and os.path.isfile(h5_main):
                        _add_candidate(h5_main)
                        mala_selected = True
                    elif sd.startswith(f"{sys_tag}_RL_MALA") and os.path.isfile(h5_main):
                        _add_candidate(h5_main)
                        rl_selected = True

                # Fallback a carpetas legacy si no estaban las nuevas
                for sd in subdirs:
                    sd_path = os.path.join(sys_dir, sd)
                    if not os.path.isdir(sd_path) or sd == "postprocess_comparison":
                        continue
                    if sd == "MALA_FixedCASSCF" and mala_selected:
                        continue
                    if sd == "RL_TD3_FixedCASSCF" and rl_selected:
                        continue
                    h5_main = os.path.join(sd_path, "run_bundle.h5")
                    if os.path.isfile(h5_main):
                        _add_candidate(h5_main)
                    else:
                        for h5_alt in sorted(glob.glob(os.path.join(sd_path, "run_bundle*.h5"))):
                            _add_candidate(h5_alt)
                            break
    else:
        # Búsqueda recursiva general cuando se pasa directamente pipeline2_pure_sampling
        # o una carpeta de búsqueda como results_nodal_crossing_search_N6
        for dirpath, dirnames, filenames in os.walk(root_abs):
            dirnames[:] = [d for d in sorted(dirnames) if d != "postprocess_comparison" and not d.startswith(".")]
            if "run_bundle.h5" in filenames:
                _add_candidate(os.path.join(dirpath, "run_bundle.h5"))
            else:
                for fn in sorted(filenames):
                    if fn.endswith(".h5") and ("run_bundle" in fn or "mala" in fn.lower() or "td3" in fn.lower()):
                        _add_candidate(os.path.join(dirpath, fn))

    return discovered


def load_bundle_walkers_and_meta(h5_path: str) -> Dict[str, Any]:
    """
    Carga el tensor `production/walkers` y los metadatos del sistema desde un
    archivo HDF5 sin requerir JAX. Soporta tanto `io_bundle.load_run_bundle`
    como lectura directa con `h5py`.
    """
    import h5py

    meta: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    walkers: Optional[np.ndarray] = None
    acc_rate: Optional[np.ndarray] = None
    actor_params: Optional[Dict[str, Any]] = None

    with h5py.File(h5_path, "r") as f:
        for k, v in f.attrs.items():
            meta[k] = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v
        if "config_json" in f.attrs:
            raw_cfg = f.attrs["config_json"]
            if isinstance(raw_cfg, (bytes, np.bytes_)):
                raw_cfg = raw_cfg.decode("utf-8")
            try:
                config = json.loads(str(raw_cfg))
            except Exception:
                config = {}
        if "meta" in f:
            for k, v in f["meta"].attrs.items():
                meta[k] = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v
            if "config_json" in f["meta"]:
                raw_cfg = f["meta"]["config_json"][()]
                if isinstance(raw_cfg, (bytes, np.bytes_)):
                    raw_cfg = raw_cfg.decode("utf-8")
                try:
                    config = json.loads(str(raw_cfg))
                except Exception:
                    pass
        if "checkpoints" in f and "actor_final" in f["checkpoints"]:
            try:
                import pickle
                actor_params = pickle.loads(np.asarray(f["checkpoints"]["actor_final"][()], dtype=np.uint8).tobytes())
            except Exception:
                actor_params = None
        if "production" in f:
            pgrp = f["production"]
            if "walkers" in pgrp:
                walkers = np.asarray(pgrp["walkers"][()], dtype=np.float64)
            if "acc_rate" in pgrp:
                acc_rate = np.asarray(pgrp["acc_rate"][()], dtype=np.float64)
            if "summary" in pgrp:
                for k, v in pgrp["summary"].attrs.items():
                    summary[k] = v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else v

    if walkers is None or walkers.size == 0:
        raise ValueError(f"El archivo {h5_path} no contiene el dataset 'production/walkers'.")

    sys_cfg = config.get("system", {}) if isinstance(config, dict) else {}
    prop_cfg = config.get("proposal", {}) if isinstance(config, dict) else {}
    samp_cfg = config.get("sampler", {}) if isinstance(config, dict) else {}

    n_from_path, w_from_path, g_from_path = _parse_n_omega_g_from_text(h5_path)

    # Normalizar forma de walkers a (n_chains, n_steps, n_batch, n_elec)
    w_arr = np.asarray(walkers, dtype=np.float64)
    if w_arr.ndim == 5 and w_arr.shape[-1] == 1:
        w_arr = w_arr[..., 0]
    elif w_arr.ndim == 4:
        # Podría ser (n_chains, n_steps, n_batch, n_elec) o (n_steps, n_batch, n_elec, 1)
        if w_arr.shape[-1] == 1:
            w_arr = w_arr[None, ..., 0]
    elif w_arr.ndim == 3:
        # (n_steps, n_batch, n_elec)
        w_arr = w_arr[None, ...]
    else:
        raise ValueError(f"Forma inesperada de production/walkers en {h5_path}: {walkers.shape}")

    n_chains, n_steps, n_batch, n_elec_tensor = w_arr.shape

    n_elec = _safe_int(sys_cfg.get("n_elec", meta.get("n_elec", n_from_path)), n_elec_tensor)
    if n_elec <= 0:
        n_elec = n_elec_tensor
    n_up = _safe_int(sys_cfg.get("n_up", meta.get("n_up", n_elec // 2)), n_elec // 2)
    n_dn = _safe_int(sys_cfg.get("n_dn", meta.get("n_dn", n_elec - n_up)), n_elec - n_up)
    omega = _safe_float(sys_cfg.get("omega", meta.get("omega", w_from_path)), 0.1)
    g_val = _safe_float(sys_cfg.get("g", meta.get("g", g_from_path)), float("nan"))
    softening = _safe_float(sys_cfg.get("softening", meta.get("softening", 0.1)), 0.1)

    tau = _safe_float(prop_cfg.get("tau", samp_cfg.get("tau", meta.get("tau", 0.1))), 0.1)
    actor_amp = _safe_float(prop_cfg.get("actor_amplitude", prop_cfg.get("drift_scale", float("nan"))))
    sigma_scale = _safe_float(prop_cfg.get("sigma_action_scale", float("nan")))
    proposal_kind = str(prop_cfg.get("kind", meta.get("proposal_kind", "")))
    run_name = str(meta.get("run_name", os.path.basename(os.path.dirname(h5_path))))

    method = _classify_method(run_name, proposal_kind, h5_path)
    pipeline = _classify_pipeline(run_name, h5_path)

    acc_mean = _safe_float(summary.get("acc_rate_mean"))
    if not np.isfinite(acc_mean) and acc_rate is not None and acc_rate.size > 0:
        acc_mean = float(np.nanmean(acc_rate))

    return {
        "h5_path": h5_path,
        "run_name": run_name,
        "method": method,
        "pipeline": pipeline,
        "proposal_kind": proposal_kind,
        "N": int(n_elec),
        "n_up": int(n_up),
        "n_dn": int(n_dn),
        "omega": float(omega),
        "g": float(g_val),
        "softening": float(softening),
        "tau": float(tau),
        "actor_amplitude": float(actor_amp),
        "sigma_action_scale": float(sigma_scale),
        "parity_equivariant": bool(prop_cfg.get("parity_equivariant", False)),
        "unclip_sigma_action": bool(prop_cfg.get("unclip_sigma_action", False)),
        "scale_clip_min": _safe_float(prop_cfg.get("scale_clip_min", -5.0), -5.0),
        "scale_clip_max": _safe_float(prop_cfg.get("scale_clip_max", 1.0), 1.0),
        "n_chains": int(n_chains),
        "n_steps": int(n_steps),
        "n_batch_saved": int(n_batch),
        "walkers": w_arr,
        "actor_params": actor_params,
        "acceptance": acc_mean,
        "ESJD": _safe_float(summary.get("esjd_mean", summary.get("ESJD"))),
        "tau_int_E": _safe_float(summary.get("tau_int_E", summary.get("tau_int_mean"))),
        "tau_int_pos": _safe_float(summary.get("tau_int_pos")),
        "E_mean": _safe_float(summary.get("energy_mean", summary.get("E_mean"))),
        "SE_E": _safe_float(summary.get("energy_err", summary.get("SE_E"))),
        "kl_total": _safe_float(summary.get("kl_casscf_total_mean")),
        "kl_up": _safe_float(summary.get("kl_casscf_up_mean")),
        "kl_down": _safe_float(summary.get("kl_casscf_down_mean")),
        "kl_spin": _safe_float(summary.get("kl_casscf_spin_mean")),
    }


# ==============================================================================
# 2. ALGORITMO VECTORIZADO DE CARACTERIZACIÓN DE FLIPS Y ANATOMÍA DE CRUCE
# ==============================================================================

def compute_flip_statistics(
    walkers: np.ndarray,
    n_up: int,
    hysteresis_frac: float = 0.25,
    fixed_delta: Optional[float] = None,
    omega: float = 0.1,
    g_val: float = 1.0,
    tau: float = 0.08,
    actor_amp: float = 0.10,
    sigma_scale: float = 0.90,
    actor_params: Optional[Dict[str, Any]] = None,
    is_neural: bool = False,
    npz_path: Optional[str] = None,
    parity_equivariant: bool = False,
    unclip_sigma_action: bool = False,
    scale_clip_min: float = -5.0,
    scale_clip_max: float = 1.0,
) -> Dict[str, Any]:
    """
    Calcula todas las métricas de intercambio entre pares de electrones (flips)
    sobre un tensor `walkers` de forma `(C, T, B, N)`:

      - Separación media entre vecinos adyacentes ordenados `d_adj` y umbral de
        histéresis `delta = hysteresis_frac * d_adj` (o `fixed_delta` si se da).
      - Flips crudos por pares (`raw_flips_per_1k`): cambios de signo de x_i - x_j
        entre t-1 y t por cada 1,000 pasos y por caminante.
      - Flips con histéresis (`hyst_flips_per_1k`): intercambios reales pozo-a-pozo
        donde x_i - x_j pasa de < -delta a > +delta (o viceversa), eliminando las
        oscilaciones térmicas microscópicas alrededor de x_i = x_j en g < 0.3.
      - Desglose exacto entre pares de espín opuesto (up-down, barrera de Wigner)
        y pares del mismo espín (up-up / down-down, nodo exacto de Pauli).
      - Duración del tránsito de barrera (`Delta t_transit`): número de pasos de
        Markov que el par pasa dentro de [-delta, +delta] durante un flip exitoso,
        y fracción de saltos en un único paso colectivo (`one_step_fraction`).
      - Anatomía temporal alineada en `t - t_flip in [-window, +window]` para
        visualizar la aproximación multi-paso y el brinco de 1 paso.
    """
    w = np.asarray(walkers, dtype=np.float64)
    if w.ndim != 4:
        raise ValueError(f"Se esperaba tensor walkers de 4 dimensiones (C, T, B, N), se recibió: {w.shape}")

    C, T, B, N = w.shape
    if T < 2 or N < 2:
        raise ValueError(f"Trayectoria demasiado corta o con N<2: shape={w.shape}")

    # 0. Recorte automático de transitorio inicial si la cadena inició descentrada (e.g. spin_ordered_offset con burn-in corto)
    if T > 4000:
        com_init = float(np.nanmean(np.abs(np.nanmean(w[:, :min(50, T), :, :], axis=-1))))
        if com_init > 2.2:
            t_trim = min(1800, T // 12)
            w = w[:, t_trim:, :, :]
            T = w.shape[1]

    # Filtrado de caminantes confinados físicamente (descarta caminantes fugados al piso numérico 1e-15 en |x| >> R_Wigner)
    # Para N=8 y g alto, el radio de confinamiento físico armónico es R ~ 28 Bohr (|x| max ~ 28).
    # Fijamos una cota física robusta que escala con sqrt((N-1)*d_adj):
    max_phys_bound = max(28.0, 4.0 * float(N + 1))
    walker_max_abs = np.nanmax(np.abs(w), axis=(1, 3))  # (C, B)
    walker_com_abs = np.nanmean(np.abs(np.nanmean(w, axis=-1)), axis=1)  # (C, B)
    walker_t_std = np.nanstd(w, axis=1)  # (C, B, N)
    walker_mean_disp = np.nanmean(walker_t_std, axis=-1)  # (C, B)

    # Un caminante sano debe estar confinado (|x| <= max_phys_bound), con centro de masa cercano a 0 (|com| <= 6.0),
    # y sin dispersión infinita por difusión libre desatada (disp <= 25.0).
    healthy_mask = (walker_max_abs <= max_phys_bound) & (walker_com_abs <= 6.0) & (walker_mean_disp <= 25.0)  # (C, B)
    if not np.any(healthy_mask):
        # Si ningún caminante cumple la cota estricta (e.g. en corridas viejas donde todos escaparon),
        # elegir el 35% de caminantes con menor valor máximo absoluto para evitar seleccionar los más alejados
        q35 = float(np.nanpercentile(walker_max_abs, 35))
        healthy_mask = (walker_max_abs <= max(q35, 15.0))
        if not np.any(healthy_mask):
            healthy_mask = np.ones((C, B), dtype=bool)

    # 1. Separación característica entre vecinos adyacentes ordenados espacialmente (sobre caminantes sanos)
    w_sorted = np.sort(w, axis=-1)
    adj_diffs = np.diff(w_sorted, axis=-1)  # (C, T, B, N-1)
    healthy_adj = adj_diffs[np.broadcast_to(healthy_mask[:, None, :, None], adj_diffs.shape)]
    d_adj = float(np.nanmedian(healthy_adj)) if healthy_adj.size > 0 else float(np.nanmean(adj_diffs))
    if not np.isfinite(d_adj) or d_adj <= 1e-8:
        d_adj = 1.0

    delta = float(fixed_delta) if (fixed_delta is not None and fixed_delta > 0) else float(hysteresis_frac * d_adj)

    # Desplazamiento cuadrático medio por paso (empírico desde las posiciones guardadas)
    step_disp_sq = np.sum(np.diff(w, axis=1) ** 2, axis=-1)  # (C, T-1, B)
    empirical_esjd = float(np.nanmean(step_disp_sq))
    rms_step_per_elec = float(np.sqrt(np.nanmean(np.diff(w, axis=1) ** 2)))

    # 2. Construir todos los pares (i, j) con 0 <= i < j < N y clasificar su espín
    pair_i, pair_j = np.triu_indices(N, k=1)
    n_pairs = len(pair_i)
    spins = np.array([1 if idx < n_up else -1 for idx in range(N)], dtype=np.int32)
    is_opp_spin = (spins[pair_i] != spins[pair_j])   # True = up-down, False = same-spin
    is_same_spin = ~is_opp_spin
    n_opp_pairs = int(np.sum(is_opp_spin))
    n_same_pairs = int(np.sum(is_same_spin))

    # Diferencias relativas d_{c, t, b, p} = x_i - x_j de forma (C, T, B, P)
    d_pairs = w[..., pair_i] - w[..., pair_j]

    # --------------------------------------------------------------------------
    # A. Conteo de Flips Crudos (delta = 0)
    # --------------------------------------------------------------------------
    raw_signs = np.sign(d_pairs)
    # Propagar hacia adelante ceros exactos si ocurrieran
    nz_raw = (raw_signs != 0)
    idx_raw = np.where(nz_raw, np.arange(T)[None, :, None, None], 0)
    np.maximum.accumulate(idx_raw, axis=1, out=idx_raw)
    raw_signs_filled = np.take_along_axis(raw_signs, idx_raw, axis=1)

    raw_flip_mask = (raw_signs_filled[:, 1:, :, :] * raw_signs_filled[:, :-1, :, :]) == -1.0  # (C, T-1, B, P)
    raw_flip_mask = raw_flip_mask & healthy_mask[:, None, :, None]

    healthy_counts_per_chain = np.maximum(1.0, np.sum(healthy_mask, axis=1).astype(np.float64))
    norm_factor_per_chain = 1000.0 / (float(T - 1) * healthy_counts_per_chain)
    raw_total_per_chain = np.sum(raw_flip_mask, axis=(1, 2, 3)) * norm_factor_per_chain
    raw_opp_per_chain = (
        np.sum(raw_flip_mask[..., is_opp_spin], axis=(1, 2, 3)) * norm_factor_per_chain
        if n_opp_pairs > 0 else np.zeros(C, dtype=np.float64)
    )
    raw_same_per_chain = (
        np.sum(raw_flip_mask[..., is_same_spin], axis=(1, 2, 3)) * norm_factor_per_chain
        if n_same_pairs > 0 else np.zeros(C, dtype=np.float64)
    )

    # --------------------------------------------------------------------------
    # B. Conteo de Flips con Histéresis (|x_i - x_j| > delta)
    # --------------------------------------------------------------------------
    # s_thresh vale +1 si x_i - x_j > delta, -1 si x_i - x_j < -delta, y 0 dentro de la barrera
    s_thresh = np.where(d_pairs > delta, 1, np.where(d_pairs < -delta, -1, 0)).astype(np.int8)
    nz_hyst = (s_thresh != 0)
    idx_hyst = np.where(nz_hyst, np.arange(T)[None, :, None, None], 0)
    np.maximum.accumulate(idx_hyst, axis=1, out=idx_hyst)
    s_committed = np.take_along_axis(s_thresh, idx_hyst, axis=1)

    # Un flip con histéresis ocurre en el paso t (entre t-1 y t, índice t-1 en diff)
    # cuando el estado comprometido pasa de -1 a +1 o de +1 a -1
    hyst_flip_mask = (s_committed[:, 1:, :, :] * s_committed[:, :-1, :, :]) == -1  # (C, T-1, B, P)
    hyst_flip_mask = hyst_flip_mask & healthy_mask[:, None, :, None]

    hyst_total_per_chain = np.sum(hyst_flip_mask, axis=(1, 2, 3)) * norm_factor_per_chain
    hyst_opp_per_chain = (
        np.sum(hyst_flip_mask[..., is_opp_spin], axis=(1, 2, 3)) * norm_factor_per_chain
        if n_opp_pairs > 0 else np.zeros(C, dtype=np.float64)
    )
    hyst_same_per_chain = (
        np.sum(hyst_flip_mask[..., is_same_spin], axis=(1, 2, 3)) * norm_factor_per_chain
        if n_same_pairs > 0 else np.zeros(C, dtype=np.float64)
    )

    # Series temporales acumuladas por paso (promediadas sobre caminantes sanos) para graficar N_flips(t)
    # Forma (C, T-1): flips por caminante en cada paso t
    raw_step_rate = np.sum(np.sum(raw_flip_mask, axis=-1), axis=-1) / healthy_counts_per_chain[:, None]
    hyst_step_rate = np.sum(np.sum(hyst_flip_mask, axis=-1), axis=-1) / healthy_counts_per_chain[:, None]
    hyst_opp_step_rate = (
        np.sum(np.sum(hyst_flip_mask[..., is_opp_spin], axis=-1), axis=-1) / healthy_counts_per_chain[:, None]
        if n_opp_pairs > 0 else np.zeros((C, T - 1), dtype=np.float64)
    )
    hyst_same_step_rate = (
        np.sum(np.sum(hyst_flip_mask[..., is_same_spin], axis=-1), axis=-1) / healthy_counts_per_chain[:, None]
        if n_same_pairs > 0 else np.zeros((C, T - 1), dtype=np.float64)
    )

    # --------------------------------------------------------------------------
    # C. Duración del Tránsito de Barrera (1 Paso vs Multi-Paso) y Saltos en el Cruce
    # --------------------------------------------------------------------------
    # Para cada flip con histéresis en t_exit = tau_idx + 1 (con tau_idx en 0..T-2),
    # el último instante previo en que el par estuvo fuera de la barrera [-delta, +delta]
    # es exactamente idx_hyst[:, tau_idx, :, :]!
    # Por lo tanto, el número de pasos empleados en cruzar la zona [-delta, +delta] es:
    #   Delta t_transit = (tau_idx + 1) - idx_hyst[:, tau_idx, :, :]
    # Si Delta t_transit == 1, el par saltó directamente de |d| > delta a |d| > delta con signo
    # opuesto en UN SOLO PASO de la cadena de Markov sin pisar la zona prohibida [-delta, +delta].
    ev_c, ev_tau, ev_b, ev_p = np.where(hyst_flip_mask)
    if len(ev_tau) > 0:
        t_enter = idx_hyst[ev_c, ev_tau, ev_b, ev_p]
        t_exit = ev_tau + 1
        transit_steps = (t_exit - t_enter).astype(np.int32)
        # Desplazamiento relativo instantáneo en el paso exacto del cruce
        jump_at_flip = np.abs(d_pairs[ev_c, t_exit, ev_b, ev_p] - d_pairs[ev_c, ev_tau, ev_b, ev_p])
        one_step_frac = float(np.mean(transit_steps == 1))
        two_step_frac = float(np.mean(transit_steps <= 2))
        transit_mean = float(np.mean(transit_steps))
        transit_median = float(np.median(transit_steps))
        jump_at_flip_mean = float(np.mean(jump_at_flip))
        jump_at_flip_std = float(np.std(jump_at_flip))
    else:
        # Si no hubo flips con histéresis pero sí crudos, medir sobre flips crudos
        rc_c, rc_tau, rc_b, rc_p = np.where(raw_flip_mask)
        transit_steps = np.array([], dtype=np.int32)
        if len(rc_tau) > 0:
            jump_at_flip = np.abs(d_pairs[rc_c, rc_tau + 1, rc_b, rc_p] - d_pairs[rc_c, rc_tau, rc_b, rc_p])
            jump_at_flip_mean = float(np.mean(jump_at_flip))
            jump_at_flip_std = float(np.std(jump_at_flip))
        else:
            jump_at_flip = np.array([], dtype=np.float64)
            jump_at_flip_mean = float("nan")
            jump_at_flip_std = float("nan")
        one_step_frac = float("nan")
        two_step_frac = float("nan")
        transit_mean = float("nan")
        transit_median = float("nan")

    # Desplazamiento relativo medio por par en pasos normales (para comparar con jump_at_flip)
    normal_pair_jump_mean = float(np.mean(np.abs(np.diff(d_pairs, axis=1))))

    # --------------------------------------------------------------------------
    # D. Cruces del origen x = 0 (para referencia con plotter.py)
    # --------------------------------------------------------------------------
    origin_signs = np.where(w > 0.25 * delta, 1, np.where(w < -0.25 * delta, -1, 0)).astype(np.int8)
    nz_orig = (origin_signs != 0)
    idx_orig = np.where(nz_orig, np.arange(T)[None, :, None, None], 0)
    np.maximum.accumulate(idx_orig, axis=1, out=idx_orig)
    orig_committed = np.take_along_axis(origin_signs, idx_orig, axis=1)
    orig_flips_mask = (orig_committed[:, 1:, :, :] * orig_committed[:, :-1, :, :]) == -1
    origin_flips_per_1k = float(np.mean(np.sum(orig_flips_mask, axis=(1, 2, 3)) * norm_factor_per_chain))

    # --------------------------------------------------------------------------
    # E. Extracción de Ventanas Microscópicas DISCRETAS [-W, +W] en pasos enteros
    #    SIN interpolación lineal fraccional (np.interp):
    #    - En k = 0 (paso entero t_0), el par está en su lado inicial: x_i(t_0) < x_j(t_0).
    #    - En k = +1 (paso entero t_0 + 1), el par YA BRINCÓ al otro lado: x_i(t_0+1) > x_j(t_0+1).
    #    - Se evalúa el Actor real (mu_theta, sigma_theta) y la magnitud real del paso
    #      |<|Delta x(k)|>| = <|x(t_0+k+1) - x(t_0+k)|> en k in [-W, +W].
    #    - Se separan eventos individuales de espín opuesto (up-down) y mismo espín (up-up/down-down).
    # --------------------------------------------------------------------------
    window = 15
    rel_steps_grid = np.arange(-window, window + 1, dtype=int)
    aligned_rel_dist: List[np.ndarray] = []
    aligned_rel_dist_norm: List[np.ndarray] = []
    aligned_xi_norm: List[np.ndarray] = []
    aligned_xj_norm: List[np.ndarray] = []
    aligned_step_mag_pair: List[np.ndarray] = []
    aligned_step_mag_spec: List[np.ndarray] = []
    ordered_all_elec_events: List[Tuple[np.ndarray, np.ndarray, int, int, float]] = []
    event_configs_for_actor: List[Tuple[np.ndarray, int, int]] = []
    example_event: Optional[Dict[str, Any]] = None

    from scripts.plot_flips_system_detailed import evaluate_legitimate_actor_on_events

    source_mask = hyst_flip_mask if len(ev_tau) > 0 else raw_flip_mask
    sc, stau, sb, sp = np.where(source_mask)
    valid_ev = np.where((stau >= window + 2) & (stau + 2 + window < T))[0]

    def _extract_discrete_event_meta(idx_ev_local: int) -> Optional[Dict[str, Any]]:
        c0, tau0, b0, p0 = int(sc[idx_ev_local]), int(stau[idx_ev_local]), int(sb[idx_ev_local]), int(sp[idx_ev_local])
        tau_exit = tau0 + 1
        if len(ev_tau) > 0:
            tau_enter = int(idx_hyst[c0, tau0, b0, p0])
            s_after = int(s_committed[c0, tau_exit, b0, p0])
        else:
            tau_enter = tau0
            s_after = int(np.sign(d_pairs[c0, tau_exit, b0, p0]))
        if s_after == 0:
            s_after = 1
        s_before = -s_after

        seg_trans = d_pairs[c0, tau_enter : tau_exit + 1, b0, p0]
        pre_candidates = np.where(np.sign(seg_trans[:-1]) == s_before)[0]
        k0 = int(pre_candidates[-1]) if len(pre_candidates) > 0 else max(0, len(seg_trans) - 2)
        t0 = tau_enter + k0

        if t0 - window < 0 or t0 + window + 1 >= T:
            return None

        a_el, b_el = int(pair_i[p0]), int(pair_j[p0])
        if s_before == -1:
            i_left, j_right = a_el, b_el
        else:
            i_left, j_right = b_el, a_el

        # Extraer pasos enteros exactos [t0 - window .. t0 + window + 1] (longitud 2*window + 2)
        w_ext = w[c0, t0 - window : t0 + window + 2, b0, :].astype(np.float64)
        w_win = w_ext[:-1, :]  # (2*window + 1, N) en k = -window .. +window
        step_disp = np.abs(np.diff(w_ext, axis=0))  # (2*window + 1, N): |x(t0+k+1) - x(t0+k)|

        xi_win = w_win[:, i_left].copy()
        xj_win = w_win[:, j_right].copy()
        dij_win = xi_win - xj_win

        d0_val = float(dij_win[window])
        d1_val = float(dij_win[window + 1])
        if d0_val > 0.0 and d1_val < 0.0:
            xi_win, xj_win = xj_win, xi_win
            i_left, j_right = j_right, i_left
            dij_win = -dij_win
            d0_val, d1_val = -d0_val, -d1_val

        # Punto medio geométrico entre los dos pozos involucrados (en k in [-8, -2])
        x_mid = 0.5 * float(
            np.mean(xi_win[max(0, window - 8) : window - 1])
            + np.mean(xj_win[max(0, window - 8) : window - 1])
        )

        pair_step_mag = 0.5 * (step_disp[:, i_left] + step_disp[:, j_right])
        spec_indices = [el for el in range(N) if el not in (i_left, j_right)]
        if spec_indices:
            spec_step_mag = np.mean(step_disp[:, spec_indices], axis=1)
        else:
            spec_step_mag = pair_step_mag.copy()

        # Ordenar los N electrones por su pozo espacial previo al flip (en [-8, -2])
        pre_mean = np.mean(w_win[max(0, window - 8) : max(1, window - 1), :], axis=0)
        pre_order = np.argsort(pre_mean)
        inv_order = np.argsort(pre_order)
        w_ord = w_win[:, pre_order]
        sp_ord = spins[pre_order]
        well_l = int(inv_order[i_left])
        well_r = int(inv_order[j_right])

        is_clean = bool(
            np.all(dij_win[window - 3 : window + 1] < 0.0)
            and np.all(dij_win[window + 1 : window + 5] > 0.0)
            and abs(well_r - well_l) == 1
        )
        jump_size = float(dij_win[window + 1] - dij_win[window])
        transit_len = int(tau_exit - tau_enter)

        return {
            "is_clean": is_clean,
            "jump_size": jump_size,
            "transit_steps": transit_len,
            "c0": c0,
            "b0": b0,
            "p0": p0,
            "t0": t0,
            "i_left": i_left,
            "j_right": j_right,
            "well_left": min(well_l, well_r),
            "well_right": max(well_l, well_r),
            "x_cross": x_mid,
            "w_win": w_win,
            "w_ord": w_ord,
            "sp_ord": sp_ord,
            "dij_win": dij_win,
            "dij_norm": dij_win / d_adj,
            "xi_norm": (xi_win - x_mid) / d_adj,
            "xj_norm": (xj_win - x_mid) / d_adj,
            "pair_step_mag": pair_step_mag,
            "spec_step_mag": spec_step_mag,
            "is_opp": bool(is_opp_spin[p0]),
        }

    opp_single_info: Optional[Dict[str, Any]] = None
    same_single_info: Optional[Dict[str, Any]] = None

    if len(valid_ev) > 0:
        opp_ev = [idx for idx in valid_ev if is_opp_spin[sp[idx]]]
        same_ev = [idx for idx in valid_ev if is_same_spin[sp[idx]]]
        chosen_indices = opp_ev if len(opp_ev) > 0 else list(valid_ev)

        if len(chosen_indices) > 500:
            sel_pos = np.linspace(0, len(chosen_indices) - 1, 500, dtype=int)
            sample_indices = [chosen_indices[ii] for ii in sel_pos]
        else:
            sample_indices = list(chosen_indices)

        extracted_opp = [m for idx_ev in sample_indices if (m := _extract_discrete_event_meta(idx_ev)) is not None]
        only_clean = [m for m in extracted_opp if m["is_clean"]]
        selected_meta = only_clean if len(only_clean) >= min(15, len(extracted_opp)) else extracted_opp

        for m in selected_meta:
            aligned_rel_dist.append(m["dij_win"])
            aligned_rel_dist_norm.append(m["dij_norm"])
            aligned_xi_norm.append(m["xi_norm"])
            aligned_xj_norm.append(m["xj_norm"])
            aligned_step_mag_pair.append(m["pair_step_mag"])
            aligned_step_mag_spec.append(m["spec_step_mag"])
            ordered_all_elec_events.append((m["w_ord"], m["sp_ord"], m["well_left"], m["well_right"], m["x_cross"]))
            event_configs_for_actor.append((m["w_win"], m["i_left"], m["j_right"]))

        if selected_meta:
            one_step_clean = [m for m in selected_meta if m["transit_steps"] == 1 and m["is_clean"]]
            pool_best = one_step_clean if one_step_clean else selected_meta
            best_m = max(pool_best, key=lambda m: m["jump_size"])
            opp_single_info = {
                "rel_steps": rel_steps_grid.copy(),
                "well_left": int(best_m["well_left"]),
                "well_right": int(best_m["well_right"]),
                "x_cross": float(best_m["x_cross"]),
                "example_x": best_m["w_ord"].copy(),
                "example_spins": best_m["sp_ord"].copy(),
                "walker_idx": int(best_m["b0"]),
                "chain_idx": int(best_m["c0"]),
                "t0": int(best_m["t0"]),
                "transit_steps": int(best_m["transit_steps"]),
                "spin_channel": "opp",
            }
            example_event = {
                "chain": int(best_m["c0"]),
                "walker": int(best_m["b0"]),
                "t_flip": int(best_m["t0"]),
                "elec_i": int(best_m["i_left"]),
                "elec_j": int(best_m["j_right"]),
                "spin_i": int(spins[best_m["i_left"]]),
                "spin_j": int(spins[best_m["j_right"]]),
                "is_opp_spin": bool(best_m["is_opp"]),
                "rel_steps": rel_steps_grid.copy(),
                "x_i": best_m["w_win"][:, best_m["i_left"]].copy(),
                "x_j": best_m["w_win"][:, best_m["j_right"]].copy(),
                "x_all": best_m["w_win"].copy(),
                "d_ij": best_m["dij_win"].copy(),
                "d_ij_norm": best_m["dij_norm"].copy(),
                "x_i_norm": best_m["xi_norm"].copy(),
                "x_j_norm": best_m["xj_norm"].copy(),
            }

        # Extraer también el mejor evento individual de MISMO ESPÍN (nodo de Pauli Psi = 0)
        same_candidates = list(same_ev)
        if not same_candidates and np.any(is_same_spin):
            rc_c_s, rc_tau_s, rc_b_s, rc_p_s = np.where(raw_flip_mask)
            valid_raw_same = [
                k for k in range(len(rc_tau_s))
                if is_same_spin[rc_p_s[k]] and (rc_tau_s[k] >= window + 2) and (rc_tau_s[k] + 2 + window < T)
            ]
            if valid_raw_same:
                # Extraer directamente de raw_flip_mask para mismo espín si no hubo con histéresis completa
                sc_orig, stau_orig, sb_orig, sp_orig = sc, stau, sb, sp
                sc, stau, sb, sp = rc_c_s, rc_tau_s, rc_b_s, rc_p_s
                same_sample_r = valid_raw_same[:250]
                extracted_same_r = [m for idx_ev in same_sample_r if (m := _extract_discrete_event_meta(idx_ev)) is not None]
                sc, stau, sb, sp = sc_orig, stau_orig, sb_orig, sp_orig
                if extracted_same_r:
                    best_same = max(extracted_same_r, key=lambda m: m["jump_size"])
                    same_single_info = {
                        "rel_steps": rel_steps_grid.copy(),
                        "well_left": int(best_same["well_left"]),
                        "well_right": int(best_same["well_right"]),
                        "x_cross": float(best_same["x_cross"]),
                        "example_x": best_same["w_ord"].copy(),
                        "example_spins": best_same["sp_ord"].copy(),
                        "walker_idx": int(best_same["b0"]),
                        "chain_idx": int(best_same["c0"]),
                        "t0": int(best_same["t0"]),
                        "transit_steps": int(best_same["transit_steps"]),
                        "spin_channel": "same",
                    }
        if same_single_info is None and len(same_ev) > 0:
            same_sample = same_ev[:250] if len(same_ev) > 250 else same_ev
            extracted_same = [m for idx_ev in same_sample if (m := _extract_discrete_event_meta(idx_ev)) is not None]
            if extracted_same:
                clean_same = [m for m in extracted_same if m["is_clean"]]
                pool_same = clean_same if clean_same else extracted_same
                best_same = max(pool_same, key=lambda m: m["jump_size"])
                same_single_info = {
                    "rel_steps": rel_steps_grid.copy(),
                    "well_left": int(best_same["well_left"]),
                    "well_right": int(best_same["well_right"]),
                    "x_cross": float(best_same["x_cross"]),
                    "example_x": best_same["w_ord"].copy(),
                    "example_spins": best_same["sp_ord"].copy(),
                    "walker_idx": int(best_same["b0"]),
                    "chain_idx": int(best_same["c0"]),
                    "t0": int(best_same["t0"]),
                    "transit_steps": int(best_same["transit_steps"]),
                    "spin_channel": "same",
                }

    # Evaluar legítimamente el Actor real sobre un subconjunto representativo de ventanas
    actor_eval_subset = event_configs_for_actor[:60]
    if actor_eval_subset:
        w_events_arr = np.stack([x[0] for x in actor_eval_subset], axis=0)
        i_left_arr = np.array([x[1] for x in actor_eval_subset], dtype=np.int32)
        j_right_arr = np.array([x[2] for x in actor_eval_subset], dtype=np.int32)
        aligned_abs_mu_actor_arr, aligned_abs_mu_arr, aligned_sigma_arr = evaluate_legitimate_actor_on_events(
            w_events_arr,
            i_left_arr=i_left_arr,
            j_right_arr=j_right_arr,
            n_up=n_up,
            omega=omega,
            g_val=g_val,
            tau=tau,
            actor_amp=actor_amp,
            sigma_scale=sigma_scale,
            actor_params=actor_params,
            is_neural=is_neural,
            npz_path=npz_path,
            parity_equivariant=parity_equivariant,
            unclip_sigma_action=unclip_sigma_action,
            scale_clip_min=scale_clip_min,
            scale_clip_max=scale_clip_max,
        )
    else:
        aligned_abs_mu_actor_arr = np.empty((0, 2 * window + 1))
        aligned_abs_mu_arr = np.empty((0, 2 * window + 1))
        aligned_sigma_arr = np.empty((0, 2 * window + 1))

    # Construir `before_after_info` con las posiciones y espines de los N electrones en pasos discretos
    before_after_info: Optional[Dict[str, Any]] = None
    if ordered_all_elec_events:
        pair_counts: Dict[Tuple[int, int], int] = {}
        for _, _, wl, wr, _ in ordered_all_elec_events:
            key_w = (min(wl, wr), max(wl, wr))
            pair_counts[key_w] = pair_counts.get(key_w, 0) + 1
        dom_wl, dom_wr = max(pair_counts.keys(), key=lambda k: pair_counts[k])
        matching_evs = [ev for ev in ordered_all_elec_events if (min(ev[2], ev[3]), max(ev[2], ev[3])) == (dom_wl, dom_wr)]
        if not matching_evs:
            matching_evs = ordered_all_elec_events

        stack_x = np.stack([ev[0] for ev in matching_evs], axis=0)  # (M, 2*W+1, N)
        ens_x = np.median(stack_x, axis=0)
        ens_std = np.std(stack_x, axis=0)
        ex_w_ord, ex_sp_ord, _, _, ex_xcr = matching_evs[0]
        if opp_single_info is None:
            opp_single_info = {
                "rel_steps": rel_steps_grid.copy(),
                "well_left": int(dom_wl),
                "well_right": int(dom_wr),
                "x_cross": float(ex_xcr),
                "example_x": ex_w_ord.copy(),
                "example_spins": ex_sp_ord.copy(),
                "walker_idx": 0,
                "chain_idx": 0,
                "t0": 0,
                "transit_steps": 1,
                "spin_channel": "opp",
            }
        before_after_info = {
            "rel_steps": rel_steps_grid.copy(),
            "well_left": int(opp_single_info["well_left"]),
            "well_right": int(opp_single_info["well_right"]),
            "x_cross": float(opp_single_info["x_cross"]),
            "example_x": opp_single_info["example_x"],
            "example_std": ens_std,
            "example_spins": opp_single_info["example_spins"],
            "walker_idx": int(opp_single_info["walker_idx"]),
            "chain_idx": int(opp_single_info["chain_idx"]),
            "t0": int(opp_single_info["t0"]),
            "transit_steps": int(opp_single_info["transit_steps"]),
            "ensemble_x": ens_x,
            "ensemble_std": ens_std,
            "ensemble_spins": ex_sp_ord,
            "opp_single": opp_single_info,
            "same_single": same_single_info,
            # Llaves planas para graficación directa y exportación CSV
            "opp_single_x": opp_single_info["example_x"],
            "opp_single_spins": opp_single_info["example_spins"],
            "opp_single_well_left": int(opp_single_info["well_left"]),
            "opp_single_well_right": int(opp_single_info["well_right"]),
            "opp_single_x_cross": float(opp_single_info["x_cross"]),
            "opp_single_t0": int(opp_single_info["t0"]),
            "opp_single_walker": int(opp_single_info["walker_idx"]),
            "opp_single_chain": int(opp_single_info["chain_idx"]),
            "opp_single_transit_steps": int(opp_single_info["transit_steps"]),
            "same_single_x": same_single_info["example_x"] if same_single_info is not None else None,
            "same_single_spins": same_single_info["example_spins"] if same_single_info is not None else None,
            "same_single_well_left": int(same_single_info["well_left"]) if same_single_info is not None else 0,
            "same_single_well_right": int(same_single_info["well_right"]) if same_single_info is not None else 1,
            "same_single_x_cross": float(same_single_info["x_cross"]) if same_single_info is not None else 0.0,
            "same_single_t0": int(same_single_info["t0"]) if same_single_info is not None else 0,
            "same_single_walker": int(same_single_info["walker_idx"]) if same_single_info is not None else 0,
            "same_single_chain": int(same_single_info["chain_idx"]) if same_single_info is not None else 0,
            "same_single_transit_steps": int(same_single_info["transit_steps"]) if same_single_info is not None else 1,
        }

    # Seleccionar el caminante SANO representativo con más flips reales para el Panel (a)
    flips_per_cb = np.sum(source_mask, axis=(1, 3)).astype(np.float64)  # (C, B)
    walker_mean_adj = np.nanmean(adj_diffs, axis=(1, 3))  # (C, B)
    expected_wigner_extent = 0.5 * float(N + 1) * d_adj + 6.0
    well_formed_mask = healthy_mask & (np.abs(walker_mean_adj - d_adj) <= 0.35 * d_adj) & (walker_max_abs <= expected_wigner_extent)
    if not np.any(well_formed_mask):
        well_formed_mask = healthy_mask & (np.abs(walker_mean_adj - d_adj) <= 0.50 * d_adj)
        if not np.any(well_formed_mask):
            well_formed_mask = healthy_mask
    masked_flips_cb = np.where(well_formed_mask, flips_per_cb, -1.0)
    best_cb_flat = int(np.argmax(masked_flips_cb))
    best_c, best_b = divmod(best_cb_flat, B)
    best_walker_trace = w[best_c, :, best_b, :].copy()  # (T, N)
    best_walker_flips_t = np.where(np.any(source_mask[best_c, :, best_b, :], axis=-1))[0] + 1

    # Histograma marginal de densidad por espín sobre los caminantes sanos
    x_max_marg = float(np.ceil(max(20.0, 0.56 * float(N) * d_adj + 5.0)))
    edges_marg = np.linspace(-x_max_marg, x_max_marg, 161)
    x_centers_marg = 0.5 * (edges_marg[:-1] + edges_marg[1:])
    w_up_healthy = w[:, :, :, :n_up][np.broadcast_to(healthy_mask[:, None, :, None], (C, T, B, n_up))]
    w_dn_healthy = w[:, :, :, n_up:][np.broadcast_to(healthy_mask[:, None, :, None], (C, T, B, N - n_up))]
    up_hist_marg, _ = np.histogram(w_up_healthy, bins=edges_marg, density=True) if w_up_healthy.size > 0 else (np.zeros(160), edges_marg)
    dn_hist_marg, _ = np.histogram(w_dn_healthy, bins=edges_marg, density=True) if w_dn_healthy.size > 0 else (np.zeros(160), edges_marg)

    aligned_arr = np.asarray(aligned_rel_dist, dtype=np.float64) if aligned_rel_dist else np.empty((0, 2 * window + 1))
    aligned_norm_arr = np.asarray(aligned_rel_dist_norm, dtype=np.float64) if aligned_rel_dist_norm else np.empty((0, 2 * window + 1))
    aligned_xi_norm_arr = np.asarray(aligned_xi_norm, dtype=np.float64) if aligned_xi_norm else np.empty((0, 2 * window + 1))
    aligned_xj_norm_arr = np.asarray(aligned_xj_norm, dtype=np.float64) if aligned_xj_norm else np.empty((0, 2 * window + 1))
    aligned_step_pair_arr = np.asarray(aligned_step_mag_pair, dtype=np.float64) if aligned_step_mag_pair else np.empty((0, 2 * window + 1))
    aligned_step_spec_arr = np.asarray(aligned_step_mag_spec, dtype=np.float64) if aligned_step_mag_spec else np.empty((0, 2 * window + 1))

    return {
        "d_adj": d_adj,
        "delta_hyst": delta,
        "hysteresis_frac": float(hysteresis_frac),
        "n_pairs_total": int(n_pairs),
        "n_opp_pairs": int(n_opp_pairs),
        "n_same_pairs": int(n_same_pairs),
        "empirical_esjd": empirical_esjd,
        "rms_step_per_elec": rms_step_per_elec,
        # Tasas crudas (flips / 1000 pasos / caminante)
        "raw_flips_total_mean": float(np.mean(raw_total_per_chain)),
        "raw_flips_total_std": float(np.std(raw_total_per_chain)),
        "raw_flips_opp_mean": float(np.mean(raw_opp_per_chain)),
        "raw_flips_opp_std": float(np.std(raw_opp_per_chain)),
        "raw_flips_same_mean": float(np.mean(raw_same_per_chain)),
        "raw_flips_same_std": float(np.std(raw_same_per_chain)),
        # Tasas con histéresis (flips / 1000 pasos / caminante)
        "hyst_flips_total_mean": float(np.mean(hyst_total_per_chain)),
        "hyst_flips_total_std": float(np.std(hyst_total_per_chain)),
        "hyst_flips_opp_mean": float(np.mean(hyst_opp_per_chain)),
        "hyst_flips_opp_std": float(np.std(hyst_opp_per_chain)),
        "hyst_flips_same_mean": float(np.mean(hyst_same_per_chain)),
        "hyst_flips_same_std": float(np.std(hyst_same_per_chain)),
        "origin_flips_per_1k": origin_flips_per_1k,
        "total_hyst_events_count": int(len(ev_tau)),
        "total_raw_events_count": int(np.sum(raw_flip_mask)),
        # Anatomía 1 paso vs multi-paso
        "one_step_fraction": one_step_frac,
        "two_step_fraction": two_step_frac,
        "transit_steps_mean": transit_mean,
        "transit_steps_median": transit_median,
        "jump_at_flip_mean": jump_at_flip_mean,
        "jump_at_flip_std": jump_at_flip_std,
        "normal_pair_jump_mean": normal_pair_jump_mean,
        # Arreglos para graficación detallada por sistema (N, g)
        "transit_steps_samples": transit_steps[:5000].copy(),
        "jump_at_flip_samples": jump_at_flip[:5000].copy(),
        "raw_step_rate": raw_step_rate,
        "hyst_step_rate": hyst_step_rate,
        "hyst_opp_step_rate": hyst_opp_step_rate,
        "hyst_same_step_rate": hyst_same_step_rate,
        "aligned_rel_dist": aligned_arr,
        "aligned_rel_dist_norm": aligned_norm_arr,
        "aligned_xi_norm": aligned_xi_norm_arr,
        "aligned_xj_norm": aligned_xj_norm_arr,
        "aligned_abs_mu_actor": aligned_abs_mu_actor_arr,
        "aligned_abs_mu": aligned_abs_mu_arr,
        "aligned_sigma": aligned_sigma_arr,
        "aligned_step_mag_pair": aligned_step_pair_arr,
        "aligned_step_mag_spec": aligned_step_spec_arr,
        "marginal_x_centers": x_centers_marg,
        "marginal_up_hist": up_hist_marg,
        "marginal_dn_hist": dn_hist_marg,
        "before_after_info": before_after_info,
        "healthy_walkers_fraction": float(np.mean(healthy_mask)),
        "example_event": example_event,
        "best_walker_trace": best_walker_trace,
        "best_walker_flips_t": best_walker_flips_t,
        "best_walker_chain": int(best_c),
        "best_walker_idx": int(best_b),
    }


# ==============================================================================
# 3. GUARDADO ATÓMICO DE CSV Y FIGURAS DE PUBLICACIÓN
# ==============================================================================

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


def _save_figure_bundle(fig: Any, out_dir: str, stem: str, args: argparse.Namespace) -> List[str]:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    if getattr(args, "no_titles", False):
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
        for ax in fig.axes:
            ax.set_title("")
    else:
        custom_sup = str(getattr(args, "custom_suptitle", "") or "").strip()
        t_prefix = str(getattr(args, "title_prefix", "") or "").strip()
        f_scale = float(getattr(args, "font_scale", 1.0))
        if custom_sup:
            full_sup = f"{t_prefix} — {custom_sup}" if t_prefix else custom_sup
            fig.suptitle(full_sup, fontsize=13.5 * f_scale, weight="bold", y=1.02)
        elif t_prefix and getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
            cur = fig._suptitle.get_text()
            if not cur.startswith(t_prefix):
                fig._suptitle.set_text(f"{t_prefix} — {cur}")

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


# ==============================================================================
# 4. GENERACIÓN DE GRÁFICAS POR SISTEMA (N, g), POR N VS g, Y GLOBAL
# ==============================================================================

def plot_per_system_flip_anatomy(
    records_ng: List[Dict[str, Any]],
    out_dir: str,
    args: argparse.Namespace,
    prod_root: Optional[str] = None,
    csv_collect_dir: Optional[str] = None,
) -> List[str]:
    """
    Genera para un sistema fijo (N, omega, g) todas las figuras rediseñadas y
    exporta las tablas CSV detalladas en `<csv_collect_dir>`:
      - Panel (a) [Arriba Izquierda]: Trazas de 1 caminante (MALA arriba, RL-MALA abajo)
        + histograma marginal de densidad por espín a un costado.
      - Panel (b) [Arriba Derecha]: Pasos necesarios para el flip Delta t_transit.
      - Panel (c) [Abajo Izquierda]: Mapa de calor de transición para MALA (rango y ampliado).
      - Panel (d) [Abajo Derecha]: Mapa de calor de transición para RL-MALA + promedio de
        |mu(t)| y sigma(t) del Actor en cada paso de la transición.
      - Figura adicional: Estado de los N electrones antes, durante y después de un flip,
        más cada panel individual en `<out_dir>/panels/`.
    """
    if not records_ng:
        return []

    from scripts.plot_flips_system_detailed import (
        export_system_flips_csvs,
        generate_all_system_flip_figures,
    )

    if csv_collect_dir:
        try:
            export_system_flips_csvs(records_ng, csv_dir=csv_collect_dir, prod_root=prod_root)
        except Exception as exc:
            print(f"  [WARN] No se pudieron exportar CSVs detallados de flips: {exc}", file=sys.stderr)

    return generate_all_system_flip_figures(records_ng, out_dir, args, prod_root=prod_root)


def _shade_physical_regimes(ax: Any, g_min: float, g_max: float, f_scale: float = 1.0, annotate: bool = False) -> None:
    """Sombrea el régimen de Hartree (g <= 0.3), crossover (0.3-0.5) y Wigner (g > 0.5)."""
    left = min(0.0, g_min)
    right = max(g_max * 1.04, 1.0)
    if left < 0.32:
        ax.axvspan(left, min(0.32, right), color="#2b8cbe", alpha=0.08, zorder=0)
    if right > 0.32:
        ax.axvspan(max(0.32, left), min(0.50, right), color="#ff7f0e", alpha=0.08, zorder=0)
    if right > 0.50:
        ax.axvspan(max(0.50, left), right, color="#d62728", alpha=0.05, zorder=0)
    ax.axvline(0.35, color="#555555", ls=":", lw=1.0, alpha=0.7)

    if annotate:
        ylim = ax.get_ylim()
        if ax.get_yscale() == "log":
            y0 = max(ylim[0], 1e-6)
            y1 = max(ylim[1], y0 * 10.0)
            y_top = y0 * ((y1 / y0) ** 0.84)
        else:
            y_top = ylim[0] + 0.86 * (ylim[1] - ylim[0])
        if g_min <= 0.35:
            ax.text(0.16, y_top, "Hartree\n($g \\lesssim 0.3$)", fontsize=8.0 * f_scale, color="#1b5e8a", ha="center", weight="bold")
        if g_max >= 0.6:
            ax.text(min(g_max * 0.62, 4.5), y_top, "Cristal de Wigner Localizado ($g > 0.4$)", fontsize=8.5 * f_scale, color="#8c1d1d", ha="center", weight="bold")


def _infer_n_same_pairs(row: Dict[str, Any], n_elec_fallback: int = 2) -> int:
    val = _safe_int(row.get("n_same_pairs"), -1)
    if val >= 0:
        return val
    n_tot = _safe_int(row.get("N"), n_elec_fallback)
    n_up = _safe_int(row.get("n_up"), n_tot // 2)
    n_dn = _safe_int(row.get("n_dn"), n_tot - n_up)
    return (n_up * (n_up - 1)) // 2 + (n_dn * (n_dn - 1)) // 2


def plot_flips_vs_g_for_N(
    rows_n: List[Dict[str, Any]],
    n_elec: int,
    pipeline: str,
    out_dir: str,
    args: argparse.Namespace,
) -> List[str]:
    """
    Genera para un N fijo la figura de 4 paneles de la evolución respecto a g:
      Panel A: Tasa de flips de espín opuesto (up-down) vs g (Histéresis vs Crudos, escala log).
      Panel B: Comparación de flips de espín opuesto (Wigner) vs mismo espín (nodos de Pauli) vs g.
      Panel C: Factor de aceleración de flips (RL-MALA / MALA) y divergencia D_KL^spin vs g.
      Panel D: Fracción de saltos en 1 paso (f_1step) y salto relativo en el cruce |Delta x_ij| vs g.
    """
    import matplotlib.pyplot as plt

    if not rows_n:
        return []

    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))

    methods = sorted(set(str(r["method"]) for r in rows_n), key=lambda m: {"MALA": 0, "Adaptive-MALA": 1, "RL-MALA": 2}.get(m, 99))
    all_g = sorted(set(_safe_float(r.get("g"), 0.0) for r in rows_n))
    g_min, g_max = min(all_g), max(all_g)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.0))
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    eps_log = 1e-3  # Piso visual para graficar 0 flips en escala logarítmica

    # --------------------------------------------------------------------------
    # Panel A: Flips up-down vs g (Histéresis vs Crudos)
    # --------------------------------------------------------------------------
    for m in methods:
        sub = sorted([r for r in rows_n if str(r["method"]) == m], key=lambda x: _safe_float(x.get("g"), 0.0))
        if not sub:
            continue
        g_arr = np.array([_safe_float(x.get("g"), 0.0) for x in sub])
        hyst_opp = np.array([max(_safe_float(x.get("hyst_flips_opp_mean"), 0.0), eps_log) for x in sub])
        hyst_err = np.array([max(_safe_float(x.get("hyst_flips_opp_std"), 0.0), 0.0) for x in sub])
        hyst_err = np.where(np.isfinite(hyst_err), hyst_err, 0.0)
        yerr_low = np.minimum(hyst_err, hyst_opp * 0.85)
        raw_opp = np.array([max(_safe_float(x.get("raw_flips_opp_mean"), 0.0), eps_log) for x in sub])

        col = METHOD_COLORS.get(m, "#333333")
        ls_h = "-" if m == "RL-MALA" else "--"
        mk_h = "o" if m == "RL-MALA" else "s"

        ax_a.errorbar(
            g_arr,
            hyst_opp,
            yerr=[yerr_low, hyst_err],
            color=col,
            ls=ls_h,
            lw=2.3 * lw_scale,
            marker=mk_h,
            markersize=6.5,
            capsize=3.5,
            label=f"{m} — Flips Reales Pozo-a-Pozo (Histéresis)",
        )
        ax_a.plot(
            g_arr,
            raw_opp,
            color=col,
            ls=":",
            lw=1.5 * lw_scale,
            marker="^",
            markersize=4.5,
            alpha=0.65,
            label=f"{m} — Cruces Crudos ($\\delta=0$, inc. jitter)",
        )

    ax_a.set_yscale("log")
    _shade_physical_regimes(ax_a, g_min, g_max, f_scale=f_scale, annotate=True)
    ax_a.set_title(f"(a) Tasa de Intercambios $\\uparrow\\downarrow$ vs $g$ ($N={n_elec}$)", fontsize=11.2 * f_scale, weight="bold")
    ax_a.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_a.set_ylabel("Flips $\\uparrow\\downarrow$ / $10^3$ pasos / caminante (log)", fontsize=10.0 * f_scale)
    ax_a.grid(True, which="both", alpha=0.26, linestyle="--")
    ax_a.legend(loc="best", fontsize=8.2 * f_scale, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel B: Flips de Espín Opuesto (Wigner) vs Mismo Espín (Nodos de Pauli)
    # --------------------------------------------------------------------------
    for m in methods:
        sub = sorted([r for r in rows_n if str(r["method"]) == m], key=lambda x: _safe_float(x.get("g"), 0.0))
        if not sub:
            continue
        g_arr = np.array([_safe_float(x.get("g"), 0.0) for x in sub])
        hyst_opp = np.array([max(_safe_float(x.get("hyst_flips_opp_mean"), 0.0), eps_log) for x in sub])
        hyst_same = np.array([max(_safe_float(x.get("hyst_flips_same_mean"), 0.0), eps_log) for x in sub])

        col = METHOD_COLORS.get(m, "#333333")
        ls_m = "-" if m == "RL-MALA" else "--"
        ax_b.plot(
            g_arr,
            hyst_opp,
            color=col,
            ls=ls_m,
            lw=2.2 * lw_scale,
            marker="o",
            markersize=6.2,
            label=f"{m}: Espín Opuesto $\\uparrow\\downarrow$ (Barrera Wigner)",
        )
        if _infer_n_same_pairs(sub[0], n_elec) > 0:
            ax_b.plot(
                g_arr,
                hyst_same,
                color=col,
                ls="-.",
                lw=1.6 * lw_scale,
                marker="D",
                markersize=5.0,
                alpha=0.78,
                label=f"{m}: Mismo Espín $\\uparrow\\uparrow / \\downarrow\\downarrow$ (Nodo Pauli $\\Psi=0$)",
            )

    ax_b.set_yscale("log")
    _shade_physical_regimes(ax_b, g_min, g_max, f_scale=f_scale, annotate=False)
    ax_b.set_title(f"(b) Desglose por Canal de Espín vs $g$ ($N={n_elec}$)", fontsize=11.2 * f_scale, weight="bold")
    ax_b.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_b.set_ylabel("Flips con histéresis / $10^3$ pasos / caminante", fontsize=10.0 * f_scale)
    ax_b.grid(True, which="both", alpha=0.26, linestyle="--")
    ax_b.legend(loc="best", fontsize=8.0 * f_scale, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel C: Correlación con D_KL^spin y Factor de Aceleración de Flips vs g
    # --------------------------------------------------------------------------
    ax_c_right = ax_c.twinx()
    for m in methods:
        sub = sorted([r for r in rows_n if str(r["method"]) == m], key=lambda x: _safe_float(x.get("g"), 0.0))
        if not sub:
            continue
        g_arr = np.array([_safe_float(x.get("g"), 0.0) for x in sub])
        kl_spin = np.array([_safe_float(x.get("kl_spin"), float("nan")) for x in sub])
        col = METHOD_COLORS.get(m, "#333333")
        ls_m = "-" if m == "RL-MALA" else "--"
        if np.any(np.isfinite(kl_spin)):
            ax_c.plot(
                g_arr,
                kl_spin,
                color=col,
                ls=ls_m,
                lw=2.2 * lw_scale,
                marker="o" if m == "RL-MALA" else "s",
                markersize=6.0,
                label=f"$D_{{\\mathrm{{KL}}}}^{{\\mathrm{{spin}}}}$ — {m}",
            )

    # Calcular el cociente de tasas de flips RL-MALA / MALA por cada g
    mala_by_g = {_safe_float(r.get("g"), 0.0): _safe_float(r.get("hyst_flips_total_mean"), 0.0) for r in rows_n if str(r["method"]) == "MALA"}
    rl_by_g = {_safe_float(r.get("g"), 0.0): _safe_float(r.get("hyst_flips_total_mean"), 0.0) for r in rows_n if str(r["method"]) == "RL-MALA"}
    common_g = sorted(set(mala_by_g.keys()) & set(rl_by_g.keys()))
    if common_g:
        ratios = [
            (rl_by_g[gg] + 1e-2) / (mala_by_g[gg] + 1e-2)
            for gg in common_g
        ]
        ax_c_right.plot(
            common_g,
            ratios,
            color="#2ca02c",
            ls=":",
            lw=2.1 * lw_scale,
            marker="^",
            markersize=6.5,
            label="Ganancia Topológica $R_{\\mathrm{flip}}^{\\mathrm{RL}} / R_{\\mathrm{flip}}^{\\mathrm{MALA}}$",
        )
        ax_c_right.axhline(1.0, color="#2ca02c", lw=0.9, ls="--", alpha=0.5)
        ax_c_right.set_ylabel("Razón de Flips $R_{\\mathrm{RL}} / R_{\\mathrm{MALA}}$", fontsize=9.8 * f_scale, color="#1f7a1f")
        ax_c_right.tick_params(axis="y", labelcolor="#1f7a1f")

    _shade_physical_regimes(ax_c, g_min, g_max, f_scale=f_scale, annotate=False)
    ax_c.set_title(f"(c) Bloqueo Ergódico ($D_{{\\mathrm{{KL}}}}^{{\\mathrm{{spin}}}}$) y Ganancia de Flips vs $g$", fontsize=11.2 * f_scale, weight="bold")
    ax_c.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_c.set_ylabel("Divergencia de Espín $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ vs CASSCF", fontsize=10.0 * f_scale)
    ax_c.grid(True, alpha=0.26, linestyle="--")

    lines_l, labels_l = ax_c.get_legend_handles_labels()
    lines_r, labels_r = ax_c_right.get_legend_handles_labels()
    if lines_l or lines_r:
        ax_c.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left", fontsize=8.2 * f_scale, framealpha=0.92)

    # --------------------------------------------------------------------------
    # Panel D: Fracción de Saltos en 1 Paso y Escala Espacial del Brinco vs g
    # --------------------------------------------------------------------------
    ax_d_right = ax_d.twinx()
    for m in methods:
        sub = sorted([r for r in rows_n if str(r["method"]) == m], key=lambda x: _safe_float(x.get("g"), 0.0))
        if not sub:
            continue
        g_arr = np.array([_safe_float(x.get("g"), 0.0) for x in sub])
        f1_raw = np.array([_safe_float(x.get("one_step_fraction"), float("nan")) * 100.0 for x in sub])
        f1_arr = np.where(np.isfinite(f1_raw), f1_raw, 0.0)
        jmp_arr = np.array([_safe_float(x.get("jump_at_flip_mean"), float("nan")) for x in sub])

        col = METHOD_COLORS.get(m, "#333333")
        ls_m = "-" if m == "RL-MALA" else "--"
        ax_d.plot(
            g_arr,
            f1_arr,
            color=col,
            ls=ls_m,
            lw=2.2 * lw_scale,
            marker="o" if m == "RL-MALA" else "s",
            markersize=6.0,
            label=f"{m}: % Cruces en 1 Paso ($\\Delta t_{{\\mathrm{{transit}}}}=1$)",
        )
        if np.any(np.isfinite(jmp_arr)):
            ax_d_right.plot(
                g_arr,
                jmp_arr,
                color=col,
                ls=":",
                lw=1.8 * lw_scale,
                marker="^",
                markersize=5.2,
                alpha=0.82,
                label=f"{m}: Salto en el cruce $\\langle |\\Delta x_{{ij}}|\\rangle$ [Bohr]",
            )

    # Graficar también la separación entre pozos d_adj(g) como referencia física
    sub_ref = sorted(rows_n, key=lambda x: _safe_float(x.get("g"), 0.0))
    g_unique = sorted(set(_safe_float(x.get("g"), 0.0) for x in sub_ref))
    d_adj_by_g = [float(np.nanmean([_safe_float(x.get("d_adj"), float("nan")) for x in sub_ref if _safe_float(x.get("g"), 0.0) == gg])) for gg in g_unique]
    if np.any(np.isfinite(d_adj_by_g)):
        ax_d_right.plot(
            g_unique,
            d_adj_by_g,
            color="#2ca02c",
            ls="-.",
            lw=1.6 * lw_scale,
            label="Separación entre pozos $d_{\\mathrm{adj}}(g)$ [Bohr]",
        )

    ax_d.set_ylim(-5.0, 105.0)
    ax_d.set_title(f"(d) Fracción de Saltos en 1 Paso y Escala del Brinco vs $g$", fontsize=11.2 * f_scale, weight="bold")
    ax_d.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_d.set_ylabel("Cruces de barrera en 1 solo paso [%]", fontsize=10.0 * f_scale)
    ax_d_right.set_ylabel("Distancia en el cruce [Bohr]", fontsize=9.8 * f_scale, color="#444444")
    ax_d.grid(True, alpha=0.26, linestyle="--")

    dl_l, dlab_l = ax_d.get_legend_handles_labels()
    dl_r, dlab_r = ax_d_right.get_legend_handles_labels()
    if dl_l or dl_r:
        ax_d.legend(dl_l + dl_r, dlab_l + dlab_r, loc="best", fontsize=7.8 * f_scale, framealpha=0.92)

    fig.suptitle(
        f"Evolución del Número de Flips y Mecanismo de Cruce respecto a $g$ — $N={n_elec}$ Electrones ({pipeline})",
        fontsize=13.5 * f_scale,
        weight="bold",
        y=0.995,
    )

    stem = f"fig_flips_vs_g_{pipeline}_N{n_elec}"
    return _save_figure_bundle(fig, out_dir, stem, args)


def plot_master_all_N_flips_vs_g(
    all_rows: List[Dict[str, Any]],
    pipeline: str,
    out_dir: str,
    args: argparse.Namespace,
) -> List[str]:
    """
    Genera la figura panorámica maestra comparando todos los tamaños de sistema
    N in {2, 4, 6, 10} a lo largo de g para MALA (líneas discontinuas) vs
    RL-MALA (líneas continuas).
    """
    import matplotlib.pyplot as plt

    rows_p = [r for r in all_rows if str(r.get("pipeline", "P2")) == pipeline]
    if not rows_p:
        return []

    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))

    n_values = sorted(set(_safe_int(r.get("N"), 0) for r in rows_p if _safe_int(r.get("N"), 0) > 0))
    all_g = sorted(set(_safe_float(r.get("g"), 0.0) for r in rows_p))
    g_min, g_max = min(all_g), max(all_g)
    eps_log = 1e-3

    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.2))
    ax_hyst, ax_raw = axes[0, 0], axes[0, 1]
    ax_ratio, ax_kl = axes[1, 0], axes[1, 1]

    for n_val in n_values:
        col_n = N_COLORS.get(n_val, "#333333")
        mk_n = N_MARKERS.get(n_val, "o")
        rows_n = [r for r in rows_p if _safe_int(r.get("N"), 0) == n_val]

        for m_name in ("MALA", "RL-MALA"):
            sub = sorted([r for r in rows_n if str(r.get("method")) == m_name], key=lambda x: _safe_float(x.get("g"), 0.0))
            if not sub:
                continue
            g_arr = np.array([_safe_float(x.get("g"), 0.0) for x in sub])
            hyst_tot = np.array([max(_safe_float(x.get("hyst_flips_total_mean"), 0.0), eps_log) for x in sub])
            raw_tot = np.array([max(_safe_float(x.get("raw_flips_total_mean"), 0.0), eps_log) for x in sub])
            kl_sp = np.array([_safe_float(x.get("kl_spin"), float("nan")) for x in sub])

            ls = "-" if m_name == "RL-MALA" else "--"
            mfc = col_n if m_name == "RL-MALA" else "white"
            alpha = 0.95 if m_name == "RL-MALA" else 0.78

            ax_hyst.plot(
                g_arr,
                hyst_tot,
                color=col_n,
                ls=ls,
                lw=2.1 * lw_scale,
                marker=mk_n,
                markersize=6.0,
                markerfacecolor=mfc,
                alpha=alpha,
                label=f"$N={n_val}$ ({m_name})",
            )
            ax_raw.plot(
                g_arr,
                raw_tot,
                color=col_n,
                ls=ls,
                lw=2.0 * lw_scale,
                marker=mk_n,
                markersize=5.8,
                markerfacecolor=mfc,
                alpha=alpha,
                label=f"$N={n_val}$ ({m_name})",
            )
            if np.any(np.isfinite(kl_sp)):
                ax_kl.plot(
                    g_arr,
                    kl_sp,
                    color=col_n,
                    ls=ls,
                    lw=2.0 * lw_scale,
                    marker=mk_n,
                    markersize=5.8,
                    markerfacecolor=mfc,
                    alpha=alpha,
                    label=f"$N={n_val}$ ({m_name})",
                )

        # Razón RL-MALA / MALA para este N
        mala_g = {_safe_float(r.get("g"), 0.0): _safe_float(r.get("hyst_flips_total_mean"), 0.0) for r in rows_n if str(r.get("method")) == "MALA"}
        rl_g = {_safe_float(r.get("g"), 0.0): _safe_float(r.get("hyst_flips_total_mean"), 0.0) for r in rows_n if str(r.get("method")) == "RL-MALA"}
        cg = sorted(set(mala_g.keys()) & set(rl_g.keys()))
        if cg:
            rat = [(rl_g[gg] + 1e-2) / (mala_g[gg] + 1e-2) for gg in cg]
            ax_ratio.plot(
                cg,
                rat,
                color=col_n,
                ls="-",
                lw=2.2 * lw_scale,
                marker=mk_n,
                markersize=6.2,
                label=f"$N={n_val}$: $R_{{\\mathrm{{RL}}}} / R_{{\\mathrm{{MALA}}}}$",
            )

    ax_hyst.set_yscale("log")
    _shade_physical_regimes(ax_hyst, g_min, g_max, f_scale=f_scale, annotate=True)
    ax_hyst.set_title("(a) Flips Reales Pozo-a-Pozo (Con Histéresis) vs $g$", fontsize=11.2 * f_scale, weight="bold")
    ax_hyst.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_hyst.set_ylabel("Flips / $10^3$ pasos / caminante (log)", fontsize=10.0 * f_scale)
    ax_hyst.grid(True, which="both", alpha=0.26, linestyle="--")
    ax_hyst.legend(loc="best", ncol=2, fontsize=7.8 * f_scale, framealpha=0.92)

    ax_raw.set_yscale("log")
    _shade_physical_regimes(ax_raw, g_min, g_max, f_scale=f_scale, annotate=False)
    ax_raw.set_title("(b) Cruces Crudos ($\\delta=0$, incluye traslape térmico en $g < 0.3$) vs $g$", fontsize=11.2 * f_scale, weight="bold")
    ax_raw.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_raw.set_ylabel("Cruces crudos / $10^3$ pasos / caminante (log)", fontsize=10.0 * f_scale)
    ax_raw.grid(True, which="both", alpha=0.26, linestyle="--")
    ax_raw.legend(loc="best", ncol=2, fontsize=7.8 * f_scale, framealpha=0.92)

    ax_ratio.axhline(1.0, color="black", ls="--", lw=1.0, alpha=0.6)
    _shade_physical_regimes(ax_ratio, g_min, g_max, f_scale=f_scale, annotate=False)
    ax_ratio.set_title("(c) Factor de Aceleración de Flips ($R_{\\mathrm{RL-MALA}} / R_{\\mathrm{MALA}}$) vs $g$", fontsize=11.2 * f_scale, weight="bold")
    ax_ratio.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_ratio.set_ylabel("Ganancia en tasa de intercambios ($\\times$)", fontsize=10.0 * f_scale)
    ax_ratio.grid(True, alpha=0.26, linestyle="--")
    ax_ratio.legend(loc="best", fontsize=8.5 * f_scale, framealpha=0.92)

    _shade_physical_regimes(ax_kl, g_min, g_max, f_scale=f_scale, annotate=False)
    ax_kl.set_title("(d) Divergencia de Espín $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ vs $g$ (Congelamiento Topológico)", fontsize=11.2 * f_scale, weight="bold")
    ax_kl.set_xlabel("Acoplamiento de interacción $g$", fontsize=10.0 * f_scale)
    ax_kl.set_ylabel("$D_{\\mathrm{KL}}^{\\mathrm{spin}}$ respecto a CASSCF exacto", fontsize=10.0 * f_scale)
    ax_kl.grid(True, alpha=0.26, linestyle="--")
    handles_kl, labels_kl = ax_kl.get_legend_handles_labels()
    if handles_kl:
        ax_kl.legend(loc="best", ncol=2, fontsize=7.8 * f_scale, framealpha=0.92)

    fig.suptitle(
        f"Síntesis Global de Intercambios Topológicos (Flips) respecto a $N$ y $g$ ({pipeline}: MALA discontinuo vs RL-MALA continuo)",
        fontsize=13.5 * f_scale,
        weight="bold",
        y=0.995,
    )

    stem = f"fig_flips_vs_g_all_N_master_{pipeline}"
    return _save_figure_bundle(fig, out_dir, stem, args)


# ==============================================================================
# 5. PUNTO DE ENTRADA CLI
# ==============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analiza y grafica el número de flips (intercambios de electrones) vs g y N a partir de la carpeta raíz de run_production_benchmark_3pipelines.sh."
    )
    p.add_argument(
        "root_dir",
        nargs="?",
        default="results_production_3pipelines_baker",
        help="Carpeta raíz del benchmark (ej. results_production_3pipelines_baker) o directorio con run_bundle.h5.",
    )
    p.add_argument("--root_dir", "--base_dir", dest="root_dir_flag", default=None, help="Alias con bandera para la carpeta raíz.")
    p.add_argument("--out_dir", default=None, help="Directorio de salida para las figuras y CSV (por defecto: <root_dir>/figures/flips_analysis).")
    p.add_argument("--hysteresis_frac", type=float, default=0.25, help="Fracción de la distancia media entre vecinos d_adj usada como umbral de histéresis delta (default: 0.25).")
    p.add_argument("--fixed_delta", type=float, default=None, help="Umbral fijo delta [Bohr] opcional en lugar de hysteresis_frac * d_adj.")
    p.add_argument("--include_pipeline3", action="store_true", help="Analizar también Pipeline 3 (entrenamiento variacional SJB) además de Pipeline 2.")
    p.add_argument("--skip_per_system_plots", action="store_true", help="Omitir las figuras individuales de cada par (N, g) y generar sólo las curvas vs g.")
    p.add_argument(
        "--only_summary_plots",
        "--from_csv",
        dest="only_summary_plots",
        action="store_true",
        help="Si ya existe flips_vs_g_summary.csv en <out_dir>, reutilizarlo directamente para generar las gráficas generales vs g sin re-leer los HDF5.",
    )
    p.add_argument(
        "--n_filter",
        type=int,
        nargs="+",
        default=None,
        help="Filtrar únicamente sistemas con N en esta lista (ej. --n_filter 2 4 6 8).",
    )
    p.add_argument(
        "--g_filter",
        type=float,
        nargs="+",
        default=None,
        help="Filtrar únicamente sistemas con g en esta lista (ej. --g_filter 2.0 3.0 5.0).",
    )
    p.add_argument(
        "--systems",
        "--system_filter",
        dest="systems_filter",
        nargs="+",
        default=None,
        help=(
            "Filtro selectivo por sistema exacto o por par N:g "
            "(ej. --systems N2:all N8:3.0,5.0 N10:all o N8_w0.1_g3.0)."
        ),
    )
    p.add_argument(
        "--skip_summary_plots",
        action="store_true",
        help="Omitir gráficas agregadas (per_N y maestra global); útil al actualizar sólo sistemas individuales.",
    )
    p.add_argument(
        "--merge_summary_csv",
        action="store_true",
        help="Combinar/actualizar filas filtradas dentro del flips_vs_g_summary.csv existente en lugar de sobrescribirlo.",
    )
    p.add_argument("--figure_format", choices=["png", "pdf", "both"], default="png", help="Formato de exportación de figuras.")
    p.add_argument("--dpi", type=int, default=300, help="Resolución en PPI/DPI para PNG.")
    p.add_argument("--font_scale", type=float, default=1.0, help="Factor de escala para tamaños de fuente.")
    p.add_argument("--line_width_scale", type=float, default=1.0, help="Factor de escala para grosores de línea.")
    p.add_argument("--title_prefix", type=str, default="", help="Prefijo opcional para títulos.")
    p.add_argument("--custom_suptitle", type=str, default="", help="Supertítulo personalizado opcional.")
    p.add_argument("--custom_subtitle", type=str, default="", help="Subtítulo personalizado opcional.")
    p.add_argument("--no_titles", action="store_true", help="Ocultar títulos en las figuras (útil para insertar en LaTeX).")
    p.add_argument("--no_suptitle", action="store_true", help="Ocultar sólo el supertítulo general.")
    p.add_argument("--no_panel_titles", action="store_true", help="Ocultar los títulos/subtítulos individuales de cada panel.")
    p.add_argument("--no_legends", action="store_true", help="Ocultar todas las leyendas.")
    p.add_argument("--save_panels", action=argparse.BooleanOptionalAction, default=True, help="Guardar cada panel individual por separado en <out_dir>/per_system/panels/.")
    p.add_argument("--heatmap_ymax", type=float, default=2.5, help="Rango vertical [-ymax, +ymax] en los mapas de calor de transición.")
    p.add_argument("--before_after_dt", type=int, default=5, help="Pasos relativos antes y después del flip para el diagrama de estado 1D.")
    p.add_argument("--show_actor_on_heatmap", action=argparse.BooleanOptionalAction, default=True, help="Superponer promedio de |mu| y sigma en el mapa de calor de RL-MALA.")
    p.add_argument("--transparent_bg", action="store_true", help="Exportar figuras con fondo transparente.")
    return p


def _parse_systems_filter(tokens: Optional[List[str]]) -> Optional[Dict[int, Optional[List[float]]]]:
    """Convierte especificaciones como ['N2:all', 'N8:3.0,5.0', 'N10_w0.1_g5.0'] en {N: [gs] | None}."""
    if not tokens:
        return None
    mapping: Dict[int, Optional[List[float]]] = {}
    for raw in tokens:
        for tok in re.split(r"[\s;]+", str(raw).strip()):
            if not tok:
                continue
            m_tag = re.match(r"^N(\d+)_w[0-9.]+_g([0-9.]+)$", tok, flags=re.IGNORECASE)
            if m_tag:
                n_v = int(m_tag.group(1))
                g_v = float(m_tag.group(2))
                cur = mapping.get(n_v, [])
                if cur is not None:
                    mapping[n_v] = [*cur, g_v]
                continue
            if ":" in tok:
                n_part, g_part = tok.split(":", 1)
                n_v = int(re.sub(r"^[Nn]", "", n_part.strip()))
                if g_part.strip().lower() in ("all", "*"):
                    mapping[n_v] = None
                else:
                    gs = [float(x) for x in g_part.split(",") if x.strip()]
                    if n_v in mapping and mapping[n_v] is None:
                        continue
                    cur = mapping.get(n_v, []) or []
                    mapping[n_v] = [*cur, *gs]
            else:
                n_v = int(re.sub(r"^[Nn]", "", tok.strip()))
                mapping[n_v] = None
    return mapping or None


def _matches_ng_filter(
    n_val: int,
    g_val: float,
    n_filter: Optional[List[int]],
    g_filter: Optional[List[float]],
    systems_map: Optional[Dict[int, Optional[List[float]]]],
) -> bool:
    if n_filter is not None and int(n_val) not in set(int(x) for x in n_filter):
        return False
    if g_filter is not None and not any(abs(float(g_val) - float(gf)) <= 1e-4 for gf in g_filter):
        return False
    if systems_map is not None:
        if int(n_val) not in systems_map:
            return False
        g_allowed = systems_map[int(n_val)]
        if g_allowed is not None and not any(abs(float(g_val) - float(ga)) <= 1e-4 for ga in g_allowed):
            return False
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    systems_map = _parse_systems_filter(args.systems_filter)

    root_dir = os.path.abspath(args.root_dir_flag if args.root_dir_flag else args.root_dir)
    if not os.path.exists(root_dir):
        print(f"[ERROR] No existe la ruta especificada: {root_dir}", file=sys.stderr)
        return 1

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(root_dir, "figures", "flips_analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "flips_vs_g_summary.csv")

    print("==============================================================================")
    print(" ANÁLISIS DE INTERCAMBIOS TOPOLÓGICOS ('FLIPS') RESPECTO A N Y g")
    print(f" Carpeta raíz de entrada : {root_dir}")
    print(f" Carpeta de salida       : {out_dir}")
    print(f" Histéresis (eta * d_adj): eta = {args.hysteresis_frac:.2f}")
    if args.n_filter or args.g_filter or systems_map:
        print(f" Filtros activos         : n={args.n_filter} | g={args.g_filter} | systems={args.systems_filter}")
    print("==============================================================================")

    records: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    if args.only_summary_plots and os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
        print(f"[+] Modo --only_summary_plots: cargando tabla existente {out_csv}")
        with open(out_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                n_r = _safe_int(r.get("N"), 0)
                g_r = _safe_float(r.get("g"), float("nan"))
                if not _matches_ng_filter(n_r, g_r, args.n_filter, args.g_filter, systems_map):
                    continue
                summary_rows.append(dict(r))
    else:
        h5_paths = discover_h5_bundles(root_dir, include_pipeline3=args.include_pipeline3)
        if not h5_paths:
            print(f"[ERROR] No se encontraron archivos run_bundle*.h5 dentro de: {root_dir}", file=sys.stderr)
            return 2

        # Pre-filtrar por ruta para evitar leer tensores pesados de sistemas excluidos
        if args.n_filter or args.g_filter or systems_map:
            filtered_paths = []
            for p_cand in h5_paths:
                n_p, _, g_p = _parse_n_omega_g_from_text(p_cand)
                if n_p is not None and g_p is not None:
                    if not _matches_ng_filter(n_p, g_p, args.n_filter, args.g_filter, systems_map):
                        continue
                filtered_paths.append(p_cand)
            h5_paths = filtered_paths

        print(f"[+] Se descubrieron {len(h5_paths)} bundles HDF5 para analizar.")
        csv_lookup = _load_csv_lookup_tables(root_dir)

        for idx, h5p in enumerate(h5_paths, start=1):
            try:
                bundle_info = load_bundle_walkers_and_meta(h5p)
                if not _matches_ng_filter(
                    int(bundle_info["N"]),
                    float(bundle_info["g"]),
                    args.n_filter,
                    args.g_filter,
                    systems_map,
                ):
                    del bundle_info["walkers"]
                    continue
                stats = compute_flip_statistics(
                    bundle_info["walkers"],
                    n_up=bundle_info["n_up"],
                    hysteresis_frac=args.hysteresis_frac,
                    fixed_delta=args.fixed_delta,
                    omega=bundle_info["omega"],
                    g_val=bundle_info["g"],
                    tau=bundle_info["tau"],
                    actor_amp=bundle_info["actor_amplitude"],
                    sigma_scale=bundle_info["sigma_action_scale"],
                    actor_params=bundle_info.get("actor_params"),
                    is_neural=(bundle_info["method"] == "RL-MALA"),
                )
                # Liberar tensor pesado de caminantes tras extraer estadísticas y trazas
                del bundle_info["walkers"]

                key = (
                    bundle_info["pipeline"],
                    int(bundle_info["N"]),
                    round(float(bundle_info["g"]), 6),
                    bundle_info["method"],
                )
                extra = csv_lookup.get(key, {})
                for k_ex in ("acceptance", "ESJD", "tau_int_E", "tau_int_pos", "E_mean", "SE_E", "kl_total", "kl_up", "kl_down", "kl_spin", "G_RL_speedup"):
                    cur_val = bundle_info.get(k_ex, float("nan"))
                    if (cur_val is None or not np.isfinite(cur_val)) and k_ex in extra:
                        bundle_info[k_ex] = extra[k_ex]

                if not np.isfinite(bundle_info.get("kl_spin", float("nan"))):
                    ku = bundle_info.get("kl_up", float("nan"))
                    kd = bundle_info.get("kl_down", float("nan"))
                    if np.isfinite(ku) and np.isfinite(kd):
                        bundle_info["kl_spin"] = 0.5 * (ku + kd)
                    else:
                        bundle_info["kl_spin"] = bundle_info.get("kl_total", float("nan"))

                if not np.isfinite(bundle_info.get("ESJD", float("nan"))):
                    bundle_info["ESJD"] = stats["empirical_esjd"]

                bundle_info["stats"] = stats
                records.append(bundle_info)

                row_csv = {
                    "pipeline": bundle_info["pipeline"],
                    "N": bundle_info["N"],
                    "n_up": bundle_info["n_up"],
                    "n_dn": bundle_info["n_dn"],
                    "omega": bundle_info["omega"],
                    "g": bundle_info["g"],
                    "method": bundle_info["method"],
                    "run_name": bundle_info["run_name"],
                    "tau": bundle_info["tau"],
                    "actor_amplitude": bundle_info["actor_amplitude"],
                    "sigma_action_scale": bundle_info["sigma_action_scale"],
                    "n_chains": bundle_info["n_chains"],
                    "n_steps": bundle_info["n_steps"],
                    "n_batch_saved": bundle_info["n_batch_saved"],
                    "n_pairs_total": stats["n_pairs_total"],
                    "n_opp_pairs": stats["n_opp_pairs"],
                    "n_same_pairs": stats["n_same_pairs"],
                    "d_adj": stats["d_adj"],
                    "delta_hyst": stats["delta_hyst"],
                    "hyst_flips_total_mean": stats["hyst_flips_total_mean"],
                    "hyst_flips_total_std": stats["hyst_flips_total_std"],
                    "hyst_flips_opp_mean": stats["hyst_flips_opp_mean"],
                    "hyst_flips_opp_std": stats["hyst_flips_opp_std"],
                    "hyst_flips_same_mean": stats["hyst_flips_same_mean"],
                    "hyst_flips_same_std": stats["hyst_flips_same_std"],
                    "raw_flips_total_mean": stats["raw_flips_total_mean"],
                    "raw_flips_total_std": stats["raw_flips_total_std"],
                    "raw_flips_opp_mean": stats["raw_flips_opp_mean"],
                    "raw_flips_opp_std": stats["raw_flips_opp_std"],
                    "raw_flips_same_mean": stats["raw_flips_same_mean"],
                    "raw_flips_same_std": stats["raw_flips_same_std"],
                    "origin_flips_per_1k": stats["origin_flips_per_1k"],
                    "one_step_fraction": stats["one_step_fraction"],
                    "two_step_fraction": stats["two_step_fraction"],
                    "transit_steps_mean": stats["transit_steps_mean"],
                    "transit_steps_median": stats["transit_steps_median"],
                    "jump_at_flip_mean": stats["jump_at_flip_mean"],
                    "normal_pair_jump_mean": stats["normal_pair_jump_mean"],
                    "acceptance": bundle_info.get("acceptance"),
                    "ESJD": bundle_info.get("ESJD"),
                    "kl_spin": bundle_info.get("kl_spin"),
                    "kl_total": bundle_info.get("kl_total"),
                    "tau_int_E": bundle_info.get("tau_int_E"),
                    "h5_path": h5p,
                }
                summary_rows.append(row_csv)

                print(
                    f"  [{idx:02d}/{len(h5_paths):02d}] {bundle_info['pipeline']} | N={bundle_info['N']:>2d} g={bundle_info['g']:<4g} | "
                    f"{bundle_info['method']:<13s} | Flips(hyst)={stats['hyst_flips_total_mean']:7.2f}/1k "
                    f"(opp={stats['hyst_flips_opp_mean']:6.2f}, same={stats['hyst_flips_same_mean']:6.2f}) | "
                    f"Flips(raw)={stats['raw_flips_total_mean']:7.2f}/1k | "
                    f"1-paso={stats['one_step_fraction']*100.0 if np.isfinite(stats['one_step_fraction']) else 0.0:5.1f}%"
                )
            except Exception as exc:
                print(f"  [WARN] No se pudo procesar {h5p}: {exc}", file=sys.stderr)

        if not summary_rows:
            print("[ERROR] Ningún archivo HDF5 pudo ser procesado exitosamente.", file=sys.stderr)
            return 3

        if args.merge_summary_csv and os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0:
            merged_map: Dict[Tuple[str, int, float, str], Dict[str, Any]] = {}
            with open(out_csv, "r", encoding="utf-8") as f_old:
                for old_r in csv.DictReader(f_old):
                    k_old = (
                        str(old_r.get("pipeline", "P2")),
                        _safe_int(old_r.get("N"), 0),
                        round(_safe_float(old_r.get("g"), 0.0), 6),
                        str(old_r.get("method", "")),
                    )
                    merged_map[k_old] = dict(old_r)
            for new_r in summary_rows:
                k_new = (
                    str(new_r.get("pipeline", "P2")),
                    _safe_int(new_r.get("N"), 0),
                    round(_safe_float(new_r.get("g"), 0.0), 6),
                    str(new_r.get("method", "")),
                )
                merged_map[k_new] = new_r
            summary_rows = list(merged_map.values())

        # Ordenar filas por pipeline, N, g, method
        summary_rows.sort(key=lambda r: (str(r["pipeline"]), _safe_int(r.get("N"), 0), _safe_float(r.get("g"), 0.0), str(r["method"])))
        fieldnames = list(summary_rows[0].keys())
        _atomic_write_csv(summary_rows, fieldnames, out_csv)
        print(f"\n[+] Tabla maestra de flips exportada en: {out_csv}")

        # Copiar también a <root_dir>/results_csv si existe esa carpeta
        csv_collect_dir = os.path.join(root_dir, "results_csv")
        if os.path.isdir(csv_collect_dir):
            _atomic_write_csv(summary_rows, fieldnames, os.path.join(csv_collect_dir, "flips_vs_g_summary.csv"))

    # Verificar disponibilidad de matplotlib antes de graficar
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        print(f"[WARN] matplotlib no está disponible en este entorno ({exc}). Se generó únicamente el CSV.", file=sys.stderr)
        return 0

    # 1. Graficar anatomía detallada por cada sistema (pipeline, N, g)
    if records and not args.skip_per_system_plots and not args.only_summary_plots:
        per_sys_dir = os.path.join(out_dir, "per_system")
        grouped_ng: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = {}
        for r in records:
            key_ng = (str(r["pipeline"]), int(r["N"]), round(float(r["g"]), 6))
            grouped_ng.setdefault(key_ng, []).append(r)

        csv_sys_collect_dir = os.path.join(root_dir, "results_csv")
        os.makedirs(csv_sys_collect_dir, exist_ok=True)
        for (pipe, n_val, g_val), recs_ng in sorted(grouped_ng.items()):
            saved_sys_figs = plot_per_system_flip_anatomy(
                recs_ng,
                per_sys_dir,
                args,
                prod_root=root_dir,
                csv_collect_dir=csv_sys_collect_dir,
            )
            # Si existe la carpeta de figuras del sistema en <root_dir>/figures/pipeline2_systems/N{N}_w{w}_g{g}, copiar ahí también
            omega_val = float(recs_ng[0]["omega"])
            sys_tag = f"N{n_val}_w{omega_val:g}_g{g_val:g}"
            dest_sys_fig_dir = os.path.join(
                root_dir,
                "figures",
                "pipeline2_systems" if pipe == "P2" else "pipeline3_systems",
                sys_tag,
            )
            if os.path.isdir(dest_sys_fig_dir):
                for sp in saved_sys_figs:
                    try:
                        import shutil
                        shutil.copy2(sp, os.path.join(dest_sys_fig_dir, os.path.basename(sp)))
                    except Exception:
                        pass

    if not args.skip_summary_plots:
        # 2. Graficar evolución vs g para cada N
        per_n_dir = os.path.join(out_dir, "per_N")
        pipelines_present = sorted(set(str(r.get("pipeline", "P2")) for r in summary_rows))
        for pipe in pipelines_present:
            n_present = sorted(set(_safe_int(r.get("N"), 0) for r in summary_rows if str(r.get("pipeline", "P2")) == pipe and _safe_int(r.get("N"), 0) > 0))
            for n_val in n_present:
                rows_n = [r for r in summary_rows if str(r.get("pipeline", "P2")) == pipe and _safe_int(r.get("N"), 0) == n_val]
                plot_flips_vs_g_for_N(rows_n, n_val, pipe, per_n_dir, args)

            # 3. Figura maestra con todos los N vs g
            saved_master = plot_master_all_N_flips_vs_g(summary_rows, pipe, out_dir, args)
            all_models_fig_dir = os.path.join(root_dir, "figures", "all_models_vs_N_and_g")
            if os.path.isdir(all_models_fig_dir):
                for sp in saved_master:
                    try:
                        import shutil
                        shutil.copy2(sp, os.path.join(all_models_fig_dir, os.path.basename(sp)))
                    except Exception:
                        pass

    print("==============================================================================")
    print(" ANÁLISIS DE FLIPS FINALIZADO EXITOSAMENTE")
    print(f" Resumen CSV : {out_csv}")
    print(f" Figuras en  : {out_dir}")
    print("==============================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
