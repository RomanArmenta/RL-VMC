#!/usr/bin/env python3
r"""
Auditoría Exhaustiva y Post-Procesamiento Masivo de Modelos en Carpetas `results*`
==================================================================================

Este script escanea recursivamente todas las carpetas cuyo nombre comience por `results*`
(generadas por `1D_queue.sh`, `run_experiments_massive*.sh`, `run_sweeps*.sh`,
`benchmark_parameter_search*.sh`, `run_massive_experiments*.sh`, `run_evaluar_wigner_casscf.sh`
o `cli/run_1d.py`), convierte automáticamente los modelos legacy (`params.json` + `*.pkl` o
corridas CASSCF) al formato unificado HDF5, ejecuta múltiples muestreos MCMC independientes
(`--n-eval-runs`, por defecto 5 semillas distintas por modelo) para obtener estadística
rigurosa con barras de error ($\mu \pm \sigma$ y $\mathrm{SEM}$), calcula observables físicos
(densidades 1D por espín, distribución de pares, cristalización de Wigner, ACF e IAT
de Sokal) y:

1. Si el sistema físico $(N_\uparrow, N_\downarrow, \omega, g)$ cuenta con una baseline CASSCF
   pre-entrenada en la carpeta `uhf*` (`casscf_state.npz`), calcula su energía variacional
   exacta $E_{\mathrm{CASSCF}}$ y sus densidades analíticas exactas de 1 cuerpo
   ($\rho_{\mathrm{tot}}(x), \rho_\uparrow(x), \rho_\downarrow(x)$), evalúa la divergencia de
   Kullback-Leibler $D_{\mathrm{KL}}(P_{\mathrm{sampled}} \parallel P_{\mathrm{CASSCF}})$ en cada
   corrida independiente y superpone la baseline CASSCF en las gráficas de auditoría
   (siguiendo el formato de `scripts/evaluar_wigner_casscf_2.py`).
2. Si NO existe baseline CASSCF para un sistema físico dado, continúa sin error y grafica
   únicamente los resultados estadísticos de la evaluación.
"""
from __future__ import annotations

import os

os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rl_vmc.config import Experiment1DConfig, PostProcessConfig
from rl_vmc.core.io_bundle import (
    convert_legacy_dir_to_h5,
    load_run_bundle_h5,
    save_run_bundle_h5,
)
from rl_vmc.physics.harmonic_1d import Harmonic1DHamiltonian
from rl_vmc.ansatze.ansatz_1d import (
    build_ansatz_1d,
    _infer_orbital_model_from_params,
    _infer_correlation_model_from_params,
)
from rl_vmc.proposals.proposals_1d import build_proposal_1d
from rl_vmc.samplers.mh_sampler_1d import MHSampler1D
from rl_vmc.postprocess.diagnostics import compute_acf_1d, compute_iat_sokal
from rl_vmc.postprocess.observables import compute_1d_observables
from rl_vmc.postprocess.casscf_baseline import (
    discover_casscf_baselines,
    find_matching_casscf_baseline,
    compute_exact_casscf_1body_density,
    compute_spin_kl_divergences,
)
from rl_vmc.postprocess.plotter import run_postprocessing_pipeline


def build_audit_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audit_results",
        description=(
            "Escanea todas las carpetas `results*`, convierte modelos legacy a HDF5, "
            "ejecuta múltiples muestreos MCMC significativos por modelo, compara contra "
            "baselines CASSCF pre-entrenadas en `uhf*` cuando existen y genera la tabla "
            "maestra de estadística y gráficas de post-producción."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--search-root",
        type=str,
        default=".",
        help="Directorio raíz donde buscar carpetas que coincidan con `--pattern`.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="results*",
        help="Patrón glob de las carpetas raíz de resultados a explorar.",
    )
    p.add_argument(
        "--uhf-root",
        type=str,
        default=".",
        help="Directorio raíz donde buscar carpetas `uhf*` con baselines CASSCF (`casscf_state.npz`).",
    )
    p.add_argument(
        "--uhf-pattern",
        type=str,
        default="uhf*",
        help="Patrón glob para localizar carpetas con modelos UHF/CASSCF pre-entrenados.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results_exhaustive_audit",
        help="Directorio de salida donde se guardarán la tabla maestra y las gráficas.",
    )
    p.add_argument(
        "--n-eval-runs",
        type=int,
        default=5,
        help="Número de corridas MCMC independientes (distintas semillas) por cada modelo para rigor estadístico.",
    )
    p.add_argument(
        "--n-burn-in",
        type=int,
        default=1000,
        help="Pasos de termalización (burn-in) descartados al inicio de cada corrida MCMC.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=5000,
        help="Pasos de producción MCMC por cada corrida de evaluación.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Número de caminantes en paralelo (0 = usar el batch_size original del modelo acotado a 512).",
    )
    p.add_argument(
        "--max-walkers-save",
        type=int,
        default=64,
        help="Número máximo de caminantes cuyas trayectorias se guardan en el HDF5 auditado.",
    )
    p.add_argument(
        "--sokal-c",
        type=float,
        default=5.0,
        help="Constante de ventana adaptativa de Sokal para el cálculo de IAT.",
    )
    p.add_argument(
        "--max-lag",
        type=int,
        default=500,
        help="Lag máximo para la función de autocorrelación (ACF).",
    )
    p.add_argument(
        "--density-bins",
        type=int,
        default=160,
        help="Número de bines para los histogramas de densidad 1D y correlación de pares.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Semilla maestra para las corridas independientes de evaluación.",
    )
    p.add_argument(
        "--generate-system-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generar las figuras comparativas de post-producción (incluyendo CASSCF si existe) para cada sistema físico.",
    )
    p.add_argument(
        "--figure-format",
        type=str,
        default="both",
        choices=["pdf", "png", "both"],
        help="Formato de exportación para las figuras de post-producción.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Reutiliza evaluaciones MCMC previas si el archivo .h5 ya existe y contiene datos de producción.",
    )
    p.add_argument(
        "--postprocess-only",
        action="store_true",
        default=False,
        help="Omite el muestreo MCMC y ejecuta directamente el análisis estadístico y gráficas sobre los bundles .h5 ya evaluados.",
    )
    return p


def _convert_legacy_casscf_sampling_dir_to_h5(
    model_dir: Path,
    output_h5_path: Path,
    casscf_baselines: list[dict[str, Any]],
) -> Path:
    """
    Convierte una carpeta producida por `scripts/evaluar_wigner_casscf_2.py`
    (`summary.csv` + `actor_params.npz` + `casscf_exact_analysis/analysis_summary.json`)
    en un bundle unificado `.h5` evaluable por el pipeline modular.
    """
    import re
    path_str = str(model_dir.resolve())
    row = {}
    if (model_dir / "summary.csv").is_file():
        try:
            summary_df = pd.read_csv(model_dir / "summary.csv")
            if not summary_df.empty:
                row = summary_df.iloc[0].to_dict()
        except Exception:
            pass

    analysis_json = model_dir / "casscf_exact_analysis" / "analysis_summary.json"
    sys_info = {}
    if analysis_json.exists():
        try:
            with analysis_json.open("r", encoding="utf-8") as f:
                sys_info = json.load(f).get("system", {})
        except Exception:
            pass

    m_n = re.search(r"[Nn]e?(\d+)", path_str)
    n_path = int(m_n.group(1)) if m_n else 4
    m_g = re.search(r"[gG]_?([0-9]+(?:\.[0-9]+)?)", path_str)
    g_path = float(m_g.group(1)) if m_g else 4.0

    n_up = int(sys_info.get("n_up", row.get("n_up", n_path // 2)))
    n_dn = int(sys_info.get("n_down", row.get("n_dn", n_path - n_up)))
    n_elec = n_up + n_dn
    omega = float(sys_info.get("omega", row.get("omega", 0.1)))
    g = float(sys_info.get("g", row.get("g", g_path)))
    softening = float(sys_info.get("a", row.get("softening", 0.1)))

    match_b = find_matching_casscf_baseline(
        casscf_baselines, n_up=n_up, n_dn=n_dn, omega=omega, g=g, softening=softening
    )
    if match_b is None:
        raise ValueError(
            f"No se encontró archivo `casscf_state.npz` para reconstruir ansatz CASSCF de {model_dir}"
        )

    npz_path = match_b["npz_path"]
    actor_amp = float(row.get("actor_amp", 0.0))
    is_rl = actor_amp > 0.0

    cfg = Experiment1DConfig()
    cfg.system.n_elec = n_elec
    cfg.system.n_up = n_up
    cfg.system.n_dn = n_dn
    cfg.system.omega = omega
    cfg.system.interaction_strength = g
    cfg.system.softening = softening
    cfg.ansatz.ansatz_type = "casscf_npz"
    cfg.ansatz.casscf_npz_path = npz_path
    cfg.ansatz.casscf_trainable = False
    cfg.proposal.proposal_type = "neural_deepset" if is_rl else "mala"
    cfg.proposal.proposal_mode = "residual_mala"
    cfg.proposal.tau = float(row.get("tau_base", 0.1))
    cfg.proposal.actor_amplitude = actor_amp
    cfg.proposal.sigma_action_scale = float(row.get("sigma_scale", 1.1))
    cfg.rl.penalty_factor = float(row.get("penalty", 0.25))
    cfg.rl.max_phys_jump_factor = float(row.get("max_jump", 2.0))
    cfg.sampler.batch_size = int(row.get("n_walkers", 100))

    ansatz = build_ansatz_1d(cfg)
    ansatz_params = ansatz.init(jax.random.PRNGKey(0), jnp.zeros((1, n_elec, 1)))

    actor_params = None
    actor_npz = model_dir / "actor_params.npz"
    if is_rl and actor_npz.exists():
        with np.load(actor_npz, allow_pickle=True) as adata:
            if "params" in adata:
                actor_params = {"params": adata["params"].item()}
            else:
                actor_params = {k: adata[k] for k in adata.files}

    class _Flags:
        save_config = True
        save_ansatz_weights = True
        save_actor_weights = True
        save_critic_weights = False
        save_training_history = False
        save_production_walkers = False
        save_walkers_max_batch = 64
        save_production_energies = False
        save_production_sampler_stats = False

    save_run_bundle_h5(
        h5_path=output_h5_path,
        config_dict=cfg.to_dict(),
        io_flags=_Flags(),
        ansatz_best=ansatz_params,
        ansatz_final=ansatz_params,
        actor_final=actor_params,
        extra_metadata={"converted_from_casscf_sampling_dir": str(model_dir)},
    )
    return output_h5_path


def discover_model_targets(
    search_root: Path, pattern: str, audit_out_dir: Path
) -> list[dict[str, Any]]:
    """
    Busca recursivamente dentro de `search_root` (usando `pattern` para las carpetas
    raíz de resultados) todos los modelos entrenados y evaluables:
    1. Archivos `.h5` unificados nativos o previamente convertidos.
    2. Carpetas legacy de muestreo CASSCF (`summary.csv` + `actor_params.npz` o `casscf_exact_analysis`).
    3. Carpetas legacy de entrenamiento/evaluación:
       - Carpetas con pesos propios (`*_ansatz_weights.pkl`, `*_actor_weights.pkl`, `params_psi.pkl`, `*weights*.pkl`).
       - Carpetas hijas/symlinks de suites de barrido (ej. `MALA_tau*`, `TD3*`, `Perm_*`).
       - Resuelve symlinks de forma segura (`followlinks=True`) evitando ciclos y deduplicando por ruta canónica.
    """
    import os
    audit_real_str = str(audit_out_dir.resolve())
    search_root_resolved = search_root.resolve()

    # Identificar carpetas raíz candidatas según pattern
    root_dirs: list[Path] = []
    # 1. Matches directos en search_root
    for p in sorted(search_root.glob(pattern)):
        if p.is_dir() and str(p.resolve()) != audit_real_str:
            root_dirs.append(p)

    # 2. Si pattern no coincidió o search_root en sí misma es la carpeta objetivo
    if not root_dirs and search_root.is_dir():
        root_dirs.append(search_root)

    visited_real_dirs: set[str] = set()
    seen_model_dirs: set[str] = set()
    seen_h5_files: set[str] = set()
    targets: list[dict[str, Any]] = []

    for top_dir in root_dirs:
        top_real = str(top_dir.resolve())
        if top_real == audit_real_str:
            continue

        suite_name = top_dir.name

        for root, dirs, files in os.walk(str(top_dir), followlinks=True):
            real_root = os.path.realpath(root)

            # Protección contra bucles cíclicos de symlinks
            if real_root in visited_real_dirs:
                dirs.clear()
                continue
            visited_real_dirs.add(real_root)

            # Omitir la carpeta de salida del auditorio
            if real_root.startswith(audit_real_str):
                dirs.clear()
                continue

            # 1. Detectar archivos HDF5 unificados (*.h5)
            for f in files:
                if f.endswith(".h5") and not f.startswith("."):
                    h5_path = Path(root) / f
                    real_h5 = os.path.realpath(str(h5_path))
                    if real_h5 not in seen_h5_files and not real_h5.startswith(audit_real_str):
                        seen_h5_files.add(real_h5)
                        targets.append(
                            {
                                "type": "h5_bundle",
                                "path": h5_path,
                                "suite": suite_name,
                                "rel_path": os.path.relpath(real_h5, str(search_root_resolved)),
                            }
                        )

            # 2. Detectar carpetas de muestreo CASSCF legacy (summary.csv + actor_params.npz o casscf_exact_analysis)
            has_summary_csv = "summary.csv" in files
            has_actor_npz = "actor_params.npz" in files
            has_casscf_analysis = ("casscf_exact_analysis" in dirs) or ("analysis_summary.json" in files)
            if (has_summary_csv or has_casscf_analysis) and (has_actor_npz or has_summary_csv):
                if real_root not in seen_model_dirs:
                    seen_model_dirs.add(real_root)
                    targets.append(
                        {
                            "type": "legacy_casscf_sampling_dir",
                            "path": Path(root),
                            "suite": suite_name,
                            "rel_path": os.path.relpath(real_root, str(search_root_resolved)),
                        }
                    )
                    continue

            # 3. Detectar carpetas legacy de entrenamiento y evaluación
            ansatz_weights = [f for f in files if f.endswith("_ansatz_weights.pkl") or f == "params_psi.pkl"]
            actor_weights = [f for f in files if f.endswith("_actor_weights.pkl") or f == "actor_params.npz"]
            any_weights = [f for f in files if "weight" in f.lower() and f.endswith(".pkl")]
            has_config = ("params.json" in files) or ("run_config.json" in files) or ("config.json" in files)
            has_metrics = any("training_metrics" in f for f in files) or ("measurement_data.csv" in files)

            is_legacy = False
            if ansatz_weights:
                is_legacy = True
            elif actor_weights:
                is_legacy = True
            elif any_weights:
                is_legacy = True
            elif has_config and has_metrics:
                # Comprobar si hay pesos en el padre o hermano (como ocurre en TD3 desacoplado)
                p_path = Path(root).parent
                if p_path.is_dir():
                    for check_dir in [p_path, *p_path.glob("MALA*"), *p_path.glob("TD3*")]:
                        if check_dir.is_dir() and any("weight" in f.name.lower() for f in check_dir.iterdir() if f.is_file()):
                            is_legacy = True
                            break

            if is_legacy and real_root not in seen_model_dirs:
                seen_model_dirs.add(real_root)
                targets.append(
                    {
                        "type": "legacy_dir",
                        "path": Path(root),
                        "suite": suite_name,
                        "rel_path": os.path.relpath(real_root, str(search_root_resolved)),
                    }
                )

    return targets


def evaluate_single_bundle(
    h5_path: Path,
    suite_name: str,
    rel_id: str,
    args: argparse.Namespace,
    casscf_baselines: list[dict[str, Any]],
    casscf_density_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Carga un bundle HDF5 (con pesos de ansatz y actor), ejecuta `args.n_eval_runs`
    muestreos MCMC independientes para medir estadística rigurosa y observables físicos,
    compara contra la baseline CASSCF analítica si existe para el sistema físico,
    y actualiza el bundle HDF5 con los resultados de post-procesamiento.
    """
    bundle = load_run_bundle_h5(h5_path, load_walkers=True)
    cfg = Experiment1DConfig.from_dict(bundle["config"])

    ckpts = bundle.get("checkpoints", {})
    ansatz_params = ckpts.get("ansatz_best")
    if ansatz_params is None:
        ansatz_params = ckpts.get("ansatz_final")
    if ansatz_params is None:
        raise ValueError(f"El bundle {h5_path} no contiene pesos de ansatz.")

    actor_params = ckpts.get("actor_final")

    # Inferir modelos efectivos de orbital y correlación si es Slater-Jastrow
    if cfg.ansatz.ansatz_type in ("slater_jastrow_1d", "slater_jastrow"):
        eff_orb_model = _infer_orbital_model_from_params(
            ansatz_params.get("orbitals", {}), cfg.ansatz.orbital_model
        )
        eff_corr_model = _infer_correlation_model_from_params(
            ansatz_params.get("correlation", {}), cfg.ansatz.correlation_model
        )
        cfg.ansatz.orbital_model = eff_orb_model
        cfg.ansatz.correlation_model = eff_corr_model
    else:
        eff_orb_model = "casscf"
        eff_corr_model = "ci_expansion"

    # Configurar tamaño de lote para la evaluación
    if args.batch_size > 0:
        cfg.sampler.batch_size = int(args.batch_size)
    else:
        cfg.sampler.batch_size = min(int(cfg.sampler.batch_size), 512)

    hamiltonian = Harmonic1DHamiltonian(cfg.system)
    ansatz = build_ansatz_1d(cfg)
    proposal = build_proposal_1d(cfg)
    sampler = MHSampler1D(cfg, ansatz, proposal)

    use_actor = bool(
        proposal.is_neural and actor_params is not None and len(actor_params) > 0
    )
    method_label = (
        f"RL-MALA ({cfg.rl.algorithm.upper()})"
        if use_actor
        else cfg.proposal.proposal_type.upper()
    )

    n_runs = max(1, int(args.n_eval_runs))
    n_burn = int(args.n_burn_in)
    n_steps = int(args.n_steps)
    batch_size = int(cfg.sampler.batch_size)

    master_key = jax.random.PRNGKey(int(args.seed))
    run_keys = jax.random.split(master_key, n_runs)

    e_means, e_vars, e_sems = [], [], []
    iats, ess_vals, ess_per_secs, wall_times = [], [], [], []
    acc_means, esjd_means, drift_means, sigma_means = [], [], [], []

    chains_walkers = []
    chains_eloc, chains_tloc, chains_vloc = [], [], []
    chains_stats: dict[str, list[np.ndarray]] = {
        "acceptance": [],
        "alpha": [],
        "proposed_disp": [],
        "accepted_disp": [],
        "esjd": [],
        "actor_drift_mag": [],
        "actor_sigma": [],
    }
    total_wall_time = 0.0

    reuse = getattr(args, "resume", False) or getattr(args, "postprocess_only", False)
    has_audited_prod = (
        "production" in bundle
        and "local_energy" in bundle["production"]
        and bundle["production"]["local_energy"] is not None
        and np.asarray(bundle["production"]["local_energy"]).size > 0
    )
    if getattr(args, "postprocess_only", False) and not has_audited_prod:
        raise RuntimeError(
            f"El bundle {h5_path} no contiene datos de producción y --postprocess-only está activo."
        )

    reused_production = False
    if reuse and has_audited_prod:
        prod_data = bundle["production"]
        prod_attrs = prod_data.get("attrs", {})
        stored_eloc = np.asarray(prod_data["local_energy"], dtype=np.float64)
        if stored_eloc.ndim == 2:
            stored_eloc = stored_eloc[None, ...]
        n_runs = stored_eloc.shape[0]
        n_steps = stored_eloc.shape[1]
        stored_batch = stored_eloc.shape[2] if stored_eloc.ndim >= 3 else 1
        attr_n_walkers = prod_attrs.get("n_walkers", None)
        if stored_batch == 1 and attr_n_walkers is not None and int(attr_n_walkers) > 0:
            batch_size = int(attr_n_walkers)
        else:
            batch_size = stored_batch
        e_var_attr = prod_attrs.get("e_var_per_chain", None)
        e_var_attr_arr = (
            np.asarray(e_var_attr, dtype=np.float64).reshape(-1)
            if e_var_attr is not None
            else None
        )
        total_wall_time = float(prod_attrs.get("wall_time_sec", 1.0))
        dt = max(total_wall_time / max(1, n_runs), 1e-6)

        for r_idx in range(n_runs):
            eloc_all = stored_eloc[r_idx]
            acf = compute_acf_1d(eloc_all, max_lag=min(args.max_lag, n_steps - 1))
            tau_int = compute_iat_sokal(acf, c_sokal=args.sokal_c)
            total_samples = float(n_steps * batch_size)
            ess = total_samples / max(tau_int, 1.0)
            ess_s = ess / dt
            e_mean = float(np.mean(eloc_all))
            if (
                stored_batch == 1
                and e_var_attr_arr is not None
                and r_idx < len(e_var_attr_arr)
                and np.isfinite(e_var_attr_arr[r_idx])
            ):
                e_var = float(e_var_attr_arr[r_idx])
            else:
                e_var = float(np.var(eloc_all))
            e_sem = float(np.sqrt(max(e_var, 0.0) / max(ess, 1.0)))

            e_means.append(e_mean)
            e_vars.append(e_var)
            e_sems.append(e_sem)
            iats.append(float(tau_int))
            ess_vals.append(float(ess))
            ess_per_secs.append(float(ess_s))
            wall_times.append(float(dt))

            s_stats = prod_data.get("sampler_stats", {})
            if "acceptance" in s_stats and len(s_stats["acceptance"]) > 0:
                acc_arr = np.asarray(s_stats["acceptance"], dtype=np.float64)
                acc_means.append(float(np.mean(acc_arr[r_idx] if acc_arr.ndim >= 2 else acc_arr)))
            else:
                acc_means.append(float("nan"))
            if "esjd" in s_stats and len(s_stats["esjd"]) > 0:
                esjd_arr = np.asarray(s_stats["esjd"], dtype=np.float64)
                esjd_means.append(float(np.mean(esjd_arr[r_idx] if esjd_arr.ndim >= 2 else esjd_arr)))
            else:
                esjd_means.append(float("nan"))
            if "actor_drift_mag" in s_stats and len(s_stats["actor_drift_mag"]) > 0:
                drift_arr = np.asarray(s_stats["actor_drift_mag"], dtype=np.float64)
                drift_means.append(float(np.mean(drift_arr[r_idx] if drift_arr.ndim >= 2 else drift_arr)))
            else:
                drift_means.append(float("nan"))
            if "actor_sigma" in s_stats and len(s_stats["actor_sigma"]) > 0:
                sig_arr = np.asarray(s_stats["actor_sigma"], dtype=np.float64)
                sigma_means.append(float(np.mean(sig_arr[r_idx] if sig_arr.ndim >= 2 else sig_arr)))
            else:
                sigma_means.append(float("nan"))

        all_walkers_arr = np.asarray(prod_data.get("walkers")) if "walkers" in prod_data else None
        if all_walkers_arr is not None and all_walkers_arr.ndim == 4:
            all_walkers_arr = all_walkers_arr[None, ...]
        reused_production = True
    else:
        for r_idx in range(n_runs):
            k_init, k_burn, k_prod = jax.random.split(run_keys[r_idx], 3)
            c_state = sampler.init_state(k_init, ansatz_params)

            if n_burn > 0:
                _, c_state = sampler.run_burn_in(
                    k_burn,
                    c_state,
                    ansatz_params,
                    actor_params,
                    n_burn,
                    use_actor=use_actor,
                )

            res = sampler.run_production_chain(
                k_prod,
                c_state,
                ansatz_params,
                actor_params,
                hamiltonian,
                measure_steps=n_steps,
                thin=1,
                jit_chunk_size=min(500, n_steps),
                use_actor=use_actor,
            )
            dt = max(float(res["wall_time_sec"]), 1e-6)
            total_wall_time += dt

            eloc_all = np.asarray(res["local_energy"])  # shape (n_steps, batch_size)
            acf = compute_acf_1d(eloc_all, max_lag=min(args.max_lag, n_steps - 1))
            tau_int = compute_iat_sokal(acf, c_sokal=args.sokal_c)

            total_samples = float(eloc_all.size)
            ess = total_samples / max(tau_int, 1.0)
            ess_s = ess / dt

            e_mean = float(np.mean(eloc_all))
            e_var = float(np.var(eloc_all))
            e_sem = float(np.sqrt(max(e_var, 0.0) / max(ess, 1.0)))

            e_means.append(e_mean)
            e_vars.append(e_var)
            e_sems.append(e_sem)
            iats.append(float(tau_int))
            ess_vals.append(float(ess))
            ess_per_secs.append(float(ess_s))
            wall_times.append(float(dt))

            acc_arr = np.asarray(res["sampler_stats"]["acceptance"])
            esjd_arr = np.asarray(res["sampler_stats"]["esjd"])
            drift_arr = np.asarray(res["sampler_stats"]["actor_drift_mag"])
            sig_arr = np.asarray(res["sampler_stats"]["actor_sigma"])

            acc_means.append(float(np.mean(acc_arr)))
            esjd_means.append(float(np.mean(esjd_arr)))
            drift_means.append(float(np.mean(drift_arr)))
            sigma_means.append(float(np.mean(sig_arr)))

            chains_walkers.append(
                np.asarray(res["walkers"][:, : max(1, int(args.max_walkers_save)), :, :])
            )
            chains_eloc.append(eloc_all)
            chains_tloc.append(np.asarray(res["kinetic_energy"]))
            chains_vloc.append(np.asarray(res["potential_energy"]))
            for sk in chains_stats.keys():
                chains_stats[sk].append(np.asarray(res["sampler_stats"][sk]))

        all_walkers_arr = np.stack(chains_walkers, axis=0)  # (n_runs, n_steps, W_sub, N, 1)

    # Calcular observables físicos con media y desviación estándar entre las n_runs cadenas
    if all_walkers_arr is not None and all_walkers_arr.size > 0:
        obs = compute_1d_observables(
            all_walkers_arr,
            n_up=cfg.system.n_up,
            n_down=cfg.system.n_dn,
            n_bins=args.density_bins,
            omega=cfg.system.omega,
        )
    else:
        obs = {
            "n_wigner_peaks": -1,
            "rms_radius": float("nan"),
            "mean_pair_dist": float("nan"),
            "x_max": 15.0,
        }

    # Buscar si el sistema físico (N_up, N_dn, omega, g) tiene baseline CASSCF pre-entrenada
    casscf_match = find_matching_casscf_baseline(
        casscf_baselines,
        n_up=cfg.system.n_up,
        n_dn=cfg.system.n_dn,
        omega=cfg.system.omega,
        g=cfg.system.interaction_strength,
        softening=cfg.system.softening,
    )

    has_casscf = False
    casscf_label = ""
    casscf_npz_path = ""
    casscf_energy = float("nan")
    delta_e_vs_casscf = float("nan")
    kl_tot_mean = float("nan")
    kl_tot_std = float("nan")
    kl_tot_sem = float("nan")
    kl_up_mean = float("nan")
    kl_dn_mean = float("nan")

    if casscf_match is not None:
        has_casscf = True
        casscf_npz_path = str(casscf_match["npz_path"])
        casscf_label = str(casscf_match["label"])
        if casscf_npz_path not in casscf_density_cache:
            casscf_density_cache[casscf_npz_path] = compute_exact_casscf_1body_density(
                casscf_npz_path,
                x_min=-float(obs["x_max"]),
                x_max=float(obs["x_max"]),
                n_grid=500,
            )
        casscf_ref = casscf_density_cache[casscf_npz_path]
        casscf_energy = float(casscf_ref["energy"])
        delta_e_vs_casscf = float(np.mean(e_means) - casscf_energy)

        # Evaluar D_KL resuelta por espín en cada una de las n_runs corridas independientes
        run_kls_tot, run_kls_up, run_kls_dn = [], [], []
        if all_walkers_arr is not None and all_walkers_arr.size > 0:
            for r_i in range(min(n_runs, all_walkers_arr.shape[0])):
                kls = compute_spin_kl_divergences(
                    all_walkers_arr[r_i],
                    casscf_ref,
                    n_bins=args.density_bins,
                    x_max=float(obs["x_max"]),
                )
                run_kls_tot.append(kls["kl_total"])
                run_kls_up.append(kls["kl_up"])
                run_kls_dn.append(kls["kl_down"])

        if run_kls_tot:
            kl_tot_mean = float(np.mean(run_kls_tot))
            kl_tot_std = float(np.std(run_kls_tot))
            kl_tot_sem = float(kl_tot_std / np.sqrt(max(1, len(run_kls_tot))))
            kl_up_mean = float(np.mean(run_kls_up))
            kl_dn_mean = float(np.mean(run_kls_dn))

    if not reused_production:
        # Actualizar el HDF5 con las cadenas de producción multi-semilla
        production_payload = {
            "n_chains": n_runs,
            "n_burn_in": n_burn,
            "measure_steps": n_steps,
            "thin": 1,
            "wall_time_sec": float(total_wall_time),
            "walkers": all_walkers_arr,
            "local_energy": np.stack(chains_eloc, axis=0),
            "kinetic_energy": np.stack(chains_tloc, axis=0),
            "potential_energy": np.stack(chains_vloc, axis=0),
            "sampler_stats": {
                sk: np.stack(slist, axis=0) for sk, slist in chains_stats.items()
            },
        }

        class _SaveAuditFlags:
            save_config = True
            save_ansatz_weights = True
            save_actor_weights = True
            save_critic_weights = True
            save_training_history = True
            save_production_walkers = True
            save_walkers_max_batch = int(args.max_walkers_save)
            save_production_energies = True
            save_production_sampler_stats = True

        trn_df = bundle.get("training_df")
        trn_records = trn_df.to_dict(orient="records") if trn_df is not None else None

        save_run_bundle_h5(
            h5_path=h5_path,
            config_dict=cfg.to_dict(),
            io_flags=_SaveAuditFlags(),
            ansatz_best=ansatz_params,
            ansatz_final=ansatz_params,
            actor_final=actor_params,
            critic1_final=ckpts.get("critic1_final"),
            critic2_final=ckpts.get("critic2_final"),
            training_history=trn_records,
            production_data=production_payload,
            extra_metadata={
                "audited_by": "cli/audit_results.py",
                "suite": suite_name,
                "has_casscf_baseline": has_casscf,
                "casscf_npz_path": casscf_npz_path,
            },
        )

    energy_mean_across_runs = float(np.mean(e_means))
    energy_std_across_runs = float(np.std(e_means))
    energy_sem_across_runs = (
        float(energy_std_across_runs / np.sqrt(n_runs))
        if n_runs > 1
        else float(np.mean(e_sems))
    )

    summary_scalar = {
        "energy_mean": energy_mean_across_runs,
        "energy_std_across_runs": energy_std_across_runs,
        "energy_sem_across_runs": energy_sem_across_runs,
        "energy_var": float(np.mean(e_vars)),
        "energy_sem_chain": float(np.mean(e_sems)),
        "iat_sokal": float(np.mean(iats)),
        "iat_std_across_runs": float(np.std(iats)),
        "ess": float(np.mean(ess_vals)),
        "ess_per_sec": float(np.mean(ess_per_secs)),
        "ess_per_sec_std": float(np.std(ess_per_secs)),
        "acceptance_mean": float(np.mean(acc_means)),
        "acceptance_std": float(np.std(acc_means)),
        "esjd_mean": float(np.mean(esjd_means)),
        "actor_drift_mean": float(np.mean(drift_means)),
        "actor_sigma_mean": float(np.mean(sigma_means)),
        "wigner_num_peaks": int(obs["n_wigner_peaks"]),
        "rms_radius": float(obs["rms_radius"]),
        "mean_pair_dist": float(obs["mean_pair_dist"]),
        "has_casscf_baseline": has_casscf,
        "casscf_label": casscf_label,
        "casscf_energy": casscf_energy,
        "delta_E_vs_CASSCF": delta_e_vs_casscf,
        "kl_casscf_total_mean": kl_tot_mean,
        "kl_casscf_total_std": kl_tot_std,
        "kl_casscf_total_sem": kl_tot_sem,
        "kl_casscf_up_mean": kl_up_mean,
        "kl_casscf_down_mean": kl_dn_mean,
        "casscf_npz_path": casscf_npz_path,
    }

    return {
        "suite": suite_name,
        "rel_path": rel_id,
        "bundle_path": str(h5_path),
        "method": method_label,
        "is_rl": bool(use_actor),
        "ansatz_type": cfg.ansatz.ansatz_type,
        "proposal_type": cfg.proposal.proposal_type,
        "proposal_mode": cfg.proposal.proposal_mode,
        "N_elec": int(cfg.system.n_elec),
        "N_up": int(cfg.system.n_up),
        "N_dn": int(cfg.system.n_dn),
        "omega": float(cfg.system.omega),
        "g": float(cfg.system.interaction_strength),
        "softening": float(cfg.system.softening),
        "orbital_model": eff_orb_model,
        "correlation_model": eff_corr_model,
        "tau": float(cfg.proposal.tau),
        "actor_amplitude": float(cfg.proposal.actor_amplitude) if use_actor else 0.0,
        "penalty_factor": float(cfg.rl.penalty_factor) if use_actor else 0.0,
        "sigma_action_scale": float(cfg.proposal.sigma_action_scale) if use_actor else 0.0,
        "n_eval_runs": n_runs,
        "n_steps_per_run": n_steps,
        "batch_size": batch_size,
        **summary_scalar,
    }


def plot_global_audit_dashboard(df: pd.DataFrame, out_dir: Path) -> None:
    """Genera un dashboard visual resumiendo todos los modelos auditados (y baselines CASSCF si existen)."""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    labels = [
        f"{r['method']}\nN={r['N_elec']}, g={r['g']}, ω={r['omega']} (τ={r['tau']})"
        for _, r in df.iterrows()
    ]
    x = np.arange(len(df))
    colors = ["#2ecc71" if r["is_rl"] else "#e74c3c" for _, r in df.iterrows()]

    # Panel 1: Energía Local Media +/- SEM entre evaluaciones + Marcador de Baseline CASSCF si existe
    ax = axes[0, 0]
    ax.bar(
        x,
        df["energy_mean"],
        yerr=df["energy_std_across_runs"],
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        label=r"Sampled $\langle E_L \rangle \pm 1\sigma$",
    )
    has_cas_mask = df["has_casscf_baseline"].astype(bool) & df["casscf_energy"].notna()
    if has_cas_mask.any():
        cas_x = x[has_cas_mask.values]
        cas_y = df.loc[has_cas_mask, "casscf_energy"].values
        ax.scatter(
            cas_x,
            cas_y,
            color="black",
            marker="D",
            s=55,
            zorder=5,
            label=r"Exact CASSCF Baseline $E_{\mathrm{CASSCF}}$",
        )
        for cx, cy in zip(cas_x, cas_y):
            ax.hlines(cy, cx - 0.4, cx + 0.4, colors="black", linestyles=":", linewidth=2.2, zorder=4)
    ax.set_ylabel(r"$\langle E_L \rangle$ (Ha)")
    ax.set_title("Ground-State Energy across Audited Models", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    e_clean = df["energy_mean"][
        np.isfinite(df["energy_mean"]) & (df["energy_mean"] > -50) & (df["energy_mean"] < 300)
    ]
    if len(e_clean) > 0:
        ax.set_ylim(max(-25.0, float(np.min(e_clean)) - 2.0), min(300.0, float(np.max(e_clean)) + 5.0))

    # Panel 2: Sokal IAT (tau_int)
    ax = axes[0, 1]
    ax.bar(
        x,
        df["iat_sokal"],
        yerr=df["iat_std_across_runs"],
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_ylabel(r"Integrated Autocorrelation Time $\tau_{\mathrm{int}}$")
    ax.set_title(r"Sokal $\tau_{\mathrm{int}}$ (Lower is Better)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    iat_clean = df["iat_sokal"][np.isfinite(df["iat_sokal"])]
    if len(iat_clean) > 0:
        ax.set_ylim(0, min(1000.0, float(np.percentile(iat_clean, 98) * 1.4)))

    # Panel 3: Throughput ESS/s (o Divergencia KL contra CASSCF si hay baselines)
    ax = axes[1, 0]
    ax.bar(
        x,
        df["ess_per_sec"],
        yerr=df["ess_per_sec_std"],
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
    )
    ax.set_ylabel("Effective Sample Size / sec (ESS/s)")
    ax.set_title("Wall-Clock Sampling Throughput (Higher is Better)", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ess_clean = df["ess_per_sec"][np.isfinite(df["ess_per_sec"])]
    if len(ess_clean) > 0:
        ax.set_ylim(0, min(200000.0, float(np.percentile(ess_clean, 98) * 1.4)))

    # Panel 4: Ganancia Topológica G_RL vs MALA
    ax = axes[1, 1]
    ax.bar(x, df["G_RL_vs_best_MALA"], color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(
        1.0, color="black", linestyle="--", linewidth=1.2, label="Best MALA Baseline (1.0x)"
    )
    ax.set_ylabel(r"Topological Efficiency Gain $G_{\mathrm{RL}}$")
    ax.set_title(
        r"Efficiency Gain $G_{\mathrm{RL}} = \tau_{\mathrm{MALA}} / \tau_{\mathrm{Model}}$",
        fontweight="bold",
    )
    ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", alpha=0.3)
    grl_clean = df["G_RL_vs_best_MALA"][np.isfinite(df["G_RL_vs_best_MALA"])]
    if len(grl_clean) > 0:
        ax.set_ylim(0, min(150.0, max(2.5, float(np.percentile(grl_clean, 98) * 1.4))))

    if len(df) > 40:
        step = max(1, len(df) // 15)
        tick_pos = x[::step]
        tick_lbl = [f"#{i+1}" for i in tick_pos]
        for a in axes.flat:
            a.set_xticks(tick_pos)
            a.set_xticklabels(tick_lbl, fontsize=8)
            a.set_xlabel("Model Index (Sorted by N, ω, g, G_RL)", fontsize=9)
    else:
        for a in axes.flat:
            a.set_xticks(x)
            a.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    fig.suptitle(
        "Exhaustive Statistical Post-Production Audit across `results*` Models",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "global_audit_dashboard.png", dpi=240)
    fig.savefig(out_dir / "global_audit_dashboard.pdf")
    plt.close(fig)

    fig.suptitle(
        "Exhaustive Statistical Post-Production Audit across `results*` Models",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "global_audit_dashboard.png", dpi=240)
    fig.savefig(out_dir / "global_audit_dashboard.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = build_audit_argparser()
    args = parser.parse_args(argv)

    search_root = Path(args.search_root).resolve()
    uhf_root = Path(args.uhf_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    converted_h5_dir = out_dir / "converted_bundles"
    converted_h5_dir.mkdir(parents=True, exist_ok=True)

    # 1. Descubrir baselines CASSCF pre-entrenadas en `uhf*`
    casscf_baselines = discover_casscf_baselines(
        search_root=uhf_root, uhf_pattern=args.uhf_pattern
    )
    casscf_density_cache: dict[str, dict[str, Any]] = {}

    # 2. Descubrir modelos a auditar en `results*`
    targets = discover_model_targets(search_root, args.pattern, out_dir)
    if not targets:
        print(
            f"[Audit] No se encontraron modelos en `{search_root}` con el patrón `{args.pattern}`."
        )
        return

    print("=" * 88)
    print(
        f"[Audit] Se encontraron {len(targets)} modelos en `{search_root}` (patrón: `{args.pattern}`)."
    )
    print(
        f"[Audit] Baselines CASSCF pre-entrenadas detectadas en `{uhf_root}`: {len(casscf_baselines)}"
    )
    print(
        f"[Audit] Evaluaciones independientes por modelo: {args.n_eval_runs} | "
        f"Burn-in: {args.n_burn_in} | Pasos/eval: {args.n_steps}"
    )
    print("=" * 88)

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(targets, 1):
        rel_id = item["rel_path"]
        suite_name = item["suite"]
        print(f"\n[{idx}/{len(targets)}] Auditando: {rel_id} ...")
        try:
            safe_name = rel_id.replace("/", "__").replace("\\", "__")
            if item["type"] == "legacy_dir":
                h5_dest = converted_h5_dir / f"{safe_name}.h5"
                if (args.resume or args.postprocess_only) and h5_dest.is_file():
                    h5_path = h5_dest
                else:
                    h5_path = convert_legacy_dir_to_h5(item["path"], output_h5_path=h5_dest)
            elif item["type"] == "legacy_casscf_sampling_dir":
                h5_dest = converted_h5_dir / f"{safe_name}.h5"
                if (args.resume or args.postprocess_only) and h5_dest.is_file():
                    h5_path = h5_dest
                else:
                    h5_path = _convert_legacy_casscf_sampling_dir_to_h5(
                        item["path"], h5_dest, casscf_baselines
                    )
            else:
                h5_path = Path(item["path"])

            rec = evaluate_single_bundle(
                h5_path,
                suite_name,
                rel_id,
                args,
                casscf_baselines=casscf_baselines,
                casscf_density_cache=casscf_density_cache,
            )
            records.append(rec)
            cas_msg = (
                f" | CASSCF E={rec['casscf_energy']:.5f} (ΔE={rec['delta_E_vs_CASSCF']:+.5f}, "
                f"KL={rec['kl_casscf_total_mean']:.4f}±{rec['kl_casscf_total_std']:.4f})"
                if rec["has_casscf_baseline"]
                else " | [Sin baseline CASSCF]"
            )
            print(
                f"   -> OK | {rec['method']} | N={rec['N_elec']}, ω={rec['omega']}, g={rec['g']} | "
                f"E={rec['energy_mean']:.5f}±{rec['energy_std_across_runs']:.5f} Ha | "
                f"IAT={rec['iat_sokal']:.2f}±{rec['iat_std_across_runs']:.2f} | "
                f"ESS/s={rec['ess_per_sec']:.1f}{cas_msg}"
            )
        except Exception as exc:
            print(f"   -> [WARN] Saltando {rel_id} por error: {exc}")

    if not records:
        print("[Audit] Ningún modelo pudo ser evaluado.")
        return

    df = pd.DataFrame(records)

    # Agrupar por sistema físico (N_elec, N_up, N_dn, omega, g) para comparar contra el mejor MALA
    df["G_RL_vs_best_MALA"] = 1.0
    df["speedup_ESS_vs_best_MALA"] = 1.0
    df["delta_E_vs_best_MALA"] = 0.0
    df["best_MALA_reference"] = ""

    group_cols = ["N_elec", "N_up", "N_dn", "omega", "g"]
    for (n_val, n_up_val, n_dn_val, om_val, g_val), grp in df.groupby(group_cols):
        mala_rows = grp[~grp["is_rl"]]
        if not mala_rows.empty:
            best_mala_idx = mala_rows["ess_per_sec"].idxmax()
            ref_row = df.loc[best_mala_idx]
        else:
            best_idx = grp["ess_per_sec"].idxmax()
            ref_row = df.loc[best_idx]

        ref_iat = max(float(ref_row["iat_sokal"]), 1e-8)
        ref_ess_s = max(float(ref_row["ess_per_sec"]), 1e-8)
        ref_e = float(ref_row["energy_mean"])
        ref_name = str(ref_row["rel_path"])

        for row_idx in grp.index:
            df.loc[row_idx, "G_RL_vs_best_MALA"] = ref_iat / max(
                float(df.loc[row_idx, "iat_sokal"]), 1e-8
            )
            df.loc[row_idx, "speedup_ESS_vs_best_MALA"] = (
                float(df.loc[row_idx, "ess_per_sec"]) / ref_ess_s
            )
            df.loc[row_idx, "delta_E_vs_best_MALA"] = (
                float(df.loc[row_idx, "energy_mean"]) - ref_e
            )
            df.loc[row_idx, "best_MALA_reference"] = ref_name

    # Ordenar y guardar tabla maestra CSV y JSON de inmediato (blindado ante fallas en plots)
    df = df.sort_values(
        by=["N_elec", "omega", "g", "G_RL_vs_best_MALA"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    csv_path = out_dir / "master_models_audit.csv"
    json_path = out_dir / "master_models_audit.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"\n[Audit] Tabla maestra CSV guardada en: {csv_path}")
    print(f"[Audit] Tabla maestra JSON guardada en: {json_path}")

    # Generar las figuras detalladas de post-producción por cada sistema físico
    if args.generate_system_plots:
        for (n_val, n_up_val, n_dn_val, om_val, g_val), grp in df.groupby(group_cols):
            if len(grp) < 1:
                continue
            try:
                sys_plot_dir = out_dir / f"system_N{n_val}_up{n_up_val}_dn{n_dn_val}_g{g_val}_om{om_val}"
                mala_rows = grp[~grp["is_rl"]]
                ref_idx = mala_rows["ess_per_sec"].idxmax() if not mala_rows.empty else grp["ess_per_sec"].idxmax()
                ordered_indices = [ref_idx] + [i for i in grp.index if i != ref_idx]
                bundle_list = [str(df.loc[i, "bundle_path"]) for i in ordered_indices]
                label_list = [
                    f"{df.loc[i, 'method']}_tau{df.loc[i, 'tau']} ({Path(df.loc[i, 'rel_path']).name})"
                    for i in ordered_indices
                ]
                matched_npz = str(df.loc[ref_idx, "casscf_npz_path"]) if df.loc[ref_idx, "has_casscf_baseline"] else None
                pp_cfg = PostProcessConfig(
                    inputs=bundle_list,
                    labels=label_list,
                    out_dir=str(sys_plot_dir),
                    casscf_npz_path=matched_npz if matched_npz else None,
                    uhf_search_root=str(uhf_root),
                    auto_casscf_baseline=True,
                    sokal_window_c=args.sokal_c,
                    max_acf_lag=args.max_lag,
                    n_density_bins=args.density_bins,
                    figure_format=args.figure_format,
                )
                run_postprocessing_pipeline(pp_cfg)
            except Exception as exc:
                print(f"   -> [WARN] Error generando plots para sistema N={n_val}, g={g_val}: {exc}")

    try:
        plot_global_audit_dashboard(df, out_dir)
    except Exception as exc:
        print(f"   -> [WARN] Error generando dashboard global: {exc}")

    print("\n" + "=" * 88)
    print(f"[Audit] Auditoría completada exitosamente para {len(df)} modelos.")
    print(f"[Audit] Tabla maestra CSV : {csv_path}")
    print(f"[Audit] Tabla maestra JSON: {json_path}")
    print(f"[Audit] Dashboard global  : {out_dir / 'global_audit_dashboard.pdf'}")
    print("=" * 88)
    cols_show = [
        "rel_path",
        "method",
        "N_elec",
        "omega",
        "g",
        "energy_mean",
        "energy_std_across_runs",
        "iat_sokal",
        "ess_per_sec",
        "G_RL_vs_best_MALA",
        "has_casscf_baseline",
        "casscf_energy",
        "kl_casscf_total_mean",
    ]
    print(df[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
