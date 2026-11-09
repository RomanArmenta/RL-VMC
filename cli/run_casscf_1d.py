#!/usr/bin/env python3
"""
CLI para el Pipeline 1: Producción de Referencias Exactas de Espacio Activo (CASSCF 1D)
=======================================================================================
Envuelve la paquetería del asesor (`uhf_1d_ho_v270826_m-determinants/src/uhf1d`) con:
1. Selección automática de hiperparámetros seguros probados por el asesor (`n_core`,
   `n_active_orbitals`, `n_basis`, `quadrature_order`, `n_final_determinants`) según
   (N, omega, g) para garantizar convergencia rápida sin cuelgues combinatorios.
2. Ejecución de `run_casscf` (precondicionador RHF DIIS + optimización variacional
   acoplada BFGS de rotaciones orbitales anti-hermíticas y diagonalización CI).
3. Generación de `casscf_state.npz`, `summary.json`, `casscf_exact_density.csv` y
   gráficas diagnósticas (`densities.png`, `ci_weights.png`, `natural_occupations.png`, etc.).
4. Registro incremental en vivo en una tabla CSV maestra (`--master_csv`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Localizar automáticamente la carpeta uhf* del asesor
UHF_DIRS = sorted(ROOT_DIR.glob("uhf*"))
for uhf_d in UHF_DIRS:
    src_d = uhf_d / "src"
    if src_d.is_dir() and str(src_d) not in sys.path:
        sys.path.insert(0, str(src_d))

from rl_vmc.core.io_bundle import (
    atomic_copy_file,
    atomic_write_csv,
    atomic_write_json,
    locked_upsert_csv,
)
from rl_vmc.postprocess.casscf_baseline import (
    compute_exact_casscf_1body_density,
    discover_casscf_baselines,
    find_matching_casscf_baseline,
)


def get_advisor_safe_casscf_defaults(
    n_up: int,
    n_dn: int,
    omega: float,
    g: float,
    softening_override: float | None = None,
) -> dict[str, Any]:
    """
    Retorna los hiperparámetros CASSCF predeterminados basados en los archivos
    `summary.json` y `examples/*.json` ya convergidos por el asesor en
    `uhf_1d_ho_v270826_m-determinants`.

    Esto mantiene la dimensión del espacio CI (`n_determinants`) entre 9 y 100,
    evitando que la transformación de 4 índices ERI + diagonalización densa en
    cada evaluación de BFGS escale combinatoriamente y cuelgue el proceso.
    """
    n_elec = n_up + n_dn
    if n_elec == 2:
        n_core = 0
        n_active_orbitals = 4
        n_basis = 36
        quadrature_order = 100
        default_softening = 0.1
    elif n_elec == 4:
        n_core = 0
        # 4 orbitales activos (36 determinantes) con base expandida n_basis=48
        n_active_orbitals = 4
        n_basis = 48
        quadrature_order = 100
        default_softening = 0.1
    elif n_elec == 6:
        # En N=6 (3 up, 3 down), congelar n_core=1 deja 2 up y 2 down activos -> 36 dets
        n_core = 1
        n_active_orbitals = 4
        n_basis = 48
        quadrature_order = 100
        default_softening = 0.1
    elif n_elec == 8:
        # En N=8 (4 up, 4 down), congelar n_core=2 deja 2 up y 2 down activos -> 36 dets
        n_core = 2
        n_active_orbitals = 4
        n_basis = 48
        quadrature_order = 100
        default_softening = 0.1
    elif n_elec == 10:
        # En N=10 (5 up, 5 down), n_core=4, n_active_orbitals=4 (16 dets), n_basis=48, quad=100
        n_core = 4
        n_active_orbitals = 4
        n_basis = 48
        quadrature_order = 100
        default_softening = 0.25
    else:
        n_core = max(0, min(n_up, n_dn) - 2)
        n_active_orbitals = 4
        n_basis = 40
        quadrature_order = 100
        default_softening = 0.1

    softening = float(softening_override) if softening_override is not None else default_softening
    # Rango espacial de graficación adaptado a la trampa armónica 1D y repulsión g
    plot_xmax = max(12.0, min(25.0, math.ceil(4.2 / math.sqrt(max(omega, 0.01)) + 0.6 * math.sqrt(max(g, 0.0)))))

    return {
        "n_up": int(n_up),
        "n_down": int(n_dn),
        "omega": float(omega),
        "g": float(g),
        "softening": float(softening),
        "n_basis": int(n_basis),
        "quadrature_order": int(quadrature_order),
        "n_core": int(n_core),
        "n_active_orbitals": int(n_active_orbitals),
        "n_final_determinants": 0,  # 0 = conservar todos los determinantes del espacio activo
        "max_iter": 200,
        "energy_tol": 1e-10,
        "density_tol": 1e-8,
        "mixing": 0.35,
        "diis": True,
        "diis_start": 3,
        "diis_size": 8,
        "casscf_max_iter": 1000,
        "casscf_energy_tol": 1e-8,
        "orbital_gradient_tol": 1e-6,
        "max_ci_dimension": 20000,
        "plot_xmax": float(plot_xmax),
        "plot_points": 500,
    }


def _append_master_csv(
    csv_path: str | Path,
    row: dict[str, Any],
    collect_dir: str | Path | None = None,
) -> None:
    locked_upsert_csv(
        pd.DataFrame([row]),
        csv_path,
        key_cols=["system_tag"],
        collect_dir=collect_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline 1: Ejecución y exportación de referencias CASSCF 1D (paquetería uhf1d)."
    )
    p.add_argument("--n_elec", type=int, default=4, help="Número total de electrones N.")
    p.add_argument("--n_up", type=int, default=None, help="Electrones spin-up (por defecto N//2).")
    p.add_argument("--n_dn", type=int, default=None, help="Electrones spin-down (por defecto N - n_up).")
    p.add_argument("--omega", type=float, default=0.1, help="Frecuencia de la trampa armónica 1D.")
    p.add_argument("--g", type=float, default=1.0, help="Intensidad de interacción repulsiva g.")
    p.add_argument(
        "--softening",
        type=float,
        default=None,
        help="Parámetro de suavizado de Coulomb a (si se omite, usa el valor seguro del asesor: 0.1 o 0.25).",
    )
    p.add_argument("--n_basis", type=int, default=None, help="Tamaño de la base de oscilador armónico.")
    p.add_argument("--quadrature_order", type=int, default=None, help="Orden de cuadratura Gauss-Hermite.")
    p.add_argument("--n_core", type=int, default=None, help="Número de orbitales core doblemente ocupados.")
    p.add_argument("--n_active_orbitals", type=int, default=None, help="Número de orbitales activos.")
    p.add_argument(
        "--n_final_determinants",
        type=int,
        default=None,
        help="Truncamiento final de determinantes (0 = conservar todos los del espacio activo).",
    )
    p.add_argument("--casscf_max_iter", type=int, default=1000, help="Máximo de iteraciones BFGS de CASSCF.")
    p.add_argument("--casscf_grad_tol", type=float, default=1e-6, help="Tolerancia de gradiente BFGS.")
    p.add_argument("--casscf_energy_tol", type=float, default=1e-8, help="Tolerancia de energía entre pasos BFGS.")
    p.add_argument("--outdir", type=str, required=True, help="Carpeta de salida para este sistema CASSCF.")
    p.add_argument(
        "--master_csv",
        type=str,
        default=None,
        help="Ruta opcional a la tabla CSV maestra donde agregar los resultados conforme terminan.",
    )
    p.add_argument(
        "--csv_collect_dir",
        type=str,
        default="results_csv",
        help="Carpeta central ligera donde se copian todos los archivos CSV con referencia al sistema.",
    )
    p.add_argument(
        "--reuse_existing",
        action="store_true",
        help="Si ya existe casscf_state.npz en --outdir, reutilizarlo sin recalcular BFGS.",
    )
    p.add_argument(
        "--copy_from_advisor_if_available",
        action="store_true",
        help="Si existe un modelo CASSCF idéntico ya convergido en uhf_1d_ho_v270826_m-determinants/results_*, copiarlo.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    n_up = args.n_up if args.n_up is not None else (args.n_elec // 2)
    n_dn = args.n_dn if args.n_dn is not None else (args.n_elec - n_up)
    n_elec = n_up + n_dn

    params = get_advisor_safe_casscf_defaults(
        n_up=n_up,
        n_dn=n_dn,
        omega=args.omega,
        g=args.g,
        softening_override=args.softening,
    )
    if args.n_basis is not None:
        params["n_basis"] = int(args.n_basis)
    if args.quadrature_order is not None:
        params["quadrature_order"] = int(args.quadrature_order)
    if args.n_core is not None:
        params["n_core"] = int(args.n_core)
    if args.n_active_orbitals is not None:
        params["n_active_orbitals"] = int(args.n_active_orbitals)
    if args.n_final_determinants is not None:
        params["n_final_determinants"] = int(args.n_final_determinants)
    params["casscf_max_iter"] = int(args.casscf_max_iter)
    params["orbital_gradient_tol"] = float(args.casscf_grad_tol)
    params["casscf_energy_tol"] = float(args.casscf_energy_tol)
    params["outdir"] = str(args.outdir)

    sys_tag = f"N{n_elec}_w{params['omega']}_g{params['g']}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_target = outdir / "casscf_state.npz"
    npz_tagged = outdir / f"casscf_state_{sys_tag}.npz"
    summary_target = outdir / "summary.json"
    summary_tagged = outdir / f"summary_casscf_{sys_tag}.json"
    csv_collect_dir = Path(args.csv_collect_dir) if args.csv_collect_dir else None
    if csv_collect_dir is not None:
        csv_collect_dir.mkdir(parents=True, exist_ok=True)

    n_act_up = params["n_up"] - params["n_core"]
    n_act_dn = params["n_down"] - params["n_core"]
    expected_n_dets = math.comb(params["n_active_orbitals"], n_act_up) * math.comb(
        params["n_active_orbitals"], n_act_dn
    )

    print("=" * 78)
    print(
        f"[Pipeline 1 - CASSCF] Sistema: {sys_tag} (up={n_up}, dn={n_dn}) | "
        f"omega={params['omega']} | g={params['g']} | a={params['softening']}"
    )
    print(
        f"[Pipeline 1 - CASSCF] Espacio Activo: n_core={params['n_core']}, "
        f"n_active_orbitals={params['n_active_orbitals']} -> {expected_n_dets} determinantes CI | "
        f"n_basis={params['n_basis']}, quad={params['quadrature_order']}, max_iter={params['casscf_max_iter']}"
    )
    print("=" * 78)

    t0 = time.perf_counter()
    reused = False

    # 1. Reutilizar si ya existe en outdir, se solicitó --reuse_existing y convergió genuinamente con base suficiente
    if args.reuse_existing and npz_target.exists() and summary_target.exists():
        try:
            with summary_target.open("r", encoding="utf-8") as f_chk:
                prev_sum = json.load(f_chk)
            prev_conv = bool(prev_sum.get("converged", False))
            prev_it = int(prev_sum.get("iterations", 0))
            prev_msg = str(prev_sum.get("optimizer_message", ""))
            prev_cfg = prev_sum.get("config", {})
            prev_max_it = int(prev_cfg.get("casscf_max_iter", prev_it))
            prev_nbasis = int(prev_cfg.get("n_basis", 0))
            prev_ncore = int(prev_cfg.get("n_core", -1))
            prev_nact = int(prev_cfg.get("n_active_orbitals", -1))
            hit_old_cap = (prev_it >= 120 and prev_max_it <= 120) or ("Maximum number of iterations" in prev_msg and prev_max_it < params["casscf_max_iter"])
            basis_ok = (prev_nbasis >= int(params["n_basis"])) and (prev_ncore == int(params["n_core"])) and (prev_nact == int(params["n_active_orbitals"]))
            if prev_conv and not hit_old_cap and basis_ok:
                print(f"[Pipeline 1 - CASSCF] Reutilizando estado convergido (b={prev_nbasis}, {prev_it} iters) en: {npz_target}")
                reused = True
            else:
                print(
                    f"[Pipeline 1 - CASSCF] Estado previo en {outdir} requiere recálculo "
                    f"(b={prev_nbasis} vs {params['n_basis']}, it={prev_it}/{prev_max_it}, conv={prev_conv}). Recalculando..."
                )
        except Exception:
            pass

    # 2. Copiar de uhf_1d_ho_v270826_m-determinants/results_* si se solicitó, coincide exactamente, tiene base >= n_basis y convergió
    if not reused and args.copy_from_advisor_if_available:
        baselines = discover_casscf_baselines(search_root=ROOT_DIR)
        match = find_matching_casscf_baseline(
            baselines,
            n_up=n_up,
            n_dn=n_dn,
            omega=params["omega"],
            g=params["g"],
            softening=params["softening"],
        )
        if match is not None and os.path.abspath(match["npz_path"]) != os.path.abspath(npz_target):
            src_dir = Path(match["npz_path"]).parent
            src_sum_file = src_dir / "summary.json"
            ok_to_copy = int(match.get("n_basis", 0)) >= int(params["n_basis"])
            if ok_to_copy and src_sum_file.exists():
                try:
                    with src_sum_file.open("r", encoding="utf-8") as fs:
                        s_meta = json.load(fs)
                    s_it = int(s_meta.get("iterations", 0))
                    s_cfg = s_meta.get("config", {})
                    s_max = int(s_cfg.get("casscf_max_iter", 500))
                    s_nbasis = int(s_cfg.get("n_basis", match.get("n_basis", 0)))
                    s_ncore = int(s_cfg.get("n_core", match.get("n_core", -1)))
                    s_nact = int(s_cfg.get("n_active_orbitals", match.get("n_active", -1)))
                    s_conv = bool(s_meta.get("converged", True))
                    if (not s_conv) or (s_it >= 120 and s_max <= 120) or (s_nbasis < int(params["n_basis"])) or (s_ncore != int(params["n_core"])) or (s_nact != int(params["n_active_orbitals"])):
                        ok_to_copy = False
                except Exception:
                    ok_to_copy = False
            if ok_to_copy:
                print(f"[Pipeline 1 - CASSCF] Copiando referencia ya convergida (b={match.get('n_basis')}) desde: {src_dir}")
                for item in src_dir.iterdir():
                    if item.is_file():
                        atomic_copy_file(item, outdir / item.name)
                reused = True

    # 3. Ejecutar cálculo variacional CASSCF desde cero usando la paquetería del asesor
    if not reused:
        import dataclasses
        from uhf1d.config import CASSCFConfig
        from uhf1d.casscf import run_casscf
        from uhf1d.plotting import make_casscf_plots

        valid_fields = {f.name for f in dataclasses.fields(CASSCFConfig)}
        casscf_kwargs = {k: v for k, v in params.items() if k in valid_fields}
        cfg = CASSCFConfig(**casscf_kwargs)
        result = run_casscf(cfg)
        result.save(npz_target)
        try:
            make_casscf_plots(result, outdir)
        except Exception as exc_p:
            print(f"  [Aviso CASSCF Plots] No se pudieron generar algunas figuras: {exc_p}")

        if result.history and "energy" in result.history and len(result.history["energy"]) > 0:
            hist_df = pd.DataFrame(
                {
                    "system": sys_tag,
                    "iteration": result.history.get("iteration", list(range(1, len(result.history["energy"]) + 1))),
                    "energy": result.history["energy"],
                }
            )
            hist_csv = outdir / f"casscf_convergence_history_{sys_tag}.csv"
            atomic_write_csv(hist_df, hist_csv, index=False)
            if csv_collect_dir is not None:
                atomic_copy_file(hist_csv, csv_collect_dir / hist_csv.name)

        initial_energy = (
            float(result.history["energy"][0])
            if (result.history and "energy" in result.history and len(result.history["energy"]) > 0)
            else float("nan")
        )
        summary_dict = {
            "system_tag": sys_tag,
            "config": result.config.to_dict(),
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "energy": float(result.energy),
            "core_energy": float(result.core_energy),
            "active_energy": float(result.active_energy),
            "initial_energy": initial_energy,
            "optimizer_message": str(result.optimizer_message),
            "n_determinants": int(len(result.ci_coefficients)),
        }
    else:
        with summary_target.open("r", encoding="utf-8") as f:
            summary_dict = json.load(f)
        summary_dict["system_tag"] = sys_tag

    wall_time = time.perf_counter() - t0

    # Copiar archivo .npz y figuras con nombre referenciado al sistema
    if npz_target.exists():
        atomic_copy_file(npz_target, npz_tagged)
    for fig_stem in ("densities", "ci_weights", "natural_occupations", "orbital_4panel", "energy_history"):
        src_fig = outdir / f"{fig_stem}.png"
        if src_fig.exists():
            atomic_copy_file(src_fig, outdir / f"{fig_stem}_{sys_tag}.png")

    # Evaluar densidad exacta de 1 cuerpo en malla uniforme y exportar CSV con referencia al sistema
    ref_density = compute_exact_casscf_1body_density(
        npz_target,
        x_min=-params["plot_xmax"],
        x_max=params["plot_xmax"],
        n_grid=params["plot_points"],
    )
    density_csv_path = outdir / "casscf_exact_density.csv"
    density_csv_tagged = outdir / f"casscf_exact_density_{sys_tag}.csv"
    df_dens = pd.DataFrame(
        {
            "system": sys_tag,
            "x": ref_density["x"],
            "rho_total": ref_density["rho_total"],
            "rho_up": ref_density["rho_up"],
            "rho_down": ref_density["rho_down"],
            "p_total": ref_density["p_total"],
            "p_up": ref_density["p_up"],
            "p_down": ref_density["p_down"],
        }
    )
    atomic_write_csv(df_dens, density_csv_path, index=False)
    atomic_write_csv(df_dens, density_csv_tagged, index=False)
    if csv_collect_dir is not None:
        atomic_write_csv(df_dens, csv_collect_dir / f"casscf_exact_density_{sys_tag}.csv", index=False)

    # Calcular radio RMS analítico sobre la densidad exacta
    x_grid = ref_density["x"]
    p_tot = ref_density["p_total"]
    dx = float(x_grid[1] - x_grid[0])
    rms_radius = float(np.sqrt(np.sum((x_grid**2) * p_tot) * dx))

    # Guardar summary.json y summary_casscf_{sys_tag}.json enriquecidos en la carpeta del modelo
    summary_dict["pipeline_stage"] = "pipeline1_casscf_reference"
    summary_dict["wall_time_sec"] = float(wall_time)
    summary_dict["reused_existing"] = bool(reused)
    summary_dict["npz_path"] = str(npz_tagged)
    summary_dict["exact_density_csv"] = str(density_csv_tagged)
    summary_dict["observables"] = {
        "energy": float(summary_dict.get("energy", ref_density["energy"])),
        "initial_energy": float(summary_dict.get("initial_energy", float("nan"))),
        "correlation_energy_gain": float(
            summary_dict.get("energy", ref_density["energy"])
            - summary_dict.get("initial_energy", float("nan"))
        ),
        "rms_radius": rms_radius,
        "n_determinants": int(summary_dict.get("n_determinants", expected_n_dets)),
    }

    atomic_write_json(summary_target, summary_dict, indent=2)
    atomic_write_json(summary_tagged, summary_dict, indent=2)

    print(
        f"[Pipeline 1 - CASSCF] Completado en {wall_time:.2f}s | "
        f"E_CASSCF = {summary_dict['observables']['energy']:.8f} Ha | "
        f"Guardado en: {outdir} ({sys_tag})"
    )

    row = {
        "system_tag": sys_tag,
        "pipeline_stage": "pipeline1_casscf_reference",
        "N": int(n_elec),
        "n_up": int(n_up),
        "n_dn": int(n_dn),
        "omega": float(params["omega"]),
        "g": float(params["g"]),
        "softening": float(params["softening"]),
        "n_core": int(params["n_core"]),
        "n_active_orbitals": int(params["n_active_orbitals"]),
        "n_basis": int(params["n_basis"]),
        "quadrature_order": int(params["quadrature_order"]),
        "n_determinants": int(summary_dict["observables"]["n_determinants"]),
        "converged": bool(summary_dict.get("converged", True)),
        "iterations": int(summary_dict.get("iterations", 0)),
        "casscf_energy": float(summary_dict["observables"]["energy"]),
        "initial_energy": float(summary_dict["observables"]["initial_energy"]),
        "correlation_energy_gain": float(summary_dict["observables"]["correlation_energy_gain"]),
        "core_energy": float(summary_dict.get("core_energy", float("nan"))),
        "active_energy": float(summary_dict.get("active_energy", float("nan"))),
        "rms_radius": rms_radius,
        "wall_time_sec": float(wall_time),
        "npz_path": str(npz_tagged),
        "outdir": str(outdir),
    }
    if args.master_csv:
        _append_master_csv(args.master_csv, row, collect_dir=csv_collect_dir)
        print(f"[Pipeline 1 - CASSCF] Fila agregada en vivo a: {args.master_csv}")
    if csv_collect_dir is not None:
        central_p1_csv = csv_collect_dir / "live_pipeline1_casscf_summary.csv"
        if not args.master_csv or Path(args.master_csv).name != central_p1_csv.name:
            _append_master_csv(central_p1_csv, row)


if __name__ == "__main__":
    main()
