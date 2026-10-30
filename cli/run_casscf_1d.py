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
import fcntl
import json
import math
import os
import shutil
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
        n_basis = 24
        quadrature_order = 80
        default_softening = 0.1
    elif n_elec == 4:
        n_core = 0
        # En N=4, el asesor utilizó n_active_orbitals=4 (36 dets) para g<=2
        # y n_active_orbitals=5 (100 dets) para g>=5.
        n_active_orbitals = 5 if g >= 4.0 else 4
        n_basis = 32
        quadrature_order = 80
        default_softening = 0.1
    elif n_elec == 6:
        # En N=6 (3 up, 3 down), congelar n_core=1 deja 2 up y 2 down activos -> 36 dets
        n_core = 1
        n_active_orbitals = 4
        n_basis = 32
        quadrature_order = 80
        default_softening = 0.1
    elif n_elec == 8:
        # En N=8 (4 up, 4 down), congelar n_core=2 deja 2 up y 2 down activos -> 36 dets
        n_core = 2
        n_active_orbitals = 4
        n_basis = 28
        quadrature_order = 80
        default_softening = 0.1
    elif n_elec == 10:
        # En N=10 (5 up, 5 down), el asesor utilizó n_core=4, n_active_orbitals=4 (16 dets)
        # y n_basis=24 con quadrature_order=80.
        n_core = 4
        n_active_orbitals = 4
        n_basis = 24
        quadrature_order = 80
        default_softening = 0.25
    else:
        n_core = max(0, min(n_up, n_dn) - 2)
        n_active_orbitals = 4
        n_basis = 24
        quadrature_order = 80
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
        "casscf_max_iter": 120,
        "casscf_energy_tol": 1e-4,
        "orbital_gradient_tol": 1e-6,
        "max_ci_dimension": 20000,
        "plot_xmax": float(plot_xmax),
        "plot_points": 500,
    }


def _append_master_csv(csv_path: str | Path, row: dict[str, Any]) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_file.with_suffix(".csv.lock")
    with lock_path.open("w") as f_lock:
        fcntl.flock(f_lock, fcntl.LOCK_EX)
        try:
            df = pd.DataFrame([row])
            write_header = not csv_file.exists() or csv_file.stat().st_size == 0
            df.to_csv(csv_file, mode="a", header=write_header, index=False)
        finally:
            fcntl.flock(f_lock, fcntl.LOCK_UN)


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
    p.add_argument("--casscf_max_iter", type=int, default=120, help="Máximo de iteraciones BFGS de CASSCF.")
    p.add_argument("--casscf_grad_tol", type=float, default=1e-6, help="Tolerancia de gradiente BFGS.")
    p.add_argument("--outdir", type=str, required=True, help="Carpeta de salida para este sistema CASSCF.")
    p.add_argument(
        "--master_csv",
        type=str,
        default=None,
        help="Ruta opcional a la tabla CSV maestra donde agregar los resultados conforme terminan.",
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
    params["casscf_grad_tol"] = float(args.casscf_grad_tol)
    params["outdir"] = str(args.outdir)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz_target = outdir / "casscf_state.npz"
    summary_target = outdir / "summary.json"

    n_act_up = params["n_up"] - params["n_core"]
    n_act_dn = params["n_down"] - params["n_core"]
    expected_n_dets = math.comb(params["n_active_orbitals"], n_act_up) * math.comb(
        params["n_active_orbitals"], n_act_dn
    )

    print("=" * 78)
    print(
        f"[Pipeline 1 - CASSCF] Sistema: N={n_elec} (up={n_up}, dn={n_dn}) | "
        f"omega={params['omega']} | g={params['g']} | a={params['softening']}"
    )
    print(
        f"[Pipeline 1 - CASSCF] Espacio Activo: n_core={params['n_core']}, "
        f"n_active_orbitals={params['n_active_orbitals']} -> {expected_n_dets} determinantes CI | "
        f"n_basis={params['n_basis']}, quad={params['quadrature_order']}"
    )
    print("=" * 78)

    t0 = time.perf_counter()
    reused = False

    # 1. Reutilizar si ya existe en outdir y se solicitó --reuse_existing
    if args.reuse_existing and npz_target.exists() and summary_target.exists():
        print(f"[Pipeline 1 - CASSCF] Reutilizando estado existente en: {npz_target}")
        reused = True

    # 2. Copiar de uhf_1d_ho_v270826_m-determinants/results_* si se solicitó y coincide exactamente
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
            print(f"[Pipeline 1 - CASSCF] Copiando referencia ya convergida desde: {src_dir}")
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, outdir / item.name)
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

        initial_energy = (
            float(result.history["energy"][0])
            if (result.history and "energy" in result.history and len(result.history["energy"]) > 0)
            else float("nan")
        )
        summary_dict = {
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

    wall_time = time.perf_counter() - t0

    # Evaluar densidad exacta de 1 cuerpo en malla uniforme y exportar CSV
    ref_density = compute_exact_casscf_1body_density(
        npz_target,
        x_min=-params["plot_xmax"],
        x_max=params["plot_xmax"],
        n_grid=params["plot_points"],
    )
    density_csv_path = outdir / "casscf_exact_density.csv"
    pd.DataFrame(
        {
            "x": ref_density["x"],
            "rho_total": ref_density["rho_total"],
            "rho_up": ref_density["rho_up"],
            "rho_down": ref_density["rho_down"],
            "p_total": ref_density["p_total"],
            "p_up": ref_density["p_up"],
            "p_down": ref_density["p_down"],
        }
    ).to_csv(density_csv_path, index=False)

    # Calcular radio RMS analítico e identificación de picos de Wigner sobre la densidad exacta
    x_grid = ref_density["x"]
    p_tot = ref_density["p_total"]
    dx = float(x_grid[1] - x_grid[0])
    rms_radius = float(np.sqrt(np.sum((x_grid**2) * p_tot) * dx))

    # Guardar summary.json enriquecido en la carpeta del modelo
    summary_dict["pipeline_stage"] = "pipeline1_casscf_reference"
    summary_dict["wall_time_sec"] = float(wall_time)
    summary_dict["reused_existing"] = bool(reused)
    summary_dict["npz_path"] = str(npz_target)
    summary_dict["exact_density_csv"] = str(density_csv_path)
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

    with summary_target.open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(
        f"[Pipeline 1 - CASSCF] Completado en {wall_time:.2f}s | "
        f"E_CASSCF = {summary_dict['observables']['energy']:.8f} Ha | "
        f"Guardado en: {outdir}"
    )

    if args.master_csv:
        row = {
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
            "npz_path": str(npz_target),
            "outdir": str(outdir),
        }
        _append_master_csv(args.master_csv, row)
        print(f"[Pipeline 1 - CASSCF] Fila agregada en vivo a: {args.master_csv}")


if __name__ == "__main__":
    main()
