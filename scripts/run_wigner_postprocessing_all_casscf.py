#!/usr/bin/env python3
"""
scripts/run_wigner_postprocessing_all_casscf.py
==============================================================================
Revisa todos los estados CASSCF generados por el benchmark de 3 pipelines
(o existentes en uhf_1d_ho_v270826_m-determinants / results_production_3pipelines)
y ejecuta el análisis y post-procesamiento de Wigner del paquete del asesor
(`wigner_post.analysis.analyze_orbital_state`) sobre cada uno.

Calcula y extrae las densidades analíticas exactas completas de CASSCF:
  - one_body_density.csv (x, rho_total, rho_up, rho_down)
  - pair_distance_distribution.csv (r, p_total, p_uu, p_dd, p_ud)
  - analysis_summary.json (sitios de Wigner, Lindemann ratio, picos de densidad)
  - Figuras: plots/density_and_pin.png, plots/pair_distributions.png

Uso:
  python3 scripts/run_wigner_postprocessing_all_casscf.py
  python3 scripts/run_wigner_postprocessing_all_casscf.py --search_dirs results_production_3pipelines_b2 uhf_1d_ho_v270826_m-determinants
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Añadir directorios de búsqueda de módulos
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "wigner_post"))

# Rutas candidatas para importar wigner_post
CANDIDATE_WIGNER_DIRS = [
    str(ROOT_DIR / "wigner_post"),
    str(ROOT_DIR / "wigner_postprocessing_v014"),
    str(ROOT_DIR.parent / "wigner_postprocessing_v014"),
    str(ROOT_DIR.parent.parent / "wigner_postprocessing_v014"),
    "/home/romanarmenta/Documents/projects/wigner_postprocessing_v014",
    "/Users/romanarmenta/Downloads/Archive/wigner_postprocessing_v014",
]
for d in CANDIDATE_WIGNER_DIRS:
    if os.path.isdir(d) and d not in sys.path:
        sys.path.insert(0, d)

# Intento de importar wigner_post
HAS_WIGNER_POST = False
try:
    from wigner_post.analysis import analyze_orbital_state
    from wigner_post.models import GridConfig
    HAS_WIGNER_POST = True
except ImportError:
    HAS_WIGNER_POST = False


def extract_casscf_metadata(npz_path: Path) -> Dict[str, Any]:
    """Lee metadatos físicos y de convergencia de un archivo casscf_state.npz."""
    import numpy as np
    data = np.load(npz_path, allow_pickle=False)
    meta = {}
    if "metadata_json" in data:
        try:
            meta = json.loads(str(data["metadata_json"]))
        except Exception:
            meta = {}

    cfg = meta.get("config", {})
    # Fallback si metadata_json no tiene config completa
    sibling_json = npz_path.parent / "summary.json"
    if sibling_json.exists() and not cfg:
        try:
            with open(sibling_json, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                cfg = s_data.get("config", s_data)
        except Exception:
            pass

    n_up = int(cfg.get("n_up", cfg.get("n_alpha", 1)))
    n_down = int(cfg.get("n_down", cfg.get("n_dn", cfg.get("n_beta", 1))))
    omega = float(cfg.get("omega", 0.1))
    g = float(cfg.get("g", cfg.get("interaction_strength", 1.0)))
    softening = float(cfg.get("softening", cfg.get("a", 0.1)))
    n_core = int(cfg.get("n_core", 0))
    n_active = int(cfg.get("n_active_orbitals", cfg.get("n_active", 4)))
    n_basis = int(cfg.get("n_basis", 32))
    energy = float(meta.get("energy", cfg.get("energy", float("nan"))))

    sys_tag = f"N{n_up + n_down}_w{omega}_g{g}"
    return {
        "npz_path": npz_path,
        "sys_tag": sys_tag,
        "n_elec": n_up + n_down,
        "n_up": n_up,
        "n_down": n_down,
        "omega": omega,
        "g": g,
        "softening": softening,
        "n_core": n_core,
        "n_active": n_active,
        "n_basis": n_basis,
        "energy": energy,
    }


def find_all_casscf_states(search_roots: List[str]) -> List[Path]:
    """Encuentra todos los archivos casscf_state.npz únicos en las carpetas de búsqueda."""
    found: List[Path] = []
    seen: set = set()

    for r in search_roots:
        p = Path(r)
        if not p.exists():
            continue
        for npz in p.rglob("casscf_state*.npz"):
            resolved = npz.resolve()
            if resolved not in seen:
                seen.add(resolved)
                found.append(npz)

    return sorted(found)


def fallback_analytical_density(npz_meta: Dict[str, Any], out_dir: Path) -> Path:
    """Fallback analítico usando rl_vmc si wigner_post no está disponible en el entorno."""
    import numpy as np
    import pandas as pd
    from rl_vmc.postprocess.casscf_baseline import load_casscf_reference_from_npz

    ref = load_casscf_reference_from_npz(str(npz_meta["npz_path"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame({
        "x": ref["x"],
        "rho_total": ref["rho_total"],
        "rho_up": ref["rho_up"],
        "rho_down": ref["rho_down"],
        "p_total": ref["p_total"],
        "p_up": ref["p_up"],
        "p_down": ref["p_down"],
    })
    csv_path = out_dir / "one_body_density.csv"
    df_out.to_csv(csv_path, index=False)
    return csv_path


def process_single_casscf(
    meta: Dict[str, Any],
    base_out_dir: Path,
    csv_collect_dir: Optional[Path] = None,
    grid_cfg: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ejecuta el post-procesamiento de Wigner sobre un archivo casscf_state.npz."""
    npz_path = meta["npz_path"]
    sys_tag = meta["sys_tag"]
    parent_folder = npz_path.parent.name
    out_dir = base_out_dir / f"{sys_tag}_{parent_folder}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    one_body_csv = out_dir / "one_body_density.csv"
    pair_dist_csv = out_dir / "pair_distance_distribution.csv"

    overrides = {
        "n_up": meta["n_up"],
        "n_down": meta["n_down"],
        "omega": meta["omega"],
        "g": meta["g"],
        "a": meta["softening"],
    }

    status = "OK"
    if HAS_WIGNER_POST:
        try:
            analyze_orbital_state(
                path=str(npz_path),
                outdir=str(out_dir),
                method="casscf",
                grid=grid_cfg or GridConfig(x_min=-15.0, x_max=15.0, n_grid=500),
                pin_x=None,
                overrides=overrides,
            )
        except Exception as e:
            status = f"ERROR_WIGNER_POST: {e}"
            # Intentar fallback analítico
            try:
                fallback_analytical_density(meta, out_dir)
                status += " (Fallback analítico generado)"
            except Exception as e2:
                status += f" (Fallback failed: {e2})"
    else:
        # Fallback analítico directo
        try:
            fallback_analytical_density(meta, out_dir)
            status = "FALLBACK_ANALYTICAL"
        except Exception as e:
            status = f"ERROR_FALLBACK: {e}"

    elapsed = time.time() - t0

    # Copiar densidad centralizada con nombre del sistema físico
    if one_body_csv.exists() and csv_collect_dir:
        csv_collect_dir.mkdir(parents=True, exist_ok=True)
        dest_csv = csv_collect_dir / f"casscf_exact_density_{sys_tag}.csv"
        shutil.copy2(one_body_csv, dest_csv)

    return {
        "sys_tag": sys_tag,
        "n_elec": meta["n_elec"],
        "omega": meta["omega"],
        "g": meta["g"],
        "softening": meta["softening"],
        "energy": meta["energy"],
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "status": status,
        "elapsed_sec": round(elapsed, 3),
        "has_density_csv": one_body_csv.exists(),
        "has_pair_csv": pair_dist_csv.exists(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta post-procesamiento de Wigner sobre todos los CASSCF del productor de 3 pipelines."
    )
    parser.add_argument(
        "--search_dirs",
        nargs="+",
        default=[
            "results_production_3pipelines_b2/pipeline1_casscf",
            "results_production_3pipelines/pipeline1_casscf",
            "uhf_1d_ho_v270826_m-determinants",
            "figures",
        ],
        help="Directorios donde buscar recursivamente archivos casscf_state.npz.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results_casscf_wigner_densities",
        help="Directorio base para guardar las densidades y figuras de Wigner.",
    )
    parser.add_argument(
        "--csv_collect_dir",
        type=str,
        default="results_csv_b2",
        help="Directorio central donde copiar los CSVs de densidad de referencia (ej. casscf_exact_density_N4_w0.1_g1.0.csv).",
    )
    parser.add_argument(
        "--x_min",
        type=float,
        default=-15.0,
        help="Límite inferior para la grilla espacial de densidad (Bohr).",
    )
    parser.add_argument(
        "--x_max",
        type=float,
        default=15.0,
        help="Límite superior para la grilla espacial de densidad (Bohr).",
    )
    parser.add_argument(
        "--n_grid",
        type=int,
        default=500,
        help="Número de puntos en la grilla espacial de densidad.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("    POST-PROCESAMIENTO DE WIGNER CASSCF EXACTO (TODOS LOS SISTEMAS)")
    print(f"    Fecha y Hora      : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    wigner_post disp. : {'SÍ (paquetería del asesor)' if HAS_WIGNER_POST else 'NO (usando fallback analítico)'}")
    print(f"    Directorios busca : {args.search_dirs}")
    print(f"    Directorio salida : {args.out_dir}")
    print(f"    CSV Collect Dir   : {args.csv_collect_dir}")
    print("=" * 80)

    npz_files = find_all_casscf_states(args.search_dirs)
    if not npz_files:
        print(f"[-] No se encontraron archivos casscf_state.npz en: {args.search_dirs}")
        return

    print(f"[+] Se encontraron {len(npz_files)} archivos CASSCF para procesar.\n")

    base_out = Path(args.out_dir)
    csv_collect = Path(args.csv_collect_dir) if args.csv_collect_dir else None
    grid_cfg = GridConfig(x_min=args.x_min, x_max=args.x_max, n_grid=args.n_grid) if HAS_WIGNER_POST else None

    results_summary = []
    for i, npz_path in enumerate(npz_files, 1):
        try:
            meta = extract_casscf_metadata(npz_path)
            print(f"[{i}/{len(npz_files)}] Procesando {meta['sys_tag']} ({npz_path.parent.name})...")
            res = process_single_casscf(meta, base_out, csv_collect, grid_cfg)
            results_summary.append(res)
            print(f"     -> Estado: {res['status']} | Tiempo: {res['elapsed_sec']}s | CSV: {res['has_density_csv']}")
        except Exception as e:
            print(f"     [-] Error procesando {npz_path}: {e}")

    # Guardar resumen global
    import pandas as pd
    df_res = pd.DataFrame(results_summary)
    summary_csv = base_out / "wigner_casscf_all_systems_summary.csv"
    base_out.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(summary_csv, index=False)

    print("\n" + "=" * 80)
    print("    POST-PROCESAMIENTO DE WIGNER COMPLETADO CON ÉXITO")
    print(f"    Sistemas Procesados : {len(results_summary)} / {len(npz_files)}")
    print(f"    Resumen Maestro CSV : {summary_csv}")
    print(f"    Densidades Copiadas : {csv_collect}")
    print("=" * 80)


if __name__ == "__main__":
    main()
