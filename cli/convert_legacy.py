#!/usr/bin/env python3
"""
CLI Convertidor Bidireccional Legacy (<->) HDF5 Unificado (`run_bundle.h5`).
Permite convertir cualquier carpeta de experimentos previa (`params.json`, `*.pkl`, `*.csv`)
al nuevo formato `.h5` unificado, y opcionalmente correr un muestreo de producción sobre
esos pesos convertidos para analizarlos en `cli/postprocess.py`.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_vmc.core.io_bundle import (
    convert_legacy_dir_to_h5,
    export_h5_to_legacy_dir,
    load_run_bundle_h5,
)
from rl_vmc.config import Experiment1DConfig
from rl_vmc.trainers.runner_1d import run_experiment_1d


def build_converter_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convertidor entre carpetas legacy (params.json, *.pkl, *.csv) y el formato unificado HDF5 (.h5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--legacy_dir",
        type=str,
        default=None,
        help="Ruta a una carpeta legacy individual para convertir a .h5.",
    )
    p.add_argument(
        "--output_h5",
        type=str,
        default=None,
        help="Ruta opcional de salida para el archivo .h5 convertido.",
    )
    p.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["MH", "MALA", "HMC", "TD3", "REINFORCE"],
        help="Método explícito si no se desea autodetección desde los archivos de la carpeta.",
    )
    p.add_argument(
        "--batch_root",
        type=str,
        default=None,
        help="Escanea recursivamente este directorio y convierte todas las carpetas legacy encontradas a .h5.",
    )
    p.add_argument(
        "--export_h5",
        type=str,
        default=None,
        help="Ruta a un archivo .h5 unificado para exportar de vuelta a archivos legacy (.json, .pkl, .csv).",
    )
    p.add_argument(
        "--export_dir",
        type=str,
        default=None,
        help="Carpeta destino al usar --export_h5.",
    )
    p.add_argument(
        "--resample_production",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tras convertir los pesos legacy a .h5, corre una cadena de producción con pesos congelados para poblar /production/walkers y /production/local_energy.",
    )
    p.add_argument(
        "--measure_steps",
        type=int,
        default=2000,
        help="Pasos de producción si se activa --resample_production.",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Número de caminantes si se activa --resample_production.",
    )
    return p


def _resample_converted_h5(h5_path: str, measure_steps: int, batch_size: int) -> None:
    bundle = load_run_bundle_h5(h5_path, load_walkers=False)
    cfg = Experiment1DConfig.from_dict(bundle["config"])
    cfg.train.mode = "eval_only"
    cfg.io.load_bundle_h5 = str(h5_path)
    cfg.io.output_h5 = str(h5_path)
    cfg.sampler.batch_size = batch_size
    cfg.eval.run_evaluation = True
    cfg.eval.eval_burn_in = min(500, measure_steps // 2)
    cfg.eval.measure_steps = measure_steps
    run_experiment_1d(cfg)


def main(argv: list[str] | None = None) -> None:
    parser = build_converter_argparser()
    args = parser.parse_args(argv)

    if args.export_h5:
        out_dir = args.export_dir or os.path.dirname(os.path.abspath(args.export_h5))
        export_h5_to_legacy_dir(args.export_h5, out_dir)
        print(f"[Convertidor] Exportado {args.export_h5} -> {out_dir}")
        return

    if args.legacy_dir:
        h5_path = convert_legacy_dir_to_h5(
            args.legacy_dir, output_h5_path=args.output_h5, method=args.method
        )
        print(f"[Convertidor] Carpeta legacy '{args.legacy_dir}' convertida -> '{h5_path}'")
        if args.resample_production:
            _resample_converted_h5(str(h5_path), args.measure_steps, args.batch_size)
        return

    if args.batch_root:
        converted = []
        for root, _, files in os.walk(args.batch_root):
            fset = set(files)
            has_legacy = any(
                k in fset or k.endswith("_weights.pkl")
                for k in fset
                if k in ("params.json", "training_metrics.csv") or k.endswith("_weights.pkl")
            )
            if has_legacy:
                h5_path = convert_legacy_dir_to_h5(root, method=args.method)
                converted.append(str(h5_path))
                print(f"[Convertidor] '{root}' -> '{h5_path}'")
                if args.resample_production:
                    _resample_converted_h5(str(h5_path), args.measure_steps, args.batch_size)
        print(f"[Convertidor] Total de carpetas legacy convertidas: {len(converted)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
