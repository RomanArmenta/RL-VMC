#!/usr/bin/env python3
"""
CLI Modular de Post-Procesamiento para Archivos Unificados HDF5 (`run_bundle.h5`).
Recibe uno o más archivos `.h5` y calcula diagnósticos MCMC, observables físicos y figuras.
Ejecuta `python cli/postprocess.py --help` para consultar todas las banderas y valores default.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_vmc.config import parse_postprocess_args
from rl_vmc.postprocess.plotter import run_postprocessing_pipeline


def main(argv: list[str] | None = None) -> None:
    cfg = parse_postprocess_args(argv)
    run_postprocessing_pipeline(cfg)


if __name__ == "__main__":
    main()
