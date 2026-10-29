#!/usr/bin/env python3
"""
CLI Unificado para el Pipeline 1D de Variational Monte Carlo (VMC) y RL-MALA.
Ejecuta `python cli/run_1d.py --help` para consultar todas las banderas y valores default.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rl_vmc.config import parse_experiment_args
from rl_vmc.trainers.runner_1d import run_experiment_1d


def main(argv: list[str] | None = None) -> None:
    cfg = parse_experiment_args(argv)
    run_experiment_1d(cfg)


if __name__ == "__main__":
    main()
