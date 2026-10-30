#!/usr/bin/env python3
"""
CLI de entrenamiento y evaluación 1D (Alias para cli/run_1d.py).
Permite invocar `python3 cli/train_1d.py` o `python3 cli/run_1d.py` indistintamente.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cli.run_1d import main

if __name__ == "__main__":
    main()
