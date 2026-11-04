"""Wigner-regime post-processing for 1D trapped-electron calculations."""

from .models import SystemInfo, GridConfig
from .classical import classical_equilibrium_positions
from .timeseries import summarize_multiwalker
from .structure import (
    sample_one_body_density,
    sample_pair_distributions,
    conditional_density_from_samples,
    localization_from_samples,
)

__all__ = [
    "SystemInfo",
    "GridConfig",
    "classical_equilibrium_positions",
    "summarize_multiwalker",
    "sample_one_body_density",
    "sample_pair_distributions",
    "conditional_density_from_samples",
    "localization_from_samples",
]

__version__ = "0.1.4"
