from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .models import SystemInfo


def classical_potential(x: np.ndarray, system: SystemInfo) -> float:
    x = np.asarray(x, dtype=float)
    v = 0.5 * system.omega**2 * np.dot(x, x)
    if system.g != 0.0:
        for i in range(len(x)):
            d = x[i] - x[i + 1:]
            v += system.g * np.sum(1.0 / np.sqrt(d * d + system.a**2))
    return float(v)


def classical_gradient(x: np.ndarray, system: SystemInfo) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    grad = system.omega**2 * x.copy()
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            d = x[i] - x[j]
            term = system.g * d / (d * d + system.a**2) ** 1.5
            grad[i] -= term
            grad[j] += term
    return grad


def classical_equilibrium_positions(system: SystemInfo, n_starts: int = 12, seed: int = 17):
    """Find sorted classical equilibrium positions for the finite trap."""
    n = system.n_electrons
    if n < 1:
        raise ValueError("at least one electron is required")
    if n == 1:
        return np.array([0.0]), 0.0
    rng = np.random.default_rng(seed)
    scale = max(system.oscillator_length, (max(system.g, 1e-8) / max(system.omega**2, 1e-8)) ** (1 / 3))
    base = np.linspace(-0.9 * scale * (n - 1) / 2, 0.9 * scale * (n - 1) / 2, n)
    starts = [base, np.zeros(n)]
    for _ in range(max(0, n_starts - len(starts))):
        starts.append(base + rng.normal(scale=0.25 * scale, size=n))
    best = None
    for x0 in starts:
        res = minimize(
            classical_potential,
            x0,
            args=(system,),
            jac=classical_gradient,
            method="BFGS",
            options={"gtol": 1e-11, "maxiter": 4000},
        )
        cand = np.sort(res.x)
        val = classical_potential(cand, system)
        if best is None or val < best[0]:
            best = (val, cand, res.success, res.message)
    return best[1], best[0]
