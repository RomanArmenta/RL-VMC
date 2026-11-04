from __future__ import annotations
import numpy as np
from scipy.special import eval_hermite, gammaln


def ho_basis(x: np.ndarray, n_basis: int, omega: float) -> np.ndarray:
    """Return normalized 1D harmonic-oscillator basis functions.

    Output shape is (n_x, n_basis), with mass=hbar=1.
    """
    x = np.asarray(x, dtype=float)
    if omega <= 0:
        raise ValueError("omega must be positive")
    z = np.sqrt(omega) * x
    gauss = np.exp(-0.5 * z * z)
    out = np.empty((x.size, n_basis), dtype=float)
    for n in range(n_basis):
        log_norm = 0.25 * np.log(omega / np.pi) - 0.5 * (n * np.log(2.0) + gammaln(n + 1.0))
        out[:, n] = np.exp(log_norm) * eval_hermite(n, z) * gauss
    return out


def orbitals_from_coefficients(x: np.ndarray, coeff: np.ndarray, basis_omega: float) -> np.ndarray:
    """Evaluate orbitals phi_p(x)=sum_n C[n,p] chi_n(x)."""
    coeff = np.asarray(coeff, dtype=float)
    if coeff.ndim != 2:
        raise ValueError("orbital coefficient matrix must have shape (n_basis, n_orbitals)")
    return ho_basis(np.asarray(x), coeff.shape[0], basis_omega) @ coeff
