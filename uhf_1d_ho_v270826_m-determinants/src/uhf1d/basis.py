import numpy as np
from scipy.special import eval_hermite, factorial


def ho_basis(x, n_basis, omega):
    """Normalized HO functions; output shape is (len(x), n_basis)."""
    x = np.asarray(x, dtype=float)
    z = np.sqrt(omega) * x
    pref = (omega / np.pi) ** 0.25 * np.exp(-0.5 * z * z)
    out = np.empty((x.size, n_basis))
    for n in range(n_basis):
        out[:, n] = pref * eval_hermite(n, z) / np.sqrt(2.0**n * factorial(n))
    return out

