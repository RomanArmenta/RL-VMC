import numpy as np


def ho_basis(x, n_basis, omega):
    """Normalized HO functions using overflow-safe 3-term recurrence; output shape is (len(x), n_basis)."""
    x = np.asarray(x, dtype=float)
    z = np.sqrt(omega) * x
    pref = (omega / np.pi) ** 0.25 * np.exp(-0.5 * z * z)
    out = np.empty((x.size, n_basis), dtype=float)
    if n_basis > 0:
        out[:, 0] = pref
    if n_basis > 1:
        out[:, 1] = np.sqrt(2.0) * z * out[:, 0]
    for n in range(2, n_basis):
        out[:, n] = np.sqrt(2.0 / n) * z * out[:, n - 1] - np.sqrt((n - 1.0) / n) * out[:, n - 2]
    return out


