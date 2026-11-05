import numpy as np
from scipy.special import roots_hermite


def build_integrals(cfg):
    """Return overlap S, one-body h, and chemists' ERIs (pq|rs)."""
    n, nq, w0 = cfg.n_basis, cfg.quadrature_order, cfg.omega
    t, weights = roots_hermite(nq)
    x = t / np.sqrt(w0)
    polys = np.empty((n, t.size), dtype=float)
    if n > 0:
        polys[0] = 1.0
    if n > 1:
        polys[1] = np.sqrt(2.0) * t
    for k in range(2, n):
        polys[k] = np.sqrt(2.0 / k) * t * polys[k - 1] - np.sqrt((k - 1.0) / k) * polys[k - 2]
    # phi_p phi_q dx = exp(-t^2) * polys[p] polys[q] / sqrt(pi) dt
    products = np.einsum("pi,qi->pqi", polys, polys) / np.sqrt(np.pi)
    s = np.einsum("i,pqi->pq", weights, products)
    h = np.diag((np.arange(n) + 0.5) * w0)
    distance = x[:, None] - x[None, :]
    kernel = cfg.g / np.sqrt(distance * distance + cfg.softening**2)
    eri = np.einsum(
        "i,j,pqi,rsj,ij->pqrs", weights, weights, products, products, kernel,
        optimize=True,
    )
    return s, h, eri

