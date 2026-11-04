from __future__ import annotations
import numpy as np
try:
    from numpy import trapezoid
except ImportError:  # NumPy < 2.0
    from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from .models import SystemInfo, GridConfig


def _as_samples(positions):
    x = np.asarray(positions, dtype=float)
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 2:
        return x
    raise ValueError("positions must have shape (time,walkers,N) or (samples,N)")


def sample_one_body_density(positions, system: SystemInfo, grid: GridConfig):
    s = _as_samples(positions)
    edges = np.linspace(grid.x_min, grid.x_max, grid.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = np.diff(edges)
    up = s[:, :system.n_up].reshape(-1)
    dn = s[:, system.n_up:].reshape(-1)
    cu = np.histogram(up, bins=edges)[0] if up.size else np.zeros(grid.n_bins)
    cd = np.histogram(dn, bins=edges)[0] if dn.size else np.zeros(grid.n_bins)
    nsamp = s.shape[0]
    rho_u = cu / (nsamp * dx) if nsamp else cu.astype(float)
    rho_d = cd / (nsamp * dx) if nsamp else cd.astype(float)
    rho = rho_u + rho_d
    p_u = rho_u / max(system.n_up, 1)
    p_d = rho_d / max(system.n_down, 1) if system.n_down else np.zeros_like(rho_d)
    p_total = rho / max(system.n_electrons, 1)
    return dict(x=centers, rho_total=rho, rho_up=rho_u, rho_down=rho_d,
                p_total=p_total, p_up=p_u, p_down=p_d)


def sample_pair_distributions(positions, system: SystemInfo, grid: GridConfig):
    s = _as_samples(positions)
    edges = np.linspace(0.0, grid.r_max, grid.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    channels = {"uu": [], "dd": [], "ud": []}
    for i in range(system.n_electrons):
        for j in range(i + 1, system.n_electrons):
            sp_i = "u" if i < system.n_up else "d"
            sp_j = "u" if j < system.n_up else "d"
            key = "uu" if sp_i == sp_j == "u" else "dd" if sp_i == sp_j == "d" else "ud"
            channels[key].append(np.abs(s[:, i] - s[:, j]))
    out = {"r": centers}
    pair_counts = {
        "uu": system.n_up * (system.n_up - 1) / 2,
        "dd": system.n_down * (system.n_down - 1) / 2,
        "ud": system.n_up * system.n_down,
    }
    for key, arrs in channels.items():
        vals = np.concatenate(arrs) if arrs else np.empty(0)
        counts = np.histogram(vals, bins=edges)[0]
        physical = counts / (s.shape[0] * dr) if s.shape[0] else counts.astype(float)
        norm = pair_counts[key]
        prob = physical / norm if norm > 0 else np.zeros_like(physical)
        out[f"pair_{key}"] = physical
        out[f"p_{key}"] = prob
        out[f"n_pairs_{key}"] = float(norm)
    return out


def conditional_density_from_samples(positions, system: SystemInfo, grid: GridConfig,
                                     x0: float, pin_spin: str = "any"):
    """Pin one electron in [x0-h,x0+h] and histogram all remaining electrons.

    Each matching pin electron contributes one conditional event. Physical conditional
    densities integrate approximately to the expected number of remaining electrons.
    """
    s = _as_samples(positions)
    if pin_spin not in ("any", "up", "down"):
        raise ValueError("pin_spin must be any/up/down")
    indices = range(system.n_electrons)
    if pin_spin == "up": indices = range(system.n_up)
    if pin_spin == "down": indices = range(system.n_up, system.n_electrons)
    edges = np.linspace(grid.x_min, grid.x_max, grid.n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:]); dx = np.diff(edges)
    cu = np.zeros(grid.n_bins); cd = np.zeros(grid.n_bins); events = 0
    for conf in s:
        for i in indices:
            if abs(conf[i] - x0) <= grid.pin_halfwidth:
                events += 1
                others_u = [conf[j] for j in range(system.n_up) if j != i]
                others_d = [conf[j] for j in range(system.n_up, system.n_electrons) if j != i]
                if others_u: cu += np.histogram(others_u, bins=edges)[0]
                if others_d: cd += np.histogram(others_d, bins=edges)[0]
    if events:
        ru = cu / (events * dx); rd = cd / (events * dx)
    else:
        ru = cu; rd = cd
    rt = ru + rd
    return dict(x=centers, rho_cond_total=rt, rho_cond_up=ru, rho_cond_down=rd,
                pin_x=float(x0), pin_spin=pin_spin, pin_halfwidth=grid.pin_halfwidth,
                pin_events=int(events))


def localization_from_samples(positions, classical_sites: np.ndarray):
    """Ordered-particle localization and Lindemann-like crossover ratio."""
    s = _as_samples(positions)
    sites = np.sort(np.asarray(classical_sites, dtype=float))
    if s.shape[1] != sites.size:
        raise ValueError("number of classical sites must equal number of electrons")
    ordered = np.sort(s, axis=1)
    mean_sites = ordered.mean(axis=0)
    widths = ordered.std(axis=0, ddof=1) if ordered.shape[0] > 1 else np.zeros(sites.size)
    dnn = np.diff(sites)
    mean_dnn = float(np.mean(dnn)) if dnn.size else np.nan
    rms_width = float(np.sqrt(np.mean(widths**2)))
    lindemann = float(rms_width / mean_dnn) if mean_dnn > 0 else np.nan
    displacement = float(np.sqrt(np.mean((mean_sites - sites) ** 2)))
    return dict(classical_sites=sites, quantum_ordered_means=mean_sites,
                ordered_widths=widths, mean_nearest_neighbor_spacing=mean_dnn,
                rms_ordered_width=rms_width, lindemann_like_ratio=lindemann,
                rms_peak_to_classical_displacement=displacement)


def density_peak_metrics(x, rho, expected_n: int | None = None):
    x = np.asarray(x); rho = np.asarray(rho)
    dx = float(np.mean(np.diff(x)))
    min_distance = max(1, int(0.25 / max(dx, 1e-12)))
    prominence = max(1e-12, 0.04 * float(np.max(rho)))
    peaks, props = find_peaks(rho, distance=min_distance, prominence=prominence)
    order = np.argsort(rho[peaks])[::-1]
    if expected_n is not None and len(order) > expected_n:
        order = order[:expected_n]
    peaks = np.sort(peaks[order])
    return dict(n_peaks=int(len(peaks)), peak_positions=x[peaks], peak_heights=rho[peaks])


def small_r_probability(r, p, cutoff: float) -> float:
    r = np.asarray(r); p = np.asarray(p)
    mask = r <= cutoff
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(p[mask], r[mask]))


def rdm_density_on_grid(phi_up, phi_down, rdm1_up, rdm1_down):
    phi_up = np.asarray(phi_up); phi_down = np.asarray(phi_down)
    gu = np.asarray(rdm1_up); gd = np.asarray(rdm1_down)
    ru = np.einsum("xp,pq,xq->x", phi_up, gu, phi_up, optimize=True).real
    rd = np.einsum("xp,pq,xq->x", phi_down, gd, phi_down, optimize=True).real
    return ru, rd


def pair_density_matrix(phi_a, phi_b, rdm2):
    """Evaluate rho2(x,y) from Gamma[p,q,r,s] efficiently."""
    pa = np.asarray(phi_a); pb = np.asarray(phi_b); G = np.asarray(rdm2)
    A = np.einsum("xp,xr->xpr", pa, pa).reshape(pa.shape[0], -1)
    B = np.einsum("yq,ys->yqs", pb, pb).reshape(pb.shape[0], -1)
    M = np.transpose(G, (0, 2, 1, 3)).reshape(A.shape[1], B.shape[1])
    return (A @ M @ B.T).real


def pair_distance_from_rho2(x, rho2, same_spin: bool):
    x = np.asarray(x); R = np.asarray(rho2)
    if len(x) < 2: raise ValueError("grid too small")
    dx = float(np.mean(np.diff(x)))
    nr = len(x)
    r = np.arange(nr) * dx
    p = np.zeros(nr)
    for k in range(nr):
        if same_spin:
            p[k] = np.sum(np.diag(R, k=k)) * dx
        else:
            p[k] = (np.sum(np.diag(R, k=k)) + (0.0 if k == 0 else np.sum(np.diag(R, k=-k)))) * dx
    return r, p


def conditional_from_rho2(x, rho_pin, rho2_same, rho2_other, x0):
    idx = int(np.argmin(np.abs(np.asarray(x) - x0)))
    denom = max(float(rho_pin[idx]), 1e-15)
    return np.asarray(rho2_same)[idx, :] / denom, np.asarray(rho2_other)[idx, :] / denom, idx


def natural_occupations(rdm1):
    g = 0.5 * (np.asarray(rdm1) + np.asarray(rdm1).T.conj())
    vals, vecs = np.linalg.eigh(g)
    order = np.argsort(vals)[::-1]
    return np.real_if_close(vals[order]), np.real_if_close(vecs[:, order])


def estimate_vqmc_rdm1(samples, x_grid, system: SystemInfo, ratio_fn, spin: str = "up"):
    """Optional off-diagonal VQMC 1-RDM estimator using a wavefunction-ratio callback.

    ratio_fn(config_batch, electron_index, xprime_grid) must return
    Psi(X with x_i -> x') / Psi(X), shape (batch, n_grid).

    The estimator bins the original x_i coordinate and evaluates the off-diagonal
    ratio to each x'. This is intentionally an API-level routine: a generic saved
    position trajectory alone is insufficient to reconstruct the VQMC 1-RDM.
    """
    s = _as_samples(samples)
    xg = np.asarray(x_grid, dtype=float)
    edges = np.empty(xg.size + 1)
    edges[1:-1] = 0.5 * (xg[:-1] + xg[1:])
    edges[0] = xg[0] - 0.5 * (xg[1] - xg[0]); edges[-1] = xg[-1] + 0.5 * (xg[-1] - xg[-2])
    dx = np.diff(edges)
    idxs = range(system.n_up) if spin == "up" else range(system.n_up, system.n_electrons)
    gamma = np.zeros((xg.size, xg.size), dtype=complex)
    for i in idxs:
        bins = np.digitize(s[:, i], edges) - 1
        ratios = np.asarray(ratio_fn(s, i, xg))
        if ratios.shape != (s.shape[0], xg.size):
            raise ValueError("ratio_fn returned an unexpected shape")
        for b in range(xg.size):
            mask = bins == b
            if np.any(mask):
                gamma[b, :] += ratios[mask].sum(axis=0) / (s.shape[0] * dx[b])
    return np.real_if_close(gamma)
