from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def compute_1d_observables(
    walkers: np.ndarray,
    n_up: int,
    n_down: int,
    n_bins: int = 200,
    x_max: float = 6.0,
    omega: float | None = None,
) -> dict:
    """
    Calcula las densidades de 1 cuerpo resueltas por espín, distribuciones de distancia de pares
    y métricas de localización de Wigner a partir de las trayectorias de caminantes guardadas.
    Soporta arreglos con múltiples corridas independientes:
      - (n_chains, n_steps, n_walkers, n_elec, 1) -> calcula media y desviación estándar entre corridas
      - (n_steps, n_walkers, n_elec, 1) o (n_samples, n_elec)
    """
    w_raw = np.asarray(walkers, dtype=np.float64)

    # Identificar si tenemos múltiples cadenas independientes (n_chains, n_steps, W, N, 1)
    if w_raw.ndim == 5:
        n_chains = w_raw.shape[0]
        n_elec = w_raw.shape[3]
        chains_list = [w_raw[c].squeeze(-1).reshape(-1, n_elec) for c in range(n_chains)]
        w = np.concatenate(chains_list, axis=0)
    elif w_raw.ndim == 4:
        n_chains = 1
        n_elec = w_raw.shape[2]
        w = w_raw.squeeze(-1).reshape(-1, n_elec)
        chains_list = [w]
    elif w_raw.ndim == 3:
        n_chains = 1
        n_elec = w_raw.shape[-1]
        w = w_raw.reshape(-1, n_elec)
        chains_list = [w]
    else:
        n_chains = 1
        n_elec = w_raw.shape[1]
        w = w_raw
        chains_list = [w]

    n_samples = w.shape[0]
    x_all = w.reshape(-1)

    # Adaptar x_max automáticamente para regímenes de trampa débil (ej. omega=0.1 -> [-15, 15])
    eff_x_max = float(x_max) if x_max is not None else 15.0
    if omega is not None and omega <= 0.25:
        eff_x_max = max(eff_x_max, 15.0)
    if x_all.size > 0:
        max_extent = float(np.percentile(np.abs(x_all), 99.8)) * 1.25
        if max_extent > eff_x_max:
            eff_x_max = float(np.ceil(max_extent))

    edges = np.linspace(-eff_x_max, eff_x_max, n_bins + 1)
    dx = edges[1] - edges[0]
    x_grid = 0.5 * (edges[:-1] + edges[1:])

    # Calcular densidades por cada cadena independiente para obtener barras de error (std / sem)
    chain_rho_tot, chain_rho_up, chain_rho_dn = [], [], []
    for wc in chains_list:
        h_tot, _ = np.histogram(wc.reshape(-1), bins=edges, density=True)
        chain_rho_tot.append(h_tot * float(n_elec))

        if n_up > 0:
            h_up, _ = np.histogram(wc[:, :n_up].reshape(-1), bins=edges, density=True)
            chain_rho_up.append(h_up * float(n_up))
        else:
            chain_rho_up.append(np.zeros_like(x_grid))

        if n_down > 0:
            h_dn, _ = np.histogram(
                wc[:, n_up : n_up + n_down].reshape(-1), bins=edges, density=True
            )
            chain_rho_dn.append(h_dn * float(n_down))
        else:
            chain_rho_dn.append(np.zeros_like(x_grid))

    arr_rho_tot = np.stack(chain_rho_tot, axis=0)
    arr_rho_up = np.stack(chain_rho_up, axis=0)
    arr_rho_dn = np.stack(chain_rho_dn, axis=0)

    rho_total = np.mean(arr_rho_tot, axis=0)
    rho_up = np.mean(arr_rho_up, axis=0)
    rho_down = np.mean(arr_rho_dn, axis=0)

    rho_total_std = np.std(arr_rho_tot, axis=0) if n_chains > 1 else np.zeros_like(x_grid)
    rho_up_std = np.std(arr_rho_up, axis=0) if n_chains > 1 else np.zeros_like(x_grid)
    rho_down_std = np.std(arr_rho_dn, axis=0) if n_chains > 1 else np.zeros_like(x_grid)

    # Suavizado para detección de picos (análisis de localización de Wigner)
    rho_smooth = gaussian_filter1d(rho_total, sigma=2.0)
    peaks, _ = find_peaks(
        rho_smooth,
        prominence=0.05 * np.max(rho_smooth) if np.max(rho_smooth) > 0 else 0.01,
        distance=max(3, n_bins // 20),
    )
    n_wigner_peaks = int(len(peaks))
    rms_radius = float(np.sqrt(np.mean(x_all ** 2))) if x_all.size > 0 else 0.0

    # Distribución de distancias entre pares g(|x_i - x_j|)
    r_edges = np.linspace(0.0, 2.0 * eff_x_max, n_bins + 1)
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])

    if n_elec >= 2:
        max_cfg = min(n_samples, 100000)
        w_sub = w[:max_cfg]
        idx_i, idx_j = np.triu_indices(n_elec, k=1)
        dists_all = np.abs(w_sub[:, idx_i] - w_sub[:, idx_j])
        g_pair_total, _ = np.histogram(dists_all.reshape(-1), bins=r_edges, density=True)
        mean_pair_dist = float(np.mean(dists_all))

        same_mask = ((idx_i < n_up) & (idx_j < n_up)) | (
            (idx_i >= n_up) & (idx_j >= n_up)
        )
        opp_mask = ~same_mask

        if np.any(same_mask):
            g_pair_same, _ = np.histogram(
                dists_all[:, same_mask].reshape(-1), bins=r_edges, density=True
            )
        else:
            g_pair_same = np.zeros_like(r_grid)

        if np.any(opp_mask):
            g_pair_opp, _ = np.histogram(
                dists_all[:, opp_mask].reshape(-1), bins=r_edges, density=True
            )
        else:
            g_pair_opp = np.zeros_like(r_grid)
    else:
        g_pair_total = np.zeros_like(r_grid)
        g_pair_same = np.zeros_like(r_grid)
        g_pair_opp = np.zeros_like(r_grid)
        mean_pair_dist = 0.0

    return {
        "n_chains_eval": int(n_chains),
        "x_max": float(eff_x_max),
        "x_grid": x_grid,
        "dx": float(dx),
        "rho_total": rho_total,
        "rho_up": rho_up,
        "rho_down": rho_down,
        "rho_total_std": rho_total_std,
        "rho_up_std": rho_up_std,
        "rho_down_std": rho_down_std,
        "r_grid": r_grid,
        "g_pair_total": g_pair_total,
        "g_pair_same": g_pair_same,
        "g_pair_opp": g_pair_opp,
        "n_wigner_peaks": n_wigner_peaks,
        "rms_radius": rms_radius,
        "mean_pair_dist": mean_pair_dist,
    }


def compute_kl_divergence_2e(
    walkers: np.ndarray,
    ansatz_apply,
    params_psi: dict,
    n_bins: int = 80,
    x_max: float = 5.0,
) -> float:
    """
    Calcula la divergencia KL empírica 2D D_KL(P_MCMC || |Psi|^2) para N=2 electrones en [-x_max, x_max]^2.
    """
    import jax.numpy as jnp

    w = np.asarray(walkers, dtype=np.float64)
    if w.ndim == 5:
        w = w.squeeze(-1).reshape(-1, w.shape[3])
    elif w.ndim == 4:
        w = w.squeeze(-1).reshape(-1, w.shape[2])
    if w.shape[1] != 2:
        return float("nan")

    edges = np.linspace(-x_max, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    da = (edges[1] - edges[0]) ** 2

    hist_2d, _, _ = np.histogram2d(
        w[:, 0], w[:, 1], bins=[edges, edges], density=True
    )
    p_emp = hist_2d * da
    p_emp = p_emp / np.sum(p_emp)

    xx, yy = np.meshgrid(centers, centers, indexing="ij")
    grid_pts = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)[:, :, None]
    log_psi_vals = np.asarray(ansatz_apply(params_psi, jnp.asarray(grid_pts)))
    log_prob = 2.0 * (log_psi_vals - np.max(log_psi_vals))
    q_exact = np.exp(log_prob).reshape(n_bins, n_bins)
    q_exact = q_exact / np.sum(q_exact)

    mask = (p_emp > 1e-12) & (q_exact > 1e-12)
    kl = float(np.sum(p_emp[mask] * np.log(p_emp[mask] / q_exact[mask])))
    return kl
