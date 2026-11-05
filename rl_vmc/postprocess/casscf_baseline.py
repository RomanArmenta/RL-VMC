from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import eval_hermite, factorial


def ho_basis_1d(x: np.ndarray, n_basis: int, omega: float) -> np.ndarray:
    """
    Evalúa las funciones base normalizadas del oscilador armónico 1D mediante
    la relación de recurrencia normalizada de 3 términos incondicionalmente estable:
    psi_n(z) = sqrt(2/n) * z * psi_{n-1}(z) - sqrt((n-1)/n) * psi_{n-2}(z)
    Inmune a desbordamientos numéricos (overflow) para n_basis >= 256.
    Retorna arreglo de forma (len(x), n_basis).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    z = np.sqrt(omega) * x_arr
    pref = (omega / np.pi) ** 0.25 * np.exp(-0.5 * z * z)
    out = np.empty((x_arr.size, n_basis), dtype=np.float64)
    
    if n_basis > 0:
        out[:, 0] = pref
    if n_basis > 1:
        out[:, 1] = np.sqrt(2.0) * z * out[:, 0]
    for n in range(2, n_basis):
        out[:, n] = np.sqrt(2.0 / n) * z * out[:, n - 1] - np.sqrt((n - 1.0) / n) * out[:, n - 2]
    return out


def _annihilate(det: int, pos: int) -> tuple[int, int]:
    """Aplica el operador de destrucción fermiónico a_{pos} sobre el bitstring `det`."""
    if ((det >> pos) & 1) == 0:
        return 0, 0
    lower = det & ((1 << pos) - 1)
    sign = -1 if (lower.bit_count() % 2) else 1
    return det ^ (1 << pos), sign


def _create(det: int, pos: int) -> tuple[int, int]:
    """Aplica el operador de creación fermiónico a^dagger_{pos} sobre el bitstring `det`."""
    if (det >> pos) & 1:
        return 0, 0
    lower = det & ((1 << pos) - 1)
    sign = -1 if (lower.bit_count() % 2) else 1
    return det | (1 << pos), sign


def load_casscf_npz_metadata(npz_path: str | Path) -> dict[str, Any]:
    """Extrae los metadatos físicos y variacionales de un archivo `casscf_state.npz`."""
    p = Path(npz_path).resolve()
    with np.load(p, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
    cfg = meta.get("config", {})
    n_up = int(cfg.get("n_up", 0))
    n_dn = int(cfg.get("n_down", cfg.get("n_dn", 0)))
    n_elec = n_up + n_dn
    omega = float(cfg.get("omega", 1.0))
    g = float(cfg.get("g", cfg.get("interaction_strength", 1.0)))
    softening = float(cfg.get("softening", 0.1))
    n_basis = int(cfg.get("n_basis", 16))
    n_core = int(cfg.get("n_core", cfg.get("n_core_orbitals", 0)))
    n_active = int(cfg.get("n_active_orbitals", cfg.get("n_active", 0)))
    energy = float(meta.get("energy", float("nan")))
    converged = bool(meta.get("converged", True))
    return {
        "npz_path": str(p),
        "folder_name": p.parent.name,
        "n_elec": n_elec,
        "n_up": n_up,
        "n_dn": n_dn,
        "omega": omega,
        "g": g,
        "softening": softening,
        "n_basis": n_basis,
        "n_core": n_core,
        "n_active": n_active,
        "energy": energy,
        "converged": converged,
        "label": f"CASSCF (b={n_basis}, c{n_core}-a{n_active})",
    }


def discover_casscf_baselines(
    search_root: str | Path = ".",
    uhf_pattern: str = "uhf*",
) -> list[dict[str, Any]]:
    """
    Escanea todas las carpetas en `search_root` que coincidan con `uhf_pattern`
    (por ejemplo, `uhf_1d_ho_v270826_m-determinants`) buscando archivos `casscf_state.npz`.
    """
    root = Path(search_root).resolve()
    candidates: list[Path] = []
    if (root / "casscf_state.npz").exists():
        candidates.append(root / "casscf_state.npz")
    for uhf_dir in sorted(root.glob(uhf_pattern)):
        if uhf_dir.is_dir():
            candidates.extend(sorted(uhf_dir.rglob("casscf_state.npz")))

    seen: set[Path] = set()
    baselines: list[dict[str, Any]] = []
    for c in candidates:
        real_c = c.resolve()
        if real_c in seen:
            continue
        seen.add(real_c)
        try:
            baselines.append(load_casscf_npz_metadata(real_c))
        except Exception:
            continue
    return baselines


def find_matching_casscf_baseline(
    baselines: list[dict[str, Any]],
    n_up: int,
    n_dn: int,
    omega: float,
    g: float,
    softening: float | None = None,
    tol: float = 1e-3,
    require_exact_softening: bool = False,
) -> dict[str, Any] | None:
    """
    Busca la mejor baseline CASSCF entrenada para el sistema físico (N_up, N_dn, omega, g).
    Si existen múltiples modelos CASSCF para el mismo sistema (distinto espacio activo o base):
      1. Prioriza aquellos cuyo `softening` coincida exactamente con el del modelo evaluado.
      2. Prioriza estados marcados como `converged=True`.
      3. Por el principio variacional, selecciona el de menor energía variacional `E_CASSCF`.
    Si no existe ninguna baseline para (N_up, N_dn, omega, g), retorna `None`.
    """
    matches = []
    for b in baselines:
        if int(b["n_up"]) != int(n_up) or int(b["n_dn"]) != int(n_dn):
            continue
        if abs(float(b["omega"]) - float(omega)) > tol:
            continue
        if abs(float(b["g"]) - float(g)) > tol:
            continue
        soft_diff = (
            abs(float(b["softening"]) - float(softening))
            if softening is not None
            else 0.0
        )
        if require_exact_softening and softening is not None and soft_diff > tol:
            continue
        matches.append((soft_diff <= tol, -soft_diff, bool(b["converged"]), -float(b["energy"]), b))

    if not matches:
        return None

    # Ordenar de mayor a menor según la tupla de prioridad y devolver el mejor
    matches.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
    return matches[0][4]


def compute_exact_casscf_1body_density(
    npz_path: str | Path,
    x_grid: np.ndarray | None = None,
    x_min: float = -15.0,
    x_max: float = 15.0,
    n_grid: int = 500,
) -> dict[str, Any]:
    """
    Calcula analíticamente la matriz de densidad reducida de 1 cuerpo (1-RDM)
    y los perfiles exactos de densidad resuelta por espín:
      rho_up(x), rho_down(x), rho_total(x) = rho_up(x) + rho_down(x)
    directamente a partir de `casscf_state.npz` (exacto a precisión de máquina ~1e-15).
    """
    p = Path(npz_path).resolve()
    with np.load(p, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata_json"]))
        c_mo = np.asarray(data["orbital_coefficients"], dtype=np.float64)
        if "selected_ci_coefficients" in data and len(data["selected_ci_coefficients"]) > 0:
            ci_coeffs = np.asarray(data["selected_ci_coefficients"], dtype=np.float64)
            dets = [int(d) for d in data["selected_ci_determinants"]]
        else:
            ci_coeffs = np.asarray(data["ci_coefficients"], dtype=np.float64)
            dets = [int(d) for d in data["ci_determinants"]]

    # Normalizar vector CI por seguridad
    norm_ci = np.linalg.norm(ci_coeffs)
    if norm_ci > 0:
        ci_coeffs = ci_coeffs / norm_ci

    cfg = meta.get("config", {})
    n_up = int(cfg.get("n_up", 2))
    n_dn = int(cfg.get("n_down", cfg.get("n_dn", 2)))
    n_elec = n_up + n_dn
    omega = float(cfg.get("omega", 0.1))
    g = float(cfg.get("g", 4.0))
    softening = float(cfg.get("softening", 0.1))
    n_basis = int(cfg.get("n_basis", c_mo.shape[0]))
    n_core = int(cfg.get("n_core", cfg.get("n_core_orbitals", 0)))
    n_act = int(cfg.get("n_active_orbitals", cfg.get("n_active", 0)))
    energy = float(meta.get("energy", float("nan")))

    if x_grid is None:
        x_arr = np.linspace(x_min, x_max, n_grid, dtype=np.float64)
    else:
        x_arr = np.asarray(x_grid, dtype=np.float64)

    # Construir 1-RDM en el espacio activo: gamma^sigma_{pq} = <Psi | a^dagger_{p,sigma} a_{q,sigma} | Psi>
    det_index = {d: idx for idx, d in enumerate(dets)}
    rdm_act_up = np.zeros((n_act, n_act), dtype=np.float64)
    rdm_act_dn = np.zeros((n_act, n_act), dtype=np.float64)

    for j_idx, det in enumerate(dets):
        cj = ci_coeffs[j_idx]
        if abs(cj) < 1e-16:
            continue
        for q in range(n_act):
            # Espín arriba (bits 0 .. n_act - 1)
            d1_u, s1_u = _annihilate(det, q)
            if s1_u != 0:
                for p_orb in range(n_act):
                    d2_u, s2_u = _create(d1_u, p_orb)
                    if s2_u != 0 and d2_u in det_index:
                        rdm_act_up[p_orb, q] += ci_coeffs[det_index[d2_u]] * cj * (s1_u * s2_u)

            # Espín abajo (bits n_act .. 2*n_act - 1)
            d1_d, s1_d = _annihilate(det, n_act + q)
            if s1_d != 0:
                for p_orb in range(n_act):
                    d2_d, s2_d = _create(d1_d, n_act + p_orb)
                    if s2_d != 0 and d2_d in det_index:
                        rdm_act_dn[p_orb, q] += ci_coeffs[det_index[d2_d]] * cj * (s1_d * s2_d)

    # Evaluar base de oscilador armónico y transformar a orbitales moleculares (MO)
    phi = ho_basis_1d(x_arr, n_basis, omega)  # (N_x, n_basis)
    # Usar einsum en float64 para evitar warnings de BLAS en números subnormales
    psi_mo = np.einsum("xb,bm->xm", phi, c_mo, optimize=True)  # (N_x, n_basis)

    rho_core = (
        np.sum(psi_mo[:, :n_core] ** 2, axis=1)
        if n_core > 0
        else np.zeros_like(x_arr)
    )
    if n_act > 0:
        psi_act = psi_mo[:, n_core : n_core + n_act]
        temp_up = np.einsum("xp,pq->xq", psi_act, rdm_act_up, optimize=True)
        temp_dn = np.einsum("xp,pq->xq", psi_act, rdm_act_dn, optimize=True)
        rho_up = rho_core + np.sum(temp_up * psi_act, axis=1)
        rho_dn = rho_core + np.sum(temp_dn * psi_act, axis=1)
    else:
        rho_up = rho_core.copy()
        rho_dn = rho_core.copy()

    rho_up = np.maximum(rho_up, 0.0)
    rho_dn = np.maximum(rho_dn, 0.0)
    rho_total = rho_up + rho_dn

    p_total = rho_total / max(float(n_elec), 1.0)
    p_up = rho_up / max(float(n_up), 1.0) if n_up > 0 else np.zeros_like(rho_up)
    p_down = rho_dn / max(float(n_dn), 1.0) if n_dn > 0 else np.zeros_like(rho_dn)

    return {
        "npz_path": str(p),
        "label": f"CASSCF (b={n_basis}, c{n_core}-a{n_act})",
        "energy": energy,
        "n_elec": n_elec,
        "n_up": n_up,
        "n_dn": n_dn,
        "omega": omega,
        "g": g,
        "softening": softening,
        "n_basis": n_basis,
        "n_core": n_core,
        "n_active": n_act,
        "x": x_arr,
        "rho_total": rho_total,
        "rho_tot": rho_total,
        "rho_up": rho_up,
        "rho_down": rho_dn,
        "rho_dn": rho_dn,
        "p_total": p_total,
        "p_up": p_up,
        "p_down": p_down,
    }


def compute_spin_kl_divergences(
    walkers: np.ndarray,
    casscf_ref: dict[str, Any],
    n_bins: int = 200,
    x_max: float = 15.0,
) -> dict[str, float]:
    """
    Calcula la divergencia de Kullback-Leibler 1D resuelta por espín:
      D_KL(P_sampled || P_CASSCF)
    para la densidad total, espín arriba y espín abajo, siguiendo exactamente la
    convención de `scripts/evaluar_wigner_casscf_2.py`.
    """
    w = np.asarray(walkers, dtype=np.float64)
    if w.ndim == 4:
        w = w.squeeze(-1).reshape(-1, w.shape[2])
    elif w.ndim == 3:
        w = w.reshape(-1, w.shape[-1])

    n_up = int(casscf_ref["n_up"])
    n_dn = int(casscf_ref["n_dn"])
    edges = np.linspace(-x_max, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    eps = 1e-12

    p_tot_ex = np.interp(centers, casscf_ref["x"], casscf_ref["p_total"]) + eps
    p_up_ex = np.interp(centers, casscf_ref["x"], casscf_ref["p_up"]) + eps
    p_dn_ex = np.interp(centers, casscf_ref["x"], casscf_ref["p_down"]) + eps

    p_tot_ex /= np.sum(p_tot_ex)
    p_up_ex /= np.sum(p_up_ex)
    p_dn_ex /= np.sum(p_dn_ex)

    h_tot, _ = np.histogram(w.reshape(-1), bins=edges, density=True)
    p_tot = (h_tot + eps) / np.sum(h_tot + eps)
    kl_tot = float(np.sum(p_tot * np.log(p_tot / p_tot_ex)))

    if n_up > 0:
        h_up, _ = np.histogram(w[:, :n_up].reshape(-1), bins=edges, density=True)
        p_up = (h_up + eps) / np.sum(h_up + eps)
        kl_up = float(np.sum(p_up * np.log(p_up / p_up_ex)))
    else:
        kl_up = float("nan")

    if n_dn > 0:
        h_dn, _ = np.histogram(w[:, n_up : n_up + n_dn].reshape(-1), bins=edges, density=True)
        p_dn = (h_dn + eps) / np.sum(h_dn + eps)
        kl_dn = float(np.sum(p_dn * np.log(p_dn / p_dn_ex)))
    else:
        kl_dn = float("nan")

    return {
        "kl_total": kl_tot,
        "kl_up": kl_up,
        "kl_down": kl_dn,
    }


def compute_kl_convergence_curve(
    walkers_chain: np.ndarray,
    casscf_ref: dict[str, Any],
    n_checkpoints: int = 20,
    n_bins: int = 200,
    x_max: float = 15.0,
) -> dict[str, np.ndarray]:
    """
    Calcula la evolución acumulada de D_KL(P_t || P_CASSCF) en función del paso MCMC `t`
    sobre una cadena de producción de forma `(n_steps, batch_size, n_elec, 1)`.
    """
    w = np.asarray(walkers_chain, dtype=np.float64)
    if w.ndim == 4:
        w = w.squeeze(-1)  # (n_steps, batch_size, n_elec)
    n_steps = w.shape[0]
    n_pts = min(max(2, n_checkpoints), n_steps)
    step_grid = np.unique(np.linspace(max(1, n_steps // n_pts), n_steps, n_pts, dtype=int))

    kl_tot_list, kl_up_list, kl_dn_list = [], [], []
    for t in step_grid:
        kls = compute_spin_kl_divergences(w[:t], casscf_ref, n_bins=n_bins, x_max=x_max)
        kl_tot_list.append(kls["kl_total"])
        kl_up_list.append(kls["kl_up"])
        kl_dn_list.append(kls["kl_down"])

    return {
        "steps": step_grid,
        "kl_total": np.asarray(kl_tot_list, dtype=np.float64),
        "kl_up": np.asarray(kl_up_list, dtype=np.float64),
        "kl_down": np.asarray(kl_dn_list, dtype=np.float64),
    }
