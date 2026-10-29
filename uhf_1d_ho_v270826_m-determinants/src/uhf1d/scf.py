from dataclasses import dataclass
import json
import numpy as np
from scipy.linalg import eigh

from .integrals import build_integrals


def _density(c, nocc):
    return c[:, :nocc] @ c[:, :nocc].T


def _jk(p, eri):
    j = np.einsum("rs,pqrs->pq", p, eri, optimize=True)
    k = np.einsum("rs,prqs->pq", p, eri, optimize=True)
    return j, k


class _DIIS:
    def __init__(self, size):
        self.size, self.focks, self.errors = size, [], []

    def update(self, f, error):
        self.focks.append(f.copy()); self.errors.append(error.ravel().copy())
        if len(self.focks) > self.size:
            self.focks.pop(0); self.errors.pop(0)

    def extrapolate(self):
        m = len(self.focks)
        b = np.empty((m + 1, m + 1)); b[-1, :] = b[:, -1] = -1; b[-1, -1] = 0
        for i in range(m):
            for j in range(m):
                b[i, j] = np.dot(self.errors[i], self.errors[j])
        rhs = np.zeros(m + 1); rhs[-1] = -1
        try:
            coeff = np.linalg.solve(b, rhs)[:-1]
        except np.linalg.LinAlgError:
            coeff = np.linalg.lstsq(b, rhs, rcond=1e-12)[0][:-1]
        return sum(c * f for c, f in zip(coeff, self.focks))


@dataclass
class UHFResult:
    config: object
    converged: bool
    iterations: int
    coefficients_up: np.ndarray
    coefficients_down: np.ndarray
    eigenvalues_up: np.ndarray
    eigenvalues_down: np.ndarray
    density_up: np.ndarray
    density_down: np.ndarray
    overlap: np.ndarray
    energies: dict
    history: dict

    def save(self, path):
        metadata = {
            "format": "uhf1d-ho-state", "format_version": 1,
            "basis": "normalized_1d_harmonic_oscillator",
            "coefficient_convention": "C[basis_index, molecular_orbital_index]",
            "spin_order": "up_then_down", "config": self.config.to_dict(),
            "converged": self.converged, "iterations": self.iterations,
            "energies": self.energies,
        }
        np.savez_compressed(
            path, coefficients_up=self.coefficients_up, coefficients_down=self.coefficients_down,
            eigenvalues_up=self.eigenvalues_up, eigenvalues_down=self.eigenvalues_down,
            occupied_coefficients_up=self.coefficients_up[:, : self.config.n_up],
            occupied_coefficients_down=self.coefficients_down[:, : self.config.n_down],
            density_matrix_up=self.density_up, density_matrix_down=self.density_down,
            overlap=self.overlap, history_iteration=np.asarray(self.history["iteration"]),
            history_energy=np.asarray(self.history["energy"]),
            history_density_rms=np.asarray(self.history["density_rms"]),
            metadata_json=np.array(json.dumps(metadata)),
        )


def run_uhf(cfg):
    cfg.validate()
    s, h, eri = build_integrals(cfg)
    eps0, c0 = eigh(h, s)
    pu, pd = _density(c0, cfg.n_up), _density(c0, cfg.n_down)
    diis_u, diis_d = _DIIS(cfg.diis_size), _DIIS(cfg.diis_size)
    history = {"iteration": [], "energy": [], "density_rms": []}
    previous_energy = np.inf
    converged = False
    for iteration in range(1, cfg.max_iter + 1):
        pt = pu + pd
        jt, _ = _jk(pt, eri); _, ku = _jk(pu, eri); _, kd = _jk(pd, eri)
        fu, fd = h + jt - ku, h + jt - kd
        erru, errd = fu @ pu @ s - s @ pu @ fu, fd @ pd @ s - s @ pd @ fd
        diis_u.update(fu, erru); diis_d.update(fd, errd)
        if cfg.diis and iteration >= cfg.diis_start and len(diis_u.focks) >= 2:
            fu, fd = diis_u.extrapolate(), diis_d.extrapolate()
        if cfg.level_shift:
            fu += cfg.level_shift * (s - s @ pu @ s)
            fd += cfg.level_shift * (s - s @ pd @ s)
        eu, cu = eigh(fu, s); ed, cd = eigh(fd, s)
        pu_raw, pd_raw = _density(cu, cfg.n_up), _density(cd, cfg.n_down)
        pu_new = (1 - cfg.mixing) * pu + cfg.mixing * pu_raw
        pd_new = (1 - cfg.mixing) * pd + cfg.mixing * pd_raw
        pt_new = pu_new + pd_new
        jt_new, _ = _jk(pt_new, eri); _, ku_new = _jk(pu_new, eri); _, kd_new = _jk(pd_new, eri)
        e1 = float(np.einsum("pq,pq", pt_new, h))
        ej = float(0.5 * np.einsum("pq,pq", pt_new, jt_new))
        ek = float(-0.5 * (np.einsum("pq,pq", pu_new, ku_new) + np.einsum("pq,pq", pd_new, kd_new)))
        energy = e1 + ej + ek
        drms = float(np.sqrt((np.mean((pu_new-pu)**2) + np.mean((pd_new-pd)**2)) / 2))
        history["iteration"].append(iteration); history["energy"].append(energy); history["density_rms"].append(drms)
        pu, pd = pu_new, pd_new
        if abs(energy - previous_energy) < cfg.energy_tol and drms < cfg.density_tol:
            converged = True; break
        previous_energy = energy
    # Final canonical orbitals and energy at the final density.
    pt = pu + pd; jt, _ = _jk(pt, eri); _, ku = _jk(pu, eri); _, kd = _jk(pd, eri)
    eu, cu = eigh(h + jt - ku, s); ed, cd = eigh(h + jt - kd, s)
    e1 = float(np.einsum("pq,pq", pt, h)); ej = float(.5*np.einsum("pq,pq", pt, jt))
    exu = float(-.5*np.einsum("pq,pq", pu, ku)); exd = float(-.5*np.einsum("pq,pq", pd, kd))
    energies = {"one_body": e1, "hartree_coulomb": ej, "exchange_up": exu,
                "exchange_down": exd, "exchange": exu+exd, "total": e1+ej+exu+exd}
    return UHFResult(cfg, converged, iteration, cu, cd, eu, ed, pu, pd, s, energies, history)

