"""Small-system, state-specific CASSCF with exact active-space CI."""
from dataclasses import dataclass
from itertools import combinations
import json
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .integrals import build_integrals


def _annihilate(det, orbital):
    mask = 1 << orbital
    if not det & mask:
        return None
    sign = -1 if (det & (mask - 1)).bit_count() % 2 else 1
    return det ^ mask, sign


def _create(det, orbital):
    mask = 1 << orbital
    if det & mask:
        return None
    sign = -1 if (det & (mask - 1)).bit_count() % 2 else 1
    return det | mask, sign


def _apply(det, operators):
    phase = 1
    for kind, orbital in operators:
        value = _annihilate(det, orbital) if kind == "a" else _create(det, orbital)
        if value is None:
            return None
        det, sign = value; phase *= sign
    return det, phase


def _determinants(norb, nup, ndown):
    result = []
    for up in combinations(range(norb), nup):
        up_bits = sum(1 << i for i in up)
        for down in combinations(range(norb), ndown):
            result.append(up_bits | sum(1 << (norb + i) for i in down))
    return result


def solve_fci(h, eri, nup, ndown, max_dimension=20000):
    """Lowest fixed-(N_up,N_down) eigenstate for spatial integrals."""
    norb = h.shape[0]; dets = _determinants(norb, nup, ndown)
    if len(dets) > max_dimension:
        raise ValueError(f"CI dimension {len(dets)} exceeds max_ci_dimension={max_dimension}")
    index = {d: i for i, d in enumerate(dets)}
    ham = np.zeros((len(dets), len(dets)))
    for col, det in enumerate(dets):
        for spin in range(2):
            offset = spin * norb
            for p in range(norb):
                for q in range(norb):
                    value = h[p, q]
                    if abs(value) < 1e-15: continue
                    applied = _apply(det, [("a", offset+q), ("c", offset+p)])
                    if applied and applied[0] in index: ham[index[applied[0]], col] += value * applied[1]
        for sp in range(2):
            for st in range(2):
                op, ot = sp*norb, st*norb
                for p in range(norb):
                    for q in range(norb):
                        for r in range(norb):
                            for s in range(norb):
                                value = 0.5 * eri[p, q, r, s]
                                if abs(value) < 1e-15: continue
                                # a†(p,sp) a†(r,st) a(s,st) a(q,sp)
                                applied = _apply(det, [("a", op+q), ("a", ot+s),
                                                       ("c", ot+r), ("c", op+p)])
                                if applied and applied[0] in index:
                                    ham[index[applied[0]], col] += value * applied[1]
    ham = (ham + ham.T) * 0.5
    values, vectors = np.linalg.eigh(ham)
    return float(values[0]), vectors[:, 0], np.asarray(dets, dtype=object)


def _transform(h, eri, c):
    ht = c.T @ h @ c
    et = np.einsum("pa,qb,rc,sd,pqrs->abcd", c, c, c, c, eri, optimize=True)
    return ht, et


def _core_active_integrals(h, eri, ncore, nactive):
    core = range(ncore); active = slice(ncore, ncore+nactive)
    ecore = 2.0 * sum(h[i, i] for i in core)
    for i in core:
        for j in core:
            ecore += 2.0 * eri[i, i, j, j] - eri[i, j, j, i]
    heff = h[active, active].copy()
    for u in range(nactive):
        for v in range(nactive):
            U, V = ncore+u, ncore+v
            for i in core:
                heff[u, v] += 2.0*eri[U, V, i, i] - eri[U, i, i, V]
    return float(ecore), heff, eri[active, active, active, active]


def _rotation_pairs(cfg):
    core = range(cfg.n_core)
    active = range(cfg.n_core, cfg.n_core + cfg.n_active_orbitals)
    virtual = range(cfg.n_core + cfg.n_active_orbitals, cfg.n_basis)
    return ([(i, a) for i in core for a in active] +
            [(i, v) for i in core for v in virtual] +
            [(a, v) for a in active for v in virtual])


def _rotation(parameters, pairs, n):
    k = np.zeros((n, n))
    for value, (p, q) in zip(parameters, pairs): k[p, q], k[q, p] = value, -value
    return expm(k)


@dataclass
class CASSCFResult:
    config: object
    converged: bool
    iterations: int
    energy: float
    core_energy: float
    active_energy: float
    orbital_coefficients: np.ndarray
    ci_coefficients: np.ndarray
    ci_determinants: np.ndarray
    overlap: np.ndarray
    history: dict
    optimizer_message: str

    def selected_ci(self):
        """Return coefficient-ranked, normalized final expansion and retained norm."""
        order = np.argsort(np.abs(self.ci_coefficients))[::-1]
        requested = self.config.n_final_determinants
        if requested > 0: order = order[:requested]
        raw = self.ci_coefficients[order]
        retained_norm = float(np.dot(raw, raw))
        normalized = raw / np.sqrt(retained_norm) if retained_norm > 0 else raw
        return normalized, self.ci_determinants[order], order, retained_norm

    def save(self, path):
        selected_c, selected_d, selected_indices, retained_norm = self.selected_ci()
        metadata = {"format": "casscf1d-ho-state", "format_version": 1,
                    "basis": "normalized_1d_harmonic_oscillator",
                    "coefficient_convention": "C[basis_index, optimized_orbital_index]",
                    "determinant_encoding": "bitstring: active up bits, then active down bits",
                    "config": self.config.to_dict(), "converged": self.converged,
                    "energy": self.energy, "core_energy": self.core_energy,
                    "active_energy": self.active_energy,
                    "full_ci_dimension": len(self.ci_coefficients),
                    "selected_ci_dimension": len(selected_c),
                    "selected_ci_retained_norm": retained_norm,
                    "selected_ci_discarded_norm": 1.0-retained_norm}
        np.savez_compressed(path, orbital_coefficients=self.orbital_coefficients,
            core_orbital_coefficients=self.orbital_coefficients[:, :self.config.n_core],
            active_orbital_coefficients=self.orbital_coefficients[:, self.config.n_core:self.config.n_core+self.config.n_active_orbitals],
            ci_coefficients=self.ci_coefficients, ci_determinants=self.ci_determinants.astype(str),
            selected_ci_coefficients=selected_c,
            selected_ci_determinants=selected_d.astype(str),
            selected_ci_full_indices=selected_indices,
            selected_ci_retained_norm=np.array(retained_norm),
            overlap=self.overlap, history_iteration=np.asarray(self.history["iteration"]),
            history_energy=np.asarray(self.history["energy"]), metadata_json=np.array(json.dumps(metadata)))


def run_casscf(cfg):
    cfg.validate(); s, h, eri = build_integrals(cfg); pairs = _rotation_pairs(cfg)
    dim = math.comb(cfg.n_active_orbitals, cfg.n_active_up) * math.comb(cfg.n_active_orbitals, cfg.n_active_down)
    if dim > cfg.max_ci_dimension:
        raise ValueError(f"CI dimension {dim} exceeds max_ci_dimension={cfg.max_ci_dimension}")
    history = {"iteration": [], "energy": []}; last = {}
    def objective(parameters):
        c = _rotation(parameters, pairs, cfg.n_basis)
        ht, et = _transform(h, eri, c)
        ecore, ha, ea = _core_active_integrals(ht, et, cfg.n_core, cfg.n_active_orbitals)
        eci, ci, dets = solve_fci(ha, ea, cfg.n_active_up, cfg.n_active_down, cfg.max_ci_dimension)
        total = ecore + eci
        last.update(c=c, ecore=ecore, eci=eci, ci=ci, dets=dets, total=total)
        return total
    counter = {"n": 0}
    def callback(x):
        counter["n"] += 1; energy = objective(x)
        history["iteration"].append(counter["n"]); history["energy"].append(energy)
        print(f"iteration: {counter["n"]}, energy: {energy}")
    x0 = np.zeros(len(pairs))
    if len(pairs):
        opt = minimize(objective, x0, method="BFGS", callback=callback,
                       options={"gtol": cfg.orbital_gradient_tol, "maxiter": cfg.casscf_max_iter})
        objective(opt.x); converged = bool(opt.success)
        # Precision loss can occur after the energy has numerically stopped changing.
        if not converged and len(history["energy"]) >= 2:
            converged = abs(history["energy"][-1]-history["energy"][-2]) < cfg.casscf_energy_tol
        message, iterations = str(opt.message), int(opt.nit)
    else:
        objective(x0); converged, message, iterations = True, "No nonredundant orbital rotations", 0
    if not history["energy"]:
        history["iteration"].append(0); history["energy"].append(last["total"])
    return CASSCFResult(cfg, converged, iterations, last["total"], last["ecore"], last["eci"],
                        last["c"], last["ci"], last["dets"], s, history, message)
