from __future__ import annotations
import numpy as np


def occupations_to_mask(occ) -> int:
    mask = 0
    for i, val in enumerate(np.asarray(occ, dtype=int).reshape(-1)):
        if val:
            mask |= 1 << i
    return mask


def _ann(mask: int, orb: int):
    if not (mask >> orb) & 1:
        return None
    sign = -1.0 if ((mask & ((1 << orb) - 1)).bit_count() % 2) else 1.0
    return mask ^ (1 << orb), sign


def _cre(mask: int, orb: int):
    if (mask >> orb) & 1:
        return None
    sign = -1.0 if ((mask & ((1 << orb) - 1)).bit_count() % 2) else 1.0
    return mask | (1 << orb), sign


def _apply_one(key, spin: str, p: int, q: int):
    up, dn = key
    mask = up if spin == "up" else dn
    a = _ann(mask, q)
    if a is None:
        return None
    mask1, s1 = a
    c = _cre(mask1, p)
    if c is None:
        return None
    mask2, s2 = c
    return ((mask2, dn) if spin == "up" else (up, mask2)), s1 * s2


def _apply_two(key, spins, p, q, r, s):
    """Apply a^+_{p,s1} a^+_{q,s2} a_{s,s2} a_{r,s1}."""
    up, dn = key
    sign = 1.0
    # Global ordering is all up orbitals, then all down orbitals. For number-conserving
    # spin blocks, cross-sector parity factors cancel. We can therefore apply within
    # the separate bit masks while preserving operator order.
    def ann_spin(u, d, sp, orb):
        nonlocal sign
        mask = u if sp == "up" else d
        z = _ann(mask, orb)
        if z is None:
            return None
        m, sg = z; sign *= sg
        return (m, d) if sp == "up" else (u, m)
    def cre_spin(u, d, sp, orb):
        nonlocal sign
        mask = u if sp == "up" else d
        z = _cre(mask, orb)
        if z is None:
            return None
        m, sg = z; sign *= sg
        return (m, d) if sp == "up" else (u, m)
    z = ann_spin(up, dn, spins[0], r)
    if z is None: return None
    up, dn = z
    z = ann_spin(up, dn, spins[1], s)
    if z is None: return None
    up, dn = z
    z = cre_spin(up, dn, spins[1], q)
    if z is None: return None
    up, dn = z
    z = cre_spin(up, dn, spins[0], p)
    if z is None: return None
    up, dn = z
    return (up, dn), sign


def ci_rdms(coefficients, occupations_up, occupations_down, compute_two_body=True):
    """Spin-resolved 1- and 2-RDMs from a determinant/CI expansion.

    Conventions:
      gamma^s[p,q] = <a^+_{p,s} a_{q,s}>
      Gamma^ss[p,q,r,s] = <a^+_p a^+_q a_s a_r>
      Gamma^ud[p,q,r,s] = <a^+_{p,u} a^+_{q,d} a_{s,d} a_{r,u}>
    """
    c = np.asarray(coefficients)
    ou = np.asarray(occupations_up, dtype=int)
    od = np.asarray(occupations_down, dtype=int)
    if ou.shape != od.shape or ou.shape[0] != c.size:
        raise ValueError("CI coefficients and occupation arrays are inconsistent")
    norb = ou.shape[1]
    norm = np.vdot(c, c).real
    if norm <= 0:
        raise ValueError("zero CI norm")
    c = c / np.sqrt(norm)
    keys = [(occupations_to_mask(ou[i]), occupations_to_mask(od[i])) for i in range(c.size)]
    index = {k: i for i, k in enumerate(keys)}

    def one(spin):
        g = np.zeros((norb, norb), dtype=complex)
        for j, key in enumerate(keys):
            for q in range(norb):
                for p in range(norb):
                    z = _apply_one(key, spin, p, q)
                    if z is None: continue
                    k2, sg = z
                    i = index.get(k2)
                    if i is not None:
                        g[p, q] += np.conjugate(c[i]) * c[j] * sg
        return np.real_if_close(g)

    gu, gd = one("up"), one("down")
    out = {"rdm1_up": gu, "rdm1_down": gd}
    if not compute_two_body:
        return out

    def two(spins):
        G = np.zeros((norb, norb, norb, norb), dtype=complex)
        for j, key in enumerate(keys):
            for r in range(norb):
                for s in range(norb):
                    for q in range(norb):
                        for p in range(norb):
                            z = _apply_two(key, spins, p, q, r, s)
                            if z is None: continue
                            k2, sg = z
                            i = index.get(k2)
                            if i is not None:
                                G[p, q, r, s] += np.conjugate(c[i]) * c[j] * sg
        return np.real_if_close(G)

    out["rdm2_uu"] = two(("up", "up"))
    out["rdm2_dd"] = two(("down", "down"))
    out["rdm2_ud"] = two(("up", "down"))
    return out


def hf_rdms_from_occupations(n_orb: int, n_up: int, n_down: int):
    gu = np.diag([1.0] * n_up + [0.0] * (n_orb - n_up))
    gd = np.diag([1.0] * n_down + [0.0] * (n_orb - n_down))
    Guu = np.einsum("pr,qs->pqrs", gu, gu) - np.einsum("ps,qr->pqrs", gu, gu)
    Gdd = np.einsum("pr,qs->pqrs", gd, gd) - np.einsum("ps,qr->pqrs", gd, gd)
    Gud = np.einsum("pr,qs->pqrs", gu, gd)
    return dict(rdm1_up=gu, rdm1_down=gd, rdm2_uu=Guu, rdm2_dd=Gdd, rdm2_ud=Gud)
