from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
import jax
import jax.numpy as jnp

from src.physics import SystemConfig, spin_labels

Array = jax.Array
OrbitalModel = Literal["ho_fixed", "ho_variational", "gaussian_basis", "backflow_ho"]


def _softplus_inverse(y: float) -> float:
    if y <= 0:
        raise ValueError("softplus inverse requires y > 0")
    return math.log(math.expm1(float(y)))


@dataclass(frozen=True)
class OrbitalConfig:
    model: OrbitalModel = "ho_fixed"
    alpha_init: float = 1.0
    min_positive: float = 1e-5

    # Gaussian-expanded orbital bank
    gaussian_basis_size: int = 10
    gaussian_center_max: float = 4.0
    gaussian_width_init: float = 0.9
    gaussian_train_width: bool = True

    # Backflow on top of HO orbitals
    backflow_components: int = 8
    backflow_r_max: float = 5.0
    backflow_width: float = 0.7
    backflow_envelope_length: float = 3.0


def _hermite_physicists(n: int, z: Array) -> Array:
    """Physicists' Hermite polynomial H_n(z), recurrence in JAX."""
    if n == 0:
        return jnp.ones_like(z)
    if n == 1:
        return 2.0 * z
    hm2 = jnp.ones_like(z)
    hm1 = 2.0 * z
    for k in range(1, n):
        h = 2.0 * z * hm1 - 2.0 * k * hm2
        hm2, hm1 = hm1, h
    return hm1


def harmonic_orbital(n: int, x: Array, alpha: Array) -> Array:
    """Normalized 1D harmonic-oscillator orbital with variational width alpha."""
    z = jnp.sqrt(alpha) * x
    norm = (alpha / jnp.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))
    return norm * _hermite_physicists(n, z) * jnp.exp(-0.5 * alpha * x**2)


def harmonic_orbital_bank(x: Array, n_orbitals: int, alpha: Array) -> Array:
    """Return (..., n_orbitals)."""
    return jnp.stack([harmonic_orbital(n, x, alpha) for n in range(n_orbitals)], axis=-1)


def _gaussian_centers(cfg: OrbitalConfig) -> Array:
    return jnp.linspace(-cfg.gaussian_center_max, cfg.gaussian_center_max, cfg.gaussian_basis_size)


def gaussian_basis_values(x: Array, centers: Array, width: Array) -> Array:
    return jnp.exp(-0.5 * ((x[..., None] - centers) / width) ** 2)


def _initial_gaussian_coefficients(system_cfg: SystemConfig, cfg: OrbitalConfig) -> Array:
    """Least-squares projection of occupied HO orbitals onto a Gaussian basis.

    This produces a stable starting point rather than random orbital matrices.
    Coefficients are intentionally unconstrained: one-particle orbitals need nodes.
    """
    n_orb = system_cfg.max_spin_occupancy
    K = cfg.gaussian_basis_size
    if K < n_orb:
        raise ValueError(
            f"gaussian_basis_size ({K}) must be >= max spin occupancy ({n_orb})"
        )

    grid = np.linspace(-cfg.gaussian_center_max * 1.3, cfg.gaussian_center_max * 1.3, 500)
    centers = np.linspace(-cfg.gaussian_center_max, cfg.gaussian_center_max, K)
    width = float(cfg.gaussian_width_init)
    B = np.exp(-0.5 * ((grid[:, None] - centers[None, :]) / width) ** 2)

    alpha = float(cfg.alpha_init)
    coeffs = []
    # NumPy recurrence for initialization only.
    z = np.sqrt(alpha) * grid
    H = [np.ones_like(z)]
    if n_orb > 1:
        H.append(2.0 * z)
    for n in range(1, n_orb - 1):
        H.append(2.0 * z * H[-1] - 2.0 * n * H[-2])
    for n in range(n_orb):
        norm = (alpha / np.pi) ** 0.25 / math.sqrt((2.0**n) * math.factorial(n))
        target = norm * H[n] * np.exp(-0.5 * alpha * grid**2)
        c, *_ = np.linalg.lstsq(B, target, rcond=1e-10)
        coeffs.append(c)
    return jnp.asarray(np.stack(coeffs, axis=0))


def init_orbital_params(key: Array, system_cfg: SystemConfig, cfg: OrbitalConfig) -> dict:
    eps = cfg.min_positive
    if cfg.model == "ho_fixed":
        return {}
    if cfg.model == "ho_variational":
        return {
            "raw_orbital_alpha": jnp.asarray(
                _softplus_inverse(max(cfg.alpha_init - eps, eps))
            )
        }
    if cfg.model == "gaussian_basis":
        params = {
            "orbital_coeff": _initial_gaussian_coefficients(system_cfg, cfg),
        }
        if cfg.gaussian_train_width:
            params["raw_gaussian_width"] = jnp.asarray(
                _softplus_inverse(max(cfg.gaussian_width_init - eps, eps))
            )
        return params
    if cfg.model == "backflow_ho":
        return {
            "raw_orbital_alpha": jnp.asarray(
                _softplus_inverse(max(cfg.alpha_init - eps, eps))
            ),
            # Separate same-spin / opposite-spin backflow channels.
            "backflow_coeff_same": jnp.zeros((cfg.backflow_components,)),
            "backflow_coeff_opposite": jnp.zeros((cfg.backflow_components,)),
        }
    raise ValueError(f"Unknown orbital model: {cfg.model}")


def physical_orbital_params(params: dict, cfg: OrbitalConfig) -> dict:
    eps = cfg.min_positive
    if cfg.model == "ho_fixed":
        return {"alpha": jnp.asarray(cfg.alpha_init)}
    if cfg.model == "ho_variational":
        return {"alpha": jax.nn.softplus(params["raw_orbital_alpha"]) + eps}
    if cfg.model == "gaussian_basis":
        width = (
            jax.nn.softplus(params["raw_gaussian_width"]) + eps
            if cfg.gaussian_train_width
            else jnp.asarray(cfg.gaussian_width_init)
        )
        # Row normalization controls the trivial determinant scale direction.
        coeff = params["orbital_coeff"]
        coeff = coeff / jnp.sqrt(jnp.sum(coeff**2, axis=1, keepdims=True) + 1e-12)
        return {
            "coeff": coeff,
            "centers": _gaussian_centers(cfg),
            "width": width,
        }
    if cfg.model == "backflow_ho":
        return {
            "alpha": jax.nn.softplus(params["raw_orbital_alpha"]) + eps,
            "backflow_coeff_same": params["backflow_coeff_same"],
            "backflow_coeff_opposite": params["backflow_coeff_opposite"],
        }
    raise ValueError(f"Unknown orbital model: {cfg.model}")


def _backflow_displacement(
    x: Array,
    system_cfg: SystemConfig,
    cfg: OrbitalConfig,
    p: dict,
) -> Array:
    """Permutation-equivariant 1D backflow displacement.

    xi_i = (1/sqrt(N-1)) sum_{j != i} (x_i-x_j) eta_sigma(r_ij)

    eta has independent same-spin and opposite-spin RBF expansions and a smooth
    long-range envelope. Initial coefficients are zero, so the model starts as
    ordinary Slater-HO.
    """
    N = system_cfg.n_electrons
    if N <= 1:
        return jnp.zeros_like(x)

    s = spin_labels(system_cfg)
    centers = jnp.linspace(0.0, cfg.backflow_r_max, cfg.backflow_components)
    width = jnp.asarray(cfg.backflow_width, dtype=x.dtype)
    ell = jnp.asarray(cfg.backflow_envelope_length, dtype=x.dtype)

    dx = x[:, None] - x[None, :]
    r = jnp.abs(dx)
    basis = jnp.exp(-0.5 * ((r[..., None] - centers) / width) ** 2)
    envelope = jnp.exp(-(r / ell) ** 2)
    eta_same = jnp.einsum("ijk,k->ij", basis, p["backflow_coeff_same"])
    eta_opp = jnp.einsum("ijk,k->ij", basis, p["backflow_coeff_opposite"])
    same = s[:, None] == s[None, :]
    eta = jnp.where(same, eta_same, eta_opp) * envelope
    mask = 1.0 - jnp.eye(N, dtype=x.dtype)
    xi = jnp.sum(mask * dx * eta, axis=1) / math.sqrt(max(N - 1, 1))
    return xi


def effective_coordinates(
    x: Array,
    params: dict,
    system_cfg: SystemConfig,
    cfg: OrbitalConfig,
) -> Array:
    if cfg.model != "backflow_ho":
        return x
    p = physical_orbital_params(params, cfg)
    return x + _backflow_displacement(x, system_cfg, cfg, p)


def orbital_bank(
    x: Array,
    params: dict,
    system_cfg: SystemConfig,
    cfg: OrbitalConfig,
) -> Array:
    """Evaluate spatial orbital bank at coordinates x, shape (N, n_orbitals)."""
    n_orb = system_cfg.max_spin_occupancy
    if n_orb == 0:
        return jnp.zeros((system_cfg.n_electrons, 0), dtype=x.dtype)

    p = physical_orbital_params(params, cfg)
    xeff = effective_coordinates(x, params, system_cfg, cfg)

    if cfg.model in ("ho_fixed", "ho_variational", "backflow_ho"):
        return harmonic_orbital_bank(xeff, n_orb, p["alpha"])

    if cfg.model == "gaussian_basis":
        B = gaussian_basis_values(xeff, p["centers"], p["width"])
        # B: (N,K), coeff: (n_orb,K) -> (N,n_orb)
        return B @ p["coeff"].T

    raise ValueError(f"Unknown orbital model: {cfg.model}")


def determinant_logabs_sign(
    x: Array,
    params: dict,
    system_cfg: SystemConfig,
    cfg: OrbitalConfig,
    node_epsilon: float = 1e-14,
) -> tuple[Array, Array]:
    """Return log|D_up D_down| and sign(D_up D_down)."""
    bank = orbital_bank(x, params, system_cfg, cfg)
    total_logabs = jnp.asarray(0.0, dtype=x.dtype)
    total_sign = jnp.asarray(1.0, dtype=x.dtype)

    for start, nspin in ((0, system_cfg.n_up), (system_cfg.n_up, system_cfg.n_down)):
        if nspin == 0:
            continue
        M = bank[start : start + nspin, :nspin]
        sign, logabs = jnp.linalg.slogdet(M)
        # Exact nodes have logabs=-inf. Keep the node effectively zero while
        # protecting downstream logs from NaN during rare numerical encounters.
        logabs = jnp.where(jnp.isfinite(logabs), logabs, jnp.log(node_epsilon))
        total_logabs = total_logabs + logabs
        total_sign = total_sign * sign

    return total_logabs, total_sign


def orbital_parameter_summary(params: dict, cfg: OrbitalConfig) -> dict[str, Array]:
    p = physical_orbital_params(params, cfg)
    if cfg.model == "ho_fixed":
        return {"orbital_alpha": p["alpha"]}
    if cfg.model == "ho_variational":
        return {"orbital_alpha": p["alpha"]}
    if cfg.model == "gaussian_basis":
        return {
            "gaussian_width": p["width"],
            "orbital_coeff_norm": jnp.sqrt(jnp.sum(p["coeff"] ** 2)),
        }
    if cfg.model == "backflow_ho":
        return {
            "orbital_alpha": p["alpha"],
            "backflow_same_rms": jnp.sqrt(jnp.mean(p["backflow_coeff_same"] ** 2)),
            "backflow_opposite_rms": jnp.sqrt(jnp.mean(p["backflow_coeff_opposite"] ** 2)),
        }
    raise ValueError(cfg.model)
