from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import itertools
import math
import jax
import jax.numpy as jnp

from src.physics import SystemConfig, pair_distances, pair_spin_masks

Array = jax.Array
CorrelationModel = Literal[
    "none",
    "jastrow",
    "spin_rbf",
    "spin_rbf_jastrow",
    "spin_rbf_threebody",
]


def _softplus_inverse(y: float) -> float:
    if y <= 0:
        raise ValueError("softplus inverse requires y > 0")
    return math.log(math.expm1(float(y)))


@dataclass(frozen=True)
class CorrelationConfig:
    model: CorrelationModel = "none"
    min_positive: float = 1e-6
    pair_smoothing: float = 1e-4

    # Jastrow initialization, independent same/opposite spin channels.
    jastrow_c_same: float = 0.15
    jastrow_c_opposite: float = 0.30
    jastrow_beta_same: float = 0.6
    jastrow_beta_opposite: float = 0.5

    # Positive pair RBF factors: f_s(r)=1+sum_k a_sk G_k(r), a_sk>0.
    rbf_components: int = 10
    rbf_r_max: float = 5.0
    rbf_width_init: float = 0.6
    rbf_coeff_init: float = 0.08

    # Positive symmetric three-electron hyperradial factor.
    threebody_components: int = 6
    threebody_q_max: float = 8.0
    threebody_width_init: float = 1.0
    threebody_coeff_init: float = 0.02


def init_correlation_params(key: Array, cfg: CorrelationConfig) -> dict:
    eps = cfg.min_positive
    if cfg.model == "none":
        return {}

    params: dict[str, Array] = {}
    if cfg.model in ("jastrow", "spin_rbf_jastrow"):
        params.update({
            "raw_c_same": jnp.asarray(_softplus_inverse(max(cfg.jastrow_c_same - eps, eps))),
            "raw_c_opposite": jnp.asarray(_softplus_inverse(max(cfg.jastrow_c_opposite - eps, eps))),
            "raw_beta_same": jnp.asarray(_softplus_inverse(max(cfg.jastrow_beta_same - eps, eps))),
            "raw_beta_opposite": jnp.asarray(_softplus_inverse(max(cfg.jastrow_beta_opposite - eps, eps))),
        })

    if cfg.model in ("spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"):
        K = cfg.rbf_components
        if K < 1:
            raise ValueError("rbf_components must be >= 1")
        raw_a = _softplus_inverse(max(cfg.rbf_coeff_init - eps, eps))
        params.update({
            "raw_pair_coeff_same": jnp.full((K,), raw_a),
            "raw_pair_coeff_opposite": jnp.full((K,), raw_a),
            "raw_pair_width_same": jnp.asarray(_softplus_inverse(max(cfg.rbf_width_init - eps, eps))),
            "raw_pair_width_opposite": jnp.asarray(_softplus_inverse(max(cfg.rbf_width_init - eps, eps))),
        })

    if cfg.model == "spin_rbf_threebody":
        K3 = cfg.threebody_components
        raw_a3 = _softplus_inverse(max(cfg.threebody_coeff_init - eps, eps))
        params.update({
            "raw_threebody_coeff": jnp.full((K3,), raw_a3),
            "raw_threebody_width": jnp.asarray(_softplus_inverse(max(cfg.threebody_width_init - eps, eps))),
        })
    return params


def physical_correlation_params(params: dict, cfg: CorrelationConfig) -> dict:
    eps = cfg.min_positive
    out = {}
    if cfg.model in ("jastrow", "spin_rbf_jastrow"):
        out.update({
            "c_same": jax.nn.softplus(params["raw_c_same"]) + eps,
            "c_opposite": jax.nn.softplus(params["raw_c_opposite"]) + eps,
            "beta_same": jax.nn.softplus(params["raw_beta_same"]) + eps,
            "beta_opposite": jax.nn.softplus(params["raw_beta_opposite"]) + eps,
        })
    if cfg.model in ("spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"):
        out.update({
            "pair_coeff_same": jax.nn.softplus(params["raw_pair_coeff_same"]) + eps,
            "pair_coeff_opposite": jax.nn.softplus(params["raw_pair_coeff_opposite"]) + eps,
            "pair_width_same": jax.nn.softplus(params["raw_pair_width_same"]) + eps,
            "pair_width_opposite": jax.nn.softplus(params["raw_pair_width_opposite"]) + eps,
        })
    if cfg.model == "spin_rbf_threebody":
        out.update({
            "threebody_coeff": jax.nn.softplus(params["raw_threebody_coeff"]) + eps,
            "threebody_width": jax.nn.softplus(params["raw_threebody_width"]) + eps,
        })
    return out


def _jastrow_log(params: dict, x: Array, system_cfg: SystemConfig, cfg: CorrelationConfig) -> Array:
    p = physical_correlation_params(params, cfg)
    r = pair_distances(x, system_cfg)
    same, opposite = pair_spin_masks(system_cfg)
    rho = jnp.sqrt(r**2 + cfg.pair_smoothing**2)
    u_same = p["c_same"] * rho / (1.0 + p["beta_same"] * rho)
    u_opp = p["c_opposite"] * rho / (1.0 + p["beta_opposite"] * rho)
    return jnp.sum(jnp.where(same, u_same, u_opp), axis=-1)


def pair_factor_curves(params: dict, r: Array, cfg: CorrelationConfig) -> tuple[Array, Array]:
    """Positive same/opposite-spin pair RBF factors on an arbitrary r-grid."""
    p = physical_correlation_params(params, cfg)
    centers = jnp.linspace(0.0, cfg.rbf_r_max, cfg.rbf_components)
    r = jnp.asarray(r)[..., None]
    gs = jnp.exp(-0.5 * ((r - centers) / p["pair_width_same"]) ** 2)
    go = jnp.exp(-0.5 * ((r - centers) / p["pair_width_opposite"]) ** 2)
    fs = 1.0 + jnp.sum(p["pair_coeff_same"] * gs, axis=-1)
    fo = 1.0 + jnp.sum(p["pair_coeff_opposite"] * go, axis=-1)
    return fs, fo


def _spin_rbf_log(params: dict, x: Array, system_cfg: SystemConfig, cfg: CorrelationConfig) -> Array:
    r = pair_distances(x, system_cfg)
    same, _ = pair_spin_masks(system_cfg)
    fs, fo = pair_factor_curves(params, r, cfg)
    f = jnp.where(same, fs, fo)
    return jnp.sum(jnp.log(f), axis=-1)


def _threebody_log(params: dict, x: Array, system_cfg: SystemConfig, cfg: CorrelationConfig) -> Array:
    if system_cfg.n_electrons < 3:
        return jnp.asarray(0.0, dtype=x.dtype)
    p = physical_correlation_params(params, cfg)
    centers = jnp.linspace(0.0, cfg.threebody_q_max, cfg.threebody_components)
    terms = []
    for i, j, k in itertools.combinations(range(system_cfg.n_electrons), 3):
        rij = jnp.abs(x[i] - x[j])
        rik = jnp.abs(x[i] - x[k])
        rjk = jnp.abs(x[j] - x[k])
        q = jnp.sqrt(rij**2 + rik**2 + rjk**2)
        basis = jnp.exp(-0.5 * ((q - centers) / p["threebody_width"]) ** 2)
        factor = 1.0 + jnp.sum(p["threebody_coeff"] * basis)
        terms.append(jnp.log(factor))
    return jnp.sum(jnp.stack(terms)) if terms else jnp.asarray(0.0, dtype=x.dtype)


def log_correlation_factor(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    cfg: CorrelationConfig,
) -> Array:
    """Return log F_corr. F_corr is always positive."""
    if cfg.model == "none":
        return jnp.asarray(0.0, dtype=x.dtype)
    if cfg.model == "jastrow":
        return _jastrow_log(params, x, system_cfg, cfg)
    if cfg.model == "spin_rbf":
        return _spin_rbf_log(params, x, system_cfg, cfg)
    if cfg.model == "spin_rbf_jastrow":
        return _spin_rbf_log(params, x, system_cfg, cfg) + _jastrow_log(params, x, system_cfg, cfg)
    if cfg.model == "spin_rbf_threebody":
        return _spin_rbf_log(params, x, system_cfg, cfg) + _threebody_log(params, x, system_cfg, cfg)
    raise ValueError(f"Unknown correlation model: {cfg.model}")


def correlation_parameter_summary(params: dict, cfg: CorrelationConfig) -> dict[str, Array]:
    if cfg.model == "none":
        return {}
    p = physical_correlation_params(params, cfg)
    out = {}
    if cfg.model in ("jastrow", "spin_rbf_jastrow"):
        out.update({
            "jastrow_c_same": p["c_same"],
            "jastrow_c_opposite": p["c_opposite"],
            "jastrow_beta_same": p["beta_same"],
            "jastrow_beta_opposite": p["beta_opposite"],
        })
    if cfg.model in ("spin_rbf", "spin_rbf_jastrow", "spin_rbf_threebody"):
        out.update({
            "pair_coeff_same_mean": jnp.mean(p["pair_coeff_same"]),
            "pair_coeff_opposite_mean": jnp.mean(p["pair_coeff_opposite"]),
            "pair_width_same": p["pair_width_same"],
            "pair_width_opposite": p["pair_width_opposite"],
        })
    if cfg.model == "spin_rbf_threebody":
        out.update({
            "threebody_coeff_mean": jnp.mean(p["threebody_coeff"]),
            "threebody_width": p["threebody_width"],
        })
    return out
