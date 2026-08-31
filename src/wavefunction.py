from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import jax
import jax.numpy as jnp

from src.physics import SystemConfig
from src.orbitals import (
    OrbitalConfig,
    init_orbital_params,
    determinant_logabs_sign,
    orbital_parameter_summary,
    physical_orbital_params,
)
from src.correlation import (
    CorrelationConfig,
    init_correlation_params,
    log_correlation_factor,
    correlation_parameter_summary,
    physical_correlation_params,
)

Array = jax.Array

WavefunctionName = Literal[
    "slater_fixed",
    "slater_varwidth",
    "slater_jastrow",
    "slater_spin_rbf",
    "slater_spin_rbf_threebody",
    "gaussian_orbitals_spin_rbf",
    "backflow_spin_rbf",
]


@dataclass(frozen=True)
class WavefunctionConfig:
    name: WavefunctionName = "slater_fixed"
    node_epsilon: float = 1e-14


def hierarchy_components(name: WavefunctionName) -> tuple[str, str]:
    mapping = {
        "slater_fixed": ("ho_fixed", "none"),
        "slater_varwidth": ("ho_variational", "none"),
        "slater_jastrow": ("ho_variational", "jastrow"),
        "slater_spin_rbf": ("ho_variational", "spin_rbf"),
        "slater_spin_rbf_threebody": ("ho_variational", "spin_rbf_threebody"),
        "gaussian_orbitals_spin_rbf": ("gaussian_basis", "spin_rbf"),
        "backflow_spin_rbf": ("backflow_ho", "spin_rbf"),
    }
    if name not in mapping:
        raise ValueError(f"Unknown wavefunction hierarchy member: {name}")
    return mapping[name]


def resolved_configs(
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> tuple[OrbitalConfig, CorrelationConfig]:
    """The named hierarchy sets model types while preserving numeric hyperparameters."""
    orb_model, corr_model = hierarchy_components(wf_cfg.name)
    orbital_cfg = OrbitalConfig(**{**orbital_cfg.__dict__, "model": orb_model})
    correlation_cfg = CorrelationConfig(**{**correlation_cfg.__dict__, "model": corr_model})
    return orbital_cfg, correlation_cfg


def init_wavefunction_params(
    key: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> dict:
    orbital_cfg, correlation_cfg = resolved_configs(wf_cfg, orbital_cfg, correlation_cfg)
    k1, k2 = jax.random.split(key)
    return {
        "orbitals": init_orbital_params(k1, system_cfg, orbital_cfg),
        "correlation": init_correlation_params(k2, correlation_cfg),
    }


def log_abs_psi(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    orbital_cfg, correlation_cfg = resolved_configs(wf_cfg, orbital_cfg, correlation_cfg)
    logdet, _ = determinant_logabs_sign(
        x,
        params["orbitals"],
        system_cfg,
        orbital_cfg,
        node_epsilon=wf_cfg.node_epsilon,
    )
    logcorr = log_correlation_factor(
        params["correlation"], x, system_cfg, correlation_cfg
    )
    return logdet + logcorr


def sign_psi(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    orbital_cfg, _ = resolved_configs(wf_cfg, orbital_cfg, correlation_cfg)
    _, sign = determinant_logabs_sign(
        x,
        params["orbitals"],
        system_cfg,
        orbital_cfg,
        node_epsilon=wf_cfg.node_epsilon,
    )
    return sign


def psi(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    return sign_psi(params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg) * jnp.exp(
        log_abs_psi(params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg)
    )


def log_probability(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    return 2.0 * log_abs_psi(
        params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )


def grad_log_probability_single(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    return jax.grad(
        lambda y: log_probability(
            params, y, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        )
    )(x)


def parameter_summary(
    params: dict,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> dict[str, Array]:
    orbital_cfg, correlation_cfg = resolved_configs(wf_cfg, orbital_cfg, correlation_cfg)
    out = {}
    out.update(orbital_parameter_summary(params["orbitals"], orbital_cfg))
    out.update(correlation_parameter_summary(params["correlation"], correlation_cfg))
    return out


def physical_wavefunction_params(
    params: dict,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> dict:
    orbital_cfg, correlation_cfg = resolved_configs(wf_cfg, orbital_cfg, correlation_cfg)
    return {
        "orbitals": physical_orbital_params(params["orbitals"], orbital_cfg),
        "correlation": physical_correlation_params(params["correlation"], correlation_cfg),
    }
