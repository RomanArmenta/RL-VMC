from __future__ import annotations
from functools import partial
import jax
import jax.numpy as jnp

from src.physics import SystemConfig, potential_energy
from src.wavefunction import WavefunctionConfig, log_abs_psi
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig

Array = jax.Array


def local_energy_single(
    params: dict,
    x: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    r"""Local energy E_L = -1/2 sum_i[d_i^2 log|Psi|+(d_i log|Psi|)^2]+V."""
    f = lambda y: log_abs_psi(
        params, y, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    grad_log = jax.grad(f)(x)
    hess_log = jax.hessian(f)(x)
    kinetic = -0.5 * (jnp.trace(hess_log) + jnp.sum(grad_log**2))
    return kinetic + potential_energy(x, system_cfg)


@partial(
    jax.jit,
    static_argnames=("system_cfg", "wf_cfg", "orbital_cfg", "correlation_cfg"),
)
def local_energies(
    params: dict,
    samples: Array,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
) -> Array:
    """Vectorized and accelerator-compiled local energies."""
    return jax.vmap(
        lambda x: local_energy_single(
            params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        )
    )(samples)
