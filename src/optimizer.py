from __future__ import annotations
from functools import partial
from typing import NamedTuple, Any
import jax
import jax.numpy as jnp

from physics import SystemConfig
from wavefunction import WavefunctionConfig, log_abs_psi
from orbitals import OrbitalConfig
from correlation import CorrelationConfig
from local_energy import local_energies


class AdamState(NamedTuple):
    """JAX-pytree Adam state so optimizer steps can themselves be JIT compiled."""
    m: Any
    v: Any
    t: Any


def _zeros_like_tree(tree):
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def adam_init(params):
    return AdamState(
        _zeros_like_tree(params), _zeros_like_tree(params),
        jnp.asarray(0, dtype=jnp.int32),
    )


def adam_update(
    params, grads, state, learning_rate=1e-2,
    beta1=0.9, beta2=0.999, eps=1e-8,
):
    t = state.t + jnp.asarray(1, dtype=state.t.dtype)
    m = jax.tree_util.tree_map(
        lambda m_, g_: beta1 * m_ + (1 - beta1) * g_, state.m, grads
    )
    v = jax.tree_util.tree_map(
        lambda v_, g_: beta2 * v_ + (1 - beta2) * (g_**2), state.v, grads
    )
    b1corr = 1.0 - beta1**t
    b2corr = 1.0 - beta2**t
    mhat = jax.tree_util.tree_map(lambda z: z / b1corr, m)
    vhat = jax.tree_util.tree_map(lambda z: z / b2corr, v)
    new_params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - learning_rate * mm / (jnp.sqrt(vv) + eps),
        params, mhat, vhat,
    )
    return new_params, AdamState(m, v, t)


@partial(
    jax.jit,
    static_argnames=("system_cfg", "wf_cfg", "orbital_cfg", "correlation_cfg"),
)
def vmc_energy_gradient(
    params,
    samples,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    """Compiled VMC covariance gradient and local-energy statistics."""
    e_loc = local_energies(
        params, samples, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    score_fn = jax.grad(
        lambda p, x: log_abs_psi(
            p, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        ),
        argnums=0,
    )
    scores = jax.vmap(lambda x: score_fn(params, x))(samples)
    e_centered = e_loc - jnp.mean(e_loc)
    n = e_centered.shape[0]

    def component_grad(score_array):
        centered = score_array - jnp.mean(score_array, axis=0)
        shape = (n,) + (1,) * (score_array.ndim - 1)
        return 2.0 * jnp.mean(e_centered.reshape(shape) * centered, axis=0)

    grads = jax.tree_util.tree_map(component_grad, scores)
    return grads, {
        "energy_mean": jnp.mean(e_loc),
        "energy_std": jnp.std(e_loc),
        "energy_variance": jnp.var(e_loc),
    }, e_loc
