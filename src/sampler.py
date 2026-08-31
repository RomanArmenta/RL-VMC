from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple
import jax
import jax.numpy as jnp

from physics import SystemConfig
from wavefunction import WavefunctionConfig, log_probability, grad_log_probability_single
from orbitals import OrbitalConfig
from correlation import CorrelationConfig

Array = jax.Array
SamplerName = Literal["metropolis", "mala"]


@dataclass(frozen=True)
class SamplerConfig:
    sampler: SamplerName = "metropolis"
    n_walkers: int = 128
    step_size: float = 0.55
    sweeps_per_iteration: int = 10
    burn_in_sweeps: int = 80
    thin: int = 2
    mala_drift_clip: float = 10.0
    jit_sweep_chunk: int = 32


class SamplerSweepMetrics(NamedTuple):
    """Averages over all walkers and all N single-electron updates in one sweep."""
    acceptance: Array
    mean_alpha: Array
    proposed_displacement: Array
    accepted_displacement: Array
    proposed_squared_displacement: Array
    accepted_squared_displacement: Array


def initialize_walkers(
    key: Array,
    system_cfg: SystemConfig,
    n_walkers: int,
    scale: float = 1.0,
) -> Array:
    """Initialize full N-electron configurations on the active JAX device."""
    x = scale * jax.random.normal(key, (n_walkers, system_cfg.n_electrons))
    offsets = jnp.linspace(-0.6, 0.6, system_cfg.n_electrons, dtype=x.dtype)
    return x + offsets[None, :]


@partial(
    jax.jit,
    static_argnames=("system_cfg", "wf_cfg", "orbital_cfg", "correlation_cfg"),
)
def batch_log_probability(
    params,
    walkers,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    """Vectorized log |Psi|^2 for all walkers, compiled for CPU/GPU."""
    return jax.vmap(
        lambda x: log_probability(
            params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        )
    )(walkers)


@partial(
    jax.jit,
    static_argnames=("system_cfg", "wf_cfg", "orbital_cfg", "correlation_cfg"),
)
def batch_grad_log_probability(
    params,
    walkers,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    """Vectorized quantum force grad log |Psi|^2 for all walkers."""
    return jax.vmap(
        lambda x: grad_log_probability_single(
            params, x, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        )
    )(walkers)


def _metric_tuple(accept, alpha, displacement):
    accepted = jnp.where(accept, displacement, 0.0)
    proposed_sq = displacement * displacement
    accepted_sq = jnp.where(accept, proposed_sq, 0.0)
    return (
        jnp.mean(accept.astype(displacement.dtype)),
        jnp.mean(alpha),
        jnp.mean(displacement),
        jnp.mean(accepted),
        jnp.mean(proposed_sq),
        jnp.mean(accepted_sq),
    )


def _rw_update_cached(
    key,
    walkers,
    current_logp,
    current_grad,
    params,
    electron_index,
    sampler_cfg,
    system_cfg,
    wf_cfg,
    orbital_cfg,
    correlation_cfg,
):
    """Random-walk update carrying the current log probability between electrons."""
    key_prop, key_acc = jax.random.split(key)
    delta = sampler_cfg.step_size * jax.random.normal(
        key_prop, (walkers.shape[0],), dtype=walkers.dtype
    )
    proposal = walkers.at[:, electron_index].add(delta)
    new_logp = batch_log_probability(
        params, proposal, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    log_alpha = new_logp - current_logp
    clipped = jnp.minimum(0.0, log_alpha)
    alpha = jnp.exp(clipped)
    log_u = jnp.log(
        jax.random.uniform(
            key_acc, (walkers.shape[0],), minval=1e-15, maxval=1.0,
            dtype=walkers.dtype,
        )
    )
    accept = log_u < clipped
    next_walkers = jnp.where(accept[:, None], proposal, walkers)
    next_logp = jnp.where(accept, new_logp, current_logp)
    displacement = jnp.abs(delta)
    metrics = _metric_tuple(accept, alpha, displacement)
    return next_walkers, next_logp, current_grad, metrics


def _mala_update_cached(
    key,
    walkers,
    current_logp,
    current_grad,
    params,
    electron_index,
    sampler_cfg,
    system_cfg,
    wf_cfg,
    orbital_cfg,
    correlation_cfg,
):
    """MALA update carrying log probability and quantum force between electrons.

    This avoids recomputing the *old* log probability and gradient for every
    electron. Only the proposed configuration requires a fresh evaluation.
    """
    key_prop, key_acc = jax.random.split(key)
    eps = jnp.asarray(sampler_cfg.step_size, dtype=walkers.dtype)
    grad_old_i = current_grad[:, electron_index]
    drift_old = 0.5 * eps**2 * grad_old_i
    if sampler_cfg.mala_drift_clip > 0:
        drift_old = jnp.clip(
            drift_old, -sampler_cfg.mala_drift_clip, sampler_cfg.mala_drift_clip
        )

    old_coord = walkers[:, electron_index]
    mean_fwd = old_coord + drift_old
    proposed_coordinate = mean_fwd + eps * jax.random.normal(
        key_prop, (walkers.shape[0],), dtype=walkers.dtype
    )
    proposal = walkers.at[:, electron_index].set(proposed_coordinate)

    new_logp = batch_log_probability(
        params, proposal, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    new_grad = batch_grad_log_probability(
        params, proposal, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    drift_new = 0.5 * eps**2 * new_grad[:, electron_index]
    if sampler_cfg.mala_drift_clip > 0:
        drift_new = jnp.clip(
            drift_new, -sampler_cfg.mala_drift_clip, sampler_cfg.mala_drift_clip
        )
    mean_rev = proposed_coordinate + drift_new

    inv_var = 1.0 / eps**2
    log_q_fwd = -0.5 * (proposed_coordinate - mean_fwd) ** 2 * inv_var
    log_q_rev = -0.5 * (old_coord - mean_rev) ** 2 * inv_var
    log_alpha = (new_logp - current_logp) + (log_q_rev - log_q_fwd)
    clipped = jnp.minimum(0.0, log_alpha)
    alpha = jnp.exp(clipped)

    log_u = jnp.log(
        jax.random.uniform(
            key_acc, (walkers.shape[0],), minval=1e-15, maxval=1.0,
            dtype=walkers.dtype,
        )
    )
    accept = log_u < clipped
    next_walkers = jnp.where(accept[:, None], proposal, walkers)
    next_logp = jnp.where(accept, new_logp, current_logp)
    next_grad = jnp.where(accept[:, None], new_grad, current_grad)
    displacement = jnp.abs(proposed_coordinate - old_coord)
    metrics = _metric_tuple(accept, alpha, displacement)
    return next_walkers, next_logp, next_grad, metrics


def _sampler_sweep_cached(
    key,
    walkers,
    current_logp,
    current_grad,
    params,
    sampler_cfg: SamplerConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    keys = jax.random.split(key, system_cfg.n_electrons)
    metric_rows = []
    for i in range(system_cfg.n_electrons):
        if sampler_cfg.sampler == "metropolis":
            walkers, current_logp, current_grad, row = _rw_update_cached(
                keys[i], walkers, current_logp, current_grad, params, i,
                sampler_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg,
            )
        elif sampler_cfg.sampler == "mala":
            walkers, current_logp, current_grad, row = _mala_update_cached(
                keys[i], walkers, current_logp, current_grad, params, i,
                sampler_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler_cfg.sampler}")
        metric_rows.append(jnp.stack(row))
    m = jnp.mean(jnp.stack(metric_rows, axis=0), axis=0)
    return walkers, current_logp, current_grad, SamplerSweepMetrics(*tuple(m))


@partial(
    jax.jit,
    static_argnames=(
        "n_sweeps", "sampler_cfg", "system_cfg", "wf_cfg", "orbital_cfg",
        "correlation_cfg",
    ),
)
def _run_chain_device(
    key,
    walkers,
    params,
    n_sweeps: int,
    sampler_cfg: SamplerConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    """Entire Markov chain kept on device and compiled through lax.scan."""
    current_logp = batch_log_probability(
        params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    if sampler_cfg.sampler == "mala":
        current_grad = batch_grad_log_probability(
            params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
        )
    else:
        current_grad = jnp.zeros_like(walkers)

    def body(carry, _):
        rng, w, lp, grad = carry
        rng, subkey = jax.random.split(rng)
        w, lp, grad, metrics = _sampler_sweep_cached(
            subkey, w, lp, grad, params, sampler_cfg, system_cfg,
            wf_cfg, orbital_cfg, correlation_cfg,
        )
        return (rng, w, lp, grad), (w, metrics)

    (key_out, walkers_out, _, _), (trajectory, metrics) = jax.lax.scan(
        body, (key, walkers, current_logp, current_grad), xs=None, length=n_sweeps
    )
    return key_out, walkers_out, trajectory, metrics


def sampler_sweep(
    key,
    walkers,
    params,
    sampler_cfg: SamplerConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
):
    """Backward-compatible one-sweep interface."""
    _, walkers, _, metrics = _run_chain_device(
        key, walkers, params, 1, sampler_cfg, system_cfg, wf_cfg,
        orbital_cfg, correlation_cfg,
    )
    return walkers, metrics.acceptance[0]


def run_chain(
    key,
    walkers,
    params,
    n_sweeps: int,
    sampler_cfg: SamplerConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    thin: int | None = None,
    return_metrics: bool = False,
):
    """Run a chain with the sweep loop compiled on the active JAX device.

    ``return_metrics=False`` preserves the historical four-value return:
    ``walkers, samples, trajectory, mean_acceptance``.

    With ``return_metrics=True`` the fourth value is a dictionary containing
    acceptance, mean acceptance probability, displacement, and ESJD-related
    statistics.  ESJD means *expected squared jump distance*.
    """
    n_sweeps = int(n_sweeps)
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be >= 1")
    thin = sampler_cfg.thin if thin is None else max(1, int(thin))
    # Run fixed-size compiled chunks.  This avoids one giant compilation for
    # every distinct chain length while still keeping the hot sweep loop on GPU.
    chunk = max(1, int(sampler_cfg.jit_sweep_chunk))
    rng = key
    remaining = n_sweeps
    trajectory_blocks = []
    metric_blocks = []
    while remaining > 0:
        block_len = min(chunk, remaining)
        rng, walkers, block_traj, block_metrics = _run_chain_device(
            rng, walkers, params, block_len, sampler_cfg, system_cfg, wf_cfg,
            orbital_cfg, correlation_cfg,
        )
        trajectory_blocks.append(block_traj)
        metric_blocks.append(block_metrics)
        remaining -= block_len

    full_trajectory = (
        trajectory_blocks[0]
        if len(trajectory_blocks) == 1
        else jnp.concatenate(trajectory_blocks, axis=0)
    )
    if len(metric_blocks) == 1:
        sweep_metrics = metric_blocks[0]
    else:
        sweep_metrics = SamplerSweepMetrics(*[
            jnp.concatenate([getattr(m, field) for m in metric_blocks], axis=0)
            for field in SamplerSweepMetrics._fields
        ])

    if thin == 1:
        trajectory = full_trajectory
    else:
        trajectory = full_trajectory[thin - 1 :: thin]
        if trajectory.shape[0] == 0:
            trajectory = full_trajectory[-1:]
    samples = trajectory.reshape((-1, system_cfg.n_electrons))

    mean_acceptance = jnp.mean(sweep_metrics.acceptance)
    if not return_metrics:
        return walkers, samples, trajectory, mean_acceptance

    # Per-electron ESJD is the average squared realized single-electron move.
    # Since each coordinate is updated exactly once per sweep, multiplying by N
    # gives the sum of squared coordinate changes over one complete sweep.
    per_electron_esjd = jnp.mean(sweep_metrics.accepted_squared_displacement)
    metrics = {
        "acceptance": mean_acceptance,
        "mean_alpha": jnp.mean(sweep_metrics.mean_alpha),
        "proposed_displacement": jnp.mean(sweep_metrics.proposed_displacement),
        "accepted_displacement": jnp.mean(sweep_metrics.accepted_displacement),
        "proposed_squared_displacement": jnp.mean(
            sweep_metrics.proposed_squared_displacement
        ),
        "accepted_squared_displacement": per_electron_esjd,
        "esjd_per_electron_update": per_electron_esjd,
        "esjd_per_sweep": per_electron_esjd * system_cfg.n_electrons,
        "sweep_acceptance": sweep_metrics.acceptance,
        "sweep_mean_alpha": sweep_metrics.mean_alpha,
        "sweep_proposed_displacement": sweep_metrics.proposed_displacement,
        "sweep_accepted_displacement": sweep_metrics.accepted_displacement,
        "sweep_proposed_squared_displacement": sweep_metrics.proposed_squared_displacement,
        "sweep_accepted_squared_displacement": sweep_metrics.accepted_squared_displacement,
    }
    return walkers, samples, trajectory, metrics
