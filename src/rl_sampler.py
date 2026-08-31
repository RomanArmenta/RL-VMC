from __future__ import annotations
from functools import partial
from typing import NamedTuple
import jax
import jax.numpy as jnp

from src.physics import SystemConfig
from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.sampler import batch_log_probability, batch_grad_log_probability
from src.rl import (
    RLConfig,
    actor_apply,
    local_state_features_from_grad,
    proposal_parameters_from_action,
)

Array = jax.Array


class RLStepBatch(NamedTuple):
    state: Array
    action: Array
    reward: Array
    next_state: Array
    done: Array
    acceptance: Array
    alpha: Array
    proposed_displacement: Array
    accepted_displacement: Array
    proposed_squared_displacement: Array
    accepted_squared_displacement: Array
    sigma: Array
    residual: Array


class RLSweepMetrics(NamedTuple):
    acceptance: Array
    mean_alpha: Array
    proposed_displacement: Array
    accepted_displacement: Array
    proposed_squared_displacement: Array
    accepted_squared_displacement: Array
    sigma: Array
    residual_abs: Array


def _proposal_and_acceptance(
    key,
    walkers,
    current_logp,
    current_grad,
    wf_params,
    actor_params,
    electron_index,
    rl_cfg,
    system_cfg,
    wf_cfg,
    orbital_cfg,
    correlation_cfg,
    exploration_std,
):
    """Common exact Metropolis-Hastings calculation with cached old target data."""
    n_walkers = walkers.shape[0]
    i = electron_index
    key_action, key_prop, key_acc = jax.random.split(key, 3)

    state_old = local_state_features_from_grad(
        walkers, current_grad, i, system_cfg, rl_cfg
    )
    base_action_old = actor_apply(actor_params, state_old, rl_cfg, system_cfg)
    aux_noise = jnp.asarray(exploration_std, dtype=walkers.dtype) * jax.random.normal(
        key_action, base_action_old.shape, dtype=walkers.dtype
    )
    action_old = jnp.clip(base_action_old + aux_noise, -1.0, 1.0)

    old_coord = walkers[:, i]
    mean_fwd, sigma_fwd, residual_fwd = proposal_parameters_from_action(
        action_old, old_coord, current_grad[:, i], rl_cfg
    )
    proposed_coord = mean_fwd + sigma_fwd * jax.random.normal(
        key_prop, (n_walkers,), dtype=walkers.dtype
    )
    proposal = walkers.at[:, i].set(proposed_coord)

    new_logp = batch_log_probability(
        wf_params, proposal, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    new_grad = batch_grad_log_probability(
        wf_params, proposal, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    state_proposed = local_state_features_from_grad(
        proposal, new_grad, i, system_cfg, rl_cfg
    )
    # The same auxiliary action perturbation is used for forward and reverse
    # proposal densities. Conditional on it, detailed balance is exact.
    action_rev = jnp.clip(
        actor_apply(actor_params, state_proposed, rl_cfg, system_cfg) + aux_noise, -1.0, 1.0
    )
    mean_rev, sigma_rev, _ = proposal_parameters_from_action(
        action_rev, proposed_coord, new_grad[:, i], rl_cfg
    )

    z_fwd = (proposed_coord - mean_fwd) / sigma_fwd
    z_rev = (old_coord - mean_rev) / sigma_rev
    log_q_fwd = -jnp.log(sigma_fwd) - 0.5 * z_fwd**2
    log_q_rev = -jnp.log(sigma_rev) - 0.5 * z_rev**2
    log_alpha = (new_logp - current_logp) + (log_q_rev - log_q_fwd)
    clipped_log_alpha = jnp.minimum(0.0, log_alpha)
    alpha = jnp.exp(clipped_log_alpha)

    log_u = jnp.log(
        jax.random.uniform(
            key_acc, (n_walkers,), minval=1e-15, maxval=1.0,
            dtype=walkers.dtype,
        )
    )
    accept = log_u < clipped_log_alpha
    next_walkers = jnp.where(accept[:, None], proposal, walkers)
    next_logp = jnp.where(accept, new_logp, current_logp)
    next_grad = jnp.where(accept[:, None], new_grad, current_grad)

    proposed_disp = jnp.abs(proposed_coord - old_coord)
    accepted_disp = jnp.where(accept, proposed_disp, 0.0)
    proposed_sq = proposed_disp**2
    accepted_sq = jnp.where(accept, proposed_sq, 0.0)

    return (
        next_walkers,
        next_logp,
        next_grad,
        state_old,
        action_old,
        accept,
        alpha,
        proposed_disp,
        accepted_disp,
        proposed_sq,
        accepted_sq,
        sigma_fwd,
        residual_fwd,
    )


@partial(
    jax.jit,
    static_argnames=(
        "electron_index", "rl_cfg", "system_cfg", "wf_cfg", "orbital_cfg",
        "correlation_cfg",
    ),
)
def rl_update_one_electron_cached(
    key: Array,
    walkers: Array,
    current_logp: Array,
    current_grad: Array,
    wf_params: dict,
    actor_params,
    electron_index: int,
    rl_cfg: RLConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    exploration_std: float = 0.0,
) -> tuple[Array, Array, Array, RLStepBatch]:
    """Training update that reuses old log-probability and quantum-force values."""
    i = int(electron_index)
    next_i = (i + 1) % system_cfg.n_electrons
    (
        next_walkers, next_logp, next_grad, state_old, action_old,
        accept, alpha, proposed_disp, accepted_disp, proposed_sq, accepted_sq,
        sigma_fwd, residual_fwd,
    ) = _proposal_and_acceptance(
        key, walkers, current_logp, current_grad, wf_params, actor_params, i,
        rl_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg, exploration_std,
    )

    next_state = local_state_features_from_grad(
        next_walkers, next_grad, next_i, system_cfg, rl_cfg
    )
    reward = (
        rl_cfg.reward_displacement_weight
        * 2.0 * jnp.log(proposed_disp + rl_cfg.reward_epsilon)
        + rl_cfg.reward_acceptance_weight
        * jnp.log(alpha + rl_cfg.reward_epsilon)
    )
    if rl_cfg.reward_clip > 0:
        reward = jnp.clip(reward, -rl_cfg.reward_clip, rl_cfg.reward_clip)

    batch = RLStepBatch(
        state=state_old,
        action=action_old,
        reward=reward,
        next_state=next_state,
        done=jnp.zeros((walkers.shape[0],), dtype=walkers.dtype),
        acceptance=accept,
        alpha=alpha,
        proposed_displacement=proposed_disp,
        accepted_displacement=accepted_disp,
        proposed_squared_displacement=proposed_sq,
        accepted_squared_displacement=accepted_sq,
        sigma=sigma_fwd,
        residual=residual_fwd,
    )
    return next_walkers, next_logp, next_grad, batch


def rl_update_one_electron(
    key: Array,
    walkers: Array,
    wf_params: dict,
    actor_params,
    electron_index: int,
    rl_cfg: RLConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    exploration_std: float = 0.0,
) -> tuple[Array, RLStepBatch]:
    """Backward-compatible one-electron update.

    Training code should prefer ``rl_update_one_electron_cached`` so old target
    quantities are not recomputed for every electron.
    """
    current_logp = batch_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    current_grad = batch_grad_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    next_walkers, _, _, batch = rl_update_one_electron_cached(
        key, walkers, current_logp, current_grad, wf_params, actor_params,
        electron_index, rl_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg,
        exploration_std,
    )
    return next_walkers, batch


def _production_rl_sweep_cached(
    key,
    walkers,
    current_logp,
    current_grad,
    wf_params,
    actor_params,
    rl_cfg,
    system_cfg,
    wf_cfg,
    orbital_cfg,
    correlation_cfg,
    exploration_std,
):
    keys = jax.random.split(key, system_cfg.n_electrons)
    rows = []
    for i in range(system_cfg.n_electrons):
        (
            walkers, current_logp, current_grad, _, _, accept, alpha,
            proposed_disp, accepted_disp, proposed_sq, accepted_sq,
            sigma, residual,
        ) = _proposal_and_acceptance(
            keys[i], walkers, current_logp, current_grad, wf_params, actor_params,
            i, rl_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg,
            exploration_std,
        )
        rows.append(jnp.stack([
            jnp.mean(accept.astype(walkers.dtype)),
            jnp.mean(alpha),
            jnp.mean(proposed_disp),
            jnp.mean(accepted_disp),
            jnp.mean(proposed_sq),
            jnp.mean(accepted_sq),
            jnp.mean(sigma),
            jnp.mean(jnp.abs(residual)),
        ]))
    m = jnp.mean(jnp.stack(rows, axis=0), axis=0)
    metrics = RLSweepMetrics(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]
    )
    return walkers, current_logp, current_grad, metrics


@partial(
    jax.jit,
    static_argnames=(
        "n_sweeps", "rl_cfg", "system_cfg", "wf_cfg", "orbital_cfg",
        "correlation_cfg",
    ),
)
def _run_rl_chain_device(
    key,
    walkers,
    wf_params,
    actor_params,
    n_sweeps: int,
    rl_cfg: RLConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    exploration_std: float,
):
    current_logp = batch_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    current_grad = batch_grad_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )

    def body(carry, _):
        rng, w, lp, grad = carry
        rng, subkey = jax.random.split(rng)
        w, lp, grad, metrics = _production_rl_sweep_cached(
            subkey, w, lp, grad, wf_params, actor_params, rl_cfg, system_cfg,
            wf_cfg, orbital_cfg, correlation_cfg, exploration_std,
        )
        return (rng, w, lp, grad), (w, metrics)

    (key_out, walkers_out, _, _), (trajectory, metrics) = jax.lax.scan(
        body, (key, walkers, current_logp, current_grad), xs=None, length=n_sweeps
    )
    return key_out, walkers_out, trajectory, metrics


def run_rl_chain(
    key: Array,
    walkers: Array,
    wf_params: dict,
    actor_params,
    n_sweeps: int,
    rl_cfg: RLConfig,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    thin: int = 1,
    exploration_std: float = 0.0,
):
    """Frozen-actor chain compiled through ``jax.lax.scan``.

    One sweep remains N sequential single-electron Metropolis-Hastings updates;
    all walkers are vectorized on the active accelerator.
    """
    n_sweeps = int(n_sweeps)
    if n_sweeps < 1:
        raise ValueError("n_sweeps must be >= 1")
    thin = max(1, int(thin))
    chunk = max(1, int(rl_cfg.jit_sweep_chunk))
    rng = key
    remaining = n_sweeps
    trajectory_blocks = []
    metric_blocks = []
    exploration = jnp.asarray(exploration_std, dtype=walkers.dtype)
    while remaining > 0:
        block_len = min(chunk, remaining)
        rng, walkers, block_traj, block_metrics = _run_rl_chain_device(
            rng, walkers, wf_params, actor_params, block_len, rl_cfg, system_cfg,
            wf_cfg, orbital_cfg, correlation_cfg, exploration,
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
        sweep_metrics = RLSweepMetrics(*[
            jnp.concatenate([getattr(m, field) for m in metric_blocks], axis=0)
            for field in RLSweepMetrics._fields
        ])

    if thin == 1:
        trajectory = full_trajectory
    else:
        trajectory = full_trajectory[thin - 1 :: thin]
        if trajectory.shape[0] == 0:
            trajectory = full_trajectory[-1:]
    samples = trajectory.reshape((-1, system_cfg.n_electrons))

    per_electron_esjd = jnp.mean(sweep_metrics.accepted_squared_displacement)
    metrics = {
        "acceptance": jnp.mean(sweep_metrics.acceptance),
        "mean_alpha": jnp.mean(sweep_metrics.mean_alpha),
        "proposed_displacement": jnp.mean(sweep_metrics.proposed_displacement),
        "accepted_displacement": jnp.mean(sweep_metrics.accepted_displacement),
        "proposed_squared_displacement": jnp.mean(
            sweep_metrics.proposed_squared_displacement
        ),
        "accepted_squared_displacement": per_electron_esjd,
        "esjd_per_electron_update": per_electron_esjd,
        "esjd_per_sweep": per_electron_esjd * system_cfg.n_electrons,
        "sigma": jnp.mean(sweep_metrics.sigma),
        "residual_abs": jnp.mean(sweep_metrics.residual_abs),
        "sweep_sigma": sweep_metrics.sigma,
        "sweep_residual_abs": sweep_metrics.residual_abs,
        "sweep_acceptance": sweep_metrics.acceptance,
        "sweep_mean_alpha": sweep_metrics.mean_alpha,
        "sweep_proposed_displacement": sweep_metrics.proposed_displacement,
        "sweep_accepted_displacement": sweep_metrics.accepted_displacement,
        "sweep_proposed_squared_displacement": sweep_metrics.proposed_squared_displacement,
        "sweep_accepted_squared_displacement": sweep_metrics.accepted_squared_displacement,
    }
    return walkers, samples, trajectory, metrics
