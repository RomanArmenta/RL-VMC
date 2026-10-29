from __future__ import annotations

from typing import Any, NamedTuple
import jax

Array = jax.Array


class SamplerState(NamedTuple):
    """Estado estandarizado del muestreador MCMC.

    walkers: shape (batch_size, n_elec, dim)
    log_psi: shape (batch_size,) -> log|Psi(walkers)|
    grad_log_psi: shape (batch_size, n_elec, dim) -> grad_R log|Psi(walkers)|
    """
    walkers: Array
    log_psi: Array
    grad_log_psi: Array


class StepStats(NamedTuple):
    """Estadísticas por paso de muestreo MCMC."""
    acceptance: Array
    alpha: Array
    proposed_disp: Array
    accepted_disp: Array
    esjd: Array
    actor_drift_mag: Array
    actor_sigma: Array
    log_q_fwd: Array
    log_p_diff: Array


class RLTransitionStep(NamedTuple):
    """Transición producida durante un paso de entrenamiento RL."""
    sampler_state: SamplerState
    state_old: Array
    action_old: Array
    reward: Array
    r_exploit: Array
    r_entropy: Array
    r_explore: Array
    next_state: Array
    stats: StepStats
