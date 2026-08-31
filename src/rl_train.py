from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from physics import SystemConfig
from wavefunction import WavefunctionConfig
from orbitals import OrbitalConfig
from correlation import CorrelationConfig
from rl import RLConfig, RLHistory, TD3Agent, ReplayBuffer, state_dim, action_dim
from rl_sampler import rl_update_one_electron_cached, run_rl_chain
from sampler import batch_log_probability, batch_grad_log_probability


def _safe_mean(values):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def train_rl_sampler(
    key,
    walkers,
    wf_params,
    system_cfg: SystemConfig,
    wf_cfg: WavefunctionConfig,
    orbital_cfg: OrbitalConfig,
    correlation_cfg: CorrelationConfig,
    rl_cfg: RLConfig,
    print_every: int = 10,
    actor_params=None,
    actor_target_params=None,
    actor_optimizer_state=None,
    history=None,
    sweep_offset: int = 0,
):
    """Train the shared actor while the current wavefunction block is frozen.

    The VMC wavefunction parameters are never changed in this stage.  The actor
    starts at ordinary MALA for ``scale_only`` and ``residual_mala`` because its
    final layer is initialized to zero.  After TD3 training, callers should
    freeze the returned actor and re-equilibrate a fresh production chain.
    """
    key_agent, key_chain, key_train = jax.random.split(key, 3)
    agent = TD3Agent(key_agent, rl_cfg, system_cfg)
    if actor_params is not None:
        agent.actor = actor_params
        agent.actor_target = (
            jax.tree_util.tree_map(lambda x: x.copy(), actor_params)
            if actor_target_params is None else actor_target_params
        )
        if actor_optimizer_state is not None:
            agent.actor_opt = actor_optimizer_state
    replay = ReplayBuffer(rl_cfg.replay_capacity, state_dim(rl_cfg, system_cfg), action_dim(rl_cfg))
    rng = np.random.default_rng(rl_cfg.seed)
    history = RLHistory() if history is None else history

    # Equilibrate with the initial actor.  For the main residual-MALA model this
    # is exactly the base MALA kernel because actor action == 0.
    if rl_cfg.burn_in_sweeps > 0:
        walkers, _, _, _ = run_rl_chain(
            key_chain, walkers, wf_params, agent.actor, rl_cfg.burn_in_sweeps,
            rl_cfg, system_cfg, wf_cfg, orbital_cfg, correlation_cfg,
            thin=max(1, rl_cfg.burn_in_sweeps), exploration_std=0.0,
        )

    # Cache the frozen target density and quantum force.  These are carried
    # through sequential electron updates, so each proposal only evaluates the
    # expensive wavefunction gradient for the proposed state.
    current_logp = batch_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )
    current_grad = batch_grad_log_probability(
        wf_params, walkers, system_cfg, wf_cfg, orbital_cfg, correlation_cfg
    )

    for sweep in range(rl_cfg.training_sweeps):
        frac = 0.0 if rl_cfg.training_sweeps <= 1 else sweep / (rl_cfg.training_sweeps - 1)
        exploration = (1.0 - frac) * rl_cfg.exploration_noise + frac * rl_cfg.exploration_noise_final

        reward_vals = []
        acc_vals = []
        alpha_vals = []
        prop_vals = []
        accepted_vals = []
        sigma_vals = []
        residual_vals = []
        q1_losses = []
        q2_losses = []
        actor_losses = []

        step_keys = jax.random.split(jax.random.fold_in(key_train, sweep), system_cfg.n_electrons)
        for i in range(system_cfg.n_electrons):
            walkers, current_logp, current_grad, batch = rl_update_one_electron_cached(
                step_keys[i], walkers, current_logp, current_grad,
                wf_params, agent.actor, i, rl_cfg, system_cfg, wf_cfg,
                orbital_cfg, correlation_cfg, exploration_std=float(exploration),
            )
            replay.add_batch(
                np.asarray(batch.state), np.asarray(batch.action), np.asarray(batch.reward),
                np.asarray(batch.next_state), np.asarray(batch.done)
            )

            reward_vals.append(np.mean(np.asarray(batch.reward)))
            acc_vals.append(np.mean(np.asarray(batch.acceptance)))
            alpha_vals.append(np.mean(np.asarray(batch.alpha)))
            prop_vals.append(np.mean(np.asarray(batch.proposed_displacement)))
            accepted_vals.append(np.mean(np.asarray(batch.accepted_displacement)))
            sigma_vals.append(np.mean(np.asarray(batch.sigma)))
            residual_vals.append(np.mean(np.abs(np.asarray(batch.residual))))

            if replay.size >= max(rl_cfg.learning_starts, rl_cfg.batch_size):
                for u in range(rl_cfg.updates_per_electron):
                    batch_rl = replay.sample(rng, rl_cfg.batch_size)
                    ku = jax.random.fold_in(step_keys[i], 10_000 + u + agent.update_count)
                    losses = agent.update(batch_rl, ku)
                    q1_losses.append(losses["critic1_loss"])
                    q2_losses.append(losses["critic2_loss"])
                    actor_losses.append(losses["actor_loss"])

        history.sweeps.append(int(sweep_offset + sweep))
        history.rewards.append(_safe_mean(reward_vals))
        history.acceptances.append(_safe_mean(acc_vals))
        history.mean_alpha.append(_safe_mean(alpha_vals))
        history.proposed_displacements.append(_safe_mean(prop_vals))
        history.accepted_displacements.append(_safe_mean(accepted_vals))
        history.sigmas.append(_safe_mean(sigma_vals))
        history.residuals.append(_safe_mean(residual_vals))
        history.critic1_losses.append(_safe_mean(q1_losses))
        history.critic2_losses.append(_safe_mean(q2_losses))
        history.actor_losses.append(_safe_mean(actor_losses))
        history.replay_sizes.append(int(replay.size))
        history.exploration_noises.append(float(exploration))

        if sweep % max(1, print_every) == 0 or sweep == rl_cfg.training_sweeps - 1:
            print(
                f"RL sweep={sweep:4d}  mode={rl_cfg.mode:14s}  "
                f"reward={history.rewards[-1]: .4f}  acc={history.acceptances[-1]:.3f}  "
                f"|dx|prop={history.proposed_displacements[-1]:.3f}  "
                f"|dx|acc={history.accepted_displacements[-1]:.3f}  "
                f"sigma={history.sigmas[-1]:.3f}  residual={history.residuals[-1]:.3f}  "
                f"buffer={replay.size}"
            )

    return {
        "actor_params": agent.actor,
        "actor_target_params": agent.actor_target,
        "critic1_params": agent.critic1,
        "critic2_params": agent.critic2,
        "walkers": walkers,
        "history": history,
        "replay_size": replay.size,
        "updates": agent.update_count,
        "actor_optimizer_state": agent.actor_opt,
        "replay_was_reset": True,
    }


def save_rl_training_metrics(history: RLHistory, path: str | Path):
    fields = [
        "sweep", "reward", "acceptance", "mean_alpha",
        "proposed_displacement", "accepted_displacement", "sigma", "residual_abs",
        "critic1_loss", "critic2_loss", "actor_loss", "replay_size", "exploration_noise",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, sweep in enumerate(history.sweeps):
            w.writerow({
                "sweep": sweep,
                "reward": history.rewards[i],
                "acceptance": history.acceptances[i],
                "mean_alpha": history.mean_alpha[i],
                "proposed_displacement": history.proposed_displacements[i],
                "accepted_displacement": history.accepted_displacements[i],
                "sigma": history.sigmas[i],
                "residual_abs": history.residuals[i],
                "critic1_loss": history.critic1_losses[i],
                "critic2_loss": history.critic2_losses[i],
                "actor_loss": history.actor_losses[i],
                "replay_size": history.replay_sizes[i],
                "exploration_noise": history.exploration_noises[i],
            })
