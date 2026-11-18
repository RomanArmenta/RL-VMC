from __future__ import annotations

import functools
import numpy as np
import jax
import jax.numpy as jnp
import optax
from rl_vmc.core.optimizers import build_lr_schedule

from rl_vmc.config import Experiment1DConfig
from rl_vmc.ansatze.ansatz_1d import BaseAnsatz1D
from rl_vmc.proposals.proposals_1d import (
    DeepSetNeuralProposal1D,
    FlatMLPNeuralProposal1D,
    gaussian_log_prob,
    _ensure_flax_vars,
)
from rl_vmc.samplers.mh_sampler_1d import MHSampler1D
from rl_vmc.rl.rewards import compute_rl_reward
from flax import linen as nn


class ReplayBuffer:
    """Buffer de experiencia circular basado en NumPy para entrenamiento off-policy TD3."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add_batch(self, state, action, reward, next_state):
        s, a, r, ns = np.asarray(state), np.asarray(action), np.asarray(reward), np.asarray(next_state)
        n = len(r)
        if n >= self.capacity:
            s, a, r, ns = s[-self.capacity :], a[-self.capacity :], r[-self.capacity :], ns[-self.capacity :]
            n = self.capacity
        idx = (np.arange(n) + self.pos) % self.capacity
        self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx] = s, a, r, ns
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int) -> dict[str, jax.Array]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "state": jnp.asarray(self.state[idx]),
            "action": jnp.asarray(self.action[idx]),
            "reward": jnp.asarray(self.reward[idx]),
            "next_state": jnp.asarray(self.next_state[idx]),
        }


class FlaxTD3Critic(nn.Module):
    hidden_dim_critic: int = 64

    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.silu(nn.Dense(self.hidden_dim_critic)(nn.silu(nn.Dense(self.hidden_dim_critic)(x))))
        return nn.Dense(1)(x)[..., 0]


class FlaxFlatCritic(nn.Module):
    @nn.compact
    def __call__(self, state: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([state, action], axis=-1)
        q1 = nn.Dense(1)(nn.relu(nn.Dense(128)(nn.relu(nn.Dense(128)(x)))))
        q2 = nn.Dense(1)(nn.relu(nn.Dense(128)(nn.relu(nn.Dense(128)(x)))))
        return q1, q2


class TD3Learner1D:
    """
    Entrenador Twin Delayed Deep Deterministic Policy Gradient (TD3) para actores 1D RL-MALA.
    Soporta tanto `neural_deepset` (`FlaxTD3Actor`/`FlaxTD3Critic`) como `neural_flat`
    (`FlaxFlatActor`/`FlaxFlatCritic`), manteniendo compatibilidad exacta con los `.pkl` existentes.
    """

    def __init__(
        self,
        cfg: Experiment1DConfig,
        proposal: DeepSetNeuralProposal1D,
        key: jax.Array,
    ):
        self.cfg = cfg
        self.proposal = proposal
        self.n_elec = cfg.system.n_elec
        self.is_flat = isinstance(proposal, FlatMLPNeuralProposal1D)
        self.state_dim = proposal.state_dim()

        self.gamma_td3 = cfg.rl.rl_gamma_td3
        self.tau_polyak = cfg.rl.tau_polyak
        self.target_noise = cfg.rl.target_noise
        self.noise_clip = cfg.rl.noise_clip
        self.policy_delay = cfg.rl.policy_delay
        self.batch_size = cfg.rl.rl_batch_size
        self.unclip_sigma_action = bool(getattr(cfg.proposal, "unclip_sigma_action", False))
        self.scale_clip_min = float(cfg.proposal.scale_clip_min)
        self.scale_clip_max = float(cfg.proposal.scale_clip_max)

        self.actor_module = proposal.actor_module
        if self.is_flat:
            self.critic_module = FlaxFlatCritic()
        else:
            self.critic_module = FlaxTD3Critic(
                hidden_dim_critic=cfg.proposal.hidden_dim_critic
            )

        k1, k2, k3 = jax.random.split(key, 3)
        dummy_s = jnp.zeros((1, self.state_dim))
        dummy_a = jnp.zeros((1, 2))

        self.actor_params = self.actor_module.init(k1, dummy_s)
        self.target_actor_params = jax.tree_util.tree_map(
            lambda x: x.copy(), self.actor_params
        )

        self.critic1_params = self.critic_module.init(k2, dummy_s, dummy_a)
        self.target_critic1_params = jax.tree_util.tree_map(
            lambda x: x.copy(), self.critic1_params
        )

        self.critic2_params = self.critic_module.init(k3, dummy_s, dummy_a)
        self.target_critic2_params = jax.tree_util.tree_map(
            lambda x: x.copy(), self.critic2_params
        )

        if cfg.train.mode == "rl_only":
            total_rl_steps = cfg.train.train_steps
        elif cfg.train.mode == "online_coupled":
            active_vmc_steps = max(
                1,
                cfg.train.train_steps
                - cfg.train.online_nodal_warmup_steps
                - cfg.train.online_rl_freeze_tail_steps,
            )
            total_rl_steps = max(
                1,
                (active_vmc_steps // max(1, cfg.train.online_rl_update_every))
                * max(1, cfg.train.online_rl_updates_per_step),
            )
        else:
            total_rl_steps = (
                cfg.train.coupled_training_cycles * cfg.train.rl_sweeps_per_cycle
            )
        actor_lr = build_lr_schedule(
            base_lr=cfg.rl.lr_actor,
            total_steps=total_rl_steps,
            use_scheduler=cfg.rl.use_lr_scheduler,
            scheduler_type=cfg.rl.lr_scheduler_type,
            warmup_steps=cfg.rl.lr_warmup_steps,
            min_ratio=cfg.rl.lr_min_ratio,
        )
        critic_lr = build_lr_schedule(
            base_lr=cfg.rl.lr_critic,
            total_steps=total_rl_steps,
            use_scheduler=cfg.rl.use_lr_scheduler,
            scheduler_type=cfg.rl.lr_scheduler_type,
            warmup_steps=cfg.rl.lr_warmup_steps,
            min_ratio=cfg.rl.lr_min_ratio,
        )
        self.opt_a = optax.adam(actor_lr)
        self.opt_c = optax.adam(critic_lr)

        self.opt_act_state = self.opt_a.init(self.actor_params)
        self.opt_c1_state = self.opt_c.init(self.critic1_params)
        self.opt_c2_state = self.opt_c.init(self.critic2_params)

        self.buffer = ReplayBuffer(
            capacity=cfg.rl.rl_replay_capacity,
            state_dim=self.state_dim,
            action_dim=2,
        )
        self.update_counter = 0
        self.use_reward_centering = bool(getattr(cfg.rl, "use_reward_centering", True))
        self.reward_centering_lr = float(getattr(cfg.rl, "reward_centering_lr", 0.01))
        self.running_reward_mean = 0.0
        self.reward_mean_initialized = False

    def get_actor_params(self) -> dict:
        return self.actor_params

    def set_actor_params(self, params: dict) -> None:
        flax_vars = _ensure_flax_vars(params)
        self.actor_params = flax_vars
        self.target_actor_params = jax.tree_util.tree_map(
            lambda x: x.copy(), flax_vars
        )
        self.opt_act_state = self.opt_a.init(self.actor_params)

    def get_critic_params(self) -> tuple[dict, dict]:
        return self.critic1_params, self.critic2_params

    def set_critic_params(self, c1: dict, c2: dict) -> None:
        v1 = _ensure_flax_vars(c1)
        v2 = _ensure_flax_vars(c2)
        self.critic1_params = v1
        self.target_critic1_params = jax.tree_util.tree_map(lambda x: x.copy(), v1)
        self.opt_c1_state = self.opt_c.init(self.critic1_params)
        self.critic2_params = v2
        self.target_critic2_params = jax.tree_util.tree_map(lambda x: x.copy(), v2)
        self.opt_c2_state = self.opt_c.init(self.critic2_params)

    def reset_one_critic(self, key: jax.Array, which: int = 1) -> None:
        dummy_s = jnp.zeros((1, self.state_dim))
        dummy_a = jnp.zeros((1, 2))
        fresh = self.critic_module.init(key, dummy_s, dummy_a)
        if which == 1:
            self.critic1_params = fresh
            self.target_critic1_params = jax.tree_util.tree_map(
                lambda x: x.copy(), fresh
            )
            self.opt_c1_state = self.opt_c.init(self.critic1_params)
        else:
            self.critic2_params = fresh
            self.target_critic2_params = jax.tree_util.tree_map(
                lambda x: x.copy(), fresh
            )
            self.opt_c2_state = self.opt_c.init(self.critic2_params)

    def push_transitions(
        self,
        rl_info: dict,
        rewards: jax.Array,
        max_walkers: int | None = None,
    ) -> None:
        r_w = np.asarray(rewards).reshape(-1)
        n_walkers = len(r_w)
        s_raw = np.asarray(rl_info["state_feat"]).reshape(n_walkers, self.n_elec, self.state_dim)
        a_raw = np.asarray(rl_info["action"]).reshape(n_walkers, self.n_elec, 2)
        ns_raw = np.asarray(rl_info["next_state_feat"]).reshape(n_walkers, self.n_elec, self.state_dim)

        if max_walkers is not None and 0 < int(max_walkers) < n_walkers:
            sel = np.random.choice(n_walkers, size=int(max_walkers), replace=False)
            s_raw = s_raw[sel]
            a_raw = a_raw[sel]
            ns_raw = ns_raw[sel]
            r_w = r_w[sel]

        s = s_raw.reshape(-1, self.state_dim)
        a = a_raw.reshape(-1, 2)
        ns = ns_raw.reshape(-1, self.state_dim)
        r = np.repeat(r_w, self.n_elec)
        self.buffer.add_batch(s, a, r, ns)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _train_networks(
        self,
        key,
        p_act,
        p_c1,
        p_c2,
        t_act,
        t_c1,
        t_c2,
        opt_act,
        opt_c1,
        opt_c2,
        batch,
        update_actor,
        update_targets,
        reward_offset,
    ):
        noise = jnp.clip(
            self.target_noise * jax.random.normal(key, batch["action"].shape),
            -self.noise_clip,
            self.noise_clip,
        )
        raw_next_action = self.actor_module.apply(t_act, batch["next_state"]) + noise
        if self.unclip_sigma_action:
            next_drift = jnp.clip(raw_next_action[..., 0:1], -1.0, 1.0)
            next_scale = jnp.clip(
                raw_next_action[..., 1:2], self.scale_clip_min, self.scale_clip_max
            )
            next_action = jnp.concatenate([next_drift, next_scale], axis=-1)
        else:
            next_action = jnp.clip(raw_next_action, -1.0, 1.0)

        if self.is_flat:
            q1_next, _ = self.critic_module.apply(
                t_c1, batch["next_state"], next_action
            )
            _, q2_next = self.critic_module.apply(
                t_c2, batch["next_state"], next_action
            )
            q_target = jnp.minimum(q1_next[..., 0], q2_next[..., 0])
        else:
            q_target = jnp.minimum(
                self.critic_module.apply(t_c1, batch["next_state"], next_action),
                self.critic_module.apply(t_c2, batch["next_state"], next_action),
            )

        centered_reward = batch["reward"] - reward_offset
        y = jax.lax.stop_gradient(centered_reward + self.gamma_td3 * q_target)

        def critic_loss(p_crit, is_q1: bool):
            if self.is_flat:
                q1, q2 = self.critic_module.apply(
                    p_crit, batch["state"], batch["action"]
                )
                q = q1[..., 0] if is_q1 else q2[..., 0]
                return jnp.mean((q - y) ** 2)
            else:
                q = self.critic_module.apply(
                    p_crit, batch["state"], batch["action"]
                )
                return jnp.mean((q - y) ** 2)

        l1, g1 = jax.value_and_grad(critic_loss)(p_c1, True)
        l2, g2 = jax.value_and_grad(critic_loss)(p_c2, False)

        u1, opt_c1 = self.opt_c.update(g1, opt_c1, p_c1)
        p_c1 = optax.apply_updates(p_c1, u1)
        u2, opt_c2 = self.opt_c.update(g2, opt_c2, p_c2)
        p_c2 = optax.apply_updates(p_c2, u2)

        def actor_update():
            def actor_loss(p_a):
                act = self.actor_module.apply(p_a, batch["state"])
                if self.is_flat:
                    q1, _ = self.critic_module.apply(p_c1, batch["state"], act)
                    return -jnp.mean(q1)
                else:
                    return -jnp.mean(
                        self.critic_module.apply(p_c1, batch["state"], act)
                    )

            l_act, g_act = jax.value_and_grad(actor_loss)(p_act)
            u_act, new_opt_act = self.opt_a.update(g_act, opt_act, p_act)
            new_p_act = optax.apply_updates(p_act, u_act)
            return (
                new_p_act,
                optax.incremental_update(new_p_act, t_act, self.tau_polyak),
                new_opt_act,
                l_act.astype(jnp.float32),
            )

        p_act, t_act, opt_act, l_act = jax.lax.cond(
            update_actor,
            actor_update,
            lambda: (
                p_act,
                t_act,
                opt_act,
                jnp.array(0.0, dtype=jnp.float32),
            ),
        )

        t_c1, t_c2 = jax.lax.cond(
            update_targets,
            lambda: (
                optax.incremental_update(p_c1, t_c1, self.tau_polyak),
                optax.incremental_update(p_c2, t_c2, self.tau_polyak),
            ),
            lambda: (t_c1, t_c2),
        )

        return (
            p_act,
            p_c1,
            p_c2,
            t_act,
            t_c1,
            t_c2,
            opt_act,
            opt_c1,
            opt_c2,
            0.5 * (l1 + l2),
            l_act,
            jnp.mean(y),
        )

    def update_step(
        self,
        key: jax.Array,
        allow_actor_update: bool = True,
    ) -> dict[str, float]:
        if self.buffer.size < self.batch_size:
            return {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "q_val": 0.0,
                "reward_mean_ema": float(self.running_reward_mean),
            }

        self.update_counter += 1
        batch = self.buffer.sample(self.batch_size)

        if self.use_reward_centering:
            batch_r_mean = float(jnp.mean(batch["reward"]))
            if not self.reward_mean_initialized:
                self.running_reward_mean = batch_r_mean
                self.reward_mean_initialized = True
            else:
                self.running_reward_mean = (
                    (1.0 - self.reward_centering_lr) * self.running_reward_mean
                    + self.reward_centering_lr * batch_r_mean
                )
            reward_offset = jnp.array(self.running_reward_mean, dtype=jnp.float32)
        else:
            reward_offset = jnp.array(0.0, dtype=jnp.float32)

        is_delay_step = (self.update_counter % self.policy_delay == 0)
        update_actor = jnp.array(bool(allow_actor_update and is_delay_step))
        update_targets = jnp.array(bool(is_delay_step))

        (
            self.actor_params,
            self.critic1_params,
            self.critic2_params,
            self.target_actor_params,
            self.target_critic1_params,
            self.target_critic2_params,
            self.opt_act_state,
            self.opt_c1_state,
            self.opt_c2_state,
            c_loss,
            a_loss,
            q_val,
        ) = self._train_networks(
            key,
            self.actor_params,
            self.critic1_params,
            self.critic2_params,
            self.target_actor_params,
            self.target_critic1_params,
            self.target_critic2_params,
            self.opt_act_state,
            self.opt_c1_state,
            self.opt_c2_state,
            batch,
            update_actor,
            update_targets,
            reward_offset,
        )
        return {
            "critic_loss": float(c_loss),
            "actor_loss": float(a_loss),
            "q_val": float(q_val),
            "reward_mean_ema": float(self.running_reward_mean),
        }


class REINFORCELearner1D:
    """
    Entrenador de gradiente de política on-policy diferenciado (vía truco de reparametrización)
    para el Actor 1D RL-MALA.
    """

    def __init__(
        self,
        cfg: Experiment1DConfig,
        proposal: DeepSetNeuralProposal1D,
        ansatz: BaseAnsatz1D,
        sampler: MHSampler1D,
        key: jax.Array,
    ):
        self.cfg = cfg
        self.proposal = proposal
        self.ansatz = ansatz
        self.sampler = sampler
        self.n_elec = cfg.system.n_elec

        dummy_w = jnp.zeros((1, self.n_elec, 1))
        dummy_g = jnp.zeros((1, self.n_elec, 1))
        self.actor_params = proposal.init_params(key, dummy_w, dummy_g)
        total_rl_steps = (
            cfg.train.train_steps
            if cfg.train.mode == "rl_only"
            else (cfg.train.coupled_training_cycles * cfg.train.rl_sweeps_per_cycle)
        )
        actor_lr = build_lr_schedule(
            base_lr=cfg.rl.lr_actor,
            total_steps=total_rl_steps,
            use_scheduler=cfg.rl.use_lr_scheduler,
            scheduler_type=cfg.rl.lr_scheduler_type,
            warmup_steps=cfg.rl.lr_warmup_steps,
            min_ratio=cfg.rl.lr_min_ratio,
        )
        self.actor_opt = optax.adam(actor_lr)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)

    def get_actor_params(self) -> dict:
        return self.actor_params

    def set_actor_params(self, params: dict) -> None:
        flax_vars = _ensure_flax_vars(params)
        self.actor_params = flax_vars
        self.actor_opt_state = self.actor_opt.init(self.actor_params)

    def get_critic_params(self) -> tuple[dict, dict]:
        return {}, {}

    def set_critic_params(self, c1: dict, c2: dict) -> None:
        pass

    @functools.partial(jax.jit, static_argnums=(0,))
    def _grad_step(self, key, actor_params, opt_state, state, params_psi):
        noise = jax.random.normal(key, shape=state.walkers.shape)

        def loss_fn(ap):
            mu_fwd, sigma_fwd, aux_fwd = self.proposal.distribution(
                ap,
                state.walkers,
                state.grad_log_psi,
                use_actor=True,
                key_explore=None,
                exploration_std=0.0,
                is_reverse=False,
            )
            w_prop = mu_fwd + sigma_fwd * noise
            log_p_prop, grad_prop = self.ansatz.value_and_grad(
                params_psi,
                w_prop,
                min_clip=self.cfg.sampler.min_clip_grads_ansatz,
                max_clip=self.cfg.sampler.max_clip_grads_ansatz,
            )
            mu_rev, sigma_rev, _ = self.proposal.distribution(
                ap,
                w_prop,
                grad_prop,
                use_actor=True,
                key_explore=None,
                exploration_std=0.0,
                is_reverse=True,
            )
            log_q_fwd = gaussian_log_prob(w_prop, mu_fwd, sigma_fwd, axis=(-2, -1))
            log_q_rev = gaussian_log_prob(
                state.walkers, mu_rev, sigma_rev, axis=(-2, -1)
            )
            log_p_diff = 2.0 * (log_p_prop - state.log_psi)
            raw_lr = log_p_diff + (log_q_rev - log_q_fwd)
            log_alpha = jnp.minimum(
                0.0,
                jnp.clip(
                    jnp.nan_to_num(raw_lr, nan=-50.0),
                    self.cfg.sampler.min_clip_log_ratio,
                    self.cfg.sampler.max_clip_log_ratio,
                ),
            )
            alpha = jnp.exp(log_alpha)
            proposed_disp = jnp.mean(jnp.abs(w_prop - state.walkers), axis=(-2, -1))
            disp_sq = jnp.sum((w_prop - state.walkers) ** 2, axis=(-2, -1))

            rl_info = {
                "alpha": alpha,
                "log_p_diff": log_p_diff,
                "log_q_fwd": log_q_fwd,
                "proposed_disp": proposed_disp,
                "disp_sq": disp_sq,
                "action": aux_fwd["action"],
            }
            rewards, _ = compute_rl_reward(rl_info, self.cfg)
            return -jnp.mean(rewards), jnp.mean(rewards)

        (loss, mean_reward), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            actor_params
        )
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(jnp.nan_to_num(g, nan=0.0), -10.0, 10.0), grads
        )
        updates, new_opt_state = self.actor_opt.update(
            grads, opt_state, actor_params
        )
        new_actor_params = optax.apply_updates(actor_params, updates)
        return new_actor_params, new_opt_state, loss, mean_reward

    def update_on_policy(
        self, key: jax.Array, state, params_psi
    ) -> tuple[object, dict[str, float]]:
        k_grad, k_step = jax.random.split(key)
        (
            self.actor_params,
            self.actor_opt_state,
            loss,
            mean_reward,
        ) = self._grad_step(
            k_grad, self.actor_params, self.actor_opt_state, state, params_psi
        )
        next_state, stats, _ = self.sampler.step(
            k_step,
            state,
            params_psi,
            self.actor_params,
            use_actor=True,
            exploration_std=0.0,
        )
        return next_state, {
            "actor_loss": float(loss),
            "critic_loss": 0.0,
            "q_val": float(mean_reward),
            "reward": float(mean_reward),
            "acceptance": float(jnp.mean(stats.acceptance)),
            "esjd": float(jnp.mean(stats.esjd)),
            "actor_drift_mag": float(jnp.mean(stats.actor_drift_mag)),
            "actor_sigma": float(jnp.mean(stats.actor_sigma)),
        }
