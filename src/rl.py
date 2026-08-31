from __future__ import annotations
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
import math
import numpy as np
import jax
import jax.numpy as jnp

from physics import SystemConfig
from optimizer import adam_init, adam_update
from actors import ActorModel, ActorArchitecture, init_mlp, mlp_apply

Array = jax.Array
RLMode = Literal["scale_only", "residual_mala", "full_gaussian"]


@dataclass(frozen=True)
class RLConfig:
    """Configuration for the shared single-electron TD3 proposal actor.

    The actor acts on one electron at a time and is shared across electron
    indices.  Its deterministic output parametrizes a Gaussian MH proposal.
    For ``residual_mala`` the proposal mean is

        x_i + 1/2 sigma_i^2 d_i log |Psi|^2 + Delta_mu_i,

    while ``sigma_i`` and ``Delta_mu_i`` are state-dependent actor outputs.
    The exact forward/reverse Gaussian proposal densities are always retained
    in the Metropolis-Hastings acceptance ratio.
    """

    mode: RLMode = "residual_mala"
    base_step_size: float = 0.20

    # Permutation-aware local state features.
    pair_feature_components: int = 6
    pair_feature_r_max: float = 5.0
    pair_feature_width: float = 0.8
    coordinate_scale: float = 3.0
    force_scale: float = 5.0

    # Actor architecture.  ``flat_mlp`` reproduces the original policy.
    # ``multibranch_deepset`` separates local/global/pair information and
    # applies shared same/opposite-spin pair encoders before pooling.
    actor_architecture: ActorArchitecture = "flat_mlp"
    trainable_embedding: bool = False
    local_embedding_dim: int = 32
    global_embedding_dim: int = 16
    pair_embedding_dim: int = 32
    fusion_dim: int = 64
    branch_hidden_layers: int = 2
    fusion_hidden_layers: int = 1
    head_hidden_dim: int = 32
    head_hidden_layers: int = 1

    # Actor output transforms.
    log_scale_range: float = 1.0
    residual_scale: float = 1.0
    min_sigma: float = 1e-4
    mala_drift_clip: float = 10.0

    # Actor / critic networks.
    hidden_dim: int = 64
    hidden_layers: int = 2

    # TD3.
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_policy_noise: float = 0.15
    target_noise_clip: float = 0.35
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_starts: int = 1024
    updates_per_electron: int = 1

    # Training chain / exploration.  Exploration noise is an auxiliary
    # variable defining a reversible mixture of proposal kernels: the SAME
    # action noise is used when evaluating q(x'|x) and q(x|x').
    training_sweeps: int = 100
    burn_in_sweeps: int = 30
    jit_sweep_chunk: int = 32
    exploration_noise: float = 0.20
    exploration_noise_final: float = 0.03

    # Reward r = beta * log(|Delta x|^2) + gamma_accept * log(alpha).
    # ``reward_acceptance_weight`` is deliberately not named ``gamma`` here,
    # because ``gamma`` above is TD3's temporal discount factor.
    reward_displacement_weight: float = 10.0
    reward_acceptance_weight: float = 0.1
    reward_epsilon: float = 1e-8
    reward_clip: float = 30.0

    seed: int = 1701

    def __post_init__(self):
        if self.base_step_size <= 0:
            raise ValueError("base_step_size must be positive")
        if self.pair_feature_components < 1:
            raise ValueError("pair_feature_components must be >= 1")
        if self.pair_feature_width <= 0:
            raise ValueError("pair_feature_width must be positive")
        if self.coordinate_scale <= 0 or self.force_scale <= 0:
            raise ValueError("feature scales must be positive")
        if self.actor_architecture not in ActorModel.AVAILABLE:
            raise ValueError(f"actor_architecture must be one of {ActorModel.AVAILABLE}")
        if self.trainable_embedding and self.actor_architecture != ActorModel.MULTIBRANCH:
            raise ValueError("trainable_embedding is only supported by multibranch_deepset")
        if min(
            self.local_embedding_dim, self.global_embedding_dim, self.pair_embedding_dim,
            self.fusion_dim, self.branch_hidden_layers, self.fusion_hidden_layers,
            self.head_hidden_dim, self.head_hidden_layers,
        ) < 1:
            raise ValueError("all multibranch actor dimensions/layer counts must be >= 1")
        if self.hidden_dim < 1 or self.hidden_layers < 1:
            raise ValueError("hidden_dim and hidden_layers must be >= 1")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size >= 1")
        if self.training_sweeps < 1:
            raise ValueError("training_sweeps must be >= 1")
        if self.reward_displacement_weight < 0 or self.reward_acceptance_weight < 0:
            raise ValueError("reward beta and reward gamma must be non-negative")


@dataclass
class RLHistory:
    sweeps: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    acceptances: list[float] = field(default_factory=list)
    mean_alpha: list[float] = field(default_factory=list)
    proposed_displacements: list[float] = field(default_factory=list)
    accepted_displacements: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=list)
    residuals: list[float] = field(default_factory=list)
    critic1_losses: list[float] = field(default_factory=list)
    critic2_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    replay_sizes: list[int] = field(default_factory=list)
    exploration_noises: list[float] = field(default_factory=list)


def action_dim(cfg: RLConfig) -> int:
    return 1 if cfg.mode == "scale_only" else 2


def state_dim(cfg: RLConfig, system_cfg: SystemConfig) -> int:
    return ActorModel(cfg, system_cfg).state_dim


def local_state_features_from_grad(
    walkers: Array,
    grad_log_prob: Array,
    electron_index: int,
    system_cfg: SystemConfig,
    cfg: RLConfig,
) -> Array:
    """Build the actor state using the selected actor architecture."""
    return ActorModel(cfg, system_cfg).build_state(
        walkers, grad_log_prob, electron_index
    )


def actor_apply(
    params, state: Array, cfg: RLConfig, system_cfg: SystemConfig
) -> Array:
    return ActorModel(cfg, system_cfg).apply(params, state)


def critic_apply(params, state: Array, action: Array) -> Array:
    # Critics intentionally remain generic flat MLPs.  For the multibranch actor
    # the replay state is distance-sorted within each spin channel, making this
    # flattened representation permutation-invariant to electron relabeling.
    x = jnp.concatenate([state, action], axis=-1)
    return jnp.squeeze(mlp_apply(params, x, final_tanh=False), axis=-1)


def init_actor(key: Array, cfg: RLConfig, system_cfg: SystemConfig):
    # Both actor architectures initialize their final action heads to zero.
    # Therefore scale_only/residual_mala recover ordinary base MALA exactly.
    return ActorModel(cfg, system_cfg).init(key)


def init_critic(key: Array, cfg: RLConfig, system_cfg: SystemConfig):
    return init_mlp(
        key, state_dim(cfg, system_cfg) + action_dim(cfg), cfg.hidden_dim,
        cfg.hidden_layers, 1, zero_last=False
    )


def proposal_parameters_from_action(
    action: Array,
    coordinate: Array,
    quantum_force: Array,
    cfg: RLConfig,
) -> tuple[Array, Array, Array]:
    """Return proposal mean, sigma, residual/learned mean offset."""
    a = jnp.asarray(action)
    if cfg.mode == "scale_only":
        scale_action = a[..., 0]
        mean_action = jnp.zeros_like(scale_action)
    else:
        mean_action = a[..., 0]
        scale_action = a[..., 1]

    sigma = cfg.base_step_size * jnp.exp(cfg.log_scale_range * scale_action)
    sigma = jnp.maximum(sigma, cfg.min_sigma)
    learned_offset = cfg.residual_scale * cfg.base_step_size * mean_action

    if cfg.mode in ("scale_only", "residual_mala"):
        drift = 0.5 * sigma**2 * quantum_force
        if cfg.mala_drift_clip > 0:
            drift = jnp.clip(drift, -cfg.mala_drift_clip, cfg.mala_drift_clip)
        residual = jnp.zeros_like(learned_offset) if cfg.mode == "scale_only" else learned_offset
        mean = coordinate + drift + residual
        return mean, sigma, residual

    if cfg.mode == "full_gaussian":
        # No explicit quantum-force drift in this ablation.
        mean = coordinate + learned_offset
        return mean, sigma, learned_offset

    raise ValueError(f"Unknown RL mode: {cfg.mode}")


def _tree_polyak(target, source, tau: float):
    return jax.tree_util.tree_map(lambda t, s: (1.0 - tau) * t + tau * s, target, source)


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim_: int, action_dim_: int, dtype=None):
        self.capacity = int(capacity)
        if dtype is None:
            dtype = np.float64 if jax.config.x64_enabled else np.float32
        self.dtype = np.dtype(dtype)
        self.state = np.zeros((capacity, state_dim_), dtype=self.dtype)
        self.action = np.zeros((capacity, action_dim_), dtype=self.dtype)
        self.reward = np.zeros((capacity,), dtype=self.dtype)
        self.next_state = np.zeros((capacity, state_dim_), dtype=self.dtype)
        self.done = np.zeros((capacity,), dtype=self.dtype)
        self.size = 0
        self.pos = 0

    def add_batch(self, state, action, reward, next_state, done=None):
        s = np.asarray(state, dtype=self.dtype)
        a = np.asarray(action, dtype=self.dtype)
        r = np.asarray(reward, dtype=self.dtype).reshape(-1)
        ns = np.asarray(next_state, dtype=self.dtype)
        d = np.zeros_like(r) if done is None else np.asarray(done, dtype=self.dtype).reshape(-1)
        n = len(r)
        if n >= self.capacity:
            s, a, r, ns, d = s[-self.capacity:], a[-self.capacity:], r[-self.capacity:], ns[-self.capacity:], d[-self.capacity:]
            n = self.capacity
        idx = (np.arange(n) + self.pos) % self.capacity
        self.state[idx] = s
        self.action[idx] = a
        self.reward[idx] = r
        self.next_state[idx] = ns
        self.done[idx] = d
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, rng: np.random.Generator, batch_size: int) -> dict[str, Array]:
        if self.size < batch_size:
            raise ValueError("Not enough replay samples")
        idx = rng.integers(0, self.size, size=int(batch_size))
        return {
            "state": jnp.asarray(self.state[idx]),
            "action": jnp.asarray(self.action[idx]),
            "reward": jnp.asarray(self.reward[idx]),
            "next_state": jnp.asarray(self.next_state[idx]),
            "done": jnp.asarray(self.done[idx]),
        }



@partial(jax.jit, static_argnames=("cfg", "system_cfg"))
def _td3_critic_update(
    actor_target, critic1, critic2, critic1_target, critic2_target,
    critic1_opt, critic2_opt, batch, key, cfg: RLConfig, system_cfg: SystemConfig,
):
    state = batch["state"]
    action = batch["action"]
    reward = batch["reward"]
    next_state = batch["next_state"]
    done = batch["done"]

    noise = cfg.target_policy_noise * jax.random.normal(
        key, (state.shape[0], action_dim(cfg)), dtype=state.dtype
    )
    noise = jnp.clip(noise, -cfg.target_noise_clip, cfg.target_noise_clip)
    target_action = jnp.clip(
        actor_apply(actor_target, next_state, cfg, system_cfg) + noise, -1.0, 1.0
    )
    target_q1 = critic_apply(critic1_target, next_state, target_action)
    target_q2 = critic_apply(critic2_target, next_state, target_action)
    target = jax.lax.stop_gradient(
        reward + cfg.gamma * (1.0 - done) * jnp.minimum(target_q1, target_q2)
    )

    def critic_loss(p, s, a, y):
        q = critic_apply(p, s, a)
        return jnp.mean((q - y) ** 2)

    loss1, grad1 = jax.value_and_grad(critic_loss)(critic1, state, action, target)
    loss2, grad2 = jax.value_and_grad(critic_loss)(critic2, state, action, target)
    critic1, critic1_opt = adam_update(
        critic1, grad1, critic1_opt, learning_rate=cfg.critic_learning_rate
    )
    critic2, critic2_opt = adam_update(
        critic2, grad2, critic2_opt, learning_rate=cfg.critic_learning_rate
    )
    return critic1, critic2, critic1_opt, critic2_opt, loss1, loss2


@partial(jax.jit, static_argnames=("cfg", "system_cfg"))
def _td3_actor_update(
    actor, critic1, critic2, actor_target, critic1_target, critic2_target,
    actor_opt, state, cfg: RLConfig, system_cfg: SystemConfig,
):
    def actor_loss(p, s):
        a = actor_apply(p, s, cfg, system_cfg)
        return -jnp.mean(critic_apply(critic1, s, a))

    loss, grad = jax.value_and_grad(actor_loss)(actor, state)
    actor, actor_opt = adam_update(
        actor, grad, actor_opt, learning_rate=cfg.actor_learning_rate
    )
    actor_target = _tree_polyak(actor_target, actor, cfg.tau)
    critic1_target = _tree_polyak(critic1_target, critic1, cfg.tau)
    critic2_target = _tree_polyak(critic2_target, critic2, cfg.tau)
    return (
        actor, actor_target, critic1_target, critic2_target, actor_opt, loss
    )


class TD3Agent:
    def __init__(self, key: Array, cfg: RLConfig, system_cfg: SystemConfig):
        self.cfg = cfg
        self.system_cfg = system_cfg
        ka, k1, k2 = jax.random.split(key, 3)
        self.actor = init_actor(ka, cfg, system_cfg)
        self.critic1 = init_critic(k1, cfg, system_cfg)
        self.critic2 = init_critic(k2, cfg, system_cfg)
        self.actor_target = jax.tree_util.tree_map(lambda x: x.copy(), self.actor)
        self.critic1_target = jax.tree_util.tree_map(lambda x: x.copy(), self.critic1)
        self.critic2_target = jax.tree_util.tree_map(lambda x: x.copy(), self.critic2)
        self.actor_opt = adam_init(self.actor)
        self.critic1_opt = adam_init(self.critic1)
        self.critic2_opt = adam_init(self.critic2)
        self.update_count = 0

    def act(self, state: Array) -> Array:
        return actor_apply(self.actor, state, self.cfg, self.system_cfg)

    def update(self, batch: dict[str, Array], key: Array) -> dict[str, float]:
        """One TD3 update; dense-network work is fused by JAX JIT on GPU/CPU."""
        cfg = self.cfg
        (
            self.critic1, self.critic2, self.critic1_opt, self.critic2_opt,
            loss1, loss2,
        ) = _td3_critic_update(
            self.actor_target, self.critic1, self.critic2,
            self.critic1_target, self.critic2_target,
            self.critic1_opt, self.critic2_opt, batch, key, cfg, self.system_cfg,
        )

        self.update_count += 1
        actor_loss_value = float("nan")
        if self.update_count % cfg.policy_delay == 0:
            (
                self.actor, self.actor_target, self.critic1_target,
                self.critic2_target, self.actor_opt, aloss,
            ) = _td3_actor_update(
                self.actor, self.critic1, self.critic2, self.actor_target,
                self.critic1_target, self.critic2_target, self.actor_opt,
                batch["state"], cfg, self.system_cfg,
            )
            actor_loss_value = float(aloss)

        return {
            "critic1_loss": float(loss1),
            "critic2_loss": float(loss2),
            "actor_loss": actor_loss_value,
        }



def copy_tree_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), tree)
