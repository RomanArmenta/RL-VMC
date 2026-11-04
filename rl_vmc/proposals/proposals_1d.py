from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

from rl_vmc.config import Experiment1DConfig


def apply_spatial_gating(x: jax.Array, actor_output: jax.Array, l_radius: float = 10.0) -> jax.Array:
    """
    Aplica una máscara suave para apagar el Actor cerca de los bordes de la trampa armónica.
    x: posiciones de los electrones. Shape (batch, n_elec) o (batch, n_elec, 1)
    actor_output: [mean, scale] del Actor. Shape (batch, 2)
    """
    x_1d = x[..., 0] if x.ndim == 3 else x
    com = jnp.mean(x_1d, axis=-1, keepdims=True)
    dist_sq = jnp.mean((x_1d - com) ** 2, axis=-1)
    eta = dist_sq / (l_radius ** 2)

    safe_eta = jnp.clip(eta, 0.50001, 0.99999)
    transition = 1.0 / (1.0 + jnp.exp(-(4 * safe_eta - 3) / (4 * safe_eta ** 2 - 6 * safe_eta + 2)))

    gamma = jnp.where(eta <= 0.5, 0.0, jnp.where(eta >= 1.0, 1.0, transition))
    gamma = gamma[..., jnp.newaxis]

    drift_gated = actor_output[..., 0:1] * (1.0 - gamma)
    scale = actor_output[..., 1:2]
    return jnp.concatenate([drift_gated, scale], axis=-1)


def build_td3_state(
    x: jax.Array,
    g: jax.Array,
    elec_idx: int | jax.Array,
    n_elec: int,
    n_up: int,
    n_dn: int,
    quantum_force_norm: float = 5.0,
    particle_factor: float = 10.0,
    dist_factor: float = 5.0,
    omega: float = 0.1,
    gs: float = 0.5,
    softening: float = 0.1,
) -> jax.Array:
    if x.ndim == 3:
        x = x[..., 0]
    if g.ndim == 3:
        g = g[..., 0]

    batch_size = x.shape[0]
    i = elec_idx
    x_i, g_i = x[:, i], g[:, i]

    spins = jnp.concatenate([jnp.ones(n_up), -jnp.ones(n_dn)])
    spin_i = spins[i]

    local_feat = jnp.stack(
        [
            x_i,
            x_i - jnp.mean(x, axis=1),
            jnp.tanh(g_i / quantum_force_norm),
            jnp.full_like(x_i, spin_i),
        ],
        axis=-1,
    )

    global_feat = jnp.asarray(
        [
            n_elec / particle_factor,
            n_up / max(n_elec, 1),
            jnp.tanh(omega),
            jnp.tanh(gs),
            jnp.tanh(softening),
        ]
    )
    global_feat = jnp.broadcast_to(global_feat[None, :], (batch_size, 5))

    dx = x[:, i, None] - x
    r = jnp.abs(dx)

    not_self = jnp.arange(n_elec) != i
    same_mask = (spins == spin_i) & not_self
    opp_mask = (spins != spin_i) & not_self

    def get_slots(r_val, dx_val, mask_val, max_slots):
        if max_slots == 0:
            return jnp.zeros((batch_size, 0, 3))
        mask_w = jnp.broadcast_to(mask_val[None, :], (batch_size, n_elec))
        keys = jnp.where(mask_w, r_val, 1e12)
        order = jnp.argsort(keys, axis=1)[:, :max_slots]
        rr = jnp.take_along_axis(r_val, order, axis=1)
        dd = jnp.take_along_axis(dx_val, order, axis=1)
        mm = jnp.take_along_axis(mask_w, order, axis=1)
        return jnp.stack(
            [
                jnp.where(mm, rr / dist_factor, 0.0),
                jnp.where(mm, jnp.sign(dd), 0.0),
                mm.astype(jnp.float32),
            ],
            axis=-1,
        )

    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0

    same_slots = get_slots(r, dx, same_mask, max_same).reshape((batch_size, -1))
    opp_slots = get_slots(r, dx, opp_mask, max_opp).reshape((batch_size, -1))

    return jnp.concatenate([local_feat, global_feat, same_slots, opp_slots], axis=-1)


class FlaxTD3Actor(nn.Module):
    n_up: int
    n_dn: int
    actor_amplitude: float = 0.05
    sigma_action_scale: float = 0.2
    hidden_dim_local: int = 32
    hidden_dim_global: int = 16
    hidden_dim_messages: int = 32
    hidden_dim_actor: int = 64

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        max_same = max(max(self.n_up - 1, 0), max(self.n_dn - 1, 0))
        max_opp = max(self.n_up, self.n_dn) if (self.n_up > 0 and self.n_dn > 0) else 0
        p = 0
        local_feat = state[..., p : p + 4]
        p += 4
        global_feat = state[..., p : p + 5]
        p += 5
        same_slots = state[..., p : p + 3 * max_same].reshape(state.shape[:-1] + (max_same, 3))
        p += 3 * max_same
        opp_slots = state[..., p : p + 3 * max_opp].reshape(state.shape[:-1] + (max_opp, 3))

        h_local = nn.silu(nn.Dense(self.hidden_dim_local)(nn.silu(nn.Dense(self.hidden_dim_local)(local_feat))))
        h_global = nn.silu(nn.Dense(self.hidden_dim_global)(nn.silu(nn.Dense(self.hidden_dim_global)(global_feat))))

        msg_same = nn.silu(nn.Dense(self.hidden_dim_messages)(nn.silu(nn.Dense(self.hidden_dim_messages)(same_slots))))
        mask_same = same_slots[..., 2:3]
        pool_same = jnp.sum(msg_same * mask_same, axis=-2) / jnp.maximum(jnp.sum(mask_same, axis=-2), 1.0)

        msg_opp = nn.silu(nn.Dense(self.hidden_dim_messages)(nn.silu(nn.Dense(self.hidden_dim_messages)(opp_slots))))
        mask_opp = opp_slots[..., 2:3]
        pool_opp = jnp.sum(msg_opp * mask_opp, axis=-2) / jnp.maximum(jnp.sum(mask_opp, axis=-2), 1.0)

        fused = jnp.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1)
        h = nn.silu(nn.Dense(self.hidden_dim_actor)(fused))

        mean = nn.tanh(nn.Dense(1, kernel_init=nn.initializers.zeros)(h))
        scale = self.sigma_action_scale * nn.tanh(nn.Dense(1, kernel_init=nn.initializers.zeros)(h))
        return jnp.concatenate([mean, scale], axis=-1)


class FlaxFlatActor(nn.Module):
    actor_amplitude: float = 0.05

    @nn.compact
    def __call__(self, state: jax.Array) -> jax.Array:
        x = nn.relu(nn.Dense(128)(state))
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(128)(x))
        drift = jnp.tanh(nn.Dense(1)(x))
        scale = nn.Dense(1)(x)
        return jnp.concatenate([drift, scale], axis=-1)


class FlaxHuzielLegacyActor(nn.Module):
    """Compatibilidad con pesos `.pkl` de `Actor_Huziel` (REINFORCE en `run_experiment_1D.py`)."""
    n_elec: int
    n_up: int
    n_dn: int
    hidden_dim: int = 64

    def setup(self):
        self.local_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        self.global_enc = nn.Sequential([nn.Dense(16), nn.silu, nn.Dense(16), nn.silu])
        self.same_spin_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        self.opp_spin_enc = nn.Sequential([nn.Dense(32), nn.silu, nn.Dense(32), nn.silu])
        self.fusion = nn.Sequential([nn.Dense(64), nn.silu, nn.Dense(64), nn.silu])
        self.out_drift = nn.Dense(1, kernel_init=nn.initializers.zeros)
        self.out_sigma = nn.Dense(1, kernel_init=nn.initializers.zeros)

    def __call__(self, walkers: jax.Array, elec_idx: int | jax.Array, grad_log_prob: jax.Array) -> jax.Array:
        x = walkers[..., 0] if walkers.ndim == 3 else walkers
        g = grad_log_prob[..., 0] if grad_log_prob.ndim == 3 else grad_log_prob
        N = self.n_elec
        x_i = x[:, elec_idx]
        g_i = g[:, elec_idx]
        spins = jnp.concatenate([jnp.ones(self.n_up), -jnp.ones(self.n_dn)])
        spin_i = spins[elec_idx]
        spin_batch = jnp.full_like(x_i, spin_i)
        com = jnp.mean(x, axis=1)
        local_feat = jnp.stack([x_i, x_i - com, nn.tanh(g_i / 5.0), spin_batch], axis=-1)
        global_feat = jnp.broadcast_to(
            jnp.stack([N / 10.0, self.n_up / max(N, 1), jnp.tanh(1.0), jnp.tanh(1.0)]),
            (x.shape[0], 4),
        )
        dx = x_i[:, None] - x
        r = jnp.abs(dx)
        direction = jnp.sign(dx)
        centers = jnp.linspace(0.0, 5.0, 6)
        basis = jnp.exp(-0.5 * ((r[..., None] - centers) / 0.8) ** 2)
        pair_input = jnp.concatenate([r[..., None], direction[..., None], basis], axis=-1)
        not_self = jnp.arange(N) != elec_idx
        same_mask = jnp.broadcast_to(((spins == spin_i) & not_self)[None, :], (x.shape[0], N))
        opp_mask = jnp.broadcast_to(((spins != spin_i) & not_self)[None, :], (x.shape[0], N))
        msg_same = self.same_spin_enc(pair_input) * same_mask[..., None]
        pool_same = jnp.sum(msg_same, axis=1) / jnp.maximum(jnp.sum(same_mask, axis=1, keepdims=True), 1.0)
        msg_opp = self.opp_spin_enc(pair_input) * opp_mask[..., None]
        pool_opp = jnp.sum(msg_opp, axis=1) / jnp.maximum(jnp.sum(opp_mask, axis=1, keepdims=True), 1.0)
        h_local = self.local_enc(local_feat)
        h_global = self.global_enc(global_feat)
        h = self.fusion(jnp.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1))
        mean = nn.tanh(self.out_drift(h))
        scale = nn.tanh(self.out_sigma(h))
        return jnp.concatenate([mean, scale], axis=-1)


def gaussian_log_prob(
    x_to: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    axis: tuple[int, ...] = (-2, -1),
) -> jax.Array:
    """
    Calcula la log-densidad de transición Gaussiana diagonal log q(x_to | x_from)
    omitiendo la constante aditiva -0.5*log(2*pi) (que se cancela exactamente en el
    cociente de Metropolis-Hastings log_q_rev - log_q_fwd) para preservar la calibración
    exacta del término de exploración CDLB (-alpha * log_q_fwd) de `src/td3_modules.py`.
    """
    sigma_safe = jnp.maximum(sigma, 1e-6)
    log_p = -jnp.log(sigma_safe) - 0.5 * ((x_to - mu) / sigma_safe) ** 2
    return jnp.sum(log_p, axis=axis)


def _ensure_flax_vars(params: dict) -> dict:
    if not params:
        return {"params": {}}
    return params if "params" in params else {"params": params}


class BaseProposal1D:
    """Interfaz base para políticas de propuesta MCMC 1D (clásicas y neuronales)."""

    is_neural: bool = False
    is_hmc: bool = False

    def __init__(self, cfg: Experiment1DConfig):
        self.cfg = cfg
        self.tau = cfg.proposal.tau
        self.rw_sigma = cfg.proposal.rw_sigma
        self.max_drift_clip = cfg.proposal.max_drift_clip
        self.drift_prefactor = 1.0 if cfg.proposal.legacy_drift_factor else 2.0
        self.n_elec = cfg.system.n_elec
        self.n_up = cfg.system.n_up
        self.n_dn = cfg.system.n_dn

    def state_dim(self) -> int:
        max_same = max(max(self.n_up - 1, 0), max(self.n_dn - 1, 0))
        max_opp = max(self.n_up, self.n_dn) if (self.n_up > 0 and self.n_dn > 0) else 0
        return 4 + 5 + 3 * max_same + 3 * max_opp

    def init_params(
        self,
        key: jax.Array,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
    ) -> dict:
        return {}

    def build_state_features(
        self,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
    ) -> jax.Array:
        """Construye las características por electrón con shape (batch, n_elec, state_dim)."""
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = grad_log_psi[..., 0] if grad_log_psi.ndim == 3 else grad_log_psi

        def per_elec(e_idx):
            return build_td3_state(
                w_1d,
                g_1d,
                e_idx,
                self.n_elec,
                self.n_up,
                self.n_dn,
                quantum_force_norm=self.cfg.proposal.force_scale,
                particle_factor=self.cfg.proposal.n_scale,
                dist_factor=self.cfg.proposal.r_scale,
                omega=self.cfg.system.omega,
                gs=self.cfg.system.interaction_strength,
                softening=self.cfg.system.softening,
            )

        feat_e_b_d = jax.vmap(per_elec)(jnp.arange(self.n_elec))
        return jnp.transpose(feat_e_b_d, (1, 0, 2))

    def distribution(
        self,
        params_actor: dict,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
        use_actor: bool = True,
        key_explore: Optional[jax.Array] = None,
        exploration_std: float = 0.0,
        is_reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict]:
        """
        Retorna (mu, sigma, aux_dict) para la transición Gaussiana q(X' | X) = N(mu, sigma^2).
        Tanto mu como sigma tienen shape (batch, n_elec, 1).
        """
        raise NotImplementedError


class MetropolisRandomWalkProposal1D(BaseProposal1D):
    """Propuesta simétrica Random-Walk Metropolis (sin deriva cuántica): X' = X + sigma * xi."""

    is_neural = False

    def distribution(
        self,
        params_actor: dict,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
        use_actor: bool = True,
        key_explore: Optional[jax.Array] = None,
        exploration_std: float = 0.0,
        is_reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict]:
        mu = walkers
        sigma = jnp.ones_like(walkers) * self.rw_sigma
        zeros = jnp.zeros_like(walkers)
        aux = {
            "residual_drift": zeros,
            "scale_log": zeros,
            "actor_drift_mag": jnp.zeros(walkers.shape[0], dtype=walkers.dtype),
            "actor_sigma": jnp.mean(sigma, axis=(-2, -1)),
            "action": jnp.concatenate([zeros, zeros], axis=-1),
        }
        return mu, sigma, aux


class MALAProposal1D(BaseProposal1D):
    """Propuesta clásica Metropolis-Adjusted Langevin Algorithm (MALA)."""

    is_neural = False

    def distribution(
        self,
        params_actor: dict,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
        use_actor: bool = True,
        key_explore: Optional[jax.Array] = None,
        exploration_std: float = 0.0,
        is_reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict]:
        drift = jnp.clip(
            self.drift_prefactor * self.tau * grad_log_psi,
            -self.max_drift_clip,
            self.max_drift_clip,
        )
        mu = walkers + drift
        sigma = jnp.ones_like(walkers) * jnp.sqrt(2.0 * self.tau)
        zeros = jnp.zeros_like(walkers)
        aux = {
            "residual_drift": zeros,
            "scale_log": zeros,
            "actor_drift_mag": jnp.zeros(walkers.shape[0], dtype=walkers.dtype),
            "actor_sigma": jnp.mean(sigma, axis=(-2, -1)),
            "action": jnp.concatenate([zeros, zeros], axis=-1),
        }
        return mu, sigma, aux


class HMCProposal1D(MALAProposal1D):
    """
    Propuesta Hamiltonian Monte Carlo (HMC) en 1D.
    Integra `hmc_leapfrog_steps` pasos simplécticos en `MHSampler1D.step_hmc`.
    """

    is_neural = False
    is_hmc = True


class DeepSetNeuralProposal1D(BaseProposal1D):
    """
    Actor Neuronal DeepSet invariante/equivariante por permutaciones para RL-MALA 1D.
    100% independiente de `src/` y compatible tanto con pesos `.pkl` de `FlaxTD3Actor` como de `Actor_Huziel`.
    """

    is_neural = True

    def __init__(self, cfg: Experiment1DConfig):
        super().__init__(cfg)
        self.actor_amp = cfg.proposal.actor_amplitude
        self.sigma_action_scale = cfg.proposal.sigma_action_scale
        self.action_rev_limit = cfg.proposal.action_rev_limit
        self.use_spatial_gating = cfg.proposal.use_spatial_gating
        self.spatial_gating_radius = cfg.proposal.spatial_gating_radius
        self.proposal_mode = cfg.proposal.proposal_mode

        self.actor_module = FlaxTD3Actor(
            n_up=self.n_up,
            n_dn=self.n_dn,
            actor_amplitude=self.actor_amp,
            sigma_action_scale=self.sigma_action_scale,
            hidden_dim_local=cfg.proposal.hidden_dim_local,
            hidden_dim_global=cfg.proposal.hidden_dim_global,
            hidden_dim_messages=cfg.proposal.hidden_dim_messages,
            hidden_dim_actor=cfg.proposal.hidden_dim_actor,
        )
        self.legacy_huziel_module = FlaxHuzielLegacyActor(
            n_elec=self.n_elec,
            n_up=self.n_up,
            n_dn=self.n_dn,
        )

    def init_params(
        self,
        key: jax.Array,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
    ) -> dict:
        dummy_s = jnp.zeros((1, self.state_dim()))
        return self.actor_module.init(key, dummy_s)

    def forward_action(
        self,
        params_actor: dict,
        state_feat: jax.Array,
        walkers: jax.Array,
        grad_log_psi: Optional[jax.Array] = None,
        use_actor: bool = True,
        key_explore: Optional[jax.Array] = None,
        exploration_std: float = 0.0,
        is_reverse: bool = False,
    ) -> jax.Array:
        if not use_actor or not params_actor:
            return jnp.zeros(
                (walkers.shape[0], walkers.shape[1], 2), dtype=walkers.dtype
            )

        flax_vars = _ensure_flax_vars(params_actor)
        inner_p = flax_vars.get("params", {})
        if isinstance(inner_p, dict) and "local_enc" in inner_p and grad_log_psi is not None:
            # Compatibilidad automática con pesos legacy Actor_Huziel
            def apply_e(e_idx):
                return self.legacy_huziel_module.apply(flax_vars, walkers, e_idx, grad_log_psi)
            base_act = jnp.transpose(jax.vmap(apply_e)(jnp.arange(self.n_elec)), (1, 0, 2))
        else:
            actor_vmap = jax.vmap(self.actor_module.apply, in_axes=(None, 1), out_axes=1)
            base_act = actor_vmap(flax_vars, state_feat)

        if self.use_spatial_gating:
            w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
            gating_vmap = jax.vmap(
                lambda act_e: apply_spatial_gating(
                    w_1d, act_e, l_radius=self.spatial_gating_radius
                ),
                in_axes=1,
                out_axes=1,
            )
            base_act = gating_vmap(base_act)

        clip_lim = self.action_rev_limit if is_reverse else 1.0
        if key_explore is not None:
            noise = exploration_std * jax.random.normal(key_explore, base_act.shape)
            action = jnp.clip(base_act + noise, -clip_lim, clip_lim)
        else:
            action = jnp.clip(base_act, -clip_lim, clip_lim)

        return action

    def distribution_from_action(
        self,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict]:
        act_drift = action[:, :, 0:1]
        act_scale = jnp.clip(
            action[:, :, 1:2],
            self.cfg.proposal.scale_clip_min,
            self.cfg.proposal.scale_clip_max,
        )

        base_step = jnp.sqrt(2.0 * self.tau)
        sigma = jnp.maximum(base_step * jnp.exp(act_scale), 1e-4)

        langevin_drift = jnp.clip(
            self.drift_prefactor * self.tau * grad_log_psi,
            -self.max_drift_clip,
            self.max_drift_clip,
        )
        residual_drift = act_drift * self.actor_amp

        if self.proposal_mode == "scale_only":
            mu = walkers + langevin_drift
            residual_drift = jnp.zeros_like(residual_drift)
        elif self.proposal_mode == "full_gaussian":
            mu = walkers + residual_drift
        else:
            # residual_mala (default)
            mu = walkers + langevin_drift + residual_drift

        aux = {
            "residual_drift": residual_drift,
            "scale_log": act_scale,
            "actor_drift_mag": jnp.mean(jnp.abs(residual_drift), axis=(-2, -1)),
            "actor_sigma": jnp.mean(sigma, axis=(-2, -1)),
            "action": action,
        }
        return mu, sigma, aux

    def distribution(
        self,
        params_actor: dict,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
        use_actor: bool = True,
        key_explore: Optional[jax.Array] = None,
        exploration_std: float = 0.0,
        is_reverse: bool = False,
    ) -> tuple[jax.Array, jax.Array, dict]:
        state_feat = self.build_state_features(walkers, grad_log_psi)
        action = self.forward_action(
            params_actor,
            state_feat,
            walkers,
            grad_log_psi=grad_log_psi,
            use_actor=use_actor,
            key_explore=key_explore,
            exploration_std=exploration_std,
            is_reverse=is_reverse,
        )
        return self.distribution_from_action(walkers, grad_log_psi, action)


class FlatMLPNeuralProposal1D(DeepSetNeuralProposal1D):
    """
    Actor Neuronal MLP Plano sin invarianza de permutación (para ablación vs DeepSet).
    100% independiente de `src/` y compatible con pesos `.pkl` de `FlaxFlatActor`.
    """

    def __init__(self, cfg: Experiment1DConfig):
        super().__init__(cfg)
        self.actor_module = FlaxFlatActor(actor_amplitude=self.actor_amp)

    def state_dim(self) -> int:
        return 2 * self.n_elec + 2

    def build_state_features(
        self,
        walkers: jax.Array,
        grad_log_psi: jax.Array,
    ) -> jax.Array:
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = grad_log_psi[..., 0] if grad_log_psi.ndim == 3 else grad_log_psi

        def per_elec(e_idx):
            tw = jax.lax.dynamic_slice_in_dim(w_1d, e_idx, 1, axis=-1)
            tg = jax.lax.dynamic_slice_in_dim(g_1d, e_idx, 1, axis=-1)
            return jnp.concatenate([tw, tg, w_1d, g_1d], axis=-1)

        feat_e_b_d = jax.vmap(per_elec)(jnp.arange(self.n_elec))
        return jnp.transpose(feat_e_b_d, (1, 0, 2))


def build_proposal_1d(cfg: Experiment1DConfig) -> BaseProposal1D:
    ptype = cfg.proposal.proposal_type
    if ptype == "metropolis":
        return MetropolisRandomWalkProposal1D(cfg)
    elif ptype == "mala":
        return MALAProposal1D(cfg)
    elif ptype == "hmc":
        return HMCProposal1D(cfg)
    elif ptype in ("neural_deepset", "neural_deepset_1d", "deepset"):
        return DeepSetNeuralProposal1D(cfg)
    elif ptype in ("neural_flat", "neural_flat_1d", "flat_mlp"):
        return FlatMLPNeuralProposal1D(cfg)
    else:
        raise ValueError(f"Tipo de propuesta no soportado: {ptype}")
