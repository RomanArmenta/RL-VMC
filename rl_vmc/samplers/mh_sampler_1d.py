from __future__ import annotations

import functools
import time
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp

from rl_vmc.config import Experiment1DConfig
from rl_vmc.core.types import SamplerState, StepStats
from rl_vmc.ansatze.ansatz_1d import BaseAnsatz1D
from rl_vmc.proposals.proposals_1d import BaseProposal1D, gaussian_log_prob
from rl_vmc.physics.harmonic_1d import Harmonic1DHamiltonian


class MHSampler1D:
    """
    Motor Metropolis-Hastings Markov Chain Monte Carlo para sistemas cuánticos 1D.
    Soporta actualizaciones de todos los electrones (`all_electrons`), barridos de un
    electrón a la vez (`single_electron`), trayectorias simplécticas HMC (`step_hmc`),
    y propuestas tanto clásicas como neuronales.
    """

    def __init__(
        self,
        cfg: Experiment1DConfig,
        ansatz: BaseAnsatz1D,
        proposal: BaseProposal1D,
    ):
        self.cfg = cfg
        self.ansatz = ansatz
        self.proposal = proposal
        self.n_elec = cfg.system.n_elec
        self.min_clip_grad = cfg.sampler.min_clip_grads_ansatz
        self.max_clip_grad = cfg.sampler.max_clip_grads_ansatz
        self.min_clip_lr = cfg.sampler.min_clip_log_ratio
        self.max_clip_lr = cfg.sampler.max_clip_log_ratio
        self.nan_clip_lr = cfg.sampler.nan_clip_log_ratio
        self.posinf_clip_lr = cfg.sampler.posinf_clip_log_ratio
        self.neginf_clip_lr = cfg.sampler.neginf_clip_log_ratio

    def init_state(
        self,
        key: jax.Array,
        params_psi: dict,
        batch_size: Optional[int] = None,
        init_mode: Optional[str] = None,
    ) -> SamplerState:
        bs = batch_size if batch_size is not None else self.cfg.sampler.batch_size
        mode = init_mode if init_mode is not None else getattr(self.cfg.sampler, "init_mode", "symmetric_gaussian")
        omega = max(float(self.cfg.system.omega), 1e-3)
        g = max(float(self.cfg.system.interaction_strength), 0.0)
        init_scale = (1.0 / jnp.sqrt(omega)) * ((1.0 + 0.25 * g) ** 0.25)
        target_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

        if mode == "spin_ordered_offset":
            base_pos = float(getattr(self.cfg.sampler, "init_offset_base", 5.0))
            step_pos = float(getattr(self.cfg.sampler, "init_offset_step", 1.0))
            offset = jnp.arange(self.n_elec, dtype=target_dtype).reshape(1, self.n_elec, 1) * step_pos
            jitter = jax.random.normal(key, (bs, self.n_elec, 1), dtype=target_dtype) * (0.12 * step_pos)
            walkers = jnp.ones((bs, self.n_elec, 1), dtype=target_dtype) * base_pos + offset + jitter
        elif mode == "alternating_spin":
            n_up = int(self.cfg.system.n_up)
            n_dn = int(self.cfg.system.n_dn)
            spread = init_scale * jnp.sqrt(max(float(self.n_elec), 2.0) / 2.5)
            if n_up > 0 and n_dn > 0 and self.n_elec >= 2:
                centers_all = jnp.linspace(-spread, spread, self.n_elec)
                up_centers = centers_all[0::2][:n_up]
                dn_centers = centers_all[1::2][:n_dn]
                if up_centers.shape[0] < n_up:
                    up_centers = jnp.linspace(-spread, spread, n_up)
                if dn_centers.shape[0] < n_dn:
                    dn_centers = jnp.linspace(-spread, spread, n_dn)
                base_centers = jnp.concatenate([up_centers, dn_centers], axis=0)[None, :, None]
            else:
                base_centers = jnp.linspace(-spread, spread, self.n_elec)[None, :, None]
            jitter_std = 0.35 * init_scale / jnp.sqrt(max(float(self.n_elec), 1.0))
            walkers = base_centers.astype(target_dtype) + jax.random.normal(key, (bs, self.n_elec, 1), dtype=target_dtype) * jitter_std
        elif mode == "uniform":
            u_half = float(getattr(self.cfg.sampler, "init_uniform_halfwidth", 0.0))
            if u_half <= 0.0:
                spread = init_scale * jnp.sqrt(max(float(self.n_elec), 2.0) / 2.5)
                u_half = float(spread * 1.5)
            walkers = jax.random.uniform(
                key, (bs, self.n_elec, 1), dtype=target_dtype, minval=-u_half, maxval=u_half
            )
        else:
            # symmetric_gaussian: isotrópico e invariante ante permutaciones de espín (como en Article_plots_11)
            std_val = float(getattr(self.cfg.sampler, "init_std", 3.0))
            if std_val <= 0.0:
                std_val = float(init_scale)
            walkers = jax.random.normal(key, (bs, self.n_elec, 1), dtype=target_dtype) * std_val

        walkers = walkers.astype(target_dtype)
        log_psi, grad_log_psi = self.ansatz.value_and_grad(
            params_psi,
            walkers,
            min_clip=self.min_clip_grad,
            max_clip=self.max_clip_grad,
        )
        return SamplerState(
            walkers=walkers.astype(target_dtype),
            log_psi=log_psi.astype(target_dtype),
            grad_log_psi=grad_log_psi.astype(target_dtype),
        )

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def step_all_electrons(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        params_actor: dict,
        use_actor: bool = True,
        exploration_std: float = 0.0,
    ) -> tuple[SamplerState, StepStats, dict]:
        k_exp, k_prop, k_acc = jax.random.split(key, 3)
        w_curr = state.walkers
        log_p_curr = state.log_psi
        grad_curr = state.grad_log_psi
        target_dtype = w_curr.dtype

        state_feat = self.proposal.build_state_features(w_curr, grad_curr)
        mu_fwd, sigma_fwd, aux_fwd = self.proposal.distribution(
            params_actor,
            w_curr,
            grad_curr,
            use_actor=use_actor,
            key_explore=k_exp,
            exploration_std=exploration_std,
            is_reverse=False,
        )

        noise = jax.random.normal(k_prop, shape=w_curr.shape, dtype=target_dtype)
        w_prop = (mu_fwd + sigma_fwd * noise).astype(target_dtype)

        log_p_prop, grad_prop = self.ansatz.value_and_grad(
            params_psi,
            w_prop,
            min_clip=self.min_clip_grad,
            max_clip=self.max_clip_grad,
        )

        mu_rev, sigma_rev, _ = self.proposal.distribution(
            params_actor,
            w_prop,
            grad_prop,
            use_actor=use_actor,
            key_explore=k_exp,
            exploration_std=exploration_std,
            is_reverse=True,
        )

        log_q_fwd = gaussian_log_prob(w_prop, mu_fwd, sigma_fwd, axis=(-2, -1))
        log_q_rev = gaussian_log_prob(w_curr, mu_rev, sigma_rev, axis=(-2, -1))

        log_p_diff = 2.0 * (log_p_prop - log_p_curr)
        raw_log_ratio = log_p_diff + (log_q_rev - log_q_fwd)
        safe_log_ratio = jnp.nan_to_num(
            raw_log_ratio,
            nan=self.nan_clip_lr,
            posinf=self.posinf_clip_lr,
            neginf=self.neginf_clip_lr,
        )
        clipped_log_ratio = jnp.clip(
            safe_log_ratio, self.min_clip_lr, self.max_clip_lr
        )
        log_alpha = jnp.minimum(0.0, clipped_log_ratio)
        alpha = jnp.exp(log_alpha)

        log_u = jnp.log(jax.random.uniform(k_acc, shape=log_alpha.shape, minval=1e-15))
        accepted = log_u < log_alpha
        acc_mask_3d = accepted[:, None, None]

        w_next = jnp.where(acc_mask_3d, w_prop, w_curr).astype(target_dtype)
        log_p_next = jnp.where(accepted, log_p_prop, log_p_curr).astype(log_p_curr.dtype)
        grad_next = jnp.where(acc_mask_3d, grad_prop, grad_curr).astype(grad_curr.dtype)

        proposed_disp = jnp.mean(jnp.abs(w_prop - w_curr), axis=(-2, -1))
        accepted_disp = jnp.mean(jnp.abs(w_next - w_curr), axis=(-2, -1))
        disp_sq = jnp.sum((w_prop - w_curr) ** 2, axis=(-2, -1))
        esjd = alpha * disp_sq

        next_state = SamplerState(
            walkers=w_next,
            log_psi=log_p_next,
            grad_log_psi=grad_next,
        )
        stats = StepStats(
            acceptance=accepted.astype(jnp.float32),
            alpha=alpha,
            proposed_disp=proposed_disp,
            accepted_disp=accepted_disp,
            esjd=esjd,
            actor_drift_mag=aux_fwd["actor_drift_mag"],
            actor_sigma=aux_fwd["actor_sigma"],
            log_q_fwd=log_q_fwd,
            log_p_diff=log_p_diff,
        )
        next_state_feat = self.proposal.build_state_features(w_next, grad_next)
        rl_info = {
            "state_feat": state_feat,
            "action": aux_fwd["action"],
            "next_state_feat": next_state_feat,
            "log_alpha": log_alpha,
            "alpha": alpha,
            "log_p_diff": log_p_diff,
            "log_q_fwd": log_q_fwd,
            "proposed_disp": proposed_disp,
            "disp_sq": disp_sq,
            "accepted": accepted,
        }
        return next_state, stats, rl_info

    @functools.partial(jax.jit, static_argnums=(0, 5))
    def step_single_electron(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        params_actor: dict,
        use_actor: bool = True,
        exploration_std: float = 0.0,
    ) -> tuple[SamplerState, StepStats, dict]:
        """Barrido completo actualizando un electrón a la vez (e = 0, ..., n_elec - 1)."""
        w_init = state.walkers
        state_feat_init = self.proposal.build_state_features(
            state.walkers, state.grad_log_psi
        )

        def body_fn(e_idx: int, carry):
            (
                k_curr,
                st,
                acc_sum,
                alpha_sum,
                pdisp_sum,
                esjd_sum,
                lqf_sum,
                lpd_sum,
                last_aux,
            ) = carry
            k_next, k_exp, k_prop, k_acc = jax.random.split(k_curr, 4)
            target_dtype = st.walkers.dtype

            mu_fwd, sigma_fwd, aux_fwd = self.proposal.distribution(
                params_actor,
                st.walkers,
                st.grad_log_psi,
                use_actor=use_actor,
                key_explore=k_exp,
                exploration_std=exploration_std,
                is_reverse=False,
            )

            noise_e = jax.random.normal(k_prop, shape=(st.walkers.shape[0], 1), dtype=target_dtype)
            mu_e = mu_fwd[:, e_idx, :]
            sigma_e = sigma_fwd[:, e_idx, :]
            w_prop_e = (mu_e + sigma_e * noise_e).astype(target_dtype)
            w_prop = st.walkers.at[:, e_idx, :].set(w_prop_e)

            log_p_prop, grad_prop = self.ansatz.value_and_grad(
                params_psi,
                w_prop,
                min_clip=self.min_clip_grad,
                max_clip=self.max_clip_grad,
            )

            mu_rev, sigma_rev, _ = self.proposal.distribution(
                params_actor,
                w_prop,
                grad_prop,
                use_actor=use_actor,
                key_explore=k_exp,
                exploration_std=exploration_std,
                is_reverse=True,
            )
            mu_rev_e = mu_rev[:, e_idx, :]
            sigma_rev_e = sigma_rev[:, e_idx, :]
            w_curr_e = st.walkers[:, e_idx, :]

            log_q_fwd = gaussian_log_prob(w_prop_e, mu_e, sigma_e, axis=(-1,))
            log_q_rev = gaussian_log_prob(w_curr_e, mu_rev_e, sigma_rev_e, axis=(-1,))

            log_p_diff = 2.0 * (log_p_prop - st.log_psi)
            raw_lr = log_p_diff + (log_q_rev - log_q_fwd)
            safe_lr = jnp.nan_to_num(
                raw_lr,
                nan=self.nan_clip_lr,
                posinf=self.posinf_clip_lr,
                neginf=self.neginf_clip_lr,
            )
            log_alpha = jnp.minimum(0.0, jnp.clip(safe_lr, self.min_clip_lr, self.max_clip_lr))
            alpha = jnp.exp(log_alpha)

            log_u = jnp.log(
                jax.random.uniform(k_acc, shape=log_alpha.shape, minval=1e-15)
            )
            accepted = log_u < log_alpha
            acc_mask_3d = accepted[:, None, None]

            w_next = jnp.where(acc_mask_3d, w_prop, st.walkers).astype(target_dtype)
            log_p_next = jnp.where(accepted, log_p_prop, st.log_psi).astype(st.log_psi.dtype)
            grad_next = jnp.where(acc_mask_3d, grad_prop, st.grad_log_psi).astype(st.grad_log_psi.dtype)

            disp_e = jnp.abs(w_prop_e[:, 0] - w_curr_e[:, 0])
            disp_e_sq = disp_e ** 2
            esjd_e = alpha * disp_e_sq

            st_next = SamplerState(
                walkers=w_next,
                log_psi=log_p_next,
                grad_log_psi=grad_next,
            )
            return (
                k_next,
                st_next,
                acc_sum + accepted.astype(jnp.float32),
                alpha_sum + alpha,
                pdisp_sum + disp_e,
                esjd_sum + esjd_e,
                lqf_sum + log_q_fwd,
                lpd_sum + log_p_diff,
                aux_fwd,
            )

        bs = state.walkers.shape[0]
        zeros_b = jnp.zeros(bs, dtype=state.walkers.dtype)
        zeros_w = jnp.zeros_like(state.walkers)
        init_aux = {
            "residual_drift": zeros_w,
            "scale_log": zeros_w,
            "actor_drift_mag": zeros_b,
            "actor_sigma": zeros_b,
            "action": jnp.concatenate([zeros_w, zeros_w], axis=-1),
        }
        init_carry = (
            key,
            state,
            zeros_b,
            zeros_b,
            zeros_b,
            zeros_b,
            zeros_b,
            zeros_b,
            init_aux,
        )
        (
            _,
            next_state,
            acc_sum,
            alpha_sum,
            pdisp_sum,
            esjd_sum,
            lqf_sum,
            lpd_sum,
            last_aux,
        ) = jax.lax.fori_loop(0, self.n_elec, body_fn, init_carry)

        inv_n = 1.0 / float(self.n_elec)
        mean_acc = acc_sum * inv_n
        mean_alpha = alpha_sum * inv_n
        mean_pdisp = pdisp_sum * inv_n
        accepted_disp = jnp.mean(jnp.abs(next_state.walkers - w_init), axis=(-2, -1))

        stats = StepStats(
            acceptance=mean_acc,
            alpha=mean_alpha,
            proposed_disp=mean_pdisp,
            accepted_disp=accepted_disp,
            esjd=esjd_sum,
            actor_drift_mag=last_aux["actor_drift_mag"],
            actor_sigma=last_aux["actor_sigma"],
            log_q_fwd=lqf_sum,
            log_p_diff=lpd_sum,
        )
        next_state_feat = self.proposal.build_state_features(
            next_state.walkers, next_state.grad_log_psi
        )
        rl_info = {
            "state_feat": state_feat_init,
            "action": last_aux["action"],
            "next_state_feat": next_state_feat,
            "log_alpha": jnp.log(jnp.clip(mean_alpha, 1e-12, 1.0)),
            "alpha": mean_alpha,
            "log_p_diff": lpd_sum,
            "log_q_fwd": lqf_sum,
            "proposed_disp": mean_pdisp,
            "disp_sq": (mean_pdisp ** 2) * float(self.n_elec),
            "accepted": mean_acc > 0.5,
        }
        return next_state, stats, rl_info

    @functools.partial(jax.jit, static_argnums=(0,))
    def step_hmc(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
    ) -> tuple[SamplerState, StepStats, dict]:
        """Paso Hamiltonian Monte Carlo (HMC) con integrador simpléctico Leapfrog."""
        k_mom, k_acc = jax.random.split(key, 2)
        w_curr = state.walkers
        log_p_curr = state.log_psi
        grad_curr = state.grad_log_psi
        target_dtype = w_curr.dtype

        eps = self.cfg.proposal.hmc_step_size
        n_leapfrog = max(1, self.cfg.proposal.hmc_leapfrog_steps)

        p_init = jax.random.normal(k_mom, shape=w_curr.shape, dtype=target_dtype)
        p = (p_init + 0.5 * eps * (2.0 * grad_curr)).astype(target_dtype)
        w = w_curr
        log_p = log_p_curr
        grad = grad_curr

        def leapfrog_body(i, carry):
            w_c, p_c, lp_c, g_c = carry
            w_n = (w_c + eps * p_c).astype(target_dtype)
            lp_n, g_n = self.ansatz.value_and_grad(
                params_psi,
                w_n,
                min_clip=self.min_clip_grad,
                max_clip=self.max_clip_grad,
            )
            lp_n = lp_n.astype(log_p_curr.dtype)
            g_n = g_n.astype(grad_curr.dtype)
            p_n = jnp.where(
                i < n_leapfrog - 1,
                p_c + eps * (2.0 * g_n),
                p_c + 0.5 * eps * (2.0 * g_n),
            ).astype(target_dtype)
            return (w_n, p_n, lp_n, g_n)

        w_prop, p_prop, log_p_prop, grad_prop = jax.lax.fori_loop(
            0, n_leapfrog, leapfrog_body, (w, p, log_p, grad)
        )

        kin_curr = 0.5 * jnp.sum(p_init ** 2, axis=(-2, -1))
        kin_prop = 0.5 * jnp.sum(p_prop ** 2, axis=(-2, -1))
        log_p_diff = 2.0 * (log_p_prop - log_p_curr)
        log_alpha = jnp.minimum(0.0, log_p_diff + kin_curr - kin_prop)
        alpha = jnp.exp(log_alpha)

        log_u = jnp.log(jax.random.uniform(k_acc, shape=log_alpha.shape, minval=1e-15))
        accepted = log_u < log_alpha
        acc_mask_3d = accepted[:, None, None]

        w_next = jnp.where(acc_mask_3d, w_prop, w_curr).astype(target_dtype)
        log_p_next = jnp.where(accepted, log_p_prop, log_p_curr).astype(log_p_curr.dtype)
        grad_next = jnp.where(acc_mask_3d, grad_prop, grad_curr).astype(grad_curr.dtype)

        proposed_disp = jnp.mean(jnp.abs(w_prop - w_curr), axis=(-2, -1))
        accepted_disp = jnp.mean(jnp.abs(w_next - w_curr), axis=(-2, -1))
        disp_sq = jnp.sum((w_prop - w_curr) ** 2, axis=(-2, -1))
        esjd = alpha * disp_sq

        zeros_b = jnp.zeros(w_curr.shape[0], dtype=w_curr.dtype)
        zeros_w = jnp.zeros_like(w_curr)
        next_state = SamplerState(
            walkers=w_next,
            log_psi=log_p_next,
            grad_log_psi=grad_next,
        )
        stats = StepStats(
            acceptance=accepted.astype(jnp.float32),
            alpha=alpha,
            proposed_disp=proposed_disp,
            accepted_disp=accepted_disp,
            esjd=esjd,
            actor_drift_mag=zeros_b,
            actor_sigma=jnp.ones_like(zeros_b) * eps,
            log_q_fwd=-kin_prop,
            log_p_diff=log_p_diff,
        )
        rl_info = {
            "state_feat": self.proposal.build_state_features(w_curr, grad_curr),
            "action": jnp.concatenate([zeros_w, zeros_w], axis=-1),
            "next_state_feat": self.proposal.build_state_features(w_next, grad_next),
            "log_alpha": log_alpha,
            "alpha": alpha,
            "log_p_diff": log_p_diff,
            "log_q_fwd": -kin_prop,
            "proposed_disp": proposed_disp,
            "disp_sq": disp_sq,
            "accepted": accepted,
        }
        return next_state, stats, rl_info

    def step(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        params_actor: dict,
        use_actor: bool = True,
        exploration_std: float = 0.0,
    ) -> tuple[SamplerState, StepStats, dict]:
        if self.cfg.proposal.proposal_type == "hmc":
            return self.step_hmc(key, state, params_psi)
        elif self.cfg.sampler.update_mode == "single_electron":
            return self.step_single_electron(
                key,
                state,
                params_psi,
                params_actor,
                use_actor=use_actor,
                exploration_std=exploration_std,
            )
        else:
            return self.step_all_electrons(
                key,
                state,
                params_psi,
                params_actor,
                use_actor=use_actor,
                exploration_std=exploration_std,
            )

    @functools.partial(jax.jit, static_argnums=(0, 5, 6))
    def run_burn_in(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        params_actor: dict,
        n_steps: int,
        use_actor: bool = True,
    ) -> tuple[jax.Array, SamplerState]:
        if n_steps <= 0:
            return key, state

        def body_fn(_, carry):
            k_c, st = carry
            k_next, k_step = jax.random.split(k_c)
            st_next, _, _ = self.step(
                k_step,
                st,
                params_psi,
                params_actor,
                use_actor=use_actor,
                exploration_std=0.0,
            )
            return (k_next, st_next)

        return jax.lax.fori_loop(0, n_steps, body_fn, (key, state))

    def run_production_chain(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        params_actor: dict,
        hamiltonian: Harmonic1DHamiltonian,
        measure_steps: int,
        thin: int = 1,
        jit_chunk_size: int = 1000,
        use_actor: bool = True,
    ) -> dict[str, np.ndarray | float]:
        """
        Ejecuta una cadena de producción de longitud `measure_steps` compilada por bloques `jit_chunk_size`.
        """
        chunk_len = min(jit_chunk_size, measure_steps)
        n_full_chunks = measure_steps // chunk_len
        remainder = measure_steps % chunk_len

        @jax.jit
        def run_chunk(k_in: jax.Array, st_in: SamplerState):
            def scan_body(carry, _):
                k_c, st = carry
                k_next, k_step = jax.random.split(k_c)
                st_next, stats, _ = self.step(
                    k_step,
                    st,
                    params_psi,
                    params_actor,
                    use_actor=use_actor,
                    exploration_std=0.0,
                )
                e_loc, t_loc, v_loc = hamiltonian.local_energy_and_components(
                    self.ansatz.apply, params_psi, st_next.walkers
                )
                step_out = {
                    "walkers": st_next.walkers,
                    "e_loc": e_loc,
                    "t_loc": t_loc,
                    "v_loc": v_loc,
                    "acceptance": jnp.mean(stats.acceptance),
                    "alpha": jnp.mean(stats.alpha),
                    "proposed_disp": jnp.mean(stats.proposed_disp),
                    "accepted_disp": jnp.mean(stats.accepted_disp),
                    "esjd": jnp.mean(stats.esjd),
                    "actor_drift_mag": jnp.mean(stats.actor_drift_mag),
                    "actor_sigma": jnp.mean(stats.actor_sigma),
                }
                return (k_next, st_next), step_out

            return jax.lax.scan(scan_body, (k_in, st_in), None, length=chunk_len)

        walkers_list = []
        e_loc_list, t_loc_list, v_loc_list = [], [], []
        acc_list, alpha_list = [], []
        pdisp_list, adisp_list, esjd_list = [], [], []
        adrift_list, asigma_list = [], []

        curr_key = key
        curr_state = state
        global_step = 0
        t0 = time.perf_counter()

        for _ in range(n_full_chunks):
            (curr_key, curr_state), out = run_chunk(curr_key, curr_state)
            jax.block_until_ready(out["e_loc"])

            idx = np.arange(chunk_len)
            mask = ((global_step + idx) % thin) == 0
            global_step += chunk_len

            walkers_list.append(np.asarray(out["walkers"])[mask])
            e_loc_list.append(np.asarray(out["e_loc"])[mask])
            t_loc_list.append(np.asarray(out["t_loc"])[mask])
            v_loc_list.append(np.asarray(out["v_loc"])[mask])
            acc_list.append(np.asarray(out["acceptance"])[mask])
            alpha_list.append(np.asarray(out["alpha"])[mask])
            pdisp_list.append(np.asarray(out["proposed_disp"])[mask])
            adisp_list.append(np.asarray(out["accepted_disp"])[mask])
            esjd_list.append(np.asarray(out["esjd"])[mask])
            adrift_list.append(np.asarray(out["actor_drift_mag"])[mask])
            asigma_list.append(np.asarray(out["actor_sigma"])[mask])

        if remainder > 0:
            for _ in range(remainder):
                curr_key, k_step = jax.random.split(curr_key)
                curr_state, stats, _ = self.step(
                    k_step,
                    curr_state,
                    params_psi,
                    params_actor,
                    use_actor=use_actor,
                    exploration_std=0.0,
                )
                if (global_step % thin) == 0:
                    e_loc, t_loc, v_loc = hamiltonian.local_energy_and_components(
                        self.ansatz.apply, params_psi, curr_state.walkers
                    )
                    walkers_list.append(np.asarray(curr_state.walkers)[None, ...])
                    e_loc_list.append(np.asarray(e_loc)[None, ...])
                    t_loc_list.append(np.asarray(t_loc)[None, ...])
                    v_loc_list.append(np.asarray(v_loc)[None, ...])
                    acc_list.append(np.asarray(jnp.mean(stats.acceptance))[None])
                    alpha_list.append(np.asarray(jnp.mean(stats.alpha))[None])
                    pdisp_list.append(np.asarray(jnp.mean(stats.proposed_disp))[None])
                    adisp_list.append(np.asarray(jnp.mean(stats.accepted_disp))[None])
                    esjd_list.append(np.asarray(jnp.mean(stats.esjd))[None])
                    adrift_list.append(np.asarray(jnp.mean(stats.actor_drift_mag))[None])
                    asigma_list.append(np.asarray(jnp.mean(stats.actor_sigma))[None])
                global_step += 1

        wall_time = time.perf_counter() - t0

        return {
            "walkers": np.concatenate(walkers_list, axis=0),
            "local_energy": np.concatenate(e_loc_list, axis=0),
            "kinetic_energy": np.concatenate(t_loc_list, axis=0),
            "potential_energy": np.concatenate(v_loc_list, axis=0),
            "sampler_stats": {
                "acceptance": np.concatenate(acc_list, axis=0),
                "alpha": np.concatenate(alpha_list, axis=0),
                "proposed_disp": np.concatenate(pdisp_list, axis=0),
                "accepted_disp": np.concatenate(adisp_list, axis=0),
                "esjd": np.concatenate(esjd_list, axis=0),
                "actor_drift_mag": np.concatenate(adrift_list, axis=0),
                "actor_sigma": np.concatenate(asigma_list, axis=0),
            },
            "wall_time_sec": float(wall_time),
            "final_state": curr_state,
        }
