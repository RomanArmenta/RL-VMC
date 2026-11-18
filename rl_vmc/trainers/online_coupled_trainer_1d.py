from __future__ import annotations

import datetime
import time
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp

from rl_vmc.config import Experiment1DConfig
from rl_vmc.core.types import SamplerState
from rl_vmc.ansatze.ansatz_1d import BaseAnsatz1D
from rl_vmc.proposals.proposals_1d import BaseProposal1D
from rl_vmc.samplers.mh_sampler_1d import MHSampler1D
from rl_vmc.rl.rewards import compute_rl_reward
from rl_vmc.rl.learners_1d import TD3Learner1D, REINFORCELearner1D


def _now_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class OnlineCoupledTrainer1D:
    """
    Entrenador Acoplado Paso a Paso en Dos Escalas de Tiempo (Pipeline 4: `online_coupled`).

    Reutiliza directamente las trazas MCMC (`rl_info`) generadas en cada barrido de
    `MHSampler1D.step` durante la optimización variacional (VMC) para entrenar al vuelo
    el sampler neuronal RL-MALA (TD3 DeepSet) sin detener VMC ni ejecutar bloques extra
    de muestreo con la función de onda congelada.

    Estructura en 4 Regímenes (Fases):
      - Régimen 1 (Nodal Warmup, 1 <= t <= N_nodal):
          Muestrea Psi_theta con MALA puro (use_actor=False, equivalente a acción a=(0,0))
          y optimiza theta mediante VMC sin poblar aún el Replay Buffer, evitando contaminar
          la memoria con el transiente inicial de relajación nodal.
      - Régimen 2 (Off-Policy Buffer Pre-fill & Critic Warmup, N_nodal < t <= N_nodal + N_critic):
          Continúa muestreando con MALA puro (a=(0,0)), recolecta gratis `rl_info` del propio
          paso de VMC, calcula la recompensa CDLB, inserta subconjuntos de caminantes en el
          Replay Buffer circular y entrena ÚNICAMENTE los Críticos gemelos Q_{vartheta_1,2}
          (con Reward Centering), manteniendo el Actor congelado en su inicialización cero.
      - Régimen 3 (Two-Timescale Step-by-Step Co-Evolution, N_nodal + N_critic < t <= T_total - N_freeze):
          Activa el Actor neuronal (use_actor=True) con política determinista (exp_std=0 por defecto
          para preservar el balance detallado exacto respecto a |Psi_{theta_t}|^2), actualiza
          simultáneamente Psi_theta (VMC) y el par Crítico-Actor (TD3) reutilizando las mismas
          muestras de cada paso.
      - Régimen 4 (Diminishing Adaptation / Tail Freeze, T_total - N_freeze < t <= T_total):
          Congela los pesos del Actor y Críticos (adaptación nula, Teorema 2 de Wang et al. 2024)
          y ejecuta los últimos pasos de pulido variacional de Psi_theta bajo un kernel MCMC
          estrictamente estacionario, seleccionando `best_params_psi`.
    """

    def __init__(
        self,
        cfg: Experiment1DConfig,
        ansatz: BaseAnsatz1D,
        proposal: BaseProposal1D,
        sampler: MHSampler1D,
        learner: TD3Learner1D | REINFORCELearner1D | None,
        vmc_update_fn: Callable,
        loaded_actor: bool = False,
    ):
        self.cfg = cfg
        self.ansatz = ansatz
        self.proposal = proposal
        self.sampler = sampler
        self.learner = learner
        self.vmc_update_fn = vmc_update_fn
        self.loaded_actor = bool(loaded_actor)

        self.total_steps = max(1, int(cfg.train.train_steps))
        self.n_nodal_warmup = max(0, int(cfg.train.online_nodal_warmup_steps))
        self.n_critic_warmup = max(0, int(cfg.train.online_critic_warmup_steps))
        self.n_freeze_tail = max(0, int(cfg.train.online_rl_freeze_tail_steps))

        # Ajustar límites si la suma de fases supera total_steps (ej. en pruebas cortas)
        if self.n_nodal_warmup + self.n_critic_warmup + self.n_freeze_tail >= self.total_steps:
            scale = self.total_steps / max(
                1, self.n_nodal_warmup + self.n_critic_warmup + self.n_freeze_tail + 1
            )
            self.n_nodal_warmup = int(round(self.n_nodal_warmup * scale))
            self.n_critic_warmup = int(round(self.n_critic_warmup * scale))
            self.n_freeze_tail = int(round(self.n_freeze_tail * scale))

        self.step_end_phase1 = self.n_nodal_warmup
        self.step_end_phase2 = self.n_nodal_warmup + self.n_critic_warmup
        self.step_end_phase3 = max(self.step_end_phase2, self.total_steps - self.n_freeze_tail)

        self.rl_update_every = max(1, int(cfg.train.online_rl_update_every))
        self.rl_updates_per_step = max(1, int(cfg.train.online_rl_updates_per_step))
        self.walkers_to_store = int(cfg.train.online_walkers_to_store)
        self.online_exp_noise = float(cfg.train.online_exp_noise)
        self.reset_critic_period = int(cfg.train.online_reset_critic_period)

    def get_phase_id_and_label(self, step: int) -> tuple[int, str]:
        if step <= self.step_end_phase1:
            return 1, "P4-R1:NodalWarmup"
        if step <= self.step_end_phase2:
            return 2, "P4-R2:CriticWarmup"
        if step <= self.step_end_phase3:
            return 3, "P4-R3:CoEvolution"
        return 4, "P4-R4:TailFreeze"

    def run(
        self,
        key: jax.Array,
        state: SamplerState,
        params_psi: dict,
        vmc_opt_state: object,
        params_actor: dict,
        best_params_psi: dict,
        best_energy: float,
    ) -> dict:
        t0_total = time.perf_counter()
        phase_wall_times = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        training_history: list[dict] = []

        ema_energy: float | None = None
        ema_var: float | None = None
        params_critic1, params_critic2 = (
            self.learner.get_critic_params()
            if isinstance(self.learner, TD3Learner1D)
            else (None, None)
        )

        print("=" * 76)
        print(f"[{_now_ts()}] [Pipeline 4 | OnlineCoupledTrainer1D] Iniciando entrenamiento acoplado paso a paso")
        print(f"  Pasos VMC Totales         : {self.total_steps} (sweeps_per_step={self.cfg.sampler.sweeps_per_step})")
        print(f"  Régimen 1 (Nodal Warmup)  : pasos 1 .. {self.step_end_phase1} ({self.n_nodal_warmup} pasos, MALA puro, sin buffer)")
        print(f"  Régimen 2 (Critic Warmup) : pasos {self.step_end_phase1 + 1} .. {self.step_end_phase2} ({self.n_critic_warmup} pasos, MALA a=0 -> Buffer + Crítico)")
        print(f"  Régimen 3 (Co-Evolución)  : pasos {self.step_end_phase2 + 1} .. {self.step_end_phase3} ({self.step_end_phase3 - self.step_end_phase2} pasos, Actor + Crítico + VMC)")
        print(f"  Régimen 4 (Tail Freeze)   : pasos {self.step_end_phase3 + 1} .. {self.total_steps} ({self.total_steps - self.step_end_phase3} pasos, Actor congelado + VMC)")
        print(f"  Parámetros RL Online      : update_every={self.rl_update_every}, UTD={self.rl_updates_per_step}, walkers_to_store={self.walkers_to_store}")
        print(f"  Reward Centering / Buffer : reward_centering={self.cfg.rl.use_reward_centering} (lr={self.cfg.rl.reward_centering_lr}), capacity={self.cfg.rl.rl_replay_capacity}")
        print("=" * 76)

        current_phase = 0
        t_phase_start = time.perf_counter()

        for step in range(1, self.total_steps + 1):
            t_step_start = time.perf_counter()
            phase_id, phase_label = self.get_phase_id_and_label(step)

            if phase_id != current_phase:
                now_perf = time.perf_counter()
                if current_phase in phase_wall_times:
                    print(
                        f"[{_now_ts()}] [Pipeline 4] Fin Régimen {current_phase} "
                        f"(tiempo crudo régimen: {phase_wall_times[current_phase]:.2f}s | "
                        f"acumulado: {now_perf - t0_total:.2f}s)"
                    )
                current_phase = phase_id
                t_phase_start = now_perf
                print(
                    f"\n[{_now_ts()}] >>> Entrando a {phase_label} (paso {step}/{self.total_steps}) <<<"
                )

            # Determinar uso del Actor y ruido de exploración según el régimen actual
            if phase_id in (1, 2):
                use_actor_now = bool(self.proposal.is_neural and self.loaded_actor)
                exp_std_now = 0.0
            elif phase_id == 3:
                use_actor_now = bool(self.proposal.is_neural)
                if self.online_exp_noise > 0.0 and self.step_end_phase3 > self.step_end_phase2:
                    frac_p3 = (step - self.step_end_phase2) / max(
                        1, self.step_end_phase3 - self.step_end_phase2
                    )
                    exp_std_now = self.online_exp_noise * max(0.0, 1.0 - frac_p3)
                else:
                    exp_std_now = 0.0
            else:
                # Régimen 4: Actor neuronal activo pero completamente determinista y congelado
                use_actor_now = bool(self.proposal.is_neural)
                exp_std_now = 0.0

            if self.learner is not None:
                params_actor = self.learner.get_actor_params()

            # 1. Muestreo MCMC de Psi_theta (reutilizando rl_info del último sweep sin costo extra)
            rl_info_last = None
            stats_last = None
            for _ in range(self.cfg.sampler.sweeps_per_step):
                key, k_s = jax.random.split(key)
                state, stats_last, rl_info_last = self.sampler.step(
                    k_s,
                    state,
                    params_psi,
                    params_actor,
                    use_actor=use_actor_now,
                    exploration_std=exp_std_now,
                )

            # 2. Recolección gratuita de transiciones y actualización RL (Regímenes 2 y 3)
            r_mean = 0.0
            r_exploit_val = 0.0
            r_entropy_val = 0.0
            r_explore_val = 0.0
            a_loss = 0.0
            c_loss = 0.0
            q_val = 0.0
            r_ema_val = 0.0

            if phase_id in (2, 3) and self.learner is not None and rl_info_last is not None:
                rewards, r_comp = compute_rl_reward(rl_info_last, self.cfg)
                r_mean = float(jnp.mean(rewards))
                r_exploit_val = float(jnp.mean(r_comp["r_exploit"]))
                r_entropy_val = float(jnp.mean(r_comp["r_entropy"]))
                r_explore_val = float(jnp.mean(r_comp["r_explore"]))

                if isinstance(self.learner, TD3Learner1D):
                    self.learner.push_transitions(
                        rl_info_last,
                        rewards,
                        max_walkers=self.walkers_to_store if self.walkers_to_store > 0 else None,
                    )

                    # Reinicio escalonado opcional de un crítico en Fase 3
                    if (
                        phase_id == 3
                        and self.reset_critic_period > 0
                        and (step - self.step_end_phase2) > 0
                        and (step - self.step_end_phase2) % self.reset_critic_period == 0
                    ):
                        key, k_rst = jax.random.split(key)
                        which_c = 1 if ((step // self.reset_critic_period) % 2 == 0) else 2
                        self.learner.reset_one_critic(k_rst, which=which_c)
                        print(
                            f"  [{_now_ts()}] [TD3 Online] Staggered reset aplicado al Crítico {which_c} en paso {step}."
                        )

                    if step % self.rl_update_every == 0:
                        allow_actor = (phase_id == 3)
                        for _ in range(self.rl_updates_per_step):
                            key, k_u = jax.random.split(key)
                            upd = self.learner.update_step(
                                k_u, allow_actor_update=allow_actor
                            )
                            c_loss = upd["critic_loss"]
                            a_loss = upd["actor_loss"]
                            q_val = upd["q_val"]
                            r_ema_val = upd.get("reward_mean_ema", 0.0)
                        params_actor = self.learner.get_actor_params()
                        params_critic1, params_critic2 = self.learner.get_critic_params()

                elif isinstance(self.learner, REINFORCELearner1D) and phase_id == 3:
                    if step % self.rl_update_every == 0:
                        key, k_u = jax.random.split(key)
                        state, upd = self.learner.update_on_policy(k_u, state, params_psi)
                        params_actor = self.learner.get_actor_params()
                        a_loss = upd["actor_loss"]
                        q_val = upd["q_val"]

            # 3. Actualización variacional de Psi_theta (VMC) con los caminantes actuales
            (
                params_psi,
                vmc_opt_state,
                e_m,
                e_v,
                e_cm,
                e_cv,
                t_m,
                v_m,
                g_norm,
            ) = self.vmc_update_fn(params_psi, vmc_opt_state, state.walkers)

            # 4. Refrescar log|Psi_theta(X_t)| y F_Q(X_t) bajo los nuevos pesos variacionales
            lp_new, grad_new = self.ansatz.value_and_grad(
                params_psi,
                state.walkers,
                min_clip=self.cfg.sampler.min_clip_grads_ansatz,
                max_clip=self.cfg.sampler.max_clip_grads_ansatz,
            )
            state = SamplerState(
                walkers=state.walkers, log_psi=lp_new, grad_log_psi=grad_new
            )

            e_val = float(e_m)
            e_clip_val = float(e_cm)
            e_clip_v = float(e_cv)

            if np.isfinite(e_clip_val) and np.isfinite(e_clip_v) and e_clip_val > 0.0:
                if ema_energy is None:
                    ema_energy = e_clip_val
                    ema_var = e_clip_v
                else:
                    ema_energy = 0.95 * ema_energy + 0.05 * e_clip_val
                    ema_var = 0.95 * ema_var + 0.05 * e_clip_v
                var_ok = (e_clip_v <= max(2.5 * ema_var, 5.0)) and (
                    float(e_v) <= max(25.0 * max(ema_var, 1.0), 50.0)
                )
                if step >= self.cfg.train.steps_for_best and var_ok and ema_energy < best_energy:
                    best_energy = ema_energy
                    best_params_psi = jax.tree_util.tree_map(lambda x: x.copy(), params_psi)

            step_dt = time.perf_counter() - t_step_start
            phase_wall_times[phase_id] = phase_wall_times.get(phase_id, 0.0) + step_dt
            elapsed_total = time.perf_counter() - t0_total

            acc_val = float(jnp.mean(stats_last.acceptance))
            esjd_val = float(jnp.mean(stats_last.esjd))
            pdisp_val = float(jnp.mean(stats_last.proposed_disp))

            if (
                step % self.cfg.train.n_log == 0
                or step == 1
                or step == self.total_steps
            ):
                training_history.append(
                    {
                        "Step": step,
                        "Phase": float(phase_id),
                        "Elapsed_Sec": float(elapsed_total),
                        "Energy_Mean": e_clip_val if np.isfinite(e_clip_val) else e_val,
                        "Energy_Var": e_clip_v if np.isfinite(e_clip_v) else float(e_v),
                        "Kinetic_Mean": float(t_m),
                        "Potential_Mean": float(v_m),
                        "Grad_Norm": float(g_norm),
                        "Acceptance": acc_val,
                        "ESJD": esjd_val,
                        "Proposed_Disp": pdisp_val,
                        "RL_Reward": r_mean,
                        "RL_Reward_EMA": r_ema_val,
                        "RL_Exploit": r_exploit_val,
                        "RL_Entropy": r_entropy_val,
                        "RL_Explore": r_explore_val,
                        "Actor_Loss": a_loss,
                        "Critic_Loss": c_loss,
                        "Q_Val": q_val,
                    }
                )

            if (
                step % self.cfg.train.n_print == 0
                or step == 1
                or step == self.total_steps
            ):
                ema_str = f"{ema_energy:.6f}" if ema_energy is not None else "n/a"
                buf_sz = (
                    self.learner.buffer.size
                    if isinstance(self.learner, TD3Learner1D)
                    else 0
                )
                print(
                    f"  [{_now_ts()}] [{phase_label} {step:5d}/{self.total_steps} | t={elapsed_total:7.2f}s] "
                    f"E={e_clip_val:.6f} (EMA={ema_str}, Var={e_clip_v:.4f}) | "
                    f"Acc={acc_val:.2%} | ESJD={esjd_val:.5f} | "
                    f"R={r_mean:+.4f} | L_c={c_loss:.4f} | L_a={a_loss:+.4f} | Buf={buf_sz}"
                )

        total_train_wall_time = time.perf_counter() - t0_total
        print("=" * 76)
        print(
            f"[{_now_ts()}] [Pipeline 4 | OnlineCoupledTrainer1D] Entrenamiento finalizado en {total_train_wall_time:.2f}s"
        )
        print(
            f"  Desglose de tiempo crudo por régimen: "
            f"R1={phase_wall_times[1]:.2f}s | R2={phase_wall_times[2]:.2f}s | "
            f"R3={phase_wall_times[3]:.2f}s | R4={phase_wall_times[4]:.2f}s"
        )
        print("=" * 76)

        return {
            "key": key,
            "state": state,
            "params_psi": params_psi,
            "vmc_opt_state": vmc_opt_state,
            "params_actor": params_actor,
            "params_critic1": params_critic1,
            "params_critic2": params_critic2,
            "best_params_psi": best_params_psi,
            "best_energy": best_energy,
            "training_history": training_history,
            "train_wall_time_sec": float(total_train_wall_time),
            "phase_wall_times_sec": {f"phase_{k}_sec": float(v) for k, v in phase_wall_times.items()},
        }
