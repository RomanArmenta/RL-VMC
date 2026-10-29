from __future__ import annotations

import jax
import jax.numpy as jnp
from rl_vmc.config import Experiment1DConfig


def compute_rl_reward(
    rl_info: dict,
    cfg: Experiment1DConfig,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """
    Calcula la recompensa por caminante para entrenar el Actor MCMC según `cfg.rl.reward_type`.
    Soporta:
      - `cdlb`: Capacitated Detailed-Balance Lower Bound (Wang et al., 2026)
      - `log_esjd`: Log Expected Squared Jump Distance regularizado
      - `barker_jump`: Barker acceptance ponderado por distancia física
      - `kappa_jump`: Distancia esperada + término de varianza de aceptación kappa*alpha*(1-alpha)
    Retorna:
      (reward, components_dict) donde cada tensor tiene shape (batch_size,).
    """
    alpha = rl_info["alpha"]
    log_p_diff = rl_info["log_p_diff"]
    log_q_fwd = rl_info["log_q_fwd"]
    disp = rl_info["proposed_disp"]

    max_phys_jump = cfg.rl.max_phys_jump_factor * jnp.sqrt(cfg.proposal.tau)
    clipped_dist = jnp.clip(disp, 0.0, max_phys_jump)

    r_exploit = alpha * log_p_diff
    r_entropy = -alpha * jnp.log(alpha + 1e-8) - (1.0 - alpha) * jnp.log(
        1.0 - alpha + 1e-8
    )
    r_explore = -alpha * log_q_fwd

    rtype = cfg.rl.reward_type
    if rtype == "cdlb":
        reward = cfg.rl.penalty_factor * (r_exploit + r_entropy + r_explore)
    elif rtype == "log_esjd":
        reward = jnp.log(cfg.rl.reward_epsilon + alpha * (clipped_dist ** 2))
    elif rtype == "barker_jump":
        barker_p = alpha / (1.0 + alpha)
        reward = barker_p * (cfg.rl.beta * clipped_dist) + cfg.rl.gamma * alpha
    elif rtype == "kappa_jump":
        reward = (alpha * clipped_dist) + cfg.rl.kappa * (1.0 - alpha) * alpha
    else:
        raise ValueError(f"Tipo de recompensa RL no soportado: {rtype}")

    reward = jnp.nan_to_num(reward, nan=-10.0, posinf=50.0, neginf=-50.0)
    reward = jnp.clip(reward, -50.0, 50.0)
    return reward, {
        "r_exploit": r_exploit,
        "r_entropy": r_entropy,
        "r_explore": r_explore,
    }
