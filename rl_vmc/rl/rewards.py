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
    if getattr(cfg.rl, "cdlb_exact_entropy", False):
        # Restaura la constante exacta de normalización Gaussiana +(N/2)*log(2*pi)
        # omitida en gaussian_log_prob (que solo cancela en MH, NO en -alpha*log_q_fwd).
        # Garantiza entropía diferencial positiva en el rango físico y elimina el exploit alpha -> 0.
        r_explore = r_explore + alpha * (
            0.5 * float(cfg.system.n_elec) * jnp.log(2.0 * jnp.pi)
        )

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

    # Regularización física par-simétrica de confinamiento armónico (opcional; 0.0 por defecto).
    # Preserva balance detallado exacto en evaluación (no modifica el kernel MH) y paridad x -> -x.
    r_confinement = jnp.zeros_like(reward)
    r_outward = jnp.zeros_like(reward)
    conf_w = float(getattr(cfg.rl, "confinement_weight", 0.0))
    out_w = float(getattr(cfg.rl, "outward_drift_penalty", 0.0))
    omega = float(cfg.system.omega)

    if conf_w > 0.0 and ("walkers_curr" in rl_info) and ("walkers_prop" in rl_info):
        w_curr = rl_info["walkers_curr"]
        w_prop = rl_info["walkers_prop"]
        v_curr = 0.5 * (omega ** 2) * jnp.mean(w_curr ** 2, axis=(-2, -1))
        v_prop = 0.5 * (omega ** 2) * jnp.mean(w_prop ** 2, axis=(-2, -1))
        r_confinement = -conf_w * (1.0 - alpha) * jax.nn.relu(v_prop - v_curr)
        reward = reward + r_confinement

    if out_w > 0.0 and ("walkers_curr" in rl_info) and ("action" in rl_info):
        w_curr = rl_info["walkers_curr"]
        x_curr = w_curr[..., 0] if w_curr.ndim == 3 else w_curr
        a_mu = rl_info["action"][..., 0]
        res_drift = float(cfg.proposal.actor_amplitude) * a_mu
        # Producto par-simétrico: bajo x -> -x, res_drift -> -res_drift y tanh(sqrt(omega)*x) -> -tanh(sqrt(omega)*x)
        outward_align = res_drift * jnp.tanh(jnp.sqrt(max(omega, 1e-6)) * x_curr)
        r_outward = -out_w * jnp.mean(jax.nn.relu(outward_align), axis=-1)
        reward = reward + r_outward

    reward = jnp.nan_to_num(reward, nan=-10.0, posinf=50.0, neginf=-50.0)
    reward = jnp.clip(reward, -50.0, 50.0)
    return reward, {
        "r_exploit": r_exploit,
        "r_entropy": r_entropy,
        "r_explore": r_explore,
        "r_confinement": r_confinement,
        "r_outward": r_outward,
    }
