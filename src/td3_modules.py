import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import numpy as np
from functools import partial

# ==========================================
# 1. REPLAY BUFFER (Basado en NumPy)
# ==========================================
class ReplayBuffer:
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
            s, a, r, ns = s[-self.capacity:], a[-self.capacity:], r[-self.capacity:], ns[-self.capacity:]
            n = self.capacity
        idx = (np.arange(n) + self.pos) % self.capacity
        self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx] = s, a, r, ns
        self.pos = int((self.pos + n) % self.capacity)
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "state": jnp.asarray(self.state[idx]),
            "action": jnp.asarray(self.action[idx]),
            "reward": jnp.asarray(self.reward[idx]),
            "next_state": jnp.asarray(self.next_state[idx])
        }

# ==========================================
# 2. EXTRACTOR DE CARACTERÍSTICAS (Deep Sets)
# ==========================================
def build_td3_state(x, g, elec_idx, n_elec, n_up, n_dn):
    # 1. BLINDAJE DE DIMENSIONES: Aseguramos que x y g sean puramente 2D [batch, N]
    if x.ndim == 3: x = x[..., 0]
    if g.ndim == 3: g = g[..., 0]
    
    batch_size = x.shape[0]
    i = elec_idx
    x_i, g_i = x[:, i], g[:, i]
    
    spins = jnp.concatenate([jnp.ones(n_up), -jnp.ones(n_dn)])
    spin_i = spins[i]
    
    # local_feat tendrá forma exacta de [batch_size, 4]
    local_feat = jnp.stack([
        x_i, 
        x_i - jnp.mean(x, axis=1), 
        jnp.tanh(g_i / 5.0), 
        jnp.full_like(x_i, spin_i)
    ], axis=-1)
    
    global_feat = jnp.asarray([n_elec / 10.0, n_up / max(n_elec, 1), jnp.tanh(1.0), jnp.tanh(1.0), jnp.tanh(0.25)])
    global_feat = jnp.broadcast_to(global_feat[None, :], (batch_size, 5))
    
    # Cálculos de distancias vectorizados
    dx = x[:, i, None] - x
    r = jnp.abs(dx)
    
    # Máscaras
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
        
        return jnp.stack([
            jnp.where(mm, rr / 5.0, 0.0), 
            jnp.where(mm, jnp.sign(dd), 0.0), 
            mm.astype(jnp.float32)
        ], axis=-1)
        
    max_same = max(max(n_up - 1, 0), max(n_dn - 1, 0))
    max_opp = max(n_up, n_dn) if (n_up > 0 and n_dn > 0) else 0
    
    same_slots = get_slots(r, dx, same_mask, max_same).reshape((batch_size, -1))
    opp_slots = get_slots(r, dx, opp_mask, max_opp).reshape((batch_size, -1))
    
    return jnp.concatenate([local_feat, global_feat, same_slots, opp_slots], axis=-1)

# ==========================================
# 3. REDES NEURONALES (FLAX)
# ==========================================
class FlaxTD3Actor(nn.Module):
    n_up: int
    n_dn: int
    actor_amplitude: float = 0.05
    
    @nn.compact
    def __call__(self, state):
        max_same, max_opp = max(max(self.n_up - 1, 0), max(self.n_dn - 1, 0)), max(self.n_up, self.n_dn) if (self.n_up > 0 and self.n_dn > 0) else 0
        p = 0
        local_feat = state[..., p : p+4]; p += 4
        global_feat = state[..., p : p+5]; p += 5
        same_slots = state[..., p : p + 3*max_same].reshape(state.shape[:-1] + (max_same, 3)); p += 3*max_same
        opp_slots = state[..., p : p + 3*max_opp].reshape(state.shape[:-1] + (max_opp, 3))
        
        h_local = nn.silu(nn.Dense(32)(nn.silu(nn.Dense(32)(local_feat))))
        h_global = nn.silu(nn.Dense(16)(nn.silu(nn.Dense(16)(global_feat))))
        
        msg_same = nn.silu(nn.Dense(32)(nn.silu(nn.Dense(32)(same_slots))))
        mask_same = same_slots[..., 2:3]
        pool_same = jnp.sum(msg_same * mask_same, axis=-2) / jnp.maximum(jnp.sum(mask_same, axis=-2), 1.0)
        
        msg_opp = nn.silu(nn.Dense(32)(nn.silu(nn.Dense(32)(opp_slots))))
        mask_opp = opp_slots[..., 2:3]
        pool_opp = jnp.sum(msg_opp * mask_opp, axis=-2) / jnp.maximum(jnp.sum(mask_opp, axis=-2), 1.0)
        
        fused = jnp.concatenate([h_local, h_global, pool_same, pool_opp], axis=-1)
        h = nn.silu(nn.Dense(64)(fused))
        
        mean = nn.tanh(nn.Dense(1, kernel_init=nn.initializers.zeros)(h))
        scale = nn.tanh(nn.Dense(1, kernel_init=nn.initializers.zeros)(h))
        return jnp.concatenate([mean, scale], axis=-1)

class FlaxTD3Critic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.silu(nn.Dense(64)(nn.silu(nn.Dense(64)(x))))
        return nn.Dense(1)(x)[..., 0]

# ==========================================
# 4. TD3 TRAINER SECUENCIAL
# ==========================================
class TD3Trainer:
    def __init__(self, ansatz_apply, n_elec, n_up, n_dn, tau, actor_amp, 
                 beta=10.0, gamma_reward=0.1, gamma_td3=0.99, tau_polyak=0.005, 
                 target_noise=0.15, noise_clip=0.35, epsilon=0.001):
        self.ansatz_apply = ansatz_apply
        self.n_elec, self.n_up, self.n_dn = n_elec, n_up, n_dn
        self.tau, self.actor_amp = tau, actor_amp
        self.beta, self.gamma_reward = beta, gamma_reward
        self.gamma_td3, self.tau_polyak = gamma_td3, tau_polyak
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.epsilon = epsilon
        
        self.actor_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp)
        self.critic_model = FlaxTD3Critic()

    @partial(jax.jit, static_argnums=(0,))
    def get_logp_and_grad(self, params_psi, walkers):
        def single_logp(w): return self.ansatz_apply(params_psi, w[None, ...])[0]
        return jax.vmap(jax.value_and_grad(single_logp))(walkers)

    @partial(jax.jit, static_argnums=(0, 7))
    def td3_update_one_electron(self, key, walkers, curr_logp, curr_grad, params_psi, params_actor, elec_idx, exp_std):
        n_w = walkers.shape[0]
        k_act, k_prop, k_acc = jax.random.split(key, 3)
        state_old = build_td3_state(walkers, curr_grad, elec_idx, self.n_elec, self.n_up, self.n_dn)
        
        base_act = self.actor_model.apply(params_actor, state_old)
        aux_noise = exp_std * jax.random.normal(k_act, base_act.shape)
        action_old = jnp.clip(base_act + aux_noise, -1.0, 1.0)
        
        base_step = jnp.sqrt(2.0 * self.tau)
        sigma_fwd = jnp.maximum(base_step * jnp.exp(action_old[..., 1]), 1e-4)
        
        # [CAMBIO 1: Agregamos el '0' para aplanar todo a (batch_size,)]
        drift_fwd = jnp.clip(self.tau * curr_grad[:, elec_idx, 0], -10.0, 10.0)
        residual_fwd = base_step * action_old[..., 0] * self.actor_amp
        
        old_coord = walkers[:, elec_idx, 0] 
        mean_fwd_dist = old_coord + drift_fwd + residual_fwd
        proposed_coord = mean_fwd_dist + sigma_fwd * jax.random.normal(k_prop, (n_w,))
        
        # [CAMBIO 2: Escribimos específicamente en la ranura aplanada]
        proposal = walkers.at[:, elec_idx, 0].set(proposed_coord)
        
        new_logp, new_grad = self.get_logp_and_grad(params_psi, proposal)
        state_prop = build_td3_state(proposal, new_grad, elec_idx, self.n_elec, self.n_up, self.n_dn)
        action_rev = jnp.clip(self.actor_model.apply(params_actor, state_prop) + aux_noise, -1.0, 1.0)
        
        sigma_rev = jnp.maximum(base_step * jnp.exp(action_rev[..., 1]), 1e-4)
        drift_rev = jnp.clip(self.tau * new_grad[:, elec_idx, 0], -10.0, 10.0)
        residual_rev = base_step * action_rev[..., 0] * self.actor_amp
        mean_rev_dist = proposed_coord + drift_rev + residual_rev
        
        log_q_fwd = -jnp.log(sigma_fwd) - 0.5 * ((proposed_coord - mean_fwd_dist) / sigma_fwd)**2
        log_q_rev = -jnp.log(sigma_rev) - 0.5 * ((old_coord - mean_rev_dist) / sigma_rev)**2
        
        clipped_log_alpha = jnp.minimum(0.0, (new_logp - curr_logp) + (log_q_rev - log_q_fwd))
        accept = jnp.log(jax.random.uniform(k_acc, (n_w,), minval=1e-15)) < clipped_log_alpha
        
        # [CAMBIO 3: Usamos [:, None, None] para proyectar booleanos sobre (batch_size, n_elec, 1)]
        next_w = jnp.where(accept[:, None, None], proposal, walkers)
        next_logp = jnp.where(accept, new_logp, curr_logp)
        next_grad = jnp.where(accept[:, None, None], new_grad, curr_grad)
        
        disp = jnp.abs(proposed_coord - old_coord)
        # reward = jnp.clip(self.beta * 2.0 * jnp.log(disp + 1e-8) + self.gamma_reward * clipped_log_alpha, -30.0, 30.0)
        # Reemplazar: reward = jnp.clip(self.beta * 2.0 * jnp.log(disp + 1e-8) + ...)
        
        # 1. Extraer probabilidad de aceptación (alpha_exact)
        alpha_exact = jnp.exp(clipped_log_alpha)
        
        # 2. Distancia recortada al salto físico máximo
        max_phys_jump = 5.0 * jnp.sqrt(self.tau) # Equivale a args.max_phys_jump_std
        clipped_dist = jnp.clip(disp, a_min=0.0, a_max=max_phys_jump)
        
        # 3. Nueva recompensa idéntica a REINFORCE (ignoramos soft_node_penalty por rendimiento)
        kappa = 0.5 # Mismo valor que args.kappa
        # reward = (alpha_exact * clipped_dist) + kappa * (1.0 - alpha_exact) * alpha_exact
        reward = jnp.log(self.epsilon + alpha_exact * clipped_dist ** 2)
        next_state = build_td3_state(next_w, next_grad, (elec_idx + 1) % self.n_elec, self.n_elec, self.n_up, self.n_dn)
        
        return next_w, next_logp, next_grad, state_old, action_old, reward, next_state, accept, disp

    @partial(jax.jit, static_argnums=(0,))
    def train_networks(self, key, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_actor):
        noise = jnp.clip(self.target_noise * jax.random.normal(key, batch["action"].shape), -self.noise_clip, self.noise_clip)
        next_action = jnp.clip(self.actor_model.apply(t_act, batch["next_state"]) + noise, -1.0, 1.0)
        y = jax.lax.stop_gradient(batch["reward"] + self.gamma_td3 * jnp.minimum(
            self.critic_model.apply(t_c1, batch["next_state"], next_action),
            self.critic_model.apply(t_c2, batch["next_state"], next_action)
        ))
        
        def critic_loss(p_crit): return jnp.mean((self.critic_model.apply(p_crit, batch["state"], batch["action"]) - y)**2)
        l1, g1 = jax.value_and_grad(critic_loss)(p_c1)
        l2, g2 = jax.value_and_grad(critic_loss)(p_c2)
        
        u1, opt_c1 = self.opt_c.update(g1, opt_c1, p_c1); p_c1 = optax.apply_updates(p_c1, u1)
        u2, opt_c2 = self.opt_c.update(g2, opt_c2, p_c2); p_c2 = optax.apply_updates(p_c2, u2)
        
        def actor_update():
            def actor_loss(p_a): return -jnp.mean(self.critic_model.apply(p_c1, batch["state"], self.actor_model.apply(p_a, batch["state"])))
            l_act, g_act = jax.value_and_grad(actor_loss)(p_act)
            u_act, new_opt_act = self.opt_a.update(g_act, opt_act, p_act)
            new_p_act = optax.apply_updates(p_act, u_act)
            return new_p_act, optax.incremental_update(new_p_act, t_act, self.tau_polyak), optax.incremental_update(p_c1, t_c1, self.tau_polyak), optax.incremental_update(p_c2, t_c2, self.tau_polyak), new_opt_act, l_act
            
        # NUEVO: Cast explícito de 0.0 a float32 para igualar a la rama verdadera
        p_act, t_act, t_c1, t_c2, opt_act, l_act = jax.lax.cond(
            update_actor, 
            actor_update, 
            lambda: (p_act, t_act, t_c1, t_c2, opt_act, jnp.array(0.0, dtype=jnp.float32))
        )
        return p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_act