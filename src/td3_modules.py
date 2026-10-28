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

def apply_spatial_gating(x, actor_output, l_radius=10.0):
    """
    Aplica una máscara suave para apagar el Actor cerca de los pozos de potencial.
    x: posiciones de los electrones. Shape (batch, n_elec) o (batch, n_elec, 1)
    actor_output: [mean, scale] del Actor. Shape (batch, 2)
    """
    if x.ndim == 3:
        x_1d = x[..., 0]
    else:
        x_1d = x
        
    # Calculamos el centro de masa para cada caminante (batch,)
    com = jnp.mean(x_1d, axis=-1, keepdims=True)
    
    # Calculamos la distancia al cuadrado aproximada de todos los electrones al centro de masa
    dist_sq = jnp.mean((x_1d - com)**2, axis=-1)
    eta = dist_sq / (l_radius ** 2)
    
    safe_eta = jnp.clip(eta, 0.50001, 0.99999) 
    transition = 1.0 / (1.0 + jnp.exp(-(4*safe_eta - 3) / (4*safe_eta**2 - 6*safe_eta + 2)))
    
    gamma = jnp.where(eta <= 0.5, 0.0,
              jnp.where(eta >= 1.0, 1.0, transition))
              
    gamma = gamma[..., jnp.newaxis] # Expandir para hacer broadcast (batch, 1)
    
    # Interpolar: Si gamma=0, drift=0 (MALA puro). Si gamma=1, drift=drift del actor
    # Solo aplicamos el gating al drift (índice 0), la escala de la varianza (índice 1) la dejamos intacta o apagada
    drift_gated = actor_output[..., 0:1] * (1.0 - gamma)
    scale = actor_output[..., 1:2]
    
    return jnp.concatenate([drift_gated, scale], axis=-1)

def build_td3_state(x, g, elec_idx, n_elec, n_up, n_dn, quantum_force_norm=5.0, particle_factor=10.0, dist_factor=5.0, omega=0.1, gs=0.5, softening=0.1):
    if x.ndim == 3: x = x[..., 0]
    if g.ndim == 3: g = g[..., 0]
    
    batch_size = x.shape[0]
    i = elec_idx
    x_i, g_i = x[:, i], g[:, i]
    
    spins = jnp.concatenate([jnp.ones(n_up), -jnp.ones(n_dn)])
    spin_i = spins[i]
    
    local_feat = jnp.stack([x_i, x_i - jnp.mean(x, axis=1), jnp.tanh(g_i / quantum_force_norm), jnp.full_like(x_i, spin_i)], axis=-1)
    
    global_feat = jnp.asarray([n_elec / particle_factor, n_up / max(n_elec, 1), jnp.tanh(omega), jnp.tanh(gs), jnp.tanh(softening)])
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
        return jnp.stack([jnp.where(mm, rr / 5.0, 0.0), jnp.where(mm, jnp.sign(dd), 0.0), mm.astype(jnp.float32)], axis=-1)
        
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
    sigma_action_scale: float = 0.2
    hidden_dim_local: int = 32
    hidden_dim_global: int = 16
    hidden_dim_messages: int = 32
    hidden_dim_actor: int = 64
    
    @nn.compact
    def __call__(self, state):
        max_same, max_opp = max(max(self.n_up - 1, 0), max(self.n_dn - 1, 0)), max(self.n_up, self.n_dn) if (self.n_up > 0 and self.n_dn > 0) else 0
        p = 0
        local_feat = state[..., p : p+4]; p += 4
        global_feat = state[..., p : p+5]; p += 5
        same_slots = state[..., p : p + 3*max_same].reshape(state.shape[:-1] + (max_same, 3)); p += 3*max_same
        opp_slots = state[..., p : p + 3*max_opp].reshape(state.shape[:-1] + (max_opp, 3))
        
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

class FlaxTD3Critic(nn.Module):

    hidden_dim_critic: int = 64

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        x = nn.silu(nn.Dense(self.hidden_dim_critic)(nn.silu(nn.Dense(self.hidden_dim_critic)(x))))
        return nn.Dense(1)(x)[..., 0]

class FlaxFlatActor(nn.Module):
    actor_amplitude: float

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(128)(state)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        drift = nn.Dense(1)(x)
        drift = jnp.tanh(drift)
        scale = nn.Dense(1)(x)
        return jnp.concatenate([drift, scale], axis=-1)

class FlaxFlatCritic(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        q1 = nn.Dense(128)(x)
        q1 = nn.relu(q1)
        q1 = nn.Dense(128)(q1)
        q1 = nn.relu(q1)
        q1 = nn.Dense(1)(q1)
        
        q2 = nn.Dense(128)(x)
        q2 = nn.relu(q2)
        q2 = nn.Dense(128)(q2)
        q2 = nn.relu(q2)
        q2 = nn.Dense(1)(q2)
        return q1, q2

# ==========================================
# 4. TD3 TRAINER SECUENCIAL
# ==========================================
# class with spatial gating in all electrons sampling

class TD3Trainer_Gating:
    def __init__(self, ansatz_apply, n_elec, n_up, n_dn, tau, actor_amp, 
                 beta=10.0, gamma_reward=0.1, gamma_td3=0.99, tau_polyak=0.005, 
                 target_noise=0.15, lambda_ = 1.0, noise_clip=0.35, epsilon=0.001, actor_type="full",
                 drift_limit=10., max_phys_jump_factor=5.0, action_rev_limit=1.0, sigma_action_scale=0.2):

        
        self.ansatz_apply = ansatz_apply
        self.n_elec, self.n_up, self.n_dn = n_elec, n_up, n_dn
        self.tau, self.actor_amp = tau, actor_amp
        self.beta, self.gamma_reward = beta, gamma_reward
        self.gamma_td3, self.tau_polyak = gamma_td3, tau_polyak
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.epsilon = epsilon
        self.actor_type = actor_type
        self.lambda_ = lambda_
        self.max_phys_jump_factor = max_phys_jump_factor
        self.drift_limit = drift_limit
        self.action_rev_limit = action_rev_limit
        self.sigma_action_scale = sigma_action_scale
        
        if self.actor_type == "flat":
            self.actor_model = FlaxFlatActor(actor_amplitude=actor_amp)
            self.critic_model = FlaxFlatCritic()
        else:
            self.actor_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp, sigma_action_scale=self.sigma_action_scale)
            self.critic_model = FlaxTD3Critic()

    @partial(jax.jit, static_argnums=(0,))
    def get_logp_and_grad(self, params_psi, walkers):
        def single_logp(w): return self.ansatz_apply(params_psi, w[None, ...])[0]
        return jax.vmap(jax.value_and_grad(single_logp))(walkers)

    @partial(jax.jit, static_argnums=(0,))
    def td3_update_all_electrons(self, key, walkers, curr_logp, curr_grad, params_psi, params_actor, exp_std):
        n_w = walkers.shape[0]
        k_act, k_prop, k_acc = jax.random.split(key, 3)
        
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = curr_grad[..., 0] if curr_grad.ndim == 3 else curr_grad
        
        # 1. Estado Global: Vectorizar sobre todos los electrones -> Shape: (N_elec, Batch, Dim)
        state_vmap = jax.vmap(build_td3_state, in_axes=(None, None, 0, None, None, None))
        state_old = state_vmap(w_1d, g_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn)
        state_old = jnp.transpose(state_old, (1, 0, 2)) # Transponer a (Batch, N_elec, Dim)
        
        # 2. Acción Global
        actor_vmap = jax.vmap(self.actor_model.apply, in_axes=(None, 1), out_axes=1)
        base_act = actor_vmap(params_actor, state_old)
        
        # === INYECCIÓN DE GATING ESPACIAL ===
        # Aplicamos el gating para cada electrón
        gating_vmap = jax.vmap(apply_spatial_gating, in_axes=(None, 1), out_axes=1)
        # Asumiendo un radio de 10.0 Bohrs. Puedes ajustarlo.
        base_act = gating_vmap(w_1d, base_act) 
        # ====================================

        aux_noise = exp_std * jax.random.normal(k_act, base_act.shape)
        action_old = jnp.clip(base_act + aux_noise, -1.0, 1.0)
        
        # 3. Propuesta Híbrida Simultánea
        base_step = jnp.sqrt(2.0 * self.tau)
        sigma_fwd = jnp.maximum(base_step * jnp.exp(action_old[..., 1]), 1e-4)
        drift_fwd = jnp.clip(self.tau * curr_grad[..., 0], -self.drift_limit, self.drift_limit)
        residual_fwd = action_old[..., 0] * self.actor_amp
        
        old_coord = walkers[..., 0]
        mean_fwd_dist = old_coord + drift_fwd + residual_fwd
        proposed_coord = mean_fwd_dist + sigma_fwd * jax.random.normal(k_prop, old_coord.shape)
        
        # Inyectar dimensión espacial
        proposal = proposed_coord[..., None] if walkers.ndim == 3 else proposed_coord
        new_logp, new_grad = self.get_logp_and_grad(params_psi, proposal)
        
        p_1d = proposal[..., 0] if proposal.ndim == 3 else proposal
        ng_1d = new_grad[..., 0] if new_grad.ndim == 3 else new_grad

        # 4. Evaluación Inversa Simultánea
        state_prop = jnp.transpose(state_vmap(p_1d, ng_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn), (1, 0, 2))
        
        # === INYECCIÓN DE GATING ESPACIAL INVERSO ===
        base_act_rev = actor_vmap(params_actor, state_prop)
        base_act_rev = gating_vmap(p_1d, base_act_rev)
        # ============================================

        action_rev = jnp.clip(base_act_rev + aux_noise, -self.action_rev_limit, self.action_rev_limit)
        
        sigma_rev = jnp.maximum(base_step * jnp.exp(action_rev[..., 1]), 1e-4)
        drift_rev = jnp.clip(self.tau * new_grad[..., 0], -self.drift_limit, self.drift_limit)
        residual_rev = action_rev[..., 0] * self.actor_amp
        mean_rev_dist = proposed_coord + drift_rev + residual_rev
        
        # 5. Kernel Asimétrico Global
        log_q_fwd = jnp.sum(-jnp.log(sigma_fwd) - 0.5 * ((proposed_coord - mean_fwd_dist) / sigma_fwd)**2, axis=-1)
        log_q_rev = jnp.sum(-jnp.log(sigma_rev) - 0.5 * ((old_coord - mean_rev_dist) / sigma_rev)**2, axis=-1)
        
        clipped_log_alpha = jnp.minimum(0.0, 2.0 * (new_logp - curr_logp) + (log_q_rev - log_q_fwd))
        accept = jnp.log(jax.random.uniform(k_acc, (n_w,), minval=1e-15)) < clipped_log_alpha
        
        next_w = jnp.where(accept[:, None, None], proposal, walkers)
        next_logp = jnp.where(accept, new_logp, curr_logp)
        next_grad = jnp.where(accept[:, None, None], new_grad, curr_grad)
        
        # 6. Recompensa (Ajustada al cálculo CDLB ya corregido)
        disp = jnp.mean(jnp.abs(proposed_coord - old_coord), axis=-1)
        alpha_exact = jnp.exp(clipped_log_alpha)
        
        log_p_diff = 2.0 * (new_logp - curr_logp)
        exploitation = alpha_exact * log_p_diff
        entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)
        exploration = -alpha_exact * log_q_fwd
        
        reward = self.lambda_ * (exploitation + entropy_alpha + exploration)
        
        nw_1d = next_w[..., 0] if next_w.ndim == 3 else next_w
        nwg_1d = next_grad[..., 0] if next_grad.ndim == 3 else next_grad
        next_state = jnp.transpose(state_vmap(nw_1d, nwg_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn), (1, 0, 2))
        
        return next_w, next_logp, next_grad, state_old, action_old, reward, exploitation, entropy_alpha, exploration, next_state, accept, disp

    @partial(jax.jit, static_argnums=(0, 7))
    def td3_update_one_electron(self, key, walkers, curr_logp, curr_grad, params_psi, params_actor, elec_idx, exp_std):
        n_w = walkers.shape[0]
        k_act, k_prop, k_acc = jax.random.split(key, 3)
        
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = curr_grad[..., 0] if curr_grad.ndim == 3 else curr_grad
        
        # Estado Inicial: Condicionado a Ablación
        if self.actor_type == "flat":
            target_w = jax.lax.dynamic_slice_in_dim(w_1d, elec_idx, 1, axis=-1)
            target_g = jax.lax.dynamic_slice_in_dim(g_1d, elec_idx, 1, axis=-1)
            state_old = jnp.concatenate([target_w, target_g, w_1d, g_1d], axis=-1)
        else:
            state_old = build_td3_state(w_1d, g_1d, elec_idx, self.n_elec, self.n_up, self.n_dn)
        
        base_act = self.actor_model.apply(params_actor, state_old)
        aux_noise = exp_std * jax.random.normal(k_act, base_act.shape)
        action_old = jnp.clip(base_act + aux_noise, -1.0, 1.0)
        
        base_step = jnp.sqrt(2.0 * self.tau)
        sigma_fwd = jnp.maximum(base_step * jnp.exp(action_old[..., 1]), 1e-4)
        
        drift_fwd = jnp.clip(self.tau * curr_grad[:, elec_idx, 0], -self.drift_limit, self.drift_limit)
        residual_fwd = action_old[..., 0] * self.actor_amp
        
        old_coord = walkers[:, elec_idx, 0] 
        mean_fwd_dist = old_coord + drift_fwd + residual_fwd
        proposed_coord = mean_fwd_dist + sigma_fwd * jax.random.normal(k_prop, (n_w,))
        
        proposal = walkers.at[:, elec_idx, 0].set(proposed_coord)
        new_logp, new_grad = self.get_logp_and_grad(params_psi, proposal)
        
        p_1d = proposal[..., 0] if proposal.ndim == 3 else proposal
        ng_1d = new_grad[..., 0] if new_grad.ndim == 3 else new_grad

        # Estado Propuesto: Condicionado a Ablación
        if self.actor_type == "flat":
            target_p = jax.lax.dynamic_slice_in_dim(p_1d, elec_idx, 1, axis=-1)
            target_ng = jax.lax.dynamic_slice_in_dim(ng_1d, elec_idx, 1, axis=-1)
            state_prop = jnp.concatenate([target_p, target_ng, p_1d, ng_1d], axis=-1)
        else:
            state_prop = build_td3_state(p_1d, ng_1d, elec_idx, self.n_elec, self.n_up, self.n_dn)

        action_rev = jnp.clip(self.actor_model.apply(params_actor, state_prop) + aux_noise, -self.action_rev_limit, self.action_rev_limit)
        
        sigma_rev = jnp.maximum(base_step * jnp.exp(action_rev[..., 1]), 1e-4)
        drift_rev = jnp.clip(self.tau * new_grad[:, elec_idx, 0], -self.drift_limit, self.drift_limit)
        residual_rev = action_rev[..., 0] * self.actor_amp
        mean_rev_dist = proposed_coord + drift_rev + residual_rev
        
        log_q_fwd = -jnp.log(sigma_fwd) - 0.5 * ((proposed_coord - mean_fwd_dist) / sigma_fwd)**2
        log_q_rev = -jnp.log(sigma_rev) - 0.5 * ((old_coord - mean_rev_dist) / sigma_rev)**2
        
        clipped_log_alpha = jnp.minimum(0.0, 2.0 * (new_logp - curr_logp) + (log_q_rev - log_q_fwd))
        accept = jnp.log(jax.random.uniform(k_acc, (n_w,), minval=1e-15)) < clipped_log_alpha
        
        next_w = jnp.where(accept[:, None, None], proposal, walkers)
        next_logp = jnp.where(accept, new_logp, curr_logp)
        next_grad = jnp.where(accept[:, None, None], new_grad, curr_grad)
        
        disp = jnp.abs(proposed_coord - old_coord)
        max_phys_jump = self.max_phys_jump_factor * jnp.sqrt(self.tau)
        clipped_dist = jnp.clip(disp, a_min=0.0, a_max=max_phys_jump)
        alpha_exact = jnp.exp(clipped_log_alpha)
        
        # RECOMPENSA CDLB (Wang et al. 2026, Eq. 11)
        # log_p_diff = 2.0 * (new_logp - curr_logp)
        # exploitation = alpha_exact * log_p_diff
        # entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)
        # exploration = -alpha_exact * log_q_fwd
        # reward = (exploitation + entropy_alpha + exploration) * 0.1

        # reward = jnp.log(self.epsilon + alpha_exact * (clipped_dist ** 2))

        # =================================================================
        # RECOMPENSA CDLB (Wang et al. 2026, Eq. 11)
        # =================================================================
        # A. Explotación: 2 * (log|Psi(X*)| - log|Psi(X)|)
        log_p_diff = 2.0 * (new_logp - curr_logp)
        exploitation = alpha_exact * log_p_diff

        # B. Entropía de Aceptación (Añadiendo 1e-8 para evitar NaN en log(0))
        entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)

        # C. Exploración: -alpha * log q(X* | X)
        # En RMALA, la densidad de propuesta ya está en la variable log_q_fwd
        exploration = -alpha_exact * log_q_fwd

        # # Recompensa combinada. Se multiplica por 0.1 para que el MSE del Crítico no explote
        reward = (exploitation + entropy_alpha + exploration) * self.lambda_
        # # =================================================================

        # Recompensa Equilibrada: Fomenta saltos físicos largos, pero castiga si la aceptación cae de un umbral
        #dist_sq = jnp.clip(disp**2, 0.0, max_phys_jump**2)
        #reward = alpha_exact * dist_sq - self.penalty_factor * (1.0 - alpha_exact)**2

        # =================================================================
        # 3. Nueva recompensa idéntica a REINFORCE (ignoramos soft_node_penalty por rendimiento)
        #kappa = 0.5 # Mismo valor que args.kappa
        # # Mi recompensa:
        #reward = (alpha_exact * clipped_dist) + kappa * (1.0 - alpha_exact) * alpha_exact
        # =================================================================

        # =================================================================
        
        # Recompensa de Huziel
        # 
        #reward = jnp.log(self.epsilon + alpha_exact * clipped_dist ** 2)

        # =================================================================


        next_idx = (elec_idx + 1) % self.n_elec
        nw_1d = next_w[..., 0] if next_w.ndim == 3 else next_w
        nwg_1d = next_grad[..., 0] if next_grad.ndim == 3 else next_grad

        # Siguiente Estado: Condicionado a Ablación
        if self.actor_type == "flat":
            target_nw = jax.lax.dynamic_slice_in_dim(nw_1d, next_idx, 1, axis=-1)
            target_nwg = jax.lax.dynamic_slice_in_dim(nwg_1d, next_idx, 1, axis=-1)
            next_state = jnp.concatenate([target_nw, target_nwg, nw_1d, nwg_1d], axis=-1)
        else:
            next_state = build_td3_state(nw_1d, nwg_1d, next_idx, self.n_elec, self.n_up, self.n_dn)
        
        return next_w, next_logp, next_grad, state_old, action_old, reward, next_state, accept, disp

    @partial(jax.jit, static_argnums=(0,))
    def train_networks(self, key, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_actor):
        noise = jnp.clip(self.target_noise * jax.random.normal(key, batch["action"].shape), -self.noise_clip, self.noise_clip)
        next_action = jnp.clip(self.actor_model.apply(t_act, batch["next_state"]) + noise, -1.0, 1.0)
        
        if self.actor_type == "flat":
            q1_next, _ = self.critic_model.apply(t_c1, batch["next_state"], next_action)
            _, q2_next = self.critic_model.apply(t_c2, batch["next_state"], next_action)
            q_target = jnp.minimum(q1_next, q2_next)
        else:
            q_target = jnp.minimum(
                self.critic_model.apply(t_c1, batch["next_state"], next_action),
                self.critic_model.apply(t_c2, batch["next_state"], next_action)
            )

        y = jax.lax.stop_gradient(batch["reward"] + self.gamma_td3 * q_target)
        
        def critic_loss(p_crit, is_q1): 
            if self.actor_type == "flat":
                q1, q2 = self.critic_model.apply(p_crit, batch["state"], batch["action"])
                q = q1 if is_q1 else q2
                return jnp.mean((q - y)**2)
            else:
                return jnp.mean((self.critic_model.apply(p_crit, batch["state"], batch["action"]) - y)**2)
        
        l1, g1 = jax.value_and_grad(critic_loss)(p_c1, True)
        l2, g2 = jax.value_and_grad(critic_loss)(p_c2, False)
        
        u1, opt_c1 = self.opt_c.update(g1, opt_c1, p_c1); p_c1 = optax.apply_updates(p_c1, u1)
        u2, opt_c2 = self.opt_c.update(g2, opt_c2, p_c2); p_c2 = optax.apply_updates(p_c2, u2)
        
        def actor_update():
            def actor_loss(p_a): 
                act = self.actor_model.apply(p_a, batch["state"])
                if self.actor_type == "flat":
                    q1, _ = self.critic_model.apply(p_c1, batch["state"], act)
                    return -jnp.mean(q1)
                else:
                    return -jnp.mean(self.critic_model.apply(p_c1, batch["state"], act))
            
            l_act, g_act = jax.value_and_grad(actor_loss)(p_act)
            u_act, new_opt_act = self.opt_a.update(g_act, opt_act, p_act)
            new_p_act = optax.apply_updates(p_act, u_act)
            return new_p_act, optax.incremental_update(new_p_act, t_act, self.tau_polyak), optax.incremental_update(p_c1, t_c1, self.tau_polyak), optax.incremental_update(p_c2, t_c2, self.tau_polyak), new_opt_act, l_act
            
        p_act, t_act, t_c1, t_c2, opt_act, l_act = jax.lax.cond(update_actor, actor_update, lambda: (p_act, t_act, t_c1, t_c2, opt_act, jnp.array(0.0, dtype=jnp.float32)))
        return p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_act


# class without the gating
class TD3Trainer:
    def __init__(self, ansatz_apply, n_elec, n_up, n_dn, tau, actor_amp, 
                 beta=10.0, gamma_reward=0.1, gamma_td3=0.99, tau_polyak=0.005, 
                 target_noise=0.15, lambda_ = 1.0, noise_clip=0.35, epsilon=0.001, actor_type="full",
                 drift_limit=10., max_phys_jump_factor=5.0, action_rev_limit=1.0, sigma_action_scale=0.2):

        
        self.ansatz_apply = ansatz_apply
        self.n_elec, self.n_up, self.n_dn = n_elec, n_up, n_dn
        self.tau, self.actor_amp = tau, actor_amp
        self.beta, self.gamma_reward = beta, gamma_reward
        self.gamma_td3, self.tau_polyak = gamma_td3, tau_polyak
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.epsilon = epsilon
        self.actor_type = actor_type
        self.lambda_ = lambda_
        self.max_phys_jump_factor = max_phys_jump_factor
        self.drift_limit = drift_limit
        self.action_rev_limit = action_rev_limit
        self.sigma_action_scale = sigma_action_scale
        
        if self.actor_type == "flat":
            self.actor_model = FlaxFlatActor(actor_amplitude=actor_amp)
            self.critic_model = FlaxFlatCritic()
        else:
            self.actor_model = FlaxTD3Actor(n_up=n_up, n_dn=n_dn, actor_amplitude=actor_amp, sigma_action_scale=self.sigma_action_scale)
            self.critic_model = FlaxTD3Critic()

    @partial(jax.jit, static_argnums=(0,))
    def get_logp_and_grad(self, params_psi, walkers):
        def single_logp(w): return self.ansatz_apply(params_psi, w[None, ...])[0]
        return jax.vmap(jax.value_and_grad(single_logp))(walkers)

    @partial(jax.jit, static_argnums=(0,))
    def td3_update_all_electrons(self, key, walkers, curr_logp, curr_grad, params_psi, params_actor, exp_std):
        n_w = walkers.shape[0]
        k_act, k_prop, k_acc = jax.random.split(key, 3)
        
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = curr_grad[..., 0] if curr_grad.ndim == 3 else curr_grad
        
        # 1. Estado Global: Vectorizar sobre todos los electrones -> Shape: (N_elec, Batch, Dim)
        state_vmap = jax.vmap(build_td3_state, in_axes=(None, None, 0, None, None, None))
        state_old = state_vmap(w_1d, g_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn)
        state_old = jnp.transpose(state_old, (1, 0, 2)) # Transponer a (Batch, N_elec, Dim)
        
        # 2. Acción Global
        actor_vmap = jax.vmap(self.actor_model.apply, in_axes=(None, 1), out_axes=1)
        base_act = actor_vmap(params_actor, state_old)
        aux_noise = exp_std * jax.random.normal(k_act, base_act.shape)
        action_old = jnp.clip(base_act + aux_noise, -1.0, 1.0)
        
        # 3. Propuesta Híbrida Simultánea
        base_step = jnp.sqrt(2.0 * self.tau)
        sigma_fwd = jnp.maximum(base_step * jnp.exp(action_old[..., 1]), 1e-4)
        drift_fwd = jnp.clip(self.tau * curr_grad[..., 0], -self.drift_limit, self.drift_limit)
        residual_fwd = action_old[..., 0] * self.actor_amp
        
        old_coord = walkers[..., 0]
        mean_fwd_dist = old_coord + drift_fwd + residual_fwd
        proposed_coord = mean_fwd_dist + sigma_fwd * jax.random.normal(k_prop, old_coord.shape)
        
        # Inyectar dimensión espacial
        proposal = proposed_coord[..., None] if walkers.ndim == 3 else proposed_coord
        new_logp, new_grad = self.get_logp_and_grad(params_psi, proposal)
        
        p_1d = proposal[..., 0] if proposal.ndim == 3 else proposal
        ng_1d = new_grad[..., 0] if new_grad.ndim == 3 else new_grad

        # 4. Evaluación Inversa Simultánea
        state_prop = jnp.transpose(state_vmap(p_1d, ng_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn), (1, 0, 2))
        action_rev = jnp.clip(actor_vmap(params_actor, state_prop) + aux_noise, -self.action_rev_limit, self.action_rev_limit)
        
        sigma_rev = jnp.maximum(base_step * jnp.exp(action_rev[..., 1]), 1e-4)
        drift_rev = jnp.clip(self.tau * new_grad[..., 0], -self.drift_limit, self.drift_limit)
        residual_rev = action_rev[..., 0] * self.actor_amp
        mean_rev_dist = proposed_coord + drift_rev + residual_rev
        
        # 5. Kernel Asimétrico Global
        log_q_fwd = jnp.sum(-jnp.log(sigma_fwd) - 0.5 * ((proposed_coord - mean_fwd_dist) / sigma_fwd)**2, axis=-1)
        log_q_rev = jnp.sum(-jnp.log(sigma_rev) - 0.5 * ((old_coord - mean_rev_dist) / sigma_rev)**2, axis=-1)
        
        clipped_log_alpha = jnp.minimum(0.0, 2.0 * (new_logp - curr_logp) + (log_q_rev - log_q_fwd))
        accept = jnp.log(jax.random.uniform(k_acc, (n_w,), minval=1e-15)) < clipped_log_alpha
        
        next_w = jnp.where(accept[:, None, None], proposal, walkers)
        next_logp = jnp.where(accept, new_logp, curr_logp)
        next_grad = jnp.where(accept[:, None, None], new_grad, curr_grad)
        
        # 6. Recompensa (Ajustada al cálculo CDLB ya corregido)
        disp = jnp.mean(jnp.abs(proposed_coord - old_coord), axis=-1)
        alpha_exact = jnp.exp(clipped_log_alpha)
        
        log_p_diff = 2.0 * (new_logp - curr_logp)
        exploitation = alpha_exact * log_p_diff
        entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)
        exploration = -alpha_exact * log_q_fwd
        
        reward = self.lambda_ * (exploitation + entropy_alpha + exploration)
        
        nw_1d = next_w[..., 0] if next_w.ndim == 3 else next_w
        nwg_1d = next_grad[..., 0] if next_grad.ndim == 3 else next_grad
        next_state = jnp.transpose(state_vmap(nw_1d, nwg_1d, jnp.arange(self.n_elec), self.n_elec, self.n_up, self.n_dn), (1, 0, 2))
        
        return next_w, next_logp, next_grad, state_old, action_old, reward, exploitation, entropy_alpha, exploration, next_state, accept, disp

    @partial(jax.jit, static_argnums=(0, 7))
    def td3_update_one_electron(self, key, walkers, curr_logp, curr_grad, params_psi, params_actor, elec_idx, exp_std):
        n_w = walkers.shape[0]
        k_act, k_prop, k_acc = jax.random.split(key, 3)
        
        w_1d = walkers[..., 0] if walkers.ndim == 3 else walkers
        g_1d = curr_grad[..., 0] if curr_grad.ndim == 3 else curr_grad
        
        # Estado Inicial: Condicionado a Ablación
        if self.actor_type == "flat":
            target_w = jax.lax.dynamic_slice_in_dim(w_1d, elec_idx, 1, axis=-1)
            target_g = jax.lax.dynamic_slice_in_dim(g_1d, elec_idx, 1, axis=-1)
            state_old = jnp.concatenate([target_w, target_g, w_1d, g_1d], axis=-1)
        else:
            state_old = build_td3_state(w_1d, g_1d, elec_idx, self.n_elec, self.n_up, self.n_dn)
        
        base_act = self.actor_model.apply(params_actor, state_old)
        aux_noise = exp_std * jax.random.normal(k_act, base_act.shape)
        action_old = jnp.clip(base_act + aux_noise, -1.0, 1.0)
        
        base_step = jnp.sqrt(2.0 * self.tau)
        sigma_fwd = jnp.maximum(base_step * jnp.exp(action_old[..., 1]), 1e-4)
        
        drift_fwd = jnp.clip(self.tau * curr_grad[:, elec_idx, 0], -self.drift_limit, self.drift_limit)
        residual_fwd = action_old[..., 0] * self.actor_amp
        
        old_coord = walkers[:, elec_idx, 0] 
        mean_fwd_dist = old_coord + drift_fwd + residual_fwd
        proposed_coord = mean_fwd_dist + sigma_fwd * jax.random.normal(k_prop, (n_w,))
        
        proposal = walkers.at[:, elec_idx, 0].set(proposed_coord)
        new_logp, new_grad = self.get_logp_and_grad(params_psi, proposal)
        
        p_1d = proposal[..., 0] if proposal.ndim == 3 else proposal
        ng_1d = new_grad[..., 0] if new_grad.ndim == 3 else new_grad

        # Estado Propuesto: Condicionado a Ablación
        if self.actor_type == "flat":
            target_p = jax.lax.dynamic_slice_in_dim(p_1d, elec_idx, 1, axis=-1)
            target_ng = jax.lax.dynamic_slice_in_dim(ng_1d, elec_idx, 1, axis=-1)
            state_prop = jnp.concatenate([target_p, target_ng, p_1d, ng_1d], axis=-1)
        else:
            state_prop = build_td3_state(p_1d, ng_1d, elec_idx, self.n_elec, self.n_up, self.n_dn)

        action_rev = jnp.clip(self.actor_model.apply(params_actor, state_prop) + aux_noise, -self.action_rev_limit, self.action_rev_limit)
        
        sigma_rev = jnp.maximum(base_step * jnp.exp(action_rev[..., 1]), 1e-4)
        drift_rev = jnp.clip(self.tau * new_grad[:, elec_idx, 0], -self.drift_limit, self.drift_limit)
        residual_rev = action_rev[..., 0] * self.actor_amp
        mean_rev_dist = proposed_coord + drift_rev + residual_rev
        
        log_q_fwd = -jnp.log(sigma_fwd) - 0.5 * ((proposed_coord - mean_fwd_dist) / sigma_fwd)**2
        log_q_rev = -jnp.log(sigma_rev) - 0.5 * ((old_coord - mean_rev_dist) / sigma_rev)**2
        
        clipped_log_alpha = jnp.minimum(0.0, 2.0 * (new_logp - curr_logp) + (log_q_rev - log_q_fwd))
        accept = jnp.log(jax.random.uniform(k_acc, (n_w,), minval=1e-15)) < clipped_log_alpha
        
        next_w = jnp.where(accept[:, None, None], proposal, walkers)
        next_logp = jnp.where(accept, new_logp, curr_logp)
        next_grad = jnp.where(accept[:, None, None], new_grad, curr_grad)
        
        disp = jnp.abs(proposed_coord - old_coord)
        max_phys_jump = self.max_phys_jump_factor * jnp.sqrt(self.tau)
        clipped_dist = jnp.clip(disp, a_min=0.0, a_max=max_phys_jump)
        alpha_exact = jnp.exp(clipped_log_alpha)
        
        # RECOMPENSA CDLB (Wang et al. 2026, Eq. 11)
        # log_p_diff = 2.0 * (new_logp - curr_logp)
        # exploitation = alpha_exact * log_p_diff
        # entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)
        # exploration = -alpha_exact * log_q_fwd
        # reward = (exploitation + entropy_alpha + exploration) * 0.1

        # reward = jnp.log(self.epsilon + alpha_exact * (clipped_dist ** 2))

        # =================================================================
        # RECOMPENSA CDLB (Wang et al. 2026, Eq. 11)
        # =================================================================
        # A. Explotación: 2 * (log|Psi(X*)| - log|Psi(X)|)
        log_p_diff = 2.0 * (new_logp - curr_logp)
        exploitation = alpha_exact * log_p_diff

        # B. Entropía de Aceptación (Añadiendo 1e-8 para evitar NaN en log(0))
        entropy_alpha = -alpha_exact * jnp.log(alpha_exact + 1e-8) - (1.0 - alpha_exact) * jnp.log(1.0 - alpha_exact + 1e-8)

        # C. Exploración: -alpha * log q(X* | X)
        # En RMALA, la densidad de propuesta ya está en la variable log_q_fwd
        exploration = -alpha_exact * log_q_fwd

        # # Recompensa combinada. Se multiplica por 0.1 para que el MSE del Crítico no explote
        reward = (exploitation + entropy_alpha + exploration) * self.lambda_
        # # =================================================================

        # Recompensa Equilibrada: Fomenta saltos físicos largos, pero castiga si la aceptación cae de un umbral
        #dist_sq = jnp.clip(disp**2, 0.0, max_phys_jump**2)
        #reward = alpha_exact * dist_sq - self.penalty_factor * (1.0 - alpha_exact)**2

        # =================================================================
        # 3. Nueva recompensa idéntica a REINFORCE (ignoramos soft_node_penalty por rendimiento)
        #kappa = 0.5 # Mismo valor que args.kappa
        # # Mi recompensa:
        #reward = (alpha_exact * clipped_dist) + kappa * (1.0 - alpha_exact) * alpha_exact
        # =================================================================

        # =================================================================
        
        # Recompensa de Huziel
        # 
        #reward = jnp.log(self.epsilon + alpha_exact * clipped_dist ** 2)

        # =================================================================


        next_idx = (elec_idx + 1) % self.n_elec
        nw_1d = next_w[..., 0] if next_w.ndim == 3 else next_w
        nwg_1d = next_grad[..., 0] if next_grad.ndim == 3 else next_grad

        # Siguiente Estado: Condicionado a Ablación
        if self.actor_type == "flat":
            target_nw = jax.lax.dynamic_slice_in_dim(nw_1d, next_idx, 1, axis=-1)
            target_nwg = jax.lax.dynamic_slice_in_dim(nwg_1d, next_idx, 1, axis=-1)
            next_state = jnp.concatenate([target_nw, target_nwg, nw_1d, nwg_1d], axis=-1)
        else:
            next_state = build_td3_state(nw_1d, nwg_1d, next_idx, self.n_elec, self.n_up, self.n_dn)
        
        return next_w, next_logp, next_grad, state_old, action_old, reward, next_state, accept, disp

    @partial(jax.jit, static_argnums=(0,))
    def train_networks(self, key, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_actor):
        noise = jnp.clip(self.target_noise * jax.random.normal(key, batch["action"].shape), -self.noise_clip, self.noise_clip)
        next_action = jnp.clip(self.actor_model.apply(t_act, batch["next_state"]) + noise, -1.0, 1.0)
        
        if self.actor_type == "flat":
            q1_next, _ = self.critic_model.apply(t_c1, batch["next_state"], next_action)
            _, q2_next = self.critic_model.apply(t_c2, batch["next_state"], next_action)
            q_target = jnp.minimum(q1_next, q2_next)
        else:
            q_target = jnp.minimum(
                self.critic_model.apply(t_c1, batch["next_state"], next_action),
                self.critic_model.apply(t_c2, batch["next_state"], next_action)
            )

        y = jax.lax.stop_gradient(batch["reward"] + self.gamma_td3 * q_target)
        
        def critic_loss(p_crit, is_q1): 
            if self.actor_type == "flat":
                q1, q2 = self.critic_model.apply(p_crit, batch["state"], batch["action"])
                q = q1 if is_q1 else q2
                return jnp.mean((q - y)**2)
            else:
                return jnp.mean((self.critic_model.apply(p_crit, batch["state"], batch["action"]) - y)**2)
        
        l1, g1 = jax.value_and_grad(critic_loss)(p_c1, True)
        l2, g2 = jax.value_and_grad(critic_loss)(p_c2, False)
        
        u1, opt_c1 = self.opt_c.update(g1, opt_c1, p_c1); p_c1 = optax.apply_updates(p_c1, u1)
        u2, opt_c2 = self.opt_c.update(g2, opt_c2, p_c2); p_c2 = optax.apply_updates(p_c2, u2)
        
        def actor_update():
            def actor_loss(p_a): 
                act = self.actor_model.apply(p_a, batch["state"])
                if self.actor_type == "flat":
                    q1, _ = self.critic_model.apply(p_c1, batch["state"], act)
                    return -jnp.mean(q1)
                else:
                    return -jnp.mean(self.critic_model.apply(p_c1, batch["state"], act))
            
            l_act, g_act = jax.value_and_grad(actor_loss)(p_act)
            u_act, new_opt_act = self.opt_a.update(g_act, opt_act, p_act)
            new_p_act = optax.apply_updates(p_act, u_act)
            return new_p_act, optax.incremental_update(new_p_act, t_act, self.tau_polyak), optax.incremental_update(p_c1, t_c1, self.tau_polyak), optax.incremental_update(p_c2, t_c2, self.tau_polyak), new_opt_act, l_act
            
        p_act, t_act, t_c1, t_c2, opt_act, l_act = jax.lax.cond(update_actor, actor_update, lambda: (p_act, t_act, t_c1, t_c2, opt_act, jnp.array(0.0, dtype=jnp.float32)))
        return p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_act