import jax
import jax.numpy as jnp
import optax
from abc import ABC, abstractmethod
from functools import partial

class BaseTrainer(ABC):
    """
    Base class for trainers. Save data from the simulation
    """
    def __init__(self, system, hamiltonian, sampler, opt_ansatz, opt_actor=None):
        self.system = system
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.opt_ansatz = opt_ansatz
        self.opt_actor = opt_actor

    @abstractmethod
    def step_ansatz(self, walkers, params_psi, opt_state_psi):
        pass

class GroundStateTrainer(BaseTrainer):
    """
    Trainer specialized in finding the truth ground state through energy minimization
    """

    def __init__(self, system, hamiltonian, sampler, opt_ansatz, opt_actor=None, max_phys_jump_std=3.0, sigma_energy_clip=5.0, energy_clip_min = -50., energy_clip_max = 0., min_clip_grads_ansatz = -20., max_clip_grads_ansatz = 20.):
        super().__init__(system, hamiltonian, sampler, opt_ansatz, opt_actor)
        self.energy_clip_min = energy_clip_min
        self.energy_clip_max = energy_clip_max
        self.sigma_energy_clip = sigma_energy_clip
        self.min_clip_grads_ansatz = min_clip_grads_ansatz
        self.max_clip_grads_ansatz = max_clip_grads_ansatz
        self.max_phys_jump_std = max_phys_jump_std

        print("rewards = beta * term_dist - gamma * term_penalty - penalty_factor * penalty_for_reject")

    @partial(jax.jit, static_argnums=(0,))
    def step_ansatz(self, walkers, params_psi, opt_state_psi):
        """
        computes the energy gradients and updates steps of the ansatz
        """
        def energy_loss(p_psi):
            # compute local energy
            el_loc = self.hamiltonian.local_energy(self.sampler.ansatz_apply, p_psi, walkers)

            # energy hard clipping
            el_hard_clip = jnp.clip(jnp.nan_to_num(el_loc, nan=0.0), self.energy_clip_min, self.energy_clip_max)
            median = jnp.median(el_hard_clip)
            mad = jnp.mean(jnp.abs(el_hard_clip - median))
            clip_low = median - self.sigma_energy_clip * mad
            clip_high = median + self.sigma_energy_clip * mad
            el_clipped = jnp.clip(el_hard_clip, clip_low, clip_high)

            # log derivative
            log_psi = self.sampler.ansatz_apply(p_psi, walkers)

            # VLC loss function
            baseline = jnp.mean(el_clipped)
            diff_energy = jax.lax.stop_gradient(el_clipped - baseline)
            loss = 2.0 * jnp.mean(diff_energy * log_psi)

            return loss, (jnp.mean(el_clipped), jnp.var(el_clipped))
        
        # autodiff
        (loss_val, (mean_E, var_E)), grads = jax.value_and_grad(energy_loss, has_aux = True)(params_psi)

        # get rid of nans
        grads_safe = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grads)

        # optax update
        updates, new_opt_state = self.opt_ansatz.update(grads_safe, opt_state_psi, params_psi)
        new_params_psi = optax.apply_updates(params_psi, updates)

        # update cache
        def sum_log_psi_new(w):
            return jnp.sum(self.sampler.ansatz_apply(new_params_psi, w))
        
        new_log_psi = self.sampler.ansatz_apply(new_params_psi, walkers)
        new_grad = jnp.clip(jax.grad(sum_log_psi_new)(walkers), self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)

        updated_cache = {"log_psi_curr": new_log_psi, "grad_curr": new_grad}

        return new_params_psi, new_opt_state, mean_E, var_E, updated_cache
    
    @partial(jax.jit, static_argnums=(0,))
    def step_actor(self, key, walkers, params_psi, params_actor, opt_state_actor, curr_cache):
        """
        Actualiza el Actor neuronal usando RL (REINFORCE) evaluando la recompensa del salto.
        """
        k_s, _ = jax.random.split(key)

        def loss_fn(p_actor, w_current, k, cache):
            # El Actor propone, Metrópolis filtra, y obtenemos las recompensas crudas (rl_data)
            w_new, stats, new_cache, prop_aux = self.sampler.step(
                k, w_current, {"ansatz": params_psi, "actor": p_actor}, cache, exploration_mode=True
            )
            
            # Desempaquetamos datos pasados por el Sampler
            rl_data = prop_aux["rl_data"]
            log_q_fwd_policy = rl_data["log_q_fwd_policy"]
            proposal_dist = rl_data["proposal_dist"]
            raw_penalty = rl_data["raw_penalty"]
            gamma = rl_data["gamma"]
            beta = rl_data["beta"]
            penalty_factor = rl_data["penalty_factor"]
            
            # Reconstruimos la recompensa (ahora tenemos alpha exacto gracias al log_ratio del step)
            safe_log_ratio = stats["safe_log_ratio_vector"] 
            alpha_exact = jnp.exp(safe_log_ratio - jax.nn.softplus(safe_log_ratio))
            
            grad_penalty = jnp.clip(raw_penalty, 0.0, 50.0)
            penalty_for_reject = (1.0 - alpha_exact)

            max_physical_jump = self.max_phys_jump_std * jnp.sqrt(self.sampler.tau)
            
            term_dist =  jnp.clip(proposal_dist, a_min=0.0, a_max=max_physical_jump) * alpha_exact
            term_penalty = grad_penalty
            
            rewards = beta * term_dist - gamma * term_penalty - penalty_factor * penalty_for_reject
            
            # Cálculo del Advantage para estabilizar REINFORCE
            mean_r = jnp.mean(rewards)
            std_r = jnp.std(rewards) + 1e-8
            advantage = (rewards - mean_r) / std_r
            
            # Loss Function: Policy Gradient (Maximizamos ventaja = Minimizamos -ventaja * prob)
            pg_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * log_q_fwd_policy)
            
            # Guardamos datos extra en stats
            stats["mean_reward"] = mean_r
            stats["reward_dist"] = jnp.mean(term_dist)
            stats["reward_penalty"] = jnp.mean(term_penalty)
            stats["penalty_for_reject"] = jnp.mean(penalty_for_reject)
            
            return pg_loss, (w_new, stats, new_cache)

        # Calculamos gradientes de la pérdida RL respecto a los pesos del Actor
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_actor, walkers, k_s, curr_cache)
        w_new, stats, new_cache = aux
        
        # Norma del gradiente para monitoreo de estabilidad
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        stats["actor_grad_norm"] = grad_norm 
        
        # Actualización Optax
        updates, new_opt_state = self.opt_actor.update(grads, opt_state_actor, params_actor)
        new_params_actor = optax.apply_updates(params_actor, updates)
        
        return new_params_actor, new_opt_state, w_new, stats, loss_val, new_cache
    

class GroundStateTrainer_test(BaseTrainer):
    """
    Trainer specialized in finding the truth ground state through energy minimization
    """

    def __init__(self, system, hamiltonian, sampler, opt_ansatz, opt_actor=None, max_phys_jump_std=3.0, sigma_energy_clip=5.0, energy_clip_min = -50., energy_clip_max = 0., min_clip_grads_ansatz = -20., max_clip_grads_ansatz = 20.):
        super().__init__(system, hamiltonian, sampler, opt_ansatz, opt_actor)
        self.energy_clip_min = energy_clip_min
        self.energy_clip_max = energy_clip_max
        self.sigma_energy_clip = sigma_energy_clip
        self.min_clip_grads_ansatz = min_clip_grads_ansatz
        self.max_clip_grads_ansatz = max_clip_grads_ansatz
        self.max_phys_jump_std = max_phys_jump_std

    @partial(jax.jit, static_argnums=(0,))
    def step_ansatz(self, walkers, params_psi, opt_state_psi):
        """
        computes the energy gradients and updates steps of the ansatz
        """
        def energy_loss(p_psi):
            # compute local energy
            el_loc = self.hamiltonian.local_energy(self.sampler.ansatz_apply, p_psi, walkers)

            # energy hard clipping
            el_hard_clip = jnp.clip(jnp.nan_to_num(el_loc, nan=0.0), self.energy_clip_min, self.energy_clip_max)
            median = jnp.median(el_hard_clip)
            mad = jnp.mean(jnp.abs(el_hard_clip - median))
            clip_low = median - self.sigma_energy_clip * mad
            clip_high = median + self.sigma_energy_clip * mad
            el_clipped = jnp.clip(el_hard_clip, clip_low, clip_high)

            # log derivative
            log_psi = self.sampler.ansatz_apply(p_psi, walkers)

            # VLC loss function
            baseline = jnp.mean(el_clipped)
            diff_energy = jax.lax.stop_gradient(el_clipped - baseline)
            loss = 2.0 * jnp.mean(diff_energy * log_psi)

            return loss, (jnp.mean(el_clipped), jnp.var(el_clipped))
        
        # autodiff
        (loss_val, (mean_E, var_E)), grads = jax.value_and_grad(energy_loss, has_aux = True)(params_psi)

        # get rid of nans
        grads_safe = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grads)

        # optax update
        updates, new_opt_state = self.opt_ansatz.update(grads_safe, opt_state_psi, params_psi)
        new_params_psi = optax.apply_updates(params_psi, updates)

        # update cache
        def sum_log_psi_new(w):
            return jnp.sum(self.sampler.ansatz_apply(new_params_psi, w))
        
        new_log_psi = self.sampler.ansatz_apply(new_params_psi, walkers)
        new_grad = jnp.clip(jax.grad(sum_log_psi_new)(walkers), self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)

        updated_cache = {"log_psi_curr": new_log_psi, "grad_curr": new_grad}

        return new_params_psi, new_opt_state, mean_E, var_E, updated_cache
    
    @partial(jax.jit, static_argnums=(0,))
    def step_actor(self, key, walkers, params_psi, params_actor, opt_state_actor, curr_cache):
        """
        Actualiza el Actor neuronal usando RL (REINFORCE) evaluando la recompensa del salto.
        """
        k_s, _ = jax.random.split(key)

        def loss_fn(p_actor, w_current, k, cache):
            # get raw data for rl
            w_new, stats, new_cache, prop_aux = self.sampler.step(
                k, w_current, {"ansatz": params_psi, "actor": p_actor}, cache, exploration_mode=True
            )
            
            # unpack data
            rl_data = prop_aux["rl_data"]
            log_q_fwd_policy = rl_data["log_q_fwd_policy"]
            proposal_dist = rl_data["proposal_dist"]
            raw_penalty = rl_data["raw_penalty"]
            gamma = rl_data["gamma"]
            beta = rl_data["beta"]
            penalty_factor = rl_data["penalty_factor"]
            
            # reward
            safe_log_ratio = stats["mean_log_ratio"] 
            alpha_exact = jnp.exp(safe_log_ratio - jax.nn.softplus(safe_log_ratio))
            
            grad_penalty = jnp.clip(raw_penalty, 0.0, 50.0)
            # penalty_for_reject = penalty_factor * (1.0 - alpha_exact)

            max_physical_jump = self.max_phys_jump_std * jnp.sqrt(self.sampler.tau)
            
            term_dist = beta * jnp.clip(proposal_dist, a_min=0.0, a_max=max_physical_jump) * alpha_exact
            term_penalty = gamma * grad_penalty
            
            # rewards = term_dist - term_penalty - penalty_for_reject
            
            p_accept = jnp.exp(jnp.clip(safe_log_ratio, a_max=0.0))
            clipped_dist = jnp.clip(proposal_dist, a_min=0.0, a_max=max_physical_jump)

            grad_threshold = 50.0 # Valor empírico ajustable
            soft_node_penalty = jnp.maximum(0.0, raw_penalty - grad_threshold) * 0.01
            rewards = (p_accept * clipped_dist) - soft_node_penalty

            # Cálculo del Advantage para estabilizar REINFORCE
            mean_r = jnp.mean(rewards)
            std_r = jnp.std(rewards) + 1e-8
            advantage = (rewards - mean_r) / std_r
            
            # Loss Function: Policy Gradient (Maximizamos ventaja = Minimizamos -ventaja * prob)
            pg_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * log_q_fwd_policy)
            
            # Guardamos datos extra en stats
            stats["mean_reward"] = mean_r
            stats["reward_dist"] = jnp.mean(term_dist)
            stats["reward_penalty"] = jnp.mean(term_penalty)
            
            return pg_loss, (w_new, stats, new_cache)

        # Calculamos gradientes de la pérdida RL respecto a los pesos del Actor
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_actor, walkers, k_s, curr_cache)
        w_new, stats, new_cache = aux
        
        # Norma del gradiente para monitoreo de estabilidad
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        stats["actor_grad_norm"] = grad_norm 
        
        # Actualización Optax
        updates, new_opt_state = self.opt_actor.update(grads, opt_state_actor, params_actor)
        new_params_actor = optax.apply_updates(params_actor, updates)
        
        return new_params_actor, new_opt_state, w_new, stats, loss_val, new_cache
    
class GroundStateTrainer_new(BaseTrainer):
    """
    Trainer specialized in finding the truth ground state through energy minimization
    """

    def __init__(self, system, hamiltonian, sampler, opt_ansatz, opt_actor=None, max_phys_jump_std=3.0, sigma_energy_clip=5.0, energy_clip_min = -50., energy_clip_max = 0., min_clip_grads_ansatz = -20., max_clip_grads_ansatz = 20.):
        super().__init__(system, hamiltonian, sampler, opt_ansatz, opt_actor)
        self.energy_clip_min = energy_clip_min
        self.energy_clip_max = energy_clip_max
        self.sigma_energy_clip = sigma_energy_clip
        self.min_clip_grads_ansatz = min_clip_grads_ansatz
        self.max_clip_grads_ansatz = max_clip_grads_ansatz
        self.max_phys_jump_std = max_phys_jump_std
        print("rewards = kappa * (1 - ratio) * alpha + alpha * |x - x_prop|^2")

    @partial(jax.jit, static_argnums=(0,))
    def step_ansatz(self, walkers, params_psi, opt_state_psi):
        """
        computes the energy gradients and updates steps of the ansatz
        """
        def energy_loss(p_psi):
            # compute local energy
            el_loc = self.hamiltonian.local_energy(self.sampler.ansatz_apply, p_psi, walkers)

            # energy hard clipping
            el_hard_clip = jnp.clip(jnp.nan_to_num(el_loc, nan=0.0), self.energy_clip_min, self.energy_clip_max)
            median = jnp.median(el_hard_clip)
            mad = jnp.mean(jnp.abs(el_hard_clip - median))
            clip_low = median - self.sigma_energy_clip * mad
            clip_high = median + self.sigma_energy_clip * mad
            el_clipped = jnp.clip(el_hard_clip, clip_low, clip_high)

            # log derivative
            log_psi = self.sampler.ansatz_apply(p_psi, walkers)

            # VLC loss function
            baseline = jnp.mean(el_clipped)
            diff_energy = jax.lax.stop_gradient(el_clipped - baseline)
            loss = 2.0 * jnp.mean(diff_energy * log_psi)

            return loss, (jnp.mean(el_clipped), jnp.var(el_clipped))
        
        # autodiff
        (loss_val, (mean_E, var_E)), grads = jax.value_and_grad(energy_loss, has_aux = True)(params_psi)

        # get rid of nans
        grads_safe = jax.tree.map(lambda x: jnp.where(jnp.isfinite(x), x, 0.0), grads)

        # optax update
        updates, new_opt_state = self.opt_ansatz.update(grads_safe, opt_state_psi, params_psi)
        new_params_psi = optax.apply_updates(params_psi, updates)

        # update cache
        def sum_log_psi_new(w):
            return jnp.sum(self.sampler.ansatz_apply(new_params_psi, w))
        
        new_log_psi = self.sampler.ansatz_apply(new_params_psi, walkers)
        new_grad = jnp.clip(jax.grad(sum_log_psi_new)(walkers), self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)

        updated_cache = {"log_psi_curr": new_log_psi, "grad_curr": new_grad}

        return new_params_psi, new_opt_state, mean_E, var_E, updated_cache
    
    @partial(jax.jit, static_argnums=(0,))
    def step_actor(self, key, walkers, params_psi, params_actor, opt_state_actor, curr_cache):
        """
        Actualiza el Actor neuronal usando RL (REINFORCE) evaluando la recompensa del salto.
        """
        k_s, _ = jax.random.split(key)

        def loss_fn(p_actor, w_current, k, cache):
            # get raw data for rl
            w_new, stats, new_cache, prop_aux = self.sampler.step(
                k, w_current, {"ansatz": params_psi, "actor": p_actor}, cache, exploration_mode=True
            )
            
            # unpack data
            rl_data = prop_aux["rl_data"]
            log_q_fwd_policy = rl_data["log_q_fwd_policy"]
            proposal_dist = rl_data["proposal_dist"]
            raw_penalty = rl_data["raw_penalty"]
            gamma = rl_data["gamma"]
            beta = rl_data["beta"]
            kappa = rl_data["kappa"]
            # penalty_factor = rl_data["penalty_factor"]
            
            # reward
            safe_log_ratio = stats["safe_log_ratio_vector"] 
            alpha_exact = jnp.exp(safe_log_ratio - jax.nn.softplus(safe_log_ratio))
            
            grad_penalty = jnp.clip(raw_penalty, 0.0, 50.0)
            # penalty_for_reject = penalty_factor * (1.0 - alpha_exact)

            max_physical_jump = self.max_phys_jump_std * jnp.sqrt(self.sampler.tau)
            
            term_dist = beta * jnp.clip(proposal_dist, a_min=0.0, a_max=max_physical_jump) * alpha_exact
            term_penalty = gamma * grad_penalty
            
            # rewards = term_dist - term_penalty - penalty_for_reject
            
            p_accept = jnp.exp(jnp.clip(safe_log_ratio, a_max=0.0))
            clipped_dist = jnp.clip(proposal_dist, a_min=0.0, a_max=max_physical_jump)

            grad_threshold = 50.0 # Valor empírico ajustable
            soft_node_penalty = jnp.maximum(0.0, raw_penalty - grad_threshold) * 0.01
            rewards = (p_accept * clipped_dist) - soft_node_penalty + kappa * (1 - p_accept) * alpha_exact

            # Cálculo del Advantage para estabilizar REINFORCE
            mean_r = jnp.mean(rewards)
            std_r = jnp.std(rewards) + 1e-8
            advantage = (rewards - mean_r) / std_r
            
            # Loss Function: Policy Gradient (Maximizamos ventaja = Minimizamos -ventaja * prob)
            pg_loss = -jnp.mean(jax.lax.stop_gradient(advantage) * log_q_fwd_policy)
            
            # Guardamos datos extra en stats
            stats["mean_reward"] = mean_r
            stats["reward_dist"] = jnp.mean(term_dist)
            stats["reward_penalty"] = jnp.mean(term_penalty)
            
            return pg_loss, (w_new, stats, new_cache)

        # Calculamos gradientes de la pérdida RL respecto a los pesos del Actor
        (loss_val, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_actor, walkers, k_s, curr_cache)
        w_new, stats, new_cache = aux
        
        # Norma del gradiente para monitoreo de estabilidad
        grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
        stats["actor_grad_norm"] = grad_norm 
        
        # Actualización Optax
        updates, new_opt_state = self.opt_actor.update(grads, opt_state_actor, params_actor)
        new_params_actor = optax.apply_updates(params_actor, updates)
        
        return new_params_actor, new_opt_state, w_new, stats, loss_val, new_cache