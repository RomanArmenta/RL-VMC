import jax
import jax.numpy as jnp
from functools import partial
from abc import ABC, abstractmethod

class BaseSampler(ABC):
    """
    abstract class for all Monte Carlo samplers. Is Metropolis-Hastings-like 
    """
    def __init__(self, ansatz_apply, system, batch_size, max_clip_log_ratio = 20., min_clip_log_ratio = -20., nan_clip_log_ratio=-20., neginf_clip_log_ratio=-20., posinf_clip_log_ratio=20.):
        self.ansatz_apply = ansatz_apply       # apply function for log wavefunction (flax notation-like)
        self.system = system       # state that store hard (ummutable<) variables (for example, nuclei positions in H10 chain)
        self.batch_size = batch_size           # number of parallel independent walkers
        self.max_clip_log_ratio = max_clip_log_ratio
        self.min_clip_log_ratio = min_clip_log_ratio
        self.nan_clip_log_ratio = nan_clip_log_ratio
        self.neginf_clip_log_ratio = neginf_clip_log_ratio
        self.posinf_clip_log_ratio = posinf_clip_log_ratio
    
    @abstractmethod
    def propose(self, key_prop, walkers, params, cache, **kwargs):
        """
        This function must return:
        - w_prop: new proposal
        - prop_log_psi: log|Psi| valued in w_prop
        - grad_prop: gradient of log|Psi| valued in w_prop
        - log_q_ratio: log ( q(w | w_prop) ) - log( q(w_prop | w) ) (kernel asymmetry)
        - prop_aux: dict with stats, rewards, general data for analysis or training
        """
        pass

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, walkers, params, cache, **kwargs):
        # a monte carlo step given the wavefunctions

        # 1. the key is splitted into two random keys
        key_prop, key_accept = jax.random.split(key)

        # extract current info of current position of electrons
        log_psi_curr = cache['log_psi_curr']

        # 2. propose the new positions given the propose function
        w_prop, prop_log_psi, grad_prop, log_q_ratio, prop_aux = self.propose(
            key_prop, walkers, params, cache, **kwargs
        )

        # 3. Metropolis ratio
        log_ratio = 2.0 * (prop_log_psi - log_psi_curr) + log_q_ratio

        # we did this to pass this safe version to the reward without gradient exploitation
        safe_log_ratio = jnp.clip(jnp.nan_to_num(log_ratio, nan=self.nan_clip_log_ratio, neginf=self.neginf_clip_log_ratio, posinf=self.posinf_clip_log_ratio), self.min_clip_log_ratio, self.max_clip_log_ratio)

        #4. accept / reject
        rnd = jnp.log(jax.random.uniform(key_accept, shape=(self.batch_size,)))
        accept = rnd < safe_log_ratio # we use this because it doesnt affect the statistics (is almost equivalent in this step)

        # 5. vmap update
        w_next = jnp.where(accept[:, None, None], w_prop, walkers)
        next_log_psi = jnp.where(accept, prop_log_psi, log_psi_curr)

        # 6. update general cache
        new_cache = {"log_psi_curr": next_log_psi}
        if grad_prop is not None:
            new_cache['grad_curr'] = jnp.where(accept[:, None, None], grad_prop, cache["grad_curr"])

        # 7. stats
        stats = self.stats_fn(walkers, w_prop, w_next, accept, safe_log_ratio, prop_aux)

        return w_next, stats, new_cache, prop_aux
        
    def stats_fn(self, w_old, w_prop, w_next, accept, safe_log_ratio, prop_aux):
        # compute EJD and other important stats

        ejd = jnp.mean(jnp.linalg.norm(w_next - w_old, axis=-1))

        stats = {
            "acceptance": jnp.mean(accept),
            "mean_log_ratio": jnp.mean(safe_log_ratio),
            "safe_log_ratio_vector": safe_log_ratio,
            "ejd": ejd
        }

        if "stats" in prop_aux:
            stats.update(prop_aux["stats"])
        
        return stats

class MetropolisHastingsSampler(BaseSampler):
    
    def __init__(self, ansatz_apply, batch_size, sigma, system):
        super().__init__(ansatz_apply, batch_size, system)
        self.sigma = sigma

    def propose(self, key_prop, walkers, params, cache, **kwargs):
        """
        This function must return:
        - w_prop: new proposal
        - prop_log_psi: log|Psi| valued in w_prop
        - grad_prop: gradient of log|Psi| valued in w_prop
        - log_q_ratio: log ( q(w | w_prop) ) - log( q(w_prop | w) ) (kernel asymmetry)
        - prop_aux: dict with stats, rewards, general data for analysis or training
        """
        w_prop = walkers + self.sigma * jax.random.normal(key_prop, walkers.shape)

        prop_log_psi = self.ansatz_apply(params["ansatz"], w_prop)

        log_q_ratio = 0.

        return w_prop, prop_log_psi, None, log_q_ratio, {}

class MALASampler(BaseSampler):
    def __init__(self, ansatz_apply, batch_size, tau, system, max_clip_grads_ansatz, min_clip_grads_ansatz, max_clip_log_ratio, min_clip_log_ratio, nan_clip_log_ratio, neginf_clip_log_ratio, posinf_clip_log_ratio):
        super().__init__(ansatz_apply=ansatz_apply, system=system, batch_size=batch_size, max_clip_log_ratio=max_clip_log_ratio, min_clip_log_ratio=min_clip_log_ratio, nan_clip_log_ratio=nan_clip_log_ratio, neginf_clip_log_ratio=neginf_clip_log_ratio, posinf_clip_log_ratio=posinf_clip_log_ratio)
        self.tau = tau
        self.sigma = jnp.sqrt(2.0 * tau)
        self.max_clip_grads_ansatz=max_clip_grads_ansatz
        self.min_clip_grads_ansatz=min_clip_grads_ansatz

    def propose(self, key_prop, walkers, params, cache, **kwargs):
        """
        This function must return:
        - w_prop: new proposal
        - prop_log_psi: log|Psi| valued in w_prop
        - grad_prop: gradient of log|Psi| valued in w_prop
        - log_q_ratio: log ( q(w | w_prop) ) - log( q(w_prop | w) ) (kernel asymmetry)
        - prop_aux: dict with stats, rewards, general data for analysis or training
        """

        grad_curr = cache["grad_curr"]
        
        # langevin proposal
        noise = jax.random.normal(key_prop, walkers.shape)
        drift_curr = self.tau * grad_curr
        
        w_prop = walkers + drift_curr + self.sigma * noise
        
        # compute things
        def log_psi_and_sum(w):
            val = self.ansatz_apply(params["ansatz"], w)
            return jnp.sum(val), val
        
        (_, log_psi_prop), raw_grad_prop = jax.value_and_grad(log_psi_and_sum, has_aux=True)(w_prop)

        grad_prop = jnp.clip(raw_grad_prop, self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)
        drift_prop = self.tau * grad_prop
        
        log_q_fwd = jnp.sum(-0.5 * ((w_prop - walkers - drift_curr) / self.sigma)**2, axis=(-1, -2))
        log_q_bwd = jnp.sum(-0.5 * ((walkers - w_prop - drift_prop) / self.sigma)**2, axis=(-1, -2))

        log_q_ratio = log_q_bwd - log_q_fwd

        return w_prop, log_psi_prop, grad_prop, log_q_ratio, {}

class RLMALASampler_(BaseSampler):
    """
    hybrid sampler guided by reinforcement learning
    combines langevin - like learned drift with a parametric neural actor
    """
    def __init__(self, ansatz_apply, system, batch_size, max_clip_log_ratio, min_clip_log_ratio, nan_clip_log_ratio, neginf_clip_log_ratio, posinf_clip_log_ratio, actor_apply, tau, gamma, beta, kappa, penalty_factor, scale_clip_min, scale_clip_max, min_clip_grads_ansatz, max_clip_grads_ansatz):
        super().__init__(ansatz_apply=ansatz_apply, system=system, batch_size=batch_size, max_clip_log_ratio=max_clip_log_ratio, min_clip_log_ratio=min_clip_log_ratio, nan_clip_log_ratio=nan_clip_log_ratio, neginf_clip_log_ratio=neginf_clip_log_ratio, posinf_clip_log_ratio=posinf_clip_log_ratio)
        self.gamma = gamma
        self.beta = beta
        self.penalty_factor = penalty_factor
        self.sigma = jnp.sqrt(2.0 * tau)
        self.tau = tau
        self.kappa = kappa
        self.scale_clip_min = scale_clip_min
        self.scale_clip_max = scale_clip_max
        self.actor_apply = actor_apply
        self.min_clip_grads_ansatz=min_clip_grads_ansatz
        self.max_clip_grads_ansatz=max_clip_grads_ansatz

    def _compute_log_q(self, x_prop, x_curr, drift_curr, sigma_curr):
        """
        this computes log q (x_prop | x_curr) assuming a gaussian distribution 
        """
        mean = x_curr + drift_curr

        # we need the the term -jnp.log(sigma_from) for covariance change
        log_prob = -0.5 * ((x_prop - mean) / sigma_curr) ** 2 - jnp.log(sigma_curr)
        
        return jnp.sum(log_prob, axis=(-1, -2))

    def propose(self, key_prop, walkers, params, cache, **kwargs):
        # we can change the exploration mode through kwargs
        exploration_mode = kwargs.get("exploration_mode", True)
        n_elec = walkers.shape[1]

        # 1. get currrent gradient
        grad_curr = cache["grad_curr"]

        # 2. pass it through the actor
        actor_vmap = jax.vmap(self.actor_apply, in_axes = (None, None, 0, None, None), out_axes=1)

        drift_actor_curr, raw_sigma_scale_curr = actor_vmap(params["actor"], walkers, jnp.arange(n_elec), self.system.coords, exploration_mode)

        safe_scale_curr = jnp.exp(jnp.clip(raw_sigma_scale_curr, self.scale_clip_min, self.scale_clip_max))

        # 3. hybrid proposal
        drift_curr = (self.tau * grad_curr) + drift_actor_curr
        sigma_curr = self.sigma * safe_scale_curr

        noise = jax.random.normal(key_prop, walkers.shape)
        w_prop = walkers + drift_curr + sigma_curr * noise

        # 4. evaluate new proposal
        def log_psi_and_sum(w):
            val = self.ansatz_apply(params["ansatz"], w)
            return jnp.sum(val), val
        
        (_, log_psi_prop), raw_grad_prop = jax.value_and_grad(log_psi_and_sum, has_aux=True)(w_prop)
        grad_prop = jnp.clip(raw_grad_prop, self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)

        # 5. actor backwards
        drift_actor_prop, raw_sigma_scale_prop = actor_vmap(params["actor"], w_prop, jnp.arange(n_elec), self.system.coords, exploration_mode)
        safe_scale_prop = jnp.exp(jnp.clip(raw_sigma_scale_prop, self.scale_clip_min, self.scale_clip_max))

        drift_prop = (self.tau * grad_prop) + drift_actor_prop
        sigma_prop = self.sigma * safe_scale_prop

        # 6. compute kernel
        log_q_fwd = self._compute_log_q(w_prop, walkers, drift_curr, sigma_curr)
        log_q_bwd = self._compute_log_q(walkers, w_prop, drift_prop, sigma_prop)
        log_q_ratio = log_q_bwd - log_q_fwd

        # we need the stop_gradient for the FEINFORCE metrics
        w_prop_stopped = jax.lax.stop_gradient(w_prop)
        log_q_fwd_policy = self._compute_log_q(w_prop_stopped, walkers, drift_curr, sigma_curr)

        # 7. package the aux for rl training
        proposal_dist = jnp.mean(jnp.linalg.norm(w_prop - walkers, axis=-1), axis=-1)
        raw_penalty = jnp.mean(jnp.linalg.norm(grad_prop, axis=-1), axis=-1)

        prop_aux = {
            "rl_data": {
                "log_q_fwd_policy": log_q_fwd_policy,
                "proposal_dist": proposal_dist,
                "raw_penalty": raw_penalty,
                "gamma": self.gamma,
                "beta": self.beta,
                "kappa": self.kappa,
                "penalty_factor": self.penalty_factor
            },
            "stats": {
                "mean_sigma": jnp.mean(sigma_curr),
                "mag_actor": jnp.mean(jnp.linalg.norm(drift_actor_curr, axis=-1)),
                "mag_mala": jnp.mean(jnp.linalg.norm(self.tau * grad_curr, axis=-1))
            }
        }

        return w_prop, log_psi_prop, grad_prop, log_q_ratio, prop_aux

class RLMALASampler(BaseSampler):
    """
    hybrid sampler guided by reinforcement learning
    combines langevin - like learned drift with a parametric neural actor
    """
    def __init__(self, ansatz_apply, system, batch_size, max_clip_log_ratio, min_clip_log_ratio, nan_clip_log_ratio, neginf_clip_log_ratio, posinf_clip_log_ratio, actor_apply, tau, gamma, beta, kappa, penalty_factor, scale_clip_min, scale_clip_max, min_clip_grads_ansatz, max_clip_grads_ansatz):
        super().__init__(ansatz_apply=ansatz_apply, system=system, batch_size=batch_size, max_clip_log_ratio=max_clip_log_ratio, min_clip_log_ratio=min_clip_log_ratio, nan_clip_log_ratio=nan_clip_log_ratio, neginf_clip_log_ratio=neginf_clip_log_ratio, posinf_clip_log_ratio=posinf_clip_log_ratio)
        self.gamma = gamma
        self.beta = beta
        self.penalty_factor = penalty_factor
        self.sigma = jnp.sqrt(2.0 * tau)
        self.tau = tau
        self.kappa = kappa
        self.scale_clip_min = scale_clip_min
        self.scale_clip_max = scale_clip_max
        self.actor_apply = actor_apply
        self.min_clip_grads_ansatz=min_clip_grads_ansatz
        self.max_clip_grads_ansatz=max_clip_grads_ansatz

    def _compute_log_q(self, x_prop, x_curr, drift_curr, sigma_curr):
        """
        Calcula log q (x_prop | x_curr) adaptándose automáticamente
        a tensores isotrópicos (diagonal) o covariantes (Cholesky).
        """
        mean = x_curr + drift_curr

        # Detección estática en tiempo de compilación JIT
        is_covariant = (sigma_curr.ndim == 4) 

        if is_covariant:
            # Matemática Covariante: sigma_curr es la matriz de Cholesky L (batch, elec, 3, 3)
            delta = x_prop - mean
            
            # 1. Invertimos la matriz L. Usamos la inversa para resolver L * y = delta
            inv_L = jnp.linalg.inv(sigma_curr)
            
            # 2. Multiplicación matricial por lotes: y = L^-1 * delta
            y = jnp.einsum('beij,bej->bei', inv_L, delta)
            
            # 3. Determinante de Cholesky: log(|det(L)|) = sum(log(diag(L)))
            log_det = jnp.sum(jnp.log(jnp.diagonal(sigma_curr, axis1=-2, axis2=-1)), axis=-1)
            
            # 4. Probabilidad Logarítmica (ignorando la constante 2*pi que se cancela en el ratio)
            log_prob = -0.5 * jnp.sum(y ** 2, axis=-1) - log_det
            
            return jnp.sum(log_prob, axis=-1) # Suma sobre todos los electrones
            
        else:
            # Matemática Element-wise (tu código original)
            log_prob = -0.5 * ((x_prop - mean) / sigma_curr) ** 2 - jnp.log(sigma_curr)
            return jnp.sum(log_prob, axis=(-1, -2))

    def propose(self, key_prop, walkers, params, cache, **kwargs):
        # we can change the exploration mode through kwargs
        exploration_mode = kwargs.get("exploration_mode", True)
        n_elec = walkers.shape[1]

        # 1. get currrent gradient
        grad_curr = cache["grad_curr"]

        # 2. pass it through the actor
        actor_vmap = jax.vmap(self.actor_apply, in_axes = (None, None, 0, None, None), out_axes=1)

        drift_actor_curr, raw_sigma_scale_curr = actor_vmap(params["actor"], walkers, jnp.arange(n_elec), self.system.coords, exploration_mode)

        safe_scale_curr = jnp.exp(jnp.clip(raw_sigma_scale_curr, self.scale_clip_min, self.scale_clip_max))

        # -----------------------------------------------------
        # 3. Hybrid Proposal (Hacia adelante)
        # -----------------------------------------------------
        drift_curr = (self.tau * grad_curr) + drift_actor_curr
        noise = jax.random.normal(key_prop, walkers.shape)

        if raw_sigma_scale_curr.ndim == 4:
            # Modo Covariante: Multiplicación matricial (einsum)
            sigma_curr = self.sigma * raw_sigma_scale_curr
            jump_curr = jnp.einsum('beij,bej->bei', sigma_curr, noise)
        else:
            # Modo Element-wise: Multiplicación escalar/diagonal
            safe_scale_curr = jnp.exp(jnp.clip(raw_sigma_scale_curr, self.scale_clip_min, self.scale_clip_max))
            sigma_curr = self.sigma * safe_scale_curr
            jump_curr = sigma_curr * noise
            
        w_prop = walkers + drift_curr + jump_curr

        # 4. evaluate new proposal
        def log_psi_and_sum(w):
            val = self.ansatz_apply(params["ansatz"], w)
            return jnp.sum(val), val
        
        (_, log_psi_prop), raw_grad_prop = jax.value_and_grad(log_psi_and_sum, has_aux=True)(w_prop)
        grad_prop = jnp.clip(raw_grad_prop, self.min_clip_grads_ansatz, self.max_clip_grads_ansatz)

        # -----------------------------------------------------
        # 5. Actor backwards (Hacia atrás)
        # -----------------------------------------------------
        drift_actor_prop, raw_sigma_scale_prop = actor_vmap(params["actor"], w_prop, jnp.arange(n_elec), self.system.coords, exploration_mode)
        drift_prop = (self.tau * grad_prop) + drift_actor_prop

        if raw_sigma_scale_prop.ndim == 4:
            # Modo Covariante
            sigma_prop = self.sigma * raw_sigma_scale_prop
        else:
            # Modo Element-wise
            safe_scale_prop = jnp.exp(jnp.clip(raw_sigma_scale_prop, self.scale_clip_min, self.scale_clip_max))
            sigma_prop = self.sigma * safe_scale_prop
            
        # 6. compute kernel
        log_q_fwd = self._compute_log_q(w_prop, walkers, drift_curr, sigma_curr)
        log_q_bwd = self._compute_log_q(walkers, w_prop, drift_prop, sigma_prop)
        log_q_ratio = log_q_bwd - log_q_fwd

        # we need the stop_gradient for the FEINFORCE metrics
        w_prop_stopped = jax.lax.stop_gradient(w_prop)
        log_q_fwd_policy = self._compute_log_q(w_prop_stopped, walkers, drift_curr, sigma_curr)

        # 7. package the aux for rl training
        proposal_dist = jnp.mean(jnp.linalg.norm(w_prop - walkers, axis=-1), axis=-1)
        raw_penalty = jnp.mean(jnp.linalg.norm(grad_prop, axis=-1), axis=-1)

        prop_aux = {
            "rl_data": {
                "log_q_fwd_policy": log_q_fwd_policy,
                "proposal_dist": proposal_dist,
                "raw_penalty": raw_penalty,
                "gamma": self.gamma,
                "beta": self.beta,
                "kappa": self.kappa,
                "penalty_factor": self.penalty_factor
            },
            "stats": {
                "mean_sigma": jnp.mean(sigma_curr),
                "mag_actor": jnp.mean(jnp.linalg.norm(drift_actor_curr, axis=-1)),
                "mag_mala": jnp.mean(jnp.linalg.norm(self.tau * grad_curr, axis=-1))
            }
        }

        return w_prop, log_psi_prop, grad_prop, log_q_ratio, prop_aux
