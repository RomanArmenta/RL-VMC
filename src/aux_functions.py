import jax
from functools import partial
import jax.numpy as jnp
import numpy as np

@partial(jax.jit, static_argnums=(0, 6, 7))
def compiled_measure_phase(sampler_obj, w_start, k_start, p_psi, p_act, init_cache, hamiltonian, n_steps=5000):
    def measure_body(carry, _):
        w, k, cache = carry
        k, subkey = jax.random.split(k)
        
        # Ejecutamos el paso usando diccionario unificado para parámetros
        params_dict = {"ansatz": p_psi, "actor": p_act}
        w_new, stats, new_cache, _ = sampler_obj.step(subkey, w, params_dict, cache, exploration_mode=False)
        
        el_loc = hamiltonian.local_energy(sampler_obj.ansatz_apply, p_psi, w_new)
        return (w_new, k, new_cache), (el_loc, stats["acceptance"])

    _, (el_history, acc_history) = jax.lax.scan(
        measure_body, (w_start, k_start, init_cache), jnp.arange(n_steps)
    )
    return el_history, acc_history

def calculate_iat(time_series, max_lag=2000):
    x = np.asarray(time_series)
    N = len(x)
    mu, var = np.mean(x), np.var(x)
    if var == 0: return 1.0
        
    x_centered = x - mu
    tau_int = 0.5
    for t in range(1, max_lag):
        Ct = np.sum(x_centered[:-t] * x_centered[t:]) / (N - t)
        rho_t = Ct / var
        tau_int += rho_t
        if rho_t < 0.05 or t >= 5 * tau_int: break
    return tau_int