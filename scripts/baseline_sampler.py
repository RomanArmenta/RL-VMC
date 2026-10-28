import os
os.environ["JAX_ENABLE_X64"] = "True"
import sys
import numpy as np
import jax
import jax.numpy as jnp
import argparse

# Rutas
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RL_VMC_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../"))
UHF1D_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../uhf_1d_ho_v270826_m-determinants/src"))
sys.path.append(RL_VMC_SRC)
sys.path.append(UHF1D_SRC)

# Importamos tu función original de casscf para que evalúe correctamente la energía
from evaluar_wigner_casscf import cargar_ansatz_casscf 

def brute_force_metropolis(log_pdf, initial_x, num_steps, step_size=0.1):
    """
    Random Walk Metropolis estándar (sin gradientes) pero muy lento y con paso ultra fino.
    Garantiza convergencia exacta por balance detallado si se corre lo suficiente.
    """
    @jax.jit
    def step(carry, key):
        x, curr_log_prob = carry
        
        # Propuesta isotrópica simple
        noise = jax.random.normal(key, x.shape) * step_size
        prop_x = x + noise
        
        # Evaluar propuesta
        prop_log_prob = log_pdf(None, prop_x) # Params = None para CASSCF estático
        
        # Aceptación
        ratio = prop_log_prob - curr_log_prob
        # Log-uniform
        log_u = jnp.log(jax.random.uniform(key, shape=(x.shape[0],)))
        
        # Máscara de aceptación (broadcasted para las dimensiones de x)
        accept = log_u < ratio
        accept_expanded = accept[:, None, None]
        
        new_x = jnp.where(accept_expanded, prop_x, x)
        new_log_prob = jnp.where(accept, prop_log_prob, curr_log_prob)
        
        return (new_x, new_log_prob), new_x

    # Evaluación inicial
    curr_log_prob = log_pdf(None, initial_x)
    
    # Pre-generar keys para lax.scan
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_steps)
    
    print(f"-> Starting Brute-Force Metropolis for {num_steps} steps...")
    _, history = jax.lax.scan(step, (initial_x, curr_log_prob), keys)
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500000, help="Número masivo de pasos para el ground truth")
    parser.add_argument("--walkers", type=int, default=500, help="Número masivo de walkers para el ground truth")
    parser.add_argument("--burn_in", type=int, default=50000, help="Descarte inicial")
    parser.add_argument("--out_file", type=str, default="ground_truth_samples.npy")
    args = parser.parse_args()

    npz_path = os.path.join(UHF1D_SRC, "../results_N4/g_4.0/cas_n4_g4_omega0.1_soft0.1_b32_c0-a4/casscf_state.npz")
    print(f"1. Loading CASSCF Ansatz from: {npz_path}")
    ansatz_apply, cfg = cargar_ansatz_casscf(npz_path)
    
    n_elec = cfg["n_up"] + cfg["n_down"]
    key = jax.random.PRNGKey(999)
    # Inicialización aleatoria forzada fuera del cero para evitar log(0) instantáneos
    initial_x = jnp.ones((args.walkers, n_elec, 1)) * 5.0 + (jnp.arange(n_elec).reshape(1, n_elec, 1) * 1.5)

    # 1. Correr Metropolis
    history = brute_force_metropolis(ansatz_apply, initial_x, args.steps, step_size=0.1)
    
    # 2. Descartar Burn-in y aplanar (Thinning opcional, aquí tomamos 1 de cada 10 para ahorrar RAM)
    valid_samples = history[args.burn_in::10] 
    flat_samples = np.array(valid_samples).reshape(-1)
    
    # 3. Guardar en disco
    print(f"-> Guardando {flat_samples.shape[0]} muestras doradas (Ground Truth) en: {args.out_file}")
    np.save(args.out_file, flat_samples)
    print("Listo. Puedes usar este archivo para evaluar KL Divergence.")