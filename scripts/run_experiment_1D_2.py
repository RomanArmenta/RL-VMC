import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import argparse
import json
import pickle
import pandas as pd
import numpy as np
from functools import partial

import jax
import jax.numpy as jnp
import optax

from src.system import Harmonic_1D
from src.hamiltonians import Harmonic1DHamiltonian
from src.wavefunctions import Ansatz1D_Backflow
from src.samplers import MALASampler, RLMALASampler_Huziel
from src.trainers import GroundStateTrainer_new
from src.aux_functions import compiled_measure_phase, format_metrics

from src.wavefunction import WavefunctionConfig
from src.orbitals import OrbitalConfig
from src.correlation import CorrelationConfig
from src.td3_modules import TD3Trainer, ReplayBuffer

@partial(jax.jit, static_argnums=(0, 6))
def termalizar_caminantes(sampler, w_start, k_start, cache_start, p_psi, p_act, n_steps):
    """Evoluciona los caminantes sin acumular gradientes ni historial para alcanzar el estado base."""
    def step_fn(carry, _):
        w_curr, k_curr, cache_curr = carry
        k_curr, subkey = jax.random.split(k_curr)
        
        params = {"ansatz": p_psi}
        if p_act is not None:
            params["actor"] = p_act
            
        w_new, _, cache_new, _ = sampler.step(subkey, w_curr, params, cache_curr, exploration_mode=True)
        return (w_new, k_curr, cache_new), None
                  
    final_carry, _ = jax.lax.scan(step_fn, (w_start, k_start, cache_start), jnp.arange(n_steps))
    return final_carry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="MALA", choices=["MALA", "TD3"])
    
    # Parámetros del Sistema
    parser.add_argument("--n_elec", type=int, default=4)
    parser.add_argument("--n_up", type=int, default=2)
    parser.add_argument("--n_dn", type=int, default=2)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--interaction_strength", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=0.1)
    
    # Configuraciones de Ansatz
    parser.add_argument("--wf_model", type=str, default="slater_jastrow")
    parser.add_argument("--orbital_model", type=str, default="backflow_ho")
    parser.add_argument("--correlation_model", type=str, default="spin_rbf_jastrow")
    parser.add_argument("--backflow_components", type=int, default=8)
    parser.add_argument("--backflow_r_max", type=float, default=5.0)
    parser.add_argument("--backflow_width", type=float, default=0.7)
    parser.add_argument("--backflow_envelope_length", type=float, default=3.0)
    
    # Parámetros Muestreo y RL
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--actor_amplitude", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--penalty_factor", type=float, default=0.1)
    parser.add_argument("--max_phys_jump_std", type=float, default=5.0)
    
    # Optimizadores
    parser.add_argument("--lr_ansatz", type=float, default=5e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-3)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    
    # Rutinas de Entrenamiento
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--measure_steps", type=int, default=2000)
    parser.add_argument("--coupled_training_cycles", type=int, default=4)
    parser.add_argument("--vmc_iterations_per_cycle", type=int, default=1250)
    parser.add_argument("--rl_sweeps_per_cycle", type=int, default=100)
    parser.add_argument("--rl_replay_capacity", type=int, default=100000)
    parser.add_argument("--rl_gamma_td3", type=float, default=0.99)
    
    # Estabilidad y Clipping
    parser.add_argument("--energy_clip_min", type=float, default=-100.0)
    parser.add_argument("--energy_clip_max", type=float, default=100.0)
    parser.add_argument("--sigma_energy_clip", type=float, default=5.0)
    parser.add_argument("--min_clip_grads_ansatz", type=float, default=-50.0)
    parser.add_argument("--max_clip_grads_ansatz", type=float, default=50.0)
    parser.add_argument("--min_clip_log_ratio", type=float, default=-50.0)
    parser.add_argument("--max_clip_log_ratio", type=float, default=50.0)
    parser.add_argument("--nan_clip_log_ratio", type=float, default=-50.0)
    parser.add_argument("--posinf_clip_log_ratio", type=float, default=50.0)
    parser.add_argument("--neginf_clip_log_ratio", type=float, default=-50.0)
    parser.add_argument("--scale_clip_min", type=float, default=-5.0)
    parser.add_argument("--scale_clip_max", type=float, default=1.0)
    
    # Tracking e I/O
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_print", type=int, default=100)
    parser.add_argument("--n_log", type=int, default=50)
    parser.add_argument("--steps_for_best", type=int, default=500)
    parser.add_argument("--results_out_dir", type=str, required=True)
    parser.add_argument("--n_ter", type=int, default=1000, help="Pasos de termalización antes de iniciar el entrenamiento")
    

    args, _ = parser.parse_known_args()
    
    out_dir = args.results_out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.json"), "w") as f: 
        json.dump(vars(args), f, indent=4)
        
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    # Configuración Física
    system = Harmonic_1D(name="Harmonic1D", n_elec=args.n_elec, n_up=args.n_up, n_dn=args.n_dn, 
                         coords=jnp.array([[0.0]]), charges=jnp.array([0]), omega=args.omega, 
                         interaction_strength=args.interaction_strength, softening=args.softening, dim=1)
    hamiltonian = Harmonic1DHamiltonian(system)
    
    wf_cfg = WavefunctionConfig(name=args.wf_model)
    orbital_cfg = OrbitalConfig(
        model=args.orbital_model, backflow_components=args.backflow_components,
        backflow_r_max=args.backflow_r_max, backflow_width=args.backflow_width, 
        backflow_envelope_length=args.backflow_envelope_length
    )
    correlation_cfg = CorrelationConfig(model=args.correlation_model)
    
    ansatz = Ansatz1D_Backflow(system, wf_cfg, orbital_cfg, correlation_cfg)
    walkers = jax.random.normal(k_init, (args.batch_size, args.n_elec, 1))
    params_psi = ansatz.init(k_init, walkers)

    opt_ansatz = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=args.lr_ansatz))
    opt_state_psi = opt_ansatz.init(params_psi)

    def sum_log_psi(w): return jnp.sum(ansatz.apply(params_psi, w))
    cache = {
        "log_psi_curr": ansatz.apply(params_psi, walkers), 
        "grad_curr": jnp.clip(jax.grad(sum_log_psi)(walkers), args.min_clip_grads_ansatz, args.max_clip_grads_ansatz)
    }

    training_history = []
    e_best = jnp.inf
    best_params_ansatz = params_psi
    p_act, p_c1, p_c2 = None, None, None
    global_start_time = time.time()

    # =========================================================
    # EJECUCIÓN MALA CLÁSICO (Sin Actor Neuronal)
    # =========================================================
    if args.method == "MALA":
        sampler = MALASampler(
            ansatz.apply, args.batch_size, args.tau, system, args.max_clip_grads_ansatz, args.min_clip_grads_ansatz, 
            args.max_clip_log_ratio, args.min_clip_log_ratio, args.nan_clip_log_ratio, args.neginf_clip_log_ratio, args.posinf_clip_log_ratio
        )
        trainer = GroundStateTrainer_new(
            system=system, hamiltonian=hamiltonian, sampler=sampler, opt_ansatz=opt_ansatz, opt_actor=None,
            sigma_energy_clip=args.sigma_energy_clip, energy_clip_min=args.energy_clip_min, energy_clip_max=args.energy_clip_max, 
            min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_std
        )

        # === NUEVO: TERMALIZACIÓN MALA ===
        print(f"\nTermalizando caminantes MALA por {args.n_ter} pasos...")
        final_carry = termalizar_caminantes(sampler, walkers, key, cache, params_psi, None, args.n_ter)
        walkers, key, cache = final_carry
        print("Termalización completada. Iniciando entrenamiento...")
        # =================================

        step_start_time = time.time()
        for step in range(args.train_steps):
            key, k_psi, k_samp = jax.random.split(key, 3)
            # 1. Optimizar Ansatz
            params_psi, opt_state_psi, e_mean, e_var, cache = trainer.step_ansatz(walkers, params_psi, opt_state_psi)
            # 2. Mover Walkers con MALA Clásico
            walkers, stats, cache, _ = sampler.step(k_samp, walkers, {"ansatz": params_psi}, cache, exploration_mode=False)

            if step > args.steps_for_best and e_mean < e_best:
                e_best = float(e_mean)
                best_params_ansatz = params_psi

            if step % args.n_log == 0:
                elapsed_time = time.time() - step_start_time
                all_stats = {"Step": step, "Time(s)": float(elapsed_time), "E_mean": float(e_mean)}
                all_stats.update(stats)
                log_data = {k: (v.item() if hasattr(v, "item") else v) for k, v in all_stats.items() if not (hasattr(v, "ndim") and v.ndim > 0)}
                training_history.append(log_data)
                if step % args.n_print == 0: print(format_metrics(log_data))
                step_start_time = time.time()

    # =========================================================
    # EJECUCIÓN TD3 (RL-MALA)
    # =========================================================
    elif args.method == "TD3":
        td3 = TD3Trainer(ansatz.apply, args.n_elec, args.n_up, args.n_dn, args.tau, args.actor_amplitude, beta=args.beta, gamma_reward=args.gamma, gamma_td3=args.rl_gamma_td3)
        
        def td3_actor_wrapper(params_actor, w, e_idx, r_nuc, exp_mode, grad):
            from src.td3_modules import build_td3_state
            w_1d, grad_1d = w[..., 0], grad[..., 0] 
            state = build_td3_state(w_1d, grad_1d, e_idx, args.n_elec, args.n_up, args.n_dn)
            action = td3.actor_model.apply(params_actor, state)
            drift_res, raw_sigma_scale = action[:, 0:1] * args.actor_amplitude, action[:, 1:2]
            return drift_res, raw_sigma_scale, {"d_act": jnp.abs(drift_res[:, 0]), "sig_scale": raw_sigma_scale[:, 0]}

        sampler = RLMALASampler_Huziel(
            ansatz_apply=ansatz.apply, system=system, batch_size=args.batch_size, actor_apply=td3_actor_wrapper, 
            tau=args.tau, gamma=args.gamma, beta=args.beta, kappa=args.kappa, penalty_factor=args.penalty_factor,
            max_clip_log_ratio=args.max_clip_log_ratio, min_clip_log_ratio=args.min_clip_log_ratio,
            nan_clip_log_ratio=args.nan_clip_log_ratio, neginf_clip_log_ratio=args.neginf_clip_log_ratio, posinf_clip_log_ratio=args.posinf_clip_log_ratio,
            scale_clip_min=args.scale_clip_min, scale_clip_max=args.scale_clip_max, 
            min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz
        )

        trainer_vmc = GroundStateTrainer_new(
            system=system, hamiltonian=hamiltonian, sampler=sampler, opt_ansatz=opt_ansatz, opt_actor=None,
            sigma_energy_clip=args.sigma_energy_clip, energy_clip_min=args.energy_clip_min, energy_clip_max=args.energy_clip_max, 
            min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_std
        )

        key, k1, k2, k3 = jax.random.split(key, 4)
        dummy_state = jnp.zeros((1, 4 + 5 + 3 * (max(max(args.n_up-1,0), max(args.n_dn-1,0)) + max(args.n_up, args.n_dn))))
        dummy_action = jnp.zeros((1, 2))
        
        p_act = td3.actor_model.init(k1, dummy_state)
        p_c1, p_c2 = td3.critic_model.init(k2, dummy_state, dummy_action), td3.critic_model.init(k3, dummy_state, dummy_action)
        t_act, t_c1, t_c2 = jax.tree_util.tree_map(lambda x: x.copy(), p_act), jax.tree_util.tree_map(lambda x: x.copy(), p_c1), jax.tree_util.tree_map(lambda x: x.copy(), p_c2)
        
        td3.opt_a, td3.opt_c = optax.adam(learning_rate=args.lr_actor), optax.adam(learning_rate=args.lr_critic)
        opt_act, opt_c1, opt_c2 = td3.opt_a.init(p_act), td3.opt_c.init(p_c1), td3.opt_c.init(p_c2)
        
        replay = ReplayBuffer(args.rl_replay_capacity, dummy_state.shape[-1], 2)
        global_step, td3_updates, last_td3_reward = 0, 0, 0.0
        last_td3_actor_loss, last_td3_critic1_loss, last_td3_critic2_loss = 0.0, 0.0, 0.0
        
        # === NUEVO: TERMALIZACIÓN TD3 ===
        print(f"\nTermalizando caminantes TD3 por {args.n_ter} pasos...")
        final_carry = termalizar_caminantes(sampler, walkers, key, cache, params_psi, p_act, args.n_ter)
        walkers, key, cache = final_carry
        print("Termalización completada. Iniciando entrenamiento acoplado...")
        # =================================

        for cycle in range(args.coupled_training_cycles):
            cycle_start_time = time.time()
            print(f"\n--- CICLO TD3 {cycle+1}/{args.coupled_training_cycles} ---")
            
            for step in range(args.vmc_iterations_per_cycle):
                key, k_psi, k_samp = jax.random.split(key, 3)
                params_psi, opt_state_psi, e_mean, e_var, cache = trainer_vmc.step_ansatz(walkers, params_psi, opt_state_psi)
                walkers, stats, cache, _ = sampler.step(k_samp, walkers, {"ansatz": params_psi, "actor": p_act}, cache, exploration_mode=False)

                stats["mean_reward"] = last_td3_reward
                stats["Actor_Loss"] = last_td3_actor_loss
                stats["Critic1_Loss"] = last_td3_critic1_loss
                stats["Critic2_Loss"] = last_td3_critic2_loss

                if global_step > args.steps_for_best and e_mean < e_best:
                    e_best = float(e_mean); best_params_ansatz = params_psi

                if global_step % args.n_log == 0:
                    all_stats = {"Step": global_step, "Time(s)": float(time.time() - cycle_start_time), "E_mean": float(e_mean), "Stage": "VMC"}
                    all_stats.update(stats)
                    log_data = {k: (v.item() if hasattr(v, "item") else v) for k, v in all_stats.items() if not (hasattr(v, "ndim") and v.ndim > 0)}
                    training_history.append(log_data)
                    if global_step % args.n_print == 0: print(format_metrics(log_data))
                global_step += 1

            frozen_wf = jax.tree_util.tree_map(jax.lax.stop_gradient, params_psi)
            curr_logp, curr_grad = td3.get_logp_and_grad(frozen_wf, walkers)
            cycle_rewards = []
            cycle_actor_loss, cycle_critic1_loss, cycle_critic2_loss = [], [], []
            
            for sweep in range(args.rl_sweeps_per_cycle):
                key, k_td3, k_train = jax.random.split(key, 3)
                exp_std = 0.2 * (1.0 - sweep / max(1, args.rl_sweeps_per_cycle))
                for i in range(args.n_elec):
                    walkers, curr_logp, curr_grad, s, a, r, ns, acc, disp = td3.td3_update_one_electron(k_td3, walkers, curr_logp, curr_grad, frozen_wf, p_act, i, exp_std)
                    replay.add_batch(s, a, r, ns)
                    cycle_rewards.append(float(jnp.mean(r)))
                    
                    # Valores "dummy" en caso de que el buffer no esté lleno
                    l1, l2, l_a = 0.0, 0.0, 0.0
                    update_act = False

                    if replay.size > args.batch_size:
                        batch = replay.sample(args.batch_size)
                        update_act = (td3_updates % 2 == 0)
                        p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_a = td3.train_networks(
                            k_train, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_act)
                        td3_updates += 1
                        
                    # Guardamos siempre (será 0 si no se entrenó, lo cual mantiene la longitud de las listas correcta)
                    cycle_critic1_loss.append(float(l1))
                    cycle_critic2_loss.append(float(l2))
                    if update_act: 
                        cycle_actor_loss.append(float(l_a))

                if sweep % max(1, args.n_print // 10) == 0:
                    print(f"TD3 Sweep {sweep:04d} | Recompensa = {float(jnp.mean(r)):.4f} | Aceptacion = {float(jnp.mean(acc)):.3f}")

            last_td3_reward = np.mean(cycle_rewards) if cycle_rewards else 0.0
            last_td3_critic1_loss = np.mean(cycle_critic1_loss) if cycle_critic1_loss else 0.0
            last_td3_critic2_loss = np.mean(cycle_critic2_loss) if cycle_critic2_loss else 0.0
            last_td3_actor_loss = np.mean(cycle_actor_loss) if cycle_actor_loss else 0.0

    # =========================================================
    # MEDICIÓN FINAL Y EXPORTACIÓN DE DATOS
    # =========================================================
    print(f"\n--- MIDIENDO ENERGÍA FINAL ({args.measure_steps} Pasos) ---")
    cache = {"log_psi_curr": ansatz.apply(best_params_ansatz, walkers), "grad_curr": jax.grad(sum_log_psi)(walkers)}

    eval_params_actor = p_act if args.method == "TD3" else None
    el_hist, acc_hist = compiled_measure_phase(sampler, walkers, key, best_params_ansatz, eval_params_actor, cache, hamiltonian, args.measure_steps)
    
    with open(os.path.join(out_dir, f"{args.method}_ansatz_weights.pkl"), "wb") as f: pickle.dump(best_params_ansatz, f)
    if args.method == "TD3":
        with open(os.path.join(out_dir, f"{args.method}_actor_weights.pkl"), "wb") as f: pickle.dump(p_act, f)
        with open(os.path.join(out_dir, "TD3_critic1_weights.pkl"), "wb") as f: pickle.dump(p_c1, f)
        with open(os.path.join(out_dir, "TD3_critic2_weights.pkl"), "wb") as f: pickle.dump(p_c2, f)

    print(f"-> Energía Final Promedio: {float(jnp.mean(jnp.array(el_hist))):.5f} Ha")
    pd.DataFrame(training_history).to_csv(os.path.join(out_dir, "training_metrics.csv"), index=False)
    pd.DataFrame({"Step": np.arange(args.measure_steps), "E_Local_Mean": np.mean(np.array(el_hist), axis=1), "Acceptance": np.array(acc_hist)}).to_csv(os.path.join(out_dir, "measurement_data.csv"), index=False)
    print(f"Tiempo Total: {(time.time() - global_start_time)/60:.2f} minutos")

if __name__ == "__main__":
    main()