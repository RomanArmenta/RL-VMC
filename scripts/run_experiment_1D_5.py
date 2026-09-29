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
from src.td3_modules import TD3Trainer, ReplayBuffer, FlaxTD3Actor

@partial(jax.jit, static_argnums=(0, 6))
def termalizar_caminantes(sampler, w_start, k_start, cache_start, p_psi, p_act, n_steps):
    """Evoluciona los caminantes sin acumular gradientes ni historial para alcanzar el estado base estacionario."""
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
    print("Iniciando RL-VMC (Staggered Transfer Learning) - Version 4 (Fully Parameterized)")

    parser = argparse.ArgumentParser(description="Pipeline de optimización VMC guiado por Reinforcement Learning (TD3)")
    
    # ==============================================================================
    # 1. PARÁMETROS PRINCIPALES
    # ==============================================================================
    parser.add_argument("--method", type=str, default="MALA", choices=["MALA", "TD3"], help="Muestreador a utilizar: MALA clásico (Langevin) o híbrido RL-MALA (TD3).")
    
    # Parámetros del Sistema Físico
    parser.add_argument("--n_elec", type=int, default=4, help="Número total de electrones en el sistema.")
    parser.add_argument("--n_up", type=int, default=2, help="Número de electrones con espín hacia arriba (up).")
    parser.add_argument("--n_dn", type=int, default=2, help="Número de electrones con espín hacia abajo (down).")
    parser.add_argument("--omega", type=float, default=0.1, help="Frecuencia angular del pozo de potencial armónico confinante.")
    parser.add_argument("--interaction_strength", type=float, default=1.0, help="Fuerza de la interacción repulsiva electrón-electrón (g).")
    parser.add_argument("--softening", type=float, default=0.1, help="Parámetro de suavizado (soft-Coulomb) para evitar singularidades en interacciones 1D.")
    
    # ==============================================================================
    # 2. ARQUITECTURA DEL ANSATZ (Función de Onda)
    # ==============================================================================
    parser.add_argument("--wf_model", type=str, default="slater_jastrow", help="Modelo base de la función de onda.")
    parser.add_argument("--orbital_model", type=str, default="backflow_ho", help="Modelo de orbitales (ej. oscilador armónico con backflow).")
    parser.add_argument("--correlation_model", type=str, default="spin_rbf_jastrow", help="Modelo de correlación (Factor de Jastrow basado en RBF).")
    parser.add_argument("--backflow_components", type=int, default=8, help="Número de componentes/canales de la transformación Backflow.")
    parser.add_argument("--backflow_r_max", type=float, default=5.0, help="Radio de corte máximo para las interacciones en el Backflow.")
    parser.add_argument("--backflow_width", type=float, default=0.7, help="Anchura inicial de las funciones de base radial en el Backflow.")
    parser.add_argument("--backflow_envelope_length", type=float, default=3.0, help="Longitud característica del decaimiento exponencial (envolvente) del Backflow.")
    
    # ==============================================================================
    # 3. MUESTREO Y REINFORCEMENT LEARNING
    # ==============================================================================
    parser.add_argument("--batch_size", type=int, default=2048, help="Número de caminantes (walkers) de Monte Carlo paralelos.")
    parser.add_argument("--tau", type=float, default=0.01, help="Tamaño de paso temporal discreto para la dinámica de Langevin (MALA).")
    parser.add_argument("--actor_amplitude", type=float, default=0.15, help="Escala de magnitud permitida para la corrección direccional (drif_res) del Actor.")
    parser.add_argument("--gamma", type=float, default=0.1, help="Ponderación histórica para las recompensas (si aplica) o hiperparámetro auxiliar RL.")
    parser.add_argument("--beta", type=float, default=10.0, help="Factor de temperatura inversa u optimización secundaria en métricas RL.")
    parser.add_argument("--kappa", type=float, default=0.5, help="Factor de escala para el término cuadrático de aceptación en recompensas alternativas.")
    parser.add_argument("--penalty_factor", type=float, default=0.1, help="Coeficiente lambda_ que pondera la penalización por entropía y baja aceptación en CDLB.")
    parser.add_argument("--max_phys_jump_factor", type=float, default=5.0, help="Multiplicador estandarizado para acotar el salto físico máximo permitido (Clipping de EJD).")
    
    # Nuevos Parámetros Dinámicos para el Estado (DeepSet Features)
    parser.add_argument("--force_scale", type=float, default=5.0, help="Constante de normalización para la fuerza cuántica de entrada en la red neuronal.")
    parser.add_argument("--n_scale", type=float, default=10.0, help="Constante de normalización para el conteo de partículas en el vector global.")
    parser.add_argument("--r_scale", type=float, default=5.0, help="Radio de escalado para distancias relativas inter-partículas en los slots DeepSet.")
    
    # Nuevos Límites de Acción del Actor
    parser.add_argument("--sigma_action_scale", type=float, default=0.2, help="Escala máxima permitida para la salida tanh que modifica la varianza isotrópica.")
    parser.add_argument("--max_drift_clip", type=float, default=10.0, help="Recorte absoluto de la deriva cruda (Langevin drift) para prevenir inestabilidad numérica.")
    parser.add_argument("--action_rev_limit", type=float, default=1.0, help="Límite estricto de recorte para la salida del Actor al evaluar la transición inversa (q_rev).")

    # ==============================================================================
    # 4. OPTIMIZADORES Y RUTINAS DE ENTRENAMIENTO
    # ==============================================================================
    parser.add_argument("--lr_ansatz", type=float, default=5e-4, help="Tasa de aprendizaje (Adam) para los parámetros de la función de onda (VMC).")
    parser.add_argument("--lr_actor", type=float, default=1e-3, help="Tasa de aprendizaje para la red Actor (TD3).")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Tasa de aprendizaje para las redes Críticas (TD3).")
    
    parser.add_argument("--train_steps", type=int, default=5000, help="Número de iteraciones totales si se usa MALA puro.")
    parser.add_argument("--measure_steps", type=int, default=2000, help="Iteraciones dedicadas puramente a medir energía sin acumular gradientes.")
    parser.add_argument("--n_ter", type=int, default=2000, help="Pasos de termalización (burn-in) obligatorios previos a cualquier optimización.")
    
    parser.add_argument("--coupled_training_cycles", type=int, default=8, help="Número de macro-ciclos alternando VMC y RL.")
    parser.add_argument("--vmc_iterations_per_cycle", type=int, default=1250, help="Pasos de optimización VMC puros en cada ciclo acoplado.")
    parser.add_argument("--rl_sweeps_per_cycle", type=int, default=100, help="Épocas de entrenamiento Bellman del agente TD3 sobre el Replay Buffer.")
    parser.add_argument("--rl_replay_capacity", type=int, default=100000, help="Capacidad máxima del Replay Buffer (transiciones almacenadas).")
    parser.add_argument("--rl_gamma_td3", type=float, default=0.99, help="Factor de descuento gamma para la Ecuación de Bellman (TD3 target).")
    
    # ==============================================================================
    # 5. ESTABILIDAD NUMÉRICA (CLIPPING)
    # ==============================================================================
    parser.add_argument("--energy_clip_min", type=float, default=-500.0, help="Límite inferior absoluto para la energía local observada.")
    parser.add_argument("--energy_clip_max", type=float, default=500.0, help="Límite superior absoluto para la energía local observada.")
    parser.add_argument("--sigma_energy_clip", type=float, default=5.0, help="Desviaciones estándar a partir de la mediana empírica para recortar energías extremas.")
    parser.add_argument("--min_clip_grads_ansatz", type=float, default=-500.0, help="Recorte mínimo para el gradiente algorítmico del Ansatz.")
    parser.add_argument("--max_clip_grads_ansatz", type=float, default=500.0, help="Recorte máximo para el gradiente algorítmico del Ansatz.")
    parser.add_argument("--min_clip_log_ratio", type=float, default=-50.0, help="Cota inferior del ratio de probabilidad asimétrica (Langevin Q ratio).")
    parser.add_argument("--max_clip_log_ratio", type=float, default=50.0, help="Cota superior del ratio de probabilidad asimétrica (Langevin Q ratio).")
    parser.add_argument("--nan_clip_log_ratio", type=float, default=-50.0, help="Valor seguro de reemplazo en caso de NaNs en el ratio de aceptación.")
    parser.add_argument("--posinf_clip_log_ratio", type=float, default=50.0, help="Valor seguro de reemplazo en caso de infinito positivo en el ratio.")
    parser.add_argument("--neginf_clip_log_ratio", type=float, default=-50.0, help="Valor seguro de reemplazo en caso de infinito negativo en el ratio.")
    parser.add_argument("--scale_clip_min", type=float, default=-5.0, help="Recorte numérico mínimo para la escala logarítmica de la varianza.")
    parser.add_argument("--scale_clip_max", type=float, default=1.0, help="Recorte numérico máximo para la escala logarítmica de la varianza.")
    
    # Tracking e I/O
    parser.add_argument("--seed", type=int, default=42, help="Semilla base aleatoria para inicializar pesos, ruido y minibatches.")
    parser.add_argument("--n_print", type=int, default=100, help="Frecuencia (pasos) para imprimir métricas en la salida estándar.")
    parser.add_argument("--n_log", type=int, default=50, help="Frecuencia (pasos) para registrar métricas en el CSV de entrenamiento.")
    parser.add_argument("--steps_for_best", type=int, default=500, help="Pasos de gracia (ignorar transitorios) antes de empezar a guardar los mejores pesos.")
    parser.add_argument("--results_out_dir", type=str, required=True, help="Directorio de exportación para datos, JSON y pesos pkl.")
    
    args, _ = parser.parse_known_args()
    
    out_dir = args.results_out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.json"), "w") as f: 
        json.dump(vars(args), f, indent=4)
        
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

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
    # EJECUCIÓN MALA CLÁSICO
    # =========================================================
    if args.method == "MALA":
        sampler = MALASampler(
            ansatz.apply, args.batch_size, args.tau, system, args.max_clip_grads_ansatz, args.min_clip_grads_ansatz, 
            args.max_clip_log_ratio, args.min_clip_log_ratio, args.nan_clip_log_ratio, args.neginf_clip_log_ratio, args.posinf_clip_log_ratio
        )
        trainer = GroundStateTrainer_new(
            system=system, hamiltonian=hamiltonian, sampler=sampler, opt_ansatz=opt_ansatz, opt_actor=None,
            sigma_energy_clip=args.sigma_energy_clip, energy_clip_min=args.energy_clip_min, energy_clip_max=args.energy_clip_max, 
            min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_factor
        )

        print(f"\nTermalizando caminantes MALA por {args.n_ter} pasos...")
        final_carry = termalizar_caminantes(sampler, walkers, key, cache, params_psi, None, args.n_ter)
        walkers, key, cache = final_carry
        print("Termalización completada. Iniciando entrenamiento...")

        step_start_time = time.time()
        for step in range(args.train_steps):
            key, k_psi, k_samp = jax.random.split(key, 3)
            params_psi, opt_state_psi, e_mean, e_var, cache = trainer.step_ansatz(walkers, params_psi, opt_state_psi)
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
        # Inyectando nuevos parámetros a la instanciación de TD3Trainer
        td3 = TD3Trainer(
            ansatz.apply, args.n_elec, args.n_up, args.n_dn, args.tau, args.actor_amplitude, 
            beta=args.beta, gamma_reward=args.gamma, gamma_td3=args.rl_gamma_td3, 
            drift_limit=args.max_drift_clip, max_phys_jump_factor=args.max_phys_jump_factor, action_rev_limit=args.action_rev_limit,
            lambda_=args.penalty_factor, sigma_action_scale=args.sigma_action_scale if hasattr(FlaxTD3Actor, 'sigma_action_scale') else 0.2
        )
        
        def td3_actor_wrapper(params_actor, w, e_idx, r_nuc, exp_mode, grad):
            from src.td3_modules import build_td3_state
            w_1d, grad_1d = w[..., 0], grad[..., 0] 
            
            # Inyectando la nueva configuración física dinámica a build_td3_state
            state = build_td3_state(
                w_1d, grad_1d, e_idx, args.n_elec, args.n_up, args.n_dn,
                quantum_force_norm=args.force_scale, particle_factor=args.n_scale,
                dist_factor=args.r_scale, omega=args.omega, gs=args.interaction_strength, softening=args.softening
            )
            
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
            min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_factor
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
        
        print(f"\nTermalizando caminantes TD3 por {args.n_ter} pasos...")
        final_carry = termalizar_caminantes(sampler, walkers, key, cache, params_psi, p_act, args.n_ter)
        walkers, key, cache = final_carry
        print("Termalización completada. Iniciando entrenamiento acoplado...")

        for cycle in range(args.coupled_training_cycles):
            cycle_start_time = time.time()
            print(f"\n--- CICLO TD3 {cycle+1}/{args.coupled_training_cycles} ---")
            
            # uncomment this shi if you want a staggered transfer learning in critics.
            # if cycle > 0:
            #     key, k_reset = jax.random.split(key)
            #     fresh_params = td3.critic_model.init(k_reset, dummy_state, dummy_action)
                
            #     if cycle % 2 == 1:
            #         print("-> Staggered Transfer: Reinicializando Crítico 1. Crítico 2 mantiene memoria.")
            #         p_c1 = fresh_params
            #         t_c1 = jax.tree_util.tree_map(lambda x: x.copy(), p_c1)
            #         opt_c1 = td3.opt_c.init(p_c1)
            #     else:
            #         print("-> Staggered Transfer: Reinicializando Crítico 2. Crítico 1 mantiene memoria.")
            #         p_c2 = fresh_params
            #         t_c2 = jax.tree_util.tree_map(lambda x: x.copy(), p_c2)
            #         opt_c2 = td3.opt_c.init(p_c2)
            
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

                # Eliminado: for i in range(args.n_elec):
                walkers, curr_logp, curr_grad, s, a, r, ns, acc, disp = td3.td3_update_all_electrons(
                    k_td3, walkers, curr_logp, curr_grad, frozen_wf, p_act, exp_std
                )
                
                # Desacoplar dimensión temporal (batch * n_elec) para el Replay Buffer
                s_flat = s.reshape(-1, s.shape[-1])
                a_flat = a.reshape(-1, a.shape[-1])
                r_flat = jnp.repeat(r, args.n_elec)
                ns_flat = ns.reshape(-1, ns.shape[-1])
                
                replay.add_batch(s_flat, a_flat, r_flat, ns_flat)
                cycle_rewards.append(float(jnp.mean(r)))

                # for i in range(args.n_elec):
                #     walkers, curr_logp, curr_grad, s, a, r, ns, acc, disp = td3.td3_update_one_electron(k_td3, walkers, curr_logp, curr_grad, frozen_wf, p_act, i, exp_std)
                #     replay.add_batch(s, a, r, ns)
                #     cycle_rewards.append(float(jnp.mean(r)))
                    
                #     l1, l2, l_a = 0.0, 0.0, 0.0
                #     update_act = False

                #     if replay.size > args.batch_size:
                #         batch = replay.sample(args.batch_size)
                #         update_act = (td3_updates % 2 == 0)
                #         p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, l1, l2, l_a = td3.train_networks(
                #             k_train, p_act, p_c1, p_c2, t_act, t_c1, t_c2, opt_act, opt_c1, opt_c2, batch, update_act)
                #         td3_updates += 1
                        
                #     cycle_critic1_loss.append(float(l1))
                #     cycle_critic2_loss.append(float(l2))
                #     if update_act: 
                #         cycle_actor_loss.append(float(l_a))

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