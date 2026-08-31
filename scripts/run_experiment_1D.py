import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys
import time
import argparse
import datetime
import pickle
import json
import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
import optax

from src.system import Harmonic_1D
from src.hamiltonians import Harmonic1DHamiltonian
from src.wavefunctions import Ansatz1D_Huziel
from src.actors import Actor_Huziel
from src.samplers import RLMALASampler
from src.trainers import GroundStateTrainer_new
from src.aux_functions import calculate_iat, compiled_measure_phase, format_metrics
from src.rl_train import train_rl_sampler
from src.rl import RLConfig
from src.physics import SystemConfig

class _MockCfg:
    def __init__(self, name=None, model=None):
        self.name = name
        self.model = model

def main():
    parser = argparse.ArgumentParser()
    
    # RL method parameter
    parser.add_argument("--rl_method", type=str, default="REINFORCE", choices=["REINFORCE", "TD3"])
    
    # physical parameters
    parser.add_argument("--n_elec", type=int, default=4)
    parser.add_argument("--n_up", type=int, default=2)
    parser.add_argument("--n_dn", type=int, default=2)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--interaction_strength", type=float, default=1.0)
    parser.add_argument("--softening", type=float, default=0.1)
    
    # ansatz conf
    parser.add_argument("--wf_model", type=str, default="slater_jastrow")
    parser.add_argument("--orbital_model", type=str, default="ho_variational")
    parser.add_argument("--correlation_model", type=str, default="spin_rbf_jastrow")
    
    # general samplers and actors parameters
    parser.add_argument("--actor_name", type=str, default="AdvisorDeepSetActor")
    parser.add_argument("--actor_amplitude", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--max_phys_jump_std", type=float, default=5.0)
    
    # REINFORCE parameters
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    
    #  TD3 parameters
    parser.add_argument("--coupled_training_cycles", type=int, default=4)
    parser.add_argument("--vmc_iterations_per_cycle", type=int, default=1250)
    parser.add_argument("--rl_sweeps_per_cycle", type=int, default=100)
    parser.add_argument("--rl_replay_capacity", type=int, default=100000)
    parser.add_argument("--rl_batch_size", type=int, default=256)
    parser.add_argument("--rl_gamma_td3", type=float, default=0.99)
    
    # optimizers
    parser.add_argument("--lr_ansatz", type=float, default=5e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-3)
    parser.add_argument("--trans_steps_sched_ansatz", type=int, default=1000)
    parser.add_argument("--decay_rate_ansatz", type=float, default=0.5)
    parser.add_argument("--trans_steps_sched_actor", type=int, default=1000)
    parser.add_argument("--decay_rate_actor", type=float, default=0.5)
    parser.add_argument("--actor_warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    
    # stability and clipping
    parser.add_argument("--energy_clip_min", type=float, default=-100.)
    parser.add_argument("--energy_clip_max", type=float, default=100.)
    parser.add_argument("--sigma_energy_clip", type=float, default=5.0)
    parser.add_argument("--min_clip_grads_ansatz", type=float, default=-50.)
    parser.add_argument("--max_clip_grads_ansatz", type=float, default=50.)
    parser.add_argument("--min_clip_log_ratio", type=float, default=-50.)
    parser.add_argument("--max_clip_log_ratio", type=float, default=50.)
    parser.add_argument("--nan_clip_log_ratio", type=float, default=-50.)
    parser.add_argument("--posinf_clip_log_ratio", type=float, default=50.)
    parser.add_argument("--neginf_clip_log_ratio", type=float, default=-50.)
    parser.add_argument("--scale_clip_min", type=float, default=-5.0)
    parser.add_argument("--scale_clip_max", type=float, default=1.0)
    parser.add_argument("--penalty_factor", type=float, default=0.1)
    
    # printing
    parser.add_argument("--measure_steps", type=int, default=2000)
    parser.add_argument("--n_print", type=int, default=100)
    parser.add_argument("--n_log", type=int, default=50)
    parser.add_argument("--steps_for_best", type=int, default=500)
    parser.add_argument("--results_out_dir", type=str, required=True)

    args = parser.parse_args()
    
    out_dir = args.results_out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
        
    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    system = Harmonic_1D(
        name="Harmonic1D", n_elec=args.n_elec, n_up=args.n_up, n_dn=args.n_dn, 
        omega=args.omega, interaction_strength=args.interaction_strength, 
        softening=args.softening, dim=1
    )
    hamiltonian = Harmonic1DHamiltonian(system)
    
    wf_cfg = _MockCfg(name=args.wf_model)
    orbital_cfg = _MockCfg(model=args.orbital_model)
    correlation_cfg = _MockCfg(model=args.correlation_model)
    ansatz = Ansatz1D_Huziel(system, wf_cfg, orbital_cfg, correlation_cfg)

    walkers_init = jax.random.normal(k_init, (args.batch_size, args.n_elec, 1))
    params_psi_master = ansatz.init(k_init, walkers_init)
    walkers = jnp.copy(walkers_init)
    params_psi = jax.tree.map(lambda x: jnp.copy(x), params_psi_master)

    sched_ansatz = optax.exponential_decay(init_value=args.lr_ansatz, transition_steps=args.trans_steps_sched_ansatz, decay_rate=args.decay_rate_ansatz)
    opt_ansatz = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=sched_ansatz))
    opt_state_psi = opt_ansatz.init(params_psi)

    actor = Actor_Huziel(n_elec=args.n_elec, n_up=args.n_up, n_dn=args.n_dn, actor_amplitude=args.actor_amplitude)
    key, k_act = jax.random.split(key)
    params_actor = actor.init(k_act, walkers, 0, system.coords, True, jnp.zeros_like(walkers))

    sampler = RLMALASampler(
        ansatz_apply=ansatz.apply, system=system, batch_size=args.batch_size,
        max_clip_log_ratio=args.max_clip_log_ratio, min_clip_log_ratio=args.min_clip_log_ratio,
        nan_clip_log_ratio=args.nan_clip_log_ratio, neginf_clip_log_ratio=args.neginf_clip_log_ratio,
        posinf_clip_log_ratio=args.posinf_clip_log_ratio, actor_apply=actor.apply,
        tau=args.tau, gamma=args.gamma, beta=args.beta, kappa=args.kappa,
        scale_clip_min=args.scale_clip_min, scale_clip_max=args.scale_clip_max,
        min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz,
        penalty_factor=args.penalty_factor
    )

    # Configuración condicional según el método
    if args.rl_method == "REINFORCE":
        sched_actor = optax.join_schedules(
            schedules=[optax.constant_schedule(0.0), optax.exponential_decay(init_value=args.lr_actor, transition_steps=args.trans_steps_sched_actor, decay_rate=args.decay_rate_actor)],
            boundaries=[args.actor_warmup_steps]
        )
        opt_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=sched_actor))
        opt_state_actor = opt_actor.init(params_actor)
        trainer = GroundStateTrainer_new(system=system, hamiltonian=hamiltonian, sampler=sampler, opt_ansatz=opt_ansatz, opt_actor=opt_actor, sigma_energy_clip=args.sigma_energy_clip, energy_clip_min=args.energy_clip_min, energy_clip_max=args.energy_clip_max, min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_std)
    
    elif args.rl_method == "TD3":
        # En TD3 el trainer no optimiza al actor directamente
        trainer = GroundStateTrainer_new(system=system, hamiltonian=hamiltonian, sampler=sampler, opt_ansatz=opt_ansatz, opt_actor=None, sigma_energy_clip=args.sigma_energy_clip, energy_clip_min=args.energy_clip_min, energy_clip_max=args.energy_clip_max, min_clip_grads_ansatz=args.min_clip_grads_ansatz, max_clip_grads_ansatz=args.max_clip_grads_ansatz, max_phys_jump_std=args.max_phys_jump_std)
        
        rl_cfg = RLConfig(
            mode="residual_mala", actor_architecture="multibranch_deepset", base_step_size=args.tau,
            actor_learning_rate=args.lr_actor, critic_learning_rate=3e-4, batch_size=args.rl_batch_size,
            replay_capacity=args.rl_replay_capacity, gamma=args.rl_gamma_td3,
            reward_displacement_weight=args.beta, reward_acceptance_weight=args.gamma, seed=args.seed
        )
        adv_system_cfg = SystemConfig(n_electrons=args.n_elec, n_up=args.n_up, omega=args.omega, interaction_strength=args.interaction_strength, softening=args.softening)
        
        actor_target_params = None
        actor_opt_state = None
        rl_history = None

    def sum_log_psi(w):
        return jnp.sum(ansatz.apply(params_psi, w))

    cache = {
        "log_psi_curr": ansatz.apply(params_psi, walkers),
        "grad_curr": jnp.clip(jax.grad(sum_log_psi)(walkers), args.min_clip_grads_ansatz, args.max_clip_grads_ansatz)
    }

    # Warmup Inicial (Aplica para ambos métodos)
    for _ in range(args.warmup_steps):
        key, key_warm = jax.random.split(key)
        walkers, stats, cache, _ = sampler.step(key_warm, walkers, {"ansatz": params_psi, "actor": params_actor}, cache, exploration_mode=True)

    training_history = []
    e_best = jnp.inf
    best_params_actor = params_actor
    best_params_ansatz = params_psi
    global_step = 0

    # =========================================================================
    # LÓGICA DE ENTRENAMIENTO: REINFORCE vs TD3
    # =========================================================================
    if args.rl_method == "REINFORCE":
        for step in range(args.train_steps):
            key, k_psi, k_act = jax.random.split(key, 3)

            params_psi, opt_state_psi, e_mean, e_var, cache = trainer.step_ansatz(walkers, params_psi, opt_state_psi)
            params_actor, opt_state_actor, walkers, stats, loss_act, cache = trainer.step_actor(k_act, walkers, params_psi, params_actor, opt_state_actor, cache)

            if step > args.steps_for_best and e_mean < e_best:
                e_best = float(e_mean)
                best_params_actor = params_actor
                best_params_ansatz = params_psi

            if step % args.n_log == 0 or step == args.train_steps - 1:
                all_stats = {"Step": step, "E_mean": float(e_mean), "E_Var": float(e_var), "Actor_Loss": float(loss_act)}
                all_stats.update(stats)
                log_data = {k: (v.item() if hasattr(v, "item") else v) for k, v in all_stats.items() if not (hasattr(v, "ndim") and v.ndim > 0)}
                training_history.append(log_data)

                if step % args.n_print == 0:
                    print(format_metrics(log_data))

    elif args.rl_method == "TD3":
        for cycle in range(args.coupled_training_cycles):
            # 1. BLOQUE VMC (Entrenar Ansatz, Actor Congelado)
            for step in range(args.vmc_iterations_per_cycle):
                key, k_psi, k_samp = jax.random.split(key, 3)
                params_psi, opt_state_psi, e_mean, e_var, cache = trainer.step_ansatz(walkers, params_psi, opt_state_psi)
                walkers, stats, cache, _ = sampler.step(k_samp, walkers, {"ansatz": params_psi, "actor": params_actor}, cache, exploration_mode=False)

                if global_step > args.steps_for_best and e_mean < e_best:
                    e_best = float(e_mean)
                    best_params_actor = params_actor
                    best_params_ansatz = params_psi

                if global_step % args.n_log == 0:
                    all_stats = {"Step": global_step, "Cycle": cycle, "E_mean": float(e_mean), "E_Var": float(e_var), "Stage": "VMC"}
                    all_stats.update(stats)
                    log_data = {k: (v.item() if hasattr(v, "item") else v) for k, v in all_stats.items() if not (hasattr(v, "ndim") and v.ndim > 0)}
                    training_history.append(log_data)
                    
                    if global_step % args.n_print == 0:
                        print(format_metrics(log_data))
                global_step += 1

            # 2. BLOQUE TD3 (Entrenar Actor, Ansatz Congelado)
            frozen_wf = jax.tree_util.tree_map(jax.lax.stop_gradient, params_psi)
            kr = jax.random.fold_in(key, 424242 + cycle)
            
            rl_result = train_rl_sampler(
                key=kr, walkers=walkers, wf_params=frozen_wf, system_cfg=adv_system_cfg,
                wf_cfg=wf_cfg, orbital_cfg=orbital_cfg, correlation_cfg=correlation_cfg, rl_cfg=rl_cfg,
                print_every=args.n_print, actor_params=params_actor, actor_target_params=actor_target_params,
                actor_optimizer_state=actor_opt_state, history=rl_history, sweep_offset=cycle * args.rl_sweeps_per_cycle
            )
            
            params_actor = rl_result["actor_params"]
            actor_target_params = rl_result["actor_target_params"]
            actor_opt_state = rl_result["actor_optimizer_state"]
            rl_history = rl_result["history"]
            walkers = rl_result["walkers"]

            # Actualizamos el mejor actor si TD3 mejoró la política
            best_params_actor = params_actor

    # =========================================================================
    # MEDICIÓN FINAL
    # =========================================================================
    el_hist, acc_hist = compiled_measure_phase(
        sampler, walkers, key, best_params_ansatz, best_params_actor, cache, hamiltonian, args.measure_steps
    )
    e_time_series = np.mean(np.array(el_hist), axis=1)

    with open(os.path.join(out_dir, f"{args.rl_method}_ansatz_weights.pkl"), "wb") as f:
        pickle.dump(best_params_ansatz, f)
    with open(os.path.join(out_dir, f"{args.rl_method}_actor_weights.pkl"), "wb") as f:
        pickle.dump(best_params_actor, f)
        
    pd.DataFrame(training_history).to_csv(os.path.join(out_dir, "training_metrics.csv"), index=False)
    pd.DataFrame({"Step": np.arange(args.measure_steps), "E_Local_Mean": e_time_series, "Acceptance": np.array(acc_hist)}).to_csv(os.path.join(out_dir, "measurement_data.csv"), index=False)

if __name__ == "__main__":
    main()