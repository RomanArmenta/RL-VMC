import os
os.environ["JAX_ENABLE_X64"] = "True"

import time
import argparse
import datetime
import pickle
import pandas as pd
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from functools import partial

from system import PhysicalSystem
from wavefunctions import DeepQMCAnsatz
from hamiltonians import FullCoulombHamiltonian
from samplers import MALASampler, RLMALASampler
from trainers import GroundStateTrainer
from actors import DirectionalActor
from aux_functions import calculate_iat, compiled_measure_phase

# Physical suystems to study
SYSTEMS = {
    "H": {
        "n_elec": 1, "n_up": 1, "n_dn": 0,
        "coords": [[1.0, 0.0, 0.0]],
        "charges": [1]
    },
    "H2+": {
        "n_elec": 1, "n_up": 1, "n_dn": 0,
        "coords": [[-0.52918, 0.0, 0.0], [0.52918, 0.0, 0.0]],
        "charges": [1, 1]
    },
    "H2": {
        "n_elec": 2, "n_up": 1, "n_dn": 1,
        "coords": [[0.0, 0.0, 0.0], [0.742, 0.0, 0.0]],
        "charges": [1, 1]
    },
    "He": {
        "n_elec": 2, "n_up": 1, "n_dn": 1,
        "coords": [[0.0, 0.0, 0.0]],
        "charges": [2]
    },
    "Be": {
        "n_elec": 4, "n_up": 2, "n_dn": 2,
        "coords": [[0.0, 0.0, 0.0]],
        "charges": [4]
    },
    "B": {
        "n_elec": 5, "n_up": 3, "n_dn": 2,
        "coords": [[0.0, 0.0, 0.0]],
        "charges": [5]
    },
    "C": {
        "n_elec": 6, "n_up": 4, "n_dn": 2,
        "coords": [[0.0, 0.0, 0.0]],
        "charges": [6]
    },
    "LiH": {
        "n_elec": 4, "n_up": 2, "n_dn": 2,
        "coords": [[0.0, 0.0, 0.0], [1.595, 0.0, 0.0]],
        "charges": [3, 1]
    },
    "CO2": {
        "n_elec": 22, "n_up": 11, "n_dn": 11,
        "coords": [[-1.161, 0.0, 0.0], [0.0, 0.0, 0.0], [1.161, 0.0, 0.0]],
        "charges": [8, 6, 8]
    },
    "Li2": {
        "n_elec": 6, "n_up": 3, "n_dn": 3,
        "coords": [[-1.3364, 0.0, 0.0], [1.3364, 0.0, 0.0]],
        "charges": [3, 3]
    },
    "Be2": {
        "n_elec": 8, "n_up": 4, "n_dn": 4,
        "coords": [[-1.230, 0.0, 0.0], [1.230, 0.0, 0.0]],
        "charges": [4, 4]
    },
    "B2": {
        "n_elec": 10, "n_up": 6, "n_dn": 4,
        "coords": [[-0.7951, 0.0, 0.0], [0.7951, 0.0, 0.0]],
        "charges": [5, 5]
    },
    "C2": {
        "n_elec": 12, "n_up": 6, "n_dn": 6,
        "coords": [[-0.621265, 0.0, 0.0], [0.621265, 0.0, 0.0]],
        "charges": [6, 6]
    },
    "N2": {
        "n_elec": 14, "n_up": 7, "n_dn": 7,
        "coords": [[-2.13534, 0.0, 0.0], [2.13534, 0.0, 0.0]],
        "charges": [7, 7]
    },
    "bicyclobutane": {
        "n_elec": 30, "n_up": 15, "n_dn": 15,
        "coords": [
            [0.7507, 0.0000, -0.3193],
            [-0.7507, 0.0000, -0.3193],
            [0.0000, 1.1350, 0.3153],
            [0.0000, -1.1350, 0.3153],
            [1.4194, 0.0000, -1.1631],
            [-1.4194, 0.0000, -1.1631],
            [0.0000, 2.0820, -0.2148],
            [0.0000, -2.0820, -0.2148],
            [0.0000, 1.2163, 1.4020],
            [0.0000, -1.2163, 1.4020]
        ],
        "charges": [6, 6, 6, 6, 1, 1, 1, 1, 1, 1]
    }
}

# main script
def main():
    parser = argparse.ArgumentParser(description="RL-VMC experiment runner")
    parser.add_argument("--system", type=str, required=True, help="Physical system's name (ex. LiH, Ne, Be)")
    parser.add_argument("--config_name", type=str, default="default", help="Ansatz configuration name from DeepQMC")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of parallel walkers")
    parser.add_argument("--tau", type=float, default=0.005, help="Time step (tau)for MALA")
    parser.add_argument("--lr_ansatz", type=float, default=5e-4, help="Learning rate for the ansatz (for energy minimization)")
    parser.add_argument("--lr_actor", type=float, default=1e-3, help="Learning rate for actor (reward maximization)")
    parser.add_argument("--train_steps", type=int, default=15_000, help="optimization steps")
    parser.add_argument("--warmup_steps", type=int, default=2_000, help="initial steps for the markov chain to termalize to the goal distribution")
    parser.add_argument("--measure_steps", type=int, default=5_000, help="measurement steps with trained models")
    parser.add_argument("--gamma", type=float, default=0.01, help="weight of the gradient penalty in the reward function")
    parser.add_argument("--beta", type=float, default=10.0, help="weight of the jump reward")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--trans_steps_sched_ansatz", type=int, default=10_000, help="steps needed for the ansatz lr to hit the lower value in the schedeule")
    parser.add_argument("--decay_rate_ansatz", type=float, default=0.5, help="decay rate, the 'velocity' in which the lr ansatz decay") 
    parser.add_argument("--trans_steps_sched_actor", type=int, default=10_000, help="steps needed for the actor lr to hit the lower value in the schedeule")
    parser.add_argument("--decay_rate_actor", type=float, default=0.5, help="decay rate, the 'velocity' in which the lr actor decay")
    parser.add_argument("--min_clip_grads_ansatz", type=float, default=-20., help="minimum value of the gradient for the ansatz to clip")
    parser.add_argument("--max_clip_grads_ansatz", type=float, default=20., help="maximum value of the gradient for the ansatz to clip")
    parser.add_argument("--n_print", type=int, default=500, help="each many steps the stats are printed in terminal")
    parser.add_argument("--n_log", type = int, default=50, help="each many steps the stats are logged or stored")
    parser.add_argument("--steps_for_best", type=int, default=500, help="number of training steps before we start to save the best parameters")
    args = parser.parse_args()

    # system setup
    if args.system not in SYSTEMS:
        raise ValueError(f"The physical system {args.system} is not found in the systems registry. Add it.")
    
    sys_config = SYSTEMS[args.system]
    molecule = PhysicalSystem(
        name = args.system,
        n_elec = sys_config["n_elec"],
        n_up = sys_config["n_up"],
        n_dn = sys_config["n_dn"],
        coords=jnp.array(sys_config["coords"]),
        charges=jnp.array(sys_config["charges"], dtype=jnp.int32)
    )

    # directory setup
    timestamp = datetime.datetime.now().strftime("%d%m%Y")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "results", f"{args.system}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=============================================================================================")
    print("Initializing VMC experiment...")
    print(f"Molecule: {molecule.name} | Batch: {args.batch_size} | DeepQMC model: {args.config_name}")
    print(f"Tau: {args.tau} | lr_ansatz: {args.lr_ansatz} | lr_actor: {args.lr_actor}")
    print(f"train_steps: {args.train_steps} | warmup_steps: {args.warmup_steps} | measure_steps: {args.measure_steps}")
    print(f"gamma: {args.gamma} | beta: {args.beta} | seed: {args.seed}")
    print(f"Results will be saved in: {out_dir}")
    print("=============================================================================================")

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    hamiltonian = FullCoulombHamiltonian(molecule)

    print(f"Initializing ansatz - {args.config_name}")
    ansatz = DeepQMCAnsatz(molecule, config_name=args.config_name)
    walkers_init = jax.random.normal(k_init, (args.batch_size, molecule.n_elec, 3)) * 0.5
    params_psi_master = ansatz.init(k_init, walkers_init)
    print("Ansatz instantiated")

    # we execute this methods (samplers)
    methods = ["MALA", "RL-MALA"]

    for method in methods:
        print("=============================================================================================")
        print(f"Starting training with {method} sampling method")
        print("=============================================================================================")

        walkers = jnp.copy(walkers_init) # copy for same initialization for all the methods
        params_psi = jax.tree.map(lambda x: jnp.copy(x), params_psi_master)
        params_actor = None

        # schedulers for training
        sched_ansatz = optax.exponential_decay(init_value = args.lr_ansatz, transition_steps=args.trans_steps_sched_ansatz, decay_rate=args.decay_rate_ansatz)
        opt_ansatz = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=sched_ansatz))
        opt_state_psi = opt_ansatz.init(params_psi)

        opt_state_actor = None

        # instantiate the sampelr
        if method == "RL-MALA":
            actor = DirectionalActor(n_elec = molecule.n_elec, n_nuclei=len(molecule.coords))
            key, k_act = jax.random.split(key)
            params_actor = actor.init(k_act, walkers, 0, molecule.coords)

            sched_actor = optax.exponential_decay(init_value=args.lr_actor, transition_steps=args.trans_steps_sched_actor, decay_rate=args.decay_rate_actor)
            opt_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=sched_actor))
            opt_state_actor = opt_actor.init(params_actor)

            sampler=RLMALASampler(ansatz.apply, actor.apply, args.batch_size, args.tau, args.gamma, args.beta, molecule)
        else:
            sampler = MALASampler(ansatz.apply, args.batch_size, args.tau, molecule)
            opt_actor = None

        trainer = GroundStateTrainer(molecule, hamiltonian, sampler, opt_ansatz, opt_actor=opt_actor)

        # init cache
        def sum_log_psi(w):
            return jnp.sum(ansatz.apply(params_psi, w))

        init_cache = {
            "log_psi_curr": ansatz.apply(params_psi, walkers),
            "grad_curr": jnp.clip(jax.grad(sum_log_psi)(walkers), args.min_clip_grads_ansatz, args.max_clip_grads_ansatz)
        }
        cache = init_cache

        # termalization
        print(f"Warm up the markov chain for {args.warmup_steps} steps...")
        for _ in range(args.warmup_steps):
            key, key_warm = jax.random.split(key)
            params_dict = {"ansatz": params_psi, "actor": params_actor}
            walkers, stats, cache, _ = sampler.step(key_warm, walkers, params_dict, cache, exploration_mode = True)

        print(f"Warm up ended with acc: {stats["acceptance"]:.2f}")

        # training loop
        training_history = []
        t_start = time.time()

        e_best = jnp.inf

        for step in range(args.train_steps):
            key, k_psi, k_act = jax.random.split(key, 3)

            # energy minimization
            params_psi, opt_state_psi, e_mean, e_var, cache = trainer.step_ansatz(
                walkers, params_psi, opt_state_psi
            )

            # RL optimization
            if method == "RL-MALA":
                params_actor, opt_state_actor, walkers, stats, loss_act, cache = trainer.step_actor(
                    k_act, walkers, params_psi, params_actor, opt_state_actor, cache
                )
            
            else:
                params_dict = {"ansatz": params_psi, "actor": None}
                walkers, stats, cache, _ = sampler.step(k_act, walkers, params_dict, cache, exploration_mode=False)

            if step > args.steps_for_best and e_best > float(e_mean):
                e_best = float(e_mean)
                best_params_actor = params_actor
                best_params_ansatz = params_psi

            # logging
            if step % args.n_log == 0 or step == args.train_steps - 1:
                log_data = {
                    "Step": step,
                    "E_mean": float(e_mean),
                    "E_Var": float(e_var),
                    "Acceptance": float(stats.get("acceptance", 0.0)),
                    "EJD": float(stats.get("ejd", 0.0)),
                    "Reward": float(stats.get("mean_reward", 0.0)) if method=="RL-MALA" else 0.0
                }
                training_history.append(log_data)

                if step % args.n_print == 0:
                    print(f"Step: {step:05d} | E: {e_mean:0.4f} Ha | Var: {e_var:6.2f} | Acc: {stats['acceptance']:.2f} | EJD: {stats['ejd']:.4f} | Best E: {e_best:.4f} Ha")


        print(f"Training ended in {time.time() - t_start:.1f} seconds.")

        # measure phase
        print(f"measurement phase: {args.measure_steps} steps")
        el_hist, acc_hist = compiled_measure_phase(
            sampler, walkers, key, params_psi, params_actor, cache, hamiltonian, args.measure_steps
        )

        e_time_series = np.mean(np.array(el_hist), axis=1) # mean energy of batch for each step
        tau_int = calculate_iat(e_time_series)
        N_eff = len(e_time_series) / (2.0 * tau_int)
        standard_error = np.std(e_time_series, ddof=1) / np.sqrt(N_eff)

        final_E = float(np.mean(e_time_series))

        print(f"Final results for[{method}]:")
        print(f" -> Energy: {final_E:.5f} ± {standard_error:.5f} Ha")
        print(f" -> Tau_int: {tau_int:.2f} steps (N_eff = {N_eff:.1f})")

        # store data 
        # parameters

        with open(os.path.join(out_dir, f"{method}_ansatz_weights.pkl"), "wb") as f: pickle.dump(params_psi, f)
        if params_actor:
            with open(os.path.join(out_dir, f"{method}_actor_weights.pkl"), "wb") as f: pickle.dump(params_actor, f)
            
        # training history
        df_train = pd.DataFrame(training_history)
        df_train.to_csv(os.path.join(out_dir, f"{args.system}_training_metrics_{method}.csv"), index=False)
        
        # measurement history
        df_measure = pd.DataFrame({
            "Step": np.arange(args.measure_steps),
            "E_Local_Mean": e_time_series,
            "Acceptance": np.array(acc_hist)
        })
        df_measure.to_csv(os.path.join(out_dir, f"{args.system}_measurement_data_{method}.csv"), index=False)

if __name__ == "__main__":
    main()