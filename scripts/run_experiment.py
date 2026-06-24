import os
os.environ["JAX_ENABLE_X64"] = "True"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

from src.system import PhysicalSystem
from src.wavefunctions import DeepQMCAnsatz
from src.hamiltonians import FullCoulombHamiltonian
from src.samplers import MALASampler, RLMALASampler
from src.trainers import GroundStateTrainer
from src.actors import DirectionalActor
from src.aux_functions import calculate_iat, compiled_measure_phase

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
    },
    "cyclobutadiene_rect": {
        "n_elec": 28, "n_up": 14, "n_dn": 14,
        "coords": [
            # Carbon
            [ 1.4929,  1.2756, 0.0],
            [-1.4929,  1.2756, 0.0],
            [-1.4929, -1.2756, 0.0],
            [ 1.4929, -1.2756, 0.0],
            # Hydrogen
            [ 2.9347,  2.7174, 0.0],
            [-2.9347,  2.7174, 0.0],
            [-2.9347, -2.7174, 0.0],
            [ 2.9347, -2.7174, 0.0]
        ],
        "charges": [6, 6, 6, 6, 1, 1, 1, 1]
    },
    "cyclobutadiene_square": {
        "n_elec": 28, "n_up": 14, "n_dn": 14,
        "coords": [
            # Carbon
            [ 1.3795,  1.3795, 0.0],
            [-1.3795,  1.3795, 0.0],
            [-1.3795, -1.3795, 0.0],
            [ 1.3795, -1.3795, 0.0],
            # Hydrogen
            [ 2.8214,  2.8214, 0.0],
            [-2.8214,  2.8214, 0.0],
            [-2.8214, -2.8214, 0.0],
            [ 2.8214, -2.8214, 0.0]
        ],
        "charges": [6, 6, 6, 6, 1, 1, 1, 1]
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
    parser.add_argument("--penalty_factor", type=float, default=0.1, help="the factor of inportance of the penalty for rejection in the reward function")
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
    parser.add_argument("--min_clip_log_ratio", type=float, default=-20., help="minimum value for the log ratio in the sampler")
    parser.add_argument("--max_clip_log_ratio", type=float, default=20., help="maximum value for the log ratio in the sampler")
    parser.add_argument("--nan_clip_log_ratio", type=float, default=-20., help="replaces nan values of the log ratio for this value")
    parser.add_argument("--posinf_clip_log_ratio", type=float, default=20., help="replaces minus infinity values of the log ratio for this value")
    parser.add_argument("--neginf_clip_log_ratio", type=float, default=-20., help="replaces plus infinity values of the log ratio for this value")
    parser.add_argument("--scale_clip_min", type=float, default=0.0, help="minimum value of the scale factor for the sigma of the proposal of an actor")
    parser.add_argument("--scale_clip_max", type=float, default=1.0, help="maximum value of the scale factor for the sigma of the proposal of an actor")
    parser.add_argument("--sigma_energy_clip", type=float, default=5.0, help="we 'clip' or limit the values to go no out of this values of sigma from its distribution")
    parser.add_argument("--energy_clip_min", type=float, default=-50., help="minimum of clip for raw energy calculations")
    parser.add_argument("--energy_clip_max", type=float, default=0.0, help="maximum of clip for raw energy calculations")
    parser.add_argument("--actor_amplitude", type=float, default=0.05, help="amplitude of the drift maximum for RL")
    parser.add_argument("--results_out_dir", type=str, default=None, help="where the results are stored")
    parser.add_argument("--n_determinants", type=int, default=10, help="number of Slater determinants in ansatz")
    parser.add_argument("--backflow_transform", type=str, default='mult', help="type of backflow transform")
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
    out_dir = args.results_out_dir if args.results_out_dir is not None else os.path.join(base_dir, "./../results", f"{args.system}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print("=============================================================================================")
    print(" Initializing VMC experiment with the following parameters:")
    print("=============================================================================================")

    print("=============================================================================================")
    for arg_name, arg_value in vars(args).items():
            # Usamos :>25 para alinear el texto a la derecha y que se vea como una tabla limpia
            print(f"{arg_name:>25}: {arg_value}")
    print(f"Results will be saved in: {out_dir}")
    print("=============================================================================================")

    key = jax.random.PRNGKey(args.seed)
    key, k_init = jax.random.split(key)

    hamiltonian = FullCoulombHamiltonian(molecule)

    print(f"Initializing ansatz | model: {"PauliNet" if args.config_name=="default" else args.config_name} | n_det: {args.n_determinants} | backflow_transform: {args.backflow_transform}")
    ansatz = DeepQMCAnsatz(molecule, config_name=args.config_name, n_determinants=args.n_determinants, backflow_transform=args.backflow_transform)
    
    # physically informed initialization
    def get_smart_initialization(key, batch_size, mol, std=0.2):
        """
        assigns initial electronic positions around the atomic nuclei
        """
        assignments = []
        # mol.charges contains atomic numbers
        for idx, z in enumerate(np.array(mol.charges)):
            assignments.extend([idx] * int(z))
            
        assignments = jnp.array(assignments[:mol.n_elec])
        
        base_coords = mol.coords[assignments]
        
        noise = jax.random.normal(key, (batch_size, mol.n_elec, 3)) * std
        return base_coords[None, ...] + noise

    walkers_init = get_smart_initialization(k_init, args.batch_size, molecule)

    params_psi_master = ansatz.init(k_init, walkers_init)
    print("Ansatz instantiated")

    # we execute this methods (samplers)
    methods = ["RL-MALA", "MALA"]

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
            actor = DirectionalActor(n_elec = molecule.n_elec, n_nuclei=len(molecule.coords), actor_amplitude=args.actor_amplitude)
            key, k_act = jax.random.split(key)
            params_actor = actor.init(k_act, walkers, 0, molecule.coords)

            sched_actor = optax.exponential_decay(init_value=args.lr_actor, transition_steps=args.trans_steps_sched_actor, decay_rate=args.decay_rate_actor)
            opt_actor = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate=sched_actor))
            opt_state_actor = opt_actor.init(params_actor)

            sampler=RLMALASampler(ansatz_apply=ansatz.apply, 
                                  system=molecule,
                                  batch_size=args.batch_size,
                                  max_clip_log_ratio=args.max_clip_log_ratio,
                                  min_clip_log_ratio=args.min_clip_log_ratio,
                                  nan_clip_log_ratio=args.nan_clip_log_ratio,
                                  neginf_clip_log_ratio=args.neginf_clip_log_ratio,
                                  posinf_clip_log_ratio=args.posinf_clip_log_ratio,
                                  actor_apply=actor.apply,
                                  tau=args.tau, 
                                  gamma=args.gamma, 
                                  beta=args.beta, 
                                  scale_clip_min=args.scale_clip_min,
                                  scale_clip_max=args.scale_clip_max,
                                  min_clip_grads_ansatz=args.min_clip_grads_ansatz,
                                  max_clip_grads_ansatz=args.max_clip_grads_ansatz,
                                  penalty_factor=args.penalty_factor)
        else:
            sampler = MALASampler(ansatz_apply=ansatz.apply, 
                                  batch_size=args.batch_size, 
                                  tau=args.tau, 
                                  system=molecule,
                                  min_clip_grads_ansatz=args.min_clip_grads_ansatz,
                                  max_clip_grads_ansatz=args.max_clip_grads_ansatz,                                
                                  max_clip_log_ratio=args.max_clip_log_ratio,
                                  min_clip_log_ratio=args.min_clip_log_ratio,
                                  nan_clip_log_ratio=args.nan_clip_log_ratio,
                                  neginf_clip_log_ratio=args.neginf_clip_log_ratio,
                                  posinf_clip_log_ratio=args.posinf_clip_log_ratio)
            opt_actor = None

        trainer = GroundStateTrainer(
            system=molecule, 
            hamiltonian=hamiltonian, 
            sampler=sampler, 
            opt_ansatz=opt_ansatz, 
            opt_actor=opt_actor,
            sigma_energy_clip=args.sigma_energy_clip,
            energy_clip_min=args.energy_clip_min,
            energy_clip_max=args.energy_clip_max,
            min_clip_grads_ansatz=args.min_clip_grads_ansatz,
            max_clip_grads_ansatz=args.max_clip_grads_ansatz)

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
                
                # lr meatric
                mean_reward = float(stats.get("mean_reward", 0.0)) if method == "RL-MALA" else 0.0
                
                # grad norm
                grad_ansatz_norm = float(jnp.linalg.norm(cache["grad_curr"]))
                
                # loss
                actor_loss_val = float(loss_act) if method == "RL-MALA" and 'loss_act' in locals() else 0.0

                log_data = {
                    "Step": step,
                    "E_mean": float(e_mean),
                    "E_Var": float(e_var),
                    "Acceptance": float(stats.get("acceptance", 0.0)),
                    "EJD": float(stats.get("ejd", 0.0)),
                    "Reward": mean_reward,
                    "Grad_Ansatz_Norm": grad_ansatz_norm,
                    "Actor_Loss": actor_loss_val
                }
                training_history.append(log_data)

                if step % args.n_print == 0:
                    if method == "RL-MALA":
                        print(f"Step: {step:05d} | E: {e_mean:0.4f} Ha | Var: {e_var:6.2f} | "
                              f"Acc: {stats['acceptance']:.2f} | EJD: {stats['ejd']:.4f} | "
                              f"Rew: {mean_reward:5.2f} | |g_psi|: {grad_ansatz_norm:5.2f} | "
                              f"L_act: {actor_loss_val:5.2f} | Best E: {e_best:.4f}")
                    else:
                        print(f"Step: {step:05d} | E: {e_mean:0.4f} Ha | Var: {e_var:6.2f} | "
                              f"Acc: {stats['acceptance']:.2f} | EJD: {stats['ejd']:.4f} | "
                              f"|g_psi|: {grad_ansatz_norm:5.2f} | Best E: {e_best:.4f}")


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