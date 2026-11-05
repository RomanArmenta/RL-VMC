from __future__ import annotations

import os
from pathlib import Path
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
from rl_vmc.core.optimizers import build_lr_schedule

from rl_vmc.config import Experiment1DConfig
from rl_vmc.core.types import SamplerState
from rl_vmc.physics.harmonic_1d import Harmonic1DHamiltonian
from rl_vmc.ansatze.ansatz_1d import build_ansatz_1d
from rl_vmc.proposals.proposals_1d import build_proposal_1d, DeepSetNeuralProposal1D
from rl_vmc.samplers.mh_sampler_1d import MHSampler1D
from rl_vmc.rl.rewards import compute_rl_reward
from rl_vmc.rl.learners_1d import TD3Learner1D, REINFORCELearner1D
from rl_vmc.core.io_bundle import (
    save_run_bundle_h5,
    load_run_bundle_h5,
    export_h5_to_legacy_dir,
)


def _load_legacy_checkpoints(legacy_dir: str | Path) -> dict:
    d = Path(legacy_dir)
    out = {}
    for fname in (
        "TD3_ansatz_weights.pkl",
        "RL-MALA_ansatz_weights.pkl",
        "MALA_ansatz_weights.pkl",
        "MH_ansatz_weights.pkl",
        "params_psi.pkl",
        "params_psi_casscf_opt.pkl",
    ):
        fpath = d / fname
        if fpath.exists():
            with fpath.open("rb") as f:
                out["ansatz_best"] = pickle.load(f)
            break
    for fname in (
        "TD3_actor_weights.pkl",
        "RL-MALA_actor_weights.pkl",
        "REINFORCE_actor_weights.pkl",
        "actor_weights.pkl",
    ):
        fpath = d / fname
        if fpath.exists():
            with fpath.open("rb") as f:
                out["actor_final"] = pickle.load(f)
            break
    c1, c2 = d / "TD3_critic1_weights.pkl", d / "TD3_critic2_weights.pkl"
    if c1.exists() and c2.exists():
        with c1.open("rb") as f:
            out["critic1_final"] = pickle.load(f)
        with c2.open("rb") as f:
            out["critic2_final"] = pickle.load(f)
    return out


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def run_experiment_1d(cfg: Experiment1DConfig) -> str:
    """
    Pipeline Maestro 1D de RL-VMC:
    1. Instancia Hamiltoniano, Ansatz, Propuesta (Clásica o Actor Neuronal) y Motor MCMC.
    2. Inicializa o restaura pesos desde un archivo `.h5` o carpeta legacy (`.pkl`).
    3. Ejecuta el régimen de entrenamiento configurado (`vmc_only`, `rl_only`, `coupled` o `eval_only`).
    4. Ejecuta la fase de Evaluación/Producción multi-cadena con pesos congelados.
    5. Guarda el archivo unificado HDF5 (`run_bundle.h5`) respetando todas las banderas de `cfg.io`.
    """
    h5_path = Path(cfg.io.output_h5)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_enable_x64", True)
    if cfg.ansatz.ansatz_type in ("casscf_npz", "casscf_1d", "casscf", "multi_sjb", "multi_slater_jastrow", "casscf_jastrow"):
        if cfg.ansatz.casscf_npz_path and os.path.exists(cfg.ansatz.casscf_npz_path):
            try:
                import json as _json
                with np.load(cfg.ansatz.casscf_npz_path, allow_pickle=False) as _cdata:
                    _cmeta = _json.loads(str(_cdata["metadata_json"])).get("config", {})
                cfg.system.n_up = int(_cmeta.get("n_up", cfg.system.n_up))
                cfg.system.n_dn = int(_cmeta.get("n_down", _cmeta.get("n_dn", cfg.system.n_dn)))
                cfg.system.n_elec = cfg.system.n_up + cfg.system.n_dn
                cfg.system.omega = float(_cmeta.get("omega", cfg.system.omega))
                cfg.system.interaction_strength = float(_cmeta.get("g", cfg.system.interaction_strength))
                cfg.system.softening = float(_cmeta.get("softening", cfg.system.softening))
            except Exception as _exc:
                print(f"  [Aviso] No se pudieron sincronizar metadatos desde {cfg.ansatz.casscf_npz_path}: {_exc}")

    print("=" * 76)
    print(f"[RL-VMC 1D] Iniciando Experimento -> {h5_path}")
    print(
        f"  Sistema   : N={cfg.system.n_elec} (up={cfg.system.n_up}, dn={cfg.system.n_dn}), "
        f"omega={cfg.system.omega}, g={cfg.system.interaction_strength}, soft={cfg.system.softening}"
    )
    print(
        f"  Ansatz    : {cfg.ansatz.ansatz_type} "
        f"(orbital={cfg.ansatz.orbital_model}, corr={cfg.ansatz.correlation_model})"
    )
    print(
        f"  Propuesta : {cfg.proposal.proposal_type} (modo={cfg.proposal.proposal_mode}, "
        f"tau={cfg.proposal.tau}, update={cfg.sampler.update_mode})"
    )
    print(f"  Régimen   : {cfg.train.mode}")
    print("=" * 76)

    key = jax.random.PRNGKey(cfg.train.seed)
    k_ansatz, k_prop, k_sampler, k_rl, key = jax.random.split(key, 5)

    hamiltonian = Harmonic1DHamiltonian(cfg.system)
    ansatz = build_ansatz_1d(cfg)
    proposal = build_proposal_1d(cfg)
    sampler = MHSampler1D(cfg, ansatz, proposal)

    dummy_walkers = jnp.zeros((cfg.sampler.batch_size, cfg.system.n_elec, 1))
    params_psi = ansatz.init(k_ansatz, dummy_walkers)
    best_params_psi = jax.tree_util.tree_map(lambda x: x.copy(), params_psi)
    best_energy = float("inf")

    params_actor = proposal.init_params(k_prop, dummy_walkers, dummy_walkers)
    params_critic1, params_critic2 = None, None

    learner = None
    if isinstance(proposal, DeepSetNeuralProposal1D):
        if cfg.rl.algorithm == "td3":
            learner = TD3Learner1D(cfg, proposal, k_rl)
            params_actor = learner.get_actor_params()
            params_critic1, params_critic2 = learner.get_critic_params()
        elif cfg.rl.algorithm == "reinforce":
            learner = REINFORCELearner1D(cfg, proposal, ansatz, sampler, k_rl)
            params_actor = learner.get_actor_params()

    # Restauración opcional de checkpoints previos (.h5 o carpeta legacy)
    loaded_actor = False
    if cfg.io.load_bundle_h5 and os.path.exists(cfg.io.load_bundle_h5):
        print(f"[IO] Cargando checkpoints desde HDF5: {cfg.io.load_bundle_h5}")
        loaded = load_run_bundle_h5(cfg.io.load_bundle_h5, load_walkers=False)
        ckpts = loaded.get("checkpoints", {})
        ans_ckpt = ckpts.get("ansatz_best", ckpts.get("ansatz_final", None))
        if ans_ckpt is not None:
            params_psi = jax.tree_util.tree_map(jnp.asarray, ans_ckpt)
            best_params_psi = jax.tree_util.tree_map(lambda x: x.copy(), params_psi)
            print("  -> Pesos del Ansatz restaurados desde HDF5.")
        if "actor_final" in ckpts and ckpts["actor_final"] is not None:
            params_actor = jax.tree_util.tree_map(jnp.asarray, ckpts["actor_final"])
            if learner is not None:
                learner.set_actor_params(params_actor)
                params_actor = learner.get_actor_params()
            loaded_actor = True
            print("  -> Pesos del Actor restaurados desde HDF5.")
        if (
            "critic1_final" in ckpts
            and "critic2_final" in ckpts
            and isinstance(learner, TD3Learner1D)
        ):
            c1 = jax.tree_util.tree_map(jnp.asarray, ckpts["critic1_final"])
            c2 = jax.tree_util.tree_map(jnp.asarray, ckpts["critic2_final"])
            learner.set_critic_params(c1, c2)

    elif cfg.io.load_legacy_dir and os.path.exists(cfg.io.load_legacy_dir):
        print(f"[IO] Cargando checkpoints desde carpeta legacy: {cfg.io.load_legacy_dir}")
        ckpts = _load_legacy_checkpoints(cfg.io.load_legacy_dir)
        if "ansatz_best" in ckpts:
            params_psi = jax.tree_util.tree_map(jnp.asarray, ckpts["ansatz_best"])
            best_params_psi = jax.tree_util.tree_map(lambda x: x.copy(), params_psi)
            print("  -> Pesos del Ansatz restaurados desde archivo .pkl legacy.")
        if "actor_final" in ckpts:
            params_actor = jax.tree_util.tree_map(jnp.asarray, ckpts["actor_final"])
            if learner is not None:
                learner.set_actor_params(params_actor)
                params_actor = learner.get_actor_params()
            loaded_actor = True
            print("  -> Pesos del Actor restaurados desde archivo .pkl legacy.")
        if (
            "critic1_final" in ckpts
            and "critic2_final" in ckpts
            and isinstance(learner, TD3Learner1D)
        ):
            c1 = jax.tree_util.tree_map(jnp.asarray, ckpts["critic1_final"])
            c2 = jax.tree_util.tree_map(jnp.asarray, ckpts["critic2_final"])
            learner.set_critic_params(c1, c2)

    # Inicializar caminantes y termalización previa al entrenamiento
    state = sampler.init_state(k_sampler, params_psi)
    if cfg.sampler.n_burn_in > 0 and cfg.train.mode != "eval_only":
        print(f"[MCMC] Termalizando caminantes iniciales ({cfg.sampler.n_burn_in} pasos)...")
        key, state = sampler.run_burn_in(
            key,
            state,
            params_psi,
            params_actor,
            cfg.sampler.n_burn_in,
            use_actor=(proposal.is_neural and loaded_actor),
        )

    # Optimizador del Ansatz para VMC con soporte para Learning Rate Scheduler
    total_vmc_steps = (
        cfg.train.train_steps
        if cfg.train.mode == "vmc_only"
        else (cfg.train.coupled_training_cycles * cfg.train.vmc_iterations_per_cycle)
    )
    ansatz_lr = build_lr_schedule(
        base_lr=cfg.train.lr_ansatz,
        total_steps=total_vmc_steps,
        use_scheduler=cfg.train.use_lr_scheduler,
        scheduler_type=cfg.train.lr_scheduler_type,
        warmup_steps=cfg.train.lr_warmup_steps,
        min_ratio=cfg.train.lr_min_ratio,
    )
    vmc_opt = optax.chain(
        optax.clip_by_global_norm(cfg.train.grad_clip_norm),
        optax.adam(ansatz_lr),
    )
    vmc_opt_state = vmc_opt.init(params_psi)

    @jax.jit
    def vmc_update_step(p_psi, opt_st, walkers):
        e_loc, t_loc, v_loc = hamiltonian.local_energy_and_components(
            ansatz.apply, p_psi, walkers
        )
        e_safe = jnp.clip(
            jnp.nan_to_num(e_loc, nan=0.0),
            cfg.train.energy_clip_min,
            cfg.train.energy_clip_max,
        )
        e_mean = jnp.mean(e_safe)
        e_var = jnp.var(e_safe)
        med = jnp.median(e_safe)
        mad = jnp.median(jnp.abs(e_safe - med)) + 1e-8
        e_clipped = jnp.clip(
            e_safe,
            med - cfg.train.sigma_energy_clip * mad,
            med + cfg.train.sigma_energy_clip * mad,
        )
        centered = jax.lax.stop_gradient(e_clipped - jnp.mean(e_clipped))

        def surrogate_loss(p):
            lp = ansatz.apply(p, walkers)
            return 2.0 * jnp.mean(centered * lp)

        grads = jax.grad(surrogate_loss)(p_psi)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0), grads
        )
        grad_norm = optax.global_norm(grads)
        updates, new_opt_st = vmc_opt.update(grads, opt_st, p_psi)
        new_p_psi = optax.apply_updates(p_psi, updates)
        return (
            new_p_psi,
            new_opt_st,
            e_mean,
            e_var,
            jnp.mean(t_loc),
            jnp.mean(v_loc),
            grad_norm,
        )

    training_history: list[dict] = []
    global_step = 0

    def _run_vmc_block(n_iters: int, use_actor_flag: bool, phase_label: str):
        nonlocal key, state, params_psi, vmc_opt_state, best_energy, best_params_psi, global_step
        for it in range(1, n_iters + 1):
            global_step += 1
            for _ in range(cfg.sampler.sweeps_per_step):
                key, k_s = jax.random.split(key)
                state, stats, _ = sampler.step(
                    k_s,
                    state,
                    params_psi,
                    params_actor,
                    use_actor=use_actor_flag,
                    exploration_std=0.0,
                )

            (
                params_psi,
                vmc_opt_state,
                e_m,
                e_v,
                t_m,
                v_m,
                g_norm,
            ) = vmc_update_step(params_psi, vmc_opt_state, state.walkers)

            lp_new, grad_new = ansatz.value_and_grad(
                params_psi,
                state.walkers,
                min_clip=cfg.sampler.min_clip_grads_ansatz,
                max_clip=cfg.sampler.max_clip_grads_ansatz,
            )
            state = SamplerState(
                walkers=state.walkers, log_psi=lp_new, grad_log_psi=grad_new
            )

            e_val = float(e_m)
            if global_step >= cfg.train.steps_for_best and e_val < best_energy:
                best_energy = e_val
                best_params_psi = jax.tree_util.tree_map(lambda x: x.copy(), params_psi)

            if it % cfg.train.n_log == 0 or it == 1 or it == n_iters:
                training_history.append(
                    {
                        "Step": global_step,
                        "Phase": 0.0,
                        "Energy_Mean": e_val,
                        "Energy_Var": float(e_v),
                        "Kinetic_Mean": float(t_m),
                        "Potential_Mean": float(v_m),
                        "Grad_Norm": float(g_norm),
                        "Acceptance": float(jnp.mean(stats.acceptance)),
                        "ESJD": float(jnp.mean(stats.esjd)),
                        "Proposed_Disp": float(jnp.mean(stats.proposed_disp)),
                        "RL_Reward": 0.0,
                        "RL_Exploit": 0.0,
                        "RL_Entropy": 0.0,
                        "RL_Explore": 0.0,
                        "Actor_Loss": 0.0,
                        "Critic_Loss": 0.0,
                    }
                )

            if it % cfg.train.n_print == 0 or it == 1 or it == n_iters:
                print(
                    f"  [{phase_label} {it:4d}/{n_iters}] "
                    f"E={e_val:.6f} (Var={float(e_v):.4f}) | "
                    f"Acc={float(jnp.mean(stats.acceptance)):.2%} | "
                    f"ESJD={float(jnp.mean(stats.esjd)):.5f}"
                )

    def _run_rl_block(n_sweeps: int, phase_label: str):
        nonlocal key, state, params_actor, params_critic1, params_critic2, global_step
        if learner is None:
            return
        for sw in range(1, n_sweeps + 1):
            global_step += 1
            frac = sw / max(1, n_sweeps)
            exp_std = (
                cfg.rl.exp_noise_init * (1.0 - frac) + cfg.rl.exp_noise_final * frac
            )

            if isinstance(learner, TD3Learner1D):
                key, k_s, k_u = jax.random.split(key, 3)
                params_actor = learner.get_actor_params()
                state, stats, rl_info = sampler.step(
                    k_s,
                    state,
                    params_psi,
                    params_actor,
                    use_actor=True,
                    exploration_std=exp_std,
                )
                rewards, r_comp = compute_rl_reward(rl_info, cfg)
                learner.push_transitions(rl_info, rewards)
                upd = learner.update_step(k_u)
                params_actor = learner.get_actor_params()
                params_critic1, params_critic2 = learner.get_critic_params()
                r_mean = float(jnp.mean(rewards))
                r_exploit_val = float(jnp.mean(r_comp["r_exploit"]))
                r_entropy_val = float(jnp.mean(r_comp["r_entropy"]))
                r_explore_val = float(jnp.mean(r_comp["r_explore"]))
                acc_val = float(jnp.mean(stats.acceptance))
                esjd_val = float(jnp.mean(stats.esjd))
                pdisp_val = float(jnp.mean(stats.proposed_disp))
                a_loss = upd["actor_loss"]
                c_loss = upd["critic_loss"]
            else:
                key, k_u = jax.random.split(key)
                state, upd = learner.update_on_policy(k_u, state, params_psi)
                params_actor = learner.get_actor_params()
                r_mean = upd["reward"]
                r_exploit_val = 0.0
                r_entropy_val = 0.0
                r_explore_val = 0.0
                acc_val = upd["acceptance"]
                esjd_val = upd["esjd"]
                pdisp_val = 0.0
                a_loss = upd["actor_loss"]
                c_loss = 0.0

            if sw % max(1, cfg.train.n_log // 2) == 0 or sw == 1 or sw == n_sweeps:
                e_loc_now = hamiltonian.local_energy(
                    ansatz.apply, params_psi, state.walkers
                )
                training_history.append(
                    {
                        "Step": global_step,
                        "Phase": 1.0,
                        "Energy_Mean": float(jnp.mean(e_loc_now)),
                        "Energy_Var": float(jnp.var(e_loc_now)),
                        "Kinetic_Mean": 0.0,
                        "Potential_Mean": 0.0,
                        "Grad_Norm": 0.0,
                        "Acceptance": acc_val,
                        "ESJD": esjd_val,
                        "Proposed_Disp": pdisp_val,
                        "RL_Reward": r_mean,
                        "RL_Exploit": r_exploit_val,
                        "RL_Entropy": r_entropy_val,
                        "RL_Explore": r_explore_val,
                        "Actor_Loss": a_loss,
                        "Critic_Loss": c_loss,
                    }
                )

            if sw % max(1, cfg.train.n_print // 2) == 0 or sw == 1 or sw == n_sweeps:
                print(
                    f"  [{phase_label} {sw:4d}/{n_sweeps}] "
                    f"Reward={r_mean:+.4f} | Acc={acc_val:.2%} | "
                    f"ESJD={esjd_val:.5f} | L_critic={c_loss:.4f}"
                )

    # =========================================================================
    # EJECUCIÓN SEGÚN cfg.train.mode
    # =========================================================================
    if cfg.train.mode == "vmc_only":
        print(f"[Train] Modo vmc_only ({cfg.train.train_steps} pasos)...")
        _run_vmc_block(
            cfg.train.train_steps,
            use_actor_flag=(proposal.is_neural and loaded_actor),
            phase_label="VMC",
        )

    elif cfg.train.mode == "rl_only":
        rl_steps = max(cfg.train.train_steps, cfg.train.rl_sweeps_per_cycle)
        print(f"[Train] Modo rl_only ({rl_steps} barridos RL)...")
        _run_rl_block(rl_steps, phase_label="RL")

    elif cfg.train.mode == "coupled":
        n_cycles = max(1, cfg.train.coupled_training_cycles)
        vmc_per_cyc = cfg.train.vmc_iterations_per_cycle
        if cfg.train.train_steps > 0 and cfg.train.train_steps != 5000:
            vmc_per_cyc = max(1, cfg.train.train_steps // n_cycles)
        print(
            f"[Train] Modo acoplado VMC + RL ({n_cycles} ciclos, {vmc_per_cyc} VMC y {cfg.train.rl_sweeps_per_cycle} RL por ciclo)..."
        )
        for cyc in range(1, n_cycles + 1):
            print(f"\n--- Ciclo Acoplado {cyc}/{n_cycles} ---")
            if (
                cfg.rl.staggered_critic_reset
                and cyc > 1
                and isinstance(learner, TD3Learner1D)
            ):
                key, k_rst = jax.random.split(key)
                which_c = 1 if (cyc % 2 == 0) else 2
                learner.reset_one_critic(k_rst, which=which_c)
                print(f"  [TD3] Staggered reset aplicado al Crítico {which_c}.")

            _run_rl_block(
                cfg.train.rl_sweeps_per_cycle, phase_label=f"C{cyc}-RL"
            )
            _run_vmc_block(
                vmc_per_cyc,
                use_actor_flag=proposal.is_neural,
                phase_label=f"C{cyc}-VMC",
            )

    elif cfg.train.mode == "eval_only":
        print("[Train] Modo eval_only: omitiendo entrenamiento.")

    if best_energy == float("inf"):
        best_params_psi = params_psi

    # =========================================================================
    # FASE DE PRODUCCIÓN / EVALUACIÓN (Pesos Congelados)
    # =========================================================================
    production_data = None
    if cfg.eval.run_evaluation and cfg.eval.measure_steps > 0:
        use_actor_eval = cfg.eval.use_actor_in_eval and proposal.is_neural
        print(
            f"\n[Eval] Ejecutando {cfg.eval.n_eval_chains} cadena(s) de producción "
            f"(burn_in={cfg.eval.eval_burn_in}, measure_steps={cfg.eval.measure_steps}, "
            f"thin={cfg.eval.thin}, use_actor={use_actor_eval})..."
        )
        chains_walkers = []
        chains_eloc, chains_tloc, chains_vloc = [], [], []
        chains_stats: dict[str, list[np.ndarray]] = {
            "acceptance": [],
            "alpha": [],
            "proposed_disp": [],
            "accepted_disp": [],
            "esjd": [],
            "actor_drift_mag": [],
            "actor_sigma": [],
        }
        total_wall_time = 0.0

        eval_init_mode = (
            cfg.sampler.init_mode
            if getattr(cfg.eval, "eval_init_mode", "same_as_sampler") == "same_as_sampler"
            else cfg.eval.eval_init_mode
        )
        for c_idx in range(cfg.eval.n_eval_chains):
            c_key = jax.random.PRNGKey(cfg.train.seed + 10000 + c_idx * 97)
            k_init, k_burn, k_prod = jax.random.split(c_key, 3)
            c_state = sampler.init_state(k_init, best_params_psi, init_mode=eval_init_mode)
            if cfg.eval.eval_burn_in > 0:
                _, c_state = sampler.run_burn_in(
                    k_burn,
                    c_state,
                    best_params_psi,
                    params_actor,
                    cfg.eval.eval_burn_in,
                    use_actor=use_actor_eval,
                )
            res = sampler.run_production_chain(
                k_prod,
                c_state,
                best_params_psi,
                params_actor,
                hamiltonian,
                measure_steps=cfg.eval.measure_steps,
                thin=cfg.eval.thin,
                jit_chunk_size=cfg.eval.jit_chunk_size,
                use_actor=use_actor_eval,
            )
            total_wall_time += res["wall_time_sec"]
            chains_walkers.append(res["walkers"])
            chains_eloc.append(res["local_energy"])
            chains_tloc.append(res["kinetic_energy"])
            chains_vloc.append(res["potential_energy"])
            for sk in chains_stats.keys():
                chains_stats[sk].append(res["sampler_stats"][sk])

            e_c = res["local_energy"]
            acc_c = res["sampler_stats"]["acceptance"]
            esjd_c = res["sampler_stats"]["esjd"]
            print(
                f"  -> [Cadena {c_idx}] <E> = {float(np.mean(e_c)):.6f} +/- "
                f"{float(np.std(e_c) / np.sqrt(e_c.size)):.6f} Ha | "
                f"Acc = {float(np.mean(acc_c)):.2%} | "
                f"ESJD = {float(np.mean(esjd_c)):.5f} | "
                f"t = {res['wall_time_sec']:.2f}s"
            )

        production_data = {
            "n_chains": cfg.eval.n_eval_chains,
            "n_burn_in": cfg.eval.eval_burn_in,
            "measure_steps": cfg.eval.measure_steps,
            "thin": cfg.eval.thin,
            "wall_time_sec": total_wall_time,
            "walkers": np.stack(chains_walkers, axis=0),
            "local_energy": np.stack(chains_eloc, axis=0),
            "kinetic_energy": np.stack(chains_tloc, axis=0),
            "potential_energy": np.stack(chains_vloc, axis=0),
            "sampler_stats": {
                sk: np.stack(slist, axis=0) for sk, slist in chains_stats.items()
            },
        }

    # =========================================================================
    # GUARDADO UNIFICADO HDF5 (+ EXPORTACIÓN LEGACY OPCIONAL)
    # =========================================================================
    saved_path = save_run_bundle_h5(
        h5_path=h5_path,
        config_dict=cfg.to_dict(),
        io_flags=cfg.io,
        ansatz_best=best_params_psi,
        ansatz_final=params_psi,
        actor_final=params_actor if proposal.is_neural else None,
        critic1_final=params_critic1,
        critic2_final=params_critic2,
        training_history=training_history if training_history else None,
        production_data=production_data,
    )
    print(f"\n[IO] Archivo unificado HDF5 guardado en: {saved_path}")

    if cfg.io.export_legacy_files:
        legacy_dir = saved_path.parent
        export_h5_to_legacy_dir(saved_path, legacy_dir)
        print(f"[IO] Archivos legacy (.json, .pkl, .csv) exportados en: {legacy_dir}")

    # =========================================================================
    # GENERACIÓN DE SUMMARY.JSON POR CARPETA Y TABLA CSV MAESTRA INCREMENTAL
    # =========================================================================
    if cfg.io.save_summary_json or cfg.io.master_csv_path:
        try:
            import json
            import fcntl
            import pandas as pd
            from dataclasses import asdict
            from rl_vmc.config import PostProcessConfig
            from rl_vmc.postprocess.diagnostics import summarize_bundle_diagnostics
            from rl_vmc.postprocess.observables import compute_1d_observables
            from rl_vmc.postprocess.casscf_baseline import (
                discover_casscf_baselines,
                find_matching_casscf_baseline,
                compute_exact_casscf_1body_density,
                compute_spin_kl_divergences,
            )

            bundle_mem = {
                "config": cfg.to_dict(),
                "training": training_history if training_history else {},
                "production": production_data if production_data else {},
            }
            pp_cfg = PostProcessConfig()
            diag = (
                summarize_bundle_diagnostics(bundle_mem, pp_cfg)
                if production_data
                else {"per_chain": [], "summary": {}}
            )
            d_sum = diag.get("summary", {})

            obs = None
            w_all = production_data.get("walkers", None) if production_data else None
            if w_all is not None and np.asarray(w_all).size > 0:
                try:
                    obs = compute_1d_observables(
                        np.asarray(w_all),
                        n_up=cfg.system.n_up,
                        n_down=cfg.system.n_dn,
                        n_bins=160,
                        x_max=15.0,
                        omega=cfg.system.omega,
                    )
                except Exception as _exc_obs:
                    print(f"  [Aviso] No se pudieron calcular observables 1D completos: {_exc_obs}")
                    obs = None

            # Resolver referencia CASSCF (explícita o descubierta automáticamente)
            ref_npz = cfg.io.reference_casscf_npz
            if not ref_npz and cfg.ansatz.ansatz_type in ("casscf_npz", "casscf_1d", "casscf", "multi_sjb", "multi_slater_jastrow"):
                ref_npz = cfg.ansatz.casscf_npz_path

            casscf_ref = None
            eff_xmax = _safe_float(obs["x_max"], 15.0) if obs is not None else 15.0
            if ref_npz and os.path.exists(ref_npz):
                try:
                    casscf_ref = compute_exact_casscf_1body_density(
                        ref_npz, x_min=-eff_xmax, x_max=eff_xmax, n_grid=500
                    )
                except Exception as exc:
                    print(f"  [Aviso] Falló lectura de referencia CASSCF ({ref_npz}): {exc}")
            else:
                baselines = discover_casscf_baselines(search_root=cfg.io.uhf_search_root or ".")
                match = find_matching_casscf_baseline(
                    baselines,
                    n_up=cfg.system.n_up,
                    n_dn=cfg.system.n_dn,
                    omega=cfg.system.omega,
                    g=cfg.system.interaction_strength,
                    softening=cfg.system.softening,
                )
                if match is not None:
                    try:
                        casscf_ref = compute_exact_casscf_1body_density(
                            match["npz_path"], x_min=-eff_xmax, x_max=eff_xmax, n_grid=500
                        )
                    except Exception:
                        casscf_ref = None

            kl_tot_mean = float("nan")
            kl_tot_std = float("nan")
            kl_up_mean = float("nan")
            kl_dn_mean = float("nan")
            if casscf_ref is not None and w_all is not None and np.asarray(w_all).size > 0:
                try:
                    w_5d = np.asarray(w_all)
                    if w_5d.ndim == 4:
                        w_5d = w_5d[None, ...]
                    kls_t, kls_u, kls_d = [], [], []
                    for c_i in range(w_5d.shape[0]):
                        kd = compute_spin_kl_divergences(
                            w_5d[c_i], casscf_ref, n_bins=160, x_max=eff_xmax
                        )
                        kls_t.append(kd["kl_total"])
                        kls_u.append(kd["kl_up"])
                        kls_d.append(kd["kl_down"])
                    kl_tot_mean = float(np.mean(kls_t))
                    kl_tot_std = float(np.std(kls_t))
                    kl_up_mean = float(np.mean(kls_u))
                    kl_dn_mean = float(np.mean(kls_d))
                except Exception as _exc_kl:
                    print(f"  [Aviso] Falló cómputo de divergencias KL: {_exc_kl}")

            e_mean_val = _safe_float(d_sum.get("E_mean"))
            t_mean_val = _safe_float(d_sum.get("T_mean"))
            v_mean_val = _safe_float(d_sum.get("V_mean"))
            virial_ratio = (
                float(2.0 * t_mean_val / v_mean_val)
                if (np.isfinite(t_mean_val) and np.isfinite(v_mean_val) and abs(v_mean_val) > 1e-12)
                else float("nan")
            )
            cas_e_val = _safe_float(casscf_ref["energy"]) if casscf_ref is not None else float("nan")
            delta_e_cas = (
                float(e_mean_val - cas_e_val)
                if (np.isfinite(e_mean_val) and np.isfinite(cas_e_val))
                else float("nan")
            )

            train_summary = {}
            sys_tag = f"N{cfg.system.n_elec}_w{cfg.system.omega}_g{cfg.system.interaction_strength}"
            run_tag = cfg.io.run_name if cfg.io.run_name else f"{cfg.train.mode}_{sys_tag}"
            safe_run_tag = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in run_tag)

            if training_history and isinstance(training_history, list) and len(training_history) > 0:
                try:
                    df_train_hist = pd.DataFrame(training_history)
                    tail = min(20, len(df_train_hist))
                    train_summary = {
                        "n_recorded_steps": int(len(df_train_hist)),
                        "final_train_energy_mean": _safe_float(df_train_hist["Energy_Mean"].tail(tail).mean()) if "Energy_Mean" in df_train_hist else float("nan"),
                        "final_train_energy_var": _safe_float(df_train_hist["Energy_Var"].tail(tail).mean()) if "Energy_Var" in df_train_hist else float("nan"),
                        "final_train_acceptance": _safe_float(df_train_hist["Acceptance"].tail(tail).mean()) if "Acceptance" in df_train_hist else float("nan"),
                        "final_train_esjd": _safe_float(df_train_hist["ESJD"].tail(tail).mean()) if "ESJD" in df_train_hist else float("nan"),
                        "final_actor_loss": _safe_float(df_train_hist["Actor_Loss"].iloc[-1]) if "Actor_Loss" in df_train_hist else float("nan"),
                        "final_critic_loss": _safe_float(df_train_hist["Critic_Loss"].iloc[-1]) if "Critic_Loss" in df_train_hist else float("nan"),
                    }
                    if getattr(cfg.io, "save_training_csv", True) or getattr(cfg.io, "save_training_history", True):
                        train_csv_sys = saved_path.parent / f"training_history_{safe_run_tag}.csv"
                        df_train_hist.to_csv(train_csv_sys, index=False)
                        if getattr(cfg.io, "csv_collect_dir", ""):
                            import shutil
                            cdir = Path(cfg.io.csv_collect_dir)
                            cdir.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(train_csv_sys, cdir / train_csv_sys.name)
                except Exception as _exc_th:
                    print(f"  [Aviso] No se pudo exportar training_history.csv: {_exc_th}")

            summary_payload = {
                "run_name": run_tag,
                "system_tag": sys_tag,
                "pipeline_stage": cfg.io.pipeline_stage,
                "output_h5": str(saved_path),
                "system": asdict(cfg.system),
                "model_hyperparameters": {
                    "ansatz": asdict(cfg.ansatz),
                    "proposal": asdict(cfg.proposal),
                },
                "training_hyperparameters": {
                    "train_mode": cfg.train.mode,
                    "seed": cfg.train.seed,
                    "sampler": asdict(cfg.sampler),
                    "train": asdict(cfg.train),
                    "rl": asdict(cfg.rl),
                    "training_final_summary": train_summary,
                },
                "evaluation_config": asdict(cfg.eval),
                "observables": {
                    "n_chains": _safe_int(d_sum.get("n_chains", 0)),
                    "E_mean": e_mean_val,
                    "SE_E": _safe_float(d_sum.get("SE_E")),
                    "E_var": _safe_float(d_sum.get("E_var")),
                    "T_mean": t_mean_val,
                    "V_mean": v_mean_val,
                    "virial_ratio_2T_over_V": virial_ratio,
                    "acceptance": _safe_float(d_sum.get("acceptance")),
                    "ESJD": _safe_float(d_sum.get("ESJD")),
                    "accepted_disp": _safe_float(d_sum.get("accepted_disp")),
                    "tau_int_E": _safe_float(d_sum.get("tau_int_E")),
                    "tau_int_pos": _safe_float(d_sum.get("tau_int_pos")),
                    "ESS": _safe_float(d_sum.get("ESS")),
                    "ESS_per_sec": _safe_float(d_sum.get("ESS_per_sec")),
                    "wall_time_sec": _safe_float(d_sum.get("wall_time_sec")),
                    "time_to_solution_sec": _safe_float(d_sum.get("time_to_solution_sec")),
                    "n_wigner_peaks": _safe_int(obs["n_wigner_peaks"], -1) if obs else -1,
                    "rms_radius": _safe_float(obs["rms_radius"]) if obs else float("nan"),
                    "mean_pair_dist": _safe_float(obs["mean_pair_dist"]) if obs else float("nan"),
                },
                "casscf_comparison": {
                    "has_casscf_baseline": bool(casscf_ref is not None),
                    "casscf_npz_path": str(casscf_ref["npz_path"]) if casscf_ref is not None else None,
                    "casscf_energy": cas_e_val,
                    "delta_E_vs_CASSCF": delta_e_cas,
                    "kl_casscf_total_mean": kl_tot_mean,
                    "kl_casscf_total_std": kl_tot_std,
                    "kl_casscf_up_mean": kl_up_mean,
                    "kl_casscf_down_mean": kl_dn_mean,
                },
            }

            if cfg.io.save_summary_json:
                try:
                    summary_json_path = saved_path.parent / "summary.json"
                    summary_sys_json_path = saved_path.parent / f"summary_{safe_run_tag}.json"
                    with summary_json_path.open("w", encoding="utf-8") as f_json:
                        json.dump(summary_payload, f_json, indent=2)
                    with summary_sys_json_path.open("w", encoding="utf-8") as f_json_sys:
                        json.dump(summary_payload, f_json_sys, indent=2)
                    print(f"[IO] Resumen estructurado guardado en: {summary_sys_json_path}")
                except Exception as _exc_sj:
                    print(f"  [Aviso IO] No se pudo guardar summary.json: {_exc_sj}")

            flat_row = {
                "run_name": run_tag,
                "system_tag": sys_tag,
                "pipeline_stage": cfg.io.pipeline_stage,
                "train_mode": cfg.train.mode,
                "ansatz_type": cfg.ansatz.ansatz_type,
                "orbital_model": cfg.ansatz.orbital_model,
                "correlation_model": cfg.ansatz.correlation_model,
                "proposal_type": cfg.proposal.proposal_type,
                "rl_algorithm": cfg.rl.algorithm if cfg.proposal.proposal_type == "neural_deepset" else "none",
                "N": cfg.system.n_elec,
                "n_up": cfg.system.n_up,
                "n_dn": cfg.system.n_dn,
                "omega": cfg.system.omega,
                "g": cfg.system.interaction_strength,
                "softening": cfg.system.softening,
                "tau": cfg.proposal.tau,
                "n_walkers": cfg.sampler.batch_size,
                "n_eval_chains": _safe_int(d_sum.get("n_chains", 0)),
                "E_mean": e_mean_val,
                "SE_E": _safe_float(d_sum.get("SE_E")),
                "E_var": _safe_float(d_sum.get("E_var")),
                "T_mean": t_mean_val,
                "V_mean": v_mean_val,
                "acceptance": _safe_float(d_sum.get("acceptance")),
                "ESJD": _safe_float(d_sum.get("ESJD")),
                "tau_int_E": _safe_float(d_sum.get("tau_int_E")),
                "tau_int_pos": _safe_float(d_sum.get("tau_int_pos")),
                "ESS": _safe_float(d_sum.get("ESS")),
                "ESS_per_sec": _safe_float(d_sum.get("ESS_per_sec")),
                "wall_time_sec": _safe_float(d_sum.get("wall_time_sec")),
                "time_to_solution_sec": _safe_float(d_sum.get("time_to_solution_sec")),
                "n_wigner_peaks": _safe_int(obs["n_wigner_peaks"], -1) if obs else -1,
                "rms_radius": _safe_float(obs["rms_radius"]) if obs else float("nan"),
                "mean_pair_dist": _safe_float(obs["mean_pair_dist"]) if obs else float("nan"),
                "has_casscf_baseline": bool(casscf_ref is not None),
                "casscf_energy": cas_e_val,
                "delta_E_vs_CASSCF": delta_e_cas,
                "kl_casscf_total_mean": kl_tot_mean,
                "kl_casscf_total_std": kl_tot_std,
                "kl_casscf_up_mean": kl_up_mean,
                "kl_casscf_down_mean": kl_dn_mean,
                "output_h5": str(saved_path),
            }
            try:
                df_row = pd.DataFrame([flat_row])
                indiv_csv_path = saved_path.parent / f"production_metrics_{safe_run_tag}.csv"
                df_row.to_csv(indiv_csv_path, index=False)

                if cfg.io.master_csv_path:
                    csv_file = Path(cfg.io.master_csv_path)
                    csv_file.parent.mkdir(parents=True, exist_ok=True)
                    lock_path = csv_file.with_suffix(".csv.lock")
                    with lock_path.open("w") as f_lock:
                        fcntl.flock(f_lock, fcntl.LOCK_EX)
                        try:
                            write_header = not csv_file.exists() or csv_file.stat().st_size == 0
                            df_row.to_csv(csv_file, mode="a", header=write_header, index=False)
                            if getattr(cfg.io, "csv_collect_dir", ""):
                                import shutil
                                cdir = Path(cfg.io.csv_collect_dir)
                                cdir.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(csv_file, cdir / csv_file.name)
                                shutil.copy2(indiv_csv_path, cdir / indiv_csv_path.name)
                        finally:
                            fcntl.flock(f_lock, fcntl.LOCK_UN)
                    print(f"[IO] Fila agregada en vivo a tabla maestra: {csv_file}")
                elif getattr(cfg.io, "csv_collect_dir", ""):
                    import shutil
                    cdir = Path(cfg.io.csv_collect_dir)
                    cdir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(indiv_csv_path, cdir / indiv_csv_path.name)
            except Exception as _exc_csv:
                print(f"  [Aviso IO] No se pudo escribir fila CSV: {_exc_csv}")
        except Exception as exc:
            print(f"[Aviso IO] No se pudo completar exportación de summary.json / master_csv: {exc}")

    return str(saved_path)
