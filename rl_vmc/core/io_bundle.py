from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import pickle
from typing import Any, Optional

import h5py
import jax
import numpy as np
import pandas as pd


def _serialize_pytree(tree: Any) -> np.ndarray:
    """Serializa un PyTree de JAX/Flax (convirtiendo hojas a numpy) en un vector uint8."""
    np_tree = jax.tree_util.tree_map(lambda x: np.asarray(x), tree)
    raw_bytes = pickle.dumps(np_tree, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def _deserialize_pytree(uint8_arr: np.ndarray) -> Any:
    """Reconstruye un PyTree desde un arreglo uint8 almacenado en HDF5."""
    raw_bytes = np.asarray(uint8_arr, dtype=np.uint8).tobytes()
    return pickle.loads(raw_bytes)


def save_run_bundle_h5(
    h5_path: str | Path,
    config_dict: dict[str, Any],
    io_flags: Any,
    ansatz_best: Any = None,
    ansatz_final: Any = None,
    actor_final: Any = None,
    critic1_final: Any = None,
    critic2_final: Any = None,
    training_history: Optional[list[dict[str, Any]]] = None,
    production_data: Optional[dict[str, Any]] = None,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Guarda todos los artefactos de una corrida en un único archivo HDF5 (.h5) respetando las banderas."""
    path = Path(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0"
        f.attrs["created_at"] = datetime.datetime.now().isoformat()
        if extra_metadata:
            for k, v in extra_metadata.items():
                f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

        # 1. Configuración
        if getattr(io_flags, "save_config", True):
            f.attrs["config_json"] = json.dumps(config_dict, indent=2)

        # 2. Checkpoints de redes entrenadas
        ckpt_grp = f.create_group("checkpoints")
        if getattr(io_flags, "save_ansatz_weights", True):
            if ansatz_best is not None:
                ckpt_grp.create_dataset("ansatz_best", data=_serialize_pytree(ansatz_best))
            if ansatz_final is not None:
                ckpt_grp.create_dataset("ansatz_final", data=_serialize_pytree(ansatz_final))

        if getattr(io_flags, "save_actor_weights", True) and actor_final is not None:
            ckpt_grp.create_dataset("actor_final", data=_serialize_pytree(actor_final))

        if getattr(io_flags, "save_critic_weights", True):
            if critic1_final is not None:
                ckpt_grp.create_dataset("critic1_final", data=_serialize_pytree(critic1_final))
            if critic2_final is not None:
                ckpt_grp.create_dataset("critic2_final", data=_serialize_pytree(critic2_final))

        # 3. Historial de Entrenamiento
        if getattr(io_flags, "save_training_history", True) and training_history:
            trn_grp = f.create_group("training")
            trn_grp.attrs["records_json"] = json.dumps(training_history)
            df_trn = pd.DataFrame(training_history)
            for col in df_trn.columns:
                series = df_trn[col]
                if pd.api.types.is_numeric_dtype(series):
                    trn_grp.create_dataset(
                        col, data=series.to_numpy(dtype=np.float64), compression="gzip"
                    )

        # 4. Datos de Producción / Evaluación MCMC
        if production_data is not None:
            prod_grp = f.create_group("production")
            for meta_key in ("wall_time_sec", "n_chains", "n_burn_in", "measure_steps", "thin"):
                if meta_key in production_data:
                    prod_grp.attrs[meta_key] = production_data[meta_key]

            if getattr(io_flags, "save_production_walkers", True) and "walkers" in production_data:
                w_arr = np.asarray(production_data["walkers"])
                max_b = getattr(io_flags, "save_walkers_max_batch", -1)
                if max_b is not None and max_b > 0 and w_arr.ndim >= 3 and w_arr.shape[2] > max_b:
                    w_arr = w_arr[:, :, :max_b, ...]
                prod_grp.create_dataset("walkers", data=w_arr, compression="gzip")

            if getattr(io_flags, "save_production_energies", True):
                for e_key in ("local_energy", "kinetic_energy", "potential_energy"):
                    if e_key in production_data and production_data[e_key] is not None:
                        prod_grp.create_dataset(
                            e_key,
                            data=np.asarray(production_data[e_key]),
                            compression="gzip",
                        )

            if getattr(io_flags, "save_production_sampler_stats", True) and "sampler_stats" in production_data:
                stats_grp = prod_grp.create_group("sampler_stats")
                for s_key, s_val in production_data["sampler_stats"].items():
                    if s_val is not None:
                        stats_grp.create_dataset(
                            s_key, data=np.asarray(s_val), compression="gzip"
                        )

    return path


def load_run_bundle_h5(h5_path: str | Path, load_walkers: bool = True) -> dict[str, Any]:
    """Carga un archivo unificado .h5 en un diccionario estructurado para evaluación o post-procesamiento."""
    path = Path(h5_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo unificado HDF5: {path}")

    bundle: dict[str, Any] = {
        "path": str(path),
        "attrs": {},
        "config": {},
        "checkpoints": {},
        "training_df": None,
        "production": {},
    }

    with h5py.File(path, "r") as f:
        for k, v in f.attrs.items():
            bundle["attrs"][k] = v
        if "config_json" in f.attrs:
            bundle["config"] = json.loads(str(f.attrs["config_json"]))

        if "checkpoints" in f:
            for ckpt_name in f["checkpoints"].keys():
                bundle["checkpoints"][ckpt_name] = _deserialize_pytree(
                    f["checkpoints"][ckpt_name][:]
                )

        if "training" in f:
            trn_grp = f["training"]
            if "records_json" in trn_grp.attrs:
                records = json.loads(str(trn_grp.attrs["records_json"]))
                bundle["training_df"] = pd.DataFrame(records)
            else:
                cols = {k: trn_grp[k][:] for k in trn_grp.keys()}
                if cols:
                    bundle["training_df"] = pd.DataFrame(cols)

        if "production" in f:
            prod_grp = f["production"]
            prod_dict: dict[str, Any] = {"attrs": dict(prod_grp.attrs), "sampler_stats": {}}
            if load_walkers and "walkers" in prod_grp:
                prod_dict["walkers"] = prod_grp["walkers"][:]
            for e_key in ("local_energy", "kinetic_energy", "potential_energy"):
                if e_key in prod_grp:
                    prod_dict[e_key] = prod_grp[e_key][:]
            if "sampler_stats" in prod_grp:
                for s_key in prod_grp["sampler_stats"].keys():
                    prod_dict["sampler_stats"][s_key] = prod_grp["sampler_stats"][s_key][:]
            bundle["production"] = prod_dict

    return bundle


def _normalize_legacy_params_to_config(legacy_params: dict[str, Any], method: str) -> dict[str, Any]:
    """Adapta un diccionario plano de params.json legacy a la estructura jerárquica de Experiment1DConfig."""
    if "system" in legacy_params and "ansatz" in legacy_params:
        return legacy_params

    n_elec = int(legacy_params.get("n_elec", 4))
    n_up = int(legacy_params.get("n_up", n_elec // 2))
    n_dn = int(legacy_params.get("n_dn", n_elec - n_up))
    g_val = float(legacy_params.get("interaction_strength", legacy_params.get("g", 1.0)))

    is_neural = method.upper() in ("TD3", "REINFORCE", "RL-MALA", "RLMALA")
    rl_algo = "reinforce" if method.upper() == "REINFORCE" else "td3"
    actor_type = legacy_params.get("actor_type", "full")
    if is_neural:
        proposal_type = "neural_flat" if actor_type == "flat" else "neural_deepset"
        proposal_mode = "scale_only" if actor_type == "scale_only" else "residual_mala"
        train_mode = "coupled"
    else:
        proposal_type = "mala"
        proposal_mode = "residual_mala"
        train_mode = "vmc_only"

    return {
        "system": {
            "n_elec": n_elec,
            "n_up": n_up,
            "n_dn": n_dn,
            "omega": float(legacy_params.get("omega", 0.1)),
            "interaction_strength": g_val,
            "softening": float(legacy_params.get("softening", 0.1)),
        },
        "ansatz": {
            "ansatz_type": "slater_jastrow_1d",
            "wf_model": str(legacy_params.get("wf_model", "slater_jastrow")),
            "orbital_model": str(legacy_params.get("orbital_model", "backflow_ho")),
            "correlation_model": str(legacy_params.get("correlation_model", "spin_rbf_jastrow")),
            "backflow_components": int(legacy_params.get("backflow_components", 8)),
            "backflow_r_max": float(legacy_params.get("backflow_r_max", 5.0)),
            "backflow_width": float(legacy_params.get("backflow_width", 0.7)),
            "backflow_envelope_length": float(legacy_params.get("backflow_envelope_length", 3.0)),
            "casscf_npz_path": None,
            "casscf_trainable": False,
        },
        "proposal": {
            "proposal_type": proposal_type,
            "proposal_mode": proposal_mode,
            "tau": float(legacy_params.get("tau", 0.01)),
            "actor_amplitude": float(legacy_params.get("actor_amplitude", 0.15)),
            "sigma_action_scale": float(legacy_params.get("sigma_action_scale", 0.2)),
            "max_drift_clip": float(legacy_params.get("max_drift_clip", 10.0)),
            "action_rev_limit": float(legacy_params.get("action_rev_limit", 1.0)),
            "scale_clip_min": float(legacy_params.get("scale_clip_min", -5.0)),
            "scale_clip_max": float(legacy_params.get("scale_clip_max", 1.0)),
            "force_scale": float(legacy_params.get("force_scale", 5.0)),
            "n_scale": float(legacy_params.get("n_scale", 10.0)),
            "r_scale": float(legacy_params.get("r_scale", 5.0)),
            "use_spatial_gating": bool(legacy_params.get("use_spatial_gating", False)),
            "spatial_gating_radius": float(legacy_params.get("spatial_gating_radius", 10.0)),
            "legacy_drift_factor": True,
        },
        "sampler": {
            "batch_size": int(legacy_params.get("batch_size", 2048)),
            "update_mode": "all_electrons",
            "sweeps_per_step": 1,
            "n_burn_in": int(legacy_params.get("n_ter", 2000)),
            "min_clip_grads_ansatz": float(legacy_params.get("min_clip_grads_ansatz", -500.0)),
            "max_clip_grads_ansatz": float(legacy_params.get("max_clip_grads_ansatz", 500.0)),
            "min_clip_log_ratio": float(legacy_params.get("min_clip_log_ratio", -50.0)),
            "max_clip_log_ratio": float(legacy_params.get("max_clip_log_ratio", 50.0)),
            "nan_clip_log_ratio": float(legacy_params.get("nan_clip_log_ratio", -50.0)),
            "posinf_clip_log_ratio": float(legacy_params.get("posinf_clip_log_ratio", 50.0)),
            "neginf_clip_log_ratio": float(legacy_params.get("neginf_clip_log_ratio", -50.0)),
        },
        "rl": {
            "algorithm": rl_algo,
            "reward_type": str(legacy_params.get("reward_type", "cdlb")),
            "lr_actor": float(legacy_params.get("lr_actor", 1e-3)),
            "lr_critic": float(legacy_params.get("lr_critic", 3e-4)),
            "rl_gamma_td3": float(legacy_params.get("rl_gamma_td3", 0.99)),
            "penalty_factor": float(legacy_params.get("penalty_factor", 0.1)),
            "beta": float(legacy_params.get("beta", 10.0)),
            "gamma": float(legacy_params.get("gamma", 0.1)),
            "kappa": float(legacy_params.get("kappa", 0.5)),
            "max_phys_jump_factor": float(legacy_params.get("max_phys_jump_factor", 5.0)),
        },
        "train": {
            "mode": train_mode,
            "seed": int(legacy_params.get("seed", 42)),
            "lr_ansatz": float(legacy_params.get("lr_ansatz", 5e-4)),
            "train_steps": int(legacy_params.get("train_steps", 5000)),
            "coupled_training_cycles": int(legacy_params.get("coupled_training_cycles", 8)),
            "vmc_iterations_per_cycle": int(legacy_params.get("vmc_iterations_per_cycle", 1250)),
            "rl_sweeps_per_cycle": int(legacy_params.get("rl_sweeps_per_cycle", 100)),
        },
        "eval": {
            "measure_steps": int(legacy_params.get("measure_steps", 2000)),
        },
        "legacy_raw_params": legacy_params,
    }


def convert_legacy_dir_to_h5(
    legacy_dir: str | Path,
    output_h5_path: Optional[str | Path] = None,
    method: Optional[str] = None,
) -> Path:
    """Convierte una carpeta de resultados previa (params.json, *.pkl, *.csv) al formato unificado .h5."""
    dir_path = Path(legacy_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"No existe la carpeta legacy: {dir_path}")

    params_file = dir_path / "params.json"
    legacy_params = {}
    if params_file.exists():
        with params_file.open("r", encoding="utf-8") as f:
            legacy_params = json.load(f)

    # Detectar método automáticamente si no fue especificado
    if method is None:
        if "method" in legacy_params:
            method = str(legacy_params["method"])
        elif "rl_method" in legacy_params:
            method = str(legacy_params["rl_method"])
        elif (dir_path / "TD3_actor_weights.pkl").exists() or (dir_path / "RL-MALA_actor_weights.pkl").exists():
            method = "TD3"
        elif (dir_path / "REINFORCE_actor_weights.pkl").exists():
            method = "REINFORCE"
        else:
            method = "MALA"

    config_dict = _normalize_legacy_params_to_config(legacy_params, method)

    # Cargar pesos .pkl disponibles
    ansatz_best = None
    for candidate in (
        f"{method}_ansatz_weights.pkl",
        "TD3_ansatz_weights.pkl",
        "REINFORCE_ansatz_weights.pkl",
        "RL-MALA_ansatz_weights.pkl",
        "MALA_ansatz_weights.pkl",
    ):
        cand_path = dir_path / candidate
        if cand_path.exists():
            with cand_path.open("rb") as f:
                ansatz_best = pickle.load(f)
            break

    actor_final = None
    for candidate in (
        f"{method}_actor_weights.pkl",
        "TD3_actor_weights.pkl",
        "REINFORCE_actor_weights.pkl",
        "RL-MALA_actor_weights.pkl",
    ):
        cand_path = dir_path / candidate
        if cand_path.exists():
            with cand_path.open("rb") as f:
                actor_final = pickle.load(f)
            break

    critic1_final, critic2_final = None, None
    c1_path = dir_path / "TD3_critic1_weights.pkl"
    c2_path = dir_path / "TD3_critic2_weights.pkl"
    if c1_path.exists():
        with c1_path.open("rb") as f:
            critic1_final = pickle.load(f)
    if c2_path.exists():
        with c2_path.open("rb") as f:
            critic2_final = pickle.load(f)

    # Cargar training_metrics.csv si existe
    training_history = None
    for trn_name in ("training_metrics.csv", f"Harmonic1D_training_metrics_{method}.csv"):
        trn_csv = dir_path / trn_name
        if trn_csv.exists():
            df_t = pd.read_csv(trn_csv)
            training_history = df_t.to_dict(orient="records")
            break

    # Cargar measurement_data.csv si existe
    production_data = None
    for meas_name in ("measurement_data.csv", f"Harmonic1D_measurement_data_{method}.csv"):
        meas_csv = dir_path / meas_name
        if meas_csv.exists():
            df_m = pd.read_csv(meas_csv)
            production_data = {
                "n_chains": 1,
                "n_burn_in": int(legacy_params.get("n_ter", 0)),
                "measure_steps": len(df_m),
                "thin": 1,
                "wall_time_sec": 0.0,
                "sampler_stats": {},
            }
            if "E_Local_Mean" in df_m.columns:
                # Shape (1, n_steps, 1)
                production_data["local_energy"] = df_m["E_Local_Mean"].to_numpy(dtype=np.float64)[None, :, None]
            if "Acceptance" in df_m.columns:
                production_data["sampler_stats"]["acceptance"] = df_m["Acceptance"].to_numpy(dtype=np.float64)[None, :]
            break

    if output_h5_path is None:
        output_h5_path = dir_path / f"{method}_bundle.h5"

    class _AllTrueFlags:
        save_config = True
        save_ansatz_weights = True
        save_actor_weights = True
        save_critic_weights = True
        save_training_history = True
        save_production_walkers = True
        save_walkers_max_batch = -1
        save_production_energies = True
        save_production_sampler_stats = True

    return save_run_bundle_h5(
        h5_path=output_h5_path,
        config_dict=config_dict,
        io_flags=_AllTrueFlags(),
        ansatz_best=ansatz_best,
        ansatz_final=ansatz_best,
        actor_final=actor_final,
        critic1_final=critic1_final,
        critic2_final=critic2_final,
        training_history=training_history,
        production_data=production_data,
        extra_metadata={"converted_from_legacy_dir": str(dir_path), "method": method},
    )


def export_h5_to_legacy_dir(h5_path: str | Path, out_dir: str | Path) -> Path:
    """Exporta un archivo unificado .h5 al formato de archivos sueltos legacy (params.json, *.pkl, *.csv)."""
    bundle = load_run_bundle_h5(h5_path, load_walkers=False)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cfg = bundle.get("config", {})
    proposal_type = cfg.get("proposal", {}).get("proposal_type", "mala")
    prefix = "TD3" if "neural" in proposal_type else "MALA"

    # Guardar params.json
    with (dest / "params.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    # Guardar checkpoints
    ckpts = bundle.get("checkpoints", {})
    if "ansatz_best" in ckpts:
        with (dest / f"{prefix}_ansatz_weights.pkl").open("wb") as f:
            pickle.dump(ckpts["ansatz_best"], f)
    if "actor_final" in ckpts:
        with (dest / f"{prefix}_actor_weights.pkl").open("wb") as f:
            pickle.dump(ckpts["actor_final"], f)
    if "critic1_final" in ckpts:
        with (dest / "TD3_critic1_weights.pkl").open("wb") as f:
            pickle.dump(ckpts["critic1_final"], f)
    if "critic2_final" in ckpts:
        with (dest / "TD3_critic2_weights.pkl").open("wb") as f:
            pickle.dump(ckpts["critic2_final"], f)

    # Guardar CSVs
    if bundle.get("training_df") is not None:
        bundle["training_df"].to_csv(dest / "training_metrics.csv", index=False)

    prod = bundle.get("production", {})
    if "local_energy" in prod:
        eloc = np.asarray(prod["local_energy"])
        e_mean_step = np.mean(eloc[0], axis=-1) if eloc.ndim == 3 else np.mean(eloc, axis=-1)
        acc = prod.get("sampler_stats", {}).get("acceptance", np.zeros((1, len(e_mean_step))))[0]
        df_m = pd.DataFrame(
            {
                "Step": np.arange(len(e_mean_step)),
                "E_Local_Mean": e_mean_step,
                "Acceptance": acc,
            }
        )
        df_m.to_csv(dest / "measurement_data.csv", index=False)

    return dest
