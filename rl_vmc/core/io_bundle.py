from __future__ import annotations

import datetime
import errno
import fcntl
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd


def _json_default(obj: Any) -> Any:
    """Convierte tipos numpy/Path comunes para serialización JSON segura."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Escribe bytes de forma atómica (archivo temporal en el mismo directorio + fsync + os.replace)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
    return target


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    """Escribe texto de forma atómica."""
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload: Any, indent: int = 2) -> Path:
    """Serializa y escribe un archivo JSON de forma atómica."""
    text = json.dumps(payload, indent=indent, default=_json_default)
    return atomic_write_text(path, text, encoding="utf-8")


def atomic_write_pickle(path: str | Path, obj: Any) -> Path:
    """Serializa y escribe un archivo pickle de forma atómica."""
    return atomic_write_bytes(path, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def atomic_write_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Escribe un DataFrame CSV de forma atómica en el mismo directorio."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.tmp.", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=index)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
    return target


def atomic_copy_file(src: str | Path, dst: str | Path) -> Path:
    """Copia un archivo a otro destino de manera atómica evitando lecturas parciales."""
    src_path = Path(src)
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst_path.name}.tmp.", dir=str(dst_path.parent))
    os.close(fd)
    try:
        shutil.copy2(src_path, tmp_name)
        with open(tmp_name, "rb") as f_sync:
            os.fsync(f_sync.fileno())
        os.replace(tmp_name, dst_path)
    except Exception:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise
    return dst_path


def locked_upsert_csv(
    df_new: pd.DataFrame,
    csv_path: str | Path,
    key_cols: list[str] | tuple[str, ...],
    collect_dir: Optional[str | Path] = None,
) -> Path:
    """Actualiza o inserta filas en un CSV maestro bajo lock exclusivo fcntl y escritura atómica."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{csv_file}.lock")

    with lock_path.open("a+", encoding="utf-8") as f_lock:
        fcntl.flock(f_lock, fcntl.LOCK_EX)
        try:
            df_old = pd.DataFrame()
            if csv_file.exists() and csv_file.stat().st_size > 0:
                try:
                    df_old = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    df_old = pd.DataFrame()
                except Exception as exc_read:
                    backup_name = csv_file.with_name(
                        f"{csv_file.name}.corrupt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    try:
                        shutil.copy2(csv_file, backup_name)
                    except OSError:
                        pass
                    print(f"  [Aviso IO] CSV previo corrupto en {csv_file} ({exc_read}); respaldado en {backup_name}")
                    df_old = pd.DataFrame()

            if df_old.empty:
                df_combined = df_new.copy()
            elif df_new.empty:
                df_combined = df_old.copy()
            else:
                df_combined = pd.concat([df_old, df_new], ignore_index=True, sort=False)

            valid_keys = [k for k in key_cols if k in df_combined.columns]
            if valid_keys and not df_combined.empty:
                if len(valid_keys) == 1:
                    dedup_key = df_combined[valid_keys[0]].astype(str)
                else:
                    dedup_key = df_combined[valid_keys].astype(str).agg("\x1f".join, axis=1)
                keep_mask = ~dedup_key.duplicated(keep="last")
                df_combined = df_combined.loc[keep_mask].reset_index(drop=True)

            atomic_write_csv(df_combined, csv_file, index=False)

            if collect_dir:
                cdir = Path(collect_dir)
                cdir.mkdir(parents=True, exist_ok=True)
                atomic_copy_file(csv_file, cdir / csv_file.name)
        finally:
            fcntl.flock(f_lock, fcntl.LOCK_UN)

    return csv_file


def _touch_dataset_endpoints(ds: h5py.Dataset) -> None:
    """Lee extremos de un dataset HDF5 para verificar que sus chunks no estén truncados."""
    if ds.size <= 0:
        raise ValueError(f"Dataset '{ds.name}' vacío (size=0)")
    if ds.ndim == 0:
        _ = ds[()]
        return
    first_idx = tuple(0 for _ in range(ds.ndim))
    last_idx = tuple(-1 for _ in range(ds.ndim))
    _ = ds[first_idx]
    _ = ds[last_idx]


def validate_run_bundle_h5(
    h5_path: str | Path,
    require_production: bool = True,
    require_walkers: bool = True,
) -> tuple[bool, str]:
    """Valida rápidamente la integridad estructural y legibilidad de un archivo run_bundle.h5."""
    path = Path(h5_path)
    if not path.is_file():
        return False, "archivo inexistente"
    try:
        if path.stat().st_size == 0:
            return False, "archivo vacío (0 bytes)"
    except OSError as exc:
        return False, f"no se pudo inspeccionar tamaño: {exc}"

    try:
        with h5py.File(path, "r") as f:
            if "config_json" not in f.attrs:
                return False, "falta atributo config_json"
            cfg_obj = json.loads(str(f.attrs["config_json"]))
            if not isinstance(cfg_obj, dict) or not cfg_obj:
                return False, "config_json inválido o vacío"

            if "checkpoints" not in f:
                return False, "falta grupo checkpoints"
            ckpt_grp = f["checkpoints"]
            for ckpt_name in ckpt_grp.keys():
                _ = _deserialize_pytree(ckpt_grp[ckpt_name][:])

            if "training" in f:
                trn_grp = f["training"]
                if "records_json" in trn_grp.attrs:
                    recs = json.loads(str(trn_grp.attrs["records_json"]))
                    if not isinstance(recs, list):
                        return False, "records_json de training no es una lista"
                for k in trn_grp.keys():
                    _touch_dataset_endpoints(trn_grp[k])

            if require_production:
                if "production" not in f:
                    return False, "falta grupo production"
                prod_grp = f["production"]
                if "local_energy" not in prod_grp:
                    return False, "falta dataset production/local_energy"
                _touch_dataset_endpoints(prod_grp["local_energy"])
                for e_key in ("kinetic_energy", "potential_energy"):
                    if e_key in prod_grp:
                        _touch_dataset_endpoints(prod_grp[e_key])
                if "sampler_stats" in prod_grp:
                    for s_key in prod_grp["sampler_stats"].keys():
                        _touch_dataset_endpoints(prod_grp["sampler_stats"][s_key])
                if require_walkers:
                    if "walkers" not in prod_grp:
                        return False, "falta dataset production/walkers"
                    _touch_dataset_endpoints(prod_grp["walkers"])
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

    return True, "ok"


def _serialize_pytree(tree: Any) -> np.ndarray:
    """Serializa un PyTree de JAX/Flax (convirtiendo hojas a numpy) en un vector uint8."""
    import jax

    np_tree = jax.tree_util.tree_map(lambda x: np.asarray(x), tree)
    raw_bytes = pickle.dumps(np_tree, protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def _deserialize_pytree(uint8_arr: np.ndarray) -> Any:
    """Reconstruye un PyTree desde un arreglo uint8 almacenado en HDF5."""
    raw_bytes = np.asarray(uint8_arr, dtype=np.uint8).tobytes()
    return pickle.loads(raw_bytes)


def _cleanup_stale_h5_temps(parent_dir: Path, target_name: str) -> int:
    """Elimina archivos temporales huérfanos de HDF5 dejados por fallos previos en el directorio."""
    removed = 0
    if not parent_dir.is_dir():
        return removed
    prefix = f".{target_name}.tmp."
    try:
        for entry in parent_dir.iterdir():
            if entry.is_file() and entry.name.startswith(prefix):
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return removed


def _ensure_disk_space(target_dir: Path, required_bytes: int, context_label: str) -> None:
    """Verifica que exista espacio libre suficiente antes de abrir/escribir HDF5 con libhdf5."""
    try:
        free_bytes = shutil.disk_usage(target_dir).free
    except OSError:
        return
    if free_bytes < required_bytes:
        free_mb = free_bytes / (1024.0 * 1024.0)
        req_mb = required_bytes / (1024.0 * 1024.0)
        raise OSError(
            errno.ENOSPC,
            f"Espacio insuficiente en disco en '{target_dir}' para guardar '{context_label}': "
            f"{free_mb:.1f} MiB libres < {req_mb:.1f} MiB requeridos",
        )


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
    """Guarda todos los artefactos de una corrida en un único archivo HDF5 (.h5) de forma atómica."""
    path = Path(h5_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_stale_h5_temps(path.parent, path.name)

    energy_mode = str(
        getattr(io_flags, "save_production_energy_mode", "step_mean") or "step_mean"
    ).lower()
    if energy_mode not in ("step_mean", "full"):
        energy_mode = "step_mean"

    w_dtype_name = str(
        getattr(io_flags, "save_production_walkers_dtype", "float32") or "float32"
    ).lower()
    w_dtype = np.float32 if w_dtype_name == "float32" else np.float64

    s_dtype_name = str(
        getattr(io_flags, "save_production_stats_dtype", "float32") or "float32"
    ).lower()
    s_dtype = np.float32 if s_dtype_name == "float32" else np.float64

    # Pre-serializar y compactar en memoria antes de abrir libhdf5 para evitar
    # escribir/fallar a mitad de disco lleno (que puede provocar SIGSEGV en HDF5).
    raw_payload_bytes = 0
    config_json_str: Optional[str] = None
    if getattr(io_flags, "save_config", True):
        config_json_str = json.dumps(config_dict, indent=2, default=_json_default)
        raw_payload_bytes += len(config_json_str.encode("utf-8"))

    ckpt_arrays: dict[str, np.ndarray] = {}
    if getattr(io_flags, "save_ansatz_weights", True):
        if ansatz_best is not None:
            ckpt_arrays["ansatz_best"] = _serialize_pytree(ansatz_best)
        if ansatz_final is not None:
            ckpt_arrays["ansatz_final"] = _serialize_pytree(ansatz_final)
    if getattr(io_flags, "save_actor_weights", True) and actor_final is not None:
        ckpt_arrays["actor_final"] = _serialize_pytree(actor_final)
    if getattr(io_flags, "save_critic_weights", True):
        if critic1_final is not None:
            ckpt_arrays["critic1_final"] = _serialize_pytree(critic1_final)
        if critic2_final is not None:
            ckpt_arrays["critic2_final"] = _serialize_pytree(critic2_final)
    for arr in ckpt_arrays.values():
        raw_payload_bytes += int(arr.nbytes)

    training_records_json: Optional[str] = None
    training_cols: dict[str, np.ndarray] = {}
    if getattr(io_flags, "save_training_history", True) and training_history:
        training_records_json = json.dumps(training_history, default=_json_default)
        raw_payload_bytes += len(training_records_json.encode("utf-8"))
        df_trn = pd.DataFrame(training_history)
        for col in df_trn.columns:
            series = df_trn[col]
            if pd.api.types.is_numeric_dtype(series):
                col_arr = series.to_numpy(dtype=np.float64)
                training_cols[col] = col_arr
                raw_payload_bytes += int(col_arr.nbytes)

    prod_attrs: dict[str, Any] = {}
    prod_walkers_arr: Optional[np.ndarray] = None
    prod_energy_arrays: dict[str, np.ndarray] = {}
    prod_stat_arrays: dict[str, np.ndarray] = {}

    if production_data is not None:
        for meta_key in ("wall_time_sec", "n_chains", "n_burn_in", "measure_steps", "thin"):
            if meta_key in production_data:
                prod_attrs[meta_key] = production_data[meta_key]

        eloc_raw = production_data.get("local_energy")
        if eloc_raw is not None:
            eloc_np = np.asarray(eloc_raw)
            eloc_3d = eloc_np[None, ...] if eloc_np.ndim == 2 else eloc_np
            inferred_nw = int(eloc_3d.shape[2]) if eloc_3d.ndim >= 3 else 1
            if (
                inferred_nw <= 1
                and "n_walkers" in production_data
                and int(production_data["n_walkers"]) > 0
            ):
                orig_nw = int(production_data["n_walkers"])
            else:
                orig_nw = inferred_nw
            prod_attrs["n_walkers"] = orig_nw
            prod_attrs["energy_storage_mode"] = energy_mode

            if inferred_nw <= 1 and "e_var_per_chain" in production_data:
                e_var_pc = np.asarray(production_data["e_var_per_chain"], dtype=np.float64)
                prod_attrs["e_var_per_chain"] = e_var_pc
                if "e_var_total" in production_data:
                    prod_attrs["e_var_total"] = float(production_data["e_var_total"])
            elif eloc_3d.size > 0:
                eloc_f64 = np.asarray(eloc_3d, dtype=np.float64)
                if eloc_f64.ndim >= 3:
                    prod_attrs["e_var_per_chain"] = np.var(eloc_f64, axis=(1, 2)).astype(
                        np.float64
                    )
                else:
                    prod_attrs["e_var_per_chain"] = np.asarray(
                        [float(np.var(eloc_f64))], dtype=np.float64
                    )
                prod_attrs["e_var_total"] = float(np.var(eloc_f64))

        if getattr(io_flags, "save_production_walkers", True) and production_data.get("walkers") is not None:
            w_arr = np.asarray(production_data["walkers"])
            max_b = getattr(io_flags, "save_walkers_max_batch", -1)
            if max_b is not None and max_b > 0 and w_arr.ndim >= 3 and w_arr.shape[2] > max_b:
                w_arr = w_arr[:, :, :max_b, ...]
            w_arr = np.ascontiguousarray(w_arr, dtype=w_dtype)
            prod_walkers_arr = w_arr
            raw_payload_bytes += int(w_arr.nbytes)

        if getattr(io_flags, "save_production_energies", True):
            for e_key in ("local_energy", "kinetic_energy", "potential_energy"):
                if e_key in production_data and production_data[e_key] is not None:
                    e_arr = np.asarray(production_data[e_key], dtype=np.float64)
                    if e_arr.ndim == 2:
                        e_arr = e_arr[None, ...]
                    if energy_mode == "step_mean" and e_arr.ndim >= 3 and e_arr.shape[2] > 1:
                        e_arr = np.mean(e_arr, axis=-1, keepdims=True, dtype=np.float64)
                    e_arr = np.ascontiguousarray(e_arr, dtype=np.float64)
                    prod_energy_arrays[e_key] = e_arr
                    raw_payload_bytes += int(e_arr.nbytes)

        if getattr(io_flags, "save_production_sampler_stats", True) and "sampler_stats" in production_data:
            for s_key, s_val in production_data["sampler_stats"].items():
                if s_val is not None:
                    s_arr = np.ascontiguousarray(np.asarray(s_val), dtype=s_dtype)
                    prod_stat_arrays[s_key] = s_arr
                    raw_payload_bytes += int(s_arr.nbytes)

    required_bytes = max(32 * 1024 * 1024, int(raw_payload_bytes) + 8 * 1024 * 1024)
    _ensure_disk_space(path.parent, required_bytes, path.name)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with h5py.File(tmp_path, "w") as f:
            f.attrs["schema_version"] = "1.1"
            f.attrs["created_at"] = datetime.datetime.now().isoformat()
            if extra_metadata:
                for k, v in extra_metadata.items():
                    f.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

            # 1. Configuración
            if config_json_str is not None:
                f.attrs["config_json"] = config_json_str

            # 2. Checkpoints de redes entrenadas
            ckpt_grp = f.create_group("checkpoints")
            for ckpt_name, ckpt_arr in ckpt_arrays.items():
                ckpt_grp.create_dataset(ckpt_name, data=ckpt_arr)

            # 3. Historial de Entrenamiento
            if training_records_json is not None:
                trn_grp = f.create_group("training")
                trn_grp.attrs["records_json"] = training_records_json
                for col, col_arr in training_cols.items():
                    trn_grp.create_dataset(col, data=col_arr, compression="gzip")

            # 4. Datos de Producción / Evaluación MCMC
            if production_data is not None:
                prod_grp = f.create_group("production")
                for meta_key, meta_val in prod_attrs.items():
                    prod_grp.attrs[meta_key] = meta_val

                if prod_walkers_arr is not None:
                    prod_grp.create_dataset("walkers", data=prod_walkers_arr, compression="gzip")

                for e_key, e_arr in prod_energy_arrays.items():
                    prod_grp.create_dataset(e_key, data=e_arr, compression="gzip")

                if prod_stat_arrays:
                    stats_grp = prod_grp.create_group("sampler_stats")
                    for s_key, s_arr in prod_stat_arrays.items():
                        stats_grp.create_dataset(s_key, data=s_arr, compression="gzip")

        need_prod = bool(
            production_data is not None
            and getattr(io_flags, "save_production_energies", True)
            and "local_energy" in prod_energy_arrays
        )
        need_walkers = bool(
            need_prod
            and getattr(io_flags, "save_production_walkers", True)
            and prod_walkers_arr is not None
        )
        ok, reason = validate_run_bundle_h5(
            tmp_path,
            require_production=need_prod,
            require_walkers=need_walkers,
        )
        if not ok:
            raise RuntimeError(f"Validación post-escritura del bundle HDF5 falló: {reason}")

        with open(tmp_path, "rb") as f_sync:
            os.fsync(f_sync.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

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
    import re
    dir_path = Path(legacy_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"No existe la carpeta legacy: {dir_path}")

    # 1. Cargar configuración JSON desde params.json o run_config.json
    legacy_params = {}
    config_candidates = [
        dir_path / "params.json",
        dir_path / "run_config.json",
        dir_path / "config.json",
    ]
    for cpath in config_candidates:
        if cpath.is_file():
            try:
                with cpath.open("r", encoding="utf-8") as f:
                    legacy_params = json.load(f)
                break
            except Exception:
                pass

    # Si no se encontró en la carpeta, buscar en carpetas ascendentes (padre / abuelo)
    if not legacy_params:
        for ancestor in (dir_path.parent, dir_path.parent.parent):
            if ancestor.is_dir():
                for cpath in (ancestor / "params.json", ancestor / "run_config.json", ancestor / "config.json"):
                    if cpath.is_file():
                        try:
                            with cpath.open("r", encoding="utf-8") as f:
                                legacy_params = json.load(f)
                            break
                        except Exception:
                            pass
                if legacy_params:
                    break

    # 2. Inferir hiperparámetros físicos faltantes a partir del nombre y ruta del directorio
    path_str = str(dir_path.resolve())
    if "n_elec" not in legacy_params:
        m_n = re.search(r"[Nn]e?(\d+)", path_str)
        if m_n:
            legacy_params["n_elec"] = int(m_n.group(1))
    if "interaction_strength" not in legacy_params and "g" not in legacy_params:
        m_g = re.search(r"[gG]_?([0-9]+(?:\.[0-9]+)?)", path_str)
        if m_g:
            legacy_params["interaction_strength"] = float(m_g.group(1))
    if "omega" not in legacy_params:
        m_om = re.search(r"[oO]m(?:ega)?_?([0-9]+(?:\.[0-9]+)?)", path_str)
        if m_om:
            legacy_params["omega"] = float(m_om.group(1))
    if "tau" not in legacy_params:
        m_tau = re.search(r"tau_?([0-9]+(?:\.[0-9]+)?)", path_str)
        if m_tau:
            legacy_params["tau"] = float(m_tau.group(1))
    if "actor_amplitude" not in legacy_params:
        m_amp = re.search(r"[Aa]mp(?:litude)?_?([0-9]+(?:\.[0-9]+)?)", path_str)
        if m_amp:
            legacy_params["actor_amplitude"] = float(m_amp.group(1))
    if "penalty_factor" not in legacy_params:
        m_pen = re.search(r"[Pp]en(?:alty)?_?([0-9]+(?:\.[0-9]+)?)", path_str)
        if m_pen:
            legacy_params["penalty_factor"] = float(m_pen.group(1))

    # Detectar método automáticamente si no fue especificado
    if method is None:
        if "method" in legacy_params:
            method = str(legacy_params["method"])
        elif "rl_method" in legacy_params:
            method = str(legacy_params["rl_method"])
        elif (dir_path / "TD3_actor_weights.pkl").exists() or (dir_path / "RL-MALA_actor_weights.pkl").exists() or (dir_path / "actor_params.npz").exists():
            method = "TD3"
        elif (dir_path / "REINFORCE_actor_weights.pkl").exists():
            method = "REINFORCE"
        elif "TD3" in dir_path.name.upper():
            method = "TD3"
        elif "REINFORCE" in dir_path.name.upper():
            method = "REINFORCE"
        else:
            method = "MALA"

    config_dict = _normalize_legacy_params_to_config(legacy_params, method)

    # 3. Cargar pesos de Ansatz (en dir_path, o fallback a parent o carpetas hermanas)
    ansatz_best = None
    ansatz_candidates = [
        f"{method}_ansatz_weights.pkl",
        "TD3_ansatz_weights.pkl",
        "REINFORCE_ansatz_weights.pkl",
        "RL-MALA_ansatz_weights.pkl",
        "MALA_ansatz_weights.pkl",
        "params_psi.pkl",
    ]
    for candidate in ansatz_candidates:
        cand_path = dir_path / candidate
        if cand_path.is_file():
            try:
                with cand_path.open("rb") as f:
                    ansatz_best = pickle.load(f)
                break
            except Exception:
                pass

    # Si no se encontró en dir_path, buscar en parent y carpetas hermanas
    if ansatz_best is None:
        parent = dir_path.parent
        # A. Checar en parent
        for candidate in ansatz_candidates:
            p_cand = parent / candidate
            if p_cand.is_file():
                try:
                    with p_cand.open("rb") as f:
                        ansatz_best = pickle.load(f)
                    break
                except Exception:
                    pass
        # B. Checar en carpetas hermanas (ej. MALA_tau*, TD3, MALA_Baselines)
        if ansatz_best is None and parent.is_dir():
            for sibling in parent.iterdir():
                if sibling.is_dir() and sibling.resolve() != dir_path.resolve():
                    for candidate in ansatz_candidates:
                        s_cand = sibling / candidate
                        if s_cand.is_file():
                            try:
                                with s_cand.open("rb") as f:
                                    ansatz_best = pickle.load(f)
                                break
                            except Exception:
                                pass
                    if ansatz_best is not None:
                        break

    # Si aún no se encontró ningún archivo de pesos, inicializar ansatz por defecto para permitir evaluar
    if ansatz_best is None:
        import jax
        import jax.numpy as jnp
        from ..ansatze.ansatz_1d import build_ansatz_1d
        from ..config import Experiment1DConfig
        tmp_cfg = Experiment1DConfig.from_dict(config_dict)
        ansatz_model = build_ansatz_1d(tmp_cfg)
        k_init = jax.random.PRNGKey(int(config_dict.get("train", {}).get("seed", 42)))
        ansatz_best = ansatz_model.init(k_init, jnp.zeros((1, tmp_cfg.system.n_elec, 1)))

    # 4. Cargar pesos de Actor si existen
    actor_final = None
    actor_candidates = [
        f"{method}_actor_weights.pkl",
        "TD3_actor_weights.pkl",
        "REINFORCE_actor_weights.pkl",
        "RL-MALA_actor_weights.pkl",
        "actor_params.npz",
    ]
    for candidate in actor_candidates:
        cand_path = dir_path / candidate
        if cand_path.is_file():
            try:
                if candidate.endswith(".npz"):
                    with np.load(cand_path, allow_pickle=True) as adata:
                        if "params" in adata:
                            actor_final = {"params": adata["params"].item()}
                        else:
                            actor_final = {k: adata[k] for k in adata.files}
                else:
                    with cand_path.open("rb") as f:
                        actor_final = pickle.load(f)
                break
            except Exception:
                pass

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
    atomic_write_json(dest / "params.json", cfg, indent=4)

    # Guardar checkpoints
    ckpts = bundle.get("checkpoints", {})
    if "ansatz_best" in ckpts:
        atomic_write_pickle(dest / f"{prefix}_ansatz_weights.pkl", ckpts["ansatz_best"])
    if "actor_final" in ckpts:
        atomic_write_pickle(dest / f"{prefix}_actor_weights.pkl", ckpts["actor_final"])
    if "critic1_final" in ckpts:
        atomic_write_pickle(dest / "TD3_critic1_weights.pkl", ckpts["critic1_final"])
    if "critic2_final" in ckpts:
        atomic_write_pickle(dest / "TD3_critic2_weights.pkl", ckpts["critic2_final"])

    # Guardar CSVs
    if bundle.get("training_df") is not None:
        atomic_write_csv(bundle["training_df"], dest / "training_metrics.csv", index=False)

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
        atomic_write_csv(df_m, dest / "measurement_data.csv", index=False)

    return dest
