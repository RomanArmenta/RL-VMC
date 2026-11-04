from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .models import SystemInfo


def _scalar(data, key, default=None):
    if key in data:
        v = data[key]
        if np.asarray(v).shape == ():
            return np.asarray(v).item()
        return v
    return default


def load_state_npz(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


SYSTEM_KEYS = ("n_up", "n_down", "omega", "g", "a")
METADATA_JSON_NAMES = (
    "summary.json", "metadata.json", "config.json",
    "casscf_summary.json", "hf_summary.json", "production_summary.json",
)


def _json_system_metadata(document):
    """Return supported system fields from a metadata JSON document."""
    if not isinstance(document, dict):
        return {}
    config = document.get("config", {})
    if not isinstance(config, dict):
        config = {}
    # Nested config is the normal result schema. Top-level fields remain useful
    # for older exporters; nested values win within one JSON document.
    merged = {**document, **config}
    result = {k: merged[k] for k in SYSTEM_KEYS if k in merged}
    if "a" not in result and "softening" in merged:
        result["a"] = merged["softening"]
    return result


def resolve_orbital_state(path, method):
    """Resolve an HF/CASSCF state file and any sibling metadata JSON."""
    p = Path(path)
    if p.is_dir():
        preferred = [p / f"{method}_state.npz", p / "state.npz"]
        state_path = next((candidate for candidate in preferred if candidate.is_file()), None)
        if state_path is None:
            npz_files = sorted(p.glob("*.npz"))
            if len(npz_files) == 1:
                state_path = npz_files[0]
            else:
                checked = ", ".join(str(x) for x in preferred)
                detail = (f"; found {len(npz_files)} NPZ files" if npz_files else "; found no NPZ files")
                raise FileNotFoundError(
                    f"could not resolve {method.upper()} state directory {p}. "
                    f"Checked {checked}{detail}"
                )
    else:
        state_path = p
    if not state_path.is_file():
        raise FileNotFoundError(f"could not find state NPZ file: {state_path}")
    if state_path.suffix.lower() != ".npz":
        raise ValueError(f"state must be an .npz file or result directory, got: {state_path}")

    data = load_state_npz(state_path)
    root = state_path.parent
    # Put the method-specific name immediately after summary.json while keeping
    # the public candidate list stable and useful in diagnostics.
    names = ["summary.json", f"{method}_summary.json"]
    names.extend(name for name in METADATA_JSON_NAMES if name not in names)
    candidates = [root / name for name in names]
    json_metadata = {}
    json_paths = []
    for candidate in candidates:
        if candidate.is_file():
            try:
                document = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"could not read metadata JSON {candidate}: {exc}") from exc
            # Earlier, more specific candidate names win, while later files may
            # fill fields omitted by summary.json.
            for key, value in _json_system_metadata(document).items():
                json_metadata.setdefault(key, value)
            json_paths.append(candidate)
    context = {
        "state_path": state_path,
        "json_paths": json_paths,
        "json_candidates": candidates,
        "npz_keys": sorted(data),
        "json_keys": sorted(json_metadata),
    }
    return data, json_metadata, context


def system_from_state(data, overrides=None, json_metadata=None, context=None):
    overrides = overrides or {}
    json_metadata = json_metadata or {}
    # Explicit precedence: CLI overrides > NPZ metadata > JSON metadata.
    def get(k, default=None):
        if k in overrides and overrides[k] is not None:
            return overrides[k]
        value = _scalar(data, k)
        return value if value is not None else json_metadata.get(k, default)
    req = {k: get(k) for k in SYSTEM_KEYS}
    missing = [k for k, v in req.items() if v is None]
    if missing:
        details = ""
        if context:
            checked_json = ", ".join(str(p) for p in context.get("json_candidates", []))
            details = (
                f" State NPZ: {context.get('state_path')}; "
                f"NPZ keys: {context.get('npz_keys', [])}; "
                f"metadata JSON files checked: [{checked_json}]; "
                f"metadata keys found: {context.get('json_keys', [])}; "
                f"CLI override keys: {sorted(overrides)}."
            )
        raise ValueError(
            f"missing system metadata keys: {missing}. Required keys are {list(SYSTEM_KEYS)}; "
            "JSON key 'softening' is accepted as 'a'." + details
        )
    return SystemInfo(int(req["n_up"]), int(req["n_down"]), float(req["omega"]), float(req["g"]), float(req["a"]))


def load_vqmc_output(path):
    p = Path(path)
    if p.is_file() and p.suffix == ".npz":
        obs_path = p; root = p.parent
    else:
        root = p
        obs_path = root / "production_observables.npz"
    if not obs_path.exists():
        raise FileNotFoundError(f"could not find {obs_path}")
    obs = load_state_npz(obs_path)
    summary = {}
    js = root / "production_summary.json"
    if js.exists():
        summary = json.loads(js.read_text(encoding="utf-8"))
    history = root / "production_analysis_history.csv"
    block = root / "production_block_error.csv"
    return dict(root=root, observables=obs, summary=summary,
                history_csv=history if history.exists() else None,
                block_csv=block if block.exists() else None)


def save_json(path, obj):
    def convert(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
            if isinstance(v, float) and not np.isfinite(v): return None
            return v
        if isinstance(v, float) and not np.isfinite(v): return None
        if isinstance(v, complex): return [v.real, v.imag]
        if isinstance(v, dict): return {k: convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [convert(x) for x in v]
        return v
    Path(path).write_text(json.dumps(convert(obj), indent=2, allow_nan=False), encoding="utf-8")
