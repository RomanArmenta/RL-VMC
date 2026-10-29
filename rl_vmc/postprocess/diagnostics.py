from __future__ import annotations

import numpy as np
from rl_vmc.config import PostProcessConfig


def compute_acf_1d(series: np.ndarray, max_lag: int = 2000) -> np.ndarray:
    """
    Calcula la función de autocorrelación normalizada rho(k) mediante FFT en O(T log T).
    Soporta series 1D `(n_steps,)` o 2D `(n_steps, batch_size)` promediando sobre canales.
    """
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    n_steps, _ = arr.shape
    eff_max_lag = min(max_lag, max(1, n_steps - 1))
    if n_steps < 4:
        return np.ones(1, dtype=np.float64)

    mean = np.mean(arr, axis=0, keepdims=True)
    centered = arr - mean
    var = np.var(centered, axis=0)
    valid = var > 1e-15
    if not np.any(valid):
        return np.ones(eff_max_lag + 1, dtype=np.float64)

    centered_valid = centered[:, valid]
    var_valid = var[valid]

    n_fft = 1 << (2 * n_steps - 1).bit_length()
    fft_f = np.fft.rfft(centered_valid, n=n_fft, axis=0)
    acov = np.fft.irfft(fft_f * np.conj(fft_f), n=n_fft, axis=0)[: eff_max_lag + 1]
    norms = np.arange(n_steps, n_steps - eff_max_lag - 1, -1, dtype=np.float64)[:, None]
    acov = acov / norms
    acf_per_channel = acov / var_valid[None, :]
    acf_mean = np.mean(acf_per_channel, axis=1)
    acf_mean[0] = 1.0
    return acf_mean


def compute_iat_sokal(acf: np.ndarray, c_sokal: float = 5.0) -> float:
    """
    Calcula el Tiempo de Autocorrelación Integrado (IAT) tau_int = 1 + 2 * sum_{k=1}^M rho(k)
    usando el criterio de ventana automática de Sokal (menor M tal que M >= c * tau_int(M)).
    """
    if len(acf) <= 1:
        return 1.0
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] <= 0.0:
            break
        tau += 2.0 * float(acf[k])
        if k >= c_sokal * tau:
            break
    return max(1.0, float(tau))


def summarize_bundle_diagnostics(
    bundle: dict,
    cfg: PostProcessConfig,
) -> dict:
    """
    Extrae y calcula todas las métricas estadísticas MCMC y físicas desde la sección `/production`
    de un archivo unificado `.h5`.
    """
    prod = bundle.get("production", {})
    if not prod or "local_energy" not in prod:
        return {"per_chain": [], "summary": {}}

    eloc_all = np.asarray(prod["local_energy"], dtype=np.float64)
    if eloc_all.ndim == 2:
        eloc_all = eloc_all[None, ...]
    n_chains, n_steps, n_walkers = eloc_all.shape

    tloc_all = prod.get("kinetic_energy", None)
    vloc_all = prod.get("potential_energy", None)
    walkers_all = prod.get("walkers", None)
    s_stats = prod.get("sampler_stats", {})
    attrs = prod.get("attrs", {})
    total_wall_time = max(1e-9, float(attrs.get("wall_time_sec", 1.0)))
    chain_wall_time = total_wall_time / max(1, n_chains)

    per_chain = []
    for c_idx in range(n_chains):
        e_c = eloc_all[c_idx]  # (n_steps, n_walkers)
        e_step_series = np.mean(e_c, axis=-1)
        e_mean = float(np.mean(e_c))
        e_var = float(np.var(e_c))

        t_mean = (
            float(np.mean(tloc_all[c_idx]))
            if tloc_all is not None and len(tloc_all) > c_idx
            else float("nan")
        )
        v_mean = (
            float(np.mean(vloc_all[c_idx]))
            if vloc_all is not None and len(vloc_all) > c_idx
            else float("nan")
        )

        # ACF sobre la serie temporal del promedio por paso E_step(t)
        acf_e = compute_acf_1d(e_step_series, max_lag=cfg.max_acf_lag)
        tau_int_e = compute_iat_sokal(acf_e, c_sokal=cfg.sokal_window_c)

        if walkers_all is not None and len(walkers_all) > c_idx:
            w_c = np.asarray(walkers_all[c_idx], dtype=np.float64)
            acf_pos = compute_acf_1d(w_c, max_lag=cfg.max_acf_lag)
            tau_int_pos = compute_iat_sokal(acf_pos, c_sokal=cfg.sokal_window_c)
        else:
            acf_pos = np.ones_like(acf_e)
            tau_int_pos = float("nan")

        total_samples = n_steps * n_walkers
        ess_steps = float(n_steps / max(1.0, tau_int_e))
        ess_total = float(total_samples / max(1.0, tau_int_e))
        ess_per_sec = float(ess_total / chain_wall_time)
        se_e = float(np.sqrt(np.var(e_step_series) / max(1.0, ess_steps)))

        acc_mean = (
            float(np.mean(s_stats["acceptance"][c_idx]))
            if "acceptance" in s_stats
            else float("nan")
        )
        esjd_mean = (
            float(np.mean(s_stats["esjd"][c_idx]))
            if "esjd" in s_stats
            else float("nan")
        )
        adisp_mean = (
            float(np.mean(s_stats["accepted_disp"][c_idx]))
            if "accepted_disp" in s_stats
            else float("nan")
        )

        tts_sec = float(
            np.var(e_step_series)
            / (max(1e-12, ess_steps / chain_wall_time) * (cfg.target_se_epsilon ** 2))
        )

        per_chain.append(
            {
                "chain_idx": c_idx,
                "n_steps": int(n_steps),
                "n_walkers": int(n_walkers),
                "total_samples": int(total_samples),
                "E_mean": e_mean,
                "E_var": e_var,
                "SE_E": se_e,
                "T_mean": t_mean,
                "V_mean": v_mean,
                "tau_int_E": tau_int_e,
                "tau_int_pos": tau_int_pos,
                "ESS": ess_total,
                "ESS_per_sec": ess_per_sec,
                "acceptance": acc_mean,
                "ESJD": esjd_mean,
                "accepted_disp": adisp_mean,
                "wall_time_sec": chain_wall_time,
                "time_to_solution_sec": tts_sec,
                "acf_energy": acf_e,
                "acf_position": acf_pos,
                "e_step_series": e_step_series,
            }
        )

    scalar_keys = [
        "E_mean",
        "E_var",
        "SE_E",
        "T_mean",
        "V_mean",
        "tau_int_E",
        "tau_int_pos",
        "ESS",
        "ESS_per_sec",
        "acceptance",
        "ESJD",
        "accepted_disp",
        "wall_time_sec",
        "time_to_solution_sec",
    ]
    summary = {"n_chains": n_chains}
    for k in scalar_keys:
        vals = [c[k] for c in per_chain if not np.isnan(c[k])]
        summary[k] = float(np.mean(vals)) if vals else float("nan")

    min_len_e = min(len(c["acf_energy"]) for c in per_chain)
    summary["acf_energy"] = np.mean(
        [c["acf_energy"][:min_len_e] for c in per_chain], axis=0
    )
    min_len_p = min(len(c["acf_position"]) for c in per_chain)
    summary["acf_position"] = np.mean(
        [c["acf_position"][:min_len_p] for c in per_chain], axis=0
    )
    summary["e_step_series"] = per_chain[0]["e_step_series"]

    return {"per_chain": per_chain, "summary": summary}


def compute_chain_diagnostics(chain_data: dict, cfg: PostProcessConfig) -> dict:
    return summarize_bundle_diagnostics({"production": chain_data}, cfg)
