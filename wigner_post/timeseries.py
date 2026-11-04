from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class TimeSeriesSummary:
    mean: float
    variance: float
    standard_error: float
    tau_int: float
    statistical_inefficiency: float
    ess: float
    n_samples: int
    n_time: int
    n_walkers: int
    estimator: str
    max_lag_used: int

    def as_dict(self):
        return asdict(self)


def autocorrelation_fft(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 2:
        return np.ones(1)
    x = x - x.mean()
    var = np.dot(x, x) / n
    if not np.isfinite(var) or var <= 0:
        m = min(n, (max_lag if max_lag is not None else n - 1) + 1)
        out = np.zeros(m)
        out[0] = 1.0
        return out
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft)
    acov = np.fft.irfft(f * np.conjugate(f), n=nfft)[:n]
    acov /= np.arange(n, 0, -1)
    acf = acov / acov[0]
    if max_lag is not None:
        acf = acf[: max_lag + 1]
    return acf


def geyer_initial_monotone(acf: np.ndarray) -> tuple[float, int]:
    """Geyer initial-positive / initial-monotone estimator.

    Returns statistical inefficiency g = 1 + 2 sum_{t>=1} rho_t.
    The IAT convention used by this package is tau_int = g/2.
    """
    r = np.asarray(acf, dtype=float)
    if r.size < 2:
        return 1.0, 0
    paired = []
    k = 0
    while 2 * k + 1 < r.size:
        gamma = r[2 * k] + r[2 * k + 1]
        if not np.isfinite(gamma) or gamma <= 0:
            break
        paired.append(float(gamma))
        k += 1
    if not paired:
        return 1.0, 0
    mono = np.minimum.accumulate(np.asarray(paired))
    mono = np.maximum(mono, 0.0)
    g = max(1.0, -1.0 + 2.0 * mono.sum())
    last_lag = min(r.size - 1, 2 * len(mono) - 1)
    return float(g), int(last_lag)


def sokal_window(acf: np.ndarray, c: float = 5.0) -> tuple[float, int]:
    r = np.asarray(acf, dtype=float)
    tau = 0.5
    last = 0
    for m in range(1, r.size):
        tau = max(0.5, 0.5 + float(np.sum(r[1:m + 1])))
        last = m
        if m >= c * tau:
            break
    return max(1.0, 2.0 * tau), last


def summarize_multiwalker(values: np.ndarray, estimator: str = "geyer", max_lag: int | None = None):
    """Estimate ACF/IAT/ESS without flattening walker and time axes."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    if v.ndim != 2:
        raise ValueError("values must have shape (time, walkers) or (time,)")
    nt, nw = v.shape
    if max_lag is None:
        max_lag = min(nt - 1, max(50, nt // 2))
    max_lag = max(0, min(int(max_lag), nt - 1))
    acfs = np.stack([autocorrelation_fft(v[:, w], max_lag=max_lag) for w in range(nw)])
    acf = np.nanmean(acfs, axis=0)
    if estimator == "geyer":
        g, lag = geyer_initial_monotone(acf)
    elif estimator == "sokal":
        g, lag = sokal_window(acf)
    else:
        raise ValueError("estimator must be 'geyer' or 'sokal'")
    flat = v.reshape(-1)
    n = flat.size
    var = float(np.var(flat, ddof=1)) if n > 1 else 0.0
    ess = min(float(n), float(n / g))
    se = float(np.sqrt(var / max(ess, 1.0)))
    report = TimeSeriesSummary(
        mean=float(np.mean(flat)), variance=var, standard_error=se,
        tau_int=float(g / 2.0), statistical_inefficiency=float(g), ess=ess,
        n_samples=int(n), n_time=int(nt), n_walkers=int(nw), estimator=estimator,
        max_lag_used=int(lag),
    )
    return report, acf, acfs


def block_error_curve(values: np.ndarray, max_points: int = 28):
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v[:, None]
    nt, nw = v.shape
    max_block = max(1, nt // 4)
    sizes = np.unique(np.geomspace(1, max_block, num=min(max_points, max_block), dtype=int))
    rows = []
    for b in sizes:
        nb = nt // b
        if nb < 2:
            continue
        blocks = v[:nb * b].reshape(nb, b, nw).mean(axis=1).reshape(-1)
        rows.append((int(b), int(blocks.size), float(np.std(blocks, ddof=1) / np.sqrt(blocks.size))))
    return rows


def running_mean(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if v.ndim == 2:
        v = v.mean(axis=1)
    return np.cumsum(v) / np.arange(1, len(v) + 1)


def precision_history(values: np.ndarray, estimator: str = "geyer", n_checkpoints: int = 24,
                      sampling_seconds: float | None = None, target_se: float | None = None,
                      minimum_ess: float | None = None):
    """Cumulative precision history and optional estimated time-to-target.

    If only total production sampling time is known, elapsed time at a checkpoint is
    estimated assuming approximately constant production throughput.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim == 1: v = v[:, None]
    nt = v.shape[0]
    lo = min(nt, max(32, nt // 50))
    checkpoints = np.unique(np.linspace(lo, nt, num=min(n_checkpoints, max(1, nt-lo+1)), dtype=int))
    rows=[]; target=None
    for t in checkpoints:
        rep, _, _ = summarize_multiwalker(v[:t], estimator=estimator, max_lag=min(t-1, max(50, t//2)))
        elapsed = None if sampling_seconds is None else float(sampling_seconds) * t / nt
        ok = ((target_se is None or rep.standard_error <= target_se) and
              (minimum_ess is None or rep.ess >= minimum_ess))
        row=dict(sweeps=int(t), mean=rep.mean, standard_error=rep.standard_error,
                 ess=rep.ess, tau_int=rep.tau_int, estimated_elapsed_seconds=elapsed,
                 target_met=bool(ok))
        rows.append(row)
        if ok and target is None: target=row.copy()
    return rows, target
