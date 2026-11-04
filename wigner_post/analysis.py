from __future__ import annotations
from pathlib import Path
import csv
import json
import numpy as np
import pandas as pd
try:
    from numpy import trapezoid
except ImportError:  # NumPy < 2.0: keep the declared NumPy 1.24 compatibility.
    from scipy.integrate import trapezoid
from .models import SystemInfo, GridConfig
from .classical import classical_equilibrium_positions
from .timeseries import summarize_multiwalker, block_error_curve, running_mean, precision_history
from .structure import (
    sample_one_body_density, sample_pair_distributions, conditional_density_from_samples,
    localization_from_samples, density_peak_metrics, small_r_probability,
    rdm_density_on_grid, pair_density_matrix, pair_distance_from_rho2,
    conditional_from_rho2, natural_occupations,
)
from .ho import orbitals_from_coefficients
from .ci import ci_rdms, hf_rdms_from_occupations
from .io import resolve_orbital_state, load_vqmc_output, system_from_state, save_json
from . import plots


def _mkdir(out):
    p = Path(out); p.mkdir(parents=True, exist_ok=True); return p


def _write_table(path, data):
    pd.DataFrame(data).to_csv(path, index=False)


def _common_wigner_metrics(density, pair, localization, system, grid):
    peaks = density_peak_metrics(density["x"], density["rho_total"], expected_n=system.n_electrons)
    cutoff = grid.small_r_cut
    if cutoff is None:
        cutoff = 0.5 * localization["mean_nearest_neighbor_spacing"] if np.isfinite(localization["mean_nearest_neighbor_spacing"]) else 0.5 * system.oscillator_length
    pud = small_r_probability(pair["r"], pair["p_ud"], cutoff) if system.n_up and system.n_down else None
    return dict(
        density_peak_count=peaks["n_peaks"], density_peak_positions=peaks["peak_positions"],
        lindemann_like_ratio=localization["lindemann_like_ratio"],
        rms_peak_to_classical_displacement=localization["rms_peak_to_classical_displacement"],
        mean_nearest_neighbor_spacing=localization["mean_nearest_neighbor_spacing"],
        opposite_spin_small_r_cut=float(cutoff), opposite_spin_small_r_probability=pud,
    )


def analyze_vqmc(path, outdir, system: SystemInfo, grid=GridConfig(), estimator="geyer", pin_x=None, pin_spin="any"):
    grid.validate(); out = _mkdir(outdir); dat = load_vqmc_output(path)
    obs = dat["observables"]
    if "positions" not in obs:
        raise ValueError("VQMC production_observables.npz must contain 'positions'")
    pos = obs["positions"]
    sites, vcl = classical_equilibrium_positions(system)
    density = sample_one_body_density(pos, system, grid)
    pair = sample_pair_distributions(pos, system, grid)
    loc = localization_from_samples(pos, sites)
    if pin_x is None:
        # Pin the left-central classical site; it is usually well sampled and informative.
        pin_x = float(sites[max(0, len(sites) // 2 - 1)])
    cond = conditional_density_from_samples(pos, system, grid, pin_x, pin_spin)
    metrics = _common_wigner_metrics(density, pair, loc, system, grid)
    summary = dict(method="RL-VQMC/VQMC", system=system.as_dict(), classical_energy=vcl,
                   classical_sites=sites, pin_x=pin_x, pin_spin=pin_spin, **metrics)

    if "local_energy" in obs:
        energy = obs["local_energy"]
        ts, acf, _ = summarize_multiwalker(energy, estimator=estimator)
        summary["energy_statistics"] = ts.as_dict()
        sampling_seconds = dat["summary"].get("sampling_seconds")
        if sampling_seconds is not None:
            summary["energy_statistics"]["sampling_seconds"] = float(sampling_seconds)
            summary["energy_statistics"]["ess_per_second"] = float(ts.ess / max(float(sampling_seconds), 1e-15))
        blocks = block_error_curve(energy)
        target_se = dat["summary"].get("configuration", {}).get("target_energy_se")
        minimum_ess = dat["summary"].get("configuration", {}).get("minimum_ess")
        phist, target = precision_history(energy, estimator=estimator, sampling_seconds=sampling_seconds,
                                          target_se=target_se, minimum_ess=minimum_ess)
        summary["precision_history_target"] = target
        if phist:
            _write_table(out / "precision_history.csv", {k:[r[k] for r in phist] for k in phist[0]})
        plots.plot_energy_trace(energy, running_mean(energy), out / "energy_trace.pdf")
        plots.plot_acf(acf, out / "energy_acf.pdf")
        plots.plot_block_curve(blocks, out / "energy_block_error.pdf")
        _write_table(out / "block_error.csv", dict(block_size=[r[0] for r in blocks], n_blocks=[r[1] for r in blocks], standard_error=[r[2] for r in blocks]))

    # Existing proposal summaries/arrays are retained if available.
    proposal = {}
    for key in ("sigma", "residual_abs"):
        if key in obs and np.asarray(obs[key]).size:
            arr = np.asarray(obs[key], dtype=float).reshape(-1)
            proposal[key] = dict(mean=float(arr.mean()), std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                                 q05=float(np.quantile(arr, .05)), q50=float(np.quantile(arr, .5)), q95=float(np.quantile(arr, .95)))
    for key in ("acceptance_fraction", "mean_acceptance_probability", "mean_proposed_absolute_displacement",
                "mean_accepted_absolute_displacement", "mean_proposed_squared_displacement",
                "mean_accepted_squared_displacement", "energy_effective_sample_size_per_second"):
        if key in dat["summary"]: proposal[key] = dat["summary"][key]
    if "mean_accepted_squared_displacement" in proposal:
        proposal["esjd_per_electron_update"] = proposal["mean_accepted_squared_displacement"]
        proposal["esjd_per_sweep"] = proposal["mean_accepted_squared_displacement"] * system.n_electrons
    if proposal: summary["proposal_diagnostics"] = proposal

    plots.plot_density(density, sites, out / "one_body_density.pdf")
    plots.plot_pairs(pair, out / "pair_distances.pdf")
    plots.plot_conditional(cond, sites, out / "conditional_density.pdf")
    _write_table(out / "one_body_density.csv", density)
    _write_table(out / "pair_distances.csv", {k: v for k, v in pair.items() if isinstance(v, np.ndarray)})
    _write_table(out / "conditional_density.csv", {k: v for k, v in cond.items() if isinstance(v, np.ndarray)})
    save_json(out / "analysis_summary.json", summary)
    return summary


def _extract_orbitals_and_rdms(data, method, system):
    basis_omega = float(np.asarray(data.get("basis_omega", data.get("omega", system.omega))).item())
    C = data.get("C")
    def first_array(*keys, default=None):
        return next((data[key] for key in keys if key in data), default)
    # Support both the original portable schema and the names emitted by the
    # companion UHF package. Full coefficient matrices are preferred; occupied
    # matrices are sufficient because HF only uses the occupied columns.
    if method == "casscf":
        # The companion CASSCF exporter provides both the complete optimized MO
        # matrix and the active block. CI RDMs live in the active orbital space.
        C = first_array("active_orbital_coefficients", "orbital_coefficients", default=C)
    Cu = first_array("C_up", "coefficients_up", "occupied_coefficients_up", default=C)
    Cd = first_array("C_down", "coefficients_down", "occupied_coefficients_down", default=C)
    if Cu is None or Cd is None:
        accepted = [
            "C", "C_up/C_down", "coefficients_up/coefficients_down",
            "occupied_coefficients_up/occupied_coefficients_down",
            "active_orbital_coefficients", "orbital_coefficients",
        ]
        raise ValueError(
            f"state does not contain recognized orbital coefficients. "
            f"Accepted keys: {accepted}; NPZ keys found: {sorted(data)}"
        )
    Cu = np.asarray(Cu, dtype=float); Cd = np.asarray(Cd, dtype=float)
    if method == "hf":
        n_orb = min(Cu.shape[1], Cd.shape[1])
        # Build RDMs in each method's MO space. If UHF has different orbital counts,
        # use independent dimensions below via direct occupied projectors.
        gu = np.diag([1.0] * system.n_up + [0.0] * (Cu.shape[1] - system.n_up))
        gd = np.diag([1.0] * system.n_down + [0.0] * (Cd.shape[1] - system.n_down))
        Guu = np.einsum("pr,qs->pqrs", gu, gu) - np.einsum("ps,qr->pqrs", gu, gu)
        Gdd = np.einsum("pr,qs->pqrs", gd, gd) - np.einsum("ps,qr->pqrs", gd, gd)
        Gud = np.einsum("pr,qs->pqrs", gu, gd)
        rdms = dict(rdm1_up=gu, rdm1_down=gd, rdm2_uu=Guu, rdm2_dd=Gdd, rdm2_ud=Gud)
    else:
        if "rdm1_up_mo" in data and "rdm1_down_mo" in data:
            rdms = dict(rdm1_up=data["rdm1_up_mo"], rdm1_down=data["rdm1_down_mo"])
            for k in ("rdm2_uu", "rdm2_dd", "rdm2_ud"):
                if f"{k}_mo" in data: rdms[k] = data[f"{k}_mo"]
        elif all(k in data for k in ("ci_coefficients", "occupations_up", "occupations_down")):
            rdms = ci_rdms(data["ci_coefficients"], data["occupations_up"], data["occupations_down"], compute_two_body=True)
        elif ("ci_determinants" in data or "selected_ci_determinants" in data):
            coeff_key = "selected_ci_coefficients" if "selected_ci_coefficients" in data else "ci_coefficients"
            det_key = "selected_ci_determinants" if "selected_ci_determinants" in data else "ci_determinants"
            if coeff_key not in data:
                raise ValueError(f"CASSCF state contains {det_key} but not matching CI coefficients")
            n_active = Cu.shape[1]
            mask = (1 << n_active) - 1
            try:
                packed = np.asarray(data[det_key]).astype(np.int64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"could not decode decimal bitstrings in {det_key}: {exc}") from exc
            occupations_up = np.array(
                [[(int(det) >> orbital) & 1 for orbital in range(n_active)] for det in packed],
                dtype=np.int8,
            )
            occupations_down = np.array(
                [[((int(det) >> n_active) & mask) >> orbital & 1 for orbital in range(n_active)] for det in packed],
                dtype=np.int8,
            )
            if np.any(occupations_up.sum(axis=1) != system.n_up) or np.any(occupations_down.sum(axis=1) != system.n_down):
                raise ValueError(
                    f"decoded {det_key} occupations do not match n_up={system.n_up}, "
                    f"n_down={system.n_down}. Expected encoding: active up bits, then active down bits; "
                    f"n_active_orbitals={n_active}."
                )
            rdms = ci_rdms(data[coeff_key], occupations_up, occupations_down, compute_two_body=True)
        else:
            raise ValueError(
                "CASSCF input needs RDMs, separate occupation arrays, or packed "
                "ci_determinants/selected_ci_determinants bitstrings"
            )
    return Cu, Cd, basis_omega, rdms


def analyze_orbital_state(path, outdir, method: str, grid=GridConfig(), pin_x=None, overrides=None):
    if method not in ("hf", "casscf"):
        raise ValueError("method must be hf or casscf")
    grid.validate(); out = _mkdir(outdir)
    data, json_metadata, load_context = resolve_orbital_state(path, method)
    system = system_from_state(data, overrides, json_metadata, load_context)
    Cu, Cd, basis_omega, rdms = _extract_orbitals_and_rdms(data, method, system)
    x = np.linspace(grid.x_min, grid.x_max, grid.n_grid)
    #print(f"grid.x_min, grid.x_max, grid.n_grid: {grid.x_min}, {grid.x_max}, {grid.n_grid}")
    phiu = orbitals_from_coefficients(x, Cu, basis_omega); phid = orbitals_from_coefficients(x, Cd, basis_omega)
    ru, rd = rdm_density_on_grid(phiu, phid, rdms["rdm1_up"], rdms["rdm1_down"])
    density = dict(x=x, rho_total=ru + rd, rho_up=ru, rho_down=rd,
                   p_total=(ru + rd) / system.n_electrons,
                   p_up=ru / max(system.n_up, 1), p_down=rd / max(system.n_down, 1) if system.n_down else np.zeros_like(rd))
    sites, vcl = classical_equilibrium_positions(system)
    if pin_x is None: pin_x = float(sites[max(0, len(sites)//2 - 1)])

    pair = None; cond = None
    if all(k in rdms for k in ("rdm2_uu", "rdm2_dd", "rdm2_ud")):
        Ruu = pair_density_matrix(phiu, phiu, rdms["rdm2_uu"])
        Rdd = pair_density_matrix(phid, phid, rdms["rdm2_dd"])
        Rud = pair_density_matrix(phiu, phid, rdms["rdm2_ud"])
        r_uu, p_uu = pair_distance_from_rho2(x, Ruu, same_spin=True)
        r_dd, p_dd = pair_distance_from_rho2(x, Rdd, same_spin=True)
        r_ud, p_ud = pair_distance_from_rho2(x, Rud, same_spin=False)
        # Restrict and normalize to probability densities for comparison across N.
        mask = r_ud <= grid.r_max; r = r_ud[mask]
        def normp(p, npairs):
            z = float(trapezoid(p[mask], r)) if npairs > 0 else 0.0
            return p[mask] / z if z > 0 else np.zeros_like(r)
        pair = dict(r=r, p_uu=normp(p_uu, system.n_up*(system.n_up-1)/2),
                    p_dd=normp(p_dd, system.n_down*(system.n_down-1)/2),
                    p_ud=normp(p_ud, system.n_up*system.n_down))
        # Pin an up electron by default when available; otherwise down.
        if system.n_up:
            cs, co, idx = conditional_from_rho2(x, ru, Ruu, Rud, pin_x)
            cond = dict(x=x, rho_cond_total=cs+co, rho_cond_up=cs, rho_cond_down=co,
                        pin_x=float(x[idx]), pin_spin="up", pin_halfwidth=0.0, pin_events=-1)
        elif system.n_down:
            cs, co, idx = conditional_from_rho2(x, rd, Rdd, Rud.T, pin_x)
            cond = dict(x=x, rho_cond_total=cs+co, rho_cond_down=cs, rho_cond_up=co,
                        pin_x=float(x[idx]), pin_spin="down", pin_halfwidth=0.0, pin_events=-1)

    occu, natu = natural_occupations(rdms["rdm1_up"]); occd, natd = natural_occupations(rdms["rdm1_down"])
    # For orbital-only methods, estimate localization widths by moments in Voronoi cells around classical sites.
    bounds = np.r_[-np.inf, 0.5*(sites[:-1]+sites[1:]), np.inf]
    means=[]; widths=[]
    for i in range(len(sites)):
        m = (x >= bounds[i]) & (x < bounds[i+1]); w = density["rho_total"][m]
        if trapezoid(w, x[m]) > 1e-12:
            z=trapezoid(w, x[m]); mu=trapezoid(x[m]*w, x[m])/z; var=trapezoid((x[m]-mu)**2*w, x[m])/z
        else: mu=sites[i]; var=np.nan
        means.append(mu); widths.append(np.sqrt(max(var,0)) if np.isfinite(var) else np.nan)
    widths=np.asarray(widths); dnn=np.diff(sites); mean_dnn=float(np.mean(dnn)) if len(dnn) else np.nan
    lind=float(np.sqrt(np.nanmean(widths**2))/mean_dnn) if mean_dnn>0 else np.nan
    loc=dict(classical_sites=sites, quantum_ordered_means=np.asarray(means), ordered_widths=widths,
             mean_nearest_neighbor_spacing=mean_dnn, rms_ordered_width=float(np.sqrt(np.nanmean(widths**2))),
             lindemann_like_ratio=lind, rms_peak_to_classical_displacement=float(np.sqrt(np.mean((np.asarray(means)-sites)**2))))
    metrics = _common_wigner_metrics(density, pair if pair is not None else {"r":np.array([0,1]),"p_ud":np.zeros(2)}, loc, system, grid)
    energy = float(np.asarray(data["energy_total"]).item()) if "energy_total" in data else None
    summary = dict(method=method.upper(), system=system.as_dict(), energy_total=energy,
                   classical_energy=vcl, classical_sites=sites, natural_occupations_up=occu,
                   natural_occupations_down=occd, **metrics)

    plots.plot_density(density, sites, out / "one_body_density.pdf")
    plots.plot_density_with_potential(density, system.omega, sites, out / "one_body_density_with_potential.pdf")
    plots.plot_orbitals_with_potential(
        x, phiu, phid, system.omega, system.n_up, system.n_down, method,
        out / "molecular_orbitals_with_potential.pdf",
    )
    plots.plot_orbital_densities_with_potential(
        x, phiu, phid, system.omega, system.n_up, system.n_down, method,
        out / "orbital_probability_densities_with_potential.pdf",
    )
    plots.plot_natural_occupations(occu, occd, out / "natural_occupations.pdf")
    _write_table(out / "one_body_density.csv", density)
    orbital_columns = {"x": x}
    probability_columns = {"x": x}
    n_plot_up = system.n_up if method == "hf" else min(phiu.shape[1], 8)
    n_plot_down = system.n_down if method == "hf" else min(phid.shape[1], 8)
    for i in range(n_plot_up):
        orbital_columns[f"phi_up_{i+1}"] = phiu[:, i]
        probability_columns[f"abs_phi_up_{i+1}_squared"] = phiu[:, i] ** 2
    for i in range(n_plot_down):
        orbital_columns[f"phi_down_{i+1}"] = phid[:, i]
        probability_columns[f"abs_phi_down_{i+1}_squared"] = phid[:, i] ** 2
    _write_table(out / "molecular_orbitals.csv", orbital_columns)
    _write_table(out / "orbital_probability_densities.csv", probability_columns)
    _write_table(out / "natural_occupations.csv", dict(index=np.arange(1,max(len(occu),len(occd))+1),
        up=np.pad(occu,(0,max(0,len(occd)-len(occu))),constant_values=np.nan),
        down=np.pad(occd,(0,max(0,len(occu)-len(occd))),constant_values=np.nan)))
    if pair is not None:
        plots.plot_pairs(pair, out / "pair_distances.pdf"); _write_table(out / "pair_distances.csv", pair)
    if cond is not None:
        plots.plot_conditional(cond, sites, out / "conditional_density.pdf"); _write_table(out / "conditional_density.csv", {k:v for k,v in cond.items() if isinstance(v,np.ndarray)})
    save_json(out / "analysis_summary.json", summary)
    return summary


def compare_analyses(analysis_dirs, labels, outdir, reference_energy=None):
    out = _mkdir(outdir)
    summaries=[]
    for d,l in zip(analysis_dirs, labels):
        s=json.loads((Path(d)/"analysis_summary.json").read_text(encoding="utf-8")); s["label"]=l; summaries.append(s)
    hf_energy = next((s.get("energy_total") or s.get("energy_statistics",{}).get("mean") for s in summaries if s.get("method")=="HF"), None)
    rows=[]
    for s in summaries:
        e=s.get("energy_total")
        if e is None: e=s.get("energy_statistics",{}).get("mean")
        frac=None
        if reference_energy is not None and hf_energy is not None and abs(hf_energy-reference_energy)>1e-14 and e is not None:
            frac=(hf_energy-e)/(hf_energy-reference_energy)
        rows.append(dict(label=s["label"], method=s.get("method"), energy=e,
                         lindemann_like_ratio=s.get("lindemann_like_ratio"), density_peak_count=s.get("density_peak_count"),
                         opposite_spin_small_r_probability=s.get("opposite_spin_small_r_probability"),
                         correlation_energy_recovery=frac,
                         ess_per_second=s.get("energy_statistics",{}).get("ess_per_second")))
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    save_json(out/"comparison.json", rows)
    return rows
