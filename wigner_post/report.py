from __future__ import annotations
from pathlib import Path
import json


def write_markdown_report(analysis_dir, output=None):
    d = Path(analysis_dir)
    s = json.loads((d / "analysis_summary.json").read_text(encoding="utf-8"))
    output = Path(output) if output else d / "REPORT.md"
    sys = s.get("system", {})
    lines = [
        f"# Wigner-regime analysis: {s.get('method','calculation')}", "",
        "## System", "",
        f"- N = {sys.get('n_electrons')} (N_up={sys.get('n_up')}, N_down={sys.get('n_down')})",
        f"- omega = {sys.get('omega')}, g = {sys.get('g')}, a = {sys.get('a')}",
        f"- lambda ~ g/sqrt(omega) = {sys.get('lambda_coupling')}", "",
        "## Finite-system Wigner crossover diagnostics", "",
        f"- density peak count: {s.get('density_peak_count')}",
        f"- Lindemann-like ratio: {s.get('lindemann_like_ratio')}",
        f"- RMS displacement from classical sites: {s.get('rms_peak_to_classical_displacement')}",
        f"- opposite-spin small-r probability: {s.get('opposite_spin_small_r_probability')}", "",
        "> These are finite trapped-system crossover diagnostics, not thermodynamic order parameters.", "",
    ]
    if "energy_statistics" in s:
        e=s["energy_statistics"]
        lines += ["## VQMC statistical efficiency", "",
                  f"- energy = {e.get('mean')} +/- {e.get('standard_error')}",
                  f"- IAT = {e.get('tau_int')}", f"- ESS = {e.get('ess')}",
                  f"- ESS/s = {e.get('ess_per_second')}", ""]
    lines += ["## Generated figures", "", "- `one_body_density.png`", "- `pair_distances.png` (when available)",
              "- `conditional_density.png` (when available)", "- `natural_occupations.png` (HF/CASSCF)",
              "- `one_body_density_with_potential.png` (HF/CASSCF)",
              "- `molecular_orbitals_with_potential.png` (HF/CASSCF)",
              "- `orbital_probability_densities_with_potential.png` (HF/CASSCF)",
              "- `energy_acf.png`, `energy_trace.png`, `energy_block_error.png` (VQMC)", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
