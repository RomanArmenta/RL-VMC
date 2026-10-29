import argparse, json
from dataclasses import fields
from pathlib import Path
import numpy as np
from .config import CASSCFConfig, UHFConfig
from .casscf import run_casscf
from .plotting import make_casscf_plots, make_plots
from .scf import run_uhf


def _parser():
    p = argparse.ArgumentParser(description="UHF or CASSCF for N electrons in a 1D harmonic trap")
    p.add_argument("--method", choices=("uhf", "casscf"), default="uhf")
    p.add_argument("--config", help="JSON config; CLI options override it")
    p.add_argument("--n-up", type=int); p.add_argument("--n-down", type=int)
    p.add_argument("--omega", type=float); p.add_argument("--g", type=float); p.add_argument("--softening", type=float)
    p.add_argument("--n-basis", type=int); p.add_argument("--quadrature-order", type=int)
    p.add_argument("--max-iter", type=int); p.add_argument("--energy-tol", type=float); p.add_argument("--density-tol", type=float)
    p.add_argument("--mixing", type=float); p.add_argument("--level-shift", type=float)
    p.add_argument("--no-diis", action="store_true"); p.add_argument("--output-dir", default="uhf_results")
    p.add_argument("--n-core", type=int); p.add_argument("--n-active-orbitals", type=int)
    p.add_argument("--casscf-max-iter", type=int); p.add_argument("--casscf-energy-tol", type=float)
    p.add_argument("--orbital-gradient-tol", type=float); p.add_argument("--max-ci-dimension", type=int)
    p.add_argument("--n-final-determinants", type=int,
                   help="retain this many largest-|CI| determinants in final export; 0 means all")
    return p


def main(argv=None):
    args = _parser().parse_args(argv); data = {}
    if args.config: data.update(json.loads(Path(args.config).read_text()))
    config_type = CASSCFConfig if args.method == "casscf" else UHFConfig
    valid = {f.name for f in fields(config_type)}
    for key, value in vars(args).items():
        name = key.replace("-", "_")
        if name in valid and value is not None: data[name] = value
    if args.no_diis: data["diis"] = False
    cfg = config_type(**data)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    if args.method == "casscf":
        result = run_casscf(cfg); result.save(out / "casscf_state.npz"); make_casscf_plots(result, out)
        selected_c, _, _, retained_norm = result.selected_ci()
        summary = {"method": "casscf", "converged": result.converged, "iterations": result.iterations,
                   "energy": result.energy, "core_energy": result.core_energy,
                   "active_energy": result.active_energy, "optimizer_message": result.optimizer_message,
                   "full_ci_dimension": len(result.ci_coefficients),
                   "selected_ci_dimension": len(selected_c), "selected_ci_retained_norm": retained_norm,
                   "config": cfg.to_dict()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        ortho = np.max(np.abs(result.orbital_coefficients.T @ result.overlap @ result.orbital_coefficients - np.eye(cfg.n_basis)))
        print(f"converged={result.converged} iterations={result.iterations} E={result.energy:.12f}")
        print(f"CI dimension={len(result.ci_coefficients)} selected={len(selected_c)} retained_norm={retained_norm:.8f}")
        print(f"max orthonormality error={ortho:.3e}")
        print(f"saved {out / 'casscf_state.npz'}")
        if not result.converged: raise SystemExit(2)
        return
    result = run_uhf(cfg)
    result.save(out / "hf_state.npz"); make_plots(result, out)
    (out / "summary.json").write_text(json.dumps({"converged": result.converged, "iterations": result.iterations,
                                                    "energies": result.energies, "config": cfg.to_dict()}, indent=2))
    ortho_u = np.max(np.abs(result.coefficients_up.T @ result.overlap @ result.coefficients_up - np.eye(cfg.n_basis)))
    ortho_d = np.max(np.abs(result.coefficients_down.T @ result.overlap @ result.coefficients_down - np.eye(cfg.n_basis)))
    print(f"converged={result.converged} iterations={result.iterations} E={result.energies['total']:.12f}")
    print(f"max orthonormality error: up={ortho_u:.3e} down={ortho_d:.3e}")
    print(f"saved {out / 'hf_state.npz'}")
    if not result.converged: raise SystemExit(2)


if __name__ == "__main__": main()
