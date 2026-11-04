from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .models import SystemInfo, GridConfig
from .classical import classical_equilibrium_positions
from .analysis import analyze_vqmc, analyze_orbital_state, compare_analyses
from .report import write_markdown_report


def _grid(a):
    return GridConfig(a.x_min,a.x_max,a.n_grid,a.n_bins,a.r_max,a.pin_halfwidth,a.small_r_cut)


def _add_grid(p):
    p.add_argument("--x-min",type=float,default=-20); p.add_argument("--x-max",type=float,default=20)
    p.add_argument("--n-grid",type=int,default=401); p.add_argument("--n-bins",type=int,default=200)
    p.add_argument("--r-max",type=float,default=10); p.add_argument("--pin-halfwidth",type=float,default=0.12)
    p.add_argument("--small-r-cut",type=float,default=None)


def _add_system(p, required=True):
    p.add_argument("--n-up",type=int,required=required); p.add_argument("--n-down",type=int,required=required)
    p.add_argument("--omega",type=float,required=required); p.add_argument("--g",type=float,required=required); p.add_argument("--a",type=float,required=required)


def main(argv=None):
    ap=argparse.ArgumentParser(prog="wigner-post",description="HF/CASSCF/RL-VQMC Wigner-regime post-processing")
    sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("classical-sites"); _add_system(p,True); p.add_argument("--output",default=None)
    p=sp.add_parser("analyze-vqmc"); p.add_argument("input"); p.add_argument("--output",required=True); _add_system(p,True); _add_grid(p)
    p.add_argument("--iat-estimator",choices=["geyer","sokal"],default="geyer"); p.add_argument("--pin-x",type=float,default=None); p.add_argument("--pin-spin",choices=["any","up","down"],default="any")
    for name in ("analyze-hf","analyze-casscf"):
        p=sp.add_parser(name); p.add_argument("state",help="state .npz file or result directory"); p.add_argument("--output",required=True); _add_grid(p); p.add_argument("--pin-x",type=float,default=None); _add_system(p,False)
    p=sp.add_parser("compare"); p.add_argument("analysis_dirs",nargs="+"); p.add_argument("--labels",nargs="+",required=True); p.add_argument("--output",required=True); p.add_argument("--reference-energy",type=float,default=None)
    p=sp.add_parser("report"); p.add_argument("analysis_dir"); p.add_argument("--output",default=None)
    a=ap.parse_args(argv)
    if a.cmd=="classical-sites":
        sys=SystemInfo(a.n_up,a.n_down,a.omega,a.g,a.a); x,v=classical_equilibrium_positions(sys)
        text="positions: "+" ".join(f"{z:.12g}" for z in x)+f"\nclassical_energy: {v:.12g}\n"
        print(text,end="");
        if a.output: Path(a.output).write_text(text,encoding="utf-8")
    elif a.cmd=="analyze-vqmc":
        sys=SystemInfo(a.n_up,a.n_down,a.omega,a.g,a.a)
        analyze_vqmc(a.input,a.output,sys,_grid(a),a.iat_estimator,a.pin_x,a.pin_spin); write_markdown_report(a.output)
    elif a.cmd in ("analyze-hf","analyze-casscf"):
        ov={k:getattr(a,k) for k in ("n_up","n_down","omega","g","a") if getattr(a,k) is not None}
        analyze_orbital_state(a.state,a.output,"hf" if a.cmd=="analyze-hf" else "casscf",_grid(a),a.pin_x,ov); write_markdown_report(a.output)
    elif a.cmd=="compare":
        if len(a.labels)!=len(a.analysis_dirs): raise SystemExit("--labels must match number of analysis directories")
        compare_analyses(a.analysis_dirs,a.labels,a.output,a.reference_energy)
    elif a.cmd=="report":
        print(write_markdown_report(a.analysis_dir,a.output))

if __name__=="__main__": main()
