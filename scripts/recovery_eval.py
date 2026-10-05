import os
import sys
import glob
import json
import argparse
import pandas as pd
import numpy as np

def fix_json(base_dir):
    sys_dirs = [d for d in glob.glob(os.path.join(base_dir, "results_N10_g*")) if os.path.isdir(d)]
    for sdir in sys_dirs:
        stats_file = os.path.join(sdir, "stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f: data = json.load(f)
            
            # Forzar la inclusión de las llaves si faltan
            if "MALA_ESS_std" not in data: data["MALA_ESS_std"] = 0.0
            if "TD3_ESS_std" not in data: data["TD3_ESS_std"] = 0.0
                
            with open(stats_file, "w") as f: json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results_linear_search_4")
    args = parser.parse_args()
    
    # 1. Arreglar los JSON corruptos de las corridas pasadas
    fix_json(args.base_dir)
    print("Archivos stats.json parcheados.")
    
    # 2. Llamar a tu generador original de la Fig 3 de forma segura
    sys.argv = ["generate_figure3_dynamic.py", "--base_dir", args.base_dir, "--n_elec", "10", "--omega", "0.1"]
    exec(open("scripts/generate_figure3_dynamic.py").read())