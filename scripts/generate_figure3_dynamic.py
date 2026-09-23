import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Carpeta base")
    parser.add_argument("--n_elec", type=int, required=True)
    parser.add_argument("--omega", type=float, default=0.1)
    args = parser.parse_args()

    # CORRECCIÓN: Filtro estricto de carpetas
    sys_dirs = [d for d in glob.glob(os.path.join(args.base_dir, f"results_N{args.n_elec}_g*")) if os.path.isdir(d)]
    
    records = []
    for sdir in sys_dirs:
        stats_file = os.path.join(sdir, "stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, "r") as f:
                data = json.load(f)
                data["lambda"] = data["g"] / np.sqrt(data["omega"])
                data["G_RL"] = data["TD3_ESS_mean"] / data["MALA_ESS_mean"]
                records.append(data)
                
    if not records:
        print(f"No hay datos procesados para N={args.n_elec}")
        return

    df = pd.DataFrame(records).sort_values("lambda")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    fig.suptitle(fr"Figure 3: Sampler Efficiency across Interaction Strengths ($N={args.n_elec}, \omega={args.omega}$)" + "\n" + r"MALA values represent $\tau$-optimized baselines", weight='bold', fontsize=15)

    ax0 = axs[0]
    ax0.errorbar(df['lambda'], df['MALA_ESS_mean'], yerr=df['MALA_ESS_std'], fmt='k--o', capsize=4, linewidth=2, label="Optimized MALA")
    ax0.errorbar(df['lambda'], df['TD3_ESS_mean'], yerr=df['TD3_ESS_std'], fmt='r-s', capsize=4, linewidth=2, markersize=8, label="RL-MALA (TD3)")
    ax0.set_yscale('log')
    ax0.set_xlabel(r"Effective Interaction $\lambda = g/\sqrt{\omega}$")
    ax0.set_ylabel("Effective Sample Size / second (ESS/s)")
    ax0.set_title("Absolute Computational Efficiency")
    ax0.legend()

    ax1 = axs[1]
    ax1.axhline(1.0, color='gray', linestyle=':', linewidth=2)
    ax1.plot(df['lambda'], df['G_RL'], 'b-^', linewidth=2, markersize=9, label=f"N = {args.n_elec}")
    ax1.set_xlabel(r"Effective Interaction $\lambda = g/\sqrt{\omega}$")
    ax1.set_ylabel(r"Relative Efficiency Gain ($G_{RL}$)")
    ax1.set_title("Algorithmic Advantage of Learned Proposals")
    ax1.legend()

    plt.tight_layout()
    out_pdf = os.path.join(args.base_dir, f"Figure3_LambdaSweep_N{args.n_elec}.pdf")
    plt.savefig(out_pdf, dpi=300, format='pdf')
    print(f"Figura 3 generada y guardada en {out_pdf}")

if __name__ == "__main__":
    main()