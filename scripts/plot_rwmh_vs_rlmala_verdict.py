#!/usr/bin/env python3
"""
scripts/plot_rwmh_vs_rlmala_verdict.py
======================================
Genera la figura diagnóstica de control estricto a igual ancho de propuesta
sigma in {1.22, 2.71, 3.30} Bohr comparando Random-Walk Metropolis-Hastings
(deriva = 0, sin red neuronal) frente a RL-MALA (6,000 pasos TD3 en Ayler)
en N=4, g=2.0 y N=6, g=1.0.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    out_dir = "figures/actor_ablation_ayler"
    os.makedirs(out_dir, exist_ok=True)

    sigmas = [
        r"$\sigma=1.22$" + "\n(techo legado)",
        r"$\sigma=2.71$" + "\n" + r"($s_\sigma=1.8$)",
        r"$\sigma=3.30$" + "\n" + r"($s_\sigma=2.0$)",
    ]
    x = np.arange(len(sigmas))
    w = 0.34

    # Datos exactos (Ayler 3x40k para RL-MALA y RWMH@1.22; control local 1x8k batch=64 para RWMH@2.71,3.30)
    data = {
        "N4": {
            "title": r"$N=4,\ \omega=0.1,\ g=2.0$ (RL-MALA aprendió deriva hacia AFUERA $\Rightarrow$ PEOR que RWMH)",
            "rwmh_acc": [45.2, 15.6, 11.1],
            "rl_acc":   [45.5, 12.3,  7.9],
            "rwmh_esjd": [2.07, 2.64, 2.73],
            "rl_esjd":   [2.12, 1.97, 1.78],
            "rwmh_flips": [2.61, 18.70, 24.58],
            "rl_flips":   [1.96, 12.32, 14.07],
        },
        "N6": {
            "title": r"$N=6,\ \omega=0.1,\ g=1.0$ (RL-MALA aprendió deriva hacia ADENTRO en $s_\sigma\geq 1.8$ $\Rightarrow$ empate / $+2\%$ a $+16\%$)",
            "rwmh_acc": [31.5, 9.2, 6.4],
            "rl_acc":   [31.5, 9.8, 6.0],
            "rwmh_esjd": [2.22, 2.84, 2.91],
            "rl_esjd":   [2.23, 3.30, 3.10],
            "rwmh_flips": [21.65, 57.04, 60.36],
            "rl_flips":   [17.76, 58.05, 63.26],
        },
    }

    plt.rcParams.update({"font.size": 9.2, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2), dpi=240)

    col_rwmh = "#e7298a"
    col_rl = "#1b9e77"

    for r_idx, key in enumerate(("N4", "N6")):
        d = data[key]
        panels = [
            ("Aceptación MH [%]", d["rwmh_acc"], d["rl_acc"], "{:.1f}%"),
            (r"ESJD [Bohr$^2$/paso]", d["rwmh_esjd"], d["rl_esjd"], "{:.2f}"),
            ("Flips totales / 1k pasos", d["rwmh_flips"], d["rl_flips"], "{:.2f}"),
        ]
        for c_idx, (ylabel, y_rw, y_rl, fmt) in enumerate(panels):
            ax = axes[r_idx, c_idx]
            b1 = ax.bar(x - w / 2, y_rw, width=w, color=col_rwmh, alpha=0.88, edgecolor="#222222", lw=0.7, label="RWMH puro (deriva=0, sin red)")
            b2 = ax.bar(x + w / 2, y_rl, width=w, color=col_rl, alpha=0.88, edgecolor="#222222", lw=0.7, label="RL-MALA paridad (Ayler 6k TD3)")
            for bars in (b1, b2):
                for b in bars:
                    h = b.get_height()
                    ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2.0, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.0, weight="bold")
            # Anotar diferencia porcentual RL vs RWMH arriba del par
            for xi, vrw, vrl in zip(x, y_rw, y_rl):
                pct = (vrl - vrw) / max(vrw, 1e-9) * 100.0
                col_txt = "#1b7837" if pct >= 0 else "#b2182b"
                ymax_pair = max(vrw, vrl)
                ax.annotate(
                    f"{pct:+.0f}%",
                    (xi, ymax_pair),
                    xytext=(0, 15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.2,
                    weight="bold",
                    color=col_txt,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=col_txt, alpha=0.90, lw=0.8),
                )
            ax.set_xticks(x)
            ax.set_xticklabels(sigmas, fontsize=8.4)
            ax.set_ylabel(ylabel, weight="bold")
            ax.set_ylim(0, max(max(y_rw), max(y_rl)) * 1.28)
            ax.grid(True, axis="y", alpha=0.25, ls="--")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="upper right", fontsize=8.2, framealpha=0.92)
            if c_idx == 1:
                ax.set_title(d["title"], fontsize=10.3, weight="bold", pad=12)

    fig.suptitle(
        r"Veredicto Crítico a Ancho Emparejado ($\sigma_{\mathrm{RWMH}} = \sigma_{\mathrm{RL}}$): ¿Supera RL-MALA Diagonal a un RWMH Isotrópico Simple?",
        fontsize=12.6,
        weight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.955], h_pad=2.4, w_pad=1.6)

    for ext in ("png", "pdf"):
        out_p = os.path.join(out_dir, f"fig_rwmh_vs_rlmala_matched_sigma_verdict.{ext}")
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        print(f"[+] Guardado: {out_p}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
