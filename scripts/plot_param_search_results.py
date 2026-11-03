#!/usr/bin/env python3
"""
scripts/plot_param_search_results.py
Genera los diagnósticos gráficos del Parameter Search en el espacio (actor_amplitude, tau):
1. Heatmaps 2D de KL Divergence (Total, Spin Up, Spin Down).
2. Heatmap 2D de tau_int (autocorrelación) y Tasa de Aceptación.
3. Curvas comparativas de Pareto: MALA vs RL-MALA para enfrentar los dos mejores samplers.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def generate_param_search_plots(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el archivo CSV: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"[+] Cargando resultados de Parameter Search desde: {csv_path} ({len(df)} modelos)")

    # Separar MALA y RL-MALA
    df_mala = df[df["method"].str.upper() == "MALA"].copy()
    df_rl = df[df["method"].str.upper().str.contains("TD3|RL")].copy()

    # Extraer tau y actor_amplitude numéricos
    df_mala["tau"] = pd.to_numeric(df_mala["tau"], errors="coerce")
    df_rl["tau"] = pd.to_numeric(df_rl["tau"], errors="coerce")
    df_rl["actor_amplitude"] = pd.to_numeric(df_rl["actor_amplitude"], errors="coerce")

    # Métricas clave
    metric_cols = ["kl_casscf_total_mean", "kl_casscf_up_mean", "kl_casscf_down_mean", "tau_int_E", "acceptance", "ESS_per_sec"]
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df_rl[c] = pd.to_numeric(df_rl[c], errors="coerce")
            if c in df_mala.columns:
                df_mala[c] = pd.to_numeric(df_mala[c], errors="coerce")

    # =========================================================================
    # FIGURA 1: Heatmaps 2D en el Espacio de Parámetros (Actor Amplitude vs Tau)
    # =========================================================================
    if not df_rl.empty and df_rl["actor_amplitude"].nunique() > 1 and df_rl["tau"].nunique() > 1:
        taus = np.sort(df_rl["tau"].unique())
        amps = np.sort(df_rl["actor_amplitude"].unique())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Panel (0,0): D_KL Total
        pivot_kl_tot = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="kl_casscf_total_mean")
        im0 = axes[0, 0].imshow(pivot_kl_tot.values, cmap="viridis_r", aspect="auto", norm=mcolors.LogNorm(vmin=max(1e-4, pivot_kl_tot.min().min()), vmax=max(1e-1, pivot_kl_tot.max().max())))
        fig.colorbar(im0, ax=axes[0, 0], label=r"$D_{\mathrm{KL}}(\rho_{\mathrm{tot}} \parallel \rho_{\mathrm{CASSCF}})$")
        axes[0, 0].set_title(r"Divergencia KL Total $D_{\mathrm{KL}}(\rho_{\mathrm{tot}})$", fontsize=12, weight="bold")

        # Panel (0,1): D_KL Spin Up
        if "kl_casscf_up_mean" in df_rl.columns and not df_rl["kl_casscf_up_mean"].isna().all():
            pivot_kl_up = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="kl_casscf_up_mean")
            im1 = axes[0, 1].imshow(pivot_kl_up.values, cmap="plasma_r", aspect="auto", norm=mcolors.LogNorm(vmin=max(1e-4, pivot_kl_up.min().min()), vmax=max(1e-1, pivot_kl_up.max().max())))
            fig.colorbar(im1, ax=axes[0, 1], label=r"$D_{\mathrm{KL}}(\rho_{\uparrow} \parallel \rho_{\uparrow}^{\mathrm{CASSCF}})$")
            axes[0, 1].set_title(r"Divergencia KL Spin $\uparrow$ ($D_{\mathrm{KL}}(\rho_{\uparrow})$)", fontsize=12, weight="bold")
        else:
            axes[0, 1].text(0.5, 0.5, "Datos de KL Spin Up no disponibles", ha="center")

        # Panel (1,0): Autocorrelación tau_int_E
        pivot_tau = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="tau_int_E")
        im2 = axes[1, 0].imshow(pivot_tau.values, cmap="magma_r", aspect="auto")
        fig.colorbar(im2, ax=axes[1, 0], label=r"Tiempo de Autocorrelación $\tau_{\mathrm{int},E}$ (pasos)")
        axes[1, 0].set_title(r"Autocorrelación de Energía $\tau_{\mathrm{int},E}$ (Menor es Mejor)", fontsize=12, weight="bold")

        # Panel (1,1): Tasa de Aceptación (%)
        pivot_acc = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="acceptance") * 100.0
        im3 = axes[1, 1].imshow(pivot_acc.values, cmap="RdYlGn", aspect="auto", vmin=30, vmax=95)
        fig.colorbar(im3, ax=axes[1, 1], label="Aceptación MCMC (%)")
        axes[1, 1].set_title("Tasa de Aceptación MCMC (%)", fontsize=12, weight="bold")

        # Ajustar ticks
        for ax in axes.flat:
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=45)
            ax.set_yticks(range(len(amps)))
            ax.set_yticklabels([f"{a:.2f}" for a in amps])
            ax.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=11)
            ax.set_ylabel("Amplitud del Actor ($A$)", fontsize=11)

        fig.tight_layout()
        fpath1 = os.path.join(out_dir, "fig_param_search_heatmaps_actor.png")
        fig.savefig(fpath1, dpi=250)
        plt.close(fig)
        print(f"[+] Guardado Heatmap 2D del Actor: {fpath1}")

    # =========================================================================
    # FIGURA 2: D_KL por Espín: Total vs Spin Up vs Spin Down
    # =========================================================================
    fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Slice a mejor amplitud del actor
    if not df_rl.empty:
        best_amp = df_rl.groupby("actor_amplitude")["kl_casscf_total_mean"].min().idxmin()
        df_best_amp = df_rl[df_rl["actor_amplitude"] == best_amp].sort_values("tau")
        
        ax2[0].plot(df_best_amp["tau"], df_best_amp["kl_casscf_total_mean"], "k-o", lw=2.2, label=r"$D_{\mathrm{KL}}$ Total")
        if "kl_casscf_up_mean" in df_best_amp.columns:
            ax2[0].plot(df_best_amp["tau"], df_best_amp["kl_casscf_up_mean"], "b--s", lw=1.8, label=r"$D_{\mathrm{KL}}$ Spin $\uparrow$")
        if "kl_casscf_down_mean" in df_best_amp.columns:
            ax2[0].plot(df_best_amp["tau"], df_best_amp["kl_casscf_down_mean"], "r:^", lw=1.8, label=r"$D_{\mathrm{KL}}$ Spin $\downarrow$")
            
        ax2[0].set_yscale("log")
        ax2[0].set_title(f"Divergencia KL por Espín en RL-MALA (Amplitud Óptima A={best_amp:.2f})", fontsize=11, weight="bold")
        ax2[0].set_xlabel(r"Tamaño de Paso $\tau$", fontsize=11)
        ax2[0].set_ylabel(r"$D_{\mathrm{KL}}(P \parallel P_{\mathrm{CASSCF}})$", fontsize=11)
        ax2[0].grid(True, alpha=0.3, which="both")
        ax2[0].legend(fontsize=9.5)

    # Enfrentamiento MALA vs RL-MALA
    if not df_mala.empty and not df_rl.empty:
        df_mala_sorted = df_mala.sort_values("tau")
        ax2[1].plot(df_mala_sorted["tau"], df_mala_sorted["kl_casscf_total_mean"], "gray", ls="--", marker="o", lw=2.0, label="MALA Estándar (Barrido $\\tau$)")
        ax2[1].plot(df_best_amp["tau"], df_best_amp["kl_casscf_total_mean"], "crimson", ls="-", marker="*", markersize=9, lw=2.4, label=f"RL-MALA TD3 ($A^*={best_amp:.2f}$)")
        
        ax2[1].set_yscale("log")
        ax2[1].set_title("Enfrentamiento de Precisión MCMC: MALA vs RL-MALA", fontsize=11, weight="bold")
        ax2[1].set_xlabel(r"Tamaño de Paso $\tau$", fontsize=11)
        ax2[1].set_ylabel(r"Divergencia KL Total $D_{\mathrm{KL}}$ (Menor es Mejor)", fontsize=11)
        ax2[1].grid(True, alpha=0.3, which="both")
        ax2[1].legend(fontsize=10)

    fig2.tight_layout()
    fpath2 = os.path.join(out_dir, "fig_param_search_spin_kl_comparison.png")
    fig2.savefig(fpath2, dpi=250)
    plt.close(fig2)
    print(f"[+] Guardada comparativa de KL por espín: {fpath2}")

    # =========================================================================
    # REPORTE DE MEJORES HIPERPARÁMETROS
    # =========================================================================
    best_rl_idx = df_rl["kl_casscf_total_mean"].idxmin() if not df_rl.empty else None
    best_mala_idx = df_mala["kl_casscf_total_mean"].idxmin() if not df_mala.empty else None

    print("\n" + "=" * 70)
    print(" RESUMEN DEL ENFRENTAMIENTO DE MEJORES SAMPLERS")
    print("=" * 70)
    if best_mala_idx is not None:
        b_m = df_mala.loc[best_mala_idx]
        print(f"[*] MEJOR MALA:")
        print(f"    tau = {b_m.get('tau', np.nan):.4f} | D_KL = {b_m.get('kl_casscf_total_mean', np.nan):.6f} | tau_int = {b_m.get('tau_int_E', np.nan):.2f} | Aceptación = {b_m.get('acceptance', 0)*100:.1f}%")
    if best_rl_idx is not None:
        b_r = df_rl.loc[best_rl_idx]
        print(f"[*] MEJOR RL-MALA (TD3):")
        print(f"    tau = {b_r.get('tau', np.nan):.4f} | Amp = {b_r.get('actor_amplitude', np.nan):.4f} | D_KL = {b_r.get('kl_casscf_total_mean', np.nan):.6f} | tau_int = {b_r.get('tau_int_E', np.nan):.2f} | Aceptación = {b_r.get('acceptance', 0)*100:.1f}%")
        if best_mala_idx is not None:
            kl_ratio = b_m.get('kl_casscf_total_mean', 1.0) / max(1e-12, b_r.get('kl_casscf_total_mean', 1.0))
            tau_ratio = b_m.get('tau_int_E', 1.0) / max(1e-12, b_r.get('tau_int_E', 1.0))
            print(f"--> Ganancia de Fidelidad KL (MALA/RL): {kl_ratio:.2f}x mejor")
            print(f"--> Ganancia de Aceleración tau_int:   {tau_ratio:.2f}x más rápido")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ploteo de resultados de parameter search")
    parser.add_argument("--csv", type=str, default="results_param_search/master_param_search_summary.csv", help="Ruta al CSV maestro de resumen")
    parser.add_argument("--out_dir", type=str, default="results_param_search/figures", help="Directorio de figuras")
    args = parser.parse_args()
    generate_param_search_plots(args.csv, args.out_dir)
