#!/usr/bin/env python3
"""
scripts/generate_parameter_search_summary_plots.py
Compila las figuras de síntesis global y análisis de hiperparámetros del muestreo
puro CASSCF (Cruce Nodal) a partir de los datos consolidados de Baker y Ayler.

Genera:
1. fig_hyperparameter_evolution_{system}.png (Evolución de métricas vs tau, A_mu con barras de error)
2. fig_param_search_heatmaps_{system}.png (Heatmaps 2D: A_mu vs tau, lambda_pen vs A_mu)
3. fig_penalty_sensitivity_{system}.png (Sensibilidad al factor de penalización lambda_pen)
4. fig_pareto_sampling_frontier.png (Frontera de Pareto global: ESJD vs Aceptación y D_KL vs tau_int)
5. fig_global_nodal_crossing_comparison.png (Comparativa directa N=4 vs N=6)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Configurar estilo visual académico limpio
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "axes.grid": True,
    "figure.autolayout": False,
})

OUT_DIR = "figures_param_search_summary"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "Wigner_N4_g4.0"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "MultiDet_N6_g1.0"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "global"), exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Carga y Normalización de Datos
# ------------------------------------------------------------------------------
def load_datasets():
    datasets = {}
    
    # Baker N4
    p_baker_n4 = "results_nodal_crossing_search_baker_N4_g4/csv_summary/nodal_crossing_summary_Wigner_N4_g4.0.csv"
    if os.path.exists(p_baker_n4):
        df = pd.read_csv(p_baker_n4)
        df["cluster"] = "baker"
        datasets["baker_n4"] = df
        
    # Baker N6
    b6_candidates = glob.glob("results_nodal_crossing_search_baker_N6_g1/*/nodal_crossing_summary_MultiDet_N6_g1.0.csv")
    if b6_candidates:
        df = pd.read_csv(b6_candidates[0])
        df["cluster"] = "baker"
        datasets["baker_n6"] = df
        
    # Ayler N4
    p_ayler_n4 = "results_nodal_crossing_search_ayler/csv_summary/nodal_crossing_summary_Wigner_N4_g4.0.csv"
    if os.path.exists(p_ayler_n4):
        df = pd.read_csv(p_ayler_n4)
        df["cluster"] = "ayler"
        datasets["ayler_n4"] = df
        
    # Ayler N6
    p_ayler_n6 = "results_nodal_crossing_search_ayler/csv_summary/nodal_crossing_summary_MultiDet_N6_g1.0.csv"
    if os.path.exists(p_ayler_n6):
        df = pd.read_csv(p_ayler_n6)
        df["cluster"] = "ayler"
        datasets["ayler_n6"] = df
        
    # Consolidar por sistema
    n4_dfs = []
    if "baker_n4" in datasets: n4_dfs.append(datasets["baker_n4"])
    if "ayler_n4" in datasets: n4_dfs.append(datasets["ayler_n4"])
    df_n4 = pd.concat(n4_dfs, ignore_index=True).drop_duplicates(subset=["label", "cluster"])
    
    n6_dfs = []
    if "baker_n6" in datasets: n6_dfs.append(datasets["baker_n6"])
    if "ayler_n6" in datasets: n6_dfs.append(datasets["ayler_n6"])
    df_n6 = pd.concat(n6_dfs, ignore_index=True).drop_duplicates(subset=["label", "cluster"])
    
    # Normalización de columnas comunes numéricas
    for df in [df_n4, df_n6]:
        df["is_neural"] = df["is_neural"].astype(str).str.lower().isin(["true", "1"])
        for col in ["tau", "actor_amplitude", "penalty_factor", "sigma_action_scale",
                    "acceptance", "ESJD", "tau_int_E", "ESS_per_sec", "kl_casscf_spin_mean",
                    "E_mean", "SE_E", "delta_E_vs_CASSCF"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
    return df_n4, df_n6

# ------------------------------------------------------------------------------
# 2. Figura de Evolución Paramétrica (4 Paneles con Barras de Error)
# ------------------------------------------------------------------------------
def plot_hyperparameter_evolution(df_sys, sys_name, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    
    df_mala = df_sys[~df_sys["is_neural"]].sort_values("tau")
    df_rl = df_sys[df_sys["is_neural"]].copy()
    
    # MALA baseline
    tau_m = df_mala["tau"].values
    spin_m = df_mala["kl_casscf_spin_mean"].values
    esjd_m = df_mala["ESJD"].values
    tau_int_m = df_mala["tau_int_E"].values
    acc_m = df_mala["acceptance"].values * 100.0
    
    # Paleta de colores para actor_amplitude
    amps = sorted(df_rl["actor_amplitude"].dropna().unique())
    colors = cm.viridis(np.linspace(0.15, 0.88, len(amps)))
    
    # Panel (0,0): D_KL Spin (Cruce Nodal) en escala logarítmica
    ax = axes[0, 0]
    ax.plot(tau_m, spin_m, "k--s", label="MALA Clásico (Línea Base)", linewidth=2.2, markersize=7)
    for amp, col in zip(amps, colors):
        sub = df_rl[df_rl["actor_amplitude"] == amp]
        # Promediar sobre penalty_factor para mostrar media y desviación estándar
        grouped = sub.groupby("tau")["kl_casscf_spin_mean"].agg(["mean", "std", "count"]).reset_index()
        taus = grouped["tau"].values
        means = grouped["mean"].values
        stds = grouped["std"].fillna(0).values
        
        ax.errorbar(taus, means, yerr=stds, fmt="-o", color=col, label=f"RL-MALA ($A_\\mu = {amp}$)",
                    capsize=4, elinewidth=1.5, linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ (Asimetría de Espín / Cruce Nodal)")
    ax.set_title(r"A) Restauración de Simetría Nodal ($D_{\mathrm{KL}}^{\mathrm{spin}} \to 0$)", weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    
    # Panel (0,1): ESJD (Movilidad)
    ax = axes[0, 1]
    ax.plot(tau_m, esjd_m, "k--s", label="MALA Clásico", linewidth=2.2, markersize=7)
    for amp, col in zip(amps, colors):
        sub = df_rl[df_rl["actor_amplitude"] == amp]
        grouped = sub.groupby("tau")["ESJD"].agg(["mean", "std"]).reset_index()
        taus = grouped["tau"].values
        means = grouped["mean"].values
        stds = grouped["std"].fillna(0).values
        ax.errorbar(taus, means, yerr=stds, fmt="-o", color=col, label=f"$A_\\mu = {amp}$",
                    capsize=4, elinewidth=1.5, linewidth=2.0)
    ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
    ax.set_ylabel(r"$\mathrm{ESJD}$ ($\text{Bohr}^2$)")
    ax.set_title(r"B) Movilidad en el Espacio de Configuración ($\mathrm{ESJD}$)", weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    
    # Panel (1,0): Autocorrelación de Energía tau_int_E
    ax = axes[1, 0]
    ax.plot(tau_m, tau_int_m, "k--s", label="MALA Clásico", linewidth=2.2, markersize=7)
    for amp, col in zip(amps, colors):
        sub = df_rl[df_rl["actor_amplitude"] == amp]
        grouped = sub.groupby("tau")["tau_int_E"].agg(["mean", "std"]).reset_index()
        taus = grouped["tau"].values
        means = grouped["mean"].values
        stds = grouped["std"].fillna(0).values
        ax.errorbar(taus, means, yerr=stds, fmt="-o", color=col, label=f"$A_\\mu = {amp}$",
                    capsize=4, elinewidth=1.5, linewidth=2.0)
    ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
    ax.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (Pasos de Autocorrelación)")
    ax.set_title(r"C) Longitud de Autocorrelación de Energía (Menor es Mejor)", weight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    
    # Panel (1,1): Tasa de Aceptación (%)
    ax = axes[1, 1]
    ax.axhspan(35, 55, color="green", alpha=0.12, label="Régimen Óptimo No Local (35-55%)")
    ax.plot(tau_m, acc_m, "k--s", label="MALA Clásico", linewidth=2.2, markersize=7)
    for amp, col in zip(amps, colors):
        sub = df_rl[df_rl["actor_amplitude"] == amp]
        grouped = sub.groupby("tau")["acceptance"].agg(["mean", "std"]).reset_index()
        taus = grouped["tau"].values
        means = grouped["mean"].values * 100.0
        stds = grouped["std"].fillna(0).values * 100.0
        ax.errorbar(taus, means, yerr=stds, fmt="-o", color=col, label=f"$A_\\mu = {amp}$",
                    capsize=4, elinewidth=1.5, linewidth=2.0)
    ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
    ax.set_ylabel("Tasa de Aceptación (%)")
    ax.set_title(r"D) Aceptación Metropolis-Hastings (%)", weight="bold")
    ax.set_ylim(bottom=15, top=100)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    
    fig.suptitle(f"Evolución Paramétrica del Sampler MALA vs RL-MALA ({sys_name})", weight="bold", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardada figura de evolución: {out_path}")

# ------------------------------------------------------------------------------
# 3. Mapas de Calor 2D (Actor Amplitude vs Tau)
# ------------------------------------------------------------------------------
def plot_heatmaps_2d(df_sys, sys_name, out_path):
    df_rl = df_sys[df_sys["is_neural"]].copy()
    if df_rl.empty or df_rl["actor_amplitude"].nunique() < 2 or df_rl["tau"].nunique() < 2:
        print(f"[Aviso] Insuficientes datos para heatmap en {sys_name}")
        return
        
    taus = np.sort(df_rl["tau"].dropna().unique())
    amps = np.sort(df_rl["actor_amplitude"].dropna().unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # 1. log10(D_KL_spin)
    pivot_kl = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="kl_casscf_spin_mean", aggfunc="min")
    vmin_kl = np.nanmin(pivot_kl.values)
    vmax_kl = np.nanmax(pivot_kl.values)
    im0 = axes[0, 0].imshow(pivot_kl.values, cmap="plasma_r", aspect="auto",
                            norm=mcolors.LogNorm(vmin=max(1e-5, vmin_kl), vmax=max(1e-2, vmax_kl)))
    cbar0 = fig.colorbar(im0, ax=axes[0, 0])
    cbar0.set_label(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ (Menor es Mejor)", weight="bold")
    axes[0, 0].set_title(r"A) Asimetría Nodal $\min_{\lambda_{\mathrm{pen}}} D_{\mathrm{KL}}^{\mathrm{spin}}$", weight="bold")
    
    # 2. ESJD
    pivot_esjd = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="ESJD", aggfunc="max")
    im1 = axes[0, 1].imshow(pivot_esjd.values, cmap="viridis", aspect="auto")
    cbar1 = fig.colorbar(im1, ax=axes[0, 1])
    cbar1.set_label(r"$\mathrm{ESJD}$ ($\text{Bohr}^2$, Mayor es Mejor)", weight="bold")
    axes[0, 1].set_title(r"B) Movilidad Cuadrática Media $\max_{\lambda} \mathrm{ESJD}$", weight="bold")
    
    # 3. tau_int_E
    pivot_tau_int = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="tau_int_E", aggfunc="min")
    im2 = axes[1, 0].imshow(pivot_tau_int.values, cmap="magma_r", aspect="auto")
    cbar2 = fig.colorbar(im2, ax=axes[1, 0])
    cbar2.set_label(r"$\tau_{\mathrm{int}}(E)$ (Pasos, Menor es Mejor)", weight="bold")
    axes[1, 0].set_title(r"C) Tiempo de Autocorrelación $\min_{\lambda} \tau_{\mathrm{int}}(E)$", weight="bold")
    
    # 4. Aceptación (%)
    pivot_acc = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="acceptance", aggfunc="mean") * 100.0
    im3 = axes[1, 1].imshow(pivot_acc.values, cmap="RdYlGn", aspect="auto", vmin=30, vmax=65)
    cbar3 = fig.colorbar(im3, ax=axes[1, 1])
    cbar3.set_label(r"Aceptación Promedio (%)", weight="bold")
    axes[1, 1].set_title(r"D) Tasa de Aceptación MCMC Media (%)", weight="bold")
    
    # Anotar ticks y etiquetas
    for ax in axes.flat:
        ax.set_xticks(range(len(taus)))
        ax.set_xticklabels([f"{t:.2f}" for t in taus])
        ax.set_yticks(range(len(amps)))
        ax.set_yticklabels([f"{a:.2f}" for a in amps])
        ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
        ax.set_ylabel(r"Amplitud del Actor $A_\mu$")
        
    fig.suptitle(f"Mapas de Calor 2D en el Espacio de Parámetros ({sys_name})", weight="bold", fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado heatmap 2D: {out_path}")

# ------------------------------------------------------------------------------
# 4. Sensibilidad al Factor de Penalización (lambda_pen)
# ------------------------------------------------------------------------------
def plot_penalty_sensitivity(df_sys, sys_name, out_path):
    df_rl = df_sys[df_sys["is_neural"]].copy()
    if df_rl.empty or df_rl["penalty_factor"].nunique() < 2:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    
    pens = sorted(df_rl["penalty_factor"].dropna().unique())
    taus = sorted(df_rl["tau"].dropna().unique())
    colors = cm.tab10(np.linspace(0, 0.8, len(taus)))
    
    for tau, col in zip(taus, colors):
        sub = df_rl[df_rl["tau"] == tau]
        # Agrupar por penalty_factor
        grp = sub.groupby("penalty_factor")[["acceptance", "kl_casscf_spin_mean"]].agg(["mean", "std"]).reset_index()
        p_vals = grp["penalty_factor"].values
        acc_means = grp["acceptance"]["mean"].values * 100.0
        acc_stds = grp["acceptance"]["std"].fillna(0).values * 100.0
        
        spin_means = grp["kl_casscf_spin_mean"]["mean"].values
        spin_stds = grp["kl_casscf_spin_mean"]["std"].fillna(0).values
        
        ax1.errorbar(p_vals, acc_means, yerr=acc_stds, fmt="-o", color=col, label=f"$\\tau = {tau}$", capsize=4)
        ax2.errorbar(p_vals, spin_means, yerr=spin_stds, fmt="-s", color=col, label=f"$\\tau = {tau}$", capsize=4)
        
    ax1.set_xlabel(r"Penalización de Rechazo $\lambda_{\mathrm{pen}}$")
    ax1.set_ylabel("Aceptación MCMC Media (%)")
    ax1.set_title(r"A) Efecto de $\lambda_{\mathrm{pen}}$ en la Aceptación", weight="bold")
    ax1.legend(frameon=True)
    
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Penalización de Rechazo $\lambda_{\mathrm{pen}}$")
    ax2.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal)")
    ax2.set_title(r"B) Efecto de $\lambda_{\mathrm{pen}}$ en el Cruce Nodal", weight="bold")
    ax2.legend(frameon=True)
    
    fig.suptitle(f"Sensibilidad al Parámetro de Recompensa CDLB $\\lambda_{{\\mathrm{{pen}}}}$ ({sys_name})", weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardada sensibilidad de penalización: {out_path}")

# ------------------------------------------------------------------------------
# 5. Frontera de Pareto Global y Comparativa Directa N4 vs N6
# ------------------------------------------------------------------------------
def plot_pareto_frontier(df_n4, df_n6, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    
    # Subplot 1: ESJD vs Aceptación
    # MALA
    for df, name, marker in [(df_n4, "Wigner N=4", "o"), (df_n6, "MultiDet N=6", "s")]:
        m = df[~df["is_neural"]]
        r = df[df["is_neural"]]
        
        ax1.scatter(m["acceptance"] * 100.0, m["ESJD"], color="black", marker=marker,
                    s=80, alpha=0.7, edgecolors="k", label=f"MALA ({name})")
        sc = ax1.scatter(r["acceptance"] * 100.0, r["ESJD"], c=r["tau"], cmap="plasma",
                         marker=marker, s=85, alpha=0.85, edgecolors="none", label=f"RL-MALA ({name})")
        
    ax1.axvspan(35, 55, color="green", alpha=0.08, label="Banda Óptima Langevin")
    ax1.set_xlabel("Tasa de Aceptación (%)")
    ax1.set_ylabel(r"$\mathrm{ESJD}$ ($\text{Bohr}^2$)")
    ax1.set_title(r"A) Movilidad Cuadrática vs Aceptación ($\mathrm{ESJD}$ Frontier)", weight="bold")
    cbar1 = fig.colorbar(sc, ax=ax1)
    cbar1.set_label(r"Paso Temporal Base $\tau$")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.8)
    
    # Subplot 2: Cruce Nodal vs Autocorrelación
    for df, name, marker in [(df_n4, "Wigner N=4", "o"), (df_n6, "MultiDet N=6", "s")]:
        m = df[~df["is_neural"]]
        r = df[df["is_neural"]]
        
        ax2.scatter(m["tau_int_E"], m["kl_casscf_spin_mean"], color="black", marker=marker,
                    s=80, alpha=0.7, edgecolors="k", label=f"MALA ({name})")
        sc2 = ax2.scatter(r["tau_int_E"], r["kl_casscf_spin_mean"], c=r["tau"], cmap="plasma",
                          marker=marker, s=85, alpha=0.85, edgecolors="none", label=f"RL-MALA ({name})")
                          
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ (pasos)")
    ax2.set_ylabel(r"Asimetría de Espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal)")
    ax2.set_title(r"B) Eficiencia de Cruce Nodal vs Autocorrelación", weight="bold")
    cbar2 = fig.colorbar(sc2, ax=ax2)
    cbar2.set_label(r"Paso Temporal Base $\tau$")
    ax2.legend(loc="upper left", frameon=True, fontsize=8.8)
    
    fig.suptitle("Frontera de Eficiencia de Pareto: Dominancia Global de RL-MALA frente a MALA", weight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardada frontera de Pareto: {out_path}")

# ------------------------------------------------------------------------------
# 6. Comparativa Cuantitativa de Cruce Nodal (N4 vs N6)
# ------------------------------------------------------------------------------
def plot_global_nodal_crossing_summary(df_n4, df_n6, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))
    
    systems = [("Wigner $N=4, g=4.0$", df_n4), ("MultiDet $N=6, g=1.0$", df_n6)]
    
    # Gráfica de Barras Comparativa: Mejor MALA vs Mejor RL-MALA
    labels = []
    spin_kl_mala = []
    spin_kl_rl = []
    esjd_mala = []
    esjd_rl = []
    
    for name, df in systems:
        labels.append(name)
        m = df[~df["is_neural"]]
        r = df[df["is_neural"]]
        
        best_m_idx = m["kl_casscf_spin_mean"].idxmin()
        best_r_idx = r["kl_casscf_spin_mean"].idxmin()
        
        spin_kl_mala.append(m.loc[best_m_idx, "kl_casscf_spin_mean"])
        spin_kl_rl.append(r.loc[best_r_idx, "kl_casscf_spin_mean"])
        
        esjd_mala.append(m.loc[best_m_idx, "ESJD"])
        esjd_rl.append(r.loc[best_r_idx, "ESJD"])
        
    x = np.arange(len(labels))
    width = 0.32
    
    # Panel 1: D_KL Spin en escala logarítmica
    rects1 = ax1.bar(x - width/2, spin_kl_mala, width, label="Mejor MALA Clásico", color="#4a4a4a", edgecolor="black")
    rects2 = ax1.bar(x + width/2, spin_kl_rl, width, label="Mejor RL-MALA (TD3 DeepSet)", color="#1b7837", edgecolor="black")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ (Asimetría de Espín, Menor es Mejor)")
    ax1.set_title(r"A) Factor de Restauración Nodal ($D_{\mathrm{KL}}^{\mathrm{spin}}$)", weight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(frameon=True)
    
    # Agregar factor de mejora encima de las barras
    for i in range(len(labels)):
        factor = spin_kl_mala[i] / spin_kl_rl[i]
        val = spin_kl_rl[i]
        ax1.annotate(f"{factor:.1f}x mejor",
                     xy=(x[i] + width/2, val),
                     xytext=(0, 10), textcoords="offset points",
                     ha="center", va="bottom", weight="bold", color="#1b7837", fontsize=10.5)
                     
    # Panel 2: ESJD (Movilidad)
    rects3 = ax2.bar(x - width/2, esjd_mala, width, label="Mejor MALA", color="#4a4a4a", edgecolor="black")
    rects4 = ax2.bar(x + width/2, esjd_rl, width, label="Mejor RL-MALA", color="#2166ac", edgecolor="black")
    ax2.set_ylabel(r"$\mathrm{ESJD}$ ($\text{Bohr}^2$, Mayor es Mejor)")
    ax2.set_title(r"B) Movilidad en el Espacio de Configuración ($\mathrm{ESJD}$)", weight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(frameon=True)
    
    for i in range(len(labels)):
        factor = esjd_rl[i] / esjd_mala[i]
        val = esjd_rl[i]
        ax2.annotate(f"+{factor:.2f}x",
                     xy=(x[i] + width/2, val),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", weight="bold", color="#2166ac", fontsize=10.5)
                     
    fig.suptitle("Comparativa Cuantitativa de Muestreo Puro: MALA vs RL-MALA", weight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Guardado resumen global de cruce nodal: {out_path}")

# ------------------------------------------------------------------------------
# Ejecución Principal
# ------------------------------------------------------------------------------
def main():
    print("==============================================================================")
    print("COMPILANDO FIGURAS DE SÍNTESIS GLOBAL DE PARAMETER SEARCH (BAKER + AYLER)")
    print("==============================================================================")
    df_n4, df_n6 = load_datasets()
    print(f"[+] Datos N=4 cargados: {len(df_n4)} corridas")
    print(f"[+] Datos N=6 cargados: {len(df_n6)} corridas")
    
    # 1. Figuras para Wigner N=4
    p_evo_n4 = os.path.join(OUT_DIR, "Wigner_N4_g4.0", "fig_hyperparameter_evolution_Wigner_N4_g4.0.png")
    plot_hyperparameter_evolution(df_n4, "Wigner $N=4, g=4.0$", p_evo_n4)
    
    p_heat_n4 = os.path.join(OUT_DIR, "Wigner_N4_g4.0", "fig_param_search_heatmaps_Wigner_N4_g4.0.png")
    plot_heatmaps_2d(df_n4, "Wigner $N=4, g=4.0$", p_heat_n4)
    
    p_pen_n4 = os.path.join(OUT_DIR, "Wigner_N4_g4.0", "fig_penalty_sensitivity_Wigner_N4_g4.0.png")
    plot_penalty_sensitivity(df_n4, "Wigner $N=4, g=4.0$", p_pen_n4)
    
    # 2. Figuras para MultiDet N=6
    p_evo_n6 = os.path.join(OUT_DIR, "MultiDet_N6_g1.0", "fig_hyperparameter_evolution_MultiDet_N6_g1.0.png")
    plot_hyperparameter_evolution(df_n6, "MultiDet $N=6, g=1.0$", p_evo_n6)
    
    p_heat_n6 = os.path.join(OUT_DIR, "MultiDet_N6_g1.0", "fig_param_search_heatmaps_MultiDet_N6_g1.0.png")
    plot_heatmaps_2d(df_n6, "MultiDet $N=6, g=1.0$", p_heat_n6)
    
    p_pen_n6 = os.path.join(OUT_DIR, "MultiDet_N6_g1.0", "fig_penalty_sensitivity_MultiDet_N6_g1.0.png")
    plot_penalty_sensitivity(df_n6, "MultiDet $N=6, g=1.0$", p_pen_n6)
    
    # 3. Figuras Globales Consolidadas
    p_pareto = os.path.join(OUT_DIR, "global", "fig_pareto_sampling_frontier.png")
    plot_pareto_frontier(df_n4, df_n6, p_pareto)
    
    p_global = os.path.join(OUT_DIR, "global", "fig_global_nodal_crossing_comparison.png")
    plot_global_nodal_crossing_summary(df_n4, df_n6, p_global)
    
    print("==============================================================================")
    print(f"Figuras guardadas exitosamente en: {OUT_DIR}")
    print("==============================================================================")

if __name__ == "__main__":
    main()
