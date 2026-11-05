#!/usr/bin/env python3
"""
scripts/plot_param_search_results.py
Genera y actualiza en tiempo real los diagnósticos gráficos globales del Parameter Search:
1. Heatmaps 2D de KL Divergence (Total, Spin-Resolved), tau_int y Tasa de Aceptación por sistema.
2. Enfrentamiento directo MALA (línea discontinua '--') vs RL-MALA (línea continua '-') vs tau y A_mu.
3. Panorama global multi-sistema conforme van terminando de converger los modelos y sistemas.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _save_fig_custom(
    fig: plt.Figure,
    out_dir: str,
    stem: str,
    args: argparse.Namespace,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if getattr(args, "no_titles", False):
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_text("")
        for ax in fig.axes:
            ax.set_title("")
    else:
        custom_sup = str(getattr(args, "custom_suptitle", "") or "").strip()
        t_prefix = str(getattr(args, "title_prefix", "") or "").strip()
        f_scale = float(getattr(args, "font_scale", 1.0))
        if custom_sup:
            full_sup = f"{t_prefix} — {custom_sup}" if t_prefix else custom_sup
            fig.suptitle(full_sup, fontsize=14.0 * f_scale, weight="bold", y=1.02)
        elif t_prefix:
            if getattr(fig, "_suptitle", None) is not None and fig._suptitle.get_text():
                cur = fig._suptitle.get_text()
                if not cur.startswith(t_prefix):
                    fig._suptitle.set_text(f"{t_prefix} — {cur}")
            else:
                fig.suptitle(t_prefix, fontsize=13.5 * f_scale, weight="bold", y=1.02)

    fmt = getattr(args, "figure_format", "png")
    dpi = int(getattr(args, "dpi", 300))
    transparent = bool(getattr(args, "transparent_bg", False))
    if fmt in ("png", "both"):
        fig.savefig(
            os.path.join(out_dir, f"{stem}.png"),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
    if fmt in ("pdf", "both"):
        fig.savefig(
            os.path.join(out_dir, f"{stem}.pdf"),
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )
    plt.close(fig)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" in df.columns:
        df = df.drop_duplicates(subset=["label"], keep="last").reset_index(drop=True)

    if "is_neural" not in df.columns:
        if "method" in df.columns:
            df["is_neural"] = df["method"].astype(str).str.upper().str.contains("TD3|RL")
        elif "proposal_type" in df.columns:
            df["is_neural"] = df["proposal_type"].astype(str).str.lower().str.contains("neural|deepset")
        else:
            df["is_neural"] = df["label"].astype(str).str.upper().str.contains("RL_MALA|TD3")
    else:
        df["is_neural"] = df["is_neural"].astype(str).str.lower().isin(["true", "1", "yes"])

    def _extract_num(pattern: str, text: str, default: float = np.nan) -> float:
        m = re.search(pattern, str(text), flags=re.IGNORECASE)
        return float(m.group(1)) if m else default

    if "tau" not in df.columns:
        df["tau"] = [
            _extract_num(r"tau([0-9.]+)", lbl, 0.10) for lbl in df.get("label", [])
        ]
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")

    if "actor_amplitude" not in df.columns:
        df["actor_amplitude"] = [
            0.0 if not isn else _extract_num(r"amp([0-9.]+)", lbl, 0.10)
            for isn, lbl in zip(df["is_neural"], df.get("label", []))
        ]
    df["actor_amplitude"] = pd.to_numeric(df["actor_amplitude"], errors="coerce").fillna(0.0)

    if "penalty_factor" not in df.columns:
        df["penalty_factor"] = [
            0.0 if not isn else _extract_num(r"pen([0-9.]+)", lbl, 0.25)
            for isn, lbl in zip(df["is_neural"], df.get("label", []))
        ]
    df["penalty_factor"] = pd.to_numeric(df["penalty_factor"], errors="coerce").fillna(0.0)

    if "sys_group" not in df.columns:
        if "N" in df.columns and "g" in df.columns:
            df["sys_group"] = [
                f"N{int(n)}_g{float(g):.1f}"
                for n, g in zip(
                    pd.to_numeric(df["N"], errors="coerce").fillna(4),
                    pd.to_numeric(df["g"], errors="coerce").fillna(4.0),
                )
            ]
        else:
            df["sys_group"] = "All"

    metric_cols = [
        "kl_casscf_total_mean",
        "kl_casscf_total_std",
        "kl_casscf_up_mean",
        "kl_casscf_down_mean",
        "kl_casscf_spin_mean",
        "kl_casscf_spin_std",
        "tau_int_E",
        "tau_int_E_std",
        "acceptance",
        "ESJD",
        "ESJD_std",
        "ESS_per_sec",
        "ESS_per_sec_std",
        "E_mean",
        "SE_E",
    ]
    for c in metric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "kl_casscf_spin_mean" not in df.columns and "kl_casscf_up_mean" in df.columns:
        df["kl_casscf_spin_mean"] = 0.5 * (
            df["kl_casscf_up_mean"] + df.get("kl_casscf_down_mean", df["kl_casscf_up_mean"])
        )
    return df


def generate_param_search_plots(csv_path: str, out_dir: str, args: argparse.Namespace) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"[plot_param_search_results] Aún no existe o está vacío: {csv_path}")
        return

    f_scale = float(getattr(args, "font_scale", 1.0))
    lw_scale = float(getattr(args, "line_width_scale", 1.0))
    plt.rcParams.update(
        {
            "font.size": 10.5 * f_scale,
            "axes.titlesize": 12.0 * f_scale,
            "axes.labelsize": 10.5 * f_scale,
            "legend.fontsize": 8.8 * f_scale,
            "lines.linewidth": 1.9 * lw_scale,
        }
    )

    df = _normalize_dataframe(pd.read_csv(csv_path))
    if df.empty:
        return
    print(f"[+] Cargando resultados de Parameter Search desde: {csv_path} ({len(df)} modelos únicos)")

    for sys_name, df_sys in df.groupby("sys_group"):
        df_mala = df_sys[~df_sys["is_neural"]].sort_values("tau").copy()
        df_rl = df_sys[df_sys["is_neural"]].sort_values(["tau", "actor_amplitude"]).copy()
        sys_suffix = f"_{sys_name}" if df["sys_group"].nunique() > 1 else ""

        # =========================================================================
        # FIGURA 1: Heatmaps 2D en el Espacio de Parámetros (Actor Amplitude vs Tau)
        # =========================================================================
        if not df_rl.empty and df_rl["actor_amplitude"].nunique() > 1 and df_rl["tau"].nunique() > 1:
            taus = np.sort(df_rl["tau"].unique())
            amps = np.sort(df_rl["actor_amplitude"].unique())

            fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))

            # Panel (0,0): D_KL Total (promediado o mínimo sobre penalty_factor)
            pivot_kl_tot = df_rl.pivot_table(
                index="actor_amplitude", columns="tau", values="kl_casscf_total_mean", aggfunc="min"
            )
            vmin_t = max(1e-4, float(np.nanmin(pivot_kl_tot.values)))
            vmax_t = max(vmin_t * 2.0, float(np.nanmax(pivot_kl_tot.values)))
            im0 = axes[0, 0].imshow(
                pivot_kl_tot.values,
                cmap="viridis_r",
                aspect="auto",
                norm=mcolors.LogNorm(vmin=vmin_t, vmax=vmax_t),
            )
            fig.colorbar(im0, ax=axes[0, 0], label=r"$D_{\mathrm{KL}}(\rho_{\mathrm{tot}} \parallel \rho_{\mathrm{CASSCF}})$")
            axes[0, 0].set_title(r"Divergencia KL Total $D_{\mathrm{KL}}(\rho_{\mathrm{tot}})$", weight="bold")

            # Panel (0,1): D_KL Spin-Averaged (Cruce Nodal)
            spin_col = "kl_casscf_spin_mean" if "kl_casscf_spin_mean" in df_rl.columns else "kl_casscf_up_mean"
            if spin_col in df_rl.columns and df_rl[spin_col].notna().any():
                pivot_kl_s = df_rl.pivot_table(
                    index="actor_amplitude", columns="tau", values=spin_col, aggfunc="min"
                )
                vmin_s = max(1e-4, float(np.nanmin(pivot_kl_s.values)))
                vmax_s = max(vmin_s * 2.0, float(np.nanmax(pivot_kl_s.values)))
                im1 = axes[0, 1].imshow(
                    pivot_kl_s.values,
                    cmap="plasma_r",
                    aspect="auto",
                    norm=mcolors.LogNorm(vmin=vmin_s, vmax=vmax_s),
                )
                fig.colorbar(im1, ax=axes[0, 1], label=r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ (Cruce Nodal)")
                axes[0, 1].set_title(r"Cruce Nodal: $D_{\mathrm{KL}}^{\mathrm{spin}}$ (Menor es Mejor)", weight="bold")
            else:
                axes[0, 1].text(0.5, 0.5, "Datos de KL Spin no disponibles", ha="center")

            # Panel (1,0): Autocorrelación tau_int_E
            pivot_tau = df_rl.pivot_table(
                index="actor_amplitude", columns="tau", values="tau_int_E", aggfunc="min"
            )
            im2 = axes[1, 0].imshow(pivot_tau.values, cmap="magma_r", aspect="auto")
            fig.colorbar(im2, ax=axes[1, 0], label=r"Tiempo de Autocorrelación $\tau_{\mathrm{int},E}$ (pasos)")
            axes[1, 0].set_title(r"Autocorrelación de Energía $\tau_{\mathrm{int},E}$ (Menor es Mejor)", weight="bold")

            # Panel (1,1): Tasa de Aceptación (%)
            pivot_acc = df_rl.pivot_table(
                index="actor_amplitude", columns="tau", values="acceptance", aggfunc="mean"
            ) * 100.0
            im3 = axes[1, 1].imshow(pivot_acc.values, cmap="RdYlGn", aspect="auto", vmin=30, vmax=95)
            fig.colorbar(im3, ax=axes[1, 1], label="Aceptación MCMC (%)")
            axes[1, 1].set_title("Tasa de Aceptación MCMC (%)", weight="bold")

            for ax in axes.flat:
                ax.set_xticks(range(len(taus)))
                ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=35)
                ax.set_yticks(range(len(amps)))
                ax.set_yticklabels([f"{a:.2f}" for a in amps])
                ax.set_xlabel(r"Tamaño de Paso Base $\tau$")
                ax.set_ylabel(r"Amplitud del Actor $A_\mu$")

            fig.suptitle(f"Mapa de Calor 2D del Actor Neuronal ({sys_name})", fontsize=14, weight="bold", y=1.01)
            fig.tight_layout()
            _save_fig_custom(fig, out_dir, f"fig_param_search_heatmaps_actor{sys_suffix}", args)

        # =========================================================================
        # FIGURA 2: Enfrentamiento MALA (discontinua '--') vs RL-MALA (continua '-')
        # =========================================================================
        fig2, ax2 = plt.subplots(1, 2, figsize=(15.5, 5.6))

        if not df_rl.empty and "kl_casscf_spin_mean" in df_rl.columns and df_rl["kl_casscf_spin_mean"].notna().any():
            best_amp = df_rl.groupby("actor_amplitude")["kl_casscf_spin_mean"].min().idxmin()
            df_best_amp = (
                df_rl[df_rl["actor_amplitude"] == best_amp]
                .sort_values("kl_casscf_spin_mean")
                .drop_duplicates(subset=["tau"], keep="first")
                .sort_values("tau")
            )

            ax2[0].plot(
                df_best_amp["tau"],
                df_best_amp["kl_casscf_total_mean"],
                "k-o",
                lw=2.2,
                label=r"RL-MALA [-] $D_{\mathrm{KL}}$ Total",
            )
            if "kl_casscf_up_mean" in df_best_amp.columns:
                ax2[0].plot(
                    df_best_amp["tau"],
                    df_best_amp["kl_casscf_up_mean"],
                    "-s",
                    color="#2980b9",
                    lw=2.0,
                    label=r"RL-MALA [-] $D_{\mathrm{KL}}$ Spin $\uparrow$",
                )
            if "kl_casscf_down_mean" in df_best_amp.columns:
                ax2[0].plot(
                    df_best_amp["tau"],
                    df_best_amp["kl_casscf_down_mean"],
                    "-^",
                    color="#27ae60",
                    lw=2.0,
                    label=r"RL-MALA [-] $D_{\mathrm{KL}}$ Spin $\downarrow$",
                )
            if not df_mala.empty and "kl_casscf_spin_mean" in df_mala.columns:
                ax2[0].plot(
                    df_mala["tau"],
                    df_mala["kl_casscf_spin_mean"],
                    "--X",
                    color="#e74c3c",
                    lw=2.2,
                    label=r"MALA [--] $D_{\mathrm{KL}}^{\mathrm{spin}}$",
                )

            ax2[0].set_yscale("log")
            ax2[0].set_title(
                f"Divergencia KL por Espín ({sys_name}, $A_\\mu^*={best_amp:.2f}$)",
                weight="bold",
            )
            ax2[0].set_xlabel(r"Tamaño de Paso Base $\tau$")
            ax2[0].set_ylabel(r"$D_{\mathrm{KL}}(P \parallel P_{\mathrm{CASSCF}})$ (log)")
            ax2[0].grid(True, alpha=0.3, which="both")
            ax2[0].legend(fontsize=9.0, frameon=True)

            # Panel derecho: Autocorrelación tau_int(E) MALA ('--') vs RL-MALA ('-')
            if not df_mala.empty:
                ax2[1].errorbar(
                    df_mala["tau"],
                    df_mala["tau_int_E"],
                    yerr=df_mala.get("tau_int_E_std", pd.Series(np.zeros(len(df_mala)))).fillna(0.0),
                    fmt="--X",
                    color="#e74c3c",
                    lw=2.3,
                    markersize=8,
                    capsize=4,
                    label="MALA Estándar [--] (Barrido $\\tau$)",
                )
            ax2[1].errorbar(
                df_best_amp["tau"],
                df_best_amp["tau_int_E"],
                yerr=df_best_amp.get("tau_int_E_std", pd.Series(np.zeros(len(df_best_amp)))).fillna(0.0),
                fmt="-o",
                color="#2ecc71",
                lw=2.5,
                markersize=8,
                capsize=4,
                label=f"RL-MALA TD3 [-] ($A_\\mu^*={best_amp:.2f}$)",
            )
            ax2[1].set_yscale("log")
            ax2[1].set_title(
                f"Autocorrelación $\\tau_{{\\mathrm{{int}}}}(E)$: MALA [--] vs RL-MALA [-] ({sys_name})",
                weight="bold",
            )
            ax2[1].set_xlabel(r"Tamaño de Paso Base $\tau$")
            ax2[1].set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (pasos, Menor es Mejor)")
            ax2[1].grid(True, alpha=0.3, which="both")
            ax2[1].legend(fontsize=9.5, frameon=True)

        fig2.tight_layout()
        _save_fig_custom(fig2, out_dir, f"fig_param_search_spin_kl_comparison{sys_suffix}", args)

    # =========================================================================
    # FIGURA 3: Panorama Global de Todos los Modelos / Sistemas Convergidos
    # =========================================================================
    if len(df) >= 2:
        fig3, axes3 = plt.subplots(1, 2, figsize=(15.5, 5.8))
        df_m_all = df[~df["is_neural"]]
        df_r_all = df[df["is_neural"]]

        # Scatter Pareto: tau_int_E vs D_KL^spin
        if "kl_casscf_spin_mean" in df.columns and df["kl_casscf_spin_mean"].notna().any():
            if not df_m_all.empty:
                axes3[0].scatter(
                    df_m_all["tau_int_E"],
                    df_m_all["kl_casscf_spin_mean"],
                    c="#e74c3c",
                    marker="X",
                    s=95,
                    alpha=0.85,
                    edgecolors="black",
                    label="MALA Clásico [--]",
                )
            if not df_r_all.empty:
                axes3[0].scatter(
                    df_r_all["tau_int_E"],
                    df_r_all["kl_casscf_spin_mean"],
                    c="#2ecc71",
                    marker="o",
                    s=85,
                    alpha=0.85,
                    edgecolors="black",
                    label="RL-MALA TD3 [-]",
                )
            axes3[0].set_xscale("log")
            axes3[0].set_yscale("log")
            axes3[0].set_xlabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ (pasos, log)")
            axes3[0].set_ylabel(r"Divergencia KL de Espín $D_{\mathrm{KL}}^{\mathrm{spin}}$ (log)")
            axes3[0].set_title("Frente de Pareto Global: Ergodicidad vs Cruce Nodal", weight="bold")
            axes3[0].grid(True, which="both", alpha=0.3)
            axes3[0].legend(frameon=True)

        # Barras de los mejores modelos por sistema (Mejor MALA vs Mejor RL-MALA)
        sys_names = sorted(df["sys_group"].unique())
        x_idx = np.arange(len(sys_names))
        width = 0.35
        mala_best_tau = []
        rl_best_tau = []
        for s_n in sys_names:
            sub_s = df[df["sys_group"] == s_n]
            sub_m = sub_s[~sub_s["is_neural"]]
            sub_r = sub_s[sub_s["is_neural"]]
            mala_best_tau.append(float(sub_m["tau_int_E"].min()) if not sub_m.empty else np.nan)
            rl_best_tau.append(float(sub_r["tau_int_E"].min()) if not sub_r.empty else np.nan)

        bm = axes3[1].bar(
            x_idx - width / 2,
            mala_best_tau,
            width,
            color="#e74c3c",
            alpha=0.85,
            edgecolor="black",
            hatch="//",
            label="Mejor MALA [--]",
        )
        br = axes3[1].bar(
            x_idx + width / 2,
            rl_best_tau,
            width,
            color="#2ecc71",
            alpha=0.85,
            edgecolor="black",
            label="Mejor RL-MALA [-]",
        )
        axes3[1].set_xticks(x_idx)
        axes3[1].set_xticklabels(sys_names)
        axes3[1].set_yscale("log")
        axes3[1].set_ylabel(r"Mínimo $\tau_{\mathrm{int}}(E)$ alcanzado (pasos, log)")
        axes3[1].set_title(r"Mejor $\tau_{\mathrm{int}}(E)$ por Sistema: MALA vs RL-MALA", weight="bold")
        axes3[1].grid(True, axis="y", which="both", alpha=0.3)
        axes3[1].legend(frameon=True)

        fig3.tight_layout()
        _save_fig_custom(fig3, out_dir, "fig_all_systems_global_overview", args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ploteo de resultados de parameter search")
    parser.add_argument(
        "--csv",
        type=str,
        default="results_param_search/master_param_search_summary.csv",
        help="Ruta al CSV maestro de resumen",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results_param_search/figures",
        help="Directorio de figuras",
    )
    parser.add_argument(
        "--figure_format",
        "--format",
        dest="figure_format",
        type=str,
        default="png",
        choices=["png", "pdf", "both"],
        help="Formato de exportación gráfica ('png' por defecto para ahorrar espacio).",
    )
    parser.add_argument(
        "--dpi",
        "--ppi",
        dest="dpi",
        type=int,
        default=300,
        help="Resolución en píxeles/puntos por pulgada (PPI / DPI).",
    )
    parser.add_argument("--title_prefix", type=str, default="", help="Prefijo de título.")
    parser.add_argument("--custom_suptitle", type=str, default="", help="Suptitle personalizado.")
    parser.add_argument(
        "--no_titles",
        "--hide_titles",
        dest="no_titles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ocultar títulos para LaTeX.",
    )
    parser.add_argument("--font_scale", type=float, default=1.0, help="Escala de fuente.")
    parser.add_argument("--line_width_scale", type=float, default=1.0, help="Escala de grosor de línea.")
    parser.add_argument(
        "--transparent_bg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fondo transparente.",
    )
    args = parser.parse_args()
    generate_param_search_plots(args.csv, args.out_dir, args)
