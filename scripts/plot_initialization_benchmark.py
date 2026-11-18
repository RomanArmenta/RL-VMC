#!/usr/bin/env python3
"""
scripts/plot_initialization_benchmark.py
==============================================================================
Generador de figuras y tablas estadísticas con barras de error para el
Benchmark de Estrategias de Inicialización de Caminantes (MALA vs RL-MALA).

Estrategias de Inicialización analizadas:
  1) alternating_spin   : Un electrón en cada Gaussiana (con jitter cuántico)
  2) symmetric_gaussian : Gaussiana centrada en x=0 con sigma que cubre la envolvente
  3) uniform            : Distribución uniforme en caja [-L, L] que encierra la nube

Estadística:
  - Agrega sobre n_exp réplicas (diferentes seeds) y n_eval cadenas MCMC.
  - Calcula la media (mean) y el error estándar de la media (SEM = std / sqrt(n)).
  - Produce barras de error en todas las métricas de eficiencia, autocorrelación,
    sesgo energético y ergodocidad nodal.
==============================================================================
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Añadir raíz del proyecto al path para importar módulos rl_vmc
ROOT_PROJECT = Path(__file__).resolve().parent.parent
if str(ROOT_PROJECT) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT))

try:
    import h5py
except ImportError:
    h5py = None

try:
    from rl_vmc.postprocess.casscf_baseline import compute_exact_casscf_1body_density
except ImportError:
    compute_exact_casscf_1body_density = None

# Colores y estilo de publicación
COLOR_MALA = "#D95F02"      # Naranja rojizo para MALA clásico
COLOR_RL_MALA = "#1B9E77"   # Verde azulado para RL-MALA
COLOR_CASSCF = "#000000"    # Negro para referencia exacta

INIT_LABELS = {
    "alternating_spin": "1 e⁻ por Gaussiana\n(Alternating Spin)",
    "randomized_wigner": "Wigner Aleatorio\n(Randomized Wigner)",
    "symmetric_gaussian": "Gaussiana x=0\n(Envolvente)",
    "uniform": "Uniforme\n(Caja [-L, L])",
}

INIT_ORDER = ["alternating_spin", "randomized_wigner", "symmetric_gaussian", "uniform"]


def setup_matplotlib_style(font_scale: float = 1.0, line_width_scale: float = 1.0) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11 * font_scale,
        "axes.labelsize": 12 * font_scale,
        "axes.titlesize": 13 * font_scale,
        "xtick.labelsize": 10 * font_scale,
        "ytick.labelsize": 10 * font_scale,
        "legend.fontsize": 10 * font_scale,
        "lines.linewidth": 1.75 * line_width_scale,
        "lines.markersize": 6 * font_scale,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "figure.autolayout": True,
    })


def parse_run_info(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Lee el archivo summary.json o production_metrics.csv de una corrida individual."""
    summary_path = run_dir / "summary.json"
    data = {}
    if summary_path.is_file():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Aviso] No se pudo leer {summary_path}: {e}")

    # Fallback o lectura complementaria desde CSV
    csv_candidates = list(run_dir.glob("production_metrics_*.csv"))
    csv_row = {}
    if csv_candidates:
        try:
            df = pd.read_csv(csv_candidates[0])
            if not df.empty:
                csv_row = df.iloc[0].to_dict()
        except Exception:
            pass

    if not data and not csv_row:
        return None

    # Extraer metadatos
    obs = data.get("observables", {})
    cass = data.get("casscf_comparison", {})
    train_h = data.get("training_hyperparameters", {})
    samp_cfg = train_h.get("sampler", {})
    eval_cfg = data.get("evaluation_config", {})
    prop_cfg = data.get("model_hyperparameters", {}).get("proposal", {})
    sys_cfg = data.get("system", {})

    init_mode = (
        samp_cfg.get("init_mode")
        or csv_row.get("init_mode")
        or eval_cfg.get("eval_init_mode")
        or "unknown"
    )
    if init_mode == "same_as_sampler":
        init_mode = samp_cfg.get("init_mode", "unknown")

    # Si aún es unknown, inferir del nombre de la carpeta
    dirname = run_dir.name
    if "alt_spin" in dirname or "alternating_spin" in dirname:
        init_mode = "alternating_spin"
    elif "rand_wigner" in dirname or "randomized_wigner" in dirname:
        init_mode = "randomized_wigner"
    elif "sym_gauss" in dirname or "symmetric_gaussian" in dirname:
        init_mode = "symmetric_gaussian"
    elif "uniform" in dirname:
        init_mode = "uniform"

    method = (
        csv_row.get("proposal_type")
        or prop_cfg.get("proposal_type")
        or ("TD3" if "RL_MALA" in dirname or "TD3" in dirname else "MALA")
    )
    if method in ("neural_deepset", "TD3", "td3"):
        model_type = "RL_MALA"
    else:
        model_type = "MALA"

    n_elec = int(sys_cfg.get("n_elec") or csv_row.get("N") or (4 if "N4" in dirname else (6 if "N6" in dirname else 8)))
    g_val = float(sys_cfg.get("interaction_strength") or csv_row.get("g") or 1.0)
    omega = float(sys_cfg.get("omega") or csv_row.get("omega") or 0.1)
    seed = int(train_h.get("seed") or csv_row.get("seed") or 0)
    casscf_npz_path = (
        cass.get("casscf_npz_path")
        or data.get("model_hyperparameters", {}).get("ansatz", {}).get("casscf_npz_path")
        or csv_row.get("casscf_npz_path")
        or ""
    )

    e_mean = float(obs.get("E_mean", csv_row.get("E_mean", float("nan"))))
    e_var = float(obs.get("E_var", csv_row.get("E_var", float("nan"))))
    se_e = float(obs.get("SE_E", csv_row.get("SE_E", float("nan"))))
    tau_e = float(obs.get("tau_int_E", csv_row.get("tau_int_E", float("nan"))))
    tau_pos = float(obs.get("tau_int_pos", csv_row.get("tau_int_pos", float("nan"))))
    ess = float(obs.get("ESS", csv_row.get("ESS", float("nan"))))
    ess_s = float(obs.get("ESS_per_sec", csv_row.get("ESS_per_sec", float("nan"))))
    acc = float(obs.get("acceptance", csv_row.get("acceptance", float("nan"))))
    esjd = float(obs.get("ESJD", csv_row.get("ESJD", float("nan"))))
    wall_time = float(obs.get("wall_time_sec", csv_row.get("wall_time_sec", float("nan"))))

    cas_e = float(cass.get("casscf_energy", csv_row.get("casscf_energy", float("nan"))))
    delta_e = float(cass.get("delta_E_vs_CASSCF", csv_row.get("delta_E_vs_CASSCF", float("nan"))))
    kl_tot = float(cass.get("kl_casscf_total_mean", csv_row.get("kl_casscf_total_mean", float("nan"))))
    kl_up = float(cass.get("kl_casscf_up_mean", csv_row.get("kl_casscf_up_mean", float("nan"))))
    kl_dn = float(cass.get("kl_casscf_down_mean", csv_row.get("kl_casscf_down_mean", float("nan"))))
    kl_spin = 0.5 * (kl_up + kl_dn) if (np.isfinite(kl_up) and np.isfinite(kl_dn)) else float("nan")

    return {
        "run_dir": str(run_dir),
        "run_name": dirname,
        "n_elec": n_elec,
        "g": g_val,
        "omega": omega,
        "system_tag": f"N{n_elec}_g{g_val}",
        "init_mode": init_mode,
        "model_type": model_type,
        "seed": seed,
        "casscf_npz_path": str(casscf_npz_path),
        "E_mean": e_mean,
        "SE_E": se_e,
        "E_var": e_var,
        "tau_int_E": tau_e,
        "tau_int_pos": tau_pos,
        "ESS": ess,
        "ESS_per_sec": ess_s,
        "acceptance": acc,
        "ESJD": esjd,
        "wall_time_sec": wall_time,
        "casscf_energy": cas_e,
        "delta_E_vs_CASSCF": delta_e,
        "kl_total": kl_tot,
        "kl_spin": kl_spin,
    }


def _discover_casscf_energies_from_jsons() -> Dict[str, float]:
    """Lee energías exactas CASSCF desde CASSCF_csv/*.json si está disponible."""
    energies: Dict[str, float] = {
        "N4_g1.0": 1.9486219356652614,
        "N6_g1.0": 4.52405213,
        "N8_g1.0": 8.07323817198525,
    }
    cas_dir = ROOT_PROJECT / "CASSCF_csv"
    if cas_dir.is_dir():
        for jp in sorted(cas_dir.glob("*.json")):
            try:
                jd = json.loads(jp.read_text(encoding="utf-8"))
                if not jd.get("converged", False):
                    continue
                jc = jd.get("config", {})
                nel = int(jc.get("n_up", 0)) + int(jc.get("n_down", 0))
                gv = float(jc.get("g", 0.0))
                ev = float(jd.get("selected_energy", jd.get("energy", float("nan"))))
                if nel > 0 and np.isfinite(ev):
                    energies[f"N{nel}_g{gv}"] = ev
            except Exception:
                continue
    return energies


def collect_benchmark_data(root_dir: Path) -> pd.DataFrame:
    """Recolecta métricas de todas las subcarpetas de corridas."""
    records = []
    # Buscar carpetas que contengan summary.json o production_metrics_*.csv
    for sfile in root_dir.glob("**/summary.json"):
        info = parse_run_info(sfile.parent)
        if info is not None:
            records.append(info)

    # Si no se encontraron por summary.json, buscar por csv
    if not records:
        for cfile in root_dir.glob("**/production_metrics_*.csv"):
            info = parse_run_info(cfile.parent)
            if info is not None:
                records.append(info)

    # También verificar si existe un master_csv en root_dir
    master_csvs = list(root_dir.glob("*master*.csv"))
    if not records and master_csvs:
        try:
            mdf = pd.read_csv(master_csvs[0])
            for _, r in mdf.iterrows():
                row_d = r.to_dict()
                dirname = str(row_d.get("run_name", ""))
                init_m = "unknown"
                if "alt_spin" in dirname or "alternating_spin" in dirname:
                    init_m = "alternating_spin"
                elif "rand_wigner" in dirname or "randomized_wigner" in dirname:
                    init_m = "randomized_wigner"
                elif "sym_gauss" in dirname or "symmetric_gaussian" in dirname:
                    init_m = "symmetric_gaussian"
                elif "uniform" in dirname:
                    init_m = "uniform"

                method = str(row_d.get("proposal_type", ""))
                m_type = "RL_MALA" if ("TD3" in dirname or "RL_MALA" in dirname or method == "neural_deepset") else "MALA"
                n_el = int(row_d.get("N", 4 if "N4" in dirname else (6 if "N6" in dirname else 8)))
                g_v = float(row_d.get("g", 1.0))

                records.append({
                    "run_dir": str(Path(str(row_d.get("output_h5", ""))).parent),
                    "run_name": dirname,
                    "n_elec": n_el,
                    "g": g_v,
                    "omega": float(row_d.get("omega", 0.1)),
                    "system_tag": f"N{n_el}_g{g_v}",
                    "init_mode": init_m,
                    "model_type": m_type,
                    "seed": int(row_d.get("seed", 0)),
                    "casscf_npz_path": str(row_d.get("casscf_npz_path", "")),
                    "E_mean": float(row_d.get("E_mean", float("nan"))),
                    "SE_E": float(row_d.get("SE_E", float("nan"))),
                    "E_var": float(row_d.get("E_var", float("nan"))),
                    "tau_int_E": float(row_d.get("tau_int_E", float("nan"))),
                    "tau_int_pos": float(row_d.get("tau_int_pos", float("nan"))),
                    "ESS": float(row_d.get("ESS", float("nan"))),
                    "ESS_per_sec": float(row_d.get("ESS_per_sec", float("nan"))),
                    "acceptance": float(row_d.get("acceptance", float("nan"))),
                    "ESJD": float(row_d.get("ESJD", float("nan"))),
                    "wall_time_sec": float(row_d.get("wall_time_sec", float("nan"))),
                    "casscf_energy": float(row_d.get("casscf_energy", float("nan"))),
                    "delta_E_vs_CASSCF": float(row_d.get("delta_E_vs_CASSCF", float("nan"))),
                    "kl_total": float(row_d.get("kl_casscf_total_mean", float("nan"))),
                    "kl_spin": float(row_d.get("kl_casscf_spin_mean", float("nan"))),
                })
        except Exception as e:
            print(f"[Aviso] No se pudo leer master_csv: {e}")

    # Fallback: Parsear directamente los archivos .log en root_dir / "logs" o root_dir
    if not records:
        import re
        log_files = sorted(list(root_dir.glob("logs/*.log")) + list(root_dir.glob("*.log")))
        cas_energies = _discover_casscf_energies_from_jsons()
        # Conservar compatibilidad si es un log histórico N8 a g=4.0 no convergido
        cas_energies.setdefault("N8_g4.0", 21.723051632802946)
        for lfile in log_files:
            bname = lfile.stem
            if bname in ("master_benchmark", "realtime_global_plots") or bname.startswith("postprocess_"):
                continue
            m = re.match(
                r"(N\d+)_w([\d\.]+)_g([\d\.]+)_([A-Za-z0-9_]+)_(alt_spin|rand_wigner|sym_gauss|uniform)_exp(\d+)",
                bname,
            )
            if not m:
                continue
            n_str, w_str, g_str, model_raw, init_tag, exp_id = m.groups()
            n_elec = int(n_str.replace("N", ""))
            g_val = float(g_str)
            omega = float(w_str)

            try:
                content = lfile.read_text(encoding="utf-8")
            except Exception:
                continue

            m_sys_hdr = re.search(r"Sistema\s*:\s*N=\d+.*?g=([\d\.]+)", content)
            if m_sys_hdr:
                g_val = float(m_sys_hdr.group(1))

            sys_tag = f"N{n_elec}_g{g_val}"
            init_mode = {
                "alt_spin": "alternating_spin",
                "rand_wigner": "randomized_wigner",
                "sym_gauss": "symmetric_gaussian",
                "uniform": "uniform",
            }.get(init_tag, init_tag)
            model_type = "RL_MALA" if ("RL_MALA" in model_raw or "TD3" in model_raw) else "MALA"

            chain_matches = re.findall(
                r"\[Cadena (\d+)\] <E> = ([\-\d\.eE\+]+) \+/- ([\-\d\.eE\+]+) Ha \| Acc = ([\d\.]+)% \| ESJD = ([\d\.]+) \| t = ([\d\.]+)s",
                content,
            )
            if not chain_matches:
                continue

            e_vals = [float(cm[1]) for cm in chain_matches]
            se_vals = [float(cm[2]) for cm in chain_matches]
            acc_vals = [float(cm[3]) / 100.0 for cm in chain_matches]
            esjd_vals = [float(cm[4]) for cm in chain_matches]
            t_vals = [float(cm[5]) for cm in chain_matches]

            cas_e = cas_energies.get(sys_tag, float("nan"))
            e_mean = float(np.mean(e_vals))
            delta_e = e_mean - cas_e if np.isfinite(cas_e) else float("nan")

            records.append({
                "run_dir": str(lfile.parent),
                "run_name": bname,
                "n_elec": n_elec,
                "g": g_val,
                "omega": omega,
                "system_tag": sys_tag,
                "init_mode": init_mode,
                "model_type": model_type,
                "seed": int(exp_id),
                "casscf_npz_path": "",
                "E_mean": e_mean,
                "SE_E": float(np.mean(se_vals)),
                "E_var": float(np.var(e_vals)),
                "tau_int_E": float("nan"),
                "tau_int_pos": float("nan"),
                "ESS": float("nan"),
                "ESS_per_sec": float("nan"),
                "acceptance": float(np.mean(acc_vals)),
                "ESJD": float(np.mean(esjd_vals)),
                "wall_time_sec": float(np.sum(t_vals)),
                "casscf_energy": cas_e,
                "delta_E_vs_CASSCF": delta_e,
                "kl_total": float("nan"),
                "kl_spin": float("nan"),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        # Deduplicar por run_name
        df = df.drop_duplicates(subset=["run_name"]).reset_index(drop=True)
    return df


def aggregate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega métricas sobre seeds/réplicas para cada (system_tag, model_type, init_mode)."""
    metrics = [
        "E_mean", "SE_E", "E_var", "tau_int_E", "tau_int_pos",
        "ESS", "ESS_per_sec", "acceptance", "ESJD", "wall_time_sec",
        "delta_E_vs_CASSCF", "kl_total", "kl_spin"
    ]

    grouped = df.groupby(["system_tag", "n_elec", "model_type", "init_mode"])
    agg_rows = []

    for (sys_tag, n_el, model_t, init_m), grp in grouped:
        n_samples = len(grp)
        row = {
            "system_tag": sys_tag,
            "n_elec": n_el,
            "model_type": model_t,
            "init_mode": init_m,
            "n_exp_runs": n_samples,
        }

        for m in metrics:
            vals = grp[m].dropna().values
            if len(vals) > 0:
                mean_v = float(np.mean(vals))
                std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                sem_v = std_v / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                row[f"{m}_mean"] = mean_v
                row[f"{m}_std"] = std_v
                row[f"{m}_sem"] = sem_v
            else:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_std"] = float("nan")
                row[f"{m}_sem"] = float("nan")

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def plot_system_initialization_comparison(
    agg_df: pd.DataFrame,
    system_tag: str,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera el panel principal de 6 subplots comparando MALA vs RL-MALA en las
    3 inicializaciones con barras de error rigurosas (SEM).
    """
    sub_df = agg_df[agg_df["system_tag"] == system_tag].copy()
    if sub_df.empty:
        print(f"[Aviso] No hay datos agregados para el sistema {system_tag}")
        return

    init_keys = [k for k in INIT_ORDER if k in sub_df["init_mode"].unique()]
    if not init_keys:
        init_keys = sorted(sub_df["init_mode"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    ax_ess = axes[0, 0]
    ax_tau = axes[0, 1]
    ax_de  = axes[0, 2]
    ax_acc = axes[1, 0]
    ax_esjd= axes[1, 1]
    ax_kl  = axes[1, 2]

    x_idx = np.arange(len(init_keys))
    bar_width = 0.35

    models = ["MALA", "RL_MALA"]
    colors = {"MALA": COLOR_MALA, "RL_MALA": COLOR_RL_MALA}
    labels = {"MALA": "MALA Clásico", "RL_MALA": "RL-MALA (TD3)"}

    for m_idx, m_name in enumerate(models):
        offset = (m_idx - 0.5) * bar_width
        m_sub = sub_df[sub_df["model_type"] == m_name].set_index("init_mode")

        def get_series(col_prefix: str) -> Tuple[np.ndarray, np.ndarray]:
            means, errs = [], []
            for k in init_keys:
                if k in m_sub.index:
                    m_val = m_sub.loc[k, f"{col_prefix}_mean"]
                    e_val = m_sub.loc[k, f"{col_prefix}_sem"]
                    means.append(m_val if np.isfinite(m_val) else 0.0)
                    errs.append(e_val if np.isfinite(e_val) else 0.0)
                else:
                    means.append(0.0)
                    errs.append(0.0)
            return np.array(means), np.array(errs)

        # 1. ESS / sec
        ess_m, ess_e = get_series("ESS_per_sec")
        ax_ess.bar(
            x_idx + offset, ess_m, bar_width, yerr=ess_e, capsize=4,
            color=colors[m_name], label=labels[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 2. tau_int(E)
        tau_m, tau_e = get_series("tau_int_E")
        ax_tau.bar(
            x_idx + offset, tau_m, bar_width, yerr=tau_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 3. |Delta E| vs CASSCF (en mHa = 1e-3 Ha)
        de_m, de_e = get_series("delta_E_vs_CASSCF")
        # Mostrar magnitud del sesgo en mHa
        de_m_mha = np.abs(de_m) * 1000.0
        de_e_mha = de_e * 1000.0
        ax_de.bar(
            x_idx + offset, de_m_mha, bar_width, yerr=de_e_mha, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 4. Acceptance (%)
        acc_m, acc_e = get_series("acceptance")
        ax_acc.bar(
            x_idx + offset, acc_m * 100.0, bar_width, yerr=acc_e * 100.0, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 5. ESJD
        esjd_m, esjd_e = get_series("ESJD")
        ax_esjd.bar(
            x_idx + offset, esjd_m, bar_width, yerr=esjd_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

        # 6. Spin KL Divergence (D_KL^spin)
        kl_m, kl_e = get_series("kl_spin")
        # En caso de que kl_spin sea 0 o NaN, intentar con kl_total
        if np.all(kl_m == 0.0):
            kl_m, kl_e = get_series("kl_total")
        ax_kl.bar(
            x_idx + offset, kl_m, bar_width, yerr=kl_e, capsize=4,
            color=colors[m_name], alpha=0.9, edgecolor="black", linewidth=0.8
        )

    # Configuración de ejes y etiquetas
    x_tick_labels = [INIT_LABELS.get(k, k) for k in init_keys]
    tick_fs = 9 if len(init_keys) >= 4 else 10
    for ax in axes[1, :]:
        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_tick_labels, rotation=0, ha="center", fontsize=tick_fs)

    ax_ess.set_title("(a) Eficiencia de Muestreo (ESS/s)")
    ax_ess.set_ylabel("Muestras Efectivas / s (mayor es mejor)")
    ax_ess.legend(loc="upper right", framealpha=0.9)

    ax_tau.set_title("(b) Tiempo Autocorrelación $\\tau_{\\mathrm{int}}(E)$")
    ax_tau.set_ylabel("Pasos MCMC (menor es mejor)")

    ax_de.set_title("(c) Sesgo Energético $|\\Delta E|$ vs CASSCF")
    ax_de.set_ylabel("mHa (menor es mejor)")

    ax_acc.set_title("(d) Tasa de Aceptación M-H")
    ax_acc.set_ylabel("Aceptación (%)")
    ax_acc.set_ylim(0, 105)

    ax_esjd.set_title("(e) Distancia Cuadrática de Salto (ESJD)")
    ax_esjd.set_ylabel("$\\mathrm{Bohr}^2$ (mayor es mejor)")

    ax_kl.set_title("(f) Divergencia $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ (Cruce Nodal)")
    ax_kl.set_ylabel("Divergencia KL (menor es mejor)")
    ax_kl.set_yscale("log")

    plt.suptitle(
        f"Benchmark de Inicialización: MALA vs RL-MALA en {system_tag} (Barras de Error: SEM sobre Réplicas)",
        fontsize=14, fontweight="bold", y=0.99
    )

    out_file = out_dir / f"benchmark_initialization_{system_tag}.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráfica] Guardada figura comparativa en: {out_file}")


def plot_speedup_scaling_summary(
    agg_df: pd.DataFrame,
    out_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera figura comparativa resumen mostrando la ganancia de velocidad (Speedup G_RL = ESS_RL / ESS_MALA)
    y reducción de autocorrelación a través de todos los sistemas e inicializaciones.
    """
    systems = sorted(agg_df["system_tag"].unique())
    if not systems:
        return

    active_inits = [k for k in INIT_ORDER if k in agg_df["init_mode"].unique()]
    if not active_inits:
        active_inits = sorted(agg_df["init_mode"].unique())

    fig_w = 14.5 if len(active_inits) >= 4 else 13.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 5.2))
    bar_width = min(0.32, 0.72 / max(1, len(systems)))
    x_base = np.arange(len(active_inits))

    for s_idx, sys_tag in enumerate(systems):
        s_df = agg_df[agg_df["system_tag"] == sys_tag]
        offset = (s_idx - (len(systems) - 1) / 2.0) * bar_width

        speedups = []
        speedup_errs = []
        tau_reductions = []

        for k in active_inits:
            m_sub = s_df[(s_df["model_type"] == "MALA") & (s_df["init_mode"] == k)]
            r_sub = s_df[(s_df["model_type"] == "RL_MALA") & (s_df["init_mode"] == k)]

            if not m_sub.empty and not r_sub.empty:
                ess_m = float(m_sub["ESS_per_sec_mean"].iloc[0])
                ess_r = float(r_sub["ESS_per_sec_mean"].iloc[0])
                sem_m = float(m_sub["ESS_per_sec_sem"].iloc[0])
                sem_r = float(r_sub["ESS_per_sec_sem"].iloc[0])

                if ess_m > 0 and np.isfinite(ess_r):
                    sp = ess_r / ess_m
                    # Propagación de error para razón sp = r / m
                    sp_err = sp * np.sqrt((sem_r / max(ess_r, 1e-9))**2 + (sem_m / max(ess_m, 1e-9))**2)
                    speedups.append(sp)
                    speedup_errs.append(sp_err)
                else:
                    speedups.append(0.0)
                    speedup_errs.append(0.0)

                tau_m = float(m_sub["tau_int_E_mean"].iloc[0])
                tau_r = float(r_sub["tau_int_E_mean"].iloc[0])
                if tau_r > 0 and np.isfinite(tau_m):
                    tau_reductions.append(tau_m / tau_r)
                else:
                    tau_reductions.append(0.0)
            else:
                speedups.append(0.0)
                speedup_errs.append(0.0)
                tau_reductions.append(0.0)

        # Plot en Panel 1: Factor de Aceleración G_RL
        ax1.bar(
            x_base + offset, speedups, bar_width, yerr=speedup_errs, capsize=4,
            label=f"Sistema {sys_tag}", alpha=0.85, edgecolor="black", linewidth=0.8
        )

        # Plot en Panel 2: Reducción de Autocorrelación
        ax2.bar(
            x_base + offset, tau_reductions, bar_width, capsize=4,
            label=f"Sistema {sys_tag}", alpha=0.85, edgecolor="black", linewidth=0.8
        )

    init_labels = [INIT_LABELS.get(k, k) for k in active_inits]
    tick_fs = 9 if len(active_inits) >= 4 else 10
    for ax in (ax1, ax2):
        ax.set_xticks(x_base)
        ax.set_xticklabels(init_labels, fontsize=tick_fs)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2, label="Paridad MALA (1.0×)")

    ax1.set_title("Factor de Aceleración Neta $G_{\\mathrm{RL}} = \\mathrm{ESS/s}_{\\mathrm{RL}} / \\mathrm{ESS/s}_{\\mathrm{MALA}}$")
    ax1.set_ylabel("Aceleración Relativa (veces mayor)")
    ax1.legend(loc="upper left")

    ax2.set_title("Factor de Descorrelación $\\tau_{\\mathrm{int}}(E)_{\\mathrm{MALA}} / \\tau_{\\mathrm{int}}(E)_{\\mathrm{RL}}$")
    ax2.set_ylabel("Reducción de Autocorrelación (veces menor)")
    ax2.legend(loc="upper left")

    plt.suptitle("Resumen Global del Impacto de la Inicialización: Resiliencia y Aceleración de RL-MALA", fontsize=13, fontweight="bold")
    out_file = out_dir / f"benchmark_initialization_summary_speedups.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráfica] Guardada figura resumen en: {out_file}")


def _read_npz_meta(npz_path: Path) -> Optional[Dict[str, Any]]:
    """Lee metadatos JSON embebidos en un .npz sin depender de numpy.load con pickle."""
    import zipfile
    try:
        with zipfile.ZipFile(npz_path, "r") as z:
            raw = z.read("metadata_json.npy")
            hlen = int.from_bytes(raw[8:10], "little")
            return json.loads(raw[10 + hlen:].decode("utf-32-le").rstrip("\x00"))
    except Exception:
        return None


def resolve_casscf_reference_state(
    system_tag: str,
    root_dir: Path,
    raw_df: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """Busca y calcula la densidad analítica CASSCF exacta para el sistema."""
    import re
    m_sys = re.match(r"N(\d+)_g([\d\.]+)", system_tag)
    n_target = int(m_sys.group(1)) if m_sys else None
    g_target = float(m_sys.group(2)) if m_sys else None

    npz_path: Optional[Path] = None

    # 1. Priorizar exactamente el .npz registrado en los resumenes del sistema
    if raw_df is not None and not raw_df.empty and "casscf_npz_path" in raw_df.columns:
        sub = raw_df[raw_df["system_tag"] == system_tag]
        for p_str in sub["casscf_npz_path"].dropna().tolist():
            if p_str and Path(p_str).is_file():
                npz_path = Path(p_str)
                break
            if p_str and (ROOT_PROJECT / p_str).is_file():
                npz_path = ROOT_PROJECT / p_str
                break

    # 2. Buscar en CASSCF_csv (estados convergidos c0-aN b=256) validando (N, g) en metadata
    if npz_path is None and n_target is not None and g_target is not None:
        search_cas_dirs = [
            ROOT_PROJECT / "CASSCF_csv",
            root_dir.parent / "CASSCF_csv",
            Path("CASSCF_csv"),
        ]
        for cdir in search_cas_dirs:
            if not cdir.is_dir():
                continue
            for cand in sorted(cdir.glob("*.npz")):
                meta = _read_npz_meta(cand)
                if not meta:
                    continue
                mcfg = meta.get("config", {})
                nel = int(mcfg.get("n_up", 0)) + int(mcfg.get("n_down", 0))
                gv = float(mcfg.get("g", -1.0))
                if nel == n_target and abs(gv - g_target) < 1e-4 and bool(meta.get("converged", False)):
                    npz_path = cand
                    break
            if npz_path is not None:
                break

    # 3. Respaldo en figures/ si no hubo match en CASSCF_csv
    if npz_path is None:
        search_dirs = [
            root_dir / "figures",
            root_dir.parent / "figures",
            Path("figures"),
            ROOT_PROJECT / "figures",
        ]
        if n_target == 4:
            target_patterns = ["*n4*g1*det6*.npz", "*n4*det6*.npz", "*n4*.npz"]
        elif n_target == 6:
            target_patterns = ["*n6*g1*det*.npz", "*n6*.npz"]
        elif n_target == 8:
            target_patterns = ["*n8*g1*det70*.npz", "*n8*det70*.npz", "*n8*.npz"]
        else:
            target_patterns = ["*.npz"]

        for sdir in search_dirs:
            if sdir.is_dir():
                for pat in target_patterns:
                    matches = list(sdir.glob(pat))
                    if matches:
                        npz_path = matches[0]
                        break
            if npz_path is not None:
                break

    if npz_path is None:
        return None

    x_max_default = {4: 14.0, 6: 18.0, 8: 22.0, 10: 26.0}.get(n_target or 8, 22.0)
    if compute_exact_casscf_1body_density is not None:
        try:
            res = compute_exact_casscf_1body_density(npz_path, x_min=-x_max_default, x_max=x_max_default, n_grid=800)
            res["npz_path"] = str(npz_path)
            res["x_max"] = x_max_default
            return res
        except Exception as e:
            print(f"[Aviso] No se pudo calcular densidad CASSCF exacta desde {npz_path}: {e}")
    return None


def find_best_models_per_initialization(
    raw_df: pd.DataFrame,
    system_tag: str,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Identifica el Mejor MALA y el Mejor RL-MALA para cada una de las 3 inicializaciones
    según el criterio de mínimo sesgo absoluto |Delta E| vs CASSCF (o mínima energía).
    """
    s_df = raw_df[raw_df["system_tag"] == system_tag].copy()
    if s_df.empty:
        return {}

    best_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for init_mode in INIT_ORDER:
        sub_init = s_df[s_df["init_mode"] == init_mode]
        if sub_init.empty:
            continue

        best_dict[init_mode] = {}
        for m_type in ("MALA", "RL_MALA"):
            m_sub = sub_init[sub_init["model_type"] == m_type].copy()
            if m_sub.empty:
                continue

            # Criterio: mínimo error absoluto |delta_E_vs_CASSCF|
            if "delta_E_vs_CASSCF" in m_sub.columns and m_sub["delta_E_vs_CASSCF"].notna().any():
                m_sub["abs_de"] = m_sub["delta_E_vs_CASSCF"].abs()
                best_row = m_sub.sort_values(by="abs_de", ascending=True).iloc[0]
            else:
                best_row = m_sub.sort_values(by="E_mean", ascending=True).iloc[0]

            best_dict[init_mode][m_type] = best_row.to_dict()

    return best_dict


def find_and_load_bundle_walkers(run_info: Dict[str, Any], root_dir: Path) -> Optional[Dict[str, Any]]:
    """Localiza y carga el archivo run_bundle.h5 para extraer las coordenadas de caminantes."""
    if h5py is None:
        return None

    r_dir = Path(run_info.get("run_dir", ""))
    r_name = str(run_info.get("run_name", ""))
    sys_tag = str(run_info.get("system_tag", ""))

    candidate_paths = [
        r_dir / "run_bundle.h5",
        r_dir if str(r_dir).endswith(".h5") else Path("non_existent"),
        root_dir / sys_tag / r_name / "run_bundle.h5",
        root_dir / r_name / "run_bundle.h5",
    ]
    # Búsqueda por glob recursivo si los anteriores fallan
    if not any(p.is_file() for p in candidate_paths):
        glob_matches = list(root_dir.glob(f"**/{r_name}/run_bundle.h5"))
        if glob_matches:
            candidate_paths.append(glob_matches[0])

    h5_path = None
    for p in candidate_paths:
        if p.is_file() and p.stat().st_size > 10000:
            h5_path = p
            break

    if h5_path is None:
        return None

    try:
        with h5py.File(h5_path, "r") as f:
            if "production" not in f or "walkers" not in f["production"]:
                return None
            w_all = f["production"]["walkers"][:]
            
        w_arr = np.asarray(w_all)
        if w_arr.ndim == 5:
            # (n_chains, n_steps, batch_size, n_elec, 1)
            w_single = w_arr[0, :, 0, :, 0]
            w_flat = w_arr.squeeze(-1).reshape(-1, w_arr.shape[3])
        elif w_arr.ndim == 4:
            # (n_chains, n_steps, batch_size, n_elec)
            w_single = w_arr[0, :, 0, :]
            w_flat = w_arr.reshape(-1, w_arr.shape[3])
        else:
            w_single = w_arr[0, :, 0]
            w_flat = w_arr.reshape(-1, w_arr.shape[-1])

        return {
            "h5_path": str(h5_path),
            "w_single": w_single,
            "w_flat": w_flat,
            "n_steps": int(w_single.shape[0]),
            "n_elec": int(w_single.shape[1]),
        }
    except Exception as exc:
        print(f"[Aviso] No se pudieron cargar caminantes desde {h5_path}: {exc}")
        return None


def plot_best_models_walker_traces(
    best_models: Dict[str, Dict[str, Dict[str, Any]]],
    system_tag: str,
    casscf_ref: Optional[Dict[str, Any]],
    out_dir: Path,
    root_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera figura de trazas temporales de 1 solo caminante junto con el histograma marginal
    y la superposición exacta CASSCF para el Mejor MALA vs Mejor RL-MALA en las 3 inicializaciones.
    """
    init_modes = [k for k in INIT_ORDER if k in best_models and len(best_models[k]) > 0]
    if not init_modes:
        return

    # Verificar si al menos un modelo tiene archivo .h5 disponible
    loaded_data: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for im in init_modes:
        for mt in ("MALA", "RL_MALA"):
            if mt in best_models[im]:
                w_dict = find_and_load_bundle_walkers(best_models[im][mt], root_dir)
                if w_dict is not None:
                    loaded_data[(im, mt)] = w_dict

    if not loaded_data:
        print(f"[Aviso Trazas] No se encontraron archivos run_bundle.h5 para {system_tag}. Omitiendo gráfica de trazas físicas.")
        return

    n_rows = len(init_modes)
    with plt.rc_context({"figure.autolayout": False}):
        fig = plt.figure(figsize=(16.0, 4.4 * n_rows))
        outer_gs = gridspec.GridSpec(n_rows, 2, hspace=0.38, wspace=0.18)

        x_max = 14.0 if "N4" in system_tag else (18.0 if "N6" in system_tag else 22.0)
        if casscf_ref is not None:
            x_max = float(casscf_ref.get("x_max", x_max))

        for r_idx, im in enumerate(init_modes):
            for c_idx, mt in enumerate(("MALA", "RL_MALA")):
                inner_gs = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=outer_gs[r_idx, c_idx], width_ratios=[3.8, 1.1], wspace=0.06
                )
                ax_main = fig.add_subplot(inner_gs[0])
                ax_hist = fig.add_subplot(inner_gs[1], sharey=ax_main)
                plt.setp(ax_hist.get_yticklabels(), visible=False)

                info = best_models[im].get(mt)
                w_res = loaded_data.get((im, mt))

                col_label = "Mejor MALA Clásico" if mt == "MALA" else "Mejor RL-MALA (TD3)"
                init_title = INIT_LABELS.get(im, im).replace("\n", " ")

                if info is None or w_res is None:
                    ax_main.text(0.5, 0.5, "Datos de caminantes no disponibles\n(run_bundle.h5 ausente)",
                                 ha="center", va="center", transform=ax_main.transAxes, color="gray")
                    ax_main.set_title(f"{col_label} | {init_title}", fontsize=10)
                    continue

                w_single = w_res["w_single"]
                n_steps, n_elec = w_single.shape
                n_up = int(info.get("n_elec", n_elec) // 2)

                # Submuestreo uniforme para renderizado nítido
                stride = max(1, n_steps // 2500)
                steps_sub = np.arange(0, n_steps, stride)
                w_sub = w_single[::stride, :]

                # Trazas de partículas coloreadas por espín
                for e_i in range(n_elec):
                    c_spin = "#2980b9" if e_i < n_up else "#c0392b"
                    lbl = r"Spin $\uparrow$" if e_i == 0 else (r"Spin $\downarrow$" if e_i == n_up else None)
                    ls = "--" if mt == "MALA" else "-"
                    ax_main.plot(steps_sub, w_sub[:, e_i], color=c_spin, linestyle=ls, lw=1.2, alpha=0.85, label=lbl)

                ax_main.set_ylim(-x_max, x_max)
                ax_main.set_ylabel(f"{init_title}\nCoordenada $x$ [Bohr]", fontsize=10)
                if r_idx == n_rows - 1:
                    ax_main.set_xlabel("Paso MCMC (Cadena 0, Caminante 0)", fontsize=10)

                # Métricas en el título
                de_val = float(info.get("delta_E_vs_CASSCF", float("nan"))) * 1000.0
                acc_val = float(info.get("acceptance", float("nan"))) * 100.0
                esjd_val = float(info.get("ESJD", float("nan")))
                title_txt = f"{col_label} ({info.get('run_name', '')})\n$\\Delta E = {de_val:+.2f}$ mHa | Acc = {acc_val:.1f}% | ESJD = {esjd_val:.3f}"
                ax_main.set_title(title_txt, fontsize=10, fontweight="bold")
                if r_idx == 0 and c_idx == 0:
                    ax_main.legend(loc="upper left", fontsize=8, framealpha=0.85)

                # Histograma marginal normalizado respecto al total real de muestras
                bins = np.linspace(-x_max, x_max, 110)
                bin_w = float(bins[1] - bins[0])
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                w_flat = w_res["w_flat"]
                up_vals = w_flat[:, :n_up].ravel()
                dn_vals = w_flat[:, n_up:].ravel()

                cnt_up, _ = np.histogram(up_vals, bins=bins)
                cnt_dn, _ = np.histogram(dn_vals, bins=bins)
                hist_up = cnt_up / max(1.0, float(up_vals.size) * bin_w)
                hist_dn = cnt_dn / max(1.0, float(dn_vals.size) * bin_w)

                ax_hist.fill_betweenx(bin_centers, 0, hist_up, color="#2980b9", alpha=0.40)
                ax_hist.plot(hist_up, bin_centers, color="#2980b9", lw=1.2)
                ax_hist.fill_betweenx(bin_centers, 0, hist_dn, color="#c0392b", alpha=0.40)
                ax_hist.plot(hist_dn, bin_centers, color="#c0392b", lw=1.2)

                # Superponer referencia CASSCF exacta
                if casscf_ref is not None:
                    # La densidad CASSCF por espín = p_total / 2 para espín cerrado
                    p_spin_cas = np.asarray(casscf_ref["p_total"], dtype=np.float64) / 2.0
                    ax_hist.plot(p_spin_cas, casscf_ref["x"], color="black", linestyle="--", lw=1.6, label="CASSCF Exacto")
                    if r_idx == 0 and c_idx == 1:
                        ax_hist.legend(loc="upper right", fontsize=7, framealpha=0.85)

                ax_hist.set_xlabel("$p_{\\sigma}(x)$", fontsize=9)
                ax_hist.set_xlim(0, max(float(np.max(hist_up)), float(np.max(hist_dn)), 0.08) * 1.25)

        fig.suptitle(
            f"Trazas Temporales de 1 Caminante y Distribuciones Marginales vs Benchmark CASSCF Exacto ({system_tag})",
            fontsize=13, fontweight="bold", y=0.995
        )

        out_file = out_dir / f"benchmark_walker_traces_{system_tag}.{fmt}"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Gráfica] Guardada figura de trazas individuales en: {out_file}")


def plot_best_models_density_distributions(
    best_models: Dict[str, Dict[str, Dict[str, Any]]],
    system_tag: str,
    casscf_ref: Optional[Dict[str, Any]],
    out_dir: Path,
    root_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> None:
    """
    Genera figura de subplots comparando la densidad de 1 cuerpo muestreada
    por el Mejor MALA y el Mejor RL-MALA contra el benchmark CASSCF exacto para cada inicialización.
    """
    init_modes = [k for k in INIT_ORDER if k in best_models and len(best_models[k]) > 0]
    if not init_modes:
        return

    n_rows = len(init_modes)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15.0, 4.0 * n_rows), sharex=True, sharey=True, squeeze=False)

    x_max = 14.0 if "N4" in system_tag else (18.0 if "N6" in system_tag else 22.0)
    if casscf_ref is not None:
        x_max = float(casscf_ref.get("x_max", x_max))

    for r_idx, im in enumerate(init_modes):
        init_title = INIT_LABELS.get(im, im).replace("\n", " ")
        for c_idx, mt in enumerate(("MALA", "RL_MALA")):
            ax = axes[r_idx, c_idx]
            info = best_models[im].get(mt)
            col_label = "Mejor MALA Clásico" if mt == "MALA" else "Mejor RL-MALA (TD3)"

            if info is None:
                ax.text(0.5, 0.5, "Modelo no evaluado", ha="center", va="center", transform=ax.transAxes, color="gray")
                continue

            w_res = find_and_load_bundle_walkers(info, root_dir)
            de_val = float(info.get("delta_E_vs_CASSCF", float("nan"))) * 1000.0
            acc_val = float(info.get("acceptance", float("nan"))) * 100.0
            esjd_val = float(info.get("ESJD", float("nan")))

            # 1. Superponer Benchmark CASSCF Exacto
            if casscf_ref is not None:
                x_cas = casscf_ref["x"]
                rho_tot_cas = casscf_ref["rho_total"]
                ax.plot(x_cas, rho_tot_cas, color="#008000", lw=2.2, linestyle="-", label="CASSCF Exacto (Total)")
                if "rho_up" in casscf_ref and "rho_down" in casscf_ref:
                    ax.plot(x_cas, casscf_ref["rho_up"], color="#2980b9", lw=1.1, linestyle=":", label=r"CASSCF $\uparrow$")
                    ax.plot(x_cas, casscf_ref["rho_down"], color="#c0392b", lw=1.1, linestyle=":", label=r"CASSCF $\downarrow$")

            # 2. Graficar densidades muestreadas si los caminantes están disponibles
            if w_res is not None:
                w_flat = w_res["w_flat"]
                n_elec = w_res["n_elec"]
                n_up = n_elec // 2
                n_dn = n_elec - n_up
                bins = np.linspace(-x_max, x_max, 140)
                bin_w = float(bins[1] - bins[0])
                bin_centers = 0.5 * (bins[:-1] + bins[1:])

                up_vals = w_flat[:, :n_up].ravel()
                dn_vals = w_flat[:, n_up:].ravel()

                # Normalización respecto al tamaño total real de muestras (preserva integral sin sesgar el borde)
                cnt_up, _ = np.histogram(up_vals, bins=bins)
                cnt_dn, _ = np.histogram(dn_vals, bins=bins)
                rho_up = (cnt_up / max(1.0, float(up_vals.size) * bin_w)) * n_up
                rho_dn = (cnt_dn / max(1.0, float(dn_vals.size) * bin_w)) * n_dn
                rho_tot = rho_up + rho_dn

                ls = "--" if mt == "MALA" else "-"
                ax.fill_between(bin_centers, rho_up, color="#2980b9", alpha=0.35, label=r"Muestreado $\rho_\uparrow(x)$")
                ax.fill_between(bin_centers, rho_dn, color="#c0392b", alpha=0.35, label=r"Muestreado $\rho_\downarrow(x)$")
                ax.plot(bin_centers, rho_tot, color="black", linestyle=ls, lw=2.0, label=f"Muestreado Total ({mt})")
            else:
                ax.text(0.5, 0.4, "Densidad muestreada requiere run_bundle.h5\n(mostrando benchmark CASSCF)",
                        ha="center", va="center", transform=ax.transAxes, color="#555555", fontsize=9)

            ax.set_xlim(-x_max, x_max)
            if c_idx == 0:
                ax.set_ylabel(f"{init_title}\nDensidad $\\rho(x)$", fontsize=10)
            if r_idx == n_rows - 1:
                ax.set_xlabel("Coordenada $x$ [Bohr]", fontsize=10)

            # Anotación cuantitativa en esquina superior
            annot_txt = f"{col_label} ({info.get('run_name', '')})\n$\\Delta E = {de_val:+.2f}$ mHa | Acc = {acc_val:.1f}% | ESJD = {esjd_val:.3f}"
            ax.set_title(annot_txt, fontsize=10, fontweight="bold")
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.suptitle(
        f"Distribuciones de Densidad Muestreadas vs Benchmark CASSCF Exacto ({system_tag})",
        fontsize=13, fontweight="bold", y=0.99
    )

    out_file = out_dir / f"benchmark_densities_vs_casscf_{system_tag}.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Gráfica] Guardada figura de densidades vs CASSCF en: {out_file}")


def export_markdown_and_csv_reports(agg_df: pd.DataFrame, out_dir: Path, root_dir: Optional[Path] = None) -> None:
    """Genera tabla ejecutiva en CSV y en Markdown."""
    csv_file = out_dir / "initialization_benchmark_summary_table.csv"
    agg_df.to_csv(csv_file, index=False)
    print(f"[Tabla] Tabla CSV agregada exportada en: {csv_file}")
    if root_dir is not None and (root_dir / "results_csv").is_dir():
        import shutil
        shutil.copy2(csv_file, root_dir / "results_csv" / csv_file.name)

    md_file = out_dir / "INITIALIZATION_BENCHMARK_REPORT.md"
    lines = [
        "# Reporte Estadístico: Benchmark de Estrategias de Inicialización (MALA vs RL-MALA)",
        "",
        "Este reporte sintetiza los resultados experimentales evaluando la sensibilidad, estabilidad y aceleración",
        "de **MALA clásico** frente a **RL-MALA (TD3)** bajo las estrategias de inicialización evaluadas:",
        "",
        "1. **`alternating_spin`**: Un electrón en cada Gaussiana (red cristalina ordenada por espín con jitter térmico).",
        "2. **`randomized_wigner`**: Red de Wigner centrada con permutación aleatoria de espines por caminante (estándar moderno).",
        "3. **`symmetric_gaussian`**: Distribución normal isotrópica $\\mathcal{N}(0, \\sigma^2)$ cubriendo la envolvente completa.",
        "4. **`uniform`**: Distribución uniforme $\\mathcal{U}(-L, L)$ sobre la caja del sistema confinante.",
        "",
        "---",
        "",
        "## Tabla Ejecutiva de Resultados (Media ± Error Estándar de la Media)",
        "",
        "| Sistema | Modelo | Modo Inicialización | Réplicas | $\\Delta E$ vs CASSCF (mHa) | $\\tau_{\\mathrm{int}}(E)$ | Aceptación (%) | $\\mathrm{ESS/s}$ | $D_{\\mathrm{KL}}^{\\mathrm{spin}}$ |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in agg_df.iterrows():
        sys_t = row["system_tag"]
        mod_t = row["model_type"]
        init_m = row["init_mode"]
        n_exp = int(row["n_exp_runs"])

        de_val = row.get("delta_E_vs_CASSCF_mean", float("nan")) * 1000.0
        de_sem = row.get("delta_E_vs_CASSCF_sem", 0.0) * 1000.0

        tau_val = row.get("tau_int_E_mean", float("nan"))
        tau_sem = row.get("tau_int_E_sem", 0.0)

        acc_val = row.get("acceptance_mean", float("nan")) * 100.0
        acc_sem = row.get("acceptance_sem", 0.0) * 100.0

        ess_val = row.get("ESS_per_sec_mean", float("nan"))
        ess_sem = row.get("ESS_per_sec_sem", 0.0)

        kl_val = row.get("kl_spin_mean", row.get("kl_total_mean", float("nan")))
        kl_sem = row.get("kl_spin_sem", row.get("kl_total_sem", 0.0))

        lines.append(
            f"| `{sys_t}` | **{mod_t}** | `{init_m}` | {n_exp} | "
            f"${de_val:+.2f} \\pm {de_sem:.2f}$ | "
            f"${tau_val:.2f} \\pm {tau_sem:.2f}$ | "
            f"${acc_val:.1f}\\% \\pm {acc_sem:.1f}\\%$ | "
            f"${ess_val:.1f} \\pm {ess_sem:.1f}$ | "
            f"${kl_val:.4f} \\pm {kl_sem:.4f}$ |"
        )

    lines.extend([
        "",
        "---",
        "Reporte generado automáticamente por `scripts/plot_initialization_benchmark.py`."
    ])

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Reporte] Reporte Markdown guardado en: {md_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graficador y Generador de Tablas del Benchmark de Inicializaciones.")
    parser.add_argument("results_dir", type=str, help="Carpeta raíz con los resultados del benchmark.")
    parser.add_argument("--out_dir", type=str, default=None, help="Carpeta de salida para figuras y reportes (por defecto: results_dir/figures).")
    parser.add_argument("--system", "-s", type=str, default="all", help="Filtrar por sistema(s) (ej. N4, N6, N8, 'N4 N8', all).")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "pdf", "both"], help="Formato de exportación gráfica.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolución en puntos por pulgada.")
    parser.add_argument("--font_scale", type=float, default=1.0, help="Escala de fuentes tipográficas.")
    parser.add_argument("--line_width_scale", type=float, default=1.0, help="Escala de grosor de línea.")
    args = parser.parse_args()

    root_dir = Path(args.results_dir)
    if not root_dir.exists():
        print(f"Error: La carpeta de resultados '{root_dir}' no existe.")
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else root_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib_style(args.font_scale, args.line_width_scale)

    print(f"[Benchmark] Recolectando corridas de inicialización desde: {root_dir}...")
    raw_df = collect_benchmark_data(root_dir)
    if raw_df.empty:
        print(f"[Aviso] No se encontraron corridas en {root_dir}. Verifica la ruta.")
        sys.exit(0)

    if args.system.lower() not in ("all", "all3", "*"):
        import re
        nums = re.findall(r"\d+", args.system)
        if nums:
            n_set = {int(x) for x in nums}
            raw_df = raw_df[raw_df["n_elec"].isin(n_set)].copy()
        else:
            raw_df = raw_df[raw_df["system_tag"].str.contains(args.system, case=False)].copy()
        if raw_df.empty:
            print(f"[Aviso] No se encontraron corridas para el sistema {args.system}.")
            sys.exit(0)

    print(f"[Benchmark] Encontradas {len(raw_df)} corridas individuales.")
    agg_df = aggregate_statistics(raw_df)

    export_markdown_and_csv_reports(agg_df, out_dir, root_dir=root_dir)

    fmts = ["png", "pdf"] if args.format == "both" else [args.format]
    for fmt in fmts:
        for sys_tag in agg_df["system_tag"].unique():
            casscf_ref = resolve_casscf_reference_state(sys_tag, root_dir, raw_df=raw_df)
            best_models = find_best_models_per_initialization(raw_df, sys_tag)

            # 1. Panel de barras comparativas agregadas
            plot_system_initialization_comparison(agg_df, sys_tag, out_dir, fmt=fmt, dpi=args.dpi)

            # 2. Trazas individuales de 1 caminante (Mejor MALA vs Mejor RL-MALA)
            plot_best_models_walker_traces(best_models, sys_tag, casscf_ref, out_dir, root_dir, fmt=fmt, dpi=args.dpi)

            # 3. Distribuciones de densidad muestreadas vs Benchmark CASSCF Exacto
            plot_best_models_density_distributions(best_models, sys_tag, casscf_ref, out_dir, root_dir, fmt=fmt, dpi=args.dpi)

        # 4. Resumen global de factores de aceleración y descorrelación
        plot_speedup_scaling_summary(agg_df, out_dir, fmt=fmt, dpi=args.dpi)

    print("[Benchmark] Post-procesamiento y graficación completados con éxito.")


if __name__ == "__main__":
    main()
