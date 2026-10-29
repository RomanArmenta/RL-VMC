# RL-VMC: Variational Quantum Monte Carlo & Reinforcement Learning MCMC

**RL-VMC** es una paquetería modular en **JAX + Flax + Optax** para simulaciones de **Monte Carlo Cuántico Variacional (VMC)** acopladas con **muestreadores de Cadenas de Markov entrenados mediante Aprendizaje por Refuerzo (RL-MCMC / RL-MALA)**.

> **Documentación de Arquitectura y Teoría Completa:** Consulta [`docs/ARCHITECTURE_1D.md`](docs/ARCHITECTURE_1D.md) para la formulación teórica detallada (Hamiltoniano 1D, Ansätze Slater-Jastrow y CASSCF, propuestas clásicas y neuronales equivariantes DeepSets, TD3/REINFORCE con recompensas CDLB/ESJD/Barker/Kappa, estimadores de Sokal y esquema HDF5), el diagrama de flujo de la arquitectura y tutoriales paso a paso.

---

## Tabla de Contenidos

1. [Estructura del Repositorio](#1-estructura-del-repositorio)
2. [Requisitos y Entorno](#2-requisitos-y-entorno)
3. [Características Principales de `rl_vmc`](#3-características-principales-de-rl_vmc)
4. [Guía de Uso por Línea de Comandos (`cli/`)](#4-guía-de-uso-por-línea-de-comandos-cli)
   - [4.1 Inspección de Banderas (`--help`)](#41-inspección-de-banderas---help)
   - [4.2 Entrenamiento y Evaluación (`cli/run_1d.py`)](#42-entrenamiento-y-evaluación-clirun_1dpy)
   - [4.3 Control Granular de Guardado (`--save-*` / `--no-save-*`)](#43-control-granular-de-guardado---save----no-save-)
   - [4.4 Post-Procesamiento Modular (`cli/postprocess.py`)](#44-post-procesamiento-modular-clipostprocesspy)
   - [4.5 Conversión Bidireccional Legacy $\leftrightarrow$ HDF5 (`cli/convert_legacy.py`)](#45-conversión-bidireccional-legacy-leftrightarrow-hdf5-cliconvert_legacypy)
   - [4.6 Auditoría Exhaustiva de Carpetas `results*` (`cli/audit_results.py`)](#46-auditoría-exhaustiva-de-carpetas-results-cliaudit_resultspy)
5. [Notebook Interactivo (`notebooks/tutorial_pipeline_1d.ipynb`)](#5-notebook-interactivo-notebookstutorial_pipeline_1dipynb)
6. [Uso Programático desde Python](#6-uso-programático-desde-python)
7. [Convivencia y Separación con la Implementación Original (`src/` y `scripts/`)](#7-convivencia-y-separación-con-la-implementación-original-src-y-scripts)

---

## 1. Estructura del Repositorio

```text
RL-VMC-main/
├── rl_vmc/                          # Paquetería modular principal (1D unificado)
│   ├── __init__.py
│   ├── config.py                    # Dataclasses jerárquicas + constructores argparse
│   ├── core/
│   │   ├── types.py                 # Estados inmutables (SamplerState, StepStats, RLTransitionStep)
│   │   └── io_bundle.py             # Esquema HDF5 unificado (run_bundle.h5) y conversores Legacy <-> HDF5
│   ├── physics/
│   │   └── harmonic_1d.py           # Hamiltoniano 1D (trampa armónica + Coulomb suavizado)
│   ├── ansatze/
│   │   └── ansatz_1d.py             # SlaterJastrow1DAnsatz y CASSCF1DAnsatz
│   ├── proposals/
│   │   └── proposals_1d.py          # Propuestas clásicas (Metropolis, MALA, HMC) y neuronales (DeepSet, FlatMLP)
│   ├── samplers/
│   │   └── mh_sampler_1d.py         # Motor MCMC Metropolis-Hastings (all_electrons, single_electron, hmc)
│   ├── rl/
│   │   ├── rewards.py               # Recompensas MCMC (cdlb, log_esjd, barker_jump, kappa_jump)
│   │   └── learners_1d.py           # Algoritmos RL: TD3Learner1D y REINFORCELearner1D
│   ├── trainers/
│   │   └── runner_1d.py             # Orquestador de los 4 regímenes (vmc_only, rl_only, coupled, eval_only)
│   └── postprocess/
│       ├── diagnostics.py           # ACF por FFT, IAT de Sokal, ESS, ESS/s, G_RL y Time-to-Solution
│       ├── observables.py           # Densidades 1D por espín, g(r_12), picos de Wigner y divergencia KL 2D
│       └── plotter.py               # Pipeline de post-procesamiento y generación de las 6 figuras PDF/PNG
├── cli/                             # Interfaces de línea de comandos (argparse)
│   ├── run_1d.py                    # Ejecución de experimentos 1D (entrenamiento y evaluación)
│   ├── postprocess.py               # Post-procesamiento independiente de uno o varios run_bundle.h5
│   └── convert_legacy.py            # Conversión bidireccional entre carpetas Legacy y run_bundle.h5
├── docs/
│   └── ARCHITECTURE_1D.md           # Documentación detallada de teoría, arquitectura y tutoriales
├── notebooks/
│   └── tutorial_pipeline_1d.ipynb   # Notebook ejecutable que prueba todas las funciones de la pipeline
├── src/                             # Módulos originales del proyecto (compatibles con convert_legacy.py)
└── scripts/                         # Scripts históricos de experimentación (1D, 2D y 3D)
```

---

## 2. Requisitos y Entorno

El paquete utiliza Python $\ge 3.10$ con precisión de doble punto flotante (`float64`) habilitada en JAX (`JAX_ENABLE_X64=True`). Las dependencias principales son:

- `jax` y `jaxlib`
- `flax`
- `optax`
- `h5py`
- `numpy`, `scipy`, `pandas`, `matplotlib`

Para verificar que tu entorno tiene todas las dependencias listas, puedes ejecutar:

```bash
python cli/run_1d.py --help
```

---

## 3. Características Principales de `rl_vmc`

| Componente | Opciones Soportadas | Banderas Principales (`cli/run_1d.py`) |
| :--- | :--- | :--- |
| **Sistema Físico 1D** | Número arbitrario de electrones $(N_\uparrow, N_\downarrow)$, frecuencia $\omega$, acoplamiento $\kappa$, suavizado $\alpha$ | `--n-up`, `--n-down`, `--omega`, `--kappa`, `--alpha` |
| **Ansatz Variacional** | `slater_jastrow`, `casscf` con orbitales (`HO`, `Poly`, `NN_Orbital_Simple`, `NN_Orbital_Full`) y correlación (`Padé-Jastrow`, `NN_Simple`, `NN_Full`) | `--ansatz-type`, `--orbital-type`, `--correlation-type`, `--n-orbitals`, `--n-dets` |
| **Propuesta MCMC** | Clásicas: `metropolis`, `mala`, `hmc`<br>Neuronales: `deepset`, `flat_mlp` | `--proposal-type`, `--action-mode`, `--use-spatial-gating`, `--base-step-size` |
| **Motor MCMC** | Movimiento conjunto (`all_electrons`), por partícula (`single_electron`) o simpléctico (`hmc`) | `--move-mode`, `--n-walkers`, `--n-burn-in` |
| **Aprendizaje por Refuerzo (RL)** | `td3` (Actor-Crítico con Replay Buffer Circular y Polyak) y `reinforce` (Policy Gradient con Baseline EMA) | `--rl-algorithm`, `--reward-type`, `--kappa-target`, `--exploration-std` |
| **Regímenes de Ejecución** | `vmc_only`, `rl_only`, `coupled`, `eval_only` | `--run-mode`, `--vmc-epochs`, `--rl-epochs` |
| **Archivo Unificado HDF5** | Guardado selectivo de pesos, curvas de entrenamiento, cadenas MCMC y métricas en `run_bundle.h5` | `--save-ansatz`, `--save-actor`, `--save-critic`, `--save-chain`, `--save-observables` |
| **Post-Procesamiento** | Cálculo de ACF, IAT ($\tau_{\text{int}}$), ESS/s, $G_{\text{RL}}$, observables físicos y 6 figuras comparativas | `cli/postprocess.py --bundles ...` |

---

## 4. Guía de Uso por Línea de Comandos (`cli/`)

### 4.1 Inspección de Banderas (`--help`)

Todas las herramientas CLI están construidas con `argparse` (`ArgumentDefaultsHelpFormatter` + `BooleanOptionalAction`), por lo que muestran todos los parámetros modificables, sus tipos y sus valores por defecto:

```bash
python cli/run_1d.py --help
python cli/postprocess.py --help
python cli/convert_legacy.py --help
```

---

### 4.2 Entrenamiento y Evaluación (`cli/run_1d.py`)

#### A) Régimen 1: Entrenar únicamente el Ansatz con un sampler clásico (`vmc_only`)

Entrena una función de onda `slater_jastrow` (`NN_Full`) usando **MALA** clásico y guarda el experimento en `results/1d_runs/mala_baseline/run_bundle.h5`:

```bash
python cli/run_1d.py \
  --run-name mala_baseline \
  --output-dir results/1d_runs \
  --run-mode vmc_only \
  --n-up 1 --n-down 1 --kappa 1.0 --alpha 0.2 \
  --ansatz-type slater_jastrow \
  --orbital-type HO \
  --correlation-type NN_Full \
  --proposal-type mala \
  --base-step-size 0.15 \
  --vmc-epochs 500 \
  --n-walkers 256 \
  --eval-steps 1000
```

#### B) Régimen 2: Entrenar un Sampler Neuronal RL sobre un Ansatz pre-entrenado (`rl_only`)

Carga el ansatz desde `mala_baseline/run_bundle.h5`, congela la función de onda y entrena un actor neuronal equivariante `deepset` con **TD3** y recompensa `cdlb`:

```bash
python cli/run_1d.py \
  --run-name rl_deepset_td3 \
  --output-dir results/1d_runs \
  --run-mode rl_only \
  --pretrained-bundle results/1d_runs/mala_baseline/run_bundle.h5 \
  --proposal-type deepset \
  --action-mode residual_mala \
  --use-spatial-gating \
  --rl-algorithm td3 \
  --reward-type cdlb \
  --rl-epochs 200 \
  --eval-steps 1000
```

#### C) Régimen 3: Co-entrenamiento acoplado VMC + RL (`coupled`)

Entrena primero o simultáneamente el ansatz y el actor neuronal en una sola ejecución:

```bash
python cli/run_1d.py \
  --run-name coupled_casscf_td3 \
  --output-dir results/1d_runs \
  --run-mode coupled \
  --ansatz-type casscf \
  --n-orbitals 4 --n-dets 4 \
  --proposal-type deepset \
  --rl-algorithm td3 \
  --reward-type log_esjd \
  --vmc-epochs 300 \
  --rl-epochs 150 \
  --eval-steps 1000
```

#### D) Régimen 4: Corrida de Producción Pura sobre modelos ya entrenados (`eval_only`)

Carga tanto el ansatz como el actor neuronal desde un `run_bundle.h5` previo y ejecuta una cadena larga de Monte Carlo para medir observables con alta precisión:

```bash
python cli/run_1d.py \
  --run-name prod_rl_deepset \
  --output-dir results/1d_runs \
  --run-mode eval_only \
  --pretrained-bundle results/1d_runs/rl_deepset_td3/run_bundle.h5 \
  --proposal-type deepset \
  --n-walkers 512 \
  --n-burn-in 500 \
  --eval-steps 5000 \
  --save-chain
```

---

### 4.3 Control Granular de Guardado (`--save-*` / `--no-save-*`)

Cada sección del archivo HDF5 y la exportación de archivos planos tienen su propia bandera booleana activable o desactivable:

| Bandera para Activar | Bandera para Desactivar | Default | Qué controla en `run_bundle.h5` |
| :--- | :--- | :---: | :--- |
| `--save-ansatz` | `--no-save-ansatz` | `True` | Pesos del ansatz en `/models/ansatz_params` |
| `--save-actor` | `--no-save-actor` | `True` | Pesos del actor RL en `/models/actor_params` |
| `--save-critic` | `--no-save-critic` | `True` | Pesos del crítico RL en `/models/critic_params` |
| `--save-train-history` | `--no-save-train-history` | `True` | Curvas de entrenamiento en `/training/vmc_history` y `/training/rl_history` |
| `--save-chain` | `--no-save-chain` | `True` | Posiciones MCMC `(T, W, N)` y energías locales `(T, W)` en `/sampling/` |
| `--save-step-metrics` | `--no-save-step-metrics` | `True` | Tasa de aceptación, ESJD y drift por paso en `/sampling/` |
| `--save-observables` | `--no-save-observables` | `True` | Diagnósticos Sokal (ACF, IAT, ESS) y observables físicos en `/postprocess/` |
| `--export-legacy` | `--no-export-legacy` | `False` | Exportación adicional a `params.json`, `*.pkl` y `*.csv` |

Ejemplo para guardar únicamente los pesos entrenados sin almacenar las cadenas de muestreo:

```bash
python cli/run_1d.py \
  --run-name lightweight_run \
  --run-mode coupled \
  --no-save-chain \
  --no-save-step-metrics
```

---

### 4.4 Post-Procesamiento Modular (`cli/postprocess.py`)

El módulo `cli/postprocess.py` recibe uno o varios archivos `run_bundle.h5`, calcula los diagnósticos de Sokal (ACF por FFT, $\tau_{\text{int}}$, ESS, ESS/s, $G_{\text{RL}}$, Time-to-Solution) y observables físicos (densidades por espín, correlación de pares $g(r_{12})$, picos de cristalización de Wigner, divergencia KL 2D para $N=2$), actualiza la sección `/postprocess/` de cada HDF5 y genera una tabla comparativa más 6 figuras listas para publicación:

```bash
python cli/postprocess.py \
  --bundles \
    results/1d_runs/mala_baseline/run_bundle.h5 \
    results/1d_runs/rl_deepset_td3/run_bundle.h5 \
  --labels "MALA Clásico" "RL-MALA (DeepSet TD3)" \
  --output-dir results/comparison_mala_vs_rl \
  --max-lag 200 \
  --n-bins 150 \
  --grid-limit 4.0 \
  --formats png pdf
```

Archivos generados en `results/comparison_mala_vs_rl/`:
- `comparison_summary.csv`: Tabla comparativa con $E_{\text{mean}}$, $\sigma_{\text{eff}}$, $\tau_{\text{int}}$, $\text{ESS}$, $\text{ESS/s}$, $G_{\text{RL}}$, tasa de aceptación, ESJD y estado de cristalización de Wigner.
- `fig1_energy_convergence.{png,pdf}`
- `fig2_autocorrelation_iat.{png,pdf}`
- `fig3_rl_training_curves.{png,pdf}`
- `fig4_wigner_density_pair_corr.{png,pdf}`
- `fig5_spatial_policy_profile.{png,pdf}`
- `fig6_kl_joint_density_2e.{png,pdf}` (cuando $N=2$)

---

### 4.5 Conversión Bidireccional Legacy $\leftrightarrow$ HDF5 (`cli/convert_legacy.py`)

Todos los modelos y corridas previas del repositorio (`params.json`, `ansatz_params.pkl`, `rl_actor_params.pkl`, `energy_history.csv`, `rl_training_history.csv`) pueden empaquetarse en un archivo `run_bundle.h5` para usarlos directamente en `cli/run_1d.py` o `cli/postprocess.py`:

```bash
# 1. Convertir una carpeta antigua (Legacy) a run_bundle.h5
python cli/convert_legacy.py \
  --mode legacy_to_h5 \
  --legacy-dir results/1D/test_run \
  --bundle-path results/1D/test_run/run_bundle.h5

# 2. Extraer un run_bundle.h5 de vuelta al formato Legacy (JSON + PKL + CSV)
python cli/convert_legacy.py \
  --mode h5_to_legacy \
  --bundle-path results/1D/test_run/run_bundle.h5 \
  --legacy-dir results/1D/test_run_exported
```

---

### 4.6 Auditoría Estadística Exhaustiva de Carpetas `results*` con Comparación Automática CASSCF (`cli/audit_results.py`)

Si tienes múltiples carpetas de resultados generadas por los scripts de producción en Bash (`1D_queue.sh`, `run_experiments_massive*.sh`, `run_sweeps*.sh`, `benchmark_parameter_search*.sh`, `run_massive_experiments*.sh`, `run_evaluar_wigner_casscf.sh`), puedes usar [`cli/audit_results.py`](cli/audit_results.py) junto con [`rl_vmc/postprocess/casscf_baseline.py`](rl_vmc/postprocess/casscf_baseline.py) para:

1. **Escaneo recursivo sin duplicados:** Explora todas las carpetas cuyo nombre empiece por `results*` (ignorando enlaces simbólicos duplicados) y detecta automáticamente todos los estados CASSCF pre-entrenados (`casscf_state.npz`) dentro de `uhf*` (ej. `uhf_1d_ho_v270826_m-determinants`).
2. **Empaquetado automático a HDF5:** Convierte cada modelo legacy (`params.json` + `*_ansatz_weights.pkl` + `*_actor_weights.pkl`, o corridas de muestreo CASSCF con `actor_params.npz` + `summary.csv`) en un archivo unificado `.h5` sin alterar los archivos originales.
3. **Rigor estadístico multi-evaluación (`--n-eval-runs`):** Ejecuta **$K$ muestreos MCMC independientes** con semillas distintas por cada modelo para estimar con precisión la media, desviación estándar entre corridas ($\pm 1\sigma$) y error estándar ($\mathrm{SEM}$) de:
   - Energía variacional $\langle E_L \rangle \pm \sigma_E$,
   - Tiempo de autocorrelación integrado de Sokal $\tau_{\text{int}} \pm \sigma_\tau$ y rendimiento $\text{ESS/s} \pm \sigma_{\text{ESS/s}}$,
   - Perfiles espaciales de densidad resueltos por espín $\bar{\rho}_{\text{tot}}(x) \pm \sigma_{\text{tot}}(x)$, $\bar{\rho}_\uparrow(x) \pm \sigma_\uparrow(x)$, $\bar{\rho}_\downarrow(x) \pm \sigma_\downarrow(x)$.
4. **Comparación automática contra Baseline CASSCF (`uhf*`):**
   - **Si existe una baseline CASSCF pre-entrenada** para el sistema físico $(N_\uparrow, N_\downarrow, \omega, g)$: calcula analíticamente su matriz de densidad reducida de 1 cuerpo (1-RDM) y sus densidades exactas $\rho_{\text{CASSCF, tot}}(x), \rho_{\text{CASSCF}, \uparrow}(x), \rho_{\text{CASSCF}, \downarrow}(x)$ directamente desde `casscf_state.npz`, evalúa la divergencia de Kullback-Leibler $D_{\text{KL}}(P_{\text{sampled}} \parallel P_{\text{CASSCF}})$ (total, espín $\uparrow$ y espín $\downarrow$) en cada corrida independiente, y superpone las curvas exactas CASSCF en todas las figuras de post-producción (`fig1`–`fig3`, `fig6_walker_traces`, `fig7_article_densities_comparison`, `fig8_article_kl_divergence` y `global_audit_dashboard.pdf`), siguiendo exactamente el estilo visual de `scripts/evaluar_wigner_casscf_2.py`.
   - **Si NO existe baseline CASSCF** para un sistema físico dado: el pipeline continúa limpiamente sin errores y genera las tablas y figuras mostrando únicamente los resultados estadísticos de la evaluación.

```bash
python cli/audit_results.py \
  --search-root . \
  --pattern "results*" \
  --uhf-root . \
  --uhf-pattern "uhf*" \
  --out-dir results_exhaustive_audit \
  --n-eval-runs 5 \
  --n-burn-in 2000 \
  --n-steps 20000 \
  --batch-size 512
```

---

## 5. Notebook Interactivo (`notebooks/tutorial_pipeline_1d.ipynb`)

El notebook [`notebooks/tutorial_pipeline_1d.ipynb`](notebooks/tutorial_pipeline_1d.ipynb) contiene una demostración completa, autónoma y ya ejecutada de toda la paquetería `rl_vmc`.

### Cómo abrirlo y ejecutarlo

```bash
jupyter notebook notebooks/tutorial_pipeline_1d.ipynb
# o en JupyterLab / VS Code / Antigravity IDE
```

### Qué incluye el Notebook

1. **Sección 1 — Configuración y Banderas CLI:** Creación de `Experiment1DConfig`, serialización JSON y parseo simulado de banderas `argparse` (`--flag` / `--no-flag`).
2. **Sección 2 — Hamiltoniano 1D y Ansätze:** Prueba de `Harmonic1DHamiltonian` y de las 8 combinaciones de `SlaterJastrow1DAnsatz` (`HO`, `Poly`, `NN_Orbital_Simple`, `NN_Orbital_Full` $\times$ `Padé-Jastrow`, `NN_Simple`, `NN_Full`) y `CASSCF1DAnsatz` (con energía local y gradientes respecto a posiciones y parámetros).
3. **Sección 3 — Políticas de Propuesta:** Prueba de las propuestas clásicas (`Metropolis`, `MALA`, `HMC`) y neuronales (`DeepSet` y `FlatMLP` con `residual_mala`, `scale_only`, `full_gaussian` y `use_spatial_gating`).
4. **Sección 4 — Motor MCMC y Recompensas RL:** Ejecución de `MHSampler1D` en modos `all_electrons`, `single_electron` y `hmc`, y evaluación de las 4 funciones de recompensa (`cdlb`, `log_esjd`, `barker_jump`, `kappa_jump`).
5. **Sección 5 — Ejecución de los 4 Regímenes del Pipeline:**
   - `vmc_only` con **MALA**.
   - `rl_only` con **TD3 + DeepSet** sobre el ansatz pre-entrenado.
   - `coupled` con **REINFORCE + FlatMLP**.
   - `eval_only` sobre el bundle entrenado con TD3.
6. **Sección 6 — Inspección del HDF5 (`run_bundle.h5`) y Conversión Legacy $\leftrightarrow$ HDF5:** Recorrido de la jerarquía HDF5 y verificación de ida y vuelta con `export_h5_to_legacy_dir` y `convert_legacy_dir_to_h5`.
7. **Sección 7 — Post-Procesamiento Modular y Figuras:** Ejecución de `run_postprocessing_pipeline` comparando los 4 experimentos, tabla de métricas Sokal (`IAT`, `ESS/s`, `G_RL`) y renderizado inline de las 6 figuras diagnósticas.

---

## 6. Uso Programático desde Python

Además de los scripts en `cli/`, puedes invocar cualquier módulo de `rl_vmc` directamente desde Python:

```python
from rl_vmc.config import Experiment1DConfig, PostProcessConfig
from rl_vmc.trainers.runner_1d import run_experiment_1d
from rl_vmc.postprocess.plotter import run_postprocessing_pipeline

# 1. Configurar y correr un experimento acoplado VMC + RL (TD3 + DeepSet)
cfg = Experiment1DConfig(run_name="demo_coupled", output_dir="results/demo")
cfg.system.n_up = 1
cfg.system.n_down = 1
cfg.ansatz.ansatz_type = "slater_jastrow"
cfg.proposal.proposal_type = "deepset"
cfg.rl.algorithm = "td3"
cfg.rl.reward_type = "cdlb"
cfg.train.run_mode = "coupled"
cfg.train.vmc_epochs = 200
cfg.train.rl_epochs = 100
cfg.eval.n_Measure_steps = 1000

result = run_experiment_1d(cfg, verbose=True)
print("Bundle HDF5 guardado en:", result["bundle_path"])

# 2. Ejecutar el post-procesamiento modular sobre el bundle resultante
pp_cfg = PostProcessConfig(
    bundle_paths=[result["bundle_path"]],
    labels=["RL-MALA DeepSet (TD3)"],
    output_dir="results/demo/postprocess",
)
summary_df = run_postprocessing_pipeline(pp_cfg, verbose=True)
print(summary_df)
```

---

## 7. Convivencia y Separación con la Implementación Original (`src/` y `scripts/`)

Las dos implementaciones dentro del repositorio son **100% independientes**:

- **Implementación Original (`src/`, `scripts/`, `*.sh`):** Permanece completamente intacta (cero líneas modificadas) y no importa nada de `rl_vmc/`. Todos tus scripts bash actuales siguen funcionando exactamente igual que antes.
- **Nueva Paquetería Modular (`rl_vmc/`, `cli/`, `docs/`, `notebooks/`):** No importa ningún módulo de `src/` ni de `scripts/`, pero es **100% compatible a nivel binario** con los diccionarios de pesos `.pkl` y `params.json` generados por la implementación original gracias a `rl_vmc/core/io_bundle.py` y `cli/audit_results.py`.

