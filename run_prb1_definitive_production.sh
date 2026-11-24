#!/usr/bin/env bash
# ==============================================================================
# run_prb1_definitive_production.sh
# ==============================================================================
# PRODUCCIÓN DEFINITIVA DE MODELOS — PRIMER ARTÍCULO (PRB Letter: Pure-CDLB RL-MALA)
#
# Ejecuta el benchmark largo y definitivo de muestreo MCMC puro sobre funciones
# de onda singlete CASSCF exactas congeladas (`--ansatz_type casscf_npz
# --no-casscf_trainable`), garantizando convergencia asintótica de:
#   1. MALA clásico en la malla completa de anchos de propuesta:
#        sigma in (0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)
#      usando la relación exacta de Langevin: tau(sigma) = sigma^2 / 2
#      (de modo que sigma_MALA = sqrt(2 * tau) == sigma idénticamente).
#   2. Random-Walk Metropolis-Hastings (RWMH) sin deriva en los mismos sigmas:
#        sigma in (0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0)
#   3. RL-MALA óptimo entrenado EXCLUSIVAMENTE con la recompensa CDLB pura:
#        - `--reward_type cdlb --cdlb_exact_entropy`
#        - `--confinement_weight 0.0 --outward_drift_penalty 0.0` (100% CDLB puro)
#        - Arquitectura más óptima: DeepSet equivariante por paridad exacta O(1)
#          (`--parity_equivariant`) + canal de escala desbloqueado calibrado
#          (`--unclip_sigma_action --scale_clip_max <sig_sc>`)
#        - Hiperparámetros óptimos de entrenamiento TD3 obtenidos en los
#          benchmarks de hyperparameter search (Baker + Ayler, 114 modelos) con
#          planificador cosenoidal de LR y centrado de recompensa (Naik et al.).
#
# Sistemas físicos (21 sistemas = 3 x 7):
#   N in (4, 6, 10)  x  g in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0),  omega = 0.1, a = 0.1
#
# Organización estricta dentro de ${BASE_OUT_DIR}:
#   ${BASE_OUT_DIR}/
#     ├── casscf_refs/   -> Estados exactos CASSCF (.npz y metadatos JSON)
#     ├── runs/          -> Bundles HDF5 por sistema/modelo (<SYS_TAG>/<MODEL>/run_bundle.h5)
#     ├── csv/           -> TODOS los CSVs (maestros, barridos en sigma, convergencia, sondas)
#     ├── logs/          -> TODOS los logs con timestamps (orquestador, CASSCF, jobs, plots)
#     └── figures/       -> TODAS las figuras de interés (por sistema y globales de artículo)
#
# Gestión de colas, multi-GPU, multi-job/GPU y protección especial de memoria en N=10:
#   - Auto-detección de GPUs (`nvidia-smi`) y cola dinámica con múltiples slots por GPU.
#   - Protección automática para N=10 (63,504 determinantes CI, b=256):
#       * Reserva exclusiva de GPU por defecto (`--n10-jobs-per-gpu 1`) aunque N=4,6
#         corran varios jobs por GPU (`--jobs-per-gpu 2` o `3`).
#       * Micro-batching real del Hessiano cinético (`--eloc_chunk_size 16` < B=128)
#         y bloques `lax.scan` compactos (`--jit_chunk_size 250`).
#       * Inicialización Wigner centrada en la densidad exacta CASSCF para evitar
#         singularidades nodales y derivas artificiales en sigma pequeños.
#
# Ejemplos de uso en Ayler (3 GPUs) o Baker (1 GPU):
#   chmod +x run_prb1_definitive_production.sh
#
#   # Corrida completa en Ayler (3 GPUs, 2 jobs/GPU en N=4,6 y 1 job/GPU automático en N=10):
#   nohup ./run_prb1_definitive_production.sh --gpus "0 1 2" --jobs-per-gpu 2 \
#       > launch_prb1_definitive.out 2>&1 &
#
#   # Verificación rápida sin ejecutar (inspecciona comandos, tau(sigma) y rutas):
#   ./run_prb1_definitive_production.sh --dry-run
#
#   # Prueba corta de humo end-to-end antes de lanzar producción larga:
#   ./run_prb1_definitive_production.sh --quick-test --systems "4:1.0"
#
#   # Re-generar únicamente tablas CSV y figuras desde HDF5 ya completados:
#   ./run_prb1_definitive_production.sh --postprocess-only
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 0. AUTO-DETECCIÓN DEL INTÉRPRETE PYTHON CON JAX + H5PY + MATPLOTLIB
# ==============================================================================
detect_python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]] && "$PYTHON_BIN" -c "import jax, flax, h5py, pandas, matplotlib" >/dev/null 2>&1; then
        echo "$PYTHON_BIN"
        return 0
    fi
    local candidates=(
        "python3"
        "python"
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
        "$HOME/miniconda3/bin/python3"
        "$HOME/anaconda3/bin/python3"
        "$HOME/.conda/envs/rl_vmc/bin/python3"
        "/opt/conda/bin/python3"
    )
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            if "$c" -c "import jax, flax, h5py, pandas, matplotlib" >/dev/null 2>&1; then
                echo "$c"
                return 0
            fi
        fi
    done
    echo "python3"
}

PYTHON_BIN="$(detect_python_bin)"

# Variables globales de entorno para precisión doble IEEE-754 e higiene de hilos/VRAM
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_cache_${USER:-rlvmc}"
mkdir -p "$MPLCONFIGDIR"

# ==============================================================================
# 1. PARÁMETROS CONFIGURABLES DEL BENCHMARK DEFINITIVO
# ==============================================================================
BASE_OUT_DIR="${BASE_OUT_DIR:-results_prb1_definitive_production}"

# Grilla física definitiva: N=(4, 6, 10) x g=(0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)
N_LIST="${N_LIST:-4 6 10}"
G_LIST="${G_LIST:-0.1 0.5 1.0 2.0 3.0 5.0 8.0}"
OMEGA="${OMEGA:-0.1}"
SOFTENING="${SOFTENING:-0.1}"
SYSTEMS_FILTER="${SYSTEMS_FILTER:-}"

# Grilla exacta de sigmas para MALA (tau = sigma^2 / 2) y RWMH (rw_sigma = sigma)
SIGMA_LIST="${SIGMA_LIST:-0.001 0.005 0.01 0.05 0.1 0.15 0.25 0.5 1.0}"

# Longitud de cadenas (Benchmark Largo: asegura convergencia de MALA, RWMH y RL-MALA)
N_WALKERS="${N_WALKERS:-128}"
RL_BURN_IN="${RL_BURN_IN:-3000}"
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-8000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-5000}"
MEASURE_STEPS="${MEASURE_STEPS:-50000}"
SEED_BASE="${SEED_BASE:-42}"

# Inicialización de entrenamiento RL y de evaluación MCMC
RL_INIT_MODE="${RL_INIT_MODE:-randomized_wigner}"
RL_INIT_STD="${RL_INIT_STD:-3.0}"
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"

# Hiperparámetros óptimos de entrenamiento TD3 + CDLB Puro
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.05}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"
RL_REPLAY_CAPACITY="${RL_REPLAY_CAPACITY:-100000}"
RL_REPLAY_CAPACITY_N10="${RL_REPLAY_CAPACITY_N10:-50000}"
POLICY_DELAY="${POLICY_DELAY:-2}"
TAU_POLYAK="${TAU_POLYAK:-0.005}"
TARGET_NOISE="${TARGET_NOISE:-0.15}"
NOISE_CLIP="${NOISE_CLIP:-0.35}"
EXP_NOISE_INIT="${EXP_NOISE_INIT:-0.20}"
EXP_NOISE_FINAL="${EXP_NOISE_FINAL:-0.02}"
REWARD_CENTERING_LR="${REWARD_CENTERING_LR:-0.01}"

# Parámetros de guardado HDF5 compacto y exacto
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-64}"
SAVE_WALKERS_MAX_BATCH_N10="${SAVE_WALKERS_MAX_BATCH_N10:-32}"
SAVE_ENERGY_MODE="${SAVE_ENERGY_MODE:-step_mean}"
SAVE_WALKERS_DTYPE="${SAVE_WALKERS_DTYPE:-float32}"
SAVE_STATS_DTYPE="${SAVE_STATS_DTYPE:-float32}"

# Protección de memoria por tamaño de sistema N (especialmente N=10)
ELOC_CHUNK_N4="${ELOC_CHUNK_N4:-128}"
ELOC_CHUNK_N6="${ELOC_CHUNK_N6:-64}"
ELOC_CHUNK_N10="${ELOC_CHUNK_N10:-16}"     # < N_WALKERS (128): activa micro-batching 8x del Hessiano 10x10
JIT_CHUNK_DEFAULT="${JIT_CHUNK_DEFAULT:-1000}"
JIT_CHUNK_N10="${JIT_CHUNK_N10:-250}"

# Parámetros CASSCF de respaldo si algún estado (ej. N=10, g=0.1, 0.5, 8.0) no está en CASSCF_csv/
CASSCF_FALLBACK_BASIS_N10="${CASSCF_FALLBACK_BASIS_N10:-64}"
CASSCF_FALLBACK_QUAD_N10="${CASSCF_FALLBACK_QUAD_N10:-500}"  # Cuadratura densa para a=0.1 sin sesgo ERI
CASSCF_FALLBACK_CORE_N10="${CASSCF_FALLBACK_CORE_N10:-3}"
CASSCF_FALLBACK_ACT_N10="${CASSCF_FALLBACK_ACT_N10:-5}"      # CAS(4,5) = 100 dets + 3 core optimizados

# Gestión de colas y GPUs
GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
N10_JOBS_PER_GPU="${N10_JOBS_PER_GPU:-1}"  # Por defecto N=10 toma 1 GPU exclusiva para no desbordar VRAM
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"
FORCE_CPU="${FORCE_CPU:-0}"
MIN_FREE_DISK_MB="${MIN_FREE_DISK_MB:-512}"

# Control de ejecución y filtros de familias de modelos
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
REALTIME_SYSTEM_POSTPROCESS="${REALTIME_SYSTEM_POSTPROCESS:-1}"
RUN_MALA="${RUN_MALA:-1}"
RUN_RWMH="${RUN_RWMH:-1}"
RUN_RL="${RUN_RL:-1}"
WITH_WIDE_RL="${WITH_WIDE_RL:-0}"

# Formato de figuras de interés
FIGURE_FORMATS="${FIGURE_FORMATS:-pdf,png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TRACE_STEPS_PLOT="${TRACE_STEPS_PLOT:-10000}"

print_help() {
    cat <<EOF
Uso: $0 [OPCIONES] [CARPETA_SALIDA]

Orquestador definitivo de producción de modelos para el Primer Artículo (PRB Letter):
  - Sistemas     : N in (${N_LIST}), g in (${G_LIST}), omega=${OMEGA}, a=${SOFTENING}
  - MALA & RWMH  : sigma in (${SIGMA_LIST}) [MALA usa tau = sigma^2 / 2 exacto]
  - RL-MALA      : Recompensa 100% CDLB pura + Actor DeepSet O(1) paridad-equivariante + TD3 óptimo

Opciones de hardware y cola:
  --gpus "0 1 2"             Lista de IDs de GPU separadas por espacio o coma (default: auto)
  --jobs-per-gpu, -j K       Trabajos concurrentes por GPU para N=4 y N=6 (default: ${JOBS_PER_GPU})
  --n10-jobs-per-gpu K       Trabajos concurrentes por GPU permitidos en N=10 (default: ${N10_JOBS_PER_GPU}, exclusivo)
  --max-parallel-jobs M      Límite global de trabajos simultáneos (default: auto = N_GPUS * JOBS_PER_GPU)
  --cpu                      Forzar ejecución en CPU (útil para pruebas locales)

Opciones de longitud y convergencia:
  --walkers B                Caminantes paralelos por cadena (default: ${N_WALKERS})
  --rl-burn-in S             Barridos de termalización previos al entrenamiento RL (default: ${RL_BURN_IN})
  --rl-steps S               Barridos de entrenamiento TD3 con CDLB pura (default: ${RL_TRAIN_STEPS})
  --chains C                 Cadenas independientes de evaluación por modelo (default: ${N_EVAL_CHAINS})
  --eval-burn-in S           Barridos de termalización descartados por cadena de evaluación (default: ${EVAL_BURN_IN})
  --measure-steps S          Barridos de medición en producción por cadena (default: ${MEASURE_STEPS})
  --quick-test               Modo rápido de validación (walkers=32, rl=400, burn=200, steps=1000, chains=2)

Filtros de sistemas y modelos:
  --systems SPEC             Filtro de sistemas, ej. "4:all 6:0.5,1.0 10:1.0" o "N6_w0.1_g1.0"
  --n-list "4 6 10"          Subconjunto de números de partículas N
  --g-list "0.1 0.5 1.0"     Subconjunto de acoplamientos g
  --sigmas "0.01 0.1 1.0"    Subconjunto de sigmas para MALA y RWMH
  --only-rl                  Ejecutar únicamente el sampler óptimo RL-MALA (CDLB)
  --only-classical           Ejecutar únicamente los barridos clásicos MALA y RWMH
  --only-mala                Ejecutar únicamente MALA
  --only-rwmh                Ejecutar únicamente RWMH
  --with-wide-rl             Añadir opcionalmente variante RL-MALA de horizonte amplio (A=0.80)

Control de flujo y salidas:
  --out-dir DIR              Carpeta raíz de resultados (default: ${BASE_OUT_DIR})
  --postprocess-only, -p     Omitir corridas MCMC y regenerar todos los CSVs y figuras
  --no-realtime-plots        Posponer el postprocesamiento por sistema hasta el cierre global
  --formats "pdf,png"        Formatos de exportación de figuras (default: ${FIGURE_FORMATS})
  --dpi DPI                  Resolución de figuras rasterizadas (default: ${FIGURE_DPI})
  --force                    Re-ejecutar aunque ya exista run_bundle.h5 válido
  --dry-run                  Imprimir la cola completa de comandos sin ejecutarlos
  --help, -h                 Mostrar esta ayuda
EOF
}

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus|--gpu|--gpu-list) GPU_LIST="${2//,/ }"; shift 2 ;;
        --jobs-per-gpu|--jobs|-j) JOBS_PER_GPU="$2"; shift 2 ;;
        --n10-jobs-per-gpu) N10_JOBS_PER_GPU="$2"; shift 2 ;;
        --max-parallel-jobs) MAX_PARALLEL_JOBS="$2"; shift 2 ;;
        --cpu) FORCE_CPU=1; shift ;;
        --walkers|-w) N_WALKERS="$2"; shift 2 ;;
        --rl-burn-in) RL_BURN_IN="$2"; shift 2 ;;
        --rl-steps|--train-steps) RL_TRAIN_STEPS="$2"; shift 2 ;;
        --chains|--eval-chains) N_EVAL_CHAINS="$2"; shift 2 ;;
        --eval-burn-in) EVAL_BURN_IN="$2"; shift 2 ;;
        --measure-steps|--steps) MEASURE_STEPS="$2"; shift 2 ;;
        --quick-test)
            N_WALKERS=32
            RL_BURN_IN=200
            RL_TRAIN_STEPS=400
            N_EVAL_CHAINS=2
            EVAL_BURN_IN=200
            MEASURE_STEPS=1000
            LR_WARMUP_STEPS=50
            SAVE_WALKERS_MAX_BATCH=16
            SAVE_WALKERS_MAX_BATCH_N10=16
            TRACE_STEPS_PLOT=1000
            shift
            ;;
        --systems|--system|-s) SYSTEMS_FILTER="$2"; shift 2 ;;
        --n-list) N_LIST="${2//,/ }"; shift 2 ;;
        --g-list) G_LIST="${2//,/ }"; shift 2 ;;
        --sigmas|--sigma-list) SIGMA_LIST="${2//,/ }"; shift 2 ;;
        --only-rl) RUN_MALA=0; RUN_RWMH=0; RUN_RL=1; shift ;;
        --only-classical) RUN_MALA=1; RUN_RWMH=1; RUN_RL=0; shift ;;
        --only-mala) RUN_MALA=1; RUN_RWMH=0; RUN_RL=0; shift ;;
        --only-rwmh) RUN_MALA=0; RUN_RWMH=1; RUN_RL=0; shift ;;
        --with-wide-rl) WITH_WIDE_RL=1; shift ;;
        --out-dir|-o) BASE_OUT_DIR="$2"; shift 2 ;;
        --postprocess-only|--plots-only|-p) POSTPROCESS_ONLY=1; shift ;;
        --no-realtime-plots) REALTIME_SYSTEM_POSTPROCESS=0; shift ;;
        --formats|--figure-formats) FIGURE_FORMATS="$2"; shift 2 ;;
        --pdf) FIGURE_FORMATS="pdf"; shift ;;
        --png) FIGURE_FORMATS="png"; shift ;;
        --both-formats) FIGURE_FORMATS="pdf,png"; shift ;;
        --dpi|--ppi) FIGURE_DPI="$2"; shift 2 ;;
        --force) FORCE_RERUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) print_help; exit 0 ;;
        *) POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    BASE_OUT_DIR="${POSITIONAL_ARGS[0]}"
fi

# Convertir ruta de salida a ruta absoluta para evitar ambigüedades
if [[ "$BASE_OUT_DIR" != /* ]]; then
    BASE_OUT_DIR="${SCRIPT_DIR}/${BASE_OUT_DIR}"
fi

# ==============================================================================
# 2. ESTRUCTURA ESTRICTA DE DIRECTORIOS DE SALIDA
# ==============================================================================
CASSCF_REF_DIR="${BASE_OUT_DIR}/casscf_refs"
RUNS_DIR="${BASE_OUT_DIR}/runs"

# Carpetas dedicadas solicitadas: CSVs, Logs y Figuras por separado
CSV_DIR="${BASE_OUT_DIR}/csv"
CSV_SYS_DIR="${CSV_DIR}/per_system"
CSV_TRAIN_DIR="${CSV_DIR}/training_histories"
CSV_PROD_DIR="${CSV_DIR}/production_metrics"
CSV_CASSCF_DIR="${CSV_DIR}/casscf_exact_densities"

LOG_DIR="${BASE_OUT_DIR}/logs"
LOG_CASSCF_DIR="${LOG_DIR}/casscf"
LOG_SAMPLING_DIR="${LOG_DIR}/sampling"
LOG_POST_DIR="${LOG_DIR}/postprocess"
TRACKER_DIR="${LOG_DIR}/.queue_tracker"
POST_PIDS_DIR="${LOG_DIR}/.postprocess_pids"

FIG_DIR="${BASE_OUT_DIR}/figures"
FIG_SYS_DIR="${FIG_DIR}/per_system"
FIG_GLOBAL_DIR="${FIG_DIR}/global_article"
FIG_CASSCF_DIR="${FIG_DIR}/casscf_diagnostics"

MASTER_LIVE_CSV="${CSV_DIR}/master_runs_live.csv"
MASTER_FINAL_CSV="${CSV_DIR}/master_definitive_summary.csv"
BEST_FAMILY_CSV="${CSV_DIR}/best_per_family_by_system.csv"
SIGMA_SWEEP_CSV="${CSV_DIR}/sigma_sweep_curves_all_systems.csv"
CASSCF_SUMMARY_CSV="${CSV_DIR}/casscf_references_summary.csv"

ORCHESTRATOR_LOG="${LOG_DIR}/orchestrator_events.log"
FAILED_JOBS_LOG="${LOG_DIR}/failed_jobs.log"

mkdir -p \
    "$CASSCF_REF_DIR" \
    "$RUNS_DIR" \
    "$CSV_DIR" "$CSV_SYS_DIR" "$CSV_TRAIN_DIR" "$CSV_PROD_DIR" "$CSV_CASSCF_DIR" \
    "$LOG_DIR" "$LOG_CASSCF_DIR" "$LOG_SAMPLING_DIR" "$LOG_POST_DIR" "$TRACKER_DIR" "$POST_PIDS_DIR" \
    "$FIG_DIR" "$FIG_SYS_DIR" "$FIG_GLOBAL_DIR" "$FIG_CASSCF_DIR"

rm -f "$TRACKER_DIR"/job_*.meta "$POST_PIDS_DIR"/post_*.pid 2>/dev/null || true
touch "$ORCHESTRATOR_LOG" "$FAILED_JOBS_LOG"

# ==============================================================================
# 3. UTILIDADES DE TIMESTAMPS, BITÁCORA Y LIMPIEZA DE SEÑALES
# ==============================================================================
ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

format_duration() {
    local secs="$1"
    if [[ "$secs" -lt 0 ]]; then secs=0; fi
    printf "%02dh:%02dm:%02ds" $((secs / 3600)) $(((secs % 3600) / 60)) $((secs % 60))
}

log_msg() {
    local level="$1"
    shift
    local line="[$(ts)] [${level}] $*"
    echo "$line"
    echo "$line" >> "$ORCHESTRATOR_LOG"
}

record_failure() {
    local stage="$1"
    local sys_tag="$2"
    local model_key="$3"
    local detail="$4"
    local line="[$(ts)] [FAIL] stage=${stage} | system=${sys_tag} | model=${model_key} | ${detail}"
    echo "$line" >&2
    echo "$line" >> "$ORCHESTRATOR_LOG"
    echo "$line" >> "$FAILED_JOBS_LOG"
    TOTAL_FAILED=$((TOTAL_FAILED + 1))
}

cleanup_on_interrupt() {
    echo ""
    log_msg "SIGNAL" "Interrupción recibida (SIGINT/SIGTERM). Deteniendo jobs activos de forma limpia..."
    for mf in "$TRACKER_DIR"/job_*.meta; do
        [[ -e "$mf" ]] || continue
        local pid=""
        # shellcheck disable=SC1090
        source "$mf"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
        rm -f "$mf"
    done
    for pf in "$POST_PIDS_DIR"/post_*.pid; do
        [[ -e "$pf" ]] || continue
        local ppid
        ppid="$(cat "$pf" 2>/dev/null || true)"
        if [[ -n "$ppid" ]] && kill -0 "$ppid" 2>/dev/null; then
            kill -TERM "$ppid" 2>/dev/null || true
        fi
        rm -f "$pf"
    done
    rm -rf "$MPLCONFIGDIR" 2>/dev/null || true
    log_msg "SIGNAL" "Procesos hijos detenidos. Puedes reanudar ejecutando el mismo comando (los .h5 completos se saltarán)."
    exit 130
}

trap cleanup_on_interrupt INT TERM

# ==============================================================================
# 4. DETECCIÓN DE GPUs Y CONFIGURACIÓN DE SLOTS DE COLA (CON CUIDADO EN N=10)
# ==============================================================================
AVAILABLE_GPUS=()
USE_CPU_BACKEND=0

if [[ "$FORCE_CPU" -eq 1 ]]; then
    AVAILABLE_GPUS=("cpu")
    USE_CPU_BACKEND=1
elif [[ "$GPU_LIST" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r gid; do
            gid="$(echo "$gid" | tr -d '[:space:]')"
            [[ -n "$gid" ]] && AVAILABLE_GPUS+=("$gid")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
    fi
    if [[ ${#AVAILABLE_GPUS[@]} -eq 0 ]]; then
        AVAILABLE_GPUS=("cpu")
        USE_CPU_BACKEND=1
    fi
else
    read -r -a AVAILABLE_GPUS <<< "$GPU_LIST"
    if [[ ${#AVAILABLE_GPUS[@]} -eq 1 && "${AVAILABLE_GPUS[0]}" == "cpu" ]]; then
        USE_CPU_BACKEND=1
    fi
fi

NUM_DEVICES="${#AVAILABLE_GPUS[@]}"
if [[ "$JOBS_PER_GPU" -lt 1 ]]; then JOBS_PER_GPU=1; fi
if [[ "$N10_JOBS_PER_GPU" -lt 1 ]]; then N10_JOBS_PER_GPU=1; fi
if [[ "$N10_JOBS_PER_GPU" -gt "$JOBS_PER_GPU" ]]; then N10_JOBS_PER_GPU="$JOBS_PER_GPU"; fi

# Peso en slots de un job N=10 dentro de una GPU:
# Si JOBS_PER_GPU=2 y N10_JOBS_PER_GPU=1 -> cada job N=10 consume 2 slots (toda la GPU exclusiva).
N10_SLOT_WEIGHT=$(( (JOBS_PER_GPU + N10_JOBS_PER_GPU - 1) / N10_JOBS_PER_GPU ))
if [[ "$N10_SLOT_WEIGHT" -lt 1 ]]; then N10_SLOT_WEIGHT=1; fi

if [[ "$MAX_PARALLEL_JOBS" == "auto" ]]; then
    MAX_TOTAL_SLOTS=$(( NUM_DEVICES * JOBS_PER_GPU ))
else
    MAX_TOTAL_SLOTS="$MAX_PARALLEL_JOBS"
fi

# Fracción de memoria XLA por job según concurrencia en GPU
MEM_FRAC_STD="$("$PYTHON_BIN" -c "j=max(1, int('$JOBS_PER_GPU')); print(f'{min(0.45, round(0.84/j, 2)):.2f}')")"
MEM_FRAC_N10="$("$PYTHON_BIN" -c "j=max(1, int('$N10_JOBS_PER_GPU')); print(f'{min(0.80, round(0.82/j, 2)):.2f}')")"

# Contadores globales de progreso
TOTAL_PLANNED_JOBS=0
TOTAL_COMPLETED_JOBS=0
TOTAL_SKIPPED_JOBS=0
TOTAL_LAUNCHED_JOBS=0
TOTAL_FAILED=0
BENCHMARK_START_EPOCH="$(date +%s)"

# ==============================================================================
# 5. FUNCIONES CIENTÍFICAS Y DE VALIDACIÓN (FILTROS, HIPERPARÁMETROS, HDF5)
# ==============================================================================
should_run_system_pair() {
    local n_req="$1"
    local g_req="$2"
    if [[ -z "$SYSTEMS_FILTER" ]]; then
        return 0
    fi
    "$PYTHON_BIN" - "$SYSTEMS_FILTER" "$n_req" "$g_req" <<'PYEOF'
import re, sys
spec, n_req, g_req = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
mp = {}
for tok in re.split(r"[\s;]+", spec.strip()):
    if not tok:
        continue
    m_tag = re.match(r"^N(\d+)_w[0-9.]+_g([0-9.]+)$", tok, flags=re.IGNORECASE)
    if m_tag:
        nv, gv = int(m_tag.group(1)), float(m_tag.group(2))
        if mp.get(nv) is not None or nv not in mp:
            mp.setdefault(nv, []).append(gv)
        continue
    if ":" in tok:
        n_p, g_p = tok.split(":", 1)
        nv = int(re.sub(r"^[Nn]", "", n_p.strip()))
        if g_p.strip().lower() in ("all", "*"):
            mp[nv] = None
        else:
            gs = [float(x) for x in g_p.split(",") if x.strip()]
            if nv in mp and mp[nv] is None:
                continue
            mp.setdefault(nv, []).extend(gs)
    else:
        nv = int(re.sub(r"^[Nn]", "", tok.strip()))
        mp[nv] = None
if n_req not in mp:
    sys.exit(1)
allowed = mp[n_req]
if allowed is None or any(abs(g_req - ga) <= 1e-4 for ga in allowed):
    sys.exit(0)
sys.exit(1)
PYEOF
}

# Exacta conversión sigma -> tau = sigma^2 / 2 en IEEE-754 float64
sigma_to_mala_tau() {
    local sig="$1"
    "$PYTHON_BIN" -c "import sys; s=float(sys.argv[1]); print(f'{0.5 * (s ** 2):.12g}')" "$sig"
}

# Hiperparámetros óptimos de RL-MALA (CDLB puro) por régimen físico (N, g)
# basados en el Hyperparameter Search de 114 modelos (Baker + Ayler, Tablas III-V PRB)
# y calibración dimensional para evitar colapso de aceptación en 10D.
# Retorna: "TAU_RL ACTOR_AMP PENALTY SIG_SCALE ELOC_CHUNK JIT_CHUNK REPLAY_CAP MAX_SAVE_BATCH"
get_optimal_rl_params() {
    local N="$1"
    local G="$2"
    "$PYTHON_BIN" - "$N" "$G" \
        "$ELOC_CHUNK_N4" "$ELOC_CHUNK_N6" "$ELOC_CHUNK_N10" \
        "$JIT_CHUNK_DEFAULT" "$JIT_CHUNK_N10" \
        "$RL_REPLAY_CAPACITY" "$RL_REPLAY_CAPACITY_N10" \
        "$SAVE_WALKERS_MAX_BATCH" "$SAVE_WALKERS_MAX_BATCH_N10" <<'PYEOF'
import sys

N = int(sys.argv[1])
g = float(sys.argv[2])
ec4, ec6, ec10 = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
jc_def, jc10 = int(sys.argv[6]), int(sys.argv[7])
rc_def, rc10 = int(sys.argv[8]), int(sys.argv[9])
sb_def, sb10 = int(sys.argv[10]), int(sys.argv[11])

strong_g = (g >= 2.5)

if N <= 4:
    # Ganadores Baker + Ayler N=4 (Tabla III): tau=0.10-0.12, A=0.12-0.15, pen=0.25-0.30, sig=1.15
    if strong_g:
        tau, amp, pen, sig = "0.12", "0.12", "0.30", "1.15"
    else:
        tau, amp, pen, sig = "0.10", "0.15", "0.25", "1.15"
    eloc_c, jit_c, rep_c, save_b = ec4, jc_def, rc_def, sb_def
elif N <= 6:
    # Ganadores Baker + Ayler N=6 (Tablas IV-V): D_KL^spin ~ 6e-5, +15% a +86% ESS/s
    if strong_g:
        tau, amp, pen, sig = "0.08", "0.10", "0.30", "1.10"
    else:
        tau, amp, pen, sig = "0.10", "0.14", "0.25", "1.15"
    eloc_c, jit_c, rep_c, save_b = ec6, jc_def, rc_def, sb_def
else:
    # N=10 (10 electrones simultáneos): calibración de escala para mantener alpha ~ 25%-45%
    # evitando el cuello de botella dimensional (sigma_well / sigma)^10 y protegiendo VRAM
    if strong_g:
        tau, amp, pen, sig = "0.06", "0.08", "0.30", "0.80"
    else:
        tau, amp, pen, sig = "0.07", "0.08", "0.25", "0.85"
    eloc_c, jit_c, rep_c, save_b = ec10, jc10, rc10, sb10

print(f"{tau} {amp} {pen} {sig} {eloc_c} {jit_c} {rep_c} {save_b}")
PYEOF
}

# Espaciado de red de Wigner centrado en 0 a partir de la densidad exacta CASSCF
# Evita derivas fuera de equilibrio para sigmas pequeños (0.001) y trampas nodales en N=10
get_wigner_centered_offsets() {
    local npz_path="$1"
    local N="$2"
    local G="$3"
    "$PYTHON_BIN" - "$npz_path" "$N" "$G" "$OMEGA" <<'PYEOF'
import math, os, sys
import numpy as np

npz_path, N, g, omega = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
d_adj = None

if os.path.isfile(npz_path):
    try:
        from rl_vmc.postprocess.casscf_baseline import compute_exact_casscf_1body_density
        xmax = max(14.0, 4.5 / math.sqrt(max(omega, 0.01)) + 0.8 * math.sqrt(max(g, 0.0)))
        ref = compute_exact_casscf_1body_density(npz_path, x_min=-xmax, x_max=xmax, n_grid=800)
        xg = np.asarray(ref["x"], dtype=np.float64)
        pt = np.asarray(ref["p_total"], dtype=np.float64)
        dx = float(xg[1] - xg[0])
        x_rms = math.sqrt(max(float(np.sum((xg ** 2) * pt) * dx), 0.25))
        if N > 1:
            # Para N sitios uniformemente espaciados centrados en 0: x_rms^2 = d_adj^2 * (N^2 - 1) / 12
            d_adj = x_rms * math.sqrt(12.0 / float(N * N - 1))
    except Exception:
        d_adj = None

if d_adj is None or not math.isfinite(d_adj) or d_adj <= 0.2:
    d_adj = 2.35 * ((1.0 + 1.8 * g) ** 0.28) * ((6.0 / max(N, 2)) ** 0.14)

d_adj = round(float(max(0.65, min(d_adj, 6.5))), 4)
base = round(-0.5 * float(N - 1) * d_adj, 4)
print(f"{base} {d_adj}")
PYEOF
}

is_bundle_complete() {
    local h5_file="$1"
    [[ -f "$h5_file" ]] || return 1
    "$PYTHON_BIN" - "$h5_file" <<'PYEOF' >/dev/null 2>&1
import json, sys, h5py, numpy as np
path = sys.argv[1]
with h5py.File(path, "r") as f:
    if "config_json" not in f.attrs or "production" not in f or "checkpoints" not in f:
        sys.exit(1)
    cfg = json.loads(str(f.attrs["config_json"]))
    if not isinstance(cfg, dict) or not cfg:
        sys.exit(1)
    prod = f["production"]
    if "walkers" not in prod or "local_energy" not in prod:
        sys.exit(1)
    if prod["walkers"].shape[0] < 1 or prod["walkers"].shape[1] < 5:
        sys.exit(1)
    el = np.asarray(prod["local_energy"][0], dtype=np.float64)
    if el.size == 0 or not np.all(np.isfinite(el)):
        sys.exit(1)
sys.exit(0)
PYEOF
}

quarantine_if_corrupt_h5() {
    local h5_file="$1"
    if [[ -f "$h5_file" ]] && ! is_bundle_complete "$h5_file"; then
        local bak="${h5_file}.corrupt.$(date +%Y%m%d_%H%M%S).bak"
        log_msg "WARN" "Respaldando bundle HDF5 incompleto/corrupto: $(basename "$(dirname "$h5_file")") -> $(basename "$bak")"
        mv "$h5_file" "$bak"
    fi
}

check_free_disk_mb() {
    local target_dir="$1"
    local avail_kb
    avail_kb=$(df -Pk "$target_dir" 2>/dev/null | awk 'NR==2 {print $4}')
    if [[ -n "${avail_kb:-}" ]]; then
        local avail_mb=$((avail_kb / 1024))
        if [[ "$avail_mb" -lt "$MIN_FREE_DISK_MB" ]]; then
            log_msg "ERROR" "Espacio libre crítico en disco (${avail_mb} MB < ${MIN_FREE_DISK_MB} MB) en ${target_dir}."
            return 1
        fi
    fi
    return 0
}

sync_stray_csvs_to_central_dir() {
    # Garantiza que cualquier CSV auxiliar quede centralizado y clasificado en ${CSV_DIR}
    local search_dir="$1"
    local default_subdir="$2"
    [[ -d "$search_dir" ]] || return 0
    mkdir -p "$default_subdir" "$CSV_TRAIN_DIR" "$CSV_PROD_DIR"
    while IFS= read -r -d '' f_csv; do
        local bname
        bname="$(basename "$f_csv")"
        if [[ "$bname" == training_history_* ]]; then
            mv -f "$f_csv" "${CSV_TRAIN_DIR}/${bname}" 2>/dev/null || true
        elif [[ "$bname" == production_metrics_* ]]; then
            mv -f "$f_csv" "${CSV_PROD_DIR}/${bname}" 2>/dev/null || true
        else
            mv -f "$f_csv" "${default_subdir}/${bname}" 2>/dev/null || true
        fi
    done < <(find "$search_dir" -type f -name "*.csv" -print0 2>/dev/null || true)
}

# ==============================================================================
# 6. ETAPA 0: DESCUBRIMIENTO Y GENERACIÓN SEGURA DE ESTADOS DE REFERENCIA CASSCF
# ==============================================================================
locate_existing_casscf_npz() {
    local n_req="$1"
    local g_req="$2"
    "$PYTHON_BIN" - "$SCRIPT_DIR" "$CASSCF_REF_DIR" "$n_req" "$g_req" "$OMEGA" "$SOFTENING" <<'PYEOF'
import glob, json, os, re, sys
import numpy as np

repo_root, ref_dir = sys.argv[1], sys.argv[2]
n_req, g_req = int(sys.argv[3]), float(sys.argv[4])
w_req, s_req = float(sys.argv[5]), float(sys.argv[6])

# 1. Revisar si ya está instalado en casscf_refs/
sys_tag = f"N{n_req}_w{w_req}_g{g_req}"
installed = os.path.join(ref_dir, sys_tag, "casscf_state.npz")
if os.path.isfile(installed):
    print(installed)
    sys.exit(0)

# 2. Buscar en CASSCF_csv/ (priorizando base grande b=256 y soft=0.1)
best_path = None
best_basis = -1
casscf_dir = os.path.join(repo_root, "CASSCF_csv")
if os.path.isdir(casscf_dir):
    for p in sorted(glob.glob(os.path.join(casscf_dir, "*.npz"))):
        b = os.path.basename(p).lower()
        m_n = re.search(r"n(\d+)_g([0-9]+(?:\.[0-9]+)?)", b)
        if not m_n:
            continue
        nv, gv = int(m_n.group(1)), float(m_n.group(2))
        if nv != n_req or abs(gv - g_req) > 1e-4:
            continue
        m_w = re.search(r"omega([0-9]+(?:\.[0-9]+)?)", b)
        if m_w and abs(float(m_w.group(1)) - w_req) > 1e-4:
            continue
        m_s = re.search(r"soft([0-9]+(?:\.[0-9]+)?)", b)
        if m_s and abs(float(m_s.group(1)) - s_req) > 1e-4:
            continue
        m_b = re.search(r"_b(\d+)", b)
        nb = int(m_b.group(1)) if m_b else 100
        if nb > best_basis:
            best_basis = nb
            best_path = p

if best_path:
    print(best_path)
    sys.exit(0)

# 3. Buscar en carpetas uhf_1d_ho_* del asesor con metadata coincidente
from rl_vmc.postprocess.casscf_baseline import discover_casscf_baselines, find_matching_casscf_baseline
baselines = discover_casscf_baselines(search_root=repo_root)
match = find_matching_casscf_baseline(
    baselines, n_up=n_req // 2, n_dn=n_req - (n_req // 2),
    omega=w_req, g=g_req, softening=s_req,
)
if match and int(match.get("n_basis", 0)) >= 32 and match.get("npz_path"):
    print(match["npz_path"])
    sys.exit(0)
PYEOF
}

prepare_all_casscf_references() {
    log_msg "STAGE-0" "=============================================================================="
    log_msg "STAGE-0" "Verificando referencias exactas CASSCF singlete para los sistemas solicitados..."
    log_msg "STAGE-0" "Carpeta de estados CASSCF : ${CASSCF_REF_DIR}"
    log_msg "STAGE-0" "Resumen CSV CASSCF        : ${CASSCF_SUMMARY_CSV}"
    log_msg "STAGE-0" "=============================================================================="

    for N in $N_LIST; do
        local N_UP=$((N / 2))
        local N_DN=$((N - N_UP))
        for G in $G_LIST; do
            should_run_system_pair "$N" "$G" || continue
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local OUT_SYS_REF="${CASSCF_REF_DIR}/${SYS_TAG}"
            local TARGET_NPZ="${OUT_SYS_REF}/casscf_state.npz"
            local CAS_LOG="${LOG_CASSCF_DIR}/casscf_${SYS_TAG}.log"

            local EXISTING_NPZ
            EXISTING_NPZ="$(locate_existing_casscf_npz "$N" "$G")"

            if [[ "$DRY_RUN" -eq 1 ]]; then
                if [[ -n "$EXISTING_NPZ" ]]; then
                    log_msg "DRY-RUN" "[CASSCF ${SYS_TAG}] Usar existente: ${EXISTING_NPZ}"
                else
                    log_msg "DRY-RUN" "[CASSCF ${SYS_TAG}] Generar con cli/run_casscf_1d.py (b=${CASSCF_FALLBACK_BASIS_N10}, quad=${CASSCF_FALLBACK_QUAD_N10}, soft=${SOFTENING})"
                fi
                continue
            fi

            if [[ -f "$TARGET_NPZ" && "$FORCE_RERUN" -eq 0 ]]; then
                log_msg "SKIP" "[CASSCF ${SYS_TAG}] Referencia ya preparada en ${TARGET_NPZ}"
                continue
            fi

            mkdir -p "$OUT_SYS_REF"
            local t0_cas
            t0_cas="$(date +%s)"

            if [[ -n "$EXISTING_NPZ" ]]; then
                log_msg "CASSCF" "[${SYS_TAG}] Importando referencia exacta convergida: $(basename "$EXISTING_NPZ")"
                env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" -u cli/run_casscf_1d.py \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFTENING" \
                    --advisor_npz_path "$EXISTING_NPZ" \
                    --outdir "$OUT_SYS_REF" \
                    --master_csv "$CASSCF_SUMMARY_CSV" \
                    --csv_collect_dir "$CSV_CASSCF_DIR" \
                    > "$CAS_LOG" 2>&1
                local rc=$?
            else
                local nb=64 quad=400 ncore=0 nact=4
                if [[ "$N" -eq 4 ]]; then
                    nb=64; quad=400; ncore=0; nact=4
                elif [[ "$N" -eq 6 ]]; then
                    nb=64; quad=450; ncore=0; nact=6
                else
                    nb="$CASSCF_FALLBACK_BASIS_N10"
                    quad="$CASSCF_FALLBACK_QUAD_N10"
                    ncore="$CASSCF_FALLBACK_CORE_N10"
                    nact="$CASSCF_FALLBACK_ACT_N10"
                fi
                log_msg "CASSCF" "[${SYS_TAG}] No hallado en CASSCF_csv/. Calculando CASSCF de alta cuadratura en CPU (b=${nb}, quad=${quad}, c=${ncore}, a=${nact}, soft=${SOFTENING})..."
                env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" -u cli/run_casscf_1d.py \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFTENING" \
                    --n_basis "$nb" --quadrature_order "$quad" \
                    --n_core "$ncore" --n_active_orbitals "$nact" \
                    --casscf_max_iter 1200 --casscf_energy_tol 1e-8 --casscf_grad_tol 1e-6 \
                    --outdir "$OUT_SYS_REF" \
                    --master_csv "$CASSCF_SUMMARY_CSV" \
                    --csv_collect_dir "$CSV_CASSCF_DIR" \
                    > "$CAS_LOG" 2>&1
                local rc=$?
            fi

            local dt_cas=$(( $(date +%s) - t0_cas ))
            if [[ "$rc" -ne 0 || ! -f "$TARGET_NPZ" ]]; then
                record_failure "CASSCF" "$SYS_TAG" "casscf_ref" "Falló preparación CASSCF (exit=${rc}). Ver ${CAS_LOG}"
                continue
            fi

            # Mover figuras diagnósticas CASSCF a figures/casscf_diagnostics/ y CSVs sueltos a csv/
            mkdir -p "${FIG_CASSCF_DIR}/${SYS_TAG}"
            for f_img in "$OUT_SYS_REF"/*_"${SYS_TAG}".png "$OUT_SYS_REF"/*_"${SYS_TAG}".pdf; do
                [[ -e "$f_img" ]] && mv -f "$f_img" "${FIG_CASSCF_DIR}/${SYS_TAG}/" 2>/dev/null || true
            done
            rm -f "$OUT_SYS_REF"/densities.png "$OUT_SYS_REF"/ci_weights.png "$OUT_SYS_REF"/natural_occupations.png \
                  "$OUT_SYS_REF"/orbital_4panel.png "$OUT_SYS_REF"/energy_history.png 2>/dev/null || true
            sync_stray_csvs_to_central_dir "$OUT_SYS_REF" "$CSV_CASSCF_DIR"

            log_msg "DONE" "[CASSCF ${SYS_TAG}] Listo en $(format_duration "$dt_cas") | Log: ${CAS_LOG}"
        done
    done
}

# ==============================================================================
# 7. GESTOR DINÁMICO DE COLAS MULTI-GPU / MULTI-JOB CON PROTECCIÓN PARA N=10
# ==============================================================================
reap_finished_sampling_jobs() {
    for mf in "$TRACKER_DIR"/job_*.meta; do
        [[ -e "$mf" ]] || continue
        local pid="" gpu="" n_elec="" sys_tag="" model_key="" slot_weight="" start_epoch="" start_ts="" log_file="" h5_path="" run_dir=""
        # shellcheck disable=SC1090
        source "$mf"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local rc=$?
            local end_epoch
            end_epoch="$(date +%s)"
            local elapsed=$((end_epoch - start_epoch))
            rm -f "$mf"

            # Centralizar cualquier CSV auxiliar escrito en run_dir hacia csv/
            [[ -n "$run_dir" ]] && sync_stray_csvs_to_central_dir "$run_dir" "$CSV_TRAIN_DIR"

            if [[ "$rc" -eq 0 ]] && is_bundle_complete "$h5_path"; then
                TOTAL_COMPLETED_JOBS=$((TOTAL_COMPLETED_JOBS + 1))
                local done_now=$((TOTAL_COMPLETED_JOBS + TOTAL_SKIPPED_JOBS))
                log_msg "DONE" "[${done_now}/${TOTAL_PLANNED_JOBS}] ${sys_tag} :: ${model_key} | GPU=${gpu} (slots=${slot_weight}) | PID=${pid} | Duración=$(format_duration "$elapsed")"
                maybe_trigger_realtime_system_postprocess "$sys_tag"
            else
                record_failure "SAMPLING" "$sys_tag" "$model_key" "Exit=${rc} tras $(format_duration "$elapsed") en GPU=${gpu} -> Log: ${log_file}"
            fi
        fi
    done
}

reap_finished_postprocess_jobs() {
    for pf in "$POST_PIDS_DIR"/post_*.pid; do
        [[ -e "$pf" ]] || continue
        local ppid
        ppid="$(cat "$pf" 2>/dev/null || true)"
        if [[ -z "$ppid" ]] || ! kill -0 "$ppid" 2>/dev/null; then
            rm -f "$pf"
        fi
    done
}

get_used_slots_on_device() {
    local target_dev="$1"
    local used=0
    for mf in "$TRACKER_DIR"/job_*.meta; do
        [[ -e "$mf" ]] || continue
        local pid="" gpu="" slot_weight=1
        # shellcheck disable=SC1090
        source "$mf"
        if [[ "$gpu" == "$target_dev" ]] && kill -0 "$pid" 2>/dev/null; then
            used=$((used + slot_weight))
        fi
    done
    echo "$used"
}

get_total_used_slots() {
    local total=0
    for mf in "$TRACKER_DIR"/job_*.meta; do
        [[ -e "$mf" ]] || continue
        local pid="" slot_weight=1
        # shellcheck disable=SC1090
        source "$mf"
        if kill -0 "$pid" 2>/dev/null; then
            total=$((total + slot_weight))
        fi
    done
    echo "$total"
}

SELECTED_DEVICE=""
wait_for_device_slot() {
    local n_elec="$1"
    local needed_slots=1
    if [[ "$n_elec" -ge 10 ]]; then
        needed_slots="$N10_SLOT_WEIGHT"
    fi
    if [[ "$needed_slots" -gt "$JOBS_PER_GPU" ]]; then
        needed_slots="$JOBS_PER_GPU"
    fi

    SELECTED_DEVICE=""
    while true; do
        reap_finished_sampling_jobs
        reap_finished_postprocess_jobs

        local total_used
        total_used="$(get_total_used_slots)"
        if [[ $((total_used + needed_slots)) -le "$MAX_TOTAL_SLOTS" ]]; then
            local best_dev=""
            local min_used=999999
            for dev in "${AVAILABLE_GPUS[@]}"; do
                local dev_used
                dev_used="$(get_used_slots_on_device "$dev")"
                if [[ $((dev_used + needed_slots)) -le "$JOBS_PER_GPU" && "$dev_used" -lt "$min_used" ]]; then
                    min_used="$dev_used"
                    best_dev="$dev"
                fi
            done
            if [[ -n "$best_dev" ]]; then
                SELECTED_DEVICE="$best_dev"
                return 0
            fi
        fi
        sleep 2
    done
}

wait_all_sampling_jobs() {
    while true; do
        reap_finished_sampling_jobs
        local used
        used="$(get_total_used_slots)"
        if [[ "$used" -eq 0 ]]; then
            break
        fi
        sleep 2
    done
}

wait_all_postprocess_jobs() {
    for pf in "$POST_PIDS_DIR"/post_*.pid; do
        [[ -e "$pf" ]] || continue
        local ppid
        ppid="$(cat "$pf" 2>/dev/null || true)"
        if [[ -n "$ppid" ]] && kill -0 "$ppid" 2>/dev/null; then
            wait "$ppid" 2>/dev/null || true
        fi
        rm -f "$pf"
    done
}

submit_sampling_job() {
    local n_elec="$1"
    local sys_tag="$2"
    local model_key="$3"
    local run_dir="$4"
    local h5_path="$5"
    shift 5
    local cmd_args=("$@")

    local log_file="${LOG_SAMPLING_DIR}/${sys_tag}__${model_key}.log"

    if [[ "$FORCE_RERUN" -eq 0 ]] && is_bundle_complete "$h5_path"; then
        TOTAL_SKIPPED_JOBS=$((TOTAL_SKIPPED_JOBS + 1))
        local done_now=$((TOTAL_COMPLETED_JOBS + TOTAL_SKIPPED_JOBS))
        log_msg "SKIP" "[${done_now}/${TOTAL_PLANNED_JOBS}] ${sys_tag} :: ${model_key} ya completado (${h5_path})"
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        TOTAL_LAUNCHED_JOBS=$((TOTAL_LAUNCHED_JOBS + 1))
        local slot_w=1
        [[ "$n_elec" -ge 10 ]] && slot_w="$N10_SLOT_WEIGHT"
        log_msg "DRY-RUN" "[${TOTAL_LAUNCHED_JOBS}/${TOTAL_PLANNED_JOBS}] (N=${n_elec}, slots=${slot_w}) ${sys_tag} :: ${model_key} -> ${PYTHON_BIN} -u cli/run_1d.py ${cmd_args[*]}"
        return 0
    fi

    quarantine_if_corrupt_h5 "$h5_path"
    check_free_disk_mb "$BASE_OUT_DIR" || {
        record_failure "DISK" "$sys_tag" "$model_key" "Abortado por falta de espacio en disco."
        return 1
    }

    wait_for_device_slot "$n_elec"
    local dev="$SELECTED_DEVICE"
    local slot_weight=1
    local mem_frac="$MEM_FRAC_STD"
    if [[ "$n_elec" -ge 10 ]]; then
        slot_weight="$N10_SLOT_WEIGHT"
        mem_frac="$MEM_FRAC_N10"
    fi

    mkdir -p "$run_dir"
    local start_epoch start_ts
    start_epoch="$(date +%s)"
    start_ts="$(ts)"

    {
        echo "=============================================================================="
        echo " [${start_ts}] INICIO DE JOB: ${sys_tag} :: ${model_key}"
        echo " Dispositivo : ${dev} (slots reservados=${slot_weight}/${JOBS_PER_GPU}, mem_frac=${mem_frac})"
        echo " Comando     : ${PYTHON_BIN} -u cli/run_1d.py ${cmd_args[*]}"
        echo "=============================================================================="
    } > "$log_file"

    if [[ "$USE_CPU_BACKEND" -eq 1 || "$dev" == "cpu" ]]; then
        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            "$PYTHON_BIN" -u cli/run_1d.py "${cmd_args[@]}" >> "$log_file" 2>&1 &
    else
        env CUDA_VISIBLE_DEVICES="$dev" \
            XLA_PYTHON_CLIENT_PREALLOCATE=false \
            XLA_PYTHON_CLIENT_MEM_FRACTION="$mem_frac" \
            TF_FORCE_GPU_ALLOW_GROWTH=true \
            "$PYTHON_BIN" -u cli/run_1d.py "${cmd_args[@]}" >> "$log_file" 2>&1 &
    fi
    local pid=$!
    TOTAL_LAUNCHED_JOBS=$((TOTAL_LAUNCHED_JOBS + 1))

    cat <<EOF > "${TRACKER_DIR}/job_${pid}.meta"
pid="${pid}"
gpu="${dev}"
n_elec="${n_elec}"
sys_tag="${sys_tag}"
model_key="${model_key}"
slot_weight="${slot_weight}"
start_epoch="${start_epoch}"
start_ts="${start_ts}"
log_file="${log_file}"
h5_path="${h5_path}"
run_dir="${run_dir}"
EOF

    local active_slots
    active_slots="$(get_total_used_slots)"
    log_msg "START" "[Lanzado ${TOTAL_LAUNCHED_JOBS} | Slots ${active_slots}/${MAX_TOTAL_SLOTS}] ${sys_tag} :: ${model_key} | GPU=${dev} (w=${slot_weight}) | PID=${pid} | Log=$(basename "$log_file")"
}

# ==============================================================================
# 8. CONSTRUCCIÓN DE LA COLA DE TRABAJOS PARA LOS 21 SISTEMAS
# ==============================================================================
count_models_per_system() {
    local n_sig=0
    for _ in $SIGMA_LIST; do n_sig=$((n_sig + 1)); done
    local k=0
    [[ "$RUN_MALA" -eq 1 ]] && k=$((k + n_sig))
    [[ "$RUN_RWMH" -eq 1 ]] && k=$((k + n_sig))
    [[ "$RUN_RL" -eq 1 ]] && k=$((k + 1))
    [[ "$RUN_RL" -eq 1 && "$WITH_WIDE_RL" -eq 1 ]] && k=$((k + 1))
    echo "$k"
}

MODELS_PER_SYSTEM="$(count_models_per_system)"

system_all_models_complete() {
    local sys_tag="$1"
    local sys_dir="${RUNS_DIR}/${sys_tag}"
    [[ -d "$sys_dir" ]] || return 1
    if [[ "$RUN_RL" -eq 1 ]]; then
        is_bundle_complete "${sys_dir}/RL_MALA_CDLB_opt/run_bundle.h5" || return 1
        if [[ "$WITH_WIDE_RL" -eq 1 ]]; then
            is_bundle_complete "${sys_dir}/RL_MALA_CDLB_wide/run_bundle.h5" || return 1
        fi
    fi
    for sig in $SIGMA_LIST; do
        if [[ "$RUN_MALA" -eq 1 ]]; then
            is_bundle_complete "${sys_dir}/MALA_sig${sig}/run_bundle.h5" || return 1
        fi
        if [[ "$RUN_RWMH" -eq 1 ]]; then
            is_bundle_complete "${sys_dir}/RWMH_sig${sig}/run_bundle.h5" || return 1
        fi
    done
    return 0
}

maybe_trigger_realtime_system_postprocess() {
    local sys_tag="$1"
    [[ "$REALTIME_SYSTEM_POSTPROCESS" -eq 1 ]] || return 0
    local done_marker="${POST_PIDS_DIR}/done_${sys_tag}.flag"
    [[ -f "$done_marker" ]] && return 0
    if system_all_models_complete "$sys_tag"; then
        touch "$done_marker"
        local plog="${LOG_POST_DIR}/postprocess_${sys_tag}.log"
        log_msg "POST-BG" "Todos los modelos de ${sys_tag} finalizaron. Generando CSVs y figuras de ${sys_tag} en CPU..."
        (
            run_system_postprocess "$sys_tag" > "$plog" 2>&1
        ) &
        local ppid=$!
        echo "$ppid" > "${POST_PIDS_DIR}/post_${sys_tag}_${ppid}.pid"
    fi
}

enqueue_all_production_jobs() {
    # Primero contabilizar total de jobs planeados para mostrar progreso exacto [k/N]
    TOTAL_PLANNED_JOBS=0
    for N in $N_LIST; do
        for G in $G_LIST; do
            should_run_system_pair "$N" "$G" || continue
            TOTAL_PLANNED_JOBS=$((TOTAL_PLANNED_JOBS + MODELS_PER_SYSTEM))
        done
    done

    log_msg "STAGE-1" "=============================================================================="
    log_msg "STAGE-1" "Iniciando cola de producción definitiva (${TOTAL_PLANNED_JOBS} corridas totales)"
    log_msg "STAGE-1" "Dispositivos : ${AVAILABLE_GPUS[*]} | Jobs/GPU (N=4,6)=${JOBS_PER_GPU} | Jobs/GPU (N=10)=${N10_JOBS_PER_GPU}"
    log_msg "STAGE-1" "Cadenas Eval : ${N_EVAL_CHAINS} x ${MEASURE_STEPS} pasos (burn-in=${EVAL_BURN_IN}, walkers=${N_WALKERS})"
    log_msg "STAGE-1" "RL-MALA CDLB : burn-in=${RL_BURN_IN}, train_steps=${RL_TRAIN_STEPS}, recompensa=100% CDLB exacta"
    log_msg "STAGE-1" "Sigmas MALA/RWMH: (${SIGMA_LIST})"
    log_msg "STAGE-1" "=============================================================================="

    local sys_idx=0
    for N in $N_LIST; do
        local N_UP=$((N / 2))
        local N_DN=$((N - N_UP))
        for G in $G_LIST; do
            should_run_system_pair "$N" "$G" || continue
            sys_idx=$((sys_idx + 1))

            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local CASSCF_NPZ="${CASSCF_REF_DIR}/${SYS_TAG}/casscf_state.npz"
            if [[ "$DRY_RUN" -eq 0 && ! -f "$CASSCF_NPZ" ]]; then
                record_failure "QUEUE" "$SYS_TAG" "ALL" "No existe ${CASSCF_NPZ}; se omite el sistema."
                continue
            fi

            read -r TAU_RL AMP_RL PEN_RL SIG_RL ELOC_CHUNK JIT_CHUNK REPLAY_CAP MAX_SAVE_B \
                <<< "$(get_optimal_rl_params "$N" "$G")"
            read -r INIT_OFFSET_BASE INIT_OFFSET_STEP \
                <<< "$(get_wigner_centered_offsets "$CASSCF_NPZ" "$N" "$G")"

            local SYS_RUN_DIR="${RUNS_DIR}/${SYS_TAG}"
            local SYS_SEED=$((SEED_BASE + N * 100 + sys_idx))

            local COMMON_SYS_ARGS=(
                --pipeline_stage "prb1_definitive_production"
                --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN"
                --omega "$OMEGA" --g "$G" --softening "$SOFTENING"
                --ansatz_type casscf_npz
                --casscf_npz_path "$CASSCF_NPZ"
                --no-casscf_trainable
                --batch_size "$N_WALKERS"
                --init_mode "$RL_INIT_MODE"
                --init_std "$RL_INIT_STD"
                --eval_init_mode "$EVAL_INIT_MODE"
                --init_offset_base "$INIT_OFFSET_BASE"
                --init_offset_step "$INIT_OFFSET_STEP"
                --n_eval_chains "$N_EVAL_CHAINS"
                --eval_burn_in "$EVAL_BURN_IN"
                --measure_steps "$MEASURE_STEPS"
                --jit_chunk_size "$JIT_CHUNK"
                --eloc_chunk_size "$ELOC_CHUNK"
                --save_walkers_max_batch "$MAX_SAVE_B"
                --save_production_energy_mode "$SAVE_ENERGY_MODE"
                --save_production_walkers_dtype "$SAVE_WALKERS_DTYPE"
                --save_production_stats_dtype "$SAVE_STATS_DTYPE"
                --no-export_legacy_files
                --reference_casscf_npz "$CASSCF_NPZ"
                --master_csv "$MASTER_LIVE_CSV"
                --csv_collect_dir "$CSV_PROD_DIR"
            )

            # ------------------------------------------------------------------
            # 1) LANZAR PRIMERO RL-MALA ÓPTIMO (100% CDLB PURO) — Job más largo
            # ------------------------------------------------------------------
            if [[ "$RUN_RL" -eq 1 ]]; then
                local RL_KEY="RL_MALA_CDLB_opt"
                local RL_DIR="${SYS_RUN_DIR}/${RL_KEY}"
                local RL_H5="${RL_DIR}/run_bundle.h5"

                submit_sampling_job "$N" "$SYS_TAG" "$RL_KEY" "$RL_DIR" "$RL_H5" \
                    --run_name "${SYS_TAG}__${RL_KEY}" \
                    --seed "$SYS_SEED" \
                    --train_mode rl_only \
                    --proposal_type neural_deepset \
                    --proposal_mode residual_mala \
                    --parity_equivariant \
                    --unclip_sigma_action \
                    --scale_clip_min -3.0 \
                    --scale_clip_max "$SIG_RL" \
                    --tau "$TAU_RL" \
                    --actor_amplitude "$AMP_RL" \
                    --sigma_action_scale "$SIG_RL" \
                    --rl_algorithm td3 \
                    --reward_type cdlb \
                    --cdlb_exact_entropy \
                    --confinement_weight 0.0 \
                    --outward_drift_penalty 0.0 \
                    --penalty_factor "$PEN_RL" \
                    --use_reward_centering \
                    --reward_centering_lr "$REWARD_CENTERING_LR" \
                    --lr_actor "$LR_ACTOR" \
                    --lr_critic "$LR_CRITIC" \
                    --use_lr_scheduler \
                    --lr_scheduler_type cosine \
                    --lr_warmup_steps "$LR_WARMUP_STEPS" \
                    --lr_min_ratio "$LR_MIN_RATIO" \
                    --rl_replay_capacity "$REPLAY_CAP" \
                    --rl_batch_size "$RL_BATCH_SIZE" \
                    --policy_delay "$POLICY_DELAY" \
                    --tau_polyak "$TAU_POLYAK" \
                    --target_noise "$TARGET_NOISE" \
                    --noise_clip "$NOISE_CLIP" \
                    --exp_noise_init "$EXP_NOISE_INIT" \
                    --exp_noise_final "$EXP_NOISE_FINAL" \
                    --n_ter "$RL_BURN_IN" \
                    --train_steps "$RL_TRAIN_STEPS" \
                    --output_h5 "$RL_H5" \
                    "${COMMON_SYS_ARGS[@]}"

                if [[ "$WITH_WIDE_RL" -eq 1 ]]; then
                    local WIDE_SIG="1.8"
                    local WIDE_AMP="0.80"
                    if [[ "$N" -ge 10 ]]; then
                        WIDE_SIG="1.20"
                        WIDE_AMP="0.45"
                    fi
                    local RLW_KEY="RL_MALA_CDLB_wide"
                    local RLW_DIR="${SYS_RUN_DIR}/${RLW_KEY}"
                    local RLW_H5="${RLW_DIR}/run_bundle.h5"

                    submit_sampling_job "$N" "$SYS_TAG" "$RLW_KEY" "$RLW_DIR" "$RLW_H5" \
                        --run_name "${SYS_TAG}__${RLW_KEY}" \
                        --seed $((SYS_SEED + 5000)) \
                        --train_mode rl_only \
                        --proposal_type neural_deepset \
                        --proposal_mode residual_mala \
                        --parity_equivariant \
                        --unclip_sigma_action \
                        --scale_clip_min -3.0 \
                        --scale_clip_max 2.5 \
                        --tau 0.10 \
                        --actor_amplitude "$WIDE_AMP" \
                        --sigma_action_scale "$WIDE_SIG" \
                        --rl_algorithm td3 \
                        --reward_type cdlb \
                        --cdlb_exact_entropy \
                        --confinement_weight 0.0 \
                        --outward_drift_penalty 0.0 \
                        --penalty_factor 0.25 \
                        --use_reward_centering \
                        --reward_centering_lr "$REWARD_CENTERING_LR" \
                        --lr_actor "$LR_ACTOR" \
                        --lr_critic "$LR_CRITIC" \
                        --use_lr_scheduler \
                        --lr_scheduler_type cosine \
                        --lr_warmup_steps "$LR_WARMUP_STEPS" \
                        --lr_min_ratio "$LR_MIN_RATIO" \
                        --rl_replay_capacity "$REPLAY_CAP" \
                        --rl_batch_size "$RL_BATCH_SIZE" \
                        --policy_delay "$POLICY_DELAY" \
                        --tau_polyak "$TAU_POLYAK" \
                        --target_noise "$TARGET_NOISE" \
                        --noise_clip "$NOISE_CLIP" \
                        --exp_noise_init "$EXP_NOISE_INIT" \
                        --exp_noise_final "$EXP_NOISE_FINAL" \
                        --n_ter "$RL_BURN_IN" \
                        --train_steps "$RL_TRAIN_STEPS" \
                        --output_h5 "$RLW_H5" \
                        "${COMMON_SYS_ARGS[@]}"
                fi
            fi

            # ------------------------------------------------------------------
            # 2) BARRIDO EN SIGMA DE MALA (tau = sigma^2 / 2) Y RWMH (rw_sigma = sigma)
            # ------------------------------------------------------------------
            local s_idx=0
            for SIG in $SIGMA_LIST; do
                s_idx=$((s_idx + 1))

                if [[ "$RUN_MALA" -eq 1 ]]; then
                    local TAU_MALA
                    TAU_MALA="$(sigma_to_mala_tau "$SIG")"
                    local MALA_KEY="MALA_sig${SIG}"
                    local MALA_DIR="${SYS_RUN_DIR}/${MALA_KEY}"
                    local MALA_H5="${MALA_DIR}/run_bundle.h5"

                    submit_sampling_job "$N" "$SYS_TAG" "$MALA_KEY" "$MALA_DIR" "$MALA_H5" \
                        --run_name "${SYS_TAG}__${MALA_KEY}" \
                        --seed $((SYS_SEED + s_idx)) \
                        --train_mode eval_only \
                        --proposal_type mala \
                        --tau "$TAU_MALA" \
                        --output_h5 "$MALA_H5" \
                        "${COMMON_SYS_ARGS[@]}"
                fi

                if [[ "$RUN_RWMH" -eq 1 ]]; then
                    local RWMH_KEY="RWMH_sig${SIG}"
                    local RWMH_DIR="${SYS_RUN_DIR}/${RWMH_KEY}"
                    local RWMH_H5="${RWMH_DIR}/run_bundle.h5"

                    submit_sampling_job "$N" "$SYS_TAG" "$RWMH_KEY" "$RWMH_DIR" "$RWMH_H5" \
                        --run_name "${SYS_TAG}__${RWMH_KEY}" \
                        --seed $((SYS_SEED + 100 + s_idx)) \
                        --train_mode eval_only \
                        --proposal_type metropolis \
                        --rw_sigma "$SIG" \
                        --output_h5 "$RWMH_H5" \
                        "${COMMON_SYS_ARGS[@]}"
                fi
            done
        done
    done
}

# ==============================================================================
# 9. MOTOR DE POSTPROCESAMIENTO CIENTÍFICO Y FIGURAS DE ARTÍCULO (EN CPU)
# ==============================================================================
run_python_postprocessor() {
    local mode="$1"      # "system" o "global"
    local sys_arg="${2:-ALL}"

    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        "$PYTHON_BIN" -u - \
        "$mode" "$sys_arg" "$BASE_OUT_DIR" "$FIGURE_FORMATS" "$FIGURE_DPI" "$TRACE_STEPS_PLOT" <<'PYEOF'
import gc
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "True")

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_vmc.config import Experiment1DConfig, PostProcessConfig
from rl_vmc.core.io_bundle import load_run_bundle_h5
from rl_vmc.postprocess.diagnostics import compute_acf_1d, compute_iat_sokal, summarize_bundle_diagnostics
from rl_vmc.postprocess.casscf_baseline import compute_exact_casscf_1body_density
from scripts.analyze_flips_vs_g import compute_flip_statistics, load_bundle_walkers_and_meta

MODE = sys.argv[1]
TARGET_SYS = sys.argv[2]
BASE_OUT = Path(sys.argv[3])
FORMATS = [f.strip().lower() for f in sys.argv[4].split(",") if f.strip()]
DPI = int(sys.argv[5])
TRACE_STEPS = int(sys.argv[6])

CASSCF_DIR = BASE_OUT / "casscf_refs"
RUNS_DIR = BASE_OUT / "runs"
CSV_DIR = BASE_OUT / "csv"
CSV_SYS_DIR = CSV_DIR / "per_system"
FIG_SYS_DIR = BASE_OUT / "figures" / "per_system"
FIG_GLOBAL_DIR = BASE_OUT / "figures" / "global_article"

PAL = {
    "mala": "#C84B31",
    "mala_alt": "#E07A5F",
    "rwmh": "#6C5B7B",
    "rwmh_alt": "#9E8FB2",
    "rl": "#0F62FE",
    "rl_alt": "#009E73",
    "rl_wide": "#117733",
    "exact": "#161616",
    "grid": "#D9DDE5",
}

plt.rcParams.update({
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 9.8,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "axes.linewidth": 0.85,
    "savefig.bbox": "tight",
})


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=DPI)
    plt.close(fig)


def kl_div_1d(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    eps = 1e-10
    ps = np.maximum(np.asarray(p, dtype=np.float64), eps)
    qs = np.maximum(np.asarray(q, dtype=np.float64), eps)
    ps = ps / (np.sum(ps) * dx)
    qs = qs / (np.sum(qs) * dx)
    return float(np.sum(ps * np.log(ps / qs)) * dx)


def parse_model_meta(model_key: str, cfg: Experiment1DConfig) -> Tuple[str, float]:
    m_m = re.match(r"^MALA_sig([0-9.eE+-]+)$", model_key)
    if m_m:
        return "MALA", float(m_m.group(1))
    m_r = re.match(r"^RWMH_sig([0-9.eE+-]+)$", model_key)
    if m_r:
        return "RWMH", float(m_r.group(1))
    if model_key.startswith("RL_MALA"):
        fam = "RL-MALA" if model_key == "RL_MALA_CDLB_opt" else model_key
        return fam, float("nan")
    ptype = cfg.proposal.proposal_type
    if ptype == "mala":
        return "MALA", math.sqrt(2.0 * float(cfg.proposal.tau))
    if ptype == "metropolis":
        return "RWMH", float(cfg.proposal.rw_sigma)
    return "RL-MALA", float("nan")


def select_representative_walker(w_chain: np.ndarray) -> int:
    T, B, N = w_chain.shape
    max_abs = np.max(np.abs(w_chain), axis=(0, 2))
    med_max = float(np.median(max_abs))
    healthy = np.where(max_abs <= max(med_max * 1.6, 25.0))[0]
    if len(healthy) == 0:
        return 0
    flips = np.zeros(len(healthy), dtype=np.int64)
    for k, b in enumerate(healthy):
        wb = w_chain[:, b, :]
        cnt = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = wb[:, i] - wb[:, j]
                cnt += int(np.sum(d[:-1] * d[1:] < 0.0))
        flips[k] = cnt
    med_f = float(np.median(flips))
    return int(healthy[int(np.argmin(np.abs(flips - med_f)))])


def evaluate_single_bundle(
    h5_path: Path,
    sys_tag: str,
    model_key: str,
    casscf_ref: Optional[Dict[str, Any]],
    npz_path: Optional[Path] = None,
) -> Dict[str, Any]:
    bundle = load_run_bundle_h5(str(h5_path))
    cfg = Experiment1DConfig.from_dict(bundle["config"])
    pp_cfg = PostProcessConfig()
    diag = summarize_bundle_diagnostics(bundle, pp_cfg)
    dsum = diag.get("summary", {})

    binfo = load_bundle_walkers_and_meta(str(h5_path))
    w = np.asarray(binfo["walkers"], dtype=np.float64)  # (C, T, B, N)
    C, T, B, N = w.shape
    n_up = int(cfg.system.n_up)

    # Estadísticas de intercambios con histéresis (Wigner opp vs Pauli same) y fracción balística f1
    fstats = compute_flip_statistics(
        w,
        n_up=n_up,
        omega=float(cfg.system.omega),
        g_val=float(cfg.system.interaction_strength),
        tau=float(cfg.proposal.tau),
        actor_amp=float(cfg.proposal.actor_amplitude),
        sigma_scale=float(cfg.proposal.sigma_action_scale),
        actor_params=None,
        is_neural=False,
        npz_path=str(npz_path) if (npz_path is not None and npz_path.is_file()) else None,
    )
    f_tot_1k = float(fstats.get("hyst_flips_total_mean", 0.0))
    f_opp_1k = float(fstats.get("hyst_flips_opp_mean", 0.0))
    f_sam_1k = float(fstats.get("hyst_flips_same_mean", 0.0))
    f1_frac = float(fstats.get("one_step_fraction", 0.0))

    # Divergencias KL de espín y densidad total frente al estado exacto CASSCF (K=400 bins)
    kl_spin, kl_up, kl_dn, kl_tot, e_cas = (float("nan"),) * 5
    if casscf_ref is not None:
        e_cas = float(casscf_ref["energy"])
        xg = np.asarray(casscf_ref["x"], dtype=np.float64)
        dx = float(xg[1] - xg[0])
        edges = np.concatenate([[xg[0] - 0.5 * dx], 0.5 * (xg[:-1] + xg[1:]), [xg[-1] + 0.5 * dx]])
        h_up, _ = np.histogram(w[..., :n_up].reshape(-1), bins=edges, density=True)
        h_dn, _ = np.histogram(w[..., n_up:].reshape(-1), bins=edges, density=True)
        h_tt, _ = np.histogram(w.reshape(-1), bins=edges, density=True)
        kl_up = kl_div_1d(h_up, casscf_ref["p_up"], dx)
        kl_dn = kl_div_1d(h_dn, casscf_ref["p_down"], dx)
        kl_spin = 0.5 * (kl_up + kl_dn)
        kl_tot = kl_div_1d(h_tt, casscf_ref["p_total"], dx)

    family, sigma_nom = parse_model_meta(model_key, cfg)
    prod_stats = bundle.get("production", {}).get("sampler_stats", {})
    if "actor_sigma" in prod_stats and np.asarray(prod_stats["actor_sigma"]).size > 0:
        sig_eff = float(np.mean(prod_stats["actor_sigma"]))
    else:
        sig_eff = sigma_nom if math.isfinite(sigma_nom) else math.sqrt(2.0 * float(cfg.proposal.tau))

    e_mean = float(dsum.get("E_mean", float("nan")))
    row = {
        "system_tag": sys_tag,
        "model_key": model_key,
        "family": family,
        "N": int(cfg.system.n_elec),
        "omega": float(cfg.system.omega),
        "g": float(cfg.system.interaction_strength),
        "sigma_nominal": sigma_nom,
        "sigma_eff": sig_eff,
        "tau": float(cfg.proposal.tau),
        "actor_amplitude": float(cfg.proposal.actor_amplitude) if family.startswith("RL") else 0.0,
        "sigma_action_scale": float(cfg.proposal.sigma_action_scale) if family.startswith("RL") else 0.0,
        "penalty_factor": float(cfg.rl.penalty_factor) if family.startswith("RL") else 0.0,
        "E_CASSCF": e_cas,
        "E_mean": e_mean,
        "SE_E": float(dsum.get("SE_E", float("nan"))),
        "Delta_E_mHa": (e_mean - e_cas) * 1e3 if (math.isfinite(e_mean) and math.isfinite(e_cas)) else float("nan"),
        "acceptance": float(dsum.get("acceptance", float("nan"))),
        "ESJD": float(dsum.get("ESJD", float("nan"))),
        "tau_int_E": float(dsum.get("tau_int_E", float("nan"))),
        "tau_int_pos": float(dsum.get("tau_int_pos", float("nan"))),
        "ESS": float(dsum.get("ESS", float("nan"))),
        "ESS_per_sec": float(dsum.get("ESS_per_sec", float("nan"))),
        "wall_time_sec": float(dsum.get("wall_time_sec", float("nan"))),
        "hyst_flips_total_1k": f_tot_1k,
        "hyst_flips_opp_1k": f_opp_1k,
        "hyst_flips_same_1k": f_sam_1k,
        "f1_ballistic_frac": f1_frac,
        "D_KL_spin": kl_spin,
        "D_KL_up": kl_up,
        "D_KL_down": kl_dn,
        "D_KL_total": kl_tot,
        "n_chains": int(C),
        "measure_steps": int(T),
        "saved_walkers_per_chain": int(B),
        "h5_path": str(h5_path),
    }
    del bundle, w
    gc.collect()
    return row


def select_best_classical_row(df_fam: pd.DataFrame) -> Optional[pd.Series]:
    if df_fam.empty:
        return None
    df_c = df_fam.copy()
    # Filtrar corridas físicamente válidas (aceptación > 0.5% y |Delta E| razonable si convergió)
    valid = df_c[df_c["acceptance"] > 0.005]
    if valid.empty:
        valid = df_c
    # Si alguna alcanza D_KL_spin < 2e-3, elegir la de mayor ESJD / menor tau_int_pos entre las ergódicas;
    # de lo contrario, elegir la de menor D_KL_spin (y mayor ESJD en empate).
    erg = valid[valid["D_KL_spin"] <= 2.0e-3]
    if not erg.empty:
        return erg.sort_values(by=["ESJD", "hyst_flips_total_1k"], ascending=[False, False]).iloc[0]
    return valid.sort_values(by=["D_KL_spin", "ESJD"], ascending=[True, False]).iloc[0]


def make_system_figures_and_csvs(sys_tag: str) -> pd.DataFrame:
    sys_run_dir = RUNS_DIR / sys_tag
    if not sys_run_dir.is_dir():
        return pd.DataFrame()

    npz_path = CASSCF_DIR / sys_tag / "casscf_state.npz"
    m_tag = re.match(r"^N(\d+)_w([0-9.]+)_g([0-9.]+)$", sys_tag)
    n_elec = int(m_tag.group(1)) if m_tag else 4
    omega = float(m_tag.group(2)) if m_tag else 0.1
    g_val = float(m_tag.group(3)) if m_tag else 1.0
    xmax = max(15.0, math.ceil(4.6 / math.sqrt(max(omega, 0.01)) + 0.9 * math.sqrt(max(g_val, 0.0))))

    casscf_ref = (
        compute_exact_casscf_1body_density(npz_path, x_min=-xmax, x_max=xmax, n_grid=400)
        if npz_path.is_file()
        else None
    )

    rows: List[Dict[str, Any]] = []
    for sub in sorted(sys_run_dir.iterdir()):
        h5p = sub / "run_bundle.h5"
        if sub.is_dir() and h5p.is_file():
            try:
                rows.append(evaluate_single_bundle(h5p, sys_tag, sub.name, casscf_ref, npz_path=npz_path))
            except Exception as exc:
                print(f"  [WARN] No se pudo evaluar {h5p}: {exc}")

    if not rows:
        return pd.DataFrame()

    df_sys = pd.DataFrame(rows)
    CSV_SYS_DIR.mkdir(parents=True, exist_ok=True)
    out_fig_dir = FIG_SYS_DIR / sys_tag
    out_fig_dir.mkdir(parents=True, exist_ok=True)

    df_mala = df_sys[df_sys["family"] == "MALA"].sort_values("sigma_nominal")
    df_rwmh = df_sys[df_sys["family"] == "RWMH"].sort_values("sigma_nominal")
    df_rl = df_sys[df_sys["model_key"] == "RL_MALA_CDLB_opt"]
    rl_row = df_rl.iloc[0] if not df_rl.empty else None
    best_mala = select_best_classical_row(df_mala)
    best_rwmh = select_best_classical_row(df_rwmh)

    # -------------------------------------------------------------------------
    # FIG 1 POR SISTEMA: Barrido completo en sigma (MALA vs RWMH) + RL-MALA óptimo
    # -------------------------------------------------------------------------
    if not df_mala.empty or not df_rwmh.empty:
        fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.4))
        panels = [
            (axes[0, 0], "acceptance", r"Acceptance $\alpha$", False, None),
            (axes[0, 1], "ESJD", r"ESJD ($a_0^2 / \mathrm{sweep}$)", False, None),
            (axes[0, 2], "hyst_flips_total_1k", r"Hysteresis Flips / $10^3$ sweeps", True, None),
            (axes[1, 0], "D_KL_spin", r"$D_{\mathrm{KL}}^{\mathrm{spin}}$ vs Exact CASSCF", True, 1e-3),
            (axes[1, 1], "tau_int_E", r"$\tau_{\mathrm{int}}(E)$ (sweeps)", True, None),
            (axes[1, 2], "tau_int_pos", r"$\tau_{\mathrm{int}}(x)$ (sweeps)", True, None),
        ]
        for ax, col, ylabel, is_log_y, hline in panels:
            if not df_mala.empty:
                ym = df_mala[col].to_numpy(dtype=np.float64)
                xm = df_mala["sigma_nominal"].to_numpy(dtype=np.float64)
                msk = np.isfinite(ym) & ((ym > 0) if is_log_y else True)
                if np.any(msk):
                    ax.plot(xm[msk], ym[msk], color=PAL["mala"], ls="--", marker="o", ms=4.5, mfc="white", mew=1.1, lw=1.5, label="MALA")
            if not df_rwmh.empty:
                yr = df_rwmh[col].to_numpy(dtype=np.float64)
                xr = df_rwmh["sigma_nominal"].to_numpy(dtype=np.float64)
                msk = np.isfinite(yr) & ((yr > 0) if is_log_y else True)
                if np.any(msk):
                    ax.plot(xr[msk], yr[msk], color=PAL["rwmh"], ls="-.", marker="D", ms=4.0, mfc="white", mew=1.0, lw=1.4, label="RWMH")
            if rl_row is not None and math.isfinite(float(rl_row[col])):
                v_rl = float(rl_row[col])
                if (not is_log_y) or v_rl > 0:
                    ax.axhline(v_rl, color=PAL["rl"], ls="-", lw=1.6, alpha=0.82, label="RL-MALA (CDLB opt.)")
                    ax.plot([float(rl_row["sigma_eff"])], [v_rl], marker="*", ms=9.5, color=PAL["rl"], mec="#082B6B", mew=0.7, zorder=10)
            if hline is not None:
                ax.axhline(hline, color="#6E7684", ls=":", lw=0.95, label="Noise floor $10^{-3}$")
            ax.set_xscale("log")
            if is_log_y:
                ax.set_yscale("log")
            ax.set_xlabel(r"Proposal width $\sigma$ ($a_0$)")
            ax.set_ylabel(ylabel)
            ax.grid(True, which="major", alpha=0.32, color=PAL["grid"])
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
        fig.suptitle(rf"Classical $\sigma$-Sweep vs Optimal Pure-CDLB RL-MALA — $N={n_elec},\; \omega={omega},\; g={g_val}$", y=1.06, fontsize=11)
        fig.tight_layout()
        save_fig(fig, out_fig_dir, f"fig1_sigma_sweep_{sys_tag}")

    # -------------------------------------------------------------------------
    # Selección de los 3 campeones por familia para Trazas + Marginales + Convergencia
    # -------------------------------------------------------------------------
    selected_models: List[Tuple[str, str, str, Path]] = []
    if best_mala is not None:
        selected_models.append((
            f"Best MALA (sig={best_mala['sigma_nominal']:g})",
            "MALA_best",
            PAL["mala"],
            Path(best_mala["h5_path"]),
        ))
    if best_rwmh is not None:
        selected_models.append((
            f"Best RWMH (sig={best_rwmh['sigma_nominal']:g})",
            "RWMH_best",
            PAL["rwmh"],
            Path(best_rwmh["h5_path"]),
        ))
    if rl_row is not None:
        selected_models.append((
            f"RL-MALA CDLB (sig_eff={rl_row['sigma_eff']:.2f})",
            "RL_MALA_CDLB_opt",
            PAL["rl"],
            Path(rl_row["h5_path"]),
        ))

    # -------------------------------------------------------------------------
    # FIG 2 POR SISTEMA: Trazas de 1 caminante + Densidades Marginales por Espín
    # FIG 3 POR SISTEMA: Convergencia acumulada D_KL^spin(t), |Delta E(t)|, tau_int(t)
    # -------------------------------------------------------------------------
    if selected_models and casscf_ref is not None:
        xg = np.asarray(casscf_ref["x"], dtype=np.float64)
        dx = float(xg[1] - xg[0])
        edges = np.concatenate([[xg[0] - 0.5 * dx], 0.5 * (xg[:-1] + xg[1:]), [xg[-1] + 0.5 * dx]])
        ex_up = np.asarray(casscf_ref["p_up"], dtype=np.float64)
        e_cas = float(casscf_ref["energy"])

        n_rows = len(selected_models)
        fig_tr = plt.figure(figsize=(8.6, 2.45 * n_rows))
        gs_tr = GridSpec(n_rows, 2, figure=fig_tr, width_ratios=[4.1, 1.2], wspace=0.06, hspace=0.18)

        fig_cv, axes_cv = plt.subplots(3, 1, figsize=(6.2, 7.6), sharex=True)

        conv_records: List[Dict[str, Any]] = []
        dens_export: Dict[str, Any] = {"x_grid": xg, "exact_p_up": ex_up, "exact_p_down": np.asarray(casscf_ref["p_down"])}

        for r_idx, (disp_lbl, short_tag, col_main, h5p) in enumerate(selected_models):
            bundle = load_run_bundle_h5(str(h5p))
            cfg = Experiment1DConfig.from_dict(bundle["config"])
            n_up = int(cfg.system.n_up)
            w = np.asarray(bundle["production"]["walkers"], dtype=np.float64)
            if w.ndim == 5:
                w = w[..., 0]
            eloc = np.asarray(bundle["production"]["local_energy"], dtype=np.float64)
            C, T, B, N = w.shape

            # 2A. Traza representativa y marginales
            ax_t = fig_tr.add_subplot(gs_tr[r_idx, 0])
            ax_h = fig_tr.add_subplot(gs_tr[r_idx, 1], sharey=ax_t)
            b_rep = select_representative_walker(w[0])
            t_lim = min(T, TRACE_STEPS)
            steps_tr = np.arange(t_lim)
            for e_i in range(N):
                c_e = col_main if e_i < n_up else "#E09F3E"
                ax_t.plot(steps_tr, w[0, :t_lim, b_rep, e_i], color=c_e, lw=0.68, alpha=0.86)
            ax_t.set_xlim(0, t_lim)
            ax_t.set_ylim(-xmax * 0.92, xmax * 0.92)
            ax_t.set_ylabel(r"$x_i$ ($a_0$)")
            ax_t.text(0.015, 0.88, disp_lbl, transform=ax_t.transAxes, fontsize=8.4, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82))
            ax_t.grid(True, alpha=0.28, color=PAL["grid"])
            if r_idx < n_rows - 1:
                plt.setp(ax_t.get_xticklabels(), visible=False)
            else:
                ax_t.set_xlabel("Monte Carlo sweep $t$")

            p_u, _ = np.histogram(w[..., :n_up].reshape(-1), bins=edges, density=True)
            p_d, _ = np.histogram(w[..., n_up:].reshape(-1), bins=edges, density=True)
            dens_export[f"{short_tag}_p_up"] = p_u
            dens_export[f"{short_tag}_p_down"] = p_d

            ax_h.fill_betweenx(xg, 0.0, p_u, color=col_main, alpha=0.24)
            ax_h.plot(p_u, xg, color=col_main, lw=1.6, label=r"$\hat{p}_\uparrow$")
            ax_h.plot(p_d, xg, color="#E09F3E", lw=1.25, ls="--", label=r"$\hat{p}_\downarrow$")
            ax_h.plot(ex_up, xg, color=PAL["exact"], lw=1.25, ls=":", label="Exact")
            ax_h.set_ylim(-xmax * 0.92, xmax * 0.92)
            plt.setp(ax_h.get_yticklabels(), visible=False)
            ax_h.grid(True, alpha=0.28, color=PAL["grid"])
            if r_idx < n_rows - 1:
                plt.setp(ax_h.get_xticklabels(), visible=False)
            else:
                ax_h.set_xlabel(r"$p_\sigma(x)$ ($a_0^{-1}$)")

            # 2B. Series acumuladas de convergencia vs t
            t_grid = np.unique(np.geomspace(max(50, T // 200), T, num=18).astype(int))
            kl_s_list, de_list, tx_list = [], [], []
            for t_end in t_grid:
                kls_c = []
                for c_i in range(C):
                    hu, _ = np.histogram(w[c_i, :t_end, :, :n_up].reshape(-1), bins=edges, density=True)
                    hd, _ = np.histogram(w[c_i, :t_end, :, n_up:].reshape(-1), bins=edges, density=True)
                    kls_c.append(0.5 * (kl_div_1d(hu, ex_up, dx) + kl_div_1d(hd, casscf_ref["p_down"], dx)))
                kl_v = float(np.mean(kls_c))
                e_cum = float(np.mean(eloc[:, :t_end]))
                de_v = max(abs(e_cum - e_cas), 1e-6)
                acf_x = compute_acf_1d(w[0, :t_end, :, :], max_lag=min(5000, max(10, t_end // 2)))
                tx_v = float(compute_iat_sokal(acf_x, c_sokal=5.0))
                kl_s_list.append(kl_v)
                de_list.append(de_v)
                tx_list.append(tx_v)
                conv_records.append({
                    "system_tag": sys_tag,
                    "model_tag": short_tag,
                    "display_label": disp_lbl,
                    "step": int(t_end),
                    "D_KL_spin_cum": kl_v,
                    "abs_Delta_E_Ha": de_v,
                    "tau_int_pos_cum": tx_v,
                })

            ls_cv = "-" if "RL" in short_tag else ("--" if "MALA" in short_tag else "-.")
            axes_cv[0].plot(t_grid, kl_s_list, color=col_main, ls=ls_cv, marker="o", ms=3.5, lw=1.55, label=disp_lbl)
            axes_cv[1].plot(t_grid, de_list, color=col_main, ls=ls_cv, marker="s", ms=3.3, lw=1.50, label=disp_lbl)
            axes_cv[2].plot(t_grid, tx_list, color=col_main, ls=ls_cv, marker="^", ms=3.5, lw=1.50, label=disp_lbl)

            del bundle, w, eloc
            gc.collect()

        save_fig(fig_tr, out_fig_dir, f"fig2_traces_and_marginals_{sys_tag}")
        pd.DataFrame(dens_export).to_csv(CSV_SYS_DIR / f"marginals_{sys_tag}.csv", index=False)

        axes_cv[0].axhline(1e-3, color="#6E7684", ls=":", lw=0.95)
        axes_cv[0].set_yscale("log")
        axes_cv[0].set_ylabel(r"$D_{\mathrm{KL}}^{\mathrm{spin}}(t)$")
        axes_cv[0].legend(loc="best", frameon=True)
        axes_cv[0].grid(True, which="major", alpha=0.30, color=PAL["grid"])
        axes_cv[0].set_title(rf"Long-Time Convergence — $N={n_elec},\; \omega={omega},\; g={g_val}$")

        axes_cv[1].set_yscale("log")
        axes_cv[1].set_ylabel(r"$|\langle E_L \rangle_t - E_{\mathrm{CASSCF}}|$ ($E_h$)")
        axes_cv[1].grid(True, which="major", alpha=0.30, color=PAL["grid"])

        axes_cv[2].set_xscale("log")
        axes_cv[2].set_ylabel(r"Cumulative $\tau_{\mathrm{int}}(x)$ (sweeps)")
        axes_cv[2].set_xlabel("Production sweeps $t$ per chain")
        axes_cv[2].grid(True, which="major", alpha=0.30, color=PAL["grid"])

        fig_cv.tight_layout()
        save_fig(fig_cv, out_fig_dir, f"fig3_convergence_kl_and_tau_{sys_tag}")
        pd.DataFrame(conv_records).to_csv(CSV_SYS_DIR / f"convergence_series_{sys_tag}.csv", index=False)

    # -------------------------------------------------------------------------
    # FIG 4 & FIG 5 POR SISTEMA: Sonda espacial mu_theta(x) y diagnóstico TD3 CDLB
    # -------------------------------------------------------------------------
    mu_sgn_weighted = float("nan")
    if rl_row is not None:
        rl_h5 = Path(rl_row["h5_path"])
        try:
            import argparse
            from scripts.probe_actor_spatial_response import (
                plot_spatial_probe_figure,
                run_spatial_actor_probe,
                save_probe_csv,
            )
            rl_bundle = load_run_bundle_h5(str(rl_h5))
            probe_res = run_spatial_actor_probe(
                rl_bundle,
                str(rl_h5),
                probe_spin="down",
                n_grid=501,
                casscf_npz_override=str(npz_path) if npz_path.is_file() else None,
            )
            xg_p = np.asarray(probe_res["x_grid"], dtype=np.float64)
            mu_p = np.asarray(probe_res["mu_actor"], dtype=np.float64)
            wt_p = np.asarray(probe_res["cond_prob"], dtype=np.float64)
            wt_sum = float(np.sum(wt_p))
            if wt_sum > 1e-12:
                mu_sgn_weighted = float(np.sum(wt_p * mu_p * np.sign(xg_p)) / wt_sum)
            fmt_probe = "both" if ("pdf" in FORMATS and "png" in FORMATS) else FORMATS[0]
            p_args = argparse.Namespace(
                font_scale=1.0,
                no_titles=False,
                no_legends=False,
                figure_format=fmt_probe,
                dpi=DPI,
            )
            plot_spatial_probe_figure(
                probe_res,
                out_dir=str(out_fig_dir),
                stem=f"fig4_actor_spatial_probe_{sys_tag}",
                args=p_args,
            )
            save_probe_csv(
                probe_res,
                str(CSV_SYS_DIR / f"actor_spatial_probe_{sys_tag}.csv"),
            )
            del rl_bundle
            gc.collect()
        except Exception as exc_pr:
            print(f"  [WARN] Sonda espacial omitida en {sys_tag}: {exc_pr}")

        # Curvas de entrenamiento RL-TD3 (recompensa CDLB pura)
        try:
            with h5py.File(str(rl_h5), "r") as f_h5:
                if "training" in f_h5 and "Step" in f_h5["training"]:
                    tr_grp = f_h5["training"]
                    tr_df = pd.DataFrame({k: np.asarray(tr_grp[k]) for k in tr_grp.keys()})
                    tr_df = tr_df.sort_values("Step")
                    fig_t, ax_t = plt.subplots(2, 2, figsize=(9.2, 5.6), sharex=True)
                    s_arr = tr_df["Step"].to_numpy()
                    ax_t[0, 0].plot(s_arr, tr_df["RL_Reward"], color=PAL["rl"], lw=1.2, label="Total CDLB")
                    ax_t[0, 0].set_ylabel("CDLB Reward")
                    ax_t[0, 0].grid(True, alpha=0.3, color=PAL["grid"])

                    ax_t[0, 1].plot(s_arr, tr_df["Acceptance"], color=PAL["rl_alt"], lw=1.2)
                    ax_t[0, 1].set_ylabel(r"Train Acceptance $\alpha$")
                    ax_t[0, 1].grid(True, alpha=0.3, color=PAL["grid"])

                    ax_t[1, 0].plot(s_arr, tr_df["ESJD"], color=PAL["rl"], lw=1.2)
                    ax_t[1, 0].set_ylabel(r"Train ESJD ($a_0^2$)")
                    ax_t[1, 0].set_xlabel("TD3 Training Sweep")
                    ax_t[1, 0].grid(True, alpha=0.3, color=PAL["grid"])

                    ax_t[1, 1].plot(s_arr, tr_df["Critic_Loss"], color=PAL["mala"], lw=1.1, label="Critic Loss")
                    ax_t[1, 1].set_yscale("log")
                    ax_t[1, 1].set_ylabel("TD3 Critic Loss")
                    ax_t[1, 1].set_xlabel("TD3 Training Sweep")
                    ax_t[1, 1].grid(True, alpha=0.3, color=PAL["grid"])

                    fig_t.suptitle(f"Pure-CDLB TD3 Training Convergence — {sys_tag}", fontsize=10.5)
                    fig_t.tight_layout()
                    save_fig(fig_t, out_fig_dir, f"fig5_rl_cdlb_training_{sys_tag}")
        except Exception as exc_tr:
            print(f"  [WARN] Gráfica de entrenamiento RL omitida en {sys_tag}: {exc_tr}")

    df_sys["mu_sgn_weighted"] = np.where(df_sys["model_key"] == "RL_MALA_CDLB_opt", mu_sgn_weighted, np.nan)
    df_sys.to_csv(CSV_SYS_DIR / f"summary_{sys_tag}.csv", index=False)
    print(f"[+] Sistema {sys_tag}: resumen CSV ({len(df_sys)} modelos) y figuras guardados.")
    return df_sys


def make_global_article_tables_and_figures() -> None:
    # Re-evaluar cualquier sistema que aún no tenga summary_<SYS_TAG>.csv actualizado
    if RUNS_DIR.is_dir():
        for sys_sub in sorted(RUNS_DIR.iterdir()):
            if not sys_sub.is_dir():
                continue
            n_h5 = len(list(sys_sub.glob("*/run_bundle.h5")))
            if n_h5 == 0:
                continue
            csv_s = CSV_SYS_DIR / f"summary_{sys_sub.name}.csv"
            need_eval = not csv_s.is_file()
            if not need_eval:
                try:
                    prev_df = pd.read_csv(csv_s)
                    if len(prev_df) < n_h5:
                        need_eval = True
                except Exception:
                    need_eval = True
            if need_eval:
                make_system_figures_and_csvs(sys_sub.name)

    sys_csvs = sorted(CSV_SYS_DIR.glob("summary_N*.csv"))
    if not sys_csvs:
        print("[WARN] No se encontraron CSVs por sistema para consolidar figuras globales.")
        return

    df_all = pd.concat([pd.read_csv(p) for p in sys_csvs], ignore_index=True)
    df_all = df_all.sort_values(["N", "g", "family", "sigma_nominal"]).reset_index(drop=True)
    df_all.to_csv(CSV_DIR / "master_definitive_summary.csv", index=False)

    # Exportar tabla dedicada del barrido en sigma
    df_all[df_all["family"].isin(["MALA", "RWMH", "RL-MALA"])].to_csv(
        CSV_DIR / "sigma_sweep_curves_all_systems.csv", index=False
    )

    # Construir tabla comparativa cabeza-a-cabeza: Mejor MALA(sigma*) vs Mejor RWMH(sigma*) vs RL-MALA óptimo
    best_rows: List[Dict[str, Any]] = []
    for (nv, gv), grp in df_all.groupby(["N", "g"]):
        bm = select_best_classical_row(grp[grp["family"] == "MALA"])
        br = select_best_classical_row(grp[grp["family"] == "RWMH"])
        rl_g = grp[grp["model_key"] == "RL_MALA_CDLB_opt"]
        rl = rl_g.iloc[0] if not rl_g.empty else None

        rec: Dict[str, Any] = {
            "N": int(nv),
            "g": float(gv),
            "system_tag": grp["system_tag"].iloc[0],
            "E_CASSCF": float(grp["E_CASSCF"].iloc[0]),
        }
        for prefix, item in [("MALA_best", bm), ("RWMH_best", br), ("RL_MALA_opt", rl)]:
            if item is not None:
                rec[f"{prefix}_sigma"] = float(item["sigma_eff"])
                rec[f"{prefix}_E_mean"] = float(item["E_mean"])
                rec[f"{prefix}_SE_E"] = float(item["SE_E"])
                rec[f"{prefix}_acc"] = float(item["acceptance"])
                rec[f"{prefix}_ESJD"] = float(item["ESJD"])
                rec[f"{prefix}_tau_E"] = float(item["tau_int_E"])
                rec[f"{prefix}_tau_x"] = float(item["tau_int_pos"])
                rec[f"{prefix}_ESS_s"] = float(item["ESS_per_sec"])
                rec[f"{prefix}_flips_1k"] = float(item["hyst_flips_total_1k"])
                rec[f"{prefix}_flips_opp_1k"] = float(item["hyst_flips_opp_1k"])
                rec[f"{prefix}_flips_same_1k"] = float(item["hyst_flips_same_1k"])
                rec[f"{prefix}_f1"] = float(item["f1_ballistic_frac"])
                rec[f"{prefix}_D_KL_spin"] = float(item["D_KL_spin"])
        if bm is not None and rl is not None:
            rec["speedup_flips_vs_MALA"] = float(rl["hyst_flips_total_1k"]) / max(float(bm["hyst_flips_total_1k"]), 1e-6)
            rec["gain_ESS_s_pct_vs_MALA"] = 100.0 * (float(rl["ESS_per_sec"]) / max(float(bm["ESS_per_sec"]), 1e-9) - 1.0)
        best_rows.append(rec)

    df_best = pd.DataFrame(best_rows).sort_values(["N", "g"]).reset_index(drop=True)
    df_best.to_csv(CSV_DIR / "best_per_family_by_system.csv", index=False)

    FIG_GLOBAL_DIR.mkdir(parents=True, exist_ok=True)
    n_vals = sorted(df_best["N"].unique())

    # -------------------------------------------------------------------------
    # FIG GLOBAL 1: Diagrama de Fase PRB (Flips totales, Canales Wigner/Pauli y D_KL^spin vs g)
    # -------------------------------------------------------------------------
    if not df_best.empty:
        fig1, axes1 = plt.subplots(1, 3, figsize=(11.6, 3.8))
        markers_n = {4: "o", 6: "s", 10: "^"}
        cols_rl = {4: "#3B82F6", 6: "#0F62FE", 10: "#002D9C"}
        cols_ma = {4: "#F28482", 6: "#C84B31", 10: "#7F1D1D"}
        cols_rw = {4: "#A594F9", 6: "#6C5B7B", 10: "#3B2F4A"}

        for nv in n_vals:
            sub = df_best[df_best["N"] == nv].sort_values("g")
            mk = markers_n.get(int(nv), "o")
            if "RL_MALA_opt_flips_1k" in sub:
                sp = sub[sub["RL_MALA_opt_flips_1k"] > 0]
                if not sp.empty:
                    axes1[0].plot(sp["g"], sp["RL_MALA_opt_flips_1k"], color=cols_rl.get(int(nv), PAL["rl"]),
                                  ls="-", marker=mk, ms=4.6, lw=1.6, label=rf"RL $N={nv}$")
                axes1[2].plot(sub["g"], sub["RL_MALA_opt_D_KL_spin"], color=cols_rl.get(int(nv), PAL["rl"]),
                              ls="-", marker=mk, ms=4.6, lw=1.6, label=rf"RL $N={nv}$")
            if "MALA_best_flips_1k" in sub:
                sm = sub[sub["MALA_best_flips_1k"] > 0]
                if not sm.empty:
                    axes1[0].plot(sm["g"], sm["MALA_best_flips_1k"], color=cols_ma.get(int(nv), PAL["mala"]),
                                  ls="--", marker=mk, ms=4.0, mfc="white", mew=1.0, lw=1.2, label=rf"MALA* $N={nv}$")
                axes1[2].plot(sub["g"], sub["MALA_best_D_KL_spin"], color=cols_ma.get(int(nv), PAL["mala"]),
                              ls="--", marker=mk, ms=4.0, mfc="white", mew=1.0, lw=1.2, label=rf"MALA* $N={nv}$")
            if "RWMH_best_flips_1k" in sub:
                sr = sub[sub["RWMH_best_flips_1k"] > 0]
                if not sr.empty:
                    axes1[0].plot(sr["g"], sr["RWMH_best_flips_1k"], color=cols_rw.get(int(nv), PAL["rwmh"]),
                                  ls=":", marker=mk, ms=3.8, mfc="white", mew=0.95, lw=1.1, label=rf"RWMH* $N={nv}$")
                axes1[2].plot(sub["g"], sub["RWMH_best_D_KL_spin"], color=cols_rw.get(int(nv), PAL["rwmh"]),
                              ls=":", marker=mk, ms=3.8, mfc="white", mew=0.95, lw=1.1, label=rf"RWMH* $N={nv}$")

        # Panel (b): canales Wigner (opp) vs Pauli (same) para N=6 (o el N intermedio disponible)
        n_focus = 6 if 6 in n_vals else n_vals[len(n_vals) // 2]
        sub_f = df_best[df_best["N"] == n_focus].sort_values("g")
        if "RL_MALA_opt_flips_opp_1k" in sub_f:
            s_ro = sub_f[sub_f["RL_MALA_opt_flips_opp_1k"] > 0]
            s_rs = sub_f[sub_f["RL_MALA_opt_flips_same_1k"] > 0]
            axes1[1].plot(s_ro["g"], s_ro["RL_MALA_opt_flips_opp_1k"], color=PAL["rl"], ls="-", marker="s", ms=4.5, lw=1.6, label=r"RL $\uparrow\!\downarrow$ (Wigner)")
            axes1[1].plot(s_rs["g"], s_rs["RL_MALA_opt_flips_same_1k"], color=PAL["rl_alt"], ls="-", marker="D", ms=4.2, lw=1.45, label="RL Pauli")
        if "MALA_best_flips_opp_1k" in sub_f:
            s_mo = sub_f[sub_f["MALA_best_flips_opp_1k"] > 0]
            s_ms = sub_f[sub_f["MALA_best_flips_same_1k"] > 0]
            if not s_mo.empty:
                axes1[1].plot(s_mo["g"], s_mo["MALA_best_flips_opp_1k"], color=PAL["mala"], ls="--", marker="s", ms=4.0, mfc="white", lw=1.2, label=r"MALA* $\uparrow\!\downarrow$")
            if not s_ms.empty:
                axes1[1].plot(s_ms["g"], s_ms["MALA_best_flips_same_1k"], color=PAL["mala_alt"], ls=":", marker="D", ms=3.8, mfc="white", lw=1.15, label="MALA* Pauli")

        for j, (ylbl, ttl) in enumerate([
            (r"Total Flips / $10^3$ sweeps", "(a) Total Spin-Exchange Rate"),
            (rf"$N={n_focus}$ Flips / $10^3$ sweeps", rf"(b) Wigner vs Pauli Channels ($N={n_focus}$)"),
            (r"$D_{\mathrm{KL}}^{\mathrm{spin}}$", "(c) Asymptotic Spin-Density KL Error"),
        ]):
            axes1[j].set_yscale("log")
            axes1[j].set_xlabel(r"Interaction strength $g$")
            axes1[j].set_ylabel(ylbl)
            axes1[j].set_title(ttl)
            axes1[j].grid(True, which="major", alpha=0.30, color=PAL["grid"])
            h_j, _ = axes1[j].get_legend_handles_labels()
            if h_j:
                axes1[j].legend(fontsize=6.8, ncol=2 if j != 1 else 1, loc="best")
        axes1[2].axhline(1e-3, color="#6E7684", ls=":", lw=0.95)
        fig1.tight_layout()
        save_fig(fig1, FIG_GLOBAL_DIR, "fig_global1_phase_diagram_N4_N6_N10")

    # -------------------------------------------------------------------------
    # FIG GLOBAL 2: Matriz de barridos en sigma (ESJD y D_KL^spin) a través de (N, g)
    # -------------------------------------------------------------------------
    g_rep = [g for g in [0.1, 1.0, 3.0, 8.0] if g in set(df_all["g"].unique())]
    if not g_rep:
        g_rep = sorted(df_all["g"].unique())[:4]
    if n_vals and g_rep:
        nr, nc = len(n_vals), len(g_rep)
        fig2, axes2 = plt.subplots(nr, nc, figsize=(3.15 * nc, 2.65 * nr), squeeze=False, sharex=True)
        for i, nv in enumerate(n_vals):
            for j, gv in enumerate(g_rep):
                ax = axes2[i, j]
                sub = df_all[(df_all["N"] == nv) & (np.isclose(df_all["g"], gv))]
                dm = sub[sub["family"] == "MALA"].sort_values("sigma_nominal")
                dr = sub[sub["family"] == "RWMH"].sort_values("sigma_nominal")
                drl = sub[sub["model_key"] == "RL_MALA_CDLB_opt"]
                if not dm.empty:
                    ax.plot(dm["sigma_nominal"], dm["ESJD"], color=PAL["mala"], ls="--", marker="o", ms=3.8, mfc="white", lw=1.35, label="MALA")
                if not dr.empty:
                    ax.plot(dr["sigma_nominal"], dr["ESJD"], color=PAL["rwmh"], ls="-.", marker="D", ms=3.5, mfc="white", lw=1.25, label="RWMH")
                if not drl.empty:
                    v_rl = float(drl["ESJD"].iloc[0])
                    s_rl = float(drl["sigma_eff"].iloc[0])
                    ax.axhline(v_rl, color=PAL["rl"], ls="-", lw=1.45, alpha=0.85, label="RL-MALA (CDLB)")
                    ax.plot([s_rl], [v_rl], marker="*", ms=8.5, color=PAL["rl"], mec="#082B6B", mew=0.6, zorder=10)
                ax.set_xscale("log")
                ax.set_title(rf"$N={nv},\; g={gv}$", fontsize=9.0)
                ax.grid(True, which="major", alpha=0.30, color=PAL["grid"])
                if j == 0:
                    ax.set_ylabel(r"ESJD ($a_0^2$)")
                if i == nr - 1:
                    ax.set_xlabel(r"$\sigma$ ($a_0$)")
        h2, l2 = axes2[0, 0].get_legend_handles_labels()
        if h2:
            fig2.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=True)
        fig2.tight_layout()
        save_fig(fig2, FIG_GLOBAL_DIR, "fig_global2_sigma_sweep_ESJD_matrix")

    # -------------------------------------------------------------------------
    # FIG GLOBAL 3: Ganancia en ESS/s y reducción de tau_int(E) vs g para N=4,6,10
    # -------------------------------------------------------------------------
    if not df_best.empty and "RL_MALA_opt_ESS_s" in df_best:
        fig3, axes3 = plt.subplots(1, len(n_vals), figsize=(3.8 * len(n_vals), 3.4), squeeze=False)
        for j, nv in enumerate(n_vals):
            ax = axes3[0, j]
            sub = df_best[df_best["N"] == nv].sort_values("g")
            if "RL_MALA_opt_ESS_s" in sub:
                ax.plot(sub["g"], sub["RL_MALA_opt_ESS_s"], color=PAL["rl"], ls="-", marker="s", ms=4.6, lw=1.65, label="RL-MALA (CDLB)")
            if "MALA_best_ESS_s" in sub:
                ax.plot(sub["g"], sub["MALA_best_ESS_s"], color=PAL["mala"], ls="--", marker="o", ms=4.2, mfc="white", lw=1.35, label=r"Best MALA ($\sigma^*$)")
            if "RWMH_best_ESS_s" in sub:
                ax.plot(sub["g"], sub["RWMH_best_ESS_s"], color=PAL["rwmh"], ls="-.", marker="D", ms=3.8, mfc="white", lw=1.25, label=r"Best RWMH ($\sigma^*$)")
            ax.set_yscale("log")
            ax.set_xlabel(r"Interaction strength $g$")
            if j == 0:
                ax.set_ylabel(r"Net GPU $\mathrm{ESS/s}$ ($\mathrm{s}^{-1}$)")
            ax.set_title(rf"$N={nv}$ Net Sampling Throughput")
            ax.grid(True, which="major", alpha=0.30, color=PAL["grid"])
            ax.legend(fontsize=7.4, loc="best")
        fig3.tight_layout()
        save_fig(fig3, FIG_GLOBAL_DIR, "fig_global3_ess_per_sec_comparison")

    print(f"[+] Consolidación global completada: {len(df_all)} corridas en master_definitive_summary.csv.")


if MODE == "system":
    make_system_figures_and_csvs(TARGET_SYS)
else:
    make_global_article_tables_and_figures()
PYEOF
}

run_system_postprocess() {
    local sys_tag="$1"
    local t0
    t0="$(date +%s)"
    log_msg "POST-SYS" "Iniciando análisis y figuras para ${sys_tag}..."
    if run_python_postprocessor "system" "$sys_tag"; then
        local dt=$(( $(date +%s) - t0 ))
        log_msg "POST-SYS" "Completado ${sys_tag} en $(format_duration "$dt") -> Figuras: ${FIG_SYS_DIR}/${sys_tag}"
    else
        record_failure "POST-SYS" "$sys_tag" "postprocess" "Falló postprocesamiento del sistema ${sys_tag}"
    fi
}

run_global_postprocess() {
    local glog="${LOG_POST_DIR}/postprocess_global_article.log"
    local t0
    t0="$(date +%s)"
    log_msg "STAGE-2" "=============================================================================="
    log_msg "STAGE-2" "Consolidando tablas CSV maestras y figuras globales del primer artículo..."
    log_msg "STAGE-2" "Carpeta CSVs    : ${CSV_DIR}"
    log_msg "STAGE-2" "Carpeta Figuras : ${FIG_GLOBAL_DIR} y ${FIG_SYS_DIR}"
    log_msg "STAGE-2" "Log Postproceso : ${glog}"
    log_msg "STAGE-2" "=============================================================================="

    if run_python_postprocessor "global" "ALL" > "$glog" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        log_msg "STAGE-2" "Postprocesamiento global finalizado con éxito en $(format_duration "$dt")."
    else
        record_failure "POST-GLOBAL" "ALL" "global_figures" "Falló consolidación global -> Ver ${glog}"
    fi

    # Asegurar que ningún archivo .csv haya quedado fuera de ${CSV_DIR} y limpiar archivos .lock vacíos
    sync_stray_csvs_to_central_dir "$RUNS_DIR" "$CSV_PROD_DIR"
    sync_stray_csvs_to_central_dir "$CASSCF_REF_DIR" "$CSV_CASSCF_DIR"
    if [[ -d "$CSV_PROD_DIR" ]]; then
        for f_tr in "$CSV_PROD_DIR"/training_history_*.csv; do
            [[ -e "$f_tr" ]] && mv -f "$f_tr" "$CSV_TRAIN_DIR/" 2>/dev/null || true
        done
    fi
    find "$CSV_DIR" -type f -name "*.lock" -delete 2>/dev/null || true
}

# ==============================================================================
# 10. EJECUCIÓN PRINCIPAL DEL ORQUESTADOR
# ==============================================================================
log_msg "INIT" "=============================================================================="
log_msg "INIT" "PRODUCCIÓN DEFINITIVA DE MODELOS — PRIMER ARTÍCULO (MALA vs RWMH vs RL-MALA CDLB)"
log_msg "INIT" "Python detectado : ${PYTHON_BIN}"
log_msg "INIT" "Carpeta Raíz     : ${BASE_OUT_DIR}"
log_msg "INIT" "  -> CSVs        : ${CSV_DIR}"
log_msg "INIT" "  -> Logs        : ${LOG_DIR}"
log_msg "INIT" "  -> Figuras     : ${FIG_DIR}"
log_msg "INIT" "  -> Bundles H5  : ${RUNS_DIR}"
log_msg "INIT" "=============================================================================="

if [[ "$POSTPROCESS_ONLY" -eq 0 ]]; then
    prepare_all_casscf_references
    enqueue_all_production_jobs

    if [[ "$DRY_RUN" -eq 1 ]]; then
        log_msg "DRY-RUN" "Simulación de cola finalizada (${TOTAL_PLANNED_JOBS} corridas listadas, 0 ejecutadas)."
        exit 0
    fi

    log_msg "WAIT" "Esperando finalización de los jobs restantes en las GPUs..."
    wait_all_sampling_jobs
    log_msg "WAIT" "Esperando tareas en segundo plano de figuras por sistema..."
    wait_all_postprocess_jobs
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    run_global_postprocess
fi

TOTAL_ELAPSED=$(( $(date +%s) - BENCHMARK_START_EPOCH ))

log_msg "SUMMARY" "=============================================================================="
log_msg "SUMMARY" "BENCHMARK DEFINITIVO DEL PRIMER ARTÍCULO FINALIZADO"
log_msg "SUMMARY" "Tiempo total transcurrido : $(format_duration "$TOTAL_ELAPSED")"
log_msg "SUMMARY" "Jobs completados nuevos   : ${TOTAL_COMPLETED_JOBS}"
log_msg "SUMMARY" "Jobs reutilizados (SKIP)  : ${TOTAL_SKIPPED_JOBS}"
log_msg "SUMMARY" "Fallos registrados        : ${TOTAL_FAILED} (detalle en ${FAILED_JOBS_LOG})"
log_msg "SUMMARY" "Tabla Maestra Consolidada : ${MASTER_FINAL_CSV}"
log_msg "SUMMARY" "Tabla Campeones (LaTeX)   : ${BEST_FAMILY_CSV}"
log_msg "SUMMARY" "Curvas Barrido en Sigma   : ${SIGMA_SWEEP_CSV}"
log_msg "SUMMARY" "Figuras Globales Artículo : ${FIG_GLOBAL_DIR}"
log_msg "SUMMARY" "Figuras por Sistema       : ${FIG_SYS_DIR}"
log_msg "SUMMARY" "Logs Completos            : ${LOG_DIR}"
log_msg "SUMMARY" "=============================================================================="

if [[ "$TOTAL_FAILED" -gt 0 ]]; then
    exit 1
fi
exit 0
