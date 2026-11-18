#!/usr/bin/env bash
# ==============================================================================
# PIPELINE 4: BÚSQUEDA DE HIPERPARÁMETROS Y ABLACIÓN DE LOS 4 REGÍMENES (AYLER / BAKER)
# ==============================================================================
# Propósito:
#   Evaluar sistemáticamente el nuevo algoritmo de Entrenamiento Variacional
#   Acoplado Online Paso a Paso (`--train_mode online_coupled`, clase
#   `OnlineCoupledTrainer1D`) sobre un sistema en concreto o un grupo pequeño de
#   sistemas representativos (2 o 3 sistemas por defecto: N=4 g=4.0 y N=6 g=1.0),
#   comparando:
#     1. Baseline 1 (`VMC_MALA`): Entrenamiento VMC con MALA puro (`vmc_only`).
#     2. Baseline 2 (`P3_BlockCoupled`): Pipeline 3 por macro-ciclos alternados (`coupled`).
#     3. Ablación de los 4 Regímenes de Pipeline 4 (`online_coupled`):
#        - Régimen A (`R_A_ColdStart_Step1`):
#            N_nodal=0, N_critic=0, N_freeze=0 (Prende y entrena Actor desde el paso 1).
#        - Régimen B (`R_B_BlindWarmup_NoCriticPretrain`):
#            N_nodal=1000, N_critic=0, N_freeze=1000 (Warmup nodal ciego sin pre-entrenar Crítico).
#        - Régimen C (`R_C_NodalAndCritic_NoTailFreeze`):
#            N_nodal=500, N_critic=500, N_freeze=0 (Warmup nodal + Crítico off-policy, sin Tail Freeze).
#        - Régimen D (`R_D_Full4Regimes`):
#            N_nodal>0, N_critic>0, N_freeze>0 (Los 4 Regímenes completos).
#     4. Búsqueda de Hiperparámetros sobre el Régimen D (Full 4 Regímenes):
#        - Partición de pasos (N_nodal, N_critic, N_freeze)
#        - Capacidad del Replay Buffer FIFO (RL_REPLAY_CAPACITY)
#        - Submuestreo de caminantes por paso al buffer (WALKERS_TO_STORE)
#        - Ratio UTD (ONLINE_RL_UPDATES_PER_STEP) y Reward Centering
#        - Escalas físicas de propuesta (tau, actor_amplitude, penalty_factor)
#
# Características QOL Incluidas:
#   - Timestamps exactos y tiempo de cómputo crudo (raw wall-clock seconds y HHh MMm SSs)
#     en cada log individual y en el log maestro.
#   - Impresión completa de todos los hiperparámetros en el encabezado de cada log individual.
#   - Tablas CSV de todas las figuras guardadas aparte en `<OUT_ROOT>/results_csv/`
#     para compartir resultados ligeros sin transferir PDFs pesados.
#   - Figuras guardadas en `.png` por defecto con opción de compilar en `.pdf` a alto DPI
#     (`--pdf --dpi 600` o `--both-formats`).
#   - Banderas `--postprocess-only`, `--only-ablations`, `--only-grid`, `--with-p3`,
#     y selección flexible de sistemas (`--systems "N4:4.0"`, `--systems "N4:2.0 N4:4.0 N6:1.0"`).
#   - Gestión inteligente de GPUs y sistema de colas dinámico:
#     * Selección explícita de GPUs (`--gpus "0 1 2"` o `--gpu 0`) o auto-detección.
#     * Soporte para >1 trabajo por GPU (`--jobs-per-gpu 2`, default 2 cuando VRAM >= 15 GB)
#       ajustando automáticamente `XLA_PYTHON_CLIENT_MEM_FRACTION` y `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
#   - Resumen ejecutivo ordenado por energía variacional, D_KL^spin y ESS/s al cierre del log maestro.
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 1. PARÁMETROS CONFIGURABLES (DESDE VARIABLES DE ENTORNO O TERMINAL)
# ==============================================================================
OUT_ROOT="${OUT_ROOT:-results_pipeline4_param_search}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SEARCH_PRESET="${SEARCH_PRESET:-fast}"      # 'fast' (12 modelos/sistema) o 'full' (24 modelos/sistema)
RUN_BASELINES="${RUN_BASELINES:-1}"         # 1 = correr VMC_MALA y P3_BlockCoupled
RUN_REGIME_ABLATION="${RUN_REGIME_ABLATION:-1}" # 1 = correr ablación de los 4 regímenes (Casos A, B, C, D)
RUN_HYPERPARAM_GRID="${RUN_HYPERPARAM_GRID:-1}" # 1 = correr búsqueda de hiperparámetros del Régimen D

# Por defecto probamos en 2 sistemas representativos (o 1 sistema concreto pasando --systems "N4:4.0")
SYSTEMS_FILTER="${SYSTEMS_FILTER:-N4:4.0 N6:1.0}"
OMEGA="${OMEGA:-0.1}"

# Gestión de GPUs y concurrencia
GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-auto}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"

# Parámetros base del Ansatz y VMC en Pipeline 4
ANSATZ_TYPE="${ANSATZ_TYPE:-slater_jastrow_1d}"
ORBITAL_MODEL="${ORBITAL_MODEL:-backflow_ho}"
CORRELATION_MODEL="${CORRELATION_MODEL:-spin_rbf_jastrow}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SWEEPS_PER_STEP="${SWEEPS_PER_STEP:-5}"
TOTAL_STEPS="${TOTAL_STEPS:-6000}"
STEPS_FOR_BEST="${STEPS_FOR_BEST:-1500}"
LR_ANSATZ="${LR_ANSATZ:-1e-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-200}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.05}"
N_BURN_IN="${N_BURN_IN:-1000}"
INIT_MODE="${INIT_MODE:-randomized_wigner}"
INIT_STD="${INIT_STD:-3.0}"

# Parámetros base de Evaluación Multi-Cadena
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-1000}"
MEASURE_STEPS="${MEASURE_STEPS:-20000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"
SAVE_ENERGY_MODE="${SAVE_ENERGY_MODE:-step_mean}"
SAVE_WALKERS_DTYPE="${SAVE_WALKERS_DTYPE:-float32}"
SAVE_STATS_DTYPE="${SAVE_STATS_DTYPE:-float32}"

# Parámetros base de RL (TD3 + CDLB)
REWARD_TYPE="${REWARD_TYPE:-cdlb}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-2.0}"
SEED="${SEED:-42}"

# Grillas personalizables para la búsqueda de hiperparámetros en Pipeline 4
# Formato de SCHEDULE_GRID: "NODAL_WARMUP:CRITIC_WARMUP:FREEZE_TAIL"
SCHEDULE_GRID_STR="${SCHEDULE_GRID_STR:-}"
REPLAY_CAP_GRID_STR="${REPLAY_CAP_GRID_STR:-}"
WALKERS_STORE_GRID_STR="${WALKERS_STORE_GRID_STR:-}"
UTD_GRID_STR="${UTD_GRID_STR:-}"
ACTOR_AMP_GRID_STR="${ACTOR_AMP_GRID_STR:-}"
PENALTY_GRID_STR="${PENALTY_GRID_STR:-}"
TAU_GRID_STR="${TAU_GRID_STR:-}"
REWARD_CENTERING_GRID_STR="${REWARD_CENTERING_GRID_STR:-}"

# Formato de figuras (PNG por defecto, opción PDF / alto DPI)
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TITLE_PREFIX="${TITLE_PREFIX:-}"
CUSTOM_SUPTITLE="${CUSTOM_SUPTITLE:-}"
NO_TITLES="${NO_TITLES:-0}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"
TRANSPARENT_BG="${TRANSPARENT_BG:-0}"
TOP_K_MODELS="${TOP_K_MODELS:-3}"
WALKER_TRACE_TOP_K="${WALKER_TRACE_TOP_K:-1}"
REALTIME_PLOT_UPDATES="${REALTIME_PLOT_UPDATES:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --postprocess-only|--postprocess_only|--plots-only|-p) POSTPROCESS_ONLY=1; shift ;;
        --force-rerun|--force_rerun) FORCE_RERUN=1; SKIP_EXISTING=0; shift ;;
        --preset) SEARCH_PRESET="$2"; shift 2 ;;
        --systems|--system|--system-filter|-s) SYSTEMS_FILTER="$2"; shift 2 ;;
        --gpu|--gpus|--gpu-list) GPU_LIST="$2"; shift 2 ;;
        --jobs-per-gpu|-j) JOBS_PER_GPU="$2"; shift 2 ;;
        --max-parallel-jobs) MAX_PARALLEL_JOBS="$2"; shift 2 ;;
        --only-ablations) RUN_BASELINES=1; RUN_REGIME_ABLATION=1; RUN_HYPERPARAM_GRID=0; shift ;;
        --only-grid) RUN_BASELINES=1; RUN_REGIME_ABLATION=0; RUN_HYPERPARAM_GRID=1; shift ;;
        --no-baselines) RUN_BASELINES=0; shift ;;
        --ansatz) ANSATZ_TYPE="$2"; shift 2 ;;
        --orbital) ORBITAL_MODEL="$2"; shift 2 ;;
        --corr) CORRELATION_MODEL="$2"; shift 2 ;;
        --batch-size|--walkers|-w) BATCH_SIZE="$2"; shift 2 ;;
        --sweeps) SWEEPS_PER_STEP="$2"; shift 2 ;;
        --total-steps|--train-steps) TOTAL_STEPS="$2"; shift 2 ;;
        --steps-for-best) STEPS_FOR_BEST="$2"; shift 2 ;;
        --lr-ansatz) LR_ANSATZ="$2"; shift 2 ;;
        --lr-actor) LR_ACTOR="$2"; shift 2 ;;
        --lr-critic) LR_CRITIC="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --eval-chains|--chains) N_EVAL_CHAINS="$2"; shift 2 ;;
        --schedule-grid) SCHEDULE_GRID_STR="$2"; shift 2 ;;
        --replay-grid) REPLAY_CAP_GRID_STR="$2"; shift 2 ;;
        --walkers-store-grid) WALKERS_STORE_GRID_STR="$2"; shift 2 ;;
        --utd-grid) UTD_GRID_STR="$2"; shift 2 ;;
        --amp-grid) ACTOR_AMP_GRID_STR="$2"; shift 2 ;;
        --pen-grid) PENALTY_GRID_STR="$2"; shift 2 ;;
        --tau-grid) TAU_GRID_STR="$2"; shift 2 ;;
        --rc-grid) REWARD_CENTERING_GRID_STR="$2"; shift 2 ;;
        --figure-format|--figure_format|--format|-f) FIGURE_FORMAT="$2"; shift 2 ;;
        --pdf) FIGURE_FORMAT="pdf"; shift ;;
        --png) FIGURE_FORMAT="png"; shift ;;
        --both-formats) FIGURE_FORMAT="both"; shift ;;
        --dpi|--ppi) FIGURE_DPI="$2"; shift 2 ;;
        --title-prefix|--title_prefix) TITLE_PREFIX="$2"; shift 2 ;;
        --custom-suptitle|--custom_suptitle) CUSTOM_SUPTITLE="$2"; shift 2 ;;
        --no-titles|--no_titles|--hide-titles) NO_TITLES=1; shift ;;
        --font-scale|--font_scale) FONT_SCALE="$2"; shift 2 ;;
        --line-width-scale|--line_width_scale) LINE_WIDTH_SCALE="$2"; shift 2 ;;
        --transparent-bg|--transparent_bg) TRANSPARENT_BG=1; shift ;;
        --top-k) TOP_K_MODELS="$2"; shift 2 ;;
        --no-realtime-plots) REALTIME_PLOT_UPDATES=0; shift ;;
        -h|--help)
            cat << 'HELP'
Uso: ./run_pipeline4_param_search.sh [carpeta_salida] [opciones]

Opciones principales:
  --systems "N4:4.0"                     Sistema(s) a evaluar (default: "N4:4.0 N6:1.0", o ej. "N4:2.0 N4:4.0 N6:1.0")
  --preset [fast|full]                   Tamaño de la grilla de búsqueda (default: fast)
  --only-ablations                       Correr únicamente Baselines + Ablación de los 4 Regímenes (A, B, C, D)
  --only-grid                            Correr únicamente Baselines + Grilla de Hiperparámetros del Régimen D
  --gpus "0 1 2" --jobs-per-gpu 2        Selección de GPUs y trabajos concurrentes por GPU en Ayler/Baker
  --total-steps 6000 --batch-size 512    Pasos VMC totales y número de caminantes por paso
  --measure-steps 20000 --chains 3       Pasos MCMC y cadenas independientes en fase de evaluación
  --schedule-grid "300:300:900 600:600:1200"  Particiones N_nodal:N_critic:N_freeze a explorar
  --replay-grid "15000 30000"            Capacidades del Replay Buffer FIFO a explorar
  --walkers-store-grid "32 64 128"       Caminantes insertados al buffer por paso VMC
  --utd-grid "1 2"                       Pasos de gradiente TD3 por paso VMC (ratio UTD)
  --amp-grid "0.10 0.15"                 Amplitudes del Actor a explorar
  --pen-grid "0.15 0.25"                 Factores de penalización CDLB a explorar
  --postprocess-only                     Regenerar sólo figuras y tablas CSV en CPU
  --pdf --dpi 600                        Exportar figuras en PDF de alta resolución (default: PNG @ 300 DPI)
  --force-rerun                          Recalcular todos los modelos desde cero
HELP
            exit 0
            ;;
        *)
            OUT_ROOT="$1"
            shift
            ;;
    esac
done

# ==============================================================================
# 2. CONFIGURACIÓN DE ENTORNO JAX/XLA Y GESTIÓN INTELIGENTE DE MEMORIA GPU
# ==============================================================================
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true

for cuda_cand in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-11*; do
    if [[ -d "$cuda_cand/bin" ]]; then
        export PATH="${cuda_cand}/bin:$PATH"
        export LD_LIBRARY_PATH="${cuda_cand}/lib64:${LD_LIBRARY_PATH:-}"
        break
    fi
done
export XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || echo "python3")}"

AVAILABLE_GPUS=()
TOTAL_VRAM_MB=0
MAX_TOTAL_JOBS=1

discover_gpus_and_memory() {
    if [[ "$GPU_LIST" != "auto" && -n "$GPU_LIST" ]]; then
        read -r -a AVAILABLE_GPUS <<< "$GPU_LIST"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        local detected
        detected=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
        if [[ -n "$detected" ]]; then
            read -r -a AVAILABLE_GPUS <<< "$detected"
        else
            AVAILABLE_GPUS=(0)
        fi
    else
        AVAILABLE_GPUS=(0)
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || echo 0)
    fi

    if [[ "$JOBS_PER_GPU" == "auto" ]]; then
        if [[ "${TOTAL_VRAM_MB:-0}" -ge 15000 ]]; then
            JOBS_PER_GPU=2
        else
            JOBS_PER_GPU=1
        fi
    fi

    if [[ "$MAX_PARALLEL_JOBS" == "auto" ]]; then
        MAX_TOTAL_JOBS=$(( ${#AVAILABLE_GPUS[@]} * JOBS_PER_GPU ))
    else
        MAX_TOTAL_JOBS="$MAX_PARALLEL_JOBS"
    fi
    [[ "$MAX_TOTAL_JOBS" -lt 1 ]] && MAX_TOTAL_JOBS=1

    if [[ "$JOBS_PER_GPU" -ge 3 ]]; then
        export XLA_PYTHON_CLIENT_MEM_FRACTION="0.28"
    elif [[ "$JOBS_PER_GPU" -eq 2 ]]; then
        export XLA_PYTHON_CLIENT_MEM_FRACTION="0.42"
    else
        export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
    fi
}

discover_gpus_and_memory

# Configurar grillas según preset y overrides
N_WARM_DEFAULT=$(( TOTAL_STEPS / 12 ))
[[ "$N_WARM_DEFAULT" -lt 150 ]] && N_WARM_DEFAULT=150
N_FREEZE_DEFAULT=$(( TOTAL_STEPS / 8 ))
[[ "$N_FREEZE_DEFAULT" -lt 250 ]] && N_FREEZE_DEFAULT=250

N_WARM_SHORT=$(( TOTAL_STEPS / 20 ))
[[ "$N_WARM_SHORT" -lt 100 ]] && N_WARM_SHORT=100
N_WARM_LONG=$(( TOTAL_STEPS / 8 ))
[[ "$N_WARM_LONG" -lt 250 ]] && N_WARM_LONG=250

if [[ "$SEARCH_PRESET" == "full" ]]; then
    SCHEDULE_GRID=(${SCHEDULE_GRID_STR:-"${N_WARM_SHORT}:${N_WARM_SHORT}:${N_FREEZE_DEFAULT}" "${N_WARM_DEFAULT}:${N_WARM_DEFAULT}:${N_FREEZE_DEFAULT}" "${N_WARM_LONG}:${N_WARM_DEFAULT}:${N_FREEZE_DEFAULT}"})
    REPLAY_CAP_GRID=(${REPLAY_CAP_GRID_STR:-15000 30000})
    WALKERS_STORE_GRID=(${WALKERS_STORE_GRID_STR:-32 64 128})
    UTD_GRID=(${UTD_GRID_STR:-1 2})
    ACTOR_AMP_GRID=(${ACTOR_AMP_GRID_STR:-0.10 0.15})
    PENALTY_GRID=(${PENALTY_GRID_STR:-0.15 0.25})
else
    # Preset 'fast': explora de manera ortogonal las dimensiones críticas de Pipeline 4
    SCHEDULE_GRID=(${SCHEDULE_GRID_STR:-"${N_WARM_SHORT}:${N_WARM_SHORT}:${N_FREEZE_DEFAULT}" "${N_WARM_DEFAULT}:${N_WARM_DEFAULT}:${N_FREEZE_DEFAULT}" "${N_WARM_LONG}:${N_WARM_DEFAULT}:${N_FREEZE_DEFAULT}"})
    REPLAY_CAP_GRID=(${REPLAY_CAP_GRID_STR:-15000 25000})
    WALKERS_STORE_GRID=(${WALKERS_STORE_GRID_STR:-32 64})
    UTD_GRID=(${UTD_GRID_STR:-1 2})
    ACTOR_AMP_GRID=(${ACTOR_AMP_GRID_STR:-0.12 0.15})
    PENALTY_GRID=(${PENALTY_GRID_STR:-0.25})
fi

PLOT_STYLE_ARGS=(
    --figure_format "$FIGURE_FORMAT"
    --dpi "$FIGURE_DPI"
    --font_scale "$FONT_SCALE"
    --line_width_scale "$LINE_WIDTH_SCALE"
)
[[ -n "$TITLE_PREFIX" ]] && PLOT_STYLE_ARGS+=(--title_prefix "$TITLE_PREFIX")
[[ -n "$CUSTOM_SUPTITLE" ]] && PLOT_STYLE_ARGS+=(--custom_suptitle "$CUSTOM_SUPTITLE")
[[ "$NO_TITLES" -eq 1 || "$NO_TITLES" == "true" ]] && PLOT_STYLE_ARGS+=(--no_titles)
[[ "$TRANSPARENT_BG" -eq 1 || "$TRANSPARENT_BG" == "true" ]] && PLOT_STYLE_ARGS+=(--transparent_bg)

LOG_DIR="${OUT_ROOT}/logs"
CSV_COLLECT_DIR="${OUT_ROOT}/results_csv"
FIG_DIR="${OUT_ROOT}/figures"
GLOBAL_FIG_DIR="${FIG_DIR}/all_models_global"
MASTER_RUNS_CSV="${OUT_ROOT}/live_p4_search_runs.csv"
MASTER_COMP_CSV="${OUT_ROOT}/live_p4_search_comparison.csv"
MASTER_LOG="${LOG_DIR}/master_p4_param_search.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$CSV_COLLECT_DIR" "$FIG_DIR" "$GLOBAL_FIG_DIR"
touch "$MASTER_LOG"

MASTER_START_EPOCH=$(date +%s)
MASTER_START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_PID=$$

TRACKER_DIR=$(mktemp -d /tmp/rl_vmc_p4_search_XXXXXX 2>/dev/null || mktemp -d -t 'rl_vmc_p4_search')
PLOT_LOCK_DIR="${TRACKER_DIR}/plot.lock"
FAILED_JOBS_DIR="${TRACKER_DIR}/failures"
mkdir -p "$FAILED_JOBS_DIR"

cleanup_on_exit() {
    local exit_status=$?
    if [ "${BASH_SUBSHELL:-0}" -gt 0 ] || { [ -n "${BASHPID:-}" ] && [ "${BASHPID}" != "${MASTER_PID}" ]; }; then
        exit "$exit_status"
    fi
    if [ -d "$TRACKER_DIR" ]; then
        for f in "$TRACKER_DIR"/job_*.meta; do
            [ -e "$f" ] || continue
            local pid
            pid=$(basename "$f" | cut -d'_' -f2)
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null || true
            fi
        done
        rm -rf "$TRACKER_DIR"
    fi
    exit "$exit_status"
}
trap cleanup_on_exit EXIT INT TERM

format_duration() {
    local dur=$1
    [[ "$dur" -lt 0 ]] && dur=0
    printf "%02dh %02dm %02ds" $((dur/3600)) $(( (dur%3600)/60 )) $((dur%60))
}

log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$MASTER_LOG"
}

log_msg "=============================================================================="
log_msg " INICIANDO BÚSQUEDA DE HIPERPARÁMETROS Y ABLACIÓN DE 4 REGÍMENES (PIPELINE 4)"
log_msg " Fecha y Hora         : $MASTER_START_TIME"
log_msg " PID del Master       : $MASTER_PID"
log_msg " Directorio raíz      : $OUT_ROOT"
log_msg " Carpeta CSVs ligeros : $CSV_COLLECT_DIR"
log_msg " Sistemas objetivo    : $SYSTEMS_FILTER (omega=$OMEGA)"
log_msg " Preset de búsqueda   : $SEARCH_PRESET (Baselines=$RUN_BASELINES, Ablación4R=$RUN_REGIME_ABLATION, Grid=$RUN_HYPERPARAM_GRID)"
log_msg " GPUs en Pool         : ${AVAILABLE_GPUS[*]} (VRAM=${TOTAL_VRAM_MB} MiB, Trabajos/GPU=$JOBS_PER_GPU, Total=$MAX_TOTAL_JOBS, XLA_MEM=$XLA_PYTHON_CLIENT_MEM_FRACTION)"
log_msg " Parámetros VMC       : ansatz=$ANSATZ_TYPE ($ORBITAL_MODEL + $CORRELATION_MODEL), B=$BATCH_SIZE, sweeps=$SWEEPS_PER_STEP, steps=$TOTAL_STEPS"
log_msg " Evaluación MultiChain: init=$EVAL_INIT_MODE, chains=$N_EVAL_CHAINS, burn_in=$EVAL_BURN_IN, steps=$MEASURE_STEPS"
log_msg " Formato de Figuras   : $FIGURE_FORMAT @ $FIGURE_DPI DPI"
log_msg "=============================================================================="

# ==============================================================================
# 3. SISTEMA DE COLAS DINÁMICO MULTI-GPU Y TRACKING DE TIEMPO CRUDO
# ==============================================================================
is_h5_completed() {
    local h5_file="$1"
    [ -f "$h5_file" ] || return 1
    [ -s "$h5_file" ] || return 1
    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" -c '
import importlib.util, sys
io_path, h5_path = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("io_bundle", io_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ok, _ = mod.validate_run_bundle_h5(h5_path, require_production=True, require_walkers=True)
sys.exit(0 if ok else 1)
' "${SCRIPT_DIR}/rl_vmc/core/io_bundle.py" "$h5_file" >/dev/null 2>&1
}

record_failure() {
    local sys_tag="$1"
    local run_name="$2"
    local detail="$3"
    local safe_tag
    safe_tag=$(printf '%s' "${sys_tag}_${run_name}" | tr -cs 'A-Za-z0-9._-' '_')
    printf '[%s] %s | %s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$sys_tag" "$run_name" "$detail" > "${FAILED_JOBS_DIR}/fail_${safe_tag}_${RANDOM}.txt"
}

cleanup_finished_jobs() {
    for f in "$TRACKER_DIR"/job_*.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local exit_code=$?
            local gpu sys run_name start_epoch start_time log_file
            # shellcheck disable=SC1090
            source "$f"
            local end_epoch end_time duration duration_formatted
            end_epoch=$(date +%s)
            end_time=$(date "+%Y-%m-%d %H:%M:%S")
            duration=$((end_epoch - start_epoch))
            duration_formatted=$(format_duration "$duration")

            log_msg "------------------------------------------------------------------------------"
            log_msg "<< [TRABAJO FINALIZADO] Modelo: $run_name (Sistema: $sys)"
            log_msg "   PID del Worker     : $pid (Código de salida: $exit_code)"
            log_msg "   GPU Liberada       : $gpu"
            log_msg "   Hora de Inicio     : $start_time"
            log_msg "   Hora de Terminación: $end_time"
            log_msg "   Tiempo Cómputo Raw : ${duration}s ($duration_formatted)"
            log_msg "   Log Individual     : $log_file"
            log_msg "=============================================================================="

            if [ "$exit_code" -ne 0 ]; then
                record_failure "$sys" "$run_name" "Exit code ${exit_code} tras ${duration}s -> ${log_file}"
            fi
            rm -f "$f"
        fi
    done
}

get_active_jobs_on_gpu() {
    local target_gpu="$1"
    local count=0
    for f in "$TRACKER_DIR"/job_*_gpu_${target_gpu}.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

get_total_active_jobs() {
    local count=0
    for f in "$TRACKER_DIR"/job_*.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

SELECTED_GPU=""
wait_for_free_gpu_slot() {
    SELECTED_GPU=""
    while true; do
        cleanup_finished_jobs
        local total_active
        total_active=$(get_total_active_jobs)
        if [ "$total_active" -lt "$MAX_TOTAL_JOBS" ]; then
            local best_gpu=""
            local min_jobs=999999
            for gpu in "${AVAILABLE_GPUS[@]}"; do
                local cur_jobs
                cur_jobs=$(get_active_jobs_on_gpu "$gpu")
                if [ "$cur_jobs" -lt "$JOBS_PER_GPU" ] && [ "$cur_jobs" -lt "$min_jobs" ]; then
                    min_jobs="$cur_jobs"
                    best_gpu="$gpu"
                fi
            done
            if [ -n "$best_gpu" ]; then
                SELECTED_GPU="$best_gpu"
                return 0
            fi
        fi
        sleep 2
    done
}

wait_all_jobs() {
    while true; do
        cleanup_finished_jobs
        local total_active
        total_active=$(get_total_active_jobs)
        if [ "$total_active" -eq 0 ]; then
            break
        fi
        sleep 2
    done
}

# ==============================================================================
# 4. HELPERS DE PARÁMETROS FÍSICOS Y POST-PROCESAMIENTO EN CPU
# ==============================================================================
get_system_default_params() {
    local N=$1
    local G=$2
    local tau="0.10"
    local actor_amp="0.15"
    local penalty="0.25"
    local sig_scale="1.1"
    local soft="0.1"
    [[ "$N" -ge 10 ]] && soft="0.25"

    local is_strong_g
    is_strong_g=$(awk -v g="$G" 'BEGIN { print (g >= 2.5) ? 1 : 0 }')
    if [ "$N" -le 4 ]; then
        tau="0.10"
        [[ "$is_strong_g" -eq 1 ]] && actor_amp="0.12" || actor_amp="0.15"
        sig_scale="1.1"
    elif [ "$N" -le 6 ]; then
        [[ "$is_strong_g" -eq 1 ]] && { tau="0.08"; actor_amp="0.12"; } || { tau="0.10"; actor_amp="0.15"; }
        sig_scale="1.1"
    else
        [[ "$is_strong_g" -eq 1 ]] && { tau="0.06"; actor_amp="0.08"; } || { tau="0.08"; actor_amp="0.10"; }
        sig_scale="0.90"
    fi
    echo "$tau $actor_amp $penalty $sig_scale $soft"
}

get_system_wigner_init_offset() {
    local N=$1
    local G=$2
    "$PYTHON_BIN" -c '
import sys
N, g = int(sys.argv[1]), float(sys.argv[2])
step_pos = round(2.4 * ((1.0 + 1.8 * g) ** 0.28) * ((6.0 / max(N, 2)) ** 0.12), 3)
base_pos = round(-0.5 * (N - 1) * step_pos, 3)
print(f"{base_pos} {step_pos}")
' "$N" "$G"
}

find_reference_casscf_npz() {
    local N=$1
    local G=$2
    "$PYTHON_BIN" - "$SCRIPT_DIR" "$N" "$G" << 'PYEOF'
import glob, os, re, sys
root, n_req, g_req = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
candidates = []
for sub in ("CASSCF_csv", "figures", "figures_allmodels_benchmarks", "results_production_3pipelines_baker/pipeline1_casscf"):
    d = os.path.join(root, sub)
    if os.path.isdir(d):
        candidates.extend(sorted(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)))
for p in candidates:
    b = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    for txt in (b, parent):
        m = re.search(r"n(\d+)_(?:w[0-9.]+_)?g([0-9]+(?:\.[0-9]+)?)", txt)
        if m and int(m.group(1)) == n_req and abs(float(m.group(2)) - g_req) <= 1e-4:
            print(p)
            sys.exit(0)
PYEOF
}

compile_system_figures() {
    local sys_tag="$1"
    local npz_path="$2"
    local sys_out="${OUT_ROOT}/${sys_tag}"
    local sys_fig="${FIG_DIR}/${sys_tag}"
    mkdir -p "$sys_fig"

    local h5_files=()
    local h5_labels=()
    for d in "${sys_out}"/*; do
        if [[ -d "$d" && -f "${d}/run_bundle.h5" ]]; then
            if is_h5_completed "${d}/run_bundle.h5"; then
                h5_files+=("${d}/run_bundle.h5")
                local bname
                bname="$(basename "$d")"
                h5_labels+=("$bname")
            fi
        fi
    done

    if [[ "${#h5_files[@]}" -eq 0 ]]; then
        log_msg "[AVISO] No se encontraron archivos run_bundle.h5 válidos en ${sys_out}."
        return 0
    fi

    local extra_casscf=()
    if [[ -n "$npz_path" && -f "$npz_path" ]]; then
        extra_casscf=(--casscf_npz_path "$npz_path")
    fi

    log_msg "[${sys_tag} | POSTPROCESS] Compilando las 8 figuras estándar, evolución de hiperparámetros y CSVs (${#h5_files[@]} modelos)..."
    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        "$PYTHON_BIN" -u cli/postprocess.py \
        --inputs "${h5_files[@]}" \
        --labels "${h5_labels[@]}" \
        "${extra_casscf[@]}" \
        --out_dir "${sys_fig}/postprocess_comparison" \
        --system_tag "P4Search_${sys_tag}" \
        --top_k_models "$TOP_K_MODELS" \
        --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
        --master_comparison_csv "$MASTER_COMP_CSV" \
        --csv_collect_dir "$CSV_COLLECT_DIR" \
        "${PLOT_STYLE_ARGS[@]}" \
        >> "${LOG_DIR}/postprocess_${sys_tag}.log" 2>&1 || log_msg "[AVISO] cli/postprocess.py reportó advertencias en ${sys_tag}"

    log_msg "[${sys_tag} | NODAL CROSSING] Compilando las 4 figuras de cruce nodal y trazas de 1 caminante..."
    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        "$PYTHON_BIN" -u cli/evaluate_nodal_crossing_1d.py \
        --inputs "${h5_files[@]}" \
        --labels "${h5_labels[@]}" \
        "${extra_casscf[@]}" \
        --system_tag "P4Search_${sys_tag}" \
        --out_dir "$sys_fig" \
        --csv_collect_dir "$CSV_COLLECT_DIR" \
        --top_k_models "$TOP_K_MODELS" \
        --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
        "${PLOT_STYLE_ARGS[@]}" \
        >> "${LOG_DIR}/nodal_crossing_${sys_tag}.log" 2>&1 || log_msg "[AVISO] cli/evaluate_nodal_crossing_1d.py reportó advertencias en ${sys_tag}"

    for f_out in "${sys_fig}/postprocess_comparison"/*.png "${sys_fig}/postprocess_comparison"/*.pdf "${sys_fig}/postprocess_comparison"/*.csv "${sys_fig}/postprocess_comparison"/*.json; do
        if [[ -e "$f_out" ]]; then
            cp -f "$f_out" "$sys_fig/" 2>/dev/null || true
        fi
    done

    if [[ -f "$MASTER_COMP_CSV" && -s "$MASTER_COMP_CSV" ]]; then
        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            "$PYTHON_BIN" -u scripts/plot_param_search_results.py \
            --csv "$MASTER_COMP_CSV" \
            --out_dir "$GLOBAL_FIG_DIR" \
            "${PLOT_STYLE_ARGS[@]}" \
            >> "${LOG_DIR}/global_param_search_plots.log" 2>&1 || true

        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            "$PYTHON_BIN" -u scripts/plot_benchmark_all_models_summary.py \
            --csv "$MASTER_COMP_CSV" \
            --out_dir "$GLOBAL_FIG_DIR" \
            "${PLOT_STYLE_ARGS[@]}" \
            >> "${LOG_DIR}/global_param_search_plots.log" 2>&1 || true
    fi
}

launch_search_job() {
    local sys_tag="$1"
    local npz_path="$2"
    local run_name="$3"
    local run_dir="$4"
    shift 4
    local cmd=("$@")
    local h5_file="${run_dir}/run_bundle.h5"

    if [[ "$SKIP_EXISTING" -eq 1 && "$FORCE_RERUN" -eq 0 ]] && is_h5_completed "$h5_file"; then
        log_msg "[SKIP] '${run_name}' ya cuenta con run_bundle.h5 válido en ${run_dir}."
        return 0
    fi

    mkdir -p "$run_dir"
    local log_file="${LOG_DIR}/${run_name}.log"

    wait_for_free_gpu_slot
    local gpu_id="$SELECTED_GPU"
    local start_epoch start_time
    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    (
        trap - EXIT INT TERM
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        local t0
        t0=$(date +%s)
        {
            echo "=============================================================================="
            echo ">> [PIPELINE 4 PARAM SEARCH] INICIANDO MODELO: ${run_name}"
            echo "   Sistema Físico     : ${sys_tag}"
            echo "   GPU Asignada       : ${gpu_id} (XLA_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION})"
            echo "   Hora de Inicio     : ${start_time}"
            echo "   Directorio Salida  : ${run_dir}"
            echo "   Comando Exacto     : ${cmd[*]}"
            echo "=============================================================================="
        } > "$log_file"

        "${cmd[@]}" >> "$log_file" 2>&1
        local rc=$?
        local t1 duration dur_fmt
        t1=$(date +%s)
        duration=$((t1 - t0))
        dur_fmt=$(printf "%02dh %02dm %02ds" $((duration/3600)) $(( (duration%3600)/60 )) $((duration%60)))

        {
            echo "------------------------------------------------------------------------------"
            echo "<< [PIPELINE 4 PARAM SEARCH] FINALIZADO: ${run_name} (Exit Code: ${rc})"
            echo "   Hora de Inicio     : ${start_time}"
            echo "   Hora de Terminación: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "   Tiempo Cómputo Raw : ${duration}s (${dur_fmt})"
            echo "=============================================================================="
        } >> "$log_file"

        if [[ "$rc" -eq 0 && "$REALTIME_PLOT_UPDATES" -eq 1 ]]; then
            if mkdir "$PLOT_LOCK_DIR" 2>/dev/null; then
                compile_system_figures "$sys_tag" "$npz_path"
                rmdir "$PLOT_LOCK_DIR" 2>/dev/null || true
            fi
        fi
        exit "$rc"
    ) &
    local worker_pid=$!

    cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${gpu_id}.meta"
gpu="$gpu_id"
sys="$sys_tag"
run_name="$run_name"
start_epoch="$start_epoch"
start_time="$start_time"
log_file="$log_file"
EOF

    log_msg ">> [TRABAJO ASIGNADO] Modelo: ${run_name} | GPU: ${gpu_id} | PID: ${worker_pid} | Log: ${log_file}"
}

# ==============================================================================
# 5. PARSEO DE SISTEMAS Y LANZAMIENTO DE BÚSQUEDA POR SISTEMA
# ==============================================================================
SYSTEM_PAIRS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && SYSTEM_PAIRS+=("$line")
done < <("$PYTHON_BIN" - "$SYSTEMS_FILTER" << 'PYEOF'
import re, sys
spec = sys.argv[1]
pairs = []
seen = set()
for tok in re.split(r"[\s;]+", spec.strip()):
    if not tok:
        continue
    m_tag = re.match(r"^N(\d+)_w[0-9.]+_g([0-9.]+)$", tok, flags=re.IGNORECASE)
    if m_tag:
        nv, gv = int(m_tag.group(1)), float(m_tag.group(2))
        if (nv, gv) not in seen:
            seen.add((nv, gv)); pairs.append((nv, gv))
        continue
    if ":" in tok:
        n_p, g_p = tok.split(":", 1)
        nv = int(re.sub(r"^[Nn]", "", n_p.strip()))
        for x in g_p.split(","):
            if x.strip():
                gv = float(x)
                if (nv, gv) not in seen:
                    seen.add((nv, gv)); pairs.append((nv, gv))
for nv, gv in pairs:
    g_str = f"{gv:.1f}" if abs(gv - round(gv)) < 1e-9 else f"{gv:g}"
    print(f"{nv} {g_str}")
PYEOF
)

run_search_for_system() {
    local N="$1"
    local G="$2"
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local SYS_OUT="${OUT_ROOT}/${SYS_TAG}"
    mkdir -p "$SYS_OUT"

    read -r TAU ACTOR_AMP PENALTY SIG_SCALE SOFT <<< "$(get_system_default_params "$N" "$G")"
    read -r INIT_OFFSET_BASE INIT_OFFSET_STEP <<< "$(get_system_wigner_init_offset "$N" "$G")"
    local CASSCF_NPZ
    CASSCF_NPZ="$(find_reference_casscf_npz "$N" "$G")"

    log_msg "=============================================================================="
    log_msg " SISTEMA EN EVALUACIÓN: ${SYS_TAG} (N=${N}, up=${N_UP}, dn=${N_DN}, g=${G}, omega=${OMEGA}, soft=${SOFT})"
    log_msg " Referencia CASSCF    : ${CASSCF_NPZ:-No encontrada (se usará búsqueda automática)}"
    log_msg " Parámetros Base      : tau=${TAU}, amp=${ACTOR_AMP}, pen=${PENALTY}, sig=${SIG_SCALE}"
    log_msg "=============================================================================="

    local COMMON_ARGS=(
        --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN"
        --omega "$OMEGA" --g "$G" --softening "$SOFT"
        --ansatz_type "$ANSATZ_TYPE"
        --orbital_model "$ORBITAL_MODEL"
        --correlation_model "$CORRELATION_MODEL"
        --batch_size "$BATCH_SIZE"
        --sweeps_per_step "$SWEEPS_PER_STEP"
        --n_ter "$N_BURN_IN"
        --lr_ansatz "$LR_ANSATZ"
        --use_lr_scheduler
        --lr_scheduler_type cosine
        --lr_warmup_steps "$LR_WARMUP_STEPS"
        --lr_min_ratio "$LR_MIN_RATIO"
        --grad_clip_norm "$GRAD_CLIP_NORM"
        --steps_for_best "$STEPS_FOR_BEST"
        --init_mode "$INIT_MODE"
        --init_std "$INIT_STD"
        --eval_init_mode "$EVAL_INIT_MODE"
        --init_offset_base "$INIT_OFFSET_BASE"
        --init_offset_step "$INIT_OFFSET_STEP"
        --n_eval_chains "$N_EVAL_CHAINS"
        --eval_burn_in "$EVAL_BURN_IN"
        --measure_steps "$MEASURE_STEPS"
        --eloc_chunk_size "$ELOC_CHUNK_SIZE"
        --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
        --save_production_energy_mode "$SAVE_ENERGY_MODE"
        --save_production_walkers_dtype "$SAVE_WALKERS_DTYPE"
        --save_production_stats_dtype "$SAVE_STATS_DTYPE"
        --master_csv "$MASTER_RUNS_CSV"
        --csv_collect_dir "$CSV_COLLECT_DIR"
        --seed "$SEED"
        --no-export_legacy_files
    )
    if [[ -n "$CASSCF_NPZ" && -f "$CASSCF_NPZ" ]]; then
        COMMON_ARGS+=(--reference_casscf_npz "$CASSCF_NPZ")
        if [[ "$ANSATZ_TYPE" == "multi_sjb" || "$ANSATZ_TYPE" == "multi_slater_jastrow" ]]; then
            COMMON_ARGS+=(--casscf_npz_path "$CASSCF_NPZ")
        fi
    fi

    if [[ "$POSTPROCESS_ONLY" -eq 1 ]]; then
        compile_system_figures "$SYS_TAG" "$CASSCF_NPZ"
        return 0
    fi

    # --------------------------------------------------------------------------
    # A. BASELINES: VMC+MALA (`vmc_only`) y PIPELINE 3 POR BLOQUES (`coupled`)
    # --------------------------------------------------------------------------
    if [[ "$RUN_BASELINES" -eq 1 ]]; then
        local rname_mala="${SYS_TAG}_01_Baseline_VMC_MALA_tau${TAU}"
        local rdir_mala="${SYS_OUT}/${rname_mala}"
        launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_mala" "$rdir_mala" \
            "$PYTHON_BIN" -u cli/run_1d.py \
            "${COMMON_ARGS[@]}" \
            --run_name "$rname_mala" \
            --pipeline_stage "p4_search_baseline_mala" \
            --train_mode vmc_only \
            --proposal_type mala \
            --tau "$TAU" \
            --train_steps "$TOTAL_STEPS" \
            --output_h5 "${rdir_mala}/run_bundle.h5"

        local p3_cycles=6
        local p3_vmc_per_cyc=$(( TOTAL_STEPS / p3_cycles ))
        local p3_rl_per_cyc=250
        local rname_p3="${SYS_TAG}_02_Baseline_P3_BlockCoupled_cyc${p3_cycles}x${p3_vmc_per_cyc}"
        local rdir_p3="${SYS_OUT}/${rname_p3}"
        launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_p3" "$rdir_p3" \
            "$PYTHON_BIN" -u cli/run_1d.py \
            "${COMMON_ARGS[@]}" \
            --run_name "$rname_p3" \
            --pipeline_stage "p4_search_baseline_p3" \
            --train_mode coupled \
            --proposal_type neural_deepset \
            --proposal_mode residual_mala \
            --rl_algorithm td3 \
            --reward_type "$REWARD_TYPE" \
            --tau "$TAU" \
            --actor_amplitude "$ACTOR_AMP" \
            --penalty_factor "$PENALTY" \
            --sigma_action_scale "$SIG_SCALE" \
            --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
            --staggered_critic_reset \
            --lr_actor "$LR_ACTOR" \
            --lr_critic "$LR_CRITIC" \
            --coupled_training_cycles "$p3_cycles" \
            --vmc_iterations_per_cycle "$p3_vmc_per_cyc" \
            --rl_sweeps_per_cycle "$p3_rl_per_cyc" \
            --train_steps "$TOTAL_STEPS" \
            --output_h5 "${rdir_p3}/run_bundle.h5"
    fi

    # --------------------------------------------------------------------------
    # B. ABLACIÓN DE LOS 4 REGÍMENES DE PIPELINE 4 (`online_coupled`)
    # --------------------------------------------------------------------------
    if [[ "$RUN_REGIME_ABLATION" -eq 1 ]]; then
        # Caso A: Cold-Start desde Paso 1 (N_nodal=0, N_critic=0, N_freeze=0)
        local rname_ca="${SYS_TAG}_03_P4_RegimeA_ColdStart_N0_C0_F0"
        local rdir_ca="${SYS_OUT}/${rname_ca}"
        launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_ca" "$rdir_ca" \
            "$PYTHON_BIN" -u cli/run_1d.py \
            "${COMMON_ARGS[@]}" \
            --run_name "$rname_ca" \
            --pipeline_stage "p4_ablation_regime_A_coldstart" \
            --train_mode online_coupled \
            --proposal_type neural_deepset \
            --proposal_mode residual_mala \
            --rl_algorithm td3 \
            --reward_type "$REWARD_TYPE" \
            --tau "$TAU" \
            --actor_amplitude "$ACTOR_AMP" \
            --penalty_factor "$PENALTY" \
            --sigma_action_scale "$SIG_SCALE" \
            --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
            --use_reward_centering \
            --rl_replay_capacity 25000 \
            --lr_actor "$LR_ACTOR" \
            --lr_critic "$LR_CRITIC" \
            --train_steps "$TOTAL_STEPS" \
            --online_nodal_warmup_steps 0 \
            --online_critic_warmup_steps 0 \
            --online_rl_freeze_tail_steps 0 \
            --online_walkers_to_store 64 \
            --output_h5 "${rdir_ca}/run_bundle.h5"

        # Caso B: Warmup Nodal Ciego sin Pre-entrenamiento de Crítico (N_nodal=2*N_WARM, N_critic=0, N_freeze=N_FREEZE)
        local n_blind=$(( 2 * N_WARM_DEFAULT ))
        local rname_cb="${SYS_TAG}_04_P4_RegimeB_BlindWarmup_N${n_blind}_C0_F${N_FREEZE_DEFAULT}"
        local rdir_cb="${SYS_OUT}/${rname_cb}"
        launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_cb" "$rdir_cb" \
            "$PYTHON_BIN" -u cli/run_1d.py \
            "${COMMON_ARGS[@]}" \
            --run_name "$rname_cb" \
            --pipeline_stage "p4_ablation_regime_B_blind_warmup" \
            --train_mode online_coupled \
            --proposal_type neural_deepset \
            --proposal_mode residual_mala \
            --rl_algorithm td3 \
            --reward_type "$REWARD_TYPE" \
            --tau "$TAU" \
            --actor_amplitude "$ACTOR_AMP" \
            --penalty_factor "$PENALTY" \
            --sigma_action_scale "$SIG_SCALE" \
            --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
            --use_reward_centering \
            --rl_replay_capacity 25000 \
            --lr_actor "$LR_ACTOR" \
            --lr_critic "$LR_CRITIC" \
            --train_steps "$TOTAL_STEPS" \
            --online_nodal_warmup_steps "$n_blind" \
            --online_critic_warmup_steps 0 \
            --online_rl_freeze_tail_steps "$N_FREEZE_DEFAULT" \
            --online_walkers_to_store 64 \
            --output_h5 "${rdir_cb}/run_bundle.h5"

        # Caso C: Warmup Nodal + Crítico Off-Policy, pero SIN Tail Freeze (N_nodal=N_WARM, N_critic=N_WARM, N_freeze=0)
        local rname_cc="${SYS_TAG}_05_P4_RegimeC_NoTailFreeze_N${N_WARM_DEFAULT}_C${N_WARM_DEFAULT}_F0"
        local rdir_cc="${SYS_OUT}/${rname_cc}"
        launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_cc" "$rdir_cc" \
            "$PYTHON_BIN" -u cli/run_1d.py \
            "${COMMON_ARGS[@]}" \
            --run_name "$rname_cc" \
            --pipeline_stage "p4_ablation_regime_C_no_tail_freeze" \
            --train_mode online_coupled \
            --proposal_type neural_deepset \
            --proposal_mode residual_mala \
            --rl_algorithm td3 \
            --reward_type "$REWARD_TYPE" \
            --tau "$TAU" \
            --actor_amplitude "$ACTOR_AMP" \
            --penalty_factor "$PENALTY" \
            --sigma_action_scale "$SIG_SCALE" \
            --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
            --use_reward_centering \
            --rl_replay_capacity 25000 \
            --lr_actor "$LR_ACTOR" \
            --lr_critic "$LR_CRITIC" \
            --train_steps "$TOTAL_STEPS" \
            --online_nodal_warmup_steps "$N_WARM_DEFAULT" \
            --online_critic_warmup_steps "$N_WARM_DEFAULT" \
            --online_rl_freeze_tail_steps 0 \
            --online_walkers_to_store 64 \
            --output_h5 "${rdir_cc}/run_bundle.h5"
    fi

    # --------------------------------------------------------------------------
    # C. BÚSQUEDA DE HIPERPARÁMETROS SOBRE EL RÉGIMEN COMPLETO DE 4 FASES
    # --------------------------------------------------------------------------
    if [[ "$RUN_HYPERPARAM_GRID" -eq 1 ]]; then
        local idx_m=6
        for sched in "${SCHEDULE_GRID[@]}"; do
            IFS=':' read -r n_nodal n_critic n_freeze <<< "$sched"
            for r_cap in "${REPLAY_CAP_GRID[@]}"; do
                for w_store in "${WALKERS_STORE_GRID[@]}"; do
                    for utd in "${UTD_GRID[@]}"; do
                        for amp in "${ACTOR_AMP_GRID[@]}"; do
                            for pen in "${PENALTY_GRID[@]}"; do
                                # En modo fast filtramos combinaciones redundantes para mantener ~12 modelos bien distribuidos
                                if [[ "$SEARCH_PRESET" == "fast" ]]; then
                                    if [[ "$sched" != "${N_WARM_DEFAULT}:${N_WARM_DEFAULT}:${N_FREEZE_DEFAULT}" ]]; then
                                        if [[ "$r_cap" != "25000" || "$w_store" != "64" || "$utd" != "1" || "$amp" != "0.15" ]]; then
                                            continue
                                        fi
                                    fi
                                    if [[ "$utd" == "2" && ( "$r_cap" != "25000" || "$w_store" != "64" ) ]]; then
                                        continue
                                    fi
                                fi
                                local idx_pad
                                idx_pad=$(printf "%02d" "$idx_m")
                                local rname_g="${SYS_TAG}_${idx_pad}_P4_Full4R_N${n_nodal}_C${n_critic}_F${n_freeze}_buf${r_cap}_w${w_store}_utd${utd}_amp${amp}_pen${pen}"
                                local rdir_g="${SYS_OUT}/${rname_g}"
                                launch_search_job "$SYS_TAG" "$CASSCF_NPZ" "$rname_g" "$rdir_g" \
                                    "$PYTHON_BIN" -u cli/run_1d.py \
                                    "${COMMON_ARGS[@]}" \
                                    --run_name "$rname_g" \
                                    --pipeline_stage "p4_grid_full_4regimes" \
                                    --train_mode online_coupled \
                                    --proposal_type neural_deepset \
                                    --proposal_mode residual_mala \
                                    --rl_algorithm td3 \
                                    --reward_type "$REWARD_TYPE" \
                                    --tau "$TAU" \
                                    --actor_amplitude "$amp" \
                                    --penalty_factor "$pen" \
                                    --sigma_action_scale "$SIG_SCALE" \
                                    --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
                                    --use_reward_centering \
                                    --rl_replay_capacity "$r_cap" \
                                    --lr_actor "$LR_ACTOR" \
                                    --lr_critic "$LR_CRITIC" \
                                    --train_steps "$TOTAL_STEPS" \
                                    --online_nodal_warmup_steps "$n_nodal" \
                                    --online_critic_warmup_steps "$n_critic" \
                                    --online_rl_freeze_tail_steps "$n_freeze" \
                                    --online_rl_update_every 1 \
                                    --online_rl_updates_per_step "$utd" \
                                    --online_walkers_to_store "$w_store" \
                                    --output_h5 "${rdir_g}/run_bundle.h5"
                                idx_m=$((idx_m + 1))
                            done
                        done
                    done
                done
            done
        done
    fi

    wait_all_jobs
    rm -rf "$PLOT_LOCK_DIR" 2>/dev/null || true
    compile_system_figures "$SYS_TAG" "$CASSCF_NPZ"
}

for pair in "${SYSTEM_PAIRS[@]}"; do
    read -r N_CUR G_CUR <<< "$pair"
    run_search_for_system "$N_CUR" "$G_CUR"
done

wait_all_jobs

# ==============================================================================
# 6. RESUMEN EJECUTIVO DE LOGS Y RANKING DE HIPERPARÁMETROS EN EL LOG MAESTRO
# ==============================================================================
MASTER_END_EPOCH=$(date +%s)
MASTER_END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_DURATION=$((MASTER_END_EPOCH - MASTER_START_EPOCH))
MASTER_DURATION_FORMATTED=$(format_duration "$MASTER_DURATION")

log_msg ""
log_msg "=============================================================================="
log_msg " RANKING CONSOLIDADO DE LA BÚSQUEDA DE HIPERPARÁMETROS (PIPELINE 4)"
log_msg "=============================================================================="
env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" - "$MASTER_COMP_CSV" "$MASTER_RUNS_CSV" "$CSV_COLLECT_DIR" << 'PYEOF' | tee -a "$MASTER_LOG"
import os, sys
import pandas as pd

comp_csv, runs_csv, collect_dir = sys.argv[1:4]
target = comp_csv if (os.path.isfile(comp_csv) and os.path.getsize(comp_csv) > 0) else runs_csv
if os.path.isfile(target) and os.path.getsize(target) > 0:
    df = pd.read_csv(target)
    if not df.empty:
        sort_cols = [c for c in ["N", "g", "E_mean", "kl_casscf_spin_mean"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)
        cols = [c for c in [
            "label", "run_name", "N", "g", "train_mode",
            "E_mean", "SE_E", "E_var", "delta_E_vs_CASSCF",
            "kl_casscf_spin_mean", "kl_casscf_total_mean",
            "acceptance", "ESJD", "tau_int_E", "G_RL_speedup", "ESS_per_sec",
            "train_wall_time_sec", "wall_time_sec"
        ] if c in df.columns]
        print(df[cols].to_string(index=False))
        ranked_csv = os.path.join(collect_dir, "pipeline4_param_search_ranked_summary.csv")
        df.to_csv(ranked_csv, index=False)
        print(f"\n[+] Tabla de ranking guardada en: {ranked_csv}")
else:
    print("  [Aviso] Aún no se encontraron filas en las tablas CSV maestras.")
PYEOF

log_msg "=============================================================================="
log_msg " BÚSQUEDA DE HIPERPARÁMETROS PIPELINE 4 FINALIZADA"
log_msg " Hora de Inicio      : $MASTER_START_TIME"
log_msg " Hora de Terminación : $MASTER_END_TIME"
log_msg " Tiempo Total Crudo  : ${MASTER_DURATION}s ($MASTER_DURATION_FORMATTED)"
log_msg " CSVs de Figuras     : $CSV_COLLECT_DIR"
log_msg " Figuras por Sistema : $FIG_DIR"
log_msg " Figuras Globales    : $GLOBAL_FIG_DIR"
log_msg " Log Maestro         : $MASTER_LOG"
log_msg "=============================================================================="
