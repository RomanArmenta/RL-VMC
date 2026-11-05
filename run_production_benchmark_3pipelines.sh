#!/usr/bin/env bash
# ==============================================================================
# MASTER PRODUCTION BENCHMARK: PIPELINES 1 & 2 (Y OPCIONAL 3) — BAKER / AYLER
# ==============================================================================
# Implementa las etapas de producción sobre el paquete modular `rl_vmc` y la
# paquetería `uhf1d` (CASSCF multi-determinante 1D) con los hiperparámetros
# óptimos de cruce nodal calibrados en Baker (Wigner_N4_g4.0 y MultiDet_N6_g1.0):
#
#   1. PIPELINE 1 (Producción de Referencias CASSCF en CPU con Base Expandida):
#      Calcula (o reutiliza si ya tiene base >= N_BASIS) la función de onda
#      multiconfiguracional CASSCF para cada sistema (N, omega, g) con
#      g in {0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0} y base expandida (N_basis=36
#      para N=2 y N_basis=48 para N>=4, quad=100) en CPU (CUDA_VISIBLE_DEVICES="").
#      Guarda inmediatamente las figuras diagnósticas de cada CASSCF al terminar.
#
#   2. PIPELINE 2 (Muestreo Puro sobre CASSCF Congelado - MALA vs RL-MALA TD3):
#      Fija exactamente los pesos de `casscf_state.npz` del Pipeline 1 y compara:
#        - MALA estándar (`--proposal_type mala --train_mode eval_only`)
#        - RL-MALA con Actor DeepSet + TD3 (`--proposal_type neural_deepset --train_mode rl_only`)
#      Incorpora la metodología validada de cruce nodal:
#        * Inicialización simétrica en entrenamiento (`--init_mode symmetric_gaussian --init_std 3.0`)
#        * Inicialización polarizada por espín en evaluación (`--eval_init_mode spin_ordered_offset`)
#        * Recompensa CDLB (`--reward_type cdlb --penalty_factor 0.25 --sigma_action_scale 1.1`)
#        * Tasa de aprendizaje constante (`--no-use_lr_scheduler --lr_actor 1e-4 --lr_critic 3e-4`)
#        * 3 cadenas de evaluación x 40,000 pasos (burn_in=200) para medir D_KL^spin y trazas
#      Al terminar cada sistema (N, g), genera en tiempo real:
#        - Las 8 figuras estándar de post-producción (`cli/postprocess.py`)
#        - Las 4 figuras de cruce nodal y trazas de 1 caminante (`cli/evaluate_nodal_crossing_1d.py`)
#        - Las 6 figuras globales de todos los modelos vs N y vs g (`scripts/plot_benchmark_all_models_summary.py`)
#
#   3. PIPELINE 3 (Opcional: Entrenamiento Variacional Slater-Jastrow-Backflow):
#      Desactivado por defecto (`PIPELINES_TO_RUN="1 2"`). Puede activarse con
#      `--run-pipeline3` o `PIPELINES_TO_RUN="1 2 3"`.
#
# Uso en Baker (1 GPU) o Ayler (3 GPUs):
#   chmod +x run_production_benchmark_3pipelines.sh
#   nohup ./run_production_benchmark_3pipelines.sh > master_3pipelines_baker.log 2>&1 &
#
# Re-generar únicamente figuras (ej. en PDF a 600 PPI para artículo):
#   ./run_production_benchmark_3pipelines.sh --postprocess-only --pdf --ppi 600
# ==============================================================================

set -u

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

BASE_OUT_DIR="${BASE_OUT_DIR:-results_production_3pipelines_baker}"
CSV_COLLECT_DIR="${CSV_COLLECT_DIR:-${BASE_OUT_DIR}/results_csv}"
# Por defecto ejecutamos Pipelines 1 y 2 como se solicitó para Baker
PIPELINES_TO_RUN="${PIPELINES_TO_RUN:-1 2}"
N_LIST="${N_LIST:-2 4 6 10}"
# Incluye g=0.1 además de toda la serie de acoplamientos
G_LIST="${G_LIST:-0.1 0.5 1.0 2.0 3.0 5.0 8.0}"
OMEGA="${OMEGA:-0.1}"
GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"
COPY_ADVISOR_CASSCF="${COPY_ADVISOR_CASSCF:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"

# Bases expandidas de CASSCF para garantizar convergencia variacional completa
CASSCF_BASIS_N2="${CASSCF_BASIS_N2:-36}"
CASSCF_BASIS_N4="${CASSCF_BASIS_N4:-48}"
CASSCF_BASIS_N6="${CASSCF_BASIS_N6:-48}"
CASSCF_BASIS_N8="${CASSCF_BASIS_N8:-48}"
CASSCF_BASIS_N10="${CASSCF_BASIS_N10:-48}"
CASSCF_QUAD="${CASSCF_QUAD:-100}"
CASSCF_MAX_ITER="${CASSCF_MAX_ITER:-1000}"
CASSCF_ENERGY_TOL="${CASSCF_ENERGY_TOL:-1e-8}"
CASSCF_GRAD_TOL="${CASSCF_GRAD_TOL:-1e-6}"

# Hiperparámetros óptimos de muestreo y cruce nodal calibrados en Baker
N_WALKERS="${N_WALKERS:-128}"
N_BURN_IN="${N_BURN_IN:-2000}"
RL_ONLY_STEPS="${RL_ONLY_STEPS:-6000}"
VMC_COUPLED_STEPS="${VMC_COUPLED_STEPS:-12000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-200}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
INIT_MODE="${INIT_MODE:-symmetric_gaussian}"
INIT_STD="${INIT_STD:-3.0}"
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"
REWARD_TYPE="${REWARD_TYPE:-cdlb}"
PENALTY_FACTOR="${PENALTY_FACTOR:-0.25}"
SIGMA_ACTION_SCALE="${SIGMA_ACTION_SCALE:-1.1}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-2.0}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-16}"
TOP_K_MODELS="${TOP_K_MODELS:-2}"
WALKER_TRACE_TOP_K="${WALKER_TRACE_TOP_K:-1}"

# Formato y estilo de figuras (por defecto solo PNG para ahorrar espacio)
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TITLE_PREFIX="${TITLE_PREFIX:-}"
CUSTOM_SUPTITLE="${CUSTOM_SUPTITLE:-}"
NO_TITLES="${NO_TITLES:-0}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"
TRANSPARENT_BG="${TRANSPARENT_BG:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --postprocess-only|--postprocess_only|--plots-only) POSTPROCESS_ONLY=1; shift ;;
        --pipelines) PIPELINES_TO_RUN="$2"; shift 2 ;;
        --run-pipeline3|--with-pipeline3) PIPELINES_TO_RUN="1 2 3"; shift ;;
        --n-list|--n_list) N_LIST="$2"; shift 2 ;;
        --g-list|--g_list) G_LIST="$2"; shift 2 ;;
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
        --force-rerun|--force_rerun) FORCE_RERUN=1; shift ;;
        *) BASE_OUT_DIR="$1"; CSV_COLLECT_DIR="${BASE_OUT_DIR}/results_csv"; shift ;;
    esac
done

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

# Estructura de carpetas de salida, figuras y tablas CSV en vivo
P1_DIR="${BASE_OUT_DIR}/pipeline1_casscf"
P2_DIR="${BASE_OUT_DIR}/pipeline2_pure_sampling"
P3_DIR="${BASE_OUT_DIR}/pipeline3_variational_training"
FIG_ROOT_DIR="${BASE_OUT_DIR}/figures"
FIG_P1_DIR="${FIG_ROOT_DIR}/pipeline1_casscf"
FIG_P2_DIR="${FIG_ROOT_DIR}/pipeline2_systems"
FIG_P3_DIR="${FIG_ROOT_DIR}/pipeline3_systems"
FIG_ALL_MODELS_DIR="${FIG_ROOT_DIR}/all_models_vs_N_and_g"
LOG_DIR="${BASE_OUT_DIR}/logs"

LIVE_CSV_P1="${BASE_OUT_DIR}/live_pipeline1_casscf_summary.csv"
LIVE_CSV_P2_RUNS="${BASE_OUT_DIR}/live_pipeline2_sampling_runs.csv"
LIVE_CSV_P2_COMP="${BASE_OUT_DIR}/live_pipeline2_sampling_comparison.csv"
LIVE_CSV_P3_RUNS="${BASE_OUT_DIR}/live_pipeline3_training_runs.csv"
LIVE_CSV_P3_COMP="${BASE_OUT_DIR}/live_pipeline3_training_comparison.csv"
LIVE_CSV_ALL="${BASE_OUT_DIR}/live_all_models_master.csv"

mkdir -p "$P1_DIR" "$P2_DIR" "$P3_DIR" "$FIG_P1_DIR" "$FIG_P2_DIR" "$FIG_P3_DIR" "$FIG_ALL_MODELS_DIR" "$LOG_DIR" "$CSV_COLLECT_DIR"

MASTER_START_EPOCH=$(date +%s)
MASTER_START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_PID=$$

# ------------------------------------------------------------------------------
# Directorio temporal atómico para tracking de trabajos y locks de graficación
# ------------------------------------------------------------------------------
TRACKER_DIR=$(mktemp -d /tmp/rl_vmc_gpu_tracker_XXXXXX 2>/dev/null || mktemp -d -t 'rl_vmc_gpu_tracker')
GLOBAL_PLOT_LOCK="${TRACKER_DIR}/global_plot.lock"

cleanup_on_exit() {
    local exit_status=$?
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

# ------------------------------------------------------------------------------
# Detección automática del Pool de GPUs (1 GPU en Baker, 3 GPUs en Ayler)
# ------------------------------------------------------------------------------
AVAILABLE_GPUS=()
MAX_TOTAL_JOBS=1

discover_gpus() {
    if [ "$GPU_LIST" != "auto" ] && [ -n "$GPU_LIST" ]; then
        read -r -a AVAILABLE_GPUS <<< "$GPU_LIST"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        local detected
        detected=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
        if [ -n "$detected" ]; then
            read -r -a AVAILABLE_GPUS <<< "$detected"
        else
            AVAILABLE_GPUS=(0)
        fi
    else
        AVAILABLE_GPUS=(0)
    fi

    if [ "$MAX_PARALLEL_JOBS" = "auto" ]; then
        MAX_TOTAL_JOBS=$(( ${#AVAILABLE_GPUS[@]} * JOBS_PER_GPU ))
    else
        MAX_TOTAL_JOBS="$MAX_PARALLEL_JOBS"
    fi
    if [ "$MAX_TOTAL_JOBS" -lt 1 ]; then
        MAX_TOTAL_JOBS=1
    fi
}

discover_gpus

echo "=============================================================================="
echo " INICIANDO BENCHMARK DE PRODUCCIÓN (BAKER / AYLER)"
echo " Fecha y Hora         : $MASTER_START_TIME"
echo " PID del Master       : $MASTER_PID"
echo " Directorio raíz      : $BASE_OUT_DIR"
echo " Pipelines activos    : $PIPELINES_TO_RUN (Postprocess-only=$POSTPROCESS_ONLY)"
echo " Sistemas N           : $N_LIST"
echo " Acoplamientos g      : $G_LIST"
echo " Frecuencia omega     : $OMEGA"
echo " Bases CASSCF (N2/N4+): ${CASSCF_BASIS_N2} / ${CASSCF_BASIS_N4} (quad=${CASSCF_QUAD})"
echo " GPUs en Pool         : ${AVAILABLE_GPUS[*]} (Trabajos/GPU=$JOBS_PER_GPU, Total=$MAX_TOTAL_JOBS)"
echo " Muestreo RL-MALA TD3 : reward=${REWARD_TYPE}, sweeps=${RL_ONLY_STEPS}, walkers=${N_WALKERS}"
echo " Evaluación CruceNod  : init=${EVAL_INIT_MODE}, chains=${N_EVAL_CHAINS}, steps=${MEASURE_STEPS}"
echo " Formato de Figuras   : ${FIGURE_FORMAT} @ ${FIGURE_DPI} PPI"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Helpers de formato de tiempo, verificación de archivos y locks
# ------------------------------------------------------------------------------
format_duration() {
    local dur=$1
    if [ "$dur" -lt 0 ]; then dur=0; fi
    printf "%02dh %02dm %02ds" $((dur/3600)) $(( (dur%3600)/60 )) $((dur%60))
}

is_h5_completed() {
    local h5_file="$1"
    if [ -f "$h5_file" ]; then
        local fsize
        fsize=$(wc -c < "$h5_file" 2>/dev/null | tr -d ' ' || echo "0")
        if [ "${fsize:-0}" -gt 10240 ]; then
            return 0
        fi
    fi
    return 1
}

acquire_global_plot_lock() {
    while ! mkdir "$GLOBAL_PLOT_LOCK" 2>/dev/null; do
        sleep 1
    done
}

release_global_plot_lock() {
    rmdir "$GLOBAL_PLOT_LOCK" 2>/dev/null || true
}

update_global_all_models_figures() {
    local reason_msg="$1"
    acquire_global_plot_lock
    local target_csv=""
    if [ -f "$LIVE_CSV_ALL" ] && [ -s "$LIVE_CSV_ALL" ]; then
        target_csv="$LIVE_CSV_ALL"
    elif [ -f "$LIVE_CSV_P2_COMP" ] && [ -s "$LIVE_CSV_P2_COMP" ]; then
        target_csv="$LIVE_CSV_P2_COMP"
    fi

    if [ -n "$target_csv" ]; then
        echo "  [Global Plots] Actualizando figuras globales vs N y g tras completar ${reason_msg}..."
        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u scripts/plot_benchmark_all_models_summary.py \
            --csv "$target_csv" \
            --out_dir "$FIG_ALL_MODELS_DIR" \
            "${PLOT_STYLE_ARGS[@]}" || true
    fi
    release_global_plot_lock
}

run_logged_command() {
    local log_prefix="$1"
    local log_file="$2"
    shift 2
    local cmd=("$@")

    local start_epoch
    start_epoch=$(date +%s)
    local start_time
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    mkdir -p "$(dirname "$log_file")"
    : > "$log_file"

    "${cmd[@]}" >> "$log_file" 2>&1 &
    local child_pid=$!

    local banner_start="==============================================================================
>> $log_prefix INICIANDO
   PID                : $child_pid
   Hora de Inicio     : $start_time
   Comando / Script   : ${cmd[*]}
   Archivo de Log     : $log_file
=============================================================================="

    echo "$banner_start"

    wait "$child_pid"
    local exit_code=$?

    local end_epoch
    end_epoch=$(date +%s)
    local end_time
    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    local duration=$((end_epoch - start_epoch))
    local duration_formatted
    duration_formatted=$(format_duration "$duration")

    local banner_end="------------------------------------------------------------------------------
<< $log_prefix TERMINADO (Código de salida: $exit_code)
   PID                : $child_pid
   Hora de Inicio     : $start_time
   Hora de Terminación: $end_time
   Tiempo transcurrido: ${duration}s ($duration_formatted)
   Archivo de Log     : $log_file
=============================================================================="

    echo "$banner_end"
    echo "$banner_end" >> "$log_file"

    return $exit_code
}

run_logged_step() {
    local step_name="$1"
    shift
    local cmd=("$@")

    local start_epoch
    start_epoch=$(date +%s)
    local start_time
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    "${cmd[@]}" &
    local child_pid=$!

    echo "=============================================================================="
    echo ">> [$step_name] INICIANDO"
    echo "   PID                : $child_pid"
    echo "   Hora de Inicio     : $start_time"
    echo "   Comando / Script   : ${cmd[*]}"
    echo "=============================================================================="

    wait "$child_pid"
    local exit_code=$?

    local end_epoch
    end_epoch=$(date +%s)
    local end_time
    end_time=$(date "+%Y-%m-%d %H:%M:%S")
    local duration=$((end_epoch - start_epoch))
    local duration_formatted
    duration_formatted=$(format_duration "$duration")

    echo "------------------------------------------------------------------------------"
    echo "<< [$step_name] TERMINADO (Código de salida: $exit_code)"
    echo "   PID                : $child_pid"
    echo "   Hora de Inicio     : $start_time"
    echo "   Hora de Terminación: $end_time"
    echo "   Tiempo transcurrido: ${duration}s ($duration_formatted)"
    echo "=============================================================================="

    return $exit_code
}

# ------------------------------------------------------------------------------
# Despachador Dinámico de GPUs para Pipelines 2 y 3
# ------------------------------------------------------------------------------
cleanup_finished_jobs() {
    local stage_name="$1"
    for f in "$TRACKER_DIR"/job_*.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local exit_code=$?
            local gpu sys start_epoch start_time script_name sys_log_file
            # shellcheck disable=SC1090
            source "$f"
            local end_epoch
            end_epoch=$(date +%s)
            local end_time
            end_time=$(date "+%Y-%m-%d %H:%M:%S")
            local duration=$((end_epoch - start_epoch))
            local duration_formatted
            duration_formatted=$(format_duration "$duration")

            {
                echo "------------------------------------------------------------------------------"
                echo "<< [$stage_name | TRABAJO FINALIZADO] Sistema: $sys"
                echo "   PID del Worker     : $pid (Código: $exit_code)"
                echo "   GPU Liberada       : $gpu"
                echo "   Script / Tarea     : $script_name"
                echo "   Hora de Inicio     : $start_time"
                echo "   Hora de Terminación: $end_time"
                echo "   Tiempo transcurrido: ${duration}s ($duration_formatted)"
                echo "   Log del Sistema    : ${LOG_DIR}/${sys_log_file}"
                echo "   Figuras Globales   : ${FIG_ALL_MODELS_DIR}"
                echo "=============================================================================="
            } >&2
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

wait_for_free_gpu_slot() {
    local stage_name="$1"
    while true; do
        cleanup_finished_jobs "$stage_name"
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
                echo "$best_gpu"
                return 0
            fi
        fi
        sleep 2
    done
}

wait_all_stage_jobs() {
    local stage_name="$1"
    while true; do
        cleanup_finished_jobs "$stage_name"
        local total_active
        total_active=$(get_total_active_jobs)
        if [ "$total_active" -eq 0 ]; then
            break
        fi
        sleep 2
    done
}

# ------------------------------------------------------------------------------
# Parámetros de Base Expandida para CASSCF por sistema físico (N, g)
# ------------------------------------------------------------------------------
get_system_casscf_params() {
    local N=$1
    local G=$2
    local n_core=0
    local n_act=4
    local n_basis="$CASSCF_BASIS_N4"
    local quad="$CASSCF_QUAD"
    local soft="0.1"

    if [ "$N" -eq 2 ]; then
        n_core=0; n_act=4; n_basis="$CASSCF_BASIS_N2"; quad="$CASSCF_QUAD"; soft="0.1"
    elif [ "$N" -eq 4 ]; then
        n_core=0; n_act=4; n_basis="$CASSCF_BASIS_N4"; quad="$CASSCF_QUAD"; soft="0.1"
    elif [ "$N" -eq 6 ]; then
        n_core=1; n_act=4; n_basis="$CASSCF_BASIS_N6"; quad="$CASSCF_QUAD"; soft="0.1"
    elif [ "$N" -eq 8 ]; then
        n_core=2; n_act=4; n_basis="$CASSCF_BASIS_N8"; quad="$CASSCF_QUAD"; soft="0.1"
    elif [ "$N" -eq 10 ]; then
        n_core=4; n_act=4; n_basis="$CASSCF_BASIS_N10"; quad="$CASSCF_QUAD"; soft="0.25"
    else
        n_core=$(( (N / 2) - 2 ))
        if [ "$n_core" -lt 0 ]; then n_core=0; fi
        n_act=4; n_basis="$CASSCF_BASIS_N4"; quad="$CASSCF_QUAD"; soft="0.1"
    fi
    echo "$n_core $n_act $n_basis $quad $soft"
}

# ------------------------------------------------------------------------------
# Hiperparámetros Óptimos de Muestreo Puro (MALA vs RL-MALA) calibrados en Baker:
# Retorna: "TAU ACTOR_AMP PENALTY SIGMA_SCALE"
# ------------------------------------------------------------------------------
get_system_p2_params() {
    local N=$1
    local G=$2
    local tau="0.10"
    local actor_amp="0.15"
    local penalty="$PENALTY_FACTOR"
    local sig_scale="$SIGMA_ACTION_SCALE"

    local is_strong_g
    is_strong_g=$(awk -v g="$G" 'BEGIN { print (g >= 2.5) ? 1 : 0 }')

    if [ "$N" -le 2 ]; then
        tau="0.10"
        if [ "$is_strong_g" -eq 1 ]; then
            actor_amp="0.12"
        else
            actor_amp="0.15"
        fi
    elif [ "$N" -le 4 ]; then
        # Calibrado directamente de Wigner_N4_g4.0: tau=0.10, amp=0.10 (g>=2.5) o 0.15 (g<2.5), pen=0.25, sig=1.1
        tau="0.10"
        if [ "$is_strong_g" -eq 1 ]; then
            actor_amp="0.10"
        else
            actor_amp="0.15"
        fi
    elif [ "$N" -le 6 ]; then
        # Calibrado directamente de MultiDet_N6_g1.0: tau=0.10, amp=0.15 (g<2.5) y tau=0.08, amp=0.12 (g>=2.5)
        if [ "$is_strong_g" -eq 1 ]; then
            tau="0.08"
            actor_amp="0.12"
        else
            tau="0.10"
            actor_amp="0.15"
        fi
    else
        # N=8, 10: movimiento simultáneo de 8-10 electrones requiere tau=0.06 para mantener aceptación ~50%
        tau="0.06"
        if [ "$is_strong_g" -eq 1 ]; then
            actor_amp="0.10"
        else
            actor_amp="0.12"
        fi
    fi

    echo "$tau $actor_amp $penalty $sig_scale"
}

# ==============================================================================
# PIPELINE 1: PRODUCCIÓN DE REFERENCIAS EXACTAS CASSCF (CPU - UNO POR UNO)
# ==============================================================================
run_pipeline_1() {
    if [ "$POSTPROCESS_ONLY" -eq 1 ]; then
        echo " [PIPELINE 1] Modo --postprocess-only activo: omitiendo cálculo CASSCF."
        return 0
    fi
    echo ""
    echo "=============================================================================="
    echo " [PIPELINE 1] Generando referencias CASSCF 1D con base expandida (uhf1d)..."
    echo " Modo de ejecución : Secuencial, uno por uno en CPU (CUDA_VISIBLE_DEVICES=\"\")"
    echo " Iteraciones máx   : $CASSCF_MAX_ITER (tol_E=$CASSCF_ENERGY_TOL, tol_grad=$CASSCF_GRAD_TOL)"
    echo " Tabla CSV en vivo : $LIVE_CSV_P1 (espejada en $CSV_COLLECT_DIR)"
    echo "=============================================================================="

    for N in $N_LIST; do
        local N_UP=$((N / 2))
        local N_DN=$((N - N_UP))
        for G in $G_LIST; do
            read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local OUT_CASSCF="${P1_DIR}/${SYS_TAG}"
            local FIG_SYS_P1="${FIG_P1_DIR}/${SYS_TAG}"
            local SYS_LOG="${LOG_DIR}/p1_casscf_${SYS_TAG}.log"

            local EXTRA_FLAGS="--reuse_existing"
            if [ "$FORCE_RERUN" -eq 1 ]; then
                EXTRA_FLAGS=""
            fi
            if [ "$COPY_ADVISOR_CASSCF" -eq 1 ]; then
                EXTRA_FLAGS="$EXTRA_FLAGS --copy_from_advisor_if_available"
            fi

            run_logged_command \
                "[PIPELINE 1 | CASSCF (b=${NBASIS}) | ${SYS_TAG}]" \
                "$SYS_LOG" \
                env CUDA_VISIBLE_DEVICES="" python3 -u cli/run_casscf_1d.py \
                    --n_elec "$N" \
                    --n_up "$N_UP" \
                    --n_dn "$N_DN" \
                    --omega "$OMEGA" \
                    --g "$G" \
                    --softening "$SOFT" \
                    --n_core "$NCORE" \
                    --n_active_orbitals "$NACT" \
                    --n_basis "$NBASIS" \
                    --quadrature_order "$QUAD" \
                    --casscf_max_iter "$CASSCF_MAX_ITER" \
                    --casscf_energy_tol "$CASSCF_ENERGY_TOL" \
                    --casscf_grad_tol "$CASSCF_GRAD_TOL" \
                    --outdir "$OUT_CASSCF" \
                    --master_csv "$LIVE_CSV_P1" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    $EXTRA_FLAGS

            # Copiar inmediatamente las figuras diagnósticas del CASSCF al directorio central de figuras
            mkdir -p "$FIG_SYS_P1"
            for f_img in "$OUT_CASSCF"/*.png "$OUT_CASSCF"/*.pdf "$OUT_CASSCF"/summary*.json; do
                [ -e "$f_img" ] && cp -f "$f_img" "$FIG_SYS_P1/" 2>/dev/null || true
            done
        done
    done
    echo ""
    echo " [PIPELINE 1] Todos los cálculos CASSCF en CPU han terminado. Resumen en: $LIVE_CSV_P1"
}

# ==============================================================================
# PIPELINE 2: MUESTREO PURO SOBRE CASSCF CONGELADO (MALA vs RL-MALA TD3)
# ==============================================================================
run_sistema_pipeline_2() {
    local GPU_ID=$1
    local N=$2
    local G=$3
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
    read -r TAU ACTOR_AMP PENALTY SIG_SCALE <<< "$(get_system_p2_params "$N" "$G")"

    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local CASSCF_NPZ="${P1_DIR}/${SYS_TAG}/casscf_state.npz"
    local SYS_OUT="${P2_DIR}/${SYS_TAG}"
    local MALA_DIR="${SYS_OUT}/${SYS_TAG}_MALA_tau${TAU}"
    local RL_DIR="${SYS_OUT}/${SYS_TAG}_RL_MALA_tau${TAU}_amp${ACTOR_AMP}_pen${PENALTY}_sig${SIG_SCALE}"
    local COMP_DIR="${SYS_OUT}/postprocess_comparison"
    local FIG_SYS_P2="${FIG_P2_DIR}/${SYS_TAG}"
    local MALA_H5="${MALA_DIR}/run_bundle.h5"
    local RL_H5="${RL_DIR}/run_bundle.h5"

    # Compatibilidad con nombres previos si ya existían corridas anteriores
    if [ -f "${SYS_OUT}/MALA_FixedCASSCF/run_bundle.h5" ] && ! is_h5_completed "$MALA_H5"; then
        MALA_DIR="${SYS_OUT}/MALA_FixedCASSCF"
        MALA_H5="${MALA_DIR}/run_bundle.h5"
    fi
    if [ -f "${SYS_OUT}/RL_TD3_FixedCASSCF/run_bundle.h5" ] && ! is_h5_completed "$RL_H5"; then
        RL_DIR="${SYS_OUT}/RL_TD3_FixedCASSCF"
        RL_H5="${RL_DIR}/run_bundle.h5"
    fi

    mkdir -p "$MALA_DIR" "$RL_DIR" "$COMP_DIR" "$FIG_SYS_P2"

    if [ "$POSTPROCESS_ONLY" -eq 0 ]; then
        if [ ! -f "$CASSCF_NPZ" ]; then
            echo "  [Aviso Pipeline 2] No se encontró $CASSCF_NPZ para ${SYS_TAG}. Ejecutando CASSCF en CPU primero..."
            run_logged_step "[Pipeline 2 | Fallback CASSCF CPU | ${SYS_TAG}]" \
                env CUDA_VISIBLE_DEVICES="" python3 -u cli/run_casscf_1d.py \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFT" \
                    --n_core "$NCORE" --n_active_orbitals "$NACT" \
                    --n_basis "$NBASIS" --quadrature_order "$QUAD" \
                    --casscf_max_iter "$CASSCF_MAX_ITER" \
                    --casscf_energy_tol "$CASSCF_ENERGY_TOL" \
                    --casscf_grad_tol "$CASSCF_GRAD_TOL" \
                    --outdir "${P1_DIR}/${SYS_TAG}" \
                    --master_csv "$LIVE_CSV_P1" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    --reuse_existing --copy_from_advisor_if_available
        fi

        if [ ! -f "$CASSCF_NPZ" ]; then
            echo "  [Error Pipeline 2] No se pudo generar $CASSCF_NPZ para ${SYS_TAG}. Omitiendo este sistema en Pipeline 2."
            return 1
        fi

        # 2A. Muestreo puro con MALA estándar sobre CASSCF congelado (desde spin_ordered_offset)
        if [ "$FORCE_RERUN" -eq 1 ] || ! is_h5_completed "$MALA_H5"; then
            run_logged_step "[Pipeline 2 | MALA (tau=${TAU}) | ${SYS_TAG} | GPU ${GPU_ID}]" \
                env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                    --run_name "P2_MALA_FixedCASSCF_${SYS_TAG}" \
                    --pipeline_stage "pipeline2_pure_sampling" \
                    --train_mode eval_only \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFT" \
                    --ansatz_type casscf_npz \
                    --casscf_npz_path "$CASSCF_NPZ" \
                    --no-casscf_trainable \
                    --proposal_type mala \
                    --tau "$TAU" \
                    --batch_size "$N_WALKERS" \
                    --init_mode "$INIT_MODE" \
                    --init_std "$INIT_STD" \
                    --eval_init_mode "$EVAL_INIT_MODE" \
                    --n_eval_chains "$N_EVAL_CHAINS" \
                    --eval_burn_in "$EVAL_BURN_IN" \
                    --measure_steps "$MEASURE_STEPS" \
                    --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                    --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                    --output_h5 "$MALA_H5" \
                    --reference_casscf_npz "$CASSCF_NPZ" \
                    --master_csv "$LIVE_CSV_P2_RUNS" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    --export_legacy_files
        else
            echo "  [SKIP] MALA ya completado para ${SYS_TAG}: $MALA_H5"
        fi

        # 2B. Entrenamiento del Actor RL-MALA (TD3 DeepSet + CDLB) sobre CASSCF congelado + Evaluación de Cruce Nodal
        if [ "$FORCE_RERUN" -eq 1 ] || ! is_h5_completed "$RL_H5"; then
            run_logged_step "[Pipeline 2 | RL-TD3 (tau=${TAU}, amp=${ACTOR_AMP}, pen=${PENALTY}) | ${SYS_TAG} | GPU ${GPU_ID}]" \
                env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                    --run_name "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
                    --pipeline_stage "pipeline2_pure_sampling" \
                    --train_mode rl_only \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFT" \
                    --ansatz_type casscf_npz \
                    --casscf_npz_path "$CASSCF_NPZ" \
                    --no-casscf_trainable \
                    --proposal_type neural_deepset \
                    --proposal_mode residual_mala \
                    --rl_algorithm td3 \
                    --reward_type "$REWARD_TYPE" \
                    --actor_amplitude "$ACTOR_AMP" \
                    --penalty_factor "$PENALTY" \
                    --sigma_action_scale "$SIG_SCALE" \
                    --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
                    --no-use_lr_scheduler \
                    --lr_actor "$LR_ACTOR" \
                    --lr_critic "$LR_CRITIC" \
                    --n_ter "$N_BURN_IN" \
                    --train_steps "$RL_ONLY_STEPS" \
                    --tau "$TAU" \
                    --batch_size "$N_WALKERS" \
                    --init_mode "$INIT_MODE" \
                    --init_std "$INIT_STD" \
                    --eval_init_mode "$EVAL_INIT_MODE" \
                    --n_eval_chains "$N_EVAL_CHAINS" \
                    --eval_burn_in "$EVAL_BURN_IN" \
                    --measure_steps "$MEASURE_STEPS" \
                    --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                    --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                    --output_h5 "$RL_H5" \
                    --reference_casscf_npz "$CASSCF_NPZ" \
                    --master_csv "$LIVE_CSV_P2_RUNS" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    --export_legacy_files
        else
            echo "  [SKIP] RL-MALA TD3 ya completado para ${SYS_TAG}: $RL_H5"
        fi
    fi

    # 2C. Post-procesamiento inmediato por sistema (8 figuras estándar + 4 figuras de cruce nodal + actualización global vs N y g)
    if is_h5_completed "$MALA_H5" && is_h5_completed "$RL_H5"; then
        run_logged_step "[Pipeline 2 | Postprocess 8 Figuras | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u cli/postprocess.py \
                --inputs "$MALA_H5" "$RL_H5" \
                --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
                --casscf_npz_path "$CASSCF_NPZ" \
                --out_dir "$COMP_DIR" \
                --system_tag "P2_${SYS_TAG}" \
                --top_k_models "$TOP_K_MODELS" \
                --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
                --master_comparison_csv "$LIVE_CSV_P2_COMP" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                "${PLOT_STYLE_ARGS[@]}"

        # Sincronizar también en LIVE_CSV_ALL para las figuras globales
        acquire_global_plot_lock
        cp -f "$LIVE_CSV_P2_COMP" "$LIVE_CSV_ALL" 2>/dev/null || true
        if [ -f "$LIVE_CSV_P3_COMP" ] && [ -s "$LIVE_CSV_P3_COMP" ]; then
            tail -n +2 "$LIVE_CSV_P3_COMP" >> "$LIVE_CSV_ALL" 2>/dev/null || true
        fi
        cp -f "$LIVE_CSV_ALL" "${CSV_COLLECT_DIR}/live_all_models_master.csv" 2>/dev/null || true
        release_global_plot_lock

        # Ejecutar evaluación dedicada de Cruce Nodal y Trazas de 1 Caminante (4 figuras de artículo)
        run_logged_step "[Pipeline 2 | Nodal Crossing & Walker Traces | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u cli/evaluate_nodal_crossing_1d.py \
                --inputs "$MALA_H5" "$RL_H5" \
                --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
                --casscf_npz_path "$CASSCF_NPZ" \
                --system_tag "$SYS_TAG" \
                --out_dir "$FIG_SYS_P2" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --top_k_models "$TOP_K_MODELS" \
                --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
                "${PLOT_STYLE_ARGS[@]}"

        # Copiar las 8 figuras estándar a la carpeta central de figuras del sistema
        for f_out in "$COMP_DIR"/*.png "$COMP_DIR"/*.pdf "$COMP_DIR"/*.csv "$COMP_DIR"/*.json; do
            [ -e "$f_out" ] && cp -f "$f_out" "$FIG_SYS_P2/" 2>/dev/null || true
        done

        # Actualizar en tiempo real las figuras globales de TODOS los modelos respecto a N y a g
        update_global_all_models_figures "P2_${SYS_TAG}"

        echo "  <- [Pipeline 2 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Figuras del sistema en: ${FIG_SYS_P2} | Globales en: ${FIG_ALL_MODELS_DIR}"
    else
        echo "  [Aviso Pipeline 2] No se encontraron ambos run_bundle.h5 completos para ${SYS_TAG}. Post-procesamiento omitido."
    fi
}

run_pipeline_2() {
    echo ""
    echo "=============================================================================="
    echo " [PIPELINE 2] Muestreo Puro sobre CASSCF Congelado (MALA vs RL-MALA TD3)"
    echo " Despacho dinámico de GPUs: ${AVAILABLE_GPUS[*]} (Límite: $MAX_TOTAL_JOBS trabajos paralelos)"
    echo " Tablas CSV en vivo       : $LIVE_CSV_P2_RUNS | $LIVE_CSV_P2_COMP"
    echo " Figuras en tiempo real   : $FIG_P2_DIR | $FIG_ALL_MODELS_DIR"
    echo "=============================================================================="

    for N in $N_LIST; do
        for G in $G_LIST; do
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local SYS_LOG="p2_sampling_${SYS_TAG}.log"
            local GPU_ID
            GPU_ID=$(wait_for_free_gpu_slot "PIPELINE 2")

            local start_epoch
            start_epoch=$(date +%s)
            local start_time
            start_time=$(date "+%Y-%m-%d %H:%M:%S")

            run_sistema_pipeline_2 "$GPU_ID" "$N" "$G" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            local worker_pid=$!

            cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$SYS_TAG"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_sistema_pipeline_2 GPU=$GPU_ID N=$N G=$G"
sys_log_file="$SYS_LOG"
EOF

            echo "=============================================================================="
            echo ">> [PIPELINE 2 | TRABAJO ASIGNADO] Sistema: ${SYS_TAG}"
            echo "   PID del Worker     : $worker_pid"
            echo "   GPU Asignada       : $GPU_ID"
            echo "   Hora de Inicio     : $start_time"
            echo "   Script / Tarea     : run_sistema_pipeline_2 GPU=$GPU_ID N=$N G=$G"
            echo "   Log del Sistema    : ${LOG_DIR}/${SYS_LOG}"
            echo "=============================================================================="
        done
    done

    wait_all_stage_jobs "PIPELINE 2"
    update_global_all_models_figures "FINAL_PIPELINE_2"
    echo ""
    echo " [PIPELINE 2] Todos los sistemas han terminado exitosamente."
}

# ==============================================================================
# PIPELINE 3: ENTRENAMIENTO VARIACIONAL SLATER-JASTROW-BACKFLOW (OPCIONAL)
# ==============================================================================
run_sistema_pipeline_3() {
    local GPU_ID=$1
    local N=$2
    local G=$3
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
    read -r TAU ACTOR_AMP PENALTY SIG_SCALE <<< "$(get_system_p2_params "$N" "$G")"

    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local CASSCF_NPZ="${P1_DIR}/${SYS_TAG}/casscf_state.npz"
    local SYS_OUT="${P3_DIR}/${SYS_TAG}"
    local VMC_MALA_DIR="${SYS_OUT}/VMC_MALA_SJB"
    local COUPLED_RL_DIR="${SYS_OUT}/Coupled_RL_TD3_SJB"
    local COMP_DIR="${SYS_OUT}/postprocess_comparison"
    local FIG_SYS_P3="${FIG_P3_DIR}/${SYS_TAG}"
    local VMC_MALA_H5="${VMC_MALA_DIR}/run_bundle.h5"
    local COUPLED_RL_H5="${COUPLED_RL_DIR}/run_bundle.h5"

    mkdir -p "$VMC_MALA_DIR" "$COUPLED_RL_DIR" "$COMP_DIR" "$FIG_SYS_P3"

    if [ "$POSTPROCESS_ONLY" -eq 0 ]; then
        if [ "$FORCE_RERUN" -eq 1 ] || ! is_h5_completed "$VMC_MALA_H5"; then
            run_logged_step "[Pipeline 3 | VMC-MALA | ${SYS_TAG} | GPU ${GPU_ID}]" \
                env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                    --run_name "P3_VMC_MALA_SJB_${SYS_TAG}" \
                    --pipeline_stage "pipeline3_variational_training" \
                    --train_mode vmc_only \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFT" \
                    --ansatz_type slater_jastrow_1d \
                    --orbital_model backflow_ho \
                    --correlation_model spin_rbf_jastrow \
                    --proposal_type mala \
                    --train_steps "$VMC_COUPLED_STEPS" \
                    --tau "$TAU" \
                    --batch_size "$N_WALKERS" \
                    --init_mode "$INIT_MODE" \
                    --init_std "$INIT_STD" \
                    --eval_init_mode "$EVAL_INIT_MODE" \
                    --n_eval_chains "$N_EVAL_CHAINS" \
                    --eval_burn_in "$EVAL_BURN_IN" \
                    --measure_steps "$MEASURE_STEPS" \
                    --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                    --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                    --output_h5 "$VMC_MALA_H5" \
                    --reference_casscf_npz "$CASSCF_NPZ" \
                    --master_csv "$LIVE_CSV_P3_RUNS" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    --export_legacy_files
        fi

        if [ "$FORCE_RERUN" -eq 1 ] || ! is_h5_completed "$COUPLED_RL_H5"; then
            run_logged_step "[Pipeline 3 | Coupled VMC+RL | ${SYS_TAG} | GPU ${GPU_ID}]" \
                env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                    --run_name "P3_Coupled_RL_TD3_SJB_${SYS_TAG}" \
                    --pipeline_stage "pipeline3_variational_training" \
                    --train_mode coupled \
                    --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                    --omega "$OMEGA" --g "$G" --softening "$SOFT" \
                    --ansatz_type slater_jastrow_1d \
                    --orbital_model backflow_ho \
                    --correlation_model spin_rbf_jastrow \
                    --proposal_type neural_deepset \
                    --proposal_mode residual_mala \
                    --rl_algorithm td3 \
                    --reward_type "$REWARD_TYPE" \
                    --actor_amplitude "$ACTOR_AMP" \
                    --penalty_factor "$PENALTY" \
                    --sigma_action_scale "$SIG_SCALE" \
                    --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
                    --lr_actor "$LR_ACTOR" \
                    --lr_critic "$LR_CRITIC" \
                    --train_steps "$VMC_COUPLED_STEPS" \
                    --tau "$TAU" \
                    --batch_size "$N_WALKERS" \
                    --init_mode "$INIT_MODE" \
                    --init_std "$INIT_STD" \
                    --eval_init_mode "$EVAL_INIT_MODE" \
                    --n_eval_chains "$N_EVAL_CHAINS" \
                    --eval_burn_in "$EVAL_BURN_IN" \
                    --measure_steps "$MEASURE_STEPS" \
                    --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                    --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                    --output_h5 "$COUPLED_RL_H5" \
                    --reference_casscf_npz "$CASSCF_NPZ" \
                    --master_csv "$LIVE_CSV_P3_RUNS" \
                    --csv_collect_dir "$CSV_COLLECT_DIR" \
                    --export_legacy_files
        fi
    fi

    if is_h5_completed "$VMC_MALA_H5" && is_h5_completed "$COUPLED_RL_H5"; then
        local EXTRA_CASSCF_FLAG=()
        if [ -f "$CASSCF_NPZ" ]; then
            EXTRA_CASSCF_FLAG=(--casscf_npz_path "$CASSCF_NPZ")
        fi

        run_logged_step "[Pipeline 3 | Postprocess | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u cli/postprocess.py \
                --inputs "$VMC_MALA_H5" "$COUPLED_RL_H5" \
                --labels "P3_VMC_MALA_SJB_${SYS_TAG}" "P3_Coupled_RL_TD3_SJB_${SYS_TAG}" \
                "${EXTRA_CASSCF_FLAG[@]}" \
                --out_dir "$COMP_DIR" \
                --system_tag "P3_${SYS_TAG}" \
                --master_comparison_csv "$LIVE_CSV_P3_COMP" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                "${PLOT_STYLE_ARGS[@]}"

        acquire_global_plot_lock
        if [ -f "$LIVE_CSV_P2_COMP" ] && [ -s "$LIVE_CSV_P2_COMP" ]; then
            cp -f "$LIVE_CSV_P2_COMP" "$LIVE_CSV_ALL" 2>/dev/null || true
            tail -n +2 "$LIVE_CSV_P3_COMP" >> "$LIVE_CSV_ALL" 2>/dev/null || true
        else
            cp -f "$LIVE_CSV_P3_COMP" "$LIVE_CSV_ALL" 2>/dev/null || true
        fi
        cp -f "$LIVE_CSV_ALL" "${CSV_COLLECT_DIR}/live_all_models_master.csv" 2>/dev/null || true
        release_global_plot_lock

        for f_out in "$COMP_DIR"/*.png "$COMP_DIR"/*.pdf "$COMP_DIR"/*.csv "$COMP_DIR"/*.json; do
            [ -e "$f_out" ] && cp -f "$f_out" "$FIG_SYS_P3/" 2>/dev/null || true
        done

        update_global_all_models_figures "P3_${SYS_TAG}"
        echo "  <- [Pipeline 3 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Gráficas en: ${FIG_SYS_P3}"
    else
        echo "  [Aviso Pipeline 3] No se encontraron ambos run_bundle.h5 para ${SYS_TAG}. Post-procesamiento omitido."
    fi
}

run_pipeline_3() {
    echo ""
    echo "=============================================================================="
    echo " [PIPELINE 3] Entrenamiento Variacional SJB (VMC+MALA vs Coupled VMC+RL-MALA)"
    echo " Despacho dinámico de GPUs: ${AVAILABLE_GPUS[*]} (Límite: $MAX_TOTAL_JOBS trabajos paralelos)"
    echo " Tablas CSV en vivo: $LIVE_CSV_P3_RUNS | $LIVE_CSV_P3_COMP"
    echo "=============================================================================="

    for N in $N_LIST; do
        for G in $G_LIST; do
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local SYS_LOG="p3_training_${SYS_TAG}.log"
            local GPU_ID
            GPU_ID=$(wait_for_free_gpu_slot "PIPELINE 3")

            local start_epoch
            start_epoch=$(date +%s)
            local start_time
            start_time=$(date "+%Y-%m-%d %H:%M:%S")

            run_sistema_pipeline_3 "$GPU_ID" "$N" "$G" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            local worker_pid=$!

            cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$SYS_TAG"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_sistema_pipeline_3 GPU=$GPU_ID N=$N G=$G"
sys_log_file="$SYS_LOG"
EOF

            echo "=============================================================================="
            echo ">> [PIPELINE 3 | TRABAJO ASIGNADO] Sistema: ${SYS_TAG}"
            echo "   PID del Worker     : $worker_pid"
            echo "   GPU Asignada       : $GPU_ID"
            echo "   Hora de Inicio     : $start_time"
            echo "   Script / Tarea     : run_sistema_pipeline_3 GPU=$GPU_ID N=$N G=$G"
            echo "   Log del Sistema    : ${LOG_DIR}/${SYS_LOG}"
            echo "=============================================================================="
        done
    done

    wait_all_stage_jobs "PIPELINE 3"
    update_global_all_models_figures "FINAL_PIPELINE_3"
    echo ""
    echo " [PIPELINE 3] Todos los sistemas han terminado exitosamente."
}

# ==============================================================================
# EJECUCIÓN SECUENCIAL DE LOS PIPELINES SELECCIONADOS
# ==============================================================================
for P_STAGE in $PIPELINES_TO_RUN; do
    if [ "$P_STAGE" = "1" ]; then
        run_pipeline_1
    elif [ "$P_STAGE" = "2" ]; then
        run_pipeline_2
    elif [ "$P_STAGE" = "3" ]; then
        run_pipeline_3
    fi
done

MASTER_END_EPOCH=$(date +%s)
MASTER_END_TIME=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_DURATION=$((MASTER_END_EPOCH - MASTER_START_EPOCH))
MASTER_DURATION_FORMATTED=$(format_duration "$MASTER_DURATION")

echo ""
echo "=============================================================================="
echo " BENCHMARK DE PRODUCCIÓN FINALIZADO EXITOSAMENTE"
echo " PID del Master      : $MASTER_PID"
echo " Hora de Inicio      : $MASTER_START_TIME"
echo " Hora de Terminación : $MASTER_END_TIME"
echo " Tiempo Total        : ${MASTER_DURATION}s ($MASTER_DURATION_FORMATTED)"
echo " Tablas CSV maestras generadas en vivo:"
echo "   - Pipeline 1 (CASSCF)       : $LIVE_CSV_P1"
echo "   - Pipeline 2 (Muestreo)     : $LIVE_CSV_P2_COMP"
echo "   - Tabla global consolidada  : $LIVE_CSV_ALL"
echo " Directorios de figuras en tiempo real:"
echo "   - Figuras CASSCF (P1)       : $FIG_P1_DIR"
echo "   - Figuras por sistema (P2)  : $FIG_P2_DIR"
echo "   - Figuras globales vs N y g : $FIG_ALL_MODELS_DIR"
echo "=============================================================================="
