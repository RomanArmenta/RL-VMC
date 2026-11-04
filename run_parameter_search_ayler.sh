#!/usr/bin/env bash
# ==============================================================================
# run_parameter_search_ayler.sh
# Parameter Search Automatizado con Sistema de Colas Dinámico para GPUs
# Sistema Objetivo: CASSCF 20 Determinantes (Base 256), N=6, w=0.1, g=1.0, a=0.1
#
# Características Principales:
#   1. Sistema de colas dinámico idéntico a los benchmarks (despacho asíncrono
#      de slots libres en GPU, tracking con archivos .meta, control de concurrencia).
#   2. Logs en TIEMPO REAL en la carpeta logs/ (PYTHONUNBUFFERED=1, python3 -u,
#      redirección directa de cada worker a su archivo .log sin buffering).
#   3. FASE 1: Muestreo PURO solamente (barrido 1D de MALA y 2D de RL-MALA TD3).
#   4. FASE 2: Entrenamiento variacional del CASSCF / Multi-SJB AL FINAL.
#   5. Learning rate scheduler configurable y conmutable (ON / OFF) para todos
#      los modelos entrenables (--use_lr_scheduler / --no-use_lr_scheduler).
#   6. FASE 3: Diagnósticos, Heatmaps 2D de KL (Total, Up, Down), tau_int y Pareto.
# ==============================================================================

set -u

# ------------------------------------------------------------------------------
# 1. Variables de Entorno y Rendimiento Numérico
# ------------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.25}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# 2. Rutas y Detección Automática de Archivos de Referencia
# ------------------------------------------------------------------------------
TARGET_CASSCF_NPZ="${TARGET_CASSCF_NPZ:-figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz}"

# Fallbacks automáticos si el archivo .npz se ubica en data/, raíz o figures/
if [ ! -f "${TARGET_CASSCF_NPZ}" ]; then
    if [ -f "cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" ]; then
        TARGET_CASSCF_NPZ="cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
    elif [ -f "data/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" ]; then
        TARGET_CASSCF_NPZ="data/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
    elif [ -f "results/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" ]; then
        TARGET_CASSCF_NPZ="results/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
    fi
fi

# Detección de CSV de densidad de referencia CASSCF opcional
REF_DENSITY_CSV="${REF_DENSITY_CSV:-results_csv_b2/casscf_exact_density_N6_w0.1_g1.0.csv}"
REF_CSV_ARG=""
if [ ! -f "${REF_DENSITY_CSV}" ]; then
    if [ -f "results_csv/casscf_exact_density_N6_w0.1_g1.0.csv" ]; then
        REF_DENSITY_CSV="results_csv/casscf_exact_density_N6_w0.1_g1.0.csv"
    elif [ -f "casscf_exact_density_N6_w0.1_g1.0.csv" ]; then
        REF_DENSITY_CSV="casscf_exact_density_N6_w0.1_g1.0.csv"
    elif [ -f "data/casscf_exact_density_N6_w0.1_g1.0.csv" ]; then
        REF_DENSITY_CSV="data/casscf_exact_density_N6_w0.1_g1.0.csv"
    else
        REF_DENSITY_CSV=""
    fi
fi
if [ -n "${REF_DENSITY_CSV}" ] && [ -f "${REF_DENSITY_CSV}" ]; then
    REF_CSV_ARG="--reference_density_csv ${REF_DENSITY_CSV}"
fi

BASE_OUT_DIR="results_param_search"
CSV_COLLECT_DIR="${BASE_OUT_DIR}/csv_summary"
FIG_OUT_DIR="${BASE_OUT_DIR}/figures"
LOG_DIR="${BASE_OUT_DIR}/logs"
TRACKER_DIR="${BASE_OUT_DIR}/.tracker"

mkdir -p "${BASE_OUT_DIR}" "${CSV_COLLECT_DIR}" "${FIG_OUT_DIR}" "${LOG_DIR}" "${TRACKER_DIR}"
rm -f "${TRACKER_DIR}"/job_*.meta 2>/dev/null || true

MASTER_LOG="${LOG_DIR}/master_parameter_search.log"

# Logger directo en tiempo real al log maestro y terminal
log_master() {
    local msg="$1"
    echo "$msg"
    echo "$msg" >> "${MASTER_LOG}"
}

# ------------------------------------------------------------------------------
# 3. Configuración del Sistema de Colas de GPUs
# ------------------------------------------------------------------------------
GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"

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

# ------------------------------------------------------------------------------
# 4. Parámetros del Parameter Search y Scheduler de Learning Rate
# ------------------------------------------------------------------------------
N_BURN_IN="${N_BURN_IN:-2500}"
MEASURE_STEPS="${MEASURE_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
VMC_TRAIN_STEPS="${VMC_TRAIN_STEPS:-1500}"

# Learning rate scheduler conmutable para modelos entrenables (VMC y RL)
USE_LR_SCHEDULER="${USE_LR_SCHEDULER:-1}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.05}"

LR_SCHEDULER_FLAGS="--use_lr_scheduler --lr_scheduler_type ${LR_SCHEDULER_TYPE} --lr_warmup_steps ${LR_WARMUP_STEPS} --lr_min_ratio ${LR_MIN_RATIO}"
if [ "$USE_LR_SCHEDULER" -eq 0 ]; then
    LR_SCHEDULER_FLAGS="--no-use_lr_scheduler"
fi

# Rejillas de Hiperparámetros de Muestreo
TAU_MALA_LIST=(0.02 0.04 0.06 0.08 0.10 0.12 0.15)
TAU_ACTOR_LIST=(0.02 0.04 0.06 0.08 0.10 0.12)
ACTOR_AMP_LIST=(0.05 0.10 0.20 0.40 0.80)

MASTER_SUMMARY_CSV="${BASE_OUT_DIR}/master_param_search_summary.csv"
echo "method,tau,actor_amplitude,label,h5_path,E_mean,SE_E,acceptance,ESJD,tau_int_E,G_RL_speedup,ESS_per_sec,casscf_energy,delta_E_vs_CASSCF,kl_casscf_total_mean,kl_casscf_total_std,kl_casscf_up_mean,kl_casscf_down_mean" > "${MASTER_SUMMARY_CSV}"

log_master "=============================================================================="
log_master "    INICIO DE PARAMETER SEARCH EN AYLER (SISTEMA DE COLAS GPU)"
log_master "    Fecha y Hora        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "    Sistema Físico      : N=6 (3 up, 3 dn), w=0.1, g=1.0, softening=0.1"
log_master "    Estado Referencia   : ${TARGET_CASSCF_NPZ}"
log_master "    Pool de GPUs        : ${AVAILABLE_GPUS[*]}"
log_master "    Trabajos por GPU    : ${JOBS_PER_GPU}"
log_master "    Concurrencia Máxima : ${MAX_TOTAL_JOBS} trabajos simultáneos"
log_master "    LR Scheduler Activo : $([ "$USE_LR_SCHEDULER" -eq 1 ] && echo "SÍ (${LR_SCHEDULER_TYPE})" || echo "NO (Constante)")"
log_master "    Logs en Tiempo Real : ${LOG_DIR}/"
log_master "=============================================================================="

if [ ! -f "${TARGET_CASSCF_NPZ}" ]; then
    log_master "[-] ERROR: No se encontró el archivo CASSCF objetivo en: ${TARGET_CASSCF_NPZ}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. Funciones Auxiliares del Gestor de Colas (Queue Manager)
# ------------------------------------------------------------------------------
format_duration() {
    local dur=$1
    if [ "$dur" -lt 0 ]; then dur=0; fi
    printf "%02dh %02dm %02ds" $((dur/3600)) $(( (dur%3600)/60 )) $((dur%60))
}

cleanup_finished_jobs() {
    local stage_name="$1"
    for f in "$TRACKER_DIR"/job_*.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null
            local exit_code=$?
            local gpu sys start_epoch start_time script_name sys_log_file sum_csv method_tag tau_val amp_val
            # shellcheck disable=SC1090
            source "$f"
            local end_epoch
            end_epoch=$(date +%s)
            local end_time
            end_time=$(date "+%Y-%m-%d %H:%M:%S")
            local duration=$((end_epoch - start_epoch))
            local duration_formatted
            duration_formatted=$(format_duration "$duration")

            # Inserción atómica y segura de métricas al CSV maestro
            if [ -n "${sum_csv:-}" ] && [ -f "$sum_csv" ]; then
                (
                    flock -x 200 2>/dev/null || true
                    tail -n +2 "$sum_csv" | awk -v m="${method_tag:-MALA}" -v t="${tau_val:-0.08}" -v a="${amp_val:-0.0}" -F',' '{OFS=","; print m, t, a, $0}' >> "${MASTER_SUMMARY_CSV}"
                ) 200>"${MASTER_SUMMARY_CSV}.lock"
            fi

            log_master "------------------------------------------------------------------------------"
            log_master "<< [$stage_name | FINALIZADO] ${sys}"
            log_master "   PID del Worker     : $pid (Código de Salida: $exit_code)"
            log_master "   GPU Liberada       : GPU $gpu"
            log_master "   Hora de Inicio     : $start_time"
            log_master "   Hora de Terminación: $end_time"
            log_master "   Tiempo Transcurrido: ${duration}s ($duration_formatted)"
            log_master "   Log en Tiempo Real : ${LOG_DIR}/${sys_log_file}"
            log_master "=============================================================================="

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
# 6. Funciones de Ejecución de Tareas en Background con Logs en Tiempo Real
# ------------------------------------------------------------------------------
run_single_sampling_worker() {
    local GPU_ID="$1"
    local METHOD="$2"
    local TAU="$3"
    local AMP="$4"
    local RUN_TAG="$5"

    local MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
    local POST_DIR="${MODEL_DIR}/postprocess"
    mkdir -p "${MODEL_DIR}" "${POST_DIR}"
    local H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"

    local EXTRA_ARGS=""
    local LABEL="MALA (tau=${TAU})"
    if [ "$METHOD" = "TD3" ]; then
        EXTRA_ARGS="--actor_amplitude ${AMP}"
        LABEL="RL-MALA (A=${AMP}, tau=${TAU})"
    fi

    # Muestreo Puro con flush en tiempo real
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} -u cli/run_1d.py \
        --method "${METHOD}" \
        --train_mode eval_only \
        --ansatz_type casscf_npz \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
        --tau "${TAU}" \
        ${EXTRA_ARGS} \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_h5 "${H5_FILE}"

    # Post-procesamiento inmediato con métricas de KL por espín y formato PNG
    ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_observables_comparison \
        --plot_kl_divergence \
        --figure_format png
}

run_single_training_worker() {
    local GPU_ID="$1"
    local METHOD="$2"
    local TRAIN_MODE="$3"
    local AMP="$4"
    local RUN_TAG="$5"

    local MODEL_DIR="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}"
    local POST_DIR="${MODEL_DIR}/postprocess"
    mkdir -p "${MODEL_DIR}" "${POST_DIR}"
    local H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"

    local EXTRA_ARGS=""
    local LABEL="VMC + MALA (Multi-SJB)"
    if [ "$METHOD" = "TD3" ]; then
        EXTRA_ARGS="--actor_amplitude ${AMP}"
        LABEL="Coupled VMC + RL-MALA (Multi-SJB)"
    fi

    # Entrenamiento con Learning Rate Scheduler conmutable
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} -u cli/train_1d.py \
        --method "${METHOD}" \
        --train_mode "${TRAIN_MODE}" \
        --ansatz_type multi_sjb \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        --correlation_model spin_rbf_jastrow \
        --orbital_model backflow_ho \
        --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
        --tau 0.08 \
        ${EXTRA_ARGS} \
        ${LR_SCHEDULER_FLAGS} \
        --train_steps "${VMC_TRAIN_STEPS}" \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_h5 "${H5_FILE}"

    # Post-procesamiento
    ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_training_metrics \
        --plot_observables_comparison \
        --plot_kl_divergence \
        --figure_format png
}

# ==============================================================================
# FASE 1: MUESTREO PURO SOLAMENTE (Cola Dinámica de GPUs)
# ==============================================================================
# Primero se realizan TODAS las tareas de muestreo con la referencia CASSCF congelada.
TOTAL_SAMPLING_RUNS=$(( ${#TAU_MALA_LIST[@]} + (${#TAU_ACTOR_LIST[@]} * ${#ACTOR_AMP_LIST[@]}) ))

log_master ""
log_master "=============================================================================="
log_master ">>> FASE 1: MUESTREO PURO SOLAMENTE (${TOTAL_SAMPLING_RUNS} corridas en total)"
log_master "    - Barrido 1D MALA    : ${#TAU_MALA_LIST[@]} valores de tau"
log_master "    - Rejilla 2D RL-MALA : ${#ACTOR_AMP_LIST[@]} amplitudes x ${#TAU_ACTOR_LIST[@]} taus"
log_master "    - Despacho Dinámico  : Concurrencia máxima = ${MAX_TOTAL_JOBS} en GPUs [${AVAILABLE_GPUS[*]}]"
log_master "=============================================================================="

# 1A. Despacho de MALA Estándar a la cola
for tau in "${TAU_MALA_LIST[@]}"; do
    RUN_TAG="MALA_tau${tau}"
    SYS_LOG="sampling_${RUN_TAG}.log"
    GPU_ID=$(wait_for_free_gpu_slot "FASE 1: MUESTREO PURO")

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    # Lanzamiento con salida desacoplada en tiempo real
    run_single_sampling_worker "$GPU_ID" "MALA" "$tau" "0.0" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
    worker_pid=$!

    cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="MALA"
tau_val="$tau"
amp_val="0.0"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_sampling_worker MALA tau=$tau"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/models/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF

    log_master ">> [FASE 1 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
done

# 1B. Despacho de RL-MALA TD3 (Rejilla 2D) a la cola
for amp in "${ACTOR_AMP_LIST[@]}"; do
    for tau in "${TAU_ACTOR_LIST[@]}"; do
        RUN_TAG="RL_MALA_amp${amp}_tau${tau}"
        SYS_LOG="sampling_${RUN_TAG}.log"
        GPU_ID=$(wait_for_free_gpu_slot "FASE 1: MUESTREO PURO")

        start_epoch=$(date +%s)
        start_time=$(date "+%Y-%m-%d %H:%M:%S")

        run_single_sampling_worker "$GPU_ID" "TD3" "$tau" "$amp" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
        worker_pid=$!

        cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="TD3"
tau_val="$tau"
amp_val="$amp"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_sampling_worker TD3 amp=$amp tau=$tau"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/models/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF

        log_master ">> [FASE 1 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
    done
done

# Esperar a que concluyan TODOS los muestreos antes de iniciar el entrenamiento
log_master ""
log_master "[*] Esperando a que todos los trabajos de muestreo en la cola terminen..."
wait_all_stage_jobs "FASE 1: MUESTREO PURO"
log_master "[+] FASE 1 COMPLETADA CON ÉXITO. Todas las corridas de muestreo han finalizado."

# ==============================================================================
# FASE 2: ENTRENAMIENTO VARIACIONAL (CASSCF / MULTI-SJB) AL FINAL
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 2: ENTRENAMIENTO VARIACIONAL VMC (Multi-SJB 20 det + Jastrow + Backflow)"
log_master "    - Se ejecuta AL FINAL, después de culminar todo el muestreo puro."
log_master "    - Modelos: VMC + MALA y Coupled VMC + RL-MALA TD3."
log_master "    - LR Scheduler: $([ "$USE_LR_SCHEDULER" -eq 1 ] && echo "ACTIVO (${LR_SCHEDULER_TYPE})" || echo "DESACTIVADO")"
log_master "=============================================================================="

# 2A. VMC + MALA Multi-SJB
RUN_TAG="VMC_MALA_MultiSJB"
SYS_LOG="train_${RUN_TAG}.log"
GPU_ID=$(wait_for_free_gpu_slot "FASE 2: ENTRENAMIENTO")

start_epoch=$(date +%s)
start_time=$(date "+%Y-%m-%d %H:%M:%S")

run_single_training_worker "$GPU_ID" "MALA" "vmc_only" "0.0" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
worker_pid=$!

cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="VMC_MALA_MultiSJB"
tau_val="0.08"
amp_val="0.0"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_training_worker VMC_MALA"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF
log_master ">> [FASE 2 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"

# 2B. Coupled VMC + RL-MALA TD3 Multi-SJB
RUN_TAG="Coupled_RL_MultiSJB"
SYS_LOG="train_${RUN_TAG}.log"
GPU_ID=$(wait_for_free_gpu_slot "FASE 2: ENTRENAMIENTO")

start_epoch=$(date +%s)
start_time=$(date "+%Y-%m-%d %H:%M:%S")

run_single_training_worker "$GPU_ID" "TD3" "coupled" "0.20" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
worker_pid=$!

cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="Coupled_RL_MultiSJB"
tau_val="0.08"
amp_val="0.20"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_training_worker Coupled_RL"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF
log_master ">> [FASE 2 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"

log_master ""
log_master "[*] Esperando a que los entrenamientos de Multi-SJB concluyan..."
wait_all_stage_jobs "FASE 2: ENTRENAMIENTO"
log_master "[+] FASE 2 COMPLETADA CON ÉXITO. Todos los entrenamientos han finalizado."

# ==============================================================================
# FASE 3: DIAGNÓSTICOS, HEATMAPS 2D Y CURVAS DE PARETO
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 3: GENERACIÓN DE HEATMAPS 2D Y ANÁLISIS DE PARETO..."
log_master "=============================================================================="

${PYTHON_BIN} -u scripts/plot_param_search_results.py \
    --csv "${MASTER_SUMMARY_CSV}" \
    --out_dir "${FIG_OUT_DIR}"

log_master "=============================================================================="
log_master "    PARAMETER SEARCH COMPLETADO CON ÉXITO EN AYLER"
log_master "    CSV Maestro  : ${MASTER_SUMMARY_CSV}"
log_master "    Figuras 2D   : ${FIG_OUT_DIR}"
log_master "    Logs Totales : ${LOG_DIR}/"
log_master "    Fecha y Hora : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "=============================================================================="
