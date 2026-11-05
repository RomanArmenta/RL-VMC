#!/usr/bin/env bash
# ==============================================================================
# run_parameter_search_ayler.sh
# Parameter Search Automatizado con Sistema de Colas Dinámico para GPUs
# Sistema Objetivo: CASSCF 20 Determinantes (Base 256), N=6, w=0.1, g=1.0, a=0.1
#
# Mejoras y Robustez (Versión Corregida):
#   1. Recibe carpeta inicial ($1 o BASE_OUT_DIR). Si ya existen corridas
#      previas completadas (bundles .h5 y resúmenes), las SALTA automáticamente,
#      recupera sus métricas al CSV maestro y garantiza que todas sus figuras
#      estén generadas y centralizadas en figures/.
#   2. Sistema de colas dinámico 100% libre de subshells y colisiones de STDOUT:
#      La asignación de GPU usa variables globales seguras. Jamás corrompe
#      CUDA_VISIBLE_DEVICES ni provoca caída a CPU.
#   3. JAX_PLATFORMS=cuda forzado para prevenir fallbacks accidentales a CPU.
#   4. FASE 1: Muestreo PURO (barrido 1D de MALA y rejilla 2D de RL-MALA TD3).
#   5. FASE 2: Entrenamiento variacional del Multi-SJB (VMC+MALA y Coupled RL+MALA).
#   6. FASE 2B: Evaluaciones de los modelos entrenados (Multi-SJB con MALA y TD3)
#      a través de múltiples configuraciones y semillas para comparar evolución.
#   7. FASE 3: Generación integral de figuras comparativas para todos los modelos,
#      heatmaps 2D, curvas de evolución de hiperparámetros y fronteras de Pareto,
#      REMARCANDO explícitamente los mejores modelos (MALA y TD3).
# ==============================================================================

set -u

# ------------------------------------------------------------------------------
# 1. Variables de Entorno y Rendimiento Numérico
# ------------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.18}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# 2. Rutas y Detección Automática de Archivos de Referencia y Directorios
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

# Detección y asignación del directorio base y opciones de formato de figuras
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TITLE_PREFIX="${TITLE_PREFIX:-}"
CUSTOM_SUPTITLE="${CUSTOM_SUPTITLE:-}"
NO_TITLES="${NO_TITLES:-0}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"
TRANSPARENT_BG="${TRANSPARENT_BG:-0}"
TARGET_BASE_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --figure-format|--figure_format|--format|-f) FIGURE_FORMAT="$2"; shift 2 ;;
        --pdf) FIGURE_FORMAT="pdf"; shift ;;
        --png) FIGURE_FORMAT="png"; shift ;;
        --dpi|--ppi) FIGURE_DPI="$2"; shift 2 ;;
        --title-prefix|--title_prefix) TITLE_PREFIX="$2"; shift 2 ;;
        --custom-suptitle|--custom_suptitle) CUSTOM_SUPTITLE="$2"; shift 2 ;;
        --no-titles|--no_titles|--hide-titles) NO_TITLES=1; shift ;;
        --font-scale|--font_scale) FONT_SCALE="$2"; shift 2 ;;
        --line-width-scale|--line_width_scale) LINE_WIDTH_SCALE="$2"; shift 2 ;;
        --transparent-bg|--transparent_bg) TRANSPARENT_BG=1; shift ;;
        *) TARGET_BASE_ARG="$1"; shift ;;
    esac
done
export FIGURE_FORMAT FIGURE_DPI TITLE_PREFIX CUSTOM_SUPTITLE NO_TITLES FONT_SCALE LINE_WIDTH_SCALE TRANSPARENT_BG

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

TARGET_BASE_ARG="${TARGET_BASE_ARG:-${BASE_OUT_DIR:-}}"
if [ -n "${TARGET_BASE_ARG}" ]; then
    BASE_OUT_DIR="${TARGET_BASE_ARG}"
elif [ -d "results_params_search" ]; then
    BASE_OUT_DIR="results_params_search"
elif [ -d "results_param_search" ]; then
    BASE_OUT_DIR="results_param_search"
else
    BASE_OUT_DIR="results_params_search"
fi
export BASE_OUT_DIR

CSV_COLLECT_DIR="${BASE_OUT_DIR}/csv_summary"
FIG_OUT_DIR="${BASE_OUT_DIR}/figures"
LOG_DIR="${BASE_OUT_DIR}/logs"
TRACKER_DIR="${BASE_OUT_DIR}/.tracker"

mkdir -p "${BASE_OUT_DIR}" "${CSV_COLLECT_DIR}" "${FIG_OUT_DIR}" "${LOG_DIR}" "${TRACKER_DIR}"
rm -f "${TRACKER_DIR}"/job_*.meta 2>/dev/null || true

MASTER_LOG="${LOG_DIR}/master_parameter_search.log"

# Logger con salida dual a terminal y archivo
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
N_BURN_IN="${N_BURN_IN:-1500}"
MEASURE_STEPS="${MEASURE_STEPS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VMC_TRAIN_STEPS="${VMC_TRAIN_STEPS:-1500}"
RL_COUPLED_AMP="${RL_COUPLED_AMP:-0.05}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-500}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-10000}"

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

# CSVs Maestros de Resumen con Mapeo de Columnas Estrictamente Alineado
MASTER_SUMMARY_HEADER="run_tag,method,tau,actor_amplitude,label,h5_path,N,n_up,n_dn,omega,g,softening,n_chains,E_mean,SE_E,E_var,T_mean,V_mean,acceptance,ESJD,accepted_disp,tau_int_E,tau_int_pos,G_RL_speedup,ESS,ESS_per_sec,wall_time_sec,time_to_solution_sec,n_wigner_peaks,rms_radius,mean_pair_dist,has_casscf_baseline,casscf_energy,delta_E_vs_CASSCF,kl_casscf_total_mean,kl_casscf_total_std,kl_casscf_up_mean,kl_casscf_down_mean,kl_divergence_2e,ESS_speedup_vs_baseline,delta_E_vs_baseline"

MASTER_SUMMARY_CSV="${BASE_OUT_DIR}/master_param_search_summary.csv"
if [ ! -f "${MASTER_SUMMARY_CSV}" ] || [ ! -s "${MASTER_SUMMARY_CSV}" ]; then
    echo "${MASTER_SUMMARY_HEADER}" > "${MASTER_SUMMARY_CSV}"
fi

EVAL_SUMMARY_CSV="${BASE_OUT_DIR}/evaluations_trained_summary.csv"
if [ ! -f "${EVAL_SUMMARY_CSV}" ] || [ ! -s "${EVAL_SUMMARY_CSV}" ]; then
    echo "${MASTER_SUMMARY_HEADER}" > "${EVAL_SUMMARY_CSV}"
fi

log_master "=============================================================================="
log_master "    INICIO DE PARAMETER SEARCH EN AYLER (ROBUSTO Y RESUMIBLE)"
log_master "    Directorio Base     : ${BASE_OUT_DIR}"
log_master "    Fecha y Hora        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "    Sistema Físico      : N=6 (3 up, 3 dn), w=0.1, g=1.0, softening=0.1"
log_master "    Estado Referencia   : ${TARGET_CASSCF_NPZ}"
log_master "    Pool de GPUs        : ${AVAILABLE_GPUS[*]}"
log_master "    Trabajos por GPU    : ${JOBS_PER_GPU}"
log_master "    Concurrencia Máxima : ${MAX_TOTAL_JOBS} trabajos simultáneos"
log_master "    LR Scheduler Activo : $([ "$USE_LR_SCHEDULER" -eq 1 ] && echo "SÍ (${LR_SCHEDULER_TYPE})" || echo "NO (Constante)")"
log_master "    Logs en Tiempo Real : ${LOG_DIR}/"
log_master "    Figuras Centralizadas: ${FIG_OUT_DIR}/"
log_master "=============================================================================="

if [ ! -f "${TARGET_CASSCF_NPZ}" ]; then
    log_master "[-] ERROR: No se encontró el archivo CASSCF objetivo en: ${TARGET_CASSCF_NPZ}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. Funciones Auxiliares del Gestor de Colas (Queue Manager Sin Subshells)
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
            wait "$pid" 2>/dev/null || true
            local exit_code=$?
            local gpu sys start_epoch start_time script_name sys_log_file sum_csv method_tag tau_val amp_val target_csv post_dir_path
            # shellcheck disable=SC1090
            source "$f"
            local end_epoch
            end_epoch=$(date +%s)
            local end_time
            end_time=$(date "+%Y-%m-%d %H:%M:%S")
            local duration=$((end_epoch - start_epoch))
            local duration_formatted
            duration_formatted=$(format_duration "$duration")

            local dest_csv="${target_csv:-$MASTER_SUMMARY_CSV}"

            # Inserción atómica y segura de métricas al CSV maestro si aún no está presente
            if [ -n "${sum_csv:-}" ] && [ -f "$sum_csv" ]; then
                if [ ! -f "${dest_csv}" ] || [ ! -s "${dest_csv}" ]; then
                    echo "${MASTER_SUMMARY_HEADER}" > "${dest_csv}"
                fi
                if ! grep -q "^${sys}," "${dest_csv}" 2>/dev/null; then
                    (
                        flock -x 200 2>/dev/null || true
                        tail -n +2 "$sum_csv" | awk -v r="${sys}" -v m="${method_tag:-MALA}" -v t="${tau_val:-0.08}" -v a="${amp_val:-0.0}" -F',' '{OFS=","; print r, m, t, a, $0}' >> "${dest_csv}"
                    ) 200>"${dest_csv}.lock"
                fi
            fi

            # Copia y centralización inmediata de figuras en figures/
            if [ -n "${post_dir_path:-}" ] && [ -d "${post_dir_path}" ]; then
                mkdir -p "${FIG_OUT_DIR}/individual_runs/${sys}"
                cp -f "${post_dir_path}"/*.png "${FIG_OUT_DIR}/individual_runs/${sys}/" 2>/dev/null || true
                for fig in "${post_dir_path}"/*.png; do
                    [ -e "$fig" ] || continue
                    local bname
                    bname=$(basename "$fig")
                    cp -f "$fig" "${FIG_OUT_DIR}/${sys}_${bname}" 2>/dev/null || true
                done
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

# Variable global que almacena la GPU asignada de forma limpia
ALLOCATED_GPU=""

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
                ALLOCATED_GPU="$best_gpu"
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
# 6. Funciones de Detección de Corridas Existentes y Generación de Figuras
# ------------------------------------------------------------------------------
check_and_register_completed_run() {
    local RUN_TAG="$1"
    local METHOD="$2"
    local TAU="$3"
    local AMP="$4"
    local H5_FILE="$5"
    local POST_DIR="$6"
    local LABEL="$7"
    local TARGET_CSV="${8:-$MASTER_SUMMARY_CSV}"

    local SUM_CSV="${POST_DIR}/summary_metrics.csv"

    # Verificar si el bundle HDF5 existe y no está vacío
    if [ -f "${H5_FILE}" ] && [ -s "${H5_FILE}" ]; then
        # Verificar si ya existen figuras generadas; si faltan, correr postprocess
        local num_figs
        num_figs=$(find "${POST_DIR}" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
        if [ ! -f "${SUM_CSV}" ] || [ "$num_figs" -eq 0 ]; then
            log_master "[*] Ejecutando post-procesamiento para completar figuras de: ${RUN_TAG}..."
            CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu ${PYTHON_BIN} -u cli/postprocess.py \
                --inputs "${H5_FILE}" \
                --labels "${LABEL}" \
                --out_dir "${POST_DIR}" \
                --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
                ${REF_CSV_ARG} \
                --system_tag "${RUN_TAG}" \
                --csv_collect_dir "${CSV_COLLECT_DIR}" \
                --plot_training_metrics \
                --plot_production_summary \
                --plot_observables_comparison \
                --plot_efficiency_bars \
                --plot_time_to_solution \
                --plot_walker_trace \
                --plot_kl_divergence \
                "${PLOT_STYLE_ARGS[@]}" >/dev/null 2>&1 || true
        fi

        # Asegurar inserción en el CSV maestro si aún no está
        if [ -f "${SUM_CSV}" ]; then
            if [ ! -f "${TARGET_CSV}" ] || [ ! -s "${TARGET_CSV}" ]; then
                echo "${MASTER_SUMMARY_HEADER}" > "${TARGET_CSV}"
            fi
            if ! grep -q "^${RUN_TAG}," "${TARGET_CSV}" 2>/dev/null; then
                (
                    flock -x 200 2>/dev/null || true
                    tail -n +2 "$SUM_CSV" | awk -v r="${RUN_TAG}" -v m="${METHOD}" -v t="${TAU}" -v a="${AMP}" -F',' '{OFS=","; print r, m, t, a, $0}' >> "${TARGET_CSV}"
                ) 200>"${TARGET_CSV}.lock"
            fi
        fi

        # Centralizar todas las figuras en figures/
        if [ -d "${POST_DIR}" ]; then
            mkdir -p "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}"
            cp -f "${POST_DIR}"/*.png "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}/" 2>/dev/null || true
            for fig in "${POST_DIR}"/*.png; do
                [ -e "$fig" ] || continue
                local bname
                bname=$(basename "$fig")
                cp -f "$fig" "${FIG_OUT_DIR}/${RUN_TAG}_${bname}" 2>/dev/null || true
            done
        fi

        log_master "[OMITIDO | YA EXISTE] ${RUN_TAG} ya finalizado con éxito. Saltando cómputo."
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------------------
# 7. Funciones de Ejecución de Trabajadores (Workers)
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

    # Muestreo Puro con JAX en GPU
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
        --jit_chunk_size "${JIT_CHUNK_SIZE}" \
        --eloc_chunk_size "${ELOC_CHUNK_SIZE}" \
        --output_h5 "${H5_FILE}"

    # Post-procesamiento exhaustivo con todas las figuras requeridas (en CPU)
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_production_summary \
        --plot_observables_comparison \
        --plot_efficiency_bars \
        --plot_time_to_solution \
        --plot_walker_trace \
        --plot_kl_divergence \
        "${PLOT_STYLE_ARGS[@]}"

    # Centralización de figuras
    if [ -d "${POST_DIR}" ]; then
        mkdir -p "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}"
        cp -f "${POST_DIR}"/*.png "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}/" 2>/dev/null || true
        for fig in "${POST_DIR}"/*.png; do
            [ -e "$fig" ] || continue
            local bname
            bname=$(basename "$fig")
            cp -f "$fig" "${FIG_OUT_DIR}/${RUN_TAG}_${bname}" 2>/dev/null || true
        done
    fi
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

    # Entrenamiento con Learning Rate Scheduler
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
        --jit_chunk_size "${JIT_CHUNK_SIZE}" \
        --eloc_chunk_size "${ELOC_CHUNK_SIZE}" \
        --output_h5 "${H5_FILE}"

    # Post-procesamiento exhaustivo incluyendo métricas de entrenamiento (en CPU)
    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_training_metrics \
        --plot_production_summary \
        --plot_observables_comparison \
        --plot_efficiency_bars \
        --plot_time_to_solution \
        --plot_walker_trace \
        --plot_kl_divergence \
        "${PLOT_STYLE_ARGS[@]}"

    if [ -d "${POST_DIR}" ]; then
        mkdir -p "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}"
        cp -f "${POST_DIR}"/*.png "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}/" 2>/dev/null || true
        for fig in "${POST_DIR}"/*.png; do
            [ -e "$fig" ] || continue
            local bname
            bname=$(basename "$fig")
            cp -f "$fig" "${FIG_OUT_DIR}/${RUN_TAG}_${bname}" 2>/dev/null || true
        done
    fi
}

run_single_eval_worker() {
    local GPU_ID="$1"
    local METHOD="$2"
    local TAU="$3"
    local AMP="$4"
    local LOAD_H5="$5"
    local SEED_VAL="$6"
    local RUN_TAG="$7"

    local MODEL_DIR="${BASE_OUT_DIR}/evaluations_trained/${RUN_TAG}"
    local POST_DIR="${MODEL_DIR}/postprocess"
    mkdir -p "${MODEL_DIR}" "${POST_DIR}"
    local H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"

    local EXTRA_ARGS=""
    local LABEL="Eval MALA (tau=${TAU}, s=${SEED_VAL}) [Multi-SJB]"
    if [ "$METHOD" = "TD3" ]; then
        EXTRA_ARGS="--actor_amplitude ${AMP}"
        LABEL="Eval RL-MALA (A=${AMP}, tau=${TAU}, s=${SEED_VAL}) [Multi-SJB]"
    fi

    env CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} -u cli/run_1d.py \
        --method "${METHOD}" \
        --train_mode eval_only \
        --ansatz_type multi_sjb \
        --correlation_model spin_rbf_jastrow \
        --orbital_model backflow_ho \
        --load_bundle_h5 "${LOAD_H5}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
        --tau "${TAU}" \
        ${EXTRA_ARGS} \
        --seed "${SEED_VAL}" \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --jit_chunk_size "${JIT_CHUNK_SIZE}" \
        --eloc_chunk_size "${ELOC_CHUNK_SIZE}" \
        --output_h5 "${H5_FILE}"

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_production_summary \
        --plot_observables_comparison \
        --plot_efficiency_bars \
        --plot_time_to_solution \
        --plot_walker_trace \
        --plot_kl_divergence \
        "${PLOT_STYLE_ARGS[@]}"

    if [ -d "${POST_DIR}" ]; then
        mkdir -p "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}"
        cp -f "${POST_DIR}"/*.png "${FIG_OUT_DIR}/individual_runs/${RUN_TAG}/" 2>/dev/null || true
        for fig in "${POST_DIR}"/*.png; do
            [ -e "$fig" ] || continue
            local bname
            bname=$(basename "$fig")
            cp -f "$fig" "${FIG_OUT_DIR}/${RUN_TAG}_${bname}" 2>/dev/null || true
        done
    fi
}

# ==============================================================================
# FASE 1: MUESTREO PURO SOLAMENTE (Cola Dinámica de GPUs)
# ==============================================================================
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
    MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
    POST_DIR="${MODEL_DIR}/postprocess"
    H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
    LABEL="MALA (tau=${tau})"

    if check_and_register_completed_run "$RUN_TAG" "MALA" "$tau" "0.0" "$H5_FILE" "$POST_DIR" "$LABEL"; then
        continue
    fi

    wait_for_free_gpu_slot "FASE 1: MUESTREO PURO"
    GPU_ID="$ALLOCATED_GPU"

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

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
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${MASTER_SUMMARY_CSV}"
EOF

    log_master ">> [FASE 1 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
done

# 1B. Despacho de RL-MALA TD3 (Rejilla 2D) a la cola
for amp in "${ACTOR_AMP_LIST[@]}"; do
    for tau in "${TAU_ACTOR_LIST[@]}"; do
        RUN_TAG="RL_MALA_amp${amp}_tau${tau}"
        SYS_LOG="sampling_${RUN_TAG}.log"
        MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
        POST_DIR="${MODEL_DIR}/postprocess"
        H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
        LABEL="RL-MALA (A=${amp}, tau=${tau})"

        if check_and_register_completed_run "$RUN_TAG" "TD3" "$tau" "$amp" "$H5_FILE" "$POST_DIR" "$LABEL"; then
            continue
        fi

        wait_for_free_gpu_slot "FASE 1: MUESTREO PURO"
        GPU_ID="$ALLOCATED_GPU"

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
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${MASTER_SUMMARY_CSV}"
EOF

        log_master ">> [FASE 1 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
    done
done

# Esperar a que concluyan TODOS los muestreos de la Fase 1
log_master ""
log_master "[*] Esperando a que todos los trabajos de muestreo en la cola terminen..."
wait_all_stage_jobs "FASE 1: MUESTREO PURO"
log_master "[+] FASE 1 COMPLETADA CON ÉXITO. Todas las corridas de muestreo han finalizado."

# ==============================================================================
# FASE 2: ENTRENAMIENTO VARIACIONAL (CASSCF / MULTI-SJB)
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 2: ENTRENAMIENTO VARIACIONAL VMC (Multi-SJB 20 det + Jastrow + Backflow)"
log_master "    - Modelos: VMC + MALA y Coupled VMC + RL-MALA TD3."
log_master "    - Amplitud Acoplada RL: ${RL_COUPLED_AMP}"
log_master "    - LR Scheduler: $([ "$USE_LR_SCHEDULER" -eq 1 ] && echo "ACTIVO (${LR_SCHEDULER_TYPE})" || echo "DESACTIVADO")"
log_master "=============================================================================="

# 2A. VMC + MALA Multi-SJB
RUN_TAG="VMC_MALA_MultiSJB"
SYS_LOG="train_${RUN_TAG}.log"
MODEL_DIR="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}"
POST_DIR="${MODEL_DIR}/postprocess"
H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
LABEL="VMC + MALA (Multi-SJB)"

if ! check_and_register_completed_run "$RUN_TAG" "VMC_MALA_MultiSJB" "0.08" "0.0" "$H5_FILE" "$POST_DIR" "$LABEL"; then
    wait_for_free_gpu_slot "FASE 2: ENTRENAMIENTO"
    GPU_ID="$ALLOCATED_GPU"

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
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${MASTER_SUMMARY_CSV}"
EOF
    log_master ">> [FASE 2 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
fi

# 2B. Coupled VMC + RL-MALA TD3 Multi-SJB
RUN_TAG="Coupled_RL_MultiSJB"
SYS_LOG="train_${RUN_TAG}.log"
MODEL_DIR="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}"
POST_DIR="${MODEL_DIR}/postprocess"
H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
LABEL="Coupled VMC + RL-MALA (Multi-SJB)"

if ! check_and_register_completed_run "$RUN_TAG" "Coupled_RL_MultiSJB" "0.08" "${RL_COUPLED_AMP}" "$H5_FILE" "$POST_DIR" "$LABEL"; then
    wait_for_free_gpu_slot "FASE 2: ENTRENAMIENTO"
    GPU_ID="$ALLOCATED_GPU"

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    run_single_training_worker "$GPU_ID" "TD3" "coupled" "${RL_COUPLED_AMP}" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
    worker_pid=$!

    cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="Coupled_RL_MultiSJB"
tau_val="0.08"
amp_val="${RL_COUPLED_AMP}"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_training_worker Coupled_RL"
sys_log_file="$SYS_LOG"
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${MASTER_SUMMARY_CSV}"
EOF
    log_master ">> [FASE 2 | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
fi

log_master ""
log_master "[*] Esperando a que los entrenamientos de Multi-SJB concluyan..."
wait_all_stage_jobs "FASE 2: ENTRENAMIENTO"
log_master "[+] FASE 2 COMPLETADA CON ÉXITO. Todos los entrenamientos han finalizado."

# ==============================================================================
# FASE 2B: EVALUACIÓN DE MODELOS ENTRENADOS (MALA vs TD3 Multi-SJB)
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 2B: EVALUACIÓN COMPARATIVA CON MODELOS ENTRENADOS"
log_master "    - Evalúa MALA vs TD3 sobre el Ansatz Multi-SJB optimizado"
log_master "    - Barrido en tau y múltiples evaluaciones para métricas reproducibles"
log_master "=============================================================================="

TRAINED_VMC_BUNDLE="${BASE_OUT_DIR}/pipeline3_multi_sjb/VMC_MALA_MultiSJB/run_bundle_VMC_MALA_MultiSJB.h5"
TRAINED_RL_BUNDLE="${BASE_OUT_DIR}/pipeline3_multi_sjb/Coupled_RL_MultiSJB/run_bundle_Coupled_RL_MultiSJB.h5"

EVAL_TAU_LIST=(0.04 0.08 0.12)
EVAL_SEEDS=(42 123 999)

# Evaluaciones MALA con el Ansatz entrenado
if [ -f "${TRAINED_VMC_BUNDLE}" ] && [ -s "${TRAINED_VMC_BUNDLE}" ]; then
    for tau in "${EVAL_TAU_LIST[@]}"; do
        for seed_idx in "${EVAL_SEEDS[@]}"; do
            RUN_TAG="eval_MultiSJB_MALA_tau${tau}_s${seed_idx}"
            SYS_LOG="eval_${RUN_TAG}.log"
            MODEL_DIR="${BASE_OUT_DIR}/evaluations_trained/${RUN_TAG}"
            POST_DIR="${MODEL_DIR}/postprocess"
            H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
            LABEL="Multi-SJB MALA (tau=${tau}, s=${seed_idx})"

            if check_and_register_completed_run "$RUN_TAG" "MALA_Trained" "$tau" "0.0" "$H5_FILE" "$POST_DIR" "$LABEL" "${EVAL_SUMMARY_CSV}"; then
                continue
            fi

            wait_for_free_gpu_slot "FASE 2B: EVALUACIONES"
            GPU_ID="$ALLOCATED_GPU"

            start_epoch=$(date +%s)
            start_time=$(date "+%Y-%m-%d %H:%M:%S")

            run_single_eval_worker "$GPU_ID" "MALA" "$tau" "0.0" "${TRAINED_VMC_BUNDLE}" "$seed_idx" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            worker_pid=$!

            cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="MALA_Trained"
tau_val="$tau"
amp_val="0.0"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_eval_worker MALA tau=$tau seed=$seed_idx"
sys_log_file="$SYS_LOG"
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${EVAL_SUMMARY_CSV}"
EOF
            log_master ">> [FASE 2B | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
        done
    done
fi

# Evaluaciones RL-MALA con el Actor y Ansatz entrenados
if [ -f "${TRAINED_RL_BUNDLE}" ] && [ -s "${TRAINED_RL_BUNDLE}" ]; then
    for tau in "${EVAL_TAU_LIST[@]}"; do
        for seed_idx in "${EVAL_SEEDS[@]}"; do
            RUN_TAG="eval_MultiSJB_TD3_tau${tau}_s${seed_idx}"
            SYS_LOG="eval_${RUN_TAG}.log"
            MODEL_DIR="${BASE_OUT_DIR}/evaluations_trained/${RUN_TAG}"
            POST_DIR="${MODEL_DIR}/postprocess"
            H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
            LABEL="Multi-SJB TD3 (A=${RL_COUPLED_AMP}, tau=${tau}, s=${seed_idx})"

            if check_and_register_completed_run "$RUN_TAG" "TD3_Trained" "$tau" "${RL_COUPLED_AMP}" "$H5_FILE" "$POST_DIR" "$LABEL" "${EVAL_SUMMARY_CSV}"; then
                continue
            fi

            wait_for_free_gpu_slot "FASE 2B: EVALUACIONES"
            GPU_ID="$ALLOCATED_GPU"

            start_epoch=$(date +%s)
            start_time=$(date "+%Y-%m-%d %H:%M:%S")

            run_single_eval_worker "$GPU_ID" "TD3" "$tau" "${RL_COUPLED_AMP}" "${TRAINED_RL_BUNDLE}" "$seed_idx" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            worker_pid=$!

            cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="TD3_Trained"
tau_val="$tau"
amp_val="${RL_COUPLED_AMP}"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_eval_worker TD3 tau=$tau seed=$seed_idx"
sys_log_file="$SYS_LOG"
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${EVAL_SUMMARY_CSV}"
EOF
            log_master ">> [FASE 2B | TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
        done
    done
fi

log_master ""
log_master "[*] Esperando a que las evaluaciones de modelos entrenados concluyan..."
wait_all_stage_jobs "FASE 2B: EVALUACIONES"
log_master "[+] FASE 2B COMPLETADA CON ÉXITO. Todas las evaluaciones han finalizado."

# ==============================================================================
# FASE 3: GENERACIÓN INTEGRAL DE FIGURAS Y REMARCADO DE MEJORES MODELOS
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 3: GENERACIÓN DE FIGURAS COMPARATIVAS (MALA vs TD3)"
log_master "    - Evolución de Hiperparámetros, Heatmaps 2D, Curvas Pareto y Rankings"
log_master "    - Destacando explícitamente los MEJORES MODELOS encontrados"
log_master "=============================================================================="

# Script integral en Python ejecutado de forma autocontenida
${PYTHON_BIN} -u - << 'PY_EOF'
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

base_dir = os.environ.get("BASE_OUT_DIR", "results_params_search")
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

csv_path = os.path.join(base_dir, "master_param_search_summary.csv")
eval_csv_path = os.path.join(base_dir, "evaluations_trained_summary.csv")

print(f"[+] Iniciando generación de figuras en: {fig_dir}")

# 1. Cargar datos del CSV maestro
df = pd.DataFrame()
if os.path.exists(csv_path) and os.path.getsize(csv_path) > 30:
    try:
        df = pd.read_csv(csv_path)
        # Limpieza y conversión numérica
        num_cols = ["tau", "actor_amplitude", "E_mean", "SE_E", "acceptance", "ESJD", "tau_int_E", "ESS_per_sec", "delta_E_vs_CASSCF", "kl_casscf_total_mean"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
    except Exception as e:
        print(f"[-] Aviso al leer {csv_path}: {e}")

if df.empty:
    print("[-] No se encontraron datos válidos en el CSV maestro para graficar.")
    sys.exit(0)

# Separar subconjuntos MALA y TD3
df_mala = df[df["method"].str.upper() == "MALA"].copy().sort_values("tau")
df_rl = df[df["method"].str.upper().str.contains("TD3|RL")].copy()

# Encontrar los MEJORES modelos
best_mala = None
if not df_mala.empty and df_mala["tau_int_E"].notna().any():
    best_mala_idx = df_mala["tau_int_E"].idxmin()
    best_mala = df_mala.loc[best_mala_idx]

best_rl = None
if not df_rl.empty and df_rl["tau_int_E"].notna().any():
    best_rl_idx = df_rl["tau_int_E"].idxmin()
    best_rl = df_rl.loc[best_rl_idx]

# ------------------------------------------------------------------------------
# FIGURA 1: EVOLUCIÓN DE HIPERPARÁMETROS MALA VS TD3 (4 PANELES)
# ------------------------------------------------------------------------------
try:
    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    casscf_ref_e = 4.524052

    # Panel (0,0): Energía Media <E> vs tau (MALA discontinuo --, RL-MALA continuo -)
    ax0 = axes[0, 0]
    if not df_mala.empty:
        ax0.plot(df_mala["tau"], df_mala["E_mean"], "o--", color="#1f77b4", lw=2.4, ms=7, label="MALA Estándar")
    if not df_rl.empty:
        amps = sorted(df_rl["actor_amplitude"].dropna().unique())
        cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(amps)))
        for i, a in enumerate(amps):
            sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
            ax0.plot(sub["tau"], sub["E_mean"], "s-", color=cmap[i], lw=1.8, ms=6, label=f"RL-MALA (A={a:.2f})")
    ax0.axhline(casscf_ref_e, color="black", ls=":", lw=1.8, label=f"CASSCF 20 det ({casscf_ref_e:.6f} Ha)")
    ax0.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=12)
    ax0.set_ylabel(r"Energía Media $\langle E \rangle$ (Ha)", fontsize=12)
    ax0.set_title(r"Evolución de Energía Variacional vs $\tau$", fontsize=13, weight="bold")
    ax0.grid(True, alpha=0.3)
    ax0.legend(fontsize=9, loc="best")

    # Panel (0,1): Tasa de Aceptación MCMC vs tau
    ax1 = axes[0, 1]
    ax1.axhspan(70, 90, color="green", alpha=0.12, label="Rango Óptimo Aceptación (70-90%)")
    if not df_mala.empty:
        ax1.plot(df_mala["tau"], df_mala["acceptance"] * 100.0, "o--", color="#1f77b4", lw=2.4, ms=7, label="MALA")
    if not df_rl.empty:
        for i, a in enumerate(amps):
            sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
            ax1.plot(sub["tau"], sub["acceptance"] * 100.0, "s-", color=cmap[i], lw=1.8, ms=6, label=f"RL A={a:.2f}")
    ax1.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=12)
    ax1.set_ylabel("Aceptación MCMC (%)", fontsize=12)
    ax1.set_title("Evolución de Tasa de Aceptación MCMC vs $\tau$", fontsize=13, weight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")

    # Panel (1,0): ESJD vs tau
    ax2 = axes[1, 0]
    if not df_mala.empty:
        ax2.plot(df_mala["tau"], df_mala["ESJD"], "o--", color="#1f77b4", lw=2.4, ms=7, label="MALA")
    if not df_rl.empty:
        for i, a in enumerate(amps):
            sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
            ax2.plot(sub["tau"], sub["ESJD"], "s-", color=cmap[i], lw=1.8, ms=6, label=f"RL A={a:.2f}")
    ax2.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=12)
    ax2.set_ylabel("ESJD (Distancia Cuadrática Media de Salto)", fontsize=12)
    ax2.set_title("Eficiencia de Exploración de Espacio de Fase (ESJD) vs $\tau$", fontsize=13, weight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc="best")

    # Panel (1,1): tau_int(E) vs tau y REMARCADO DE LOS MEJORES MODELOS
    ax3 = axes[1, 1]
    if not df_mala.empty:
        ax3.plot(df_mala["tau"], df_mala["tau_int_E"], "o--", color="#1f77b4", lw=2.4, ms=7, label="MALA")
    if not df_rl.empty:
        for i, a in enumerate(amps):
            sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
            ax3.plot(sub["tau"], sub["tau_int_E"], "s-", color=cmap[i], lw=1.8, ms=6, label=f"RL A={a:.2f}")

    # Remarcado explícito con estrellas doradas y rojas
    if best_mala is not None:
        bm_tau = float(best_mala["tau"])
        bm_tint = float(best_mala["tau_int_E"])
        ax3.scatter([bm_tau], [bm_tint], color="gold", edgecolor="black", s=280, marker="*", zorder=10, label=f"★ Mejor MALA: tau={bm_tau:.2f} (tint={bm_tint:.1f})")
        ax3.annotate(f"Mejor MALA\n$\\tau={bm_tau:.2f}, \\tau_{{int}}={bm_tint:.1f}$",
                     xy=(bm_tau, bm_tint), xytext=(bm_tau, bm_tint + 6),
                     arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4", lw=1.5),
                     ha="center", fontsize=9.5, weight="bold")

    if best_rl is not None:
        br_tau = float(best_rl["tau"])
        br_tint = float(best_rl["tau_int_E"])
        br_amp = float(best_rl["actor_amplitude"])
        ax3.scatter([br_tau], [br_tint], color="crimson", edgecolor="black", s=280, marker="*", zorder=10, label=f"★ Mejor RL-MALA: A={br_amp:.2f}, tau={br_tau:.2f}")
        ax3.annotate(f"Mejor RL-MALA\n$A={br_amp:.2f}, \\tau={br_tau:.2f}$\n$\\tau_{{int}}={br_tint:.1f}$",
                     xy=(br_tau, br_tint), xytext=(br_tau, br_tint + 12),
                     arrowprops=dict(facecolor="crimson", shrink=0.08, width=1.5, headwidth=6),
                     bbox=dict(boxstyle="round,pad=0.3", fc="#ffe8e8", ec="crimson", lw=1.5),
                     ha="center", fontsize=9.5, weight="bold")

    ax3.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=12)
    ax3.set_ylabel(r"Tiempo de Autocorrelación $\tau_{\mathrm{int}}(E)$ (Pasos - Menor es Mejor)", fontsize=12)
    ax3.set_title(r"Descorrelación MCMC: $\tau_{\mathrm{int}}(E)$ vs $\tau$ (Mejores Modelos Destacados)", fontsize=13, weight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9.5, loc="best")

    fig1.tight_layout()
    fpath1 = os.path.join(fig_dir, "fig1_hyperparam_evolution_mala_vs_td3.png")
    fig1.savefig(fpath1, dpi=250)
    plt.close(fig1)
    print(f"[+] Guardada Figura 1 (Evolución de Hiperparámetros): {fpath1}")
except Exception as e:
    print(f"[-] Error al generar Figura 1: {e}")

# ------------------------------------------------------------------------------
# FIGURA 2: HEATMAPS 2D EN EL ESPACIO DE PARÁMETROS DEL ACTOR (TD3)
# ------------------------------------------------------------------------------
try:
    if not df_rl.empty and df_rl["actor_amplitude"].nunique() > 1 and df_rl["tau"].nunique() > 1:
        taus = np.sort(df_rl["tau"].dropna().unique())
        amps = np.sort(df_rl["actor_amplitude"].dropna().unique())

        fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Autocorrelación tau_int_E
        p_tint = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="tau_int_E", aggfunc="mean")
        im0 = axes2[0, 0].imshow(p_tint.values, cmap="magma_r", aspect="auto")
        fig2.colorbar(im0, ax=axes2[0, 0], label=r"$\tau_{\mathrm{int}}(E)$ (pasos)")
        axes2[0, 0].set_title(r"Autocorrelación $\tau_{\mathrm{int}}(E)$ (Menor es Mejor)", fontsize=12, weight="bold")

        # 2. Aceptación %
        p_acc = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="acceptance", aggfunc="mean") * 100.0
        im1 = axes2[0, 1].imshow(p_acc.values, cmap="RdYlGn", aspect="auto", vmin=40, vmax=95)
        fig2.colorbar(im1, ax=axes2[0, 1], label="Aceptación (%)")
        axes2[0, 1].set_title("Tasa de Aceptación MCMC (%)", fontsize=12, weight="bold")

        # 3. ESJD
        p_esjd = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="ESJD", aggfunc="mean")
        im2 = axes2[1, 0].imshow(p_esjd.values, cmap="viridis", aspect="auto")
        fig2.colorbar(im2, ax=axes2[1, 0], label="ESJD")
        axes2[1, 0].set_title("Distancia Cuadrática de Salto (ESJD)", fontsize=12, weight="bold")

        # 4. KL Total
        p_kl = df_rl.pivot_table(index="actor_amplitude", columns="tau", values="kl_casscf_total_mean", aggfunc="mean")
        im3 = axes2[1, 1].imshow(p_kl.values, cmap="plasma_r", aspect="auto")
        fig2.colorbar(im3, ax=axes2[1, 1], label=r"$D_{\mathrm{KL}}(\rho \parallel \rho_{\mathrm{CASSCF}})$")
        axes2[1, 1].set_title(r"Divergencia KL Total $D_{\mathrm{KL}}$", fontsize=12, weight="bold")

        # Resaltar la celda óptima en cada panel
        if best_rl is not None:
            opt_tau = float(best_rl["tau"])
            opt_amp = float(best_rl["actor_amplitude"])
            if opt_tau in taus and opt_amp in amps:
                c_idx = np.where(taus == opt_tau)[0][0]
                r_idx = np.where(amps == opt_amp)[0][0]
                for ax in axes2.flat:
                    rect = plt.Rectangle((c_idx - 0.45, r_idx - 0.45), 0.9, 0.9, fill=False, edgecolor="gold", lw=3.0)
                    ax.add_patch(rect)
                    ax.scatter([c_idx], [r_idx], color="gold", marker="*", s=220, zorder=5)

        for ax in axes2.flat:
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels([f"{t:.2f}" for t in taus], rotation=30)
            ax.set_yticks(range(len(amps)))
            ax.set_yticklabels([f"{a:.2f}" for a in amps])
            ax.set_xlabel(r"Tamaño de Paso $\tau$", fontsize=11)
            ax.set_ylabel("Amplitud del Actor ($A$)", fontsize=11)

        fig2.tight_layout()
        fpath2 = os.path.join(fig_dir, "fig2_2d_heatmaps_td3_parameters.png")
        fig2.savefig(fpath2, dpi=250)
        plt.close(fig2)
        print(f"[+] Guardada Figura 2 (Heatmaps 2D del Actor): {fpath2}")
except Exception as e:
    print(f"[-] Error al generar Figura 2: {e}")

# ------------------------------------------------------------------------------
# FIGURA 3: FRONTERA DE EFICIENCIA DE PARETO (ACCURACY VS TIME)
# ------------------------------------------------------------------------------
try:
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: tau_int vs Aceptación
    axp1 = axes3[0]
    if not df_mala.empty:
        axp1.scatter(df_mala["acceptance"] * 100.0, df_mala["tau_int_E"], color="#1f77b4", s=90, label="MALA Estándar", zorder=3)
        axp1.plot(df_mala["acceptance"] * 100.0, df_mala["tau_int_E"], color="#1f77b4", ls=":", lw=1.5, alpha=0.6)
    if not df_rl.empty:
        sc = axp1.scatter(df_rl["acceptance"] * 100.0, df_rl["tau_int_E"], c=df_rl["actor_amplitude"], cmap="plasma", s=90, label="RL-MALA TD3", zorder=3)
        plt.colorbar(sc, ax=axp1, label="Amplitud Actor ($A$)")

    if best_mala is not None:
        axp1.scatter([best_mala["acceptance"] * 100.0], [best_mala["tau_int_E"]], color="gold", edgecolor="black", s=320, marker="*", zorder=10, label="★ Mejor MALA")
    if best_rl is not None:
        axp1.scatter([best_rl["acceptance"] * 100.0], [best_rl["tau_int_E"]], color="crimson", edgecolor="black", s=320, marker="*", zorder=10, label="★ Mejor RL-MALA")

    axp1.set_xlabel("Aceptación MCMC (%)", fontsize=11)
    axp1.set_ylabel(r"Autocorrelación $\tau_{\mathrm{int}}(E)$ (Pasos - Menor es Mejor)", fontsize=11)
    axp1.set_title(r"Frontera Pareto: Autocorrelación vs Aceptación", fontsize=12, weight="bold")
    axp1.grid(True, alpha=0.3)
    axp1.legend(fontsize=9.5)

    # Panel 2: ESS por segundo vs Error Delta E vs CASSCF
    axp2 = axes3[1]
    if not df_mala.empty and "ESS_per_sec" in df_mala.columns and "delta_E_vs_CASSCF" in df_mala.columns:
        axp2.scatter(np.abs(df_mala["delta_E_vs_CASSCF"]), df_mala["ESS_per_sec"], color="#1f77b4", s=90, label="MALA", zorder=3)
    if not df_rl.empty and "ESS_per_sec" in df_rl.columns and "delta_E_vs_CASSCF" in df_rl.columns:
        axp2.scatter(np.abs(df_rl["delta_E_vs_CASSCF"]), df_rl["ESS_per_sec"], color="purple", s=90, label="RL-MALA TD3", zorder=3)

    if best_mala is not None and "ESS_per_sec" in best_mala:
        axp2.scatter([abs(best_mala["delta_E_vs_CASSCF"])], [best_mala["ESS_per_sec"]], color="gold", edgecolor="black", s=320, marker="*", zorder=10)
    if best_rl is not None and "ESS_per_sec" in best_rl:
        axp2.scatter([abs(best_rl["delta_E_vs_CASSCF"])], [best_rl["ESS_per_sec"]], color="crimson", edgecolor="black", s=320, marker="*", zorder=10)

    axp2.set_xlabel(r"$|\Delta E|$ vs CASSCF (Ha - Menor es Mejor)", fontsize=11)
    axp2.set_ylabel("Muestras Efectivas / Segundo (ESS/s - Mayor es Mejor)", fontsize=11)
    axp2.set_title("Frontera Pareto: Rendimiento (ESS/s) vs Precisión Energética", fontsize=12, weight="bold")
    axp2.grid(True, alpha=0.3)
    axp2.legend(fontsize=9.5)

    fig3.tight_layout()
    fpath3 = os.path.join(fig_dir, "fig3_pareto_sampling_efficiency.png")
    fig3.savefig(fpath3, dpi=250)
    plt.close(fig3)
    print(f"[+] Guardada Figura 3 (Frontera de Pareto): {fpath3}")
except Exception as e:
    print(f"[-] Error al generar Figura 3: {e}")

# ------------------------------------------------------------------------------
# FIGURA 4: EVALUACIONES DE MODELOS ENTRENADOS (MULTI-SJB)
# ------------------------------------------------------------------------------
try:
    if os.path.exists(eval_csv_path) and os.path.getsize(eval_csv_path) > 30:
        df_ev = pd.read_csv(eval_csv_path)
        for c in ["tau", "actor_amplitude", "E_mean", "SE_E", "acceptance", "ESJD", "tau_int_E", "ESS_per_sec", "delta_E_vs_CASSCF"]:
            if c in df_ev.columns:
                df_ev[c] = pd.to_numeric(df_ev[c], errors="coerce")

        if not df_ev.empty:
            fig4, ax4 = plt.subplots(1, 2, figsize=(15, 6))

            # Panel 1: Energía de modelos entrenados vs CASSCF
            agg_e = df_ev.groupby(["method", "tau"])["E_mean"].agg(["mean", "std"]).reset_index()
            for m in agg_e["method"].unique():
                sub = agg_e[agg_e["method"] == m]
                ax4[0].errorbar(sub["tau"], sub["mean"], yerr=sub["std"].fillna(0), marker="o", lw=2.2, capsize=4, label=f"Trained {m}")
            ax4[0].axhline(casscf_ref_e, color="black", ls=":", label=f"CASSCF Exact ({casscf_ref_e:.6f} Ha)")
            ax4[0].set_xlabel(r"Paso $\tau$", fontsize=11)
            ax4[0].set_ylabel(r"$\langle E \rangle$ (Ha)", fontsize=11)
            ax4[0].set_title("Energía Variacional de Modelos Entrenados Multi-SJB", fontsize=12, weight="bold")
            ax4[0].grid(True, alpha=0.3)
            ax4[0].legend(fontsize=9.5)

            # Panel 2: tau_int de modelos entrenados
            agg_t = df_ev.groupby(["method", "tau"])["tau_int_E"].agg(["mean", "std"]).reset_index()
            for m in agg_t["method"].unique():
                sub = agg_t[agg_t["method"] == m]
                ax4[1].errorbar(sub["tau"], sub["mean"], yerr=sub["std"].fillna(0), marker="s", lw=2.2, capsize=4, label=f"Trained {m}")
            ax4[1].set_xlabel(r"Paso $\tau$", fontsize=11)
            ax4[1].set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (pasos)", fontsize=11)
            ax4[1].set_title(r"Velocidad de Muestreo $\tau_{\mathrm{int}}$ en Modelos Entrenados", fontsize=12, weight="bold")
            ax4[1].grid(True, alpha=0.3)
            ax4[1].legend(fontsize=9.5)

            fig4.tight_layout()
            fpath4 = os.path.join(fig_dir, "fig4_trained_models_eval_comparison.png")
            fig4.savefig(fpath4, dpi=250)
            plt.close(fig4)
            print(f"[+] Guardada Figura 4 (Modelos Entrenados Multi-SJB): {fpath4}")
except Exception as e:
    print(f"[-] Aviso al procesar Figura 4: {e}")

# ------------------------------------------------------------------------------
# REPORTE DE TEXTO DE MEJORES MODELOS
# ------------------------------------------------------------------------------
report_path = os.path.join(fig_dir, "best_models_ranking.txt")
with open(report_path, "w") as rf:
    rf.write("==============================================================================\n")
    rf.write("       RANKING DE MEJORES CONFIGURACIONES EN PARAMETER SEARCH\n")
    rf.write("==============================================================================\n\n")
    if best_mala is not None:
        rf.write("[★] MEJOR CONFIGURACIÓN MALA ESTÁNDAR:\n")
        rf.write(f"    - Tamaño de Paso (tau)      : {best_mala.get('tau', np.nan):.4f}\n")
        rf.write(f"    - Energía Media <E>         : {best_mala.get('E_mean', np.nan):.6f} +/- {best_mala.get('SE_E', 0):.6f} Ha\n")
        rf.write(f"    - Tasa de Aceptación MCMC   : {best_mala.get('acceptance', 0)*100.0:.2f}%\n")
        rf.write(f"    - Autocorrelación tau_int   : {best_mala.get('tau_int_E', np.nan):.2f} pasos\n")
        rf.write(f"    - Distancia de Salto (ESJD) : {best_mala.get('ESJD', np.nan):.5f}\n")
        rf.write(f"    - Muestras Efectivas/seg    : {best_mala.get('ESS_per_sec', np.nan):.1f} ESS/s\n\n")

    if best_rl is not None:
        rf.write("[★] MEJOR CONFIGURACIÓN RL-MALA (TD3):\n")
        rf.write(f"    - Amplitud del Actor (A)    : {best_rl.get('actor_amplitude', np.nan):.4f}\n")
        rf.write(f"    - Tamaño de Paso (tau)      : {best_rl.get('tau', np.nan):.4f}\n")
        rf.write(f"    - Energía Media <E>         : {best_rl.get('E_mean', np.nan):.6f} +/- {best_rl.get('SE_E', 0):.6f} Ha\n")
        rf.write(f"    - Tasa de Aceptación MCMC   : {best_rl.get('acceptance', 0)*100.0:.2f}%\n")
        rf.write(f"    - Autocorrelación tau_int   : {best_rl.get('tau_int_E', np.nan):.2f} pasos\n")
        rf.write(f"    - Distancia de Salto (ESJD) : {best_rl.get('ESJD', np.nan):.5f}\n")
        rf.write(f"    - Muestras Efectivas/seg    : {best_rl.get('ESS_per_sec', np.nan):.1f} ESS/s\n\n")

    if best_mala is not None and best_rl is not None:
        tau_speedup = float(best_mala.get('tau_int_E', 1.0)) / max(1e-6, float(best_rl.get('tau_int_E', 1.0)))
        rf.write(f"[+] GANANCIA POR REINFORCEMENT LEARNING G_RL (tau_int MALA / tau_int RL): {tau_speedup:.2f}x\n")
    rf.write("==============================================================================\n")

print(f"[+] Reporte de mejores modelos guardado en: {report_path}")
PY_EOF

# Resumen de verificación de figuras generadas
TOTAL_FIGS=$(find "${FIG_OUT_DIR}" -type f -name "*.png" 2>/dev/null | wc -l)

log_master "=============================================================================="
log_master "    PARAMETER SEARCH COMPLETADO CON ÉXITO EN AYLER"
log_master "    Directorio Base    : ${BASE_OUT_DIR}"
log_master "    CSV Maestro        : ${MASTER_SUMMARY_CSV}"
log_master "    Evaluaciones CSV   : ${EVAL_SUMMARY_CSV}"
log_master "    Directorio Figuras : ${FIG_OUT_DIR}/ (${TOTAL_FIGS} archivos PNG)"
log_master "    Ranking y Reporte  : ${FIG_OUT_DIR}/best_models_ranking.txt"
log_master "    Fecha y Hora       : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "=============================================================================="
