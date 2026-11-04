#!/usr/bin/env bash
# ==============================================================================
# run_parameter_search_baker.sh
# ==============================================================================
# Parameter Search Condensado para BAKER (Servidor de 1 Sola GPU)
#
# Objetivo:
#   Optimizar sistemáticamente el Actor Neuronal (TD3) frente a MALA para romper
#   la barrera de G_RL ~ 1.0 observada en los benchmarks previos, explorando:
#     1. Amplitudes del actor en el rango óptimo (A in [0.03, 0.08, 0.16, 0.30]).
#     2. Régimen de paso temporal tau (pequeño vs intermedio vs grande).
#     3. Funciones de recompensa: CDLB vs Log-ESJD vs Barker.
#
# Optimizaciones para 1 Sola GPU (Baker):
#   - Concurrencia secuencial estricta (1 job por GPU).
#   - Entrenamiento condensado: 4 macro-ciclos de 100 VMC + 50 RL (600 pasos totales).
#   - Muestreo rápido y de bajo consumo: batch_size=256, measure_steps=2500.
#   - Reanudación inteligente: Salta cualquier corrida previamente calculada.
#   - Generación automática de todas las figuras en formato PNG al finalizar.
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

# Gestión conservadora de VRAM para servidor de 1 GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.35}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_DEVICE_MAX_CONNECTIONS=1

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# 2. Configuración de Rutas y Directorios
# ------------------------------------------------------------------------------
BASE_OUT_DIR="${1:-${BASE_OUT_DIR:-results_baker_search}}"
LOG_DIR="${BASE_OUT_DIR}/logs"
CSV_COLLECT_DIR="${BASE_OUT_DIR}/csv_summary"
FIG_OUT_DIR="${BASE_OUT_DIR}/figures"
TRACKER_DIR="${BASE_OUT_DIR}/.tracker"

mkdir -p "${BASE_OUT_DIR}" "${LOG_DIR}" "${CSV_COLLECT_DIR}" "${FIG_OUT_DIR}" "${TRACKER_DIR}"
rm -f "${TRACKER_DIR}"/job_*.meta 2>/dev/null || true

MASTER_LOG="${LOG_DIR}/master_baker_search.log"

log_master() {
    local msg="$1"
    echo "$msg"
    echo "$msg" >> "${MASTER_LOG}"
}

# Detección de estados CASSCF de referencia
TARGET_CASSCF_NPZ="${TARGET_CASSCF_NPZ:-figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz}"
if [ ! -f "${TARGET_CASSCF_NPZ}" ]; then
    for cand in \
        "cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" \
        "data/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" \
        "figures_allmodels_benchmarks/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"; do
        if [ -f "$cand" ]; then
            TARGET_CASSCF_NPZ="$cand"
            break
        fi
    done
fi

REF_DENSITY_CSV="${REF_DENSITY_CSV:-results_csv_b2/casscf_exact_density_N6_w0.1_g1.0.csv}"
REF_CSV_ARG=""
if [ -f "${REF_DENSITY_CSV}" ]; then
    REF_CSV_ARG="--reference_density_csv ${REF_DENSITY_CSV}"
fi

# ------------------------------------------------------------------------------
# 3. Configuración de 1 Sola GPU en Baker
# ------------------------------------------------------------------------------
AVAILABLE_GPUS=(0)
JOBS_PER_GPU=1
MAX_TOTAL_JOBS=1

# ------------------------------------------------------------------------------
# 4. Hiperparámetros Condensados y Grillas Focalizadas
# ------------------------------------------------------------------------------
N_BURN_IN="${N_BURN_IN:-1200}"
MEASURE_STEPS="${MEASURE_STEPS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-500}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-10000}"

# Parámetros de Entrenamiento Condensado
CONDENSED_TRAIN_CYCLES=4
VMC_STEPS_PER_CYCLE=100
RL_SWEEPS_PER_CYCLE=50
CONDENSED_TOTAL_STEPS=$(( CONDENSED_TRAIN_CYCLES * VMC_STEPS_PER_CYCLE ))

# Grillas Focalizadas para romper G_RL ~ 1.0:
# A. Barrido fino de MALA (referencia exacta)
TAU_MALA_LIST=(0.02 0.04 0.08 0.12)

# B. Amplitudes del Actor y pasos temporales
ACTOR_AMP_LIST=(0.03 0.08 0.16 0.30)
TAU_ACTOR_LIST=(0.04 0.08 0.12)

# Encabezado Unificado de 41 Columnas
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
log_master "    INICIO DE PARAMETER SEARCH EN BAKER (MODO 1 GPU CONDENSADO)"
log_master "    Fecha y Hora        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "    Directorio Base     : ${BASE_OUT_DIR}"
log_master "    Estado Referencia   : ${TARGET_CASSCF_NPZ}"
log_master "    Configuración MCMC  : Batch=${BATCH_SIZE}, Pasos=${MEASURE_STEPS}, Burn-in=${N_BURN_IN}"
log_master "    Entrenamiento VMC   : ${CONDENSED_TRAIN_CYCLES} ciclos (${CONDENSED_TOTAL_STEPS} pasos VMC, $(( CONDENSED_TRAIN_CYCLES * RL_SWEEPS_PER_CYCLE )) sweeps RL)"
log_master "    Amplitudes Actor    : [${ACTOR_AMP_LIST[*]}]"
log_master "    Taus Evaluados      : [${TAU_ACTOR_LIST[*]}]"
log_master "=============================================================================="

# ------------------------------------------------------------------------------
# 5. Funciones Auxiliares del Gestor de Cola de 1 GPU
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
            local gpu sys start_epoch start_time script_name sys_log_file sum_csv method_tag tau_val amp_val post_dir_path target_csv
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

            # Inserción atómica y segura al CSV maestro
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

            # Centralización inmediata de figuras en figures/
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
            log_master "   Duración           : ${duration}s ($duration_formatted)"
            log_master "   Log en Tiempo Real : ${LOG_DIR}/${sys_log_file}"
            log_master "=============================================================================="

            rm -f "$f"
        fi
    done
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

ALLOCATED_GPU="0"

wait_for_free_gpu_slot() {
    local stage_name="$1"
    while true; do
        cleanup_finished_jobs "$stage_name"
        local total_active
        total_active=$(get_total_active_jobs)
        if [ "$total_active" -lt "$MAX_TOTAL_JOBS" ]; then
            ALLOCATED_GPU="0"
            return 0
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
# 6. Detección Inteligente de Corridas Previas (Reanudación)
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

    if [ -f "${H5_FILE}" ] && [ -s "${H5_FILE}" ]; then
        local num_figs
        num_figs=$(find "${POST_DIR}" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l)
        if [ ! -f "${SUM_CSV}" ] || [ "$num_figs" -eq 0 ]; then
            log_master "[*] Completando post-procesamiento faltante para: ${RUN_TAG}..."
            ${PYTHON_BIN} -u cli/postprocess.py \
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
                --figure_format png \
                --dpi 200 >/dev/null 2>&1 || true
        fi

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

        log_master "[OMITIDO | YA EXISTE] ${RUN_TAG} completado previamente. Saltando cómputo."
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------------------
# 7. Funciones de Ejecución de Tareas (Workers)
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

    # Muestreo con JAX
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

    # Post-procesamiento
    ${PYTHON_BIN} -u cli/postprocess.py \
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
        --figure_format png \
        --dpi 200

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

    # Entrenamiento condensado para 1 sola GPU
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
        --coupled_training_cycles "${CONDENSED_TRAIN_CYCLES}" \
        --vmc_iterations_per_cycle "${VMC_STEPS_PER_CYCLE}" \
        --rl_sweeps_per_cycle "${RL_SWEEPS_PER_CYCLE}" \
        --train_steps "${CONDENSED_TOTAL_STEPS}" \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --jit_chunk_size "${JIT_CHUNK_SIZE}" \
        --eloc_chunk_size "${ELOC_CHUNK_SIZE}" \
        --output_h5 "${H5_FILE}"

    ${PYTHON_BIN} -u cli/postprocess.py \
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
        --figure_format png \
        --dpi 200

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

    ${PYTHON_BIN} -u cli/postprocess.py \
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
        --figure_format png \
        --dpi 200

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
# FASE 1: MUESTREO PURO CONDENSADO (MALA vs RL-MALA en CASSCF Congelado)
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 1: MUESTREO PURO CONDENSADO EN 1 GPU"
log_master "    - MALA Referencia : ${#TAU_MALA_LIST[@]} valores de tau"
log_master "    - Rejilla RL-MALA : ${#ACTOR_AMP_LIST[@]} amplitudes x ${#TAU_ACTOR_LIST[@]} taus"
log_master "=============================================================================="

# 1A. MALA
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

    wait_for_free_gpu_slot "FASE 1: MALA"
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
    log_master ">> [FASE 1 | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
done

# 1B. RL-MALA (Barrido 2D focalizado)
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

        wait_for_free_gpu_slot "FASE 1: RL-MALA"
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
        log_master ">> [FASE 1 | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
    done
done

log_master ""
log_master "[*] Esperando finalización de la Fase 1..."
wait_all_stage_jobs "FASE 1: MUESTREO PURO"
log_master "[+] FASE 1 COMPLETADA CON ÉXITO."

# ==============================================================================
# FASE 2: ENTRENAMIENTO VARIACIONAL MULTI-SJB CONDENSADO
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 2: ENTRENAMIENTO VARIACIONAL MULTI-SJB CONDENSADO (4 CICLOS)"
log_master "=============================================================================="

# 2A. VMC + MALA
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
    log_master ">> [FASE 2 | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
fi

# 2B. Coupled VMC + RL-MALA TD3
RUN_TAG="Coupled_RL_MultiSJB"
SYS_LOG="train_${RUN_TAG}.log"
MODEL_DIR="${BASE_OUT_DIR}/pipeline3_multi_sjb/${RUN_TAG}"
POST_DIR="${MODEL_DIR}/postprocess"
H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
LABEL="Coupled VMC + RL-MALA (Multi-SJB)"

if ! check_and_register_completed_run "$RUN_TAG" "Coupled_RL_MultiSJB" "0.08" "0.08" "$H5_FILE" "$POST_DIR" "$LABEL"; then
    wait_for_free_gpu_slot "FASE 2: ENTRENAMIENTO"
    GPU_ID="$ALLOCATED_GPU"

    start_epoch=$(date +%s)
    start_time=$(date "+%Y-%m-%d %H:%M:%S")

    run_single_training_worker "$GPU_ID" "TD3" "coupled" "0.08" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
    worker_pid=$!

    cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="Coupled_RL_MultiSJB"
tau_val="0.08"
amp_val="0.08"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_training_worker Coupled_RL"
sys_log_file="$SYS_LOG"
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${MASTER_SUMMARY_CSV}"
EOF
    log_master ">> [FASE 2 | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
fi

log_master ""
log_master "[*] Esperando finalización de la Fase 2..."
wait_all_stage_jobs "FASE 2: ENTRENAMIENTO"
log_master "[+] FASE 2 COMPLETADA CON ÉXITO."

# ==============================================================================
# FASE 2B: EVALUACIONES DE MODELOS ENTRENADOS
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 2B: EVALUACIONES RÁPIDAS DE MODELOS ENTRENADOS"
log_master "=============================================================================="

TRAINED_VMC_BUNDLE="${BASE_OUT_DIR}/pipeline3_multi_sjb/VMC_MALA_MultiSJB/run_bundle_VMC_MALA_MultiSJB.h5"
TRAINED_RL_BUNDLE="${BASE_OUT_DIR}/pipeline3_multi_sjb/Coupled_RL_MultiSJB/run_bundle_Coupled_RL_MultiSJB.h5"

EVAL_TAU_LIST=(0.04 0.08)
EVAL_SEEDS=(42 123)

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

            wait_for_free_gpu_slot "FASE 2B: EVAL"
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
            log_master ">> [FASE 2B | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
        done
    done
fi

if [ -f "${TRAINED_RL_BUNDLE}" ] && [ -s "${TRAINED_RL_BUNDLE}" ]; then
    for tau in "${EVAL_TAU_LIST[@]}"; do
        for seed_idx in "${EVAL_SEEDS[@]}"; do
            RUN_TAG="eval_MultiSJB_TD3_tau${tau}_s${seed_idx}"
            SYS_LOG="eval_${RUN_TAG}.log"
            MODEL_DIR="${BASE_OUT_DIR}/evaluations_trained/${RUN_TAG}"
            POST_DIR="${MODEL_DIR}/postprocess"
            H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
            LABEL="Multi-SJB TD3 (A=0.08, tau=${tau}, s=${seed_idx})"

            if check_and_register_completed_run "$RUN_TAG" "TD3_Trained" "$tau" "0.08" "$H5_FILE" "$POST_DIR" "$LABEL" "${EVAL_SUMMARY_CSV}"; then
                continue
            fi

            wait_for_free_gpu_slot "FASE 2B: EVAL"
            GPU_ID="$ALLOCATED_GPU"

            start_epoch=$(date +%s)
            start_time=$(date "+%Y-%m-%d %H:%M:%S")

            run_single_eval_worker "$GPU_ID" "TD3" "$tau" "0.08" "${TRAINED_RL_BUNDLE}" "$seed_idx" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            worker_pid=$!

            cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
method_tag="TD3_Trained"
tau_val="$tau"
amp_val="0.08"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_single_eval_worker TD3 tau=$tau seed=$seed_idx"
sys_log_file="$SYS_LOG"
post_dir_path="${POST_DIR}"
sum_csv="${POST_DIR}/summary_metrics.csv"
target_csv="${EVAL_SUMMARY_CSV}"
EOF
            log_master ">> [FASE 2B | ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
        done
    done
fi

log_master ""
log_master "[*] Esperando finalización de la Fase 2B..."
wait_all_stage_jobs "FASE 2B: EVALUACIONES"
log_master "[+] FASE 2B COMPLETADA CON ÉXITO."

# ==============================================================================
# FASE 3: SÍNTESIS GRÁFICA Y RANKING DE MEJORES MODELOS
# ==============================================================================
log_master ""
log_master "=============================================================================="
log_master ">>> FASE 3: SÍNTESIS GRÁFICA Y RANKING EJECUTIVO (BAKER)"
log_master "=============================================================================="

${PYTHON_BIN} -u - << 'PY_EOF'
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base_dir = os.environ.get("BASE_OUT_DIR", "results_baker_search")
fig_dir = os.path.join(base_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)

csv_path = os.path.join(base_dir, "master_param_search_summary.csv")
eval_csv_path = os.path.join(base_dir, "evaluations_trained_summary.csv")

if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 30:
    print("[-] CSV maestro no encontrado o vacío.")
    sys.exit(0)

df = pd.read_csv(csv_path)
num_cols = ["tau", "actor_amplitude", "E_mean", "SE_E", "acceptance", "ESJD", "tau_int_E", "ESS_per_sec", "delta_E_vs_CASSCF", "kl_casscf_total_mean"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df_mala = df[df["method"].str.upper() == "MALA"].copy().sort_values("tau")
df_rl = df[df["method"].str.upper().str.contains("TD3|RL")].copy()

best_mala = df_mala.loc[df_mala["tau_int_E"].idxmin()] if not df_mala.empty and df_mala["tau_int_E"].notna().any() else None
best_rl = df_rl.loc[df_rl["tau_int_E"].idxmin()] if not df_rl.empty and df_rl["tau_int_E"].notna().any() else None

# FIGURA 1: Comparativa MALA vs RL-MALA
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
casscf_ref_e = 4.524052

# Panel (0,0): Energía
ax0 = axes[0, 0]
if not df_mala.empty:
    ax0.plot(df_mala["tau"], df_mala["E_mean"], "o-", color="#1f77b4", lw=2.2, label="MALA Estándar")
if not df_rl.empty:
    amps = sorted(df_rl["actor_amplitude"].dropna().unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(amps)))
    for i, a in enumerate(amps):
        sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
        ax0.plot(sub["tau"], sub["E_mean"], "s--", color=cmap[i], lw=1.8, label=f"RL-MALA (A={a:.2f})")
ax0.axhline(casscf_ref_e, color="black", ls=":", label=f"CASSCF Exact ({casscf_ref_e:.5f} Ha)")
ax0.set_xlabel(r"Tamaño de Paso $\tau$")
ax0.set_ylabel(r"$\langle E \rangle$ (Ha)")
ax0.set_title("Energía Variacional vs $\tau$", weight="bold")
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=9)

# Panel (0,1): Aceptación
ax1 = axes[0, 1]
ax1.axhspan(70, 90, color="green", alpha=0.12, label="Rango Óptimo (70-90%)")
if not df_mala.empty:
    ax1.plot(df_mala["tau"], df_mala["acceptance"] * 100.0, "o-", color="#1f77b4", lw=2.2, label="MALA")
if not df_rl.empty:
    for i, a in enumerate(amps):
        sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
        ax1.plot(sub["tau"], sub["acceptance"] * 100.0, "s--", color=cmap[i], lw=1.8, label=f"RL A={a:.2f}")
ax1.set_xlabel(r"Tamaño de Paso $\tau$")
ax1.set_ylabel("Aceptación MCMC (%)")
ax1.set_title("Tasa de Aceptación vs $\tau$", weight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Panel (1,0): ESJD
ax2 = axes[1, 0]
if not df_mala.empty:
    ax2.plot(df_mala["tau"], df_mala["ESJD"], "o-", color="#1f77b4", lw=2.2, label="MALA")
if not df_rl.empty:
    for i, a in enumerate(amps):
        sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
        ax2.plot(sub["tau"], sub["ESJD"], "s--", color=cmap[i], lw=1.8, label=f"RL A={a:.2f}")
ax2.set_xlabel(r"Tamaño de Paso $\tau$")
ax2.set_ylabel("ESJD (Salto Cuadrático Medio)")
ax2.set_title("Movilidad Espacial (ESJD) vs $\tau$", weight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)

# Panel (1,1): Autocorrelación con Estrellas en Mejores Modelos
ax3 = axes[1, 1]
if not df_mala.empty:
    ax3.plot(df_mala["tau"], df_mala["tau_int_E"], "o-", color="#1f77b4", lw=2.2, label="MALA")
if not df_rl.empty:
    for i, a in enumerate(amps):
        sub = df_rl[df_rl["actor_amplitude"] == a].sort_values("tau")
        ax3.plot(sub["tau"], sub["tau_int_E"], "s--", color=cmap[i], lw=1.8, label=f"RL A={a:.2f}")

if best_mala is not None:
    bm_tau = float(best_mala["tau"])
    bm_tint = float(best_mala["tau_int_E"])
    ax3.scatter([bm_tau], [bm_tint], color="gold", edgecolor="black", s=250, marker="*", zorder=10, label=f"★ Mejor MALA: tau={bm_tau:.2f} (tint={bm_tint:.1f})")

if best_rl is not None:
    br_tau = float(best_rl["tau"])
    br_tint = float(best_rl["tau_int_E"])
    br_amp = float(best_rl["actor_amplitude"])
    ax3.scatter([br_tau], [br_tint], color="crimson", edgecolor="black", s=250, marker="*", zorder=10, label=f"★ Mejor RL: A={br_amp:.2f}, tau={br_tau:.2f} (tint={br_tint:.1f})")

ax3.set_xlabel(r"Tamaño de Paso $\tau$")
ax3.set_ylabel(r"$\tau_{\mathrm{int}}(E)$ (Pasos - Menor es Mejor)")
ax3.set_title(r"Descorrelación $\tau_{\mathrm{int}}(E)$ (Mejores Modelos Destacados)", weight="bold")
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)

fig.tight_layout()
out_f1 = os.path.join(fig_dir, "fig1_baker_hyperparam_evolution.png")
fig.savefig(out_f1, dpi=250)
plt.close(fig)
print(f"[+] Figura guardada: {out_f1}")

# Reporte en texto
rep_file = os.path.join(fig_dir, "baker_best_models_ranking.txt")
with open(rep_file, "w") as f:
    f.write("==============================================================================\n")
    f.write("      RANKING DE MEJORES CONFIGURACIONES EN BAKER (1 GPU CONDENSADO)\n")
    f.write("==============================================================================\n\n")
    if best_mala is not None:
        f.write(f"[★] MEJOR MALA: tau={best_mala.get('tau', np.nan):.4f} | tau_int={best_mala.get('tau_int_E', np.nan):.2f} | Acc={best_mala.get('acceptance', 0)*100:.1f}% | ESS/s={best_mala.get('ESS_per_sec', 0):.1f}\n")
    if best_rl is not None:
        f.write(f"[★] MEJOR RL-MALA: A={best_rl.get('actor_amplitude', np.nan):.4f} | tau={best_rl.get('tau', np.nan):.4f} | tau_int={best_rl.get('tau_int_E', np.nan):.2f} | Acc={best_rl.get('acceptance', 0)*100:.1f}% | ESS/s={best_rl.get('ESS_per_sec', 0):.1f}\n")
    if best_mala is not None and best_rl is not None:
        g_speedup = float(best_mala.get('tau_int_E', 1.0)) / max(1e-5, float(best_rl.get('tau_int_E', 1.0)))
        f.write(f"\n[+] GANANCIA POR REINFORCEMENT LEARNING G_RL: {g_speedup:.2f}x\n")
    f.write("==============================================================================\n")
print(f"[+] Reporte guardado: {rep_file}")
PY_EOF

log_master "=============================================================================="
log_master "    PARAMETER SEARCH EN BAKER COMPLETADO CON ÉXITO"
log_master "    Directorio Base    : ${BASE_OUT_DIR}"
log_master "    CSV Maestro        : ${MASTER_SUMMARY_CSV}"
log_master "    Directorio Figuras : ${FIG_OUT_DIR}/"
log_master "    Reporte y Ranking  : ${FIG_OUT_DIR}/baker_best_models_ranking.txt"
log_master "    Fecha y Hora       : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "=============================================================================="
