#!/usr/bin/env bash
# ==============================================================================
# run_parameter_search_ayler.sh
# Parameter Search Automatizado: Actor Amplitude, Tau Size y Benchmark con MALA
# Sistema Objetivo: CASSCF 20 Determinantes (Base 256), N=6, w=0.1, g=1.0, a=0.1
# Optimizado para ejecución en Ayler (Cluster / Workstation con GPU o CPU)
# ==============================================================================

set -uo pipefail

export JAX_ENABLE_X64=True
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 1. Configuración de Rutas y Sistema Objetivo
TARGET_CASSCF_NPZ="figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
BASE_OUT_DIR="results_param_search"
CSV_COLLECT_DIR="${BASE_OUT_DIR}/csv_summary"
FIG_OUT_DIR="${BASE_OUT_DIR}/figures"
LOG_DIR="${BASE_OUT_DIR}/logs"

mkdir -p "${BASE_OUT_DIR}" "${CSV_COLLECT_DIR}" "${FIG_OUT_DIR}" "${LOG_DIR}"

MASTER_LOG="${LOG_DIR}/master_parameter_search.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

echo "=============================================================================="
echo "    INICIO DE PARAMETER SEARCH EN AYLER: ACTOR AMPLITUDE & TAU SIZE"
echo "    Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "    Sistema     : N=6 (3 up, 3 dn), w=0.1, g=1.0, softening=0.1 (CASSCF 20-det)"
echo "    Referencia  : ${TARGET_CASSCF_NPZ}"
echo "    Python Bin  : ${PYTHON_BIN}"
echo "=============================================================================="

if [ ! -f "${TARGET_CASSCF_NPZ}" ]; then
    echo "[-] ERROR: No se encontró el archivo CASSCF objetivo en: ${TARGET_CASSCF_NPZ}"
    exit 1
fi

# 2. Hiperparámetros de Producción y Muestreo
N_BURN_IN=2500
MEASURE_STEPS=8000
BATCH_SIZE=1024

# Rejilla de Búsqueda
TAU_MALA_LIST=(0.02 0.04 0.06 0.08 0.10 0.12 0.15)
TAU_ACTOR_LIST=(0.02 0.04 0.06 0.08 0.10 0.12)
ACTOR_AMP_LIST=(0.05 0.10 0.20 0.40 0.80)

MASTER_SUMMARY_CSV="${BASE_OUT_DIR}/master_param_search_summary.csv"
echo "method,tau,actor_amplitude,label,h5_path,E_mean,SE_E,acceptance,ESJD,tau_int_E,G_RL_speedup,ESS_per_sec,casscf_energy,delta_E_vs_CASSCF,kl_casscf_total_mean,kl_casscf_total_std,kl_casscf_up_mean,kl_casscf_down_mean" > "${MASTER_SUMMARY_CSV}"

# Función auxiliar de log con tiempos y PIDs
log_step_info() {
    local script_name="$1"
    local start_epoch="$2"
    local pid="$3"
    local end_epoch
    end_epoch=$(date +%s)
    local elapsed=$(( end_epoch - start_epoch ))
    local end_time
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo "  [INFO] Script: ${script_name} | PID: ${pid} | Fin: ${end_time} | Duración: ${elapsed}s"
}

# ==============================================================================
# FASE 1: BARRIDO DE MALA ESTÁNDAR (Variando tau_mala)
# ==============================================================================
echo -e "\n>>> FASE 1: BARRIDO UNIDIMENSIONAL DE MALA ESTÁNDAR (${#TAU_MALA_LIST[@]} valores)..."

for tau in "${TAU_MALA_LIST[@]}"; do
    RUN_TAG="MALA_tau${tau}"
    MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
    POST_DIR="${MODEL_DIR}/postprocess"
    mkdir -p "${MODEL_DIR}" "${POST_DIR}"
    
    H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
    echo -e "\n------------------------------------------------------------------------------"
    echo " [*] Ejecutando MALA con tau=${tau}..."
    START_T=$(date +%s)
    
    ${PYTHON_BIN} -u cli/run_1d.py \
        --method MALA \
        --ansatz_type casscf_npz \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
        --tau "${tau}" \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_h5 "${H5_FILE}" &
    PID=$!
    wait ${PID}
    log_step_info "run_1d.py (MALA tau=${tau})" "${START_T}" "${PID}"
    
    # Post-procesamiento
    START_POST=$(date +%s)
    ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "MALA (tau=${tau})" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_observables_comparison &
    PID_POST=$!
    wait ${PID_POST}
    log_step_info "postprocess.py (${RUN_TAG})" "${START_POST}" "${PID_POST}"
    
    # Anexar métricas al CSV maestro
    SUM_CSV="${POST_DIR}/summary_metrics.csv"
    if [ -f "${SUM_CSV}" ]; then
        tail -n +2 "${SUM_CSV}" | awk -v m="MALA" -v t="${tau}" -v a="0.0" -F',' '{OFS=","; print m, t, a, $0}' >> "${MASTER_SUMMARY_CSV}"
    fi
done

# ==============================================================================
# FASE 2: REJILLA 2D DE RL-MALA (TD3 Actor: tau vs actor_amplitude)
# ==============================================================================
TOTAL_RL_RUNS=$(( ${#TAU_ACTOR_LIST[@]} * ${#ACTOR_AMP_LIST[@]} ))
echo -e "\n>>> FASE 2: REJILLA 2D DE RL-MALA (TD3 ACTOR: ${TOTAL_RL_RUNS} combinaciones)..."

for amp in "${ACTOR_AMP_LIST[@]}"; do
    for tau in "${TAU_ACTOR_LIST[@]}"; do
        RUN_TAG="RL_MALA_amp${amp}_tau${tau}"
        MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
        POST_DIR="${MODEL_DIR}/postprocess"
        mkdir -p "${MODEL_DIR}" "${POST_DIR}"
        
        H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"
        echo -e "\n------------------------------------------------------------------------------"
        echo " [*] Ejecutando RL-MALA TD3 con actor_amplitude=${amp}, tau=${tau}..."
        START_T=$(date +%s)
        
        ${PYTHON_BIN} -u cli/run_1d.py \
            --method TD3 \
            --ansatz_type casscf_npz \
            --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
            --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
            --tau "${tau}" \
            --actor_amplitude "${amp}" \
            --n_ter "${N_BURN_IN}" \
            --measure_steps "${MEASURE_STEPS}" \
            --batch_size "${BATCH_SIZE}" \
            --output_h5 "${H5_FILE}" &
        PID=$!
        wait ${PID}
        log_step_info "run_1d.py (RL-MALA amp=${amp} tau=${tau})" "${START_T}" "${PID}"
        
        # Post-procesamiento
        START_POST=$(date +%s)
        ${PYTHON_BIN} -u cli/postprocess.py \
            --inputs "${H5_FILE}" \
            --labels "RL-MALA (A=${amp}, tau=${tau})" \
            --out_dir "${POST_DIR}" \
            --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
            --system_tag "${RUN_TAG}" \
            --csv_collect_dir "${CSV_COLLECT_DIR}" \
            --plot_observables_comparison &
        PID_POST=$!
        wait ${PID_POST}
        log_step_info "postprocess.py (${RUN_TAG})" "${START_POST}" "${PID_POST}"
        
        # Anexar métricas al CSV maestro
        SUM_CSV="${POST_DIR}/summary_metrics.csv"
        if [ -f "${SUM_CSV}" ]; then
            tail -n +2 "${SUM_CSV}" | awk -v m="TD3" -v t="${tau}" -v a="${amp}" -F',' '{OFS=","; print m, t, a, $0}' >> "${MASTER_SUMMARY_CSV}"
        fi
    done
done

# ==============================================================================
# FASE 3: PIPELINE 3 CON ANSATZ MULTIDETERMINANTE (Multi-SJB)
# ==============================================================================
echo -e "\n>>> FASE 3: PIPELINE 3 - ENTRENAMIENTO VARIACIONAL VMC (Multi-SJB con 20 det)..."

P3_MODEL_DIR="${BASE_OUT_DIR}/pipeline3_multi_sjb"
P3_MALA_DIR="${P3_MODEL_DIR}/VMC_MALA_MultiSJB"
P3_RL_DIR="${P3_MODEL_DIR}/Coupled_RL_MultiSJB"
mkdir -p "${P3_MALA_DIR}" "${P3_RL_DIR}"

H5_P3_MALA="${P3_MALA_DIR}/run_bundle_P3_MALA_MultiSJB.h5"
H5_P3_RL="${P3_RL_DIR}/run_bundle_P3_RL_MultiSJB.h5"

echo -e "\n [*] Entrenando VMC + MALA con Multi-SJB (20 determinantes + Jastrow + Backflow)..."
START_T=$(date +%s)
${PYTHON_BIN} -u cli/train_1d.py \
    --method MALA \
    --ansatz_type multi_sjb \
    --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
    --correlation_model spin_rbf_jastrow \
    --orbital_model backflow_ho \
    --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
    --tau 0.08 \
    --vmc_train_steps 1500 \
    --n_ter "${N_BURN_IN}" \
    --measure_steps "${MEASURE_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_h5 "${H5_P3_MALA}" &
PID=$!
wait ${PID}
log_step_info "train_1d.py (VMC+MALA Multi-SJB)" "${START_T}" "${PID}"

echo -e "\n [*] Entrenando Coupled VMC + RL-MALA TD3 con Multi-SJB..."
START_T=$(date +%s)
${PYTHON_BIN} -u cli/train_1d.py \
    --method TD3 \
    --ansatz_type multi_sjb \
    --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
    --correlation_model spin_rbf_jastrow \
    --orbital_model backflow_ho \
    --n_elec 6 --n_up 3 --n_dn 3 --omega 0.1 --interaction_strength 1.0 --softening 0.1 \
    --tau 0.08 \
    --actor_amplitude 0.20 \
    --vmc_train_steps 1500 \
    --n_ter "${N_BURN_IN}" \
    --measure_steps "${MEASURE_STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --output_h5 "${H5_P3_RL}" &
PID=$!
wait ${PID}
log_step_info "train_1d.py (Coupled RL Multi-SJB)" "${START_T}" "${PID}"

# Post-procesamiento comparativo de Pipeline 3
P3_POST_DIR="${P3_MODEL_DIR}/postprocess"
${PYTHON_BIN} -u cli/postprocess.py \
    --inputs "${H5_P3_MALA}" "${H5_P3_RL}" \
    --labels "VMC + MALA (Multi-SJB)" "Coupled VMC + RL-MALA (Multi-SJB)" \
    --out_dir "${P3_POST_DIR}" \
    --casscf_npz_path "${TARGET_CASSCF_NPZ}" \
    --system_tag "P3_MultiSJB_N6_w0.1_g1.0" \
    --csv_collect_dir "${CSV_COLLECT_DIR}" \
    --plot_training_metrics \
    --plot_observables_comparison

# ==============================================================================
# FASE 4: AGREGACIÓN DE RESULTADOS Y GENERACIÓN DE FIGURAS 2D
# ==============================================================================
echo -e "\n>>> FASE 4: GENERACIÓN DE HEATMAPS 2D Y ANÁLISIS DE PARETO..."
${PYTHON_BIN} -u scripts/plot_param_search_results.py \
    --csv "${MASTER_SUMMARY_CSV}" \
    --out_dir "${FIG_OUT_DIR}"

echo "=============================================================================="
echo "    PARAMETER SEARCH COMPLETADO CON ÉXITO EN AYLER"
echo "    CSV Maestro  : ${MASTER_SUMMARY_CSV}"
echo "    Figuras 2D   : ${FIG_OUT_DIR}"
echo "    Fecha y Hora : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=============================================================================="
