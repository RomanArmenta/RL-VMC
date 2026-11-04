#!/usr/bin/env bash
# ==============================================================================
# run_targeted_pipeline2_grid_search.sh
# ==============================================================================
# Grid Search Focalizado de Pipeline 2 (Muestreo Puro sobre CASSCF Congelado)
#
# Características Principales:
#   1. Sistemas seleccionados fácilmente modificables:
#      Default: [N=4, g=0.5], [N=4, g=1.0], [N=10, g=0.5], [N=10, g=5.0]
#   2. Grid search sobre 'actor_amplitude' para RL-MALA (TD3 Actor).
#   3. Grid search sobre 'tau_mala' para MALA Estándar.
#   4. Reutiliza los estados CASSCF generados en el Pipeline 1.
#   5. Sistema de colas dinámico idéntico a los benchmarks (despacho asíncrono
#      de slots libres en GPU, tracking con archivos .meta, control de concurrencia).
#   6. Logs en TIEMPO REAL en la carpeta logs/ (PYTHONUNBUFFERED=1, python3 -u,
#      redirección desacoplada a archivos de log individuales).
#   7. Master log detallado con tiempos de corrida (inicio, fin, PID, duración)
#      y tabla ejecutiva de resumen en términos de precisión física y muestreo.
# ==============================================================================

set -u

# ------------------------------------------------------------------------------
# 1. Configuración de Sistemas Objetivo (FÁCILMENTE MODIFICABLE)
# Formato: "N:g"
# ------------------------------------------------------------------------------
SYSTEMS_LIST=(
    "4:0.5"
    "4:1.0"
    "10:0.5"
    "10:5.0"
)

OMEGA="${OMEGA:-0.1}"
BASE_OUT_DIR="results_targeted_pipeline2"
LOG_DIR="${BASE_OUT_DIR}/logs"
CSV_COLLECT_DIR="${BASE_OUT_DIR}/csv_summary"
FIG_OUT_DIR="${BASE_OUT_DIR}/figures"
TRACKER_DIR="${BASE_OUT_DIR}/.tracker"

mkdir -p "${BASE_OUT_DIR}" "${LOG_DIR}" "${CSV_COLLECT_DIR}" "${FIG_OUT_DIR}" "${TRACKER_DIR}"
rm -f "${TRACKER_DIR}"/job_*.meta 2>/dev/null || true

MASTER_LOG="${LOG_DIR}/master_targeted_pipeline2_search.log"

log_master() {
    local msg="$1"
    echo "$msg"
    echo "$msg" >> "${MASTER_LOG}"
}

# ------------------------------------------------------------------------------
# 2. Entorno y Rendimiento Numérico
# ------------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.20}"
export TF_FORCE_GPU_ALLOW_GROWTH=true

PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# 3. Configuración de GPUs y Colas
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
# 4. Hiperparámetros de la Rejilla de Muestreo
# ------------------------------------------------------------------------------
N_BURN_IN="${N_BURN_IN:-2500}"
MEASURE_STEPS="${MEASURE_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

# Grillas de Parámetros
# A. Amplitudes del Actor para TD3 (RL-MALA)
ACTOR_AMP_LIST=(0.02 0.05 0.10 0.15 0.20 0.30 0.40)

# B. Tamaños de paso para MALA estándar (adaptados según N)
get_mala_tau_list() {
    local N=$1
    if [ "$N" -ge 10 ]; then
        echo "0.002 0.005 0.01 0.02 0.03 0.05"
    else
        echo "0.02 0.04 0.06 0.08 0.10 0.12 0.15"
    fi
}

get_nominal_rl_tau() {
    local N=$1
    if [ "$N" -ge 10 ]; then
        echo "0.01"
    else
        echo "0.08"
    fi
}

# Archivo de resumen maestro
TARGETED_SUMMARY_CSV="${BASE_OUT_DIR}/targeted_pipeline2_grid_summary.csv"
echo "sys_tag,N,g,method,param_name,param_val,h5_path,E_mean,SE_E,acceptance,ESJD,tau_int_E,G_RL_speedup,ESS_per_sec,casscf_energy,delta_E_vs_CASSCF,kl_casscf_total_mean,kl_casscf_up_mean,kl_casscf_down_mean" > "${TARGETED_SUMMARY_CSV}"

log_master "=============================================================================="
log_master "    INICIO DE GRID SEARCH FOCALIZADO DE PIPELINE 2 (MUESTREO PURO)"
log_master "    Fecha y Hora        : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "    Sistemas a Evaluar  : ${SYSTEMS_LIST[*]}"
log_master "    Frecuencia Omega    : ${OMEGA}"
log_master "    Barrido TD3 (Actor) : act_amp in [${ACTOR_AMP_LIST[*]}]"
log_master "    Pool de GPUs        : ${AVAILABLE_GPUS[*]}"
log_master "    Trabajos por GPU    : ${JOBS_PER_GPU}"
log_master "    Concurrencia Máxima : ${MAX_TOTAL_JOBS} trabajos simultáneos"
log_master "    Logs en Tiempo Real : ${LOG_DIR}/"
log_master "=============================================================================="

# ------------------------------------------------------------------------------
# 5. Búsqueda de CASSCF de Referencia por Sistema
# ------------------------------------------------------------------------------
find_casscf_for_system() {
    local N=$1
    local G=$2
    local sys_tag="N${N}_w${OMEGA}_g${G}"

    # Candidatos directos
    local candidates=(
        "results_production_3pipelines_b2/pipeline1_casscf/${sys_tag}/casscf_state.npz"
        "results_production_3pipelines/pipeline1_casscf/${sys_tag}/casscf_state.npz"
        "figures/cas_n${N}_g${G}_omega${OMEGA}_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
        "figures/cas_n${N}_g${G}_omega${OMEGA}_soft0.1_b32_casscf_state.npz"
    )

    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            echo "$c"
            return 0
        fi
    done

    # Búsqueda en paquete uhf1d
    local uhf_match
    uhf_match=$(find uhf_1d_ho_v270826_m-determinants -path "*N${N}*/*g_${G}*/*casscf_state.npz" 2>/dev/null | head -n 1)
    if [ -n "$uhf_match" ] && [ -f "$uhf_match" ]; then
        echo "$uhf_match"
        return 0
    fi

    uhf_match=$(find uhf_1d_ho_v270826_m-determinants -path "*N${N}*/*g${G}*/*casscf_state.npz" 2>/dev/null | head -n 1)
    if [ -n "$uhf_match" ] && [ -f "$uhf_match" ]; then
        echo "$uhf_match"
        return 0
    fi

    echo ""
}

# Detección de CSV de densidad analítica previa
find_ref_density_csv() {
    local N=$1
    local G=$2
    local sys_tag="N${N}_w${OMEGA}_g${G}"
    local candidates=(
        "results_csv_b2/casscf_exact_density_${sys_tag}.csv"
        "results_csv/casscf_exact_density_${sys_tag}.csv"
        "results_casscf_wigner_densities/${sys_tag}*/one_body_density.csv"
    )
    for c in "${candidates[@]}"; do
        for f in $c; do
            if [ -f "$f" ]; then
                echo "$f"
                return 0
            fi
        done
    done
    echo ""
}

# ------------------------------------------------------------------------------
# 6. Funciones del Gestor de Colas
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
            local gpu sys start_epoch start_time script_name sys_log_file sum_csv method_tag param_name param_val sys_tag_val n_val g_val
            # shellcheck disable=SC1090
            source "$f"
            local end_epoch
            end_epoch=$(date +%s)
            local end_time
            end_time=$(date "+%Y-%m-%d %H:%M:%S")
            local duration=$((end_epoch - start_epoch))
            local duration_formatted
            duration_formatted=$(format_duration "$duration")

            # Inserción atómica al CSV maestro
            if [ -n "${sum_csv:-}" ] && [ -f "$sum_csv" ]; then
                (
                    flock -x 200 2>/dev/null || true
                    tail -n +2 "$sum_csv" | awk -v s="${sys_tag_val}" -v n="${n_val}" -v g="${g_val}" -v m="${method_tag}" -v pn="${param_name}" -v pv="${param_val}" -F',' '{OFS=","; print s, n, g, m, pn, pv, $0}' >> "${TARGETED_SUMMARY_CSV}"
                ) 200>"${TARGETED_SUMMARY_CSV}.lock"
            fi

            log_master "------------------------------------------------------------------------------"
            log_master "<< [$stage_name | FINALIZADO] ${sys}"
            log_master "   PID del Worker     : $pid (Código: $exit_code)"
            log_master "   GPU Liberada       : GPU $gpu"
            log_master "   Hora de Inicio     : $start_time"
            log_master "   Hora de Fin        : $end_time"
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
# 7. Worker de Muestreo (MALA o TD3) con Flush en Tiempo Real
# ------------------------------------------------------------------------------
run_targeted_sampling_worker() {
    local GPU_ID="$1"
    local N="$2"
    local G="$3"
    local METHOD="$4"
    local TAU="$5"
    local ACT_AMP="$6"
    local CASSCF_NPZ="$7"
    local REF_DENSITY="$8"
    local RUN_TAG="$9"

    local N_UP=$(( N / 2 ))
    local N_DN=$(( N - N_UP ))
    local SOFT="0.1"
    if [ "$N" -ge 10 ] && (( $(echo "$G >= 3.0" | bc -l 2>/dev/null || echo 0) )); then
        SOFT="0.25"
    fi

    local MODEL_DIR="${BASE_OUT_DIR}/models/${RUN_TAG}"
    local POST_DIR="${MODEL_DIR}/postprocess"
    mkdir -p "${MODEL_DIR}" "${POST_DIR}"
    local H5_FILE="${MODEL_DIR}/run_bundle_${RUN_TAG}.h5"

    local EXTRA_ARGS=""
    local LABEL="MALA (tau=${TAU})"
    if [ "$METHOD" = "TD3" ]; then
        EXTRA_ARGS="--actor_amplitude ${ACT_AMP}"
        LABEL="RL-MALA TD3 (A=${ACT_AMP}, tau=${TAU})"
    fi

    local REF_CSV_ARG=""
    if [ -n "$REF_DENSITY" ] && [ -f "$REF_DENSITY" ]; then
        REF_CSV_ARG="--reference_density_csv ${REF_DENSITY}"
    fi

    # Ejecución no bufferizada
    env CUDA_VISIBLE_DEVICES="${GPU_ID}" ${PYTHON_BIN} -u cli/run_1d.py \
        --method "${METHOD}" \
        --train_mode eval_only \
        --ansatz_type casscf_npz \
        --casscf_npz_path "${CASSCF_NPZ}" \
        --n_elec "${N}" --n_up "${N_UP}" --n_dn "${N_DN}" \
        --omega "${OMEGA}" --interaction_strength "${G}" --softening "${SOFT}" \
        --tau "${TAU}" \
        ${EXTRA_ARGS} \
        --n_ter "${N_BURN_IN}" \
        --measure_steps "${MEASURE_STEPS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_h5 "${H5_FILE}"

    # Post-procesamiento
    ${PYTHON_BIN} -u cli/postprocess.py \
        --inputs "${H5_FILE}" \
        --labels "${LABEL}" \
        --out_dir "${POST_DIR}" \
        --casscf_npz_path "${CASSCF_NPZ}" \
        ${REF_CSV_ARG} \
        --system_tag "${RUN_TAG}" \
        --csv_collect_dir "${CSV_COLLECT_DIR}" \
        --plot_observables_comparison \
        --plot_kl_divergence \
        --figure_format png
}

# ------------------------------------------------------------------------------
# 8. Despacho de Corridas por Sistema
# ------------------------------------------------------------------------------
for entry in "${SYSTEMS_LIST[@]}"; do
    IFS=":" read -r N G <<< "$entry"
    SYS_TAG="N${N}_w${OMEGA}_g${G}"

    log_master ""
    log_master "=============================================================================="
    log_master ">>> PREPARANDO SISTEMA: ${SYS_TAG} (N=${N}, g=${G})"
    log_master "=============================================================================="

    CASSCF_NPZ=$(find_casscf_for_system "$N" "$G")
    if [ -z "$CASSCF_NPZ" ] || [ ! -f "$CASSCF_NPZ" ]; then
        log_master "[-] AVISO: No se encontró estado CASSCF para ${SYS_TAG}. Omitiendo sistema."
        continue
    fi
    log_master "[+] Estado CASSCF Encontrado : ${CASSCF_NPZ}"

    REF_CSV=$(find_ref_density_csv "$N" "$G")
    if [ -n "$REF_CSV" ]; then
        log_master "[+] Densidad CASSCF Exacta  : ${REF_CSV}"
    fi

    # 8A. Despacho de MALA (Barrido de tau_mala)
    TAUS_TO_RUN=$(get_mala_tau_list "$N")
    log_master "[*] Despachando barrido de MALA: tau in [${TAUS_TO_RUN}]"

    for tau in $TAUS_TO_RUN; do
        RUN_TAG="${SYS_TAG}_MALA_tau${tau}"
        SYS_LOG="sampling_${RUN_TAG}.log"
        GPU_ID=$(wait_for_free_gpu_slot "GRID SEARCH P2")

        start_epoch=$(date +%s)
        start_time=$(date "+%Y-%m-%d %H:%M:%S")

        run_targeted_sampling_worker "$GPU_ID" "$N" "$G" "MALA" "$tau" "0.0" "$CASSCF_NPZ" "$REF_CSV" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
        worker_pid=$!

        cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
sys_tag_val="$SYS_TAG"
n_val="$N"
g_val="$G"
method_tag="MALA"
param_name="tau_mala"
param_val="$tau"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_targeted_sampling_worker MALA tau=$tau"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/models/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF
        log_master ">> [TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
    done

    # 8B. Despacho de RL-MALA TD3 (Barrido de actor_amplitude)
    RL_TAU=$(get_nominal_rl_tau "$N")
    log_master "[*] Despachando barrido de RL-MALA TD3: act_amp in [${ACTOR_AMP_LIST[*]}] (tau=${RL_TAU})"

    for amp in "${ACTOR_AMP_LIST[@]}"; do
        RUN_TAG="${SYS_TAG}_RL_TD3_amp${amp}_tau${RL_TAU}"
        SYS_LOG="sampling_${RUN_TAG}.log"
        GPU_ID=$(wait_for_free_gpu_slot "GRID SEARCH P2")

        start_epoch=$(date +%s)
        start_time=$(date "+%Y-%m-%d %H:%M:%S")

        run_targeted_sampling_worker "$GPU_ID" "$N" "$G" "TD3" "$RL_TAU" "$amp" "$CASSCF_NPZ" "$REF_CSV" "$RUN_TAG" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
        worker_pid=$!

        cat << EOF > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
gpu="$GPU_ID"
sys="$RUN_TAG"
sys_tag_val="$SYS_TAG"
n_val="$N"
g_val="$G"
method_tag="TD3"
param_name="actor_amplitude"
param_val="$amp"
start_epoch="$start_epoch"
start_time="$start_time"
script_name="run_targeted_sampling_worker TD3 amp=$amp"
sys_log_file="$SYS_LOG"
sum_csv="${BASE_OUT_DIR}/models/${RUN_TAG}/postprocess/summary_metrics.csv"
EOF
        log_master ">> [TRABAJO ASIGNADO] ${RUN_TAG} -> GPU ${GPU_ID} | PID: ${worker_pid} | Log: ${LOG_DIR}/${SYS_LOG}"
    done
done

# Esperar a que terminen todos los trabajos
log_master ""
log_master "[*] Esperando a que concluyan todos los trabajos encolados de Pipeline 2..."
wait_all_stage_jobs "GRID SEARCH P2"
log_master "[+] TODOS LOS MUESTREOS HAN FINALIZADO EXITOSAMENTE."

# ------------------------------------------------------------------------------
# 9. Resumen Ejecutivo en Términos de Precisión y Tiempos de Corrida
# ------------------------------------------------------------------------------
log_master ""
log_master "=============================================================================="
log_master "                    RESUMEN EJECUTIVO DE RESULTADOS Y PRECISIÓN"
log_master "=============================================================================="

${PYTHON_BIN} - << 'PY_REPORT'
import os
import pandas as pd
import numpy as np

csv_path = "results_targeted_pipeline2/targeted_pipeline2_grid_summary.csv"
if not os.path.exists(csv_path):
    print("No se encontró el archivo de resumen:", csv_path)
    exit(0)

df = pd.read_csv(csv_path)
if df.empty:
    print("La tabla de resumen está vacía.")
    exit(0)

# Limpieza y conversión numérica
num_cols = ["E_mean", "SE_E", "acceptance", "tau_int_E", "delta_E_vs_CASSCF", "kl_casscf_total_mean"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("\n" + "=" * 90)
print(f"{'SISTEMA':<16} | {'METODO':<6} | {'PARAMETRO':<10} | {'VALOR':<6} | {'E_mean (Ha)':<11} | {'SE(E)':<9} | {'tau_int':<7} | {'D_KL':<8} | {'Acc %':<5}")
print("=" * 90)

for (sys_tag), g_df in df.groupby("sys_tag"):
    # Mejor MALA
    mala_df = g_df[g_df["method"] == "MALA"]
    if not mala_df.empty:
        best_mala = mala_df.sort_values("kl_casscf_total_mean").iloc[0]
        acc_pct = f"{best_mala.get('acceptance', 0)*100:.1f}%"
        print(f"{sys_tag:<16} | {'MALA':<6} | {'tau_mala':<10} | {str(best_mala.get('param_val', '')):<6} | {best_mala.get('E_mean', np.nan):<11.5f} | {best_mala.get('SE_E', np.nan):<9.2e} | {best_mala.get('tau_int_E', np.nan):<7.2f} | {best_mala.get('kl_casscf_total_mean', np.nan):<8.4f} | {acc_pct:<5}")

    # Mejor RL-MALA (TD3)
    rl_df = g_df[g_df["method"] == "TD3"]
    if not rl_df.empty:
        best_rl = rl_df.sort_values("kl_casscf_total_mean").iloc[0]
        acc_pct = f"{best_rl.get('acceptance', 0)*100:.1f}%"
        print(f"{sys_tag:<16} | {'TD3':<6} | {'act_amp':<10} | {str(best_rl.get('param_val', '')):<6} | {best_rl.get('E_mean', np.nan):<11.5f} | {best_rl.get('SE_E', np.nan):<9.2e} | {best_rl.get('tau_int_E', np.nan):<7.2f} | {best_rl.get('kl_casscf_total_mean', np.nan):<8.4f} | {acc_pct:<5}")
    print("-" * 90)

print("=" * 90 + "\n")
PY_REPORT

log_master "=============================================================================="
log_master "    GRID SEARCH FOCALIZADO COMPLETADO CON ÉXITO"
log_master "    CSV Maestro  : ${TARGETED_SUMMARY_CSV}"
log_master "    Logs Totales : ${LOG_DIR}/"
log_master "    Fecha y Hora : $(date '+%Y-%m-%d %H:%M:%S %Z')"
log_master "=============================================================================="
