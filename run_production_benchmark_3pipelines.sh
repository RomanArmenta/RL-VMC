#!/usr/bin/env bash
# ==============================================================================
# MASTER PRODUCTION BENCHMARK: 3 PIPELINES (CASSCF -> PURE SAMPLING -> COUPLED VMC)
# ==============================================================================
# Implementa las tres etapas de producción solicitadas sobre el paquete modular `rl_vmc`
# y la paquetería `uhf1d` (CASSCF multi-determinante 1D) del asesor:
#
#   1. PIPELINE 1 (Producción de Referencias CASSCF en CPU, estrictamente secuencial):
#      Calcula (o reutiliza) la función de onda multiconfiguracional CASSCF para
#      cada sistema (N, omega, g) uno por uno en CPU (CUDA_VISIBLE_DEVICES="").
#      No inicia el siguiente sistema hasta que el anterior haya concluido por completo.
#      Registra script lanzado, hora de inicio, hora de fin, tiempo transcurrido y PID.
#
#   2. PIPELINE 2 (Muestreo Puro sobre CASSCF Congelado - MALA vs RL-MALA TD3):
#      Fija exactamente los pesos de la función de onda multi-determinante
#      `casscf_state.npz` del Pipeline 1 y compara:
#        - MALA estándar (`--proposal_type mala --train_mode eval_only`)
#        - RL-MALA con Actor DeepSet + TD3 (`--proposal_type neural_deepset --train_mode rl_only`)
#      Los sistemas se reparten dinámicamente entre las GPUs disponibles conforme
#      se van desocupando. Se monitorea cada ejecución con PID, tiempos y comandos.
#
#   3. PIPELINE 3 (Entrenamiento Variacional de Función de Onda Neuronal):
#      Entrena desde cero el Ansatz variacional Slater-Jastrow con Backflow Neuronal
#      y Jastrow RBF resuelto por espín comparando:
#        - VMC + MALA (`--proposal_type mala --train_mode vmc_only`)
#        - VMC + RL-MALA Acoplado (`--proposal_type neural_deepset --train_mode coupled`)
#      Se reparten los trabajos dinámicamente en GPU conforme se van liberando slots.
#
# Uso:
#   chmod +x run_production_benchmark_3pipelines.sh
#   nohup ./run_production_benchmark_3pipelines.sh > master_3pipelines.log 2>&1 &
#
# Variables de entorno opcionales para personalizar la corrida:
#   BASE_OUT_DIR="results_production_3pipelines"
#   PIPELINES_TO_RUN="1 2 3"
#   N_LIST="2 4 6 10"
#   G_LIST="0.5 1.0 2.0 3.0 5.0 8.0"
#   OMEGA="0.1"
#   GPU_LIST="auto"           # "auto" o lista de GPUs p.ej. "0 1"
#   JOBS_PER_GPU="1"          # Trabajos simultáneos por GPU (default: 1)
#   MAX_PARALLEL_JOBS="auto"  # Límite global ("auto" = GPUs * JOBS_PER_GPU)
#   COPY_ADVISOR_CASSCF="1"   # 1 = reutilizar casscf_state.npz del asesor si coincide exactamente
#   FORCE_RERUN="0"           # 1 = recalcular aunque ya exista run_bundle.h5
# ==============================================================================

set -u

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export TF_FORCE_GPU_ALLOW_GROWTH=true

BASE_OUT_DIR="${BASE_OUT_DIR:-results_production_3pipelines_b2}"
CSV_COLLECT_DIR="${CSV_COLLECT_DIR:-results_csv_b2}"
PIPELINES_TO_RUN="${PIPELINES_TO_RUN:-1 2 3}"
N_LIST="${N_LIST:-2 4 6 10}"
G_LIST="${G_LIST:-0.5 1.0 2.0 3.0 5.0 8.0}"
OMEGA="${OMEGA:-0.1}"
GPU_LIST="${GPU_LIST:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"
COPY_ADVISOR_CASSCF="${COPY_ADVISOR_CASSCF:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Hiperparámetros de CASSCF (Benchmark 2: corridas largas para convergencia completa)
CASSCF_MAX_ITER="${CASSCF_MAX_ITER:-1000}"
CASSCF_ENERGY_TOL="${CASSCF_ENERGY_TOL:-1e-8}"
CASSCF_GRAD_TOL="${CASSCF_GRAD_TOL:-1e-4}"

# Hiperparámetros de muestreo y entrenamiento
N_WALKERS="${N_WALKERS:-256}"
RL_ONLY_STEPS="${RL_ONLY_STEPS:-10000}"
VMC_COUPLED_STEPS="${VMC_COUPLED_STEPS:-15000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-5}"
EVAL_BURN_IN="${EVAL_BURN_IN:-2500}"
MEASURE_STEPS="${MEASURE_STEPS:-8000}"

# Estructura de carpetas de salida y tablas CSV en vivo
P1_DIR="${BASE_OUT_DIR}/pipeline1_casscf"
P2_DIR="${BASE_OUT_DIR}/pipeline2_pure_sampling"
P3_DIR="${BASE_OUT_DIR}/pipeline3_variational_training"
LOG_DIR="${BASE_OUT_DIR}/logs"

LIVE_CSV_P1="${BASE_OUT_DIR}/live_pipeline1_casscf_summary.csv"
LIVE_CSV_P2_RUNS="${BASE_OUT_DIR}/live_pipeline2_sampling_runs.csv"
LIVE_CSV_P2_COMP="${BASE_OUT_DIR}/live_pipeline2_sampling_comparison.csv"
LIVE_CSV_P3_RUNS="${BASE_OUT_DIR}/live_pipeline3_training_runs.csv"
LIVE_CSV_P3_COMP="${BASE_OUT_DIR}/live_pipeline3_training_comparison.csv"
LIVE_CSV_ALL="${BASE_OUT_DIR}/live_all_models_master.csv"

mkdir -p "$P1_DIR" "$P2_DIR" "$P3_DIR" "$LOG_DIR" "$CSV_COLLECT_DIR"

MASTER_START_EPOCH=$(date +%s)
MASTER_START_TIME=$(date "+%Y-%m-%d %H:%M:%S")
MASTER_PID=$$

# ------------------------------------------------------------------------------
# Directorio temporal atómico para tracking de trabajos y asignación de GPUs
# ------------------------------------------------------------------------------
TRACKER_DIR=$(mktemp -d /tmp/rl_vmc_gpu_tracker_XXXXXX 2>/dev/null || mktemp -d -t 'rl_vmc_gpu_tracker')

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
# Detección y configuración del Pool de GPUs
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
echo " INICIANDO BENCHMARK DE PRODUCCIÓN EN 3 PIPELINES"
echo " Fecha y Hora        : $MASTER_START_TIME"
echo " PID del Master      : $MASTER_PID"
echo " Directorio raíz     : $BASE_OUT_DIR"
echo " Pipelines activos   : $PIPELINES_TO_RUN"
echo " Sistemas N          : $N_LIST"
echo " Acoplamientos g     : $G_LIST"
echo " Frecuencia omega    : $OMEGA"
echo " GPUs en Pool        : ${AVAILABLE_GPUS[*]}"
echo " Trabajos por GPU    : $JOBS_PER_GPU"
echo " Concurrencia total  : $MAX_TOTAL_JOBS trabajos simultáneos"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Helpers de formato de tiempo y logging detallado
# ------------------------------------------------------------------------------
format_duration() {
    local dur=$1
    if [ "$dur" -lt 0 ]; then dur=0; fi
    printf "%02dh %02dm %02ds" $((dur/3600)) $(( (dur%3600)/60 )) $((dur%60))
}

# Ejecuta un comando síncrono (usado en Pipeline 1), registrando
# el script lanzado, hora de inicio, fin, duración y PID.
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

# Ejecuta un sub-paso interno dentro de los workers de Pipeline 2 y 3,
# registrando explícitamente en el log del sistema el script, PID y tiempos.
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
# Parámetros seguros por sistema físico (N, g) basados en los summary.json del asesor
# ------------------------------------------------------------------------------
get_system_casscf_params() {
    local N=$1
    local G=$2
    local n_core=0
    local n_act=4
    local n_basis=32
    local quad=80
    local soft="0.1"

    if [ "$N" -eq 2 ]; then
        n_core=0; n_act=4; n_basis=24; quad=80; soft="0.1"
    elif [ "$N" -eq 4 ]; then
        n_core=0; n_act=4; n_basis=32; quad=80; soft="0.1"
    elif [ "$N" -eq 6 ]; then
        n_core=1; n_act=4; n_basis=32; quad=80; soft="0.1"
    elif [ "$N" -eq 8 ]; then
        n_core=2; n_act=4; n_basis=28; quad=80; soft="0.1"
    elif [ "$N" -eq 10 ]; then
        # n_core=4, n_active_orbitals=4, n_basis=32 (expandido para convergencia a g>=3.0), softening=0.25
        n_core=4; n_act=4; n_basis=32; quad=80; soft="0.25"
    else
        n_core=$(( (N / 2) - 2 ))
        if [ "$n_core" -lt 0 ]; then n_core=0; fi
        n_act=4; n_basis=32; quad=80; soft="0.1"
    fi
    echo "$n_core $n_act $n_basis $quad $soft"
}

get_system_tau() {
    local N=$1
    if [ "$N" -ge 10 ]; then
        echo "0.005"
    else
        echo "0.01"
    fi
}

# ==============================================================================
# PIPELINE 1: PRODUCCIÓN DE REFERENCIAS EXACTAS CASSCF (CPU - UNO POR UNO)
# ==============================================================================
run_pipeline_1() {
    echo ""
    echo "=============================================================================="
    echo " [PIPELINE 1] Generando referencias CASSCF 1D con paquetería uhf1d..."
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
            local SYS_LOG="${LOG_DIR}/p1_casscf_${SYS_TAG}.log"

            local EXTRA_FLAGS="--reuse_existing"
            if [ "$FORCE_RERUN" -eq 1 ]; then
                EXTRA_FLAGS=""
            fi
            if [ "$COPY_ADVISOR_CASSCF" -eq 1 ]; then
                EXTRA_FLAGS="$EXTRA_FLAGS --copy_from_advisor_if_available"
            fi

            # Ejecución estrictamente uno por uno en CPU (CUDA_VISIBLE_DEVICES="")
            run_logged_command \
                "[PIPELINE 1 | CASSCF | ${SYS_TAG}]" \
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
        done
    done
    echo ""
    echo " [PIPELINE 1] Todos los cálculos CASSCF en CPU han terminado. Resumen en: $LIVE_CSV_P1"
}

# ==============================================================================
# PIPELINE 2: MUESTREO PURO SOBRE CASSCF CONGELADO (MALA vs RL-MALA)
# ==============================================================================
run_sistema_pipeline_2() {
    local GPU_ID=$1
    local N=$2
    local G=$3
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
    local TAU
    TAU=$(get_system_tau "$N")

    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local CASSCF_NPZ="${P1_DIR}/${SYS_TAG}/casscf_state.npz"
    local SYS_OUT="${P2_DIR}/${SYS_TAG}"
    local MALA_DIR="${SYS_OUT}/MALA_FixedCASSCF"
    local RL_DIR="${SYS_OUT}/RL_TD3_FixedCASSCF"
    local COMP_DIR="${SYS_OUT}/postprocess_comparison"
    local MALA_H5="${MALA_DIR}/run_bundle_P2_MALA_FixedCASSCF_${SYS_TAG}.h5"
    local RL_H5="${RL_DIR}/run_bundle_P2_RL_TD3_FixedCASSCF_${SYS_TAG}.h5"

    mkdir -p "$MALA_DIR" "$RL_DIR" "$COMP_DIR"

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

    # 2A. Muestreo puro con MALA estándar sobre CASSCF congelado
    if [ "$FORCE_RERUN" -eq 1 ] || { [ ! -f "$MALA_H5" ] && [ ! -f "${MALA_DIR}/run_bundle.h5" ]; }; then
        run_logged_step "[Pipeline 2 | MALA | ${SYS_TAG} | GPU ${GPU_ID}]" \
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
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --output_h5 "$MALA_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P2_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --export_legacy_files
        if [ -f "$MALA_H5" ]; then
            ln -sf "$(basename "$MALA_H5")" "${MALA_DIR}/run_bundle.h5" 2>/dev/null || cp -f "$MALA_H5" "${MALA_DIR}/run_bundle.h5"
        fi
    elif [ -f "${MALA_DIR}/run_bundle.h5" ] && [ ! -f "$MALA_H5" ]; then
        MALA_H5="${MALA_DIR}/run_bundle.h5"
    fi

    # 2B. Entrenamiento del Actor RL-MALA (TD3 DeepSet) sobre CASSCF congelado + Evaluación
    if [ "$FORCE_RERUN" -eq 1 ] || { [ ! -f "$RL_H5" ] && [ ! -f "${RL_DIR}/run_bundle.h5" ]; }; then
        run_logged_step "[Pipeline 2 | RL-TD3 | ${SYS_TAG} | GPU ${GPU_ID}]" \
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
                --rl_algorithm td3 \
                --reward_type log_esjd \
                --actor_amplitude 0.02 \
                --train_steps "$RL_ONLY_STEPS" \
                --tau "$TAU" \
                --batch_size "$N_WALKERS" \
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --output_h5 "$RL_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P2_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --export_legacy_files
        if [ -f "$RL_H5" ]; then
            ln -sf "$(basename "$RL_H5")" "${RL_DIR}/run_bundle.h5" 2>/dev/null || cp -f "$RL_H5" "${RL_DIR}/run_bundle.h5"
        fi
    elif [ -f "${RL_DIR}/run_bundle.h5" ] && [ ! -f "$RL_H5" ]; then
        RL_H5="${RL_DIR}/run_bundle.h5"
    fi

    # 2C. Post-procesamiento inmediato por sistema (figuras + CSV comparativo en vivo)
    if [ -f "$MALA_H5" ] && [ -f "$RL_H5" ]; then
        run_logged_step "[Pipeline 2 | Postprocess | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" python3 -u cli/postprocess.py \
                --inputs "$MALA_H5" "$RL_H5" \
                --labels "MALA (Fixed CASSCF)" "RL-MALA TD3 (Fixed CASSCF)" \
                --casscf_npz_path "$CASSCF_NPZ" \
                --out_dir "$COMP_DIR" \
                --system_tag "P2_${SYS_TAG}" \
                --master_comparison_csv "$LIVE_CSV_P2_COMP" \
                --csv_collect_dir "$CSV_COLLECT_DIR"

        run_logged_step "[Pipeline 2 | Postprocess Master | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" python3 -u cli/postprocess.py \
                --inputs "$MALA_H5" "$RL_H5" \
                --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
                --casscf_npz_path "$CASSCF_NPZ" \
                --out_dir "$COMP_DIR" \
                --system_tag "P2_${SYS_TAG}" \
                --master_comparison_csv "$LIVE_CSV_ALL" \
                --csv_collect_dir "$CSV_COLLECT_DIR"
        echo "  <- [Pipeline 2 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Gráficas y summary.json en: ${COMP_DIR}"
    else
        echo "  [Aviso Pipeline 2] No se encontraron ambos run_bundle.h5 para ${SYS_TAG}. Post-procesamiento omitido."
    fi
}

run_pipeline_2() {
    echo ""
    echo "=============================================================================="
    echo " [PIPELINE 2] Muestreo Puro sobre CASSCF Congelado (MALA vs RL-MALA TD3)"
    echo " Despacho dinámico de GPUs: ${AVAILABLE_GPUS[*]} (Límite: $MAX_TOTAL_JOBS trabajos paralelos)"
    echo " Tablas CSV en vivo: $LIVE_CSV_P2_RUNS | $LIVE_CSV_P2_COMP"
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

            # Lanzamiento en background con GPU asignada
            run_sistema_pipeline_2 "$GPU_ID" "$N" "$G" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            local worker_pid=$!

            # Registro de metadata para monitoreo dinámico
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
    echo ""
    echo " [PIPELINE 2] Todos los sistemas han terminado exitosamente."
}

# ==============================================================================
# PIPELINE 3: ENTRENAMIENTO VARIACIONAL SLATER-JASTROW-BACKFLOW (MALA vs COUPLED RL-MALA)
# ==============================================================================
run_sistema_pipeline_3() {
    local GPU_ID=$1
    local N=$2
    local G=$3
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
    local TAU
    TAU=$(get_system_tau "$N")

    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local CASSCF_NPZ="${P1_DIR}/${SYS_TAG}/casscf_state.npz"
    local SYS_OUT="${P3_DIR}/${SYS_TAG}"
    local VMC_MALA_DIR="${SYS_OUT}/VMC_MALA_SJB"
    local COUPLED_RL_DIR="${SYS_OUT}/Coupled_RL_TD3_SJB"
    local COMP_DIR="${SYS_OUT}/postprocess_comparison"
    local VMC_MALA_H5="${VMC_MALA_DIR}/run_bundle_P3_VMC_MALA_SJB_${SYS_TAG}.h5"
    local COUPLED_RL_H5="${COUPLED_RL_DIR}/run_bundle_P3_Coupled_RL_TD3_SJB_${SYS_TAG}.h5"

    mkdir -p "$VMC_MALA_DIR" "$COUPLED_RL_DIR" "$COMP_DIR"

    # 3A. Entrenamiento VMC con MALA estándar (Slater-Jastrow + Backflow + Spin-RBF Jastrow)
    if [ "$FORCE_RERUN" -eq 1 ] || { [ ! -f "$VMC_MALA_H5" ] && [ ! -f "${VMC_MALA_DIR}/run_bundle.h5" ]; }; then
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
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --output_h5 "$VMC_MALA_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P3_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --export_legacy_files
        if [ -f "$VMC_MALA_H5" ]; then
            ln -sf "$(basename "$VMC_MALA_H5")" "${VMC_MALA_DIR}/run_bundle.h5" 2>/dev/null || cp -f "$VMC_MALA_H5" "${VMC_MALA_DIR}/run_bundle.h5"
        fi
    elif [ -f "${VMC_MALA_DIR}/run_bundle.h5" ] && [ ! -f "$VMC_MALA_H5" ]; then
        VMC_MALA_H5="${VMC_MALA_DIR}/run_bundle.h5"
    fi

    # 3B. Entrenamiento Acoplado VMC + RL-MALA (Coupled Trainer con TD3 DeepSet)
    if [ "$FORCE_RERUN" -eq 1 ] || { [ ! -f "$COUPLED_RL_H5" ] && [ ! -f "${COUPLED_RL_DIR}/run_bundle.h5" ]; }; then
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
                --rl_algorithm td3 \
                --reward_type log_esjd \
                --actor_amplitude 0.02 \
                --train_steps "$VMC_COUPLED_STEPS" \
                --tau "$TAU" \
                --batch_size "$N_WALKERS" \
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --output_h5 "$COUPLED_RL_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P3_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --export_legacy_files
        if [ -f "$COUPLED_RL_H5" ]; then
            ln -sf "$(basename "$COUPLED_RL_H5")" "${COUPLED_RL_DIR}/run_bundle.h5" 2>/dev/null || cp -f "$COUPLED_RL_H5" "${COUPLED_RL_DIR}/run_bundle.h5"
        fi
    elif [ -f "${COUPLED_RL_DIR}/run_bundle.h5" ] && [ ! -f "$COUPLED_RL_H5" ]; then
        COUPLED_RL_H5="${COUPLED_RL_DIR}/run_bundle.h5"
    fi

    # 3C. Post-procesamiento inmediato por sistema
    if [ -f "$VMC_MALA_H5" ] && [ -f "$COUPLED_RL_H5" ]; then
        local EXTRA_CASSCF_FLAG=""
        if [ -f "$CASSCF_NPZ" ]; then
            EXTRA_CASSCF_FLAG="--casscf_npz_path $CASSCF_NPZ"
        fi

        run_logged_step "[Pipeline 3 | Postprocess | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" python3 -u cli/postprocess.py \
                --inputs "$VMC_MALA_H5" "$COUPLED_RL_H5" \
                --labels "VMC + MALA (SJB)" "Coupled VMC + RL-MALA TD3 (SJB)" \
                $EXTRA_CASSCF_FLAG \
                --out_dir "$COMP_DIR" \
                --system_tag "P3_${SYS_TAG}" \
                --master_comparison_csv "$LIVE_CSV_P3_COMP" \
                --csv_collect_dir "$CSV_COLLECT_DIR"

        run_logged_step "[Pipeline 3 | Postprocess Master | ${SYS_TAG}]" \
            env CUDA_VISIBLE_DEVICES="" python3 -u cli/postprocess.py \
                --inputs "$VMC_MALA_H5" "$COUPLED_RL_H5" \
                --labels "P3_VMC_MALA_SJB_${SYS_TAG}" "P3_Coupled_RL_TD3_SJB_${SYS_TAG}" \
                $EXTRA_CASSCF_FLAG \
                --out_dir "$COMP_DIR" \
                --system_tag "P3_${SYS_TAG}" \
                --master_comparison_csv "$LIVE_CSV_ALL" \
                --csv_collect_dir "$CSV_COLLECT_DIR"
        echo "  <- [Pipeline 3 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Gráficas y summary.json en: ${COMP_DIR}"
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

            # Lanzamiento en background con GPU asignada
            run_sistema_pipeline_3 "$GPU_ID" "$N" "$G" > "${LOG_DIR}/${SYS_LOG}" 2>&1 &
            local worker_pid=$!

            # Registro de metadata para monitoreo dinámico
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
echo " BENCHMARK DE 3 PIPELINES FINALIZADO EXITOSAMENTE"
echo " PID del Master      : $MASTER_PID"
echo " Hora de Inicio      : $MASTER_START_TIME"
echo " Hora de Terminación : $MASTER_END_TIME"
echo " Tiempo Total        : ${MASTER_DURATION}s ($MASTER_DURATION_FORMATTED)"
echo " Tablas CSV maestras generadas en vivo:"
echo "   - Pipeline 1 (CASSCF)       : $LIVE_CSV_P1"
echo "   - Pipeline 2 (Muestreo)     : $LIVE_CSV_P2_COMP"
echo "   - Pipeline 3 (Entrenamiento): $LIVE_CSV_P3_COMP"
echo "   - Tabla global consolidada  : $LIVE_CSV_ALL"
echo "=============================================================================="
