#!/usr/bin/env bash
# ==============================================================================
# MASTER PRODUCTION BENCHMARK: 3 PIPELINES (CASSCF -> PURE SAMPLING -> COUPLED VMC)
# ==============================================================================
# Implementa las tres etapas de producción solicitadas sobre el paquete modular `rl_vmc`
# y la paquetería `uhf1d` (CASSCF multi-determinante 1D) del asesor:
#
#   1. PIPELINE 1 (Producción de Referencias CASSCF):
#      Calcula (o reutiliza) la función de onda multiconfiguracional CASSCF para
#      cada sistema (N, omega, g) usando parámetros seguros de espacio activo
#      (n_core, n_active_orbitals, n_basis, quadrature_order) extraídos de los
#      `summary.json` ya convergidos por el asesor.
#
#   2. PIPELINE 2 (Muestreo Puro sobre CASSCF Congelado):
#      Fija exactamente los pesos de la función de onda multi-determinante
#      `casscf_state.npz` del Pipeline 1 (`--ansatz_type casscf_npz --casscf_trainable False`)
#      y compara:
#        - MALA estándar (`--proposal_type mala --train_mode eval_only`)
#        - RL-MALA con Actor DeepSet + TD3 (`--proposal_type neural_deepset --train_mode rl_only`)
#      Al terminar cada sistema (N, g), genera de inmediato las gráficas comparativas
#      y actualiza en vivo las tablas CSV y `summary.json`.
#
#   3. PIPELINE 3 (Entrenamiento Variacional de Función de Onda Neuronal):
#      Entrena desde cero el Ansatz variacional Slater-Jastrow con Backflow Neuronal
#      y Jastrow RBF resuelto por espín (`--ansatz_type slater_jastrow_1d
#      --orbital_model backflow_ho --correlation_model spin_rbf_jastrow`) comparando:
#        - VMC + MALA (`--proposal_type mala --train_mode vmc_only`)
#        - VMC + RL-MALA Acoplado (`--proposal_type neural_deepset --train_mode coupled`)
#      Evalúa ambos modelos con múltiples cadenas independientes y genera al instante
#      las figuras comparativas superponiendo la referencia CASSCF del Pipeline 1.
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
#   MAX_PARALLEL_JOBS="4"
#   COPY_ADVISOR_CASSCF="1"   # 1 = reutilizar casscf_state.npz del asesor si coincide exactamente
#   FORCE_RERUN="0"           # 1 = recalcular aunque ya exista run_bundle.h5
# ==============================================================================

set -u

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export TF_FORCE_GPU_ALLOW_GROWTH=true

BASE_OUT_DIR="${BASE_OUT_DIR:-results_production_3pipelines}"
PIPELINES_TO_RUN="${PIPELINES_TO_RUN:-1 2 3}"
N_LIST="${N_LIST:-2 4 6 10}"
G_LIST="${G_LIST:-0.5 1.0 2.0 3.0 5.0 8.0}"
OMEGA="${OMEGA:-0.1}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"
COPY_ADVISOR_CASSCF="${COPY_ADVISOR_CASSCF:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

# Hiperparámetros de muestreo y entrenamiento
N_WALKERS="${N_WALKERS:-256}"
RL_ONLY_STEPS="${RL_ONLY_STEPS:-10000}"
VMC_COUPLED_STEPS="${VMC_COUPLED_STEPS:-15000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-5}"
EVAL_BURN_IN="${EVAL_BURN_IN:-1000}"
MEASURE_STEPS="${MEASURE_STEPS:-6000}"

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

mkdir -p "$P1_DIR" "$P2_DIR" "$P3_DIR" "$LOG_DIR"

echo "=============================================================================="
echo " INICIANDO BENCHMARK DE PRODUCCIÓN EN 3 PIPELINES: $(date)"
echo " Directorio raíz     : $BASE_OUT_DIR"
echo " Pipelines activos   : $PIPELINES_TO_RUN"
echo " Sistemas N          : $N_LIST"
echo " Acoplamientos g     : $G_LIST"
echo " Frecuencia omega    : $OMEGA"
echo " Trabajos paralelos  : $MAX_PARALLEL_JOBS"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# Despachador de GPUs (o CPU si nvidia-smi no está disponible)
# ------------------------------------------------------------------------------
get_free_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return 0
    fi
    while true; do
        local active_jobs
        active_jobs=$(jobs -p | wc -l)
        if [ "$active_jobs" -lt "$MAX_PARALLEL_JOBS" ]; then
            local gpu_id
            gpu_id=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
                     sort -t',' -k2,2n -k3,3n | \
                     head -n 1 | \
                     awk -F', ' '{if ($2 < 16000) print $1; else print -1}')
            if [ -n "$gpu_id" ] && [ "$gpu_id" != "-1" ]; then
                echo "$gpu_id"
                return 0
            fi
        fi
        sleep 4
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
        n_core=0
        # Para g >= 4.0 el asesor utiliza 5 orbitales activos (100 determinantes)
        local is_strong
        is_strong=$(awk -v g="$G" 'BEGIN {print (g >= 4.0) ? 1 : 0}')
        if [ "$is_strong" -eq 1 ]; then
            n_act=5
        else
            n_act=4
        fi
        n_basis=32; quad=80; soft="0.1"
    elif [ "$N" -eq 6 ]; then
        n_core=1; n_act=4; n_basis=32; quad=80; soft="0.1"
    elif [ "$N" -eq 8 ]; then
        n_core=2; n_act=4; n_basis=28; quad=80; soft="0.1"
    elif [ "$N" -eq 10 ]; then
        # En los summary.json de N=10 del asesor: n_core=4, n_active_orbitals=4, n_basis=24, softening=0.25
        n_core=4; n_act=4; n_basis=24; quad=80; soft="0.25"
    else
        n_core=$(( (N / 2) - 2 ))
        if [ "$n_core" -lt 0 ]; then n_core=0; fi
        n_act=4; n_basis=24; quad=80; soft="0.1"
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
# PIPELINE 1: PRODUCCIÓN DE REFERENCIAS EXACTAS DE ESPACIO ACTIVO (CASSCF 1D)
# ==============================================================================
run_pipeline_1() {
    echo ""
    echo "------------------------------------------------------------------------------"
    echo " [PIPELINE 1] Generando referencias CASSCF 1D con paquetería uhf1d..."
    echo " Tabla CSV en vivo: $LIVE_CSV_P1"
    echo "------------------------------------------------------------------------------"

    for N in $N_LIST; do
        local N_UP=$((N / 2))
        local N_DN=$((N - N_UP))
        for G in $G_LIST; do
            read -r NCORE NACT NBASIS QUAD SOFT <<< "$(get_system_casscf_params "$N" "$G")"
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local OUT_CASSCF="${P1_DIR}/${SYS_TAG}"

            local EXTRA_FLAGS="--reuse_existing"
            if [ "$FORCE_RERUN" -eq 1 ]; then
                EXTRA_FLAGS=""
            fi
            if [ "$COPY_ADVISOR_CASSCF" -eq 1 ]; then
                EXTRA_FLAGS="$EXTRA_FLAGS --copy_from_advisor_if_available"
            fi

            echo "  -> [Pipeline 1] Ejecutando CASSCF para ${SYS_TAG} (n_core=${NCORE}, n_act=${NACT}, n_basis=${NBASIS}, a=${SOFT})..."
            python3 cli/run_casscf_1d.py \
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
                --outdir "$OUT_CASSCF" \
                --master_csv "$LIVE_CSV_P1" \
                $EXTRA_FLAGS \
                > "${LOG_DIR}/p1_casscf_${SYS_TAG}.log" 2>&1
        done
    done
    echo " [PIPELINE 1] Completado. Resumen guardado en: $LIVE_CSV_P1"
}

# ==============================================================================
# PIPELINE 2: MUESTREO PURO SOBRE LOS MODELOS CASSCF CONGELADOS (MALA vs RL-MALA)
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

    mkdir -p "$MALA_DIR" "$RL_DIR" "$COMP_DIR"

    if [ ! -f "$CASSCF_NPZ" ]; then
        echo "  [Aviso Pipeline 2] No se encontró $CASSCF_NPZ para ${SYS_TAG}. Ejecutando Pipeline 1 primero..."
        python3 cli/run_casscf_1d.py \
            --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
            --omega "$OMEGA" --g "$G" --softening "$SOFT" \
            --n_core "$NCORE" --n_active_orbitals "$NACT" \
            --n_basis "$NBASIS" --quadrature_order "$QUAD" \
            --outdir "${P1_DIR}/${SYS_TAG}" \
            --master_csv "$LIVE_CSV_P1" \
            --reuse_existing --copy_from_advisor_if_available
    fi

    if [ ! -f "$CASSCF_NPZ" ]; then
        echo "  [Error Pipeline 2] No se pudo generar $CASSCF_NPZ para ${SYS_TAG}. Omitiendo este sistema en Pipeline 2."
        return 1
    fi

    echo "  -> [Pipeline 2 | GPU ${GPU_ID}] Iniciando muestreo puro sobre CASSCF congelado: ${SYS_TAG} (tau=${TAU})"

    # 2A. Muestreo puro con MALA estándar sobre CASSCF congelado
    if [ "$FORCE_RERUN" -eq 1 ] || [ ! -f "${MALA_DIR}/run_bundle.h5" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 cli/run_1d.py \
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
            --output_h5 "${MALA_DIR}/run_bundle.h5" \
            --reference_casscf_npz "$CASSCF_NPZ" \
            --master_csv "$LIVE_CSV_P2_RUNS" \
            --export_legacy_files
    fi

    # 2B. Entrenamiento del Actor RL-MALA (TD3 DeepSet) sobre CASSCF congelado + Evaluación
    if [ "$FORCE_RERUN" -eq 1 ] || [ ! -f "${RL_DIR}/run_bundle.h5" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 cli/run_1d.py \
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
            --output_h5 "${RL_DIR}/run_bundle.h5" \
            --reference_casscf_npz "$CASSCF_NPZ" \
            --master_csv "$LIVE_CSV_P2_RUNS" \
            --export_legacy_files
    fi

    # 2C. Post-procesamiento inmediato por sistema (figuras + CSV comparativo en vivo)
    if [ -f "${MALA_DIR}/run_bundle.h5" ] && [ -f "${RL_DIR}/run_bundle.h5" ]; then
        CUDA_VISIBLE_DEVICES="" python3 cli/postprocess.py \
            --inputs "${MALA_DIR}/run_bundle.h5" "${RL_DIR}/run_bundle.h5" \
            --labels "MALA (Fixed CASSCF)" "RL-MALA TD3 (Fixed CASSCF)" \
            --casscf_npz_path "$CASSCF_NPZ" \
            --out_dir "$COMP_DIR" \
            --master_comparison_csv "$LIVE_CSV_P2_COMP"

        CUDA_VISIBLE_DEVICES="" python3 cli/postprocess.py \
            --inputs "${MALA_DIR}/run_bundle.h5" "${RL_DIR}/run_bundle.h5" \
            --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
            --casscf_npz_path "$CASSCF_NPZ" \
            --out_dir "$COMP_DIR" \
            --master_comparison_csv "$LIVE_CSV_ALL" >/dev/null 2>&1
        echo "  <- [Pipeline 2 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Gráficas y summary.json en: ${COMP_DIR}"
    else
        echo "  [Aviso Pipeline 2] No se encontraron ambos run_bundle.h5 para ${SYS_TAG}. Post-procesamiento omitido."
    fi
}

run_pipeline_2() {
    echo ""
    echo "------------------------------------------------------------------------------"
    echo " [PIPELINE 2] Muestreo Puro sobre CASSCF Congelado (MALA vs RL-MALA TD3)"
    echo " Tablas CSV en vivo: $LIVE_CSV_P2_RUNS | $LIVE_CSV_P2_COMP"
    echo "------------------------------------------------------------------------------"

    for N in $N_LIST; do
        for G in $G_LIST; do
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local GPU_ID
            GPU_ID=$(get_free_gpu)
            run_sistema_pipeline_2 "$GPU_ID" "$N" "$G" \
                > "${LOG_DIR}/p2_sampling_${SYS_TAG}.log" 2>&1 &
            sleep 2
        done
    done
    wait
    echo " [PIPELINE 2] Todos los sistemas han terminado."
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

    mkdir -p "$VMC_MALA_DIR" "$COUPLED_RL_DIR" "$COMP_DIR"

    echo "  -> [Pipeline 3 | GPU ${GPU_ID}] Iniciando entrenamiento variacional SJB: ${SYS_TAG} (tau=${TAU})"

    # 3A. Entrenamiento VMC con MALA estándar (Slater-Jastrow + Backflow + Spin-RBF Jastrow)
    if [ "$FORCE_RERUN" -eq 1 ] || [ ! -f "${VMC_MALA_DIR}/run_bundle.h5" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 cli/run_1d.py \
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
            --output_h5 "${VMC_MALA_DIR}/run_bundle.h5" \
            --reference_casscf_npz "$CASSCF_NPZ" \
            --master_csv "$LIVE_CSV_P3_RUNS" \
            --export_legacy_files
    fi

    # 3B. Entrenamiento Acoplado VMC + RL-MALA (Coupled Trainer con TD3 DeepSet)
    if [ "$FORCE_RERUN" -eq 1 ] || [ ! -f "${COUPLED_RL_DIR}/run_bundle.h5" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 cli/run_1d.py \
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
            --output_h5 "${COUPLED_RL_DIR}/run_bundle.h5" \
            --reference_casscf_npz "$CASSCF_NPZ" \
            --master_csv "$LIVE_CSV_P3_RUNS" \
            --export_legacy_files
    fi

    # 3C. Post-procesamiento inmediato por sistema (comparando VMC-MALA vs Coupled RL-MALA + Baseline CASSCF)
    if [ -f "${VMC_MALA_DIR}/run_bundle.h5" ] && [ -f "${COUPLED_RL_DIR}/run_bundle.h5" ]; then
        local EXTRA_CASSCF_ARG=""
        if [ -f "$CASSCF_NPZ" ]; then
            EXTRA_CASSCF_ARG="--casscf_npz_path $CASSCF_NPZ"
        fi

        CUDA_VISIBLE_DEVICES="" python3 cli/postprocess.py \
            --inputs "${VMC_MALA_DIR}/run_bundle.h5" "${COUPLED_RL_DIR}/run_bundle.h5" \
            --labels "VMC + MALA (SJB)" "Coupled VMC + RL-MALA TD3 (SJB)" \
            $EXTRA_CASSCF_ARG \
            --out_dir "$COMP_DIR" \
            --master_comparison_csv "$LIVE_CSV_P3_COMP"

        CUDA_VISIBLE_DEVICES="" python3 cli/postprocess.py \
            --inputs "${VMC_MALA_DIR}/run_bundle.h5" "${COUPLED_RL_DIR}/run_bundle.h5" \
            --labels "P3_VMC_MALA_SJB_${SYS_TAG}" "P3_Coupled_RL_TD3_SJB_${SYS_TAG}" \
            $EXTRA_CASSCF_ARG \
            --out_dir "$COMP_DIR" \
            --master_comparison_csv "$LIVE_CSV_ALL" >/dev/null 2>&1

        echo "  <- [Pipeline 3 | GPU ${GPU_ID}] Finalizado ${SYS_TAG}. Gráficas y summary.json en: ${COMP_DIR}"
    else
        echo "  [Aviso Pipeline 3] No se encontraron ambos run_bundle.h5 para ${SYS_TAG}. Post-procesamiento omitido."
    fi
}

run_pipeline_3() {
    echo ""
    echo "------------------------------------------------------------------------------"
    echo " [PIPELINE 3] Entrenamiento Variacional SJB (VMC+MALA vs Coupled VMC+RL-MALA)"
    echo " Tablas CSV en vivo: $LIVE_CSV_P3_RUNS | $LIVE_CSV_P3_COMP"
    echo "------------------------------------------------------------------------------"

    for N in $N_LIST; do
        for G in $G_LIST; do
            local SYS_TAG="N${N}_w${OMEGA}_g${G}"
            local GPU_ID
            GPU_ID=$(get_free_gpu)
            run_sistema_pipeline_3 "$GPU_ID" "$N" "$G" \
                > "${LOG_DIR}/p3_training_${SYS_TAG}.log" 2>&1 &
            sleep 2
        done
    done
    wait
    echo " [PIPELINE 3] Todos los sistemas han terminado."
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

echo ""
echo "=============================================================================="
echo " BENCHMARK DE 3 PIPELINES FINALIZADO EXITOSAMENTE: $(date)"
echo " Tablas CSV maestras generadas en vivo:"
echo "   - Pipeline 1 (CASSCF)       : $LIVE_CSV_P1"
echo "   - Pipeline 2 (Muestreo)     : $LIVE_CSV_P2_COMP"
echo "   - Pipeline 3 (Entrenamiento): $LIVE_CSV_P3_COMP"
echo "   - Tabla global consolidada  : $LIVE_CSV_ALL"
echo "=============================================================================="
