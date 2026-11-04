#!/usr/bin/env bash
# ==============================================================================
# RL-VMC 1D: ORQUESTADOR UNIVERSAL DE CRUCE DE BARRERAS NODALES (BAKER & AYLER)
# ==============================================================================
# Propósito:
#   Entrenar y evaluar samplers neuronales RL-MALA (TD3 con arquitectura DeepSet)
#   que superen al algoritmo MALA clásico mediante el cruce activo de barreras
#   nodales y de repulsión de Wigner, replicando y sistematizando el mecanismo
#   físico de `Article_plots_11/evaluar_wigner_casscf.py`.
#
# Características de Diseño:
#   1. AUTO-ADAPTACIÓN BAKER (1 GPU) / AYLER (Multi-GPU):
#      Detecta automáticamente el número de GPUs y la VRAM disponible sin
#      cambiar una sola línea de código.
#      - En Baker (1 GPU): ejecuta hasta 2 trabajos ligeros en paralelo sobre la
#        misma GPU (si VRAM >= 15 GB) con XLA_PYTHON_CLIENT_MEM_FRACTION=0.40
#        y asignación bajo demanda (`platform`), aprovechando al máximo los
#        núcleos CUDA sin agotar la memoria ni colgar el servidor.
#      - En Ayler (3 GPUs): distribuye 1 trabajo por GPU en paralelo.
#   2. CORRECCIÓN CRÍTICA DE ENTRENAMIENTO (`--train_mode rl_only`):
#      Entrena genuinamente los pesos del Actor TD3 sobre la función de onda
#      CASSCF congelada antes de evaluar (evitando el modo `eval_only` que
#      dejaba los pesos del Actor en cero = MALA puro).
#   3. HIPERPARÁMETROS DESBLOQUEADOS PARA CRUCE NODAL:
#      - `sigma_action_scale = 1.1` (permite expandir la varianza de propuesta
#        sigma = sqrt(2*tau)*exp(a_sigma) hasta ~2.7x cerca de las barreras).
#      - `penalty_factor (lambda) = 0.15, 0.25` (recompensa CDLB calibrada).
#      - `lr_actor = 1e-5`, `lr_critic = 3e-5` (aprendizaje TD3 estable).
#      - `eval_init_mode = spin_ordered_offset` (prueba explícita de cruce de
#        barreras nodales entre sectores de espín: up-up-down-down -> simétrico).
#   4. REANUDACIÓN AUTOMÁTICA (CHECKPOINT-SKIP):
#      Si una corrida ya cuenta con `run_bundle.h5` y `summary.json` válidos,
#      se salta automáticamente al reanudar.
#
# Uso directo (en Baker o Ayler):
#   chmod +x run_parameter_search_baker.sh
#   ./run_parameter_search_baker.sh [CARPETA_SALIDA]
#
# Uso sobrescribiendo parámetros desde la terminal:
#   RL_TRAIN_STEPS=10000 MEASURE_STEPS=100000 ./run_parameter_search_baker.sh
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# SECCIÓN 1: PARÁMETROS MODULARES Y DESCRIPTIVOS (MODIFICABLES)
# ==============================================================================

# --- 1.1 Directorios de Salida y Reanudación ---
# Puedes pasar la carpeta destino como primer argumento ($1) o vía OUT_ROOT.
OUT_ROOT="${1:-${OUT_ROOT:-results_nodal_crossing_search}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # 1 = saltar corridas ya terminadas; 0 = recalcular todo

# --- 1.2 Selección de Sistemas Físicos a Evaluar ---
# Sistema A (Recomendado Principal): Molécula de Wigner con barreras profundas
# (N=4, g=4.0, omega=0.1, softening=0.1) — mismo sistema de Article_plots_11.
RUN_SYSTEM_WIGNER_N4_G4="${RUN_SYSTEM_WIGNER_N4_G4:-1}"
NPZ_WIGNER_N4_G4="${NPZ_WIGNER_N4_G4:-uhf_1d_ho_v270826_m-determinants/results_N4/g_4.0/cas_n4_g4_omega0.1_soft0.1_b32_c0-a4/casscf_state.npz}"

# Sistema B (Opcional): Sistema multideterminante de 6 electrones
# (N=6, g=1.0, omega=0.1, softening=0.1, 20 determinantes, base=256).
RUN_SYSTEM_MULTIDET_N6_G1="${RUN_SYSTEM_MULTIDET_N6_G1:-1}"
NPZ_MULTIDET_N6_G1="${NPZ_MULTIDET_N6_G1:-figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz}"

# --- 1.3 Grillas de Búsqueda de Hiperparámetros (MALA vs RL-MALA TD3) ---
# MALA_TAU_GRID   : Tamaños de paso temporal tau para el baseline MALA clásico.
# TD3_TAU_GRID    : Tamaños de paso base tau para el sampler neuronal RL-MALA.
# ACTOR_AMP_GRID  : Amplitud máxima del empuje direccional del Actor (A * tanh).
# PENALTY_GRID    : Factor lambda de la recompensa CDLB (controla aceptación vs exploración).
# SIGMA_SCALE_GRID: Escala máxima del segundo canal del Actor para expandir sigma.
MALA_TAU_GRID=(${MALA_TAU_GRID_STR:-0.04 0.08 0.10 0.12})
TD3_TAU_GRID=(${TD3_TAU_GRID_STR:-0.08 0.10})
ACTOR_AMP_GRID=(${ACTOR_AMP_GRID_STR:-0.10 0.15 0.25})
PENALTY_GRID=(${PENALTY_GRID_STR:-0.15 0.25})
SIGMA_SCALE_GRID=(${SIGMA_SCALE_GRID_STR:-1.1})

# --- 1.4 Hiperparámetros de Entrenamiento RL (TD3 sobre CASSCF congelado) ---
# RL_TRAIN_STEPS     : Barridos MCMC de entrenamiento TD3 (6000 es rápido y converge
#                      el cruce nodal; en Article_plots_11 se usaron 20000).
# N_WALKERS_TRAIN    : Número de caminantes paralelos durante entrenamiento y evaluación.
#                      (100-128 caminantes replican Article_plots_11 con bajo uso de VRAM).
# N_BURN_IN_TRAIN    : Pasos de termalización previos al entrenamiento del Actor.
# RL_LR_ACTOR        : Learning rate del Actor (1e-5 evita saturación prematura de tanh).
# RL_LR_CRITIC       : Learning rate de los Críticos (3e-5 mantiene MSE estable ~0.1).
# RL_USE_LR_SCHEDULER: false = mantiene LR constante como en Article_plots_11.
# MAX_PHYS_JUMP      : Cota física de salto en múltiplos de sqrt(tau) (2.0 en Article_plots_11).
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-6000}"
N_WALKERS="${N_WALKERS:-128}"
N_BURN_IN_TRAIN="${N_BURN_IN_TRAIN:-2000}"
RL_LR_ACTOR="${RL_LR_ACTOR:-1e-5}"
RL_LR_CRITIC="${RL_LR_CRITIC:-3e-5}"
RL_USE_LR_SCHEDULER="${RL_USE_LR_SCHEDULER:-false}"
MAX_PHYS_JUMP="${MAX_PHYS_JUMP:-2.0}"
RL_REPLAY_CAPACITY="${RL_REPLAY_CAPACITY:-50000}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"

# --- 1.5 Configuración de la Fase de Evaluación / Cruce de Barreras Nodales ---
# EVAL_INIT_MODE     : Modo de inicialización de caminantes al evaluar:
#                      - 'spin_ordered_offset': inicializa electrones ordenados por espín
#                        (up a la izquierda, down a la derecha, desplazados en +5 Bohr),
#                        replicando exactamente Article_plots_11 para medir cuántos pasos
#                        tarda el sampler en cruzar las barreras nodales y despolarizar el espín.
#                      - 'symmetric_gaussian': inicialización Gaussiana isotrópica.
# EVAL_BURN_IN       : Pasos descartados antes de medir en evaluación (200 pasos bastan para
#                      caer al pozo armónico y observar toda la dinámica de cruce nodal).
# MEASURE_STEPS      : Pasos MCMC de producción por cadena (40000 para barrido rápido;
#                      subir a 200000-500000 para réplica larga de artículo).
# N_EVAL_CHAINS      : Cadenas independientes de evaluación.
# JIT_CHUNK_SIZE     : Tamaño de bloque compilado lax.scan (protege VRAM).
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"
EVAL_BURN_IN="${EVAL_BURN_IN:-200}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-1}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-2000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-4096}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-128}"

# --- 1.6 Fase 3 (Opcional): Entrenamiento Acoplado Multi-SJB + TD3 ---
# Activar con RUN_PHASE3_MULTISJB=1 si además del muestreo sobre CASSCF congelado
# deseas entrenar el ansatz variacional Multi-SJB con el mejor Actor encontrado.
RUN_PHASE3_MULTISJB="${RUN_PHASE3_MULTISJB:-0}"
COUPLED_CYCLES="${COUPLED_CYCLES:-6}"
VMC_ITERS_PER_CYCLE="${VMC_ITERS_PER_CYCLE:-400}"
RL_SWEEPS_PER_CYCLE="${RL_SWEEPS_PER_CYCLE:-300}"
LR_ANSATZ="${LR_ANSATZ:-2e-4}"

# --- 1.7 Semilla Global y Formatos de Gráficas ---
SEED="${SEED:-42}"
FIGURE_FORMAT="${FIGURE_FORMAT:-both}"  # png, pdf, o both

# ==============================================================================
# SECCIÓN 2: PROTECCIÓN DE MEMORIA Y AUTO-DETECCIÓN DE GPUs (BAKER / AYLER)
# ==============================================================================

# Limitar hilos de CPU por proceso para no saturar RAM ni núcleos del host
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# Desactivar pre-asignación masiva de VRAM en JAX/XLA (obligatorio para no colgar Baker/Ayler)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

AVAILABLE_GPUS=()
TOTAL_VRAM_MB=0

discover_hardware() {
    if [[ -n "${GPU_LIST:-}" ]]; then
        read -r -a AVAILABLE_GPUS <<< "$GPU_LIST"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && AVAILABLE_GPUS+=("$line")
        done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
        TOTAL_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || echo 0)
    fi

    local n_gpus="${#AVAILABLE_GPUS[@]}"
    if [[ "$n_gpus" -eq 0 ]]; then
        AVAILABLE_GPUS=("")
        JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.70}"
    elif [[ "$n_gpus" -eq 1 ]]; then
        # Modo BAKER (1 sola GPU): si tiene >= 15 GB de VRAM y N_WALKERS <= 256,
        # podemos correr 2 trabajos ligeros simultáneos en la misma GPU sin riesgo de OOM.
        if [[ -z "${JOBS_PER_GPU:-}" ]]; then
            if [[ "${TOTAL_VRAM_MB:-0}" -ge 15000 && "$N_WALKERS" -le 256 ]]; then
                JOBS_PER_GPU=2
            else
                JOBS_PER_GPU=1
            fi
        fi
        if [[ "$JOBS_PER_GPU" -ge 2 ]]; then
            export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.40}"
        else
            export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.75}"
        fi
    else
        # Modo AYLER (Múltiples GPUs): 1 trabajo por GPU por defecto
        JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.45}"
    fi
}

discover_hardware
NUM_GPUS="${#AVAILABLE_GPUS[@]}"
if [[ -z "${AVAILABLE_GPUS[0]}" ]]; then
    NUM_GPUS=0
    MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-1}"
else
    MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-$(( NUM_GPUS * JOBS_PER_GPU ))}"
fi

# Construir tabla de slots virtuales (cada slot se mapea a una GPU física)
SLOT_GPUS=()
if [[ "$NUM_GPUS" -eq 0 ]]; then
    for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
        SLOT_GPUS+=("")
    done
else
    for (( j=0; j<JOBS_PER_GPU; j++ )); do
        for g in "${AVAILABLE_GPUS[@]}"; do
            SLOT_GPUS+=("$g")
        done
    done
fi

# Directorios de trabajo y logs en tiempo real
LOG_DIR="${OUT_ROOT}/logs"
CSV_SUMMARY_DIR="${OUT_ROOT}/csv_summary"
FIG_DIR="${OUT_ROOT}/figures"
MASTER_CSV="${CSV_SUMMARY_DIR}/all_models_master.csv"
MASTER_LOG="${LOG_DIR}/master_search.log"

mkdir -p "$LOG_DIR" "$CSV_SUMMARY_DIR" "$FIG_DIR"
touch "$MASTER_LOG"

log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "[$ts] $*" | tee -a "$MASTER_LOG"
}

log_msg "=============================================================================="
log_msg "INICIANDO BÚSQUEDA DE PARÁMETROS Y CRUCE DE BARRERAS NODALES (BAKER / AYLER)"
log_msg "=============================================================================="
log_msg "Directorio raíz        : ${OUT_ROOT}"
log_msg "GPUs físicas detectadas: ${NUM_GPUS} (${AVAILABLE_GPUS[*]:-CPU}) | VRAM por GPU: ${TOTAL_VRAM_MB} MB"
log_msg "Trabajos por GPU       : ${JOBS_PER_GPU} | Concurrencia total: ${MAX_CONCURRENT_JOBS}"
log_msg "XLA_MEM_FRACTION       : ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
log_msg "Entrenamiento TD3      : rl_only (${RL_TRAIN_STEPS} sweeps, N_walkers=${N_WALKERS}, lr_act=${RL_LR_ACTOR}, lr_crit=${RL_LR_CRITIC})"
log_msg "Evaluación MCMC        : init=${EVAL_INIT_MODE}, burn_in=${EVAL_BURN_IN}, measure_steps=${MEASURE_STEPS}"
log_msg "Grilla MALA (tau)      : ${MALA_TAU_GRID[*]}"
log_msg "Grilla TD3 (tau)       : ${TD3_TAU_GRID[*]}"
log_msg "Grilla TD3 (actor_amp) : ${ACTOR_AMP_GRID[*]}"
log_msg "Grilla TD3 (penalty)   : ${PENALTY_GRID[*]}"
log_msg "Grilla TD3 (sigma_sc)  : ${SIGMA_SCALE_GRID[*]}"
log_msg "=============================================================================="

# ==============================================================================
# SECCIÓN 3: COLA DE TRABAJOS CON ASIGNACIÓN DINÁMICA DE SLOTS GPU
# ==============================================================================

declare -a SLOT_PIDS=()
declare -a SLOT_NAMES=()
for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
    SLOT_PIDS[$s]=0
    SLOT_NAMES[$s]=""
done

is_run_completed() {
    local dir="$1"
    if [[ "$SKIP_EXISTING" -eq 1 && -f "${dir}/run_bundle.h5" && -f "${dir}/summary.json" ]]; then
        return 0
    fi
    return 1
}

wait_for_free_slot() {
    while true; do
        for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
            local pid="${SLOT_PIDS[$s]}"
            if [[ "$pid" -eq 0 ]]; then
                FREE_SLOT=$s
                return 0
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                wait "$pid" 2>/dev/null || true
                log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]}') liberado por '${SLOT_NAMES[$s]}'"
                SLOT_PIDS[$s]=0
                SLOT_NAMES[$s]=""
                FREE_SLOT=$s
                return 0
            fi
        done
        sleep 2
    done
}

wait_all_slots() {
    for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
        local pid="${SLOT_PIDS[$s]}"
        if [[ "$pid" -ne 0 ]]; then
            wait "$pid" 2>/dev/null || true
            log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]}') liberado por '${SLOT_NAMES[$s]}'"
            SLOT_PIDS[$s]=0
            SLOT_NAMES[$s]=""
        fi
    done
}

launch_job() {
    local run_name="$1"
    local run_dir="$2"
    shift 2
    local cmd=("$@")

    if is_run_completed "$run_dir"; then
        log_msg "[SKIP] '${run_name}' ya cuenta con run_bundle.h5 y summary.json en ${run_dir}."
        return 0
    fi

    mkdir -p "$run_dir"
    local log_file="${LOG_DIR}/${run_name}.log"

    wait_for_free_slot
    local slot="$FREE_SLOT"
    local gpu_id="${SLOT_GPUS[$slot]}"

    log_msg "[RUN] Lanzando '${run_name}' en Slot #${slot} (GPU='${gpu_id:-CPU}') -> Log: ${log_file}"
    (
        if [[ -n "$gpu_id" ]]; then
            export CUDA_VISIBLE_DEVICES="$gpu_id"
        fi
        "${cmd[@]}" > "$log_file" 2>&1
    ) &
    SLOT_PIDS[$slot]=$!
    SLOT_NAMES[$slot]="$run_name"
}

# ==============================================================================
# SECCIÓN 4: EJECUCIÓN DEL PIPELINE POR SISTEMA FÍSICO
# ==============================================================================

run_system_pipeline() {
    local sys_tag="$1"
    local npz_path="$2"
    local n_elec="$3"
    local n_up="$4"
    local n_dn="$5"
    local omega="$6"
    local g_val="$7"
    local soft="$8"

    if [[ ! -f "$npz_path" ]]; then
        log_msg "[AVISO] No se encontró el archivo CASSCF '${npz_path}' para ${sys_tag}. Saltando sistema."
        return 0
    fi

    local sys_out="${OUT_ROOT}/${sys_tag}"
    local sys_fig="${FIG_DIR}/${sys_tag}"
    mkdir -p "$sys_out" "$sys_fig"

    log_msg "------------------------------------------------------------------------------"
    log_msg "SISTEMA: ${sys_tag} (N=${n_elec}, up=${n_up}, dn=${n_dn}, g=${g_val}, omega=${omega})"
    log_msg "Archivo CASSCF: ${npz_path}"
    log_msg "------------------------------------------------------------------------------"

    local common_args=(
        --n_elec "$n_elec"
        --n_up "$n_up"
        --n_dn "$n_dn"
        --omega "$omega"
        --interaction_strength "$g_val"
        --softening "$soft"
        --batch_size "$N_WALKERS"
        --init_mode "symmetric_gaussian"
        --init_std 3.0
        --eval_init_mode "$EVAL_INIT_MODE"
        --n_eval_chains "$N_EVAL_CHAINS"
        --eval_burn_in "$EVAL_BURN_IN"
        --measure_steps "$MEASURE_STEPS"
        --jit_chunk_size "$JIT_CHUNK_SIZE"
        --eloc_chunk_size "$ELOC_CHUNK_SIZE"
        --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
        --reference_casscf_npz "$npz_path"
        --master_csv "$MASTER_CSV"
        --csv_collect_dir "$CSV_SUMMARY_DIR"
        --seed "$SEED"
    )

    # --------------------------------------------------------------------------
    # FASE 1: BARRIDO DE BASELINES MALA CLÁSICO (CASSCF CONGELADO)
    # --------------------------------------------------------------------------
    log_msg "[${sys_tag} - FASE 1] Muestreo MALA Clásico sobre CASSCF congelado..."
    for tau in "${MALA_TAU_GRID[@]}"; do
        local rname="${sys_tag}_MALA_tau${tau}"
        local rdir="${sys_out}/${rname}"
        launch_job "$rname" "$rdir" \
            python3 -u cli/run_1d.py \
            "${common_args[@]}" \
            --method MALA \
            --train_mode eval_only \
            --ansatz_type casscf_npz \
            --casscf_npz_path "$npz_path" \
            --no-casscf_trainable \
            --tau "$tau" \
            --pipeline_stage "phase1_mala_baseline" \
            --run_name "$rname" \
            --results_out_dir "$rdir"
    done

    # --------------------------------------------------------------------------
    # FASE 2: ENTRENAMIENTO Y EVALUACIÓN RL-MALA (TD3 CON CRUCE NODAL)
    # ¡IMPORTANTE!: Usa `--train_mode rl_only` para entrenar el Actor TD3 sobre
    # el estado CASSCF antes de correr la evaluación de cruce nodal.
    # --------------------------------------------------------------------------
    log_msg "[${sys_tag} - FASE 2] Entrenamiento TD3 (--train_mode rl_only) + Evaluación de Cruce Nodal..."
    local sched_flag="--no-use_lr_scheduler"
    if [[ "$RL_USE_LR_SCHEDULER" == "true" || "$RL_USE_LR_SCHEDULER" == "1" ]]; then
        sched_flag="--use_lr_scheduler"
    fi

    for tau in "${TD3_TAU_GRID[@]}"; do
        for amp in "${ACTOR_AMP_GRID[@]}"; do
            for pen in "${PENALTY_GRID[@]}"; do
                for sig in "${SIGMA_SCALE_GRID[@]}"; do
                    local rname="${sys_tag}_RL_MALA_tau${tau}_amp${amp}_pen${pen}_sig${sig}"
                    local rdir="${sys_out}/${rname}"
                    launch_job "$rname" "$rdir" \
                        python3 -u cli/run_1d.py \
                        "${common_args[@]}" \
                        --method TD3 \
                        --train_mode rl_only \
                        --train_steps "$RL_TRAIN_STEPS" \
                        --n_burn_in "$N_BURN_IN_TRAIN" \
                        --ansatz_type casscf_npz \
                        --casscf_npz_path "$npz_path" \
                        --no-casscf_trainable \
                        --tau "$tau" \
                        --actor_amplitude "$amp" \
                        --sigma_action_scale "$sig" \
                        --reward_type cdlb \
                        --penalty_factor "$pen" \
                        --max_phys_jump_factor "$MAX_PHYS_JUMP" \
                        --lr_actor "$RL_LR_ACTOR" \
                        --lr_critic "$RL_LR_CRITIC" \
                        "$sched_flag" \
                        --rl_replay_capacity "$RL_REPLAY_CAPACITY" \
                        --rl_batch_size "$RL_BATCH_SIZE" \
                        --use_actor_in_eval \
                        --pipeline_stage "phase2_rl_nodal_crossing" \
                        --run_name "$rname" \
                        --results_out_dir "$rdir"
                done
            done
        done
    done

    # Esperar a que terminen Fase 1 y Fase 2 de este sistema
    wait_all_slots

    # --------------------------------------------------------------------------
    # FASE 3 (OPCIONAL): ENTRENAMIENTO ACOPLADO MULTI-SJB + TD3
    # --------------------------------------------------------------------------
    if [[ "$RUN_PHASE3_MULTISJB" -eq 1 ]]; then
        log_msg "[${sys_tag} - FASE 3] Entrenamiento acoplado Multi-SJB + TD3..."
        local rname_p3="${sys_tag}_Coupled_MultiSJB_TD3"
        local rdir_p3="${sys_out}/${rname_p3}"
        launch_job "$rname_p3" "$rdir_p3" \
            python3 -u cli/run_1d.py \
            "${common_args[@]}" \
            --method TD3 \
            --train_mode coupled \
            --ansatz_type multi_sjb \
            --casscf_npz_path "$npz_path" \
            --coupled_training_cycles "$COUPLED_CYCLES" \
            --vmc_iterations_per_cycle "$VMC_ITERS_PER_CYCLE" \
            --rl_sweeps_per_cycle "$RL_SWEEPS_PER_CYCLE" \
            --lr_ansatz "$LR_ANSATZ" \
            --lr_actor "$RL_LR_ACTOR" \
            --lr_critic "$RL_LR_CRITIC" \
            "$sched_flag" \
            --tau "0.10" \
            --actor_amplitude "0.15" \
            --sigma_action_scale "1.1" \
            --penalty_factor "0.25" \
            --max_phys_jump_factor "$MAX_PHYS_JUMP" \
            --pipeline_stage "phase3_coupled_multisjb" \
            --run_name "$rname_p3" \
            --results_out_dir "$rdir_p3"
        wait_all_slots
    fi

    # --------------------------------------------------------------------------
    # FASE 4: POST-PROCESAMIENTO Y GRÁFICAS DE CRUCE NODAL (TIPO ARTICLE_PLOTS_11)
    # --------------------------------------------------------------------------
    log_msg "[${sys_tag} - FASE 4] Generando diagnósticos, D_KL por espín y figuras de artículo..."
    local h5_files=()
    local h5_labels=()

    for d in "${sys_out}"/*; do
        if [[ -d "$d" && -f "${d}/run_bundle.h5" ]]; then
            h5_files+=("${d}/run_bundle.h5")
            local bname
            bname="$(basename "$d")"
            # Etiqueta corta quitando el prefijo del sistema
            h5_labels+=("${bname#${sys_tag}_}")
        fi
    done

    if [[ "${#h5_files[@]}" -gt 0 ]]; then
        # 4.1 Post-procesamiento general (tablas de autocorrelación, ESS/s, energías)
        python3 -u cli/postprocess_1d.py \
            --inputs "${h5_files[@]}" \
            --labels "${h5_labels[@]}" \
            --casscf_npz_path "$npz_path" \
            --out_dir "${sys_fig}/postprocess_standard" \
            --csv_collect_dir "$CSV_SUMMARY_DIR" \
            --system_tag "$sys_tag" \
            --figure_format "$FIGURE_FORMAT" \
            > "${LOG_DIR}/postprocess_${sys_tag}.log" 2>&1 || log_msg "[AVISO] postprocess_1d reportó advertencias en ${sys_tag}"

        # 4.2 Evaluador especializado de Cruce de Barreras Nodales (réplica Article_plots_11 + Ranking)
        python3 -u cli/evaluate_nodal_crossing_1d.py \
            --inputs "${h5_files[@]}" \
            --labels "${h5_labels[@]}" \
            --casscf_npz_path "$npz_path" \
            --out_dir "$sys_fig" \
            --csv_collect_dir "$CSV_SUMMARY_DIR" \
            --system_tag "$sys_tag" \
            --figure_format "$FIGURE_FORMAT" \
            > "${LOG_DIR}/nodal_crossing_eval_${sys_tag}.log" 2>&1 || log_msg "[AVISO] evaluate_nodal_crossing_1d reportó advertencias en ${sys_tag}"

        log_msg "[OK] Figuras y tablas CSV de ${sys_tag} guardadas en: ${sys_fig} y ${CSV_SUMMARY_DIR}"
    else
        log_msg "[AVISO] No se encontraron archivos .h5 para post-procesar en ${sys_out}."
    fi
}

# Ejecutar los sistemas habilitados
if [[ "$RUN_SYSTEM_WIGNER_N4_G4" -eq 1 ]]; then
    run_system_pipeline "Wigner_N4_g4.0" "$NPZ_WIGNER_N4_G4" 4 2 2 0.1 4.0 0.1
fi

if [[ "$RUN_SYSTEM_MULTIDET_N6_G1" -eq 1 ]]; then
    run_system_pipeline "MultiDet_N6_g1.0" "$NPZ_MULTIDET_N6_G1" 6 3 3 0.1 1.0 0.1
fi

log_msg "=============================================================================="
log_msg "PIPELINE COMPLETADO EXITOSAMENTE"
log_msg "  -> Tablas CSV concentradas en : ${CSV_SUMMARY_DIR}"
log_msg "  -> Figuras generadas en       : ${FIG_DIR}"
log_msg "  -> Bitácora maestra en        : ${MASTER_LOG}"
log_msg "=============================================================================="
