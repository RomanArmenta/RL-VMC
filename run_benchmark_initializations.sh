#!/usr/bin/env bash
# ==============================================================================
# BENCHMARK DE ESTRATEGIAS DE INICIALIZACIÓN: MALA vs RL-MALA EN CASSCF 1D
# ==============================================================================
# Propósito:
#   Evaluar sistemáticamente la sensibilidad, resiliencia y convergencia de
#   muestreadores MALA clásico y neuronales RL-MALA (TD3 DeepSet) frente a
#   tres estrategias de inicialización espacial de los caminantes MCMC:
#
#   1) alternating_spin   : "Un electrón en cada Gaussiana" (centros espaciales
#                           escalados a la trampa con jitter térmico local).
#   2) symmetric_gaussian : "Gaussiana centrada en x=0 cubriendo la envolvente"
#                           (sigma calibrado al radio fermiónico del sistema).
#   3) uniform            : "Uniforme en la caja de confinamiento [-L, L]"
#                           (partículas distribuidas sin sesgo armónico inicial).
#
# Sistemas de Referencia (en figures/):
#   - Referencia 1: N=4, g=1.0, omega=0.1, softening=0.1, base 256, 6 determinantes
#     NPZ: figures/casscf-state_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6.npz
#     JSON: figures/casscf-summary_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6_summary.json
#   - Referencia 2: N=8, g=1.0, omega=0.1, softening=0.1, base 256, det 70
#     NPZ: figures/casscf_state_cas_n8_g1_omega0.1_soft0.1_b256_c0-a8_det70.npz
#     JSON: figures/summary_cas_n8_g1_omega0.1_soft0.1_b256_c0-a8_det70.json
#
# Hiperparámetros Óptimos y Extrapolación Prudente para N=8:
#   - N=4:
#       * MALA    : tau = 0.10
#       * RL-MALA : tau = 0.10, A_mu = 0.10, lambda_pen = 0.25, sigma_scale = 1.10
#   - N=8 (Extrapolación Prudente por Dimensión Espacial R^8):
#       * MALA    : tau = 0.08 (escalamiento conservador para acotar rechazos en 8D)
#       * RL-MALA : tau = 0.08, A_mu = 0.10, lambda_pen = 0.25, sigma_scale = 1.10
#
# Rigor Estadístico:
#   - Cada combinación (Sistema × Modelo × Inicialización) se ejecuta n_exp veces
#     con semillas estocásticas independientes para evaluar repetibilidad.
#   - Cada corrida evalúa n_eval cadenas MCMC independientes.
#   - El post-procesador genera gráficas con barras de error rigurosas (SEM).
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN PREDETERMINADA Y PARÁMETROS MODIFICABLES
# ==============================================================================

OUT_ROOT="${OUT_ROOT:-results_benchmark_initializations}"
SYSTEM="${SYSTEM:-all}"
INIT_MODE_FILTER="${INIT_MODE_FILTER:-all}"

N_EXP="${N_EXP:-3}"              # Número de réplicas independientes con distinta semilla
N_EVAL="${N_EVAL:-3}"            # Número de cadenas de evaluación por corrida (barras de error)
BASE_SEED="${BASE_SEED:-42}"

# Hiperparámetros MCMC y de Producción
MEASURE_STEPS="${MEASURE_STEPS:-15000}"
EVAL_BURN_IN="${EVAL_BURN_IN:-1000}"
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-4000}"
N_BURN_IN_TRAIN="${N_BURN_IN_TRAIN:-1000}"
N_WALKERS="${N_WALKERS:-128}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-1000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-128}"

# Formato y estilo de figuras
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"

POSTPROCESS_ONLY=0
SKIP_EXISTING=1

# Procesamiento de argumentos por línea de comandos
while [[ $# -gt 0 ]]; do
    case "$1" in
        --system|-s)
            SYSTEM="$2"
            shift 2
            ;;
        --n-exp|--n_exp|-e)
            N_EXP="$2"
            shift 2
            ;;
        --n-eval|--n_eval|-c)
            N_EVAL="$2"
            shift 2
            ;;
        --init-mode|--init_mode)
            INIT_MODE_FILTER="$2"
            shift 2
            ;;
        --walkers|-w)
            N_WALKERS="$2"
            shift 2
            ;;
        --measure-steps|-m)
            MEASURE_STEPS="$2"
            shift 2
            ;;
        --rl-steps|-r)
            RL_TRAIN_STEPS="$2"
            shift 2
            ;;
        --gpu|--gpus)
            GPU_ARG="$2"
            shift 2
            ;;
        --jobs-per-gpu|-j)
            JOBS_PER_GPU="$2"
            shift 2
            ;;
        --format|-f)
            FIGURE_FORMAT="$2"
            shift 2
            ;;
        --pdf)
            FIGURE_FORMAT="pdf"
            shift
            ;;
        --png)
            FIGURE_FORMAT="png"
            shift
            ;;
        --dpi)
            FIGURE_DPI="$2"
            shift 2
            ;;
        --postprocess-only|-p)
            POSTPROCESS_ONLY=1
            shift
            ;;
        --force-recalculate)
            SKIP_EXISTING=0
            shift
            ;;
        --help|-h)
            echo "Uso: $0 [OPCIONES] [CARPETA_SALIDA]"
            echo ""
            echo "Opciones principales:"
            echo "  --system, -s [N4|N8|all]        Sistema CASSCF a evaluar (default: all)"
            echo "  --init-mode [all|alternating_spin|symmetric_gaussian|uniform]"
            echo "                                    Filtra la estrategia de inicialización (default: all)"
            echo "  --n-exp, -e N                     Número de réplicas independientes por configuración (default: 3)"
            echo "  --n-eval, -c N                    Número de cadenas MCMC de evaluación por réplica (default: 3)"
            echo "  --measure-steps, -m N             Pasos MCMC de producción por cadena (default: 15000)"
            echo "  --rl-steps, -r N                  Pasos de entrenamiento TD3 para RL-MALA (default: 4000)"
            echo "  --walkers, -w N                   Número de caminantes (default: 128)"
            echo "  --gpu [ID|'ID1 ID2'|auto]         Asignación de GPU(s) (default: auto)"
            echo "  --jobs-per-gpu, -j N              Trabajos paralelos por GPU (default: 1)"
            echo "  --format, -f [png|pdf|both]       Formato de exportación gráfica (default: png)"
            echo "  --dpi N                           Resolución de las figuras en DPI (default: 300)"
            echo "  --postprocess-only, -p            Solo compila figuras y tablas sobre resultados existentes"
            echo "  --force-recalculate               Re-ejecuta corridas aunque ya existan"
            echo "  [CARPETA_SALIDA]                  Directorio de salida (default: results_benchmark_initializations)"
            exit 0
            ;;
        *)
            OUT_ROOT="$1"
            shift
            ;;
    esac
done

LOG_DIR="${OUT_ROOT}/logs"
FIG_DIR="${OUT_ROOT}/figures"
MASTER_CSV="${OUT_ROOT}/master_initialization_benchmark.csv"
MASTER_LOG="${LOG_DIR}/master_benchmark.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$FIG_DIR"

log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $1" | tee -a "$MASTER_LOG"
}

# ==============================================================================
# SECCIÓN 2: PROTECCIÓN DE MEMORIA Y AUTO-DETECCIÓN DE GPUs / ENTORNO
# ==============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.25}"
export JAX_ENABLE_X64=True

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || echo "python3")}"

# Detección dinámica de GPUs
GPU_ARG="${GPU_ARG:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

if [[ "$GPU_ARG" != "auto" && -n "$GPU_ARG" ]]; then
    read -r -a GPUS <<< "$GPU_ARG"
elif command -v nvidia-smi >/dev/null 2>&1; then
    DETECTED_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | tr '\n' ' ')
    if [[ -n "$DETECTED_GPUS" ]]; then
        read -r -a GPUS <<< "$DETECTED_GPUS"
    else
        GPUS=(-1)
    fi
else
    GPUS=(-1)
fi

SLOT_GPUS=()
if [[ "${GPUS[0]}" -eq -1 ]]; then
    # Modo CPU
    for (( j=0; j<JOBS_PER_GPU; j++ )); do
        SLOT_GPUS+=("")
    done
else
    for gpu_id in "${GPUS[@]}"; do
        for (( j=0; j<JOBS_PER_GPU; j++ )); do
            SLOT_GPUS+=("$gpu_id")
        done
    done
fi

MAX_CONCURRENT_JOBS="${#SLOT_GPUS[@]}"
if [[ "$MAX_CONCURRENT_JOBS" -lt 1 ]]; then
    MAX_CONCURRENT_JOBS=1
    SLOT_GPUS+=("")
fi

declare -a SLOT_PIDS
declare -a SLOT_NAMES
declare -a SLOT_RUN_DIRS
for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
    SLOT_PIDS[$s]=0
    SLOT_NAMES[$s]=""
    SLOT_RUN_DIRS[$s]=""
done

FREE_SLOT=-1

is_run_completed() {
    local rdir="$1"
    if [[ "$SKIP_EXISTING" -eq 0 ]]; then
        return 1
    fi
    if [[ -f "${rdir}/run_bundle.h5" ]]; then
        local sz=0
        sz=$(wc -c < "${rdir}/run_bundle.h5" 2>/dev/null || echo 0)
        if [[ "$sz" -gt 10000 ]]; then
            return 0
        fi
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
                local prev_run="${SLOT_NAMES[$s]}"
                log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]:-CPU}') liberado por '${prev_run}'"
                SLOT_PIDS[$s]=0
                SLOT_NAMES[$s]=""
                SLOT_RUN_DIRS[$s]=""
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
            log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]:-CPU}') liberado por '${SLOT_NAMES[$s]}'"
            SLOT_PIDS[$s]=0
            SLOT_NAMES[$s]=""
            SLOT_RUN_DIRS[$s]=""
        fi
    done
}

launch_job() {
    local sys_tag="$1"
    local npz_path="$2"
    local run_name="$3"
    local run_dir="$4"
    shift 4
    local cmd=("$@")

    if is_run_completed "$run_dir"; then
        log_msg "[SKIP] '${run_name}' ya completado en ${run_dir}."
        return 0
    fi

    mkdir -p "$run_dir"
    local log_file="${LOG_DIR}/${run_name}.log"

    wait_for_free_slot
    local slot="$FREE_SLOT"
    local gpu_id="${SLOT_GPUS[$slot]}"

    log_msg "[RUN] Lanzando '${run_name}' en Slot #${slot} (Dispositivo: ${gpu_id:-CPU}) -> Log: ${log_file}"
    (
        if [[ -n "$gpu_id" && "$gpu_id" != "-1" ]]; then
            export CUDA_VISIBLE_DEVICES="$gpu_id"
        else
            export CUDA_VISIBLE_DEVICES=""
            export JAX_PLATFORMS=cpu
        fi
        local t0
        t0=$(date +%s)
        "${cmd[@]}" > "$log_file" 2>&1
        local exit_code=$?
        local t_elapsed=$(( $(date +%s) - t0 ))
        if [[ "$exit_code" -ne 0 ]]; then
            log_msg "[ALERTA] '${run_name}' falló con código ${exit_code} tras ${t_elapsed}s. Revisa: ${log_file}"
        fi
    ) &

    SLOT_PIDS[$slot]=$!
    SLOT_NAMES[$slot]="$run_name"
    SLOT_RUN_DIRS[$slot]="$run_dir"
    sleep 1
}

# ==============================================================================
# SECCIÓN 3: DEFINICIÓN DE SISTEMAS Y ESTRATEGIAS DE INICIALIZACIÓN
# ==============================================================================

# 1. Referencias CASSCF
NPZ_N4="figures/casscf-state_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6.npz"
NPZ_N8="figures/casscf_state_cas_n8_g1_omega0.1_soft0.1_b256_c0-a8_det70.npz"

# Comprobación de existencia
if [[ ! -f "$NPZ_N4" ]]; then
    log_msg "[ERROR FATAL] No se encontró el archivo CASSCF de N=4 en: $NPZ_N4"
    exit 1
fi
if [[ ! -f "$NPZ_N8" ]]; then
    log_msg "[ERROR FATAL] No se encontró el archivo CASSCF de N=8 en: $NPZ_N8"
    exit 1
fi

# Parámetros físicos N=4
N4_ELEC=4
N4_UP=2
N4_DN=2
N4_OMEGA=0.1
N4_G=1.0
N4_SOFT=0.1
N4_MALA_TAU=0.10
N4_RL_TAU=0.10
N4_RL_AMP=0.10
N4_RL_PEN=0.25
N4_RL_SIG=1.10
N4_SIGMA_ENV=6.0          # Sigma que cubre la envolvente fermiónica en x=0 (Bohr)
N4_BOX_HALF=10.0          # Semiancho de caja para distribución uniforme [-L, L] (Bohr)

# Parámetros físicos N=8 (Extrapolación Prudente)
N8_ELEC=8
N8_UP=4
N8_DN=4
N8_OMEGA=0.1
N8_G=1.0
N8_SOFT=0.1
N8_MALA_TAU=0.08          # Extrapolación prudente: reducción a 0.08 para acotar rechazos en 8D
N8_RL_TAU=0.08            # Extrapolación prudente: reducción a 0.08
N8_RL_AMP=0.10            # Desplazamiento residual acotado al espaciado inter-partícula
N8_RL_PEN=0.25
N8_RL_SIG=1.10
N8_SIGMA_ENV=9.0          # Sigma que cubre la envolvente de 8 fermiones en x=0 (Bohr)
N8_BOX_HALF=15.0          # Semiancho de caja para distribución uniforme [-L, L] (Bohr)

# Lista de inicializaciones a evaluar
ALL_INITS=("alternating_spin" "symmetric_gaussian" "uniform")
ACTIVE_INITS=()
if [[ "$INIT_MODE_FILTER" == "all" ]]; then
    ACTIVE_INITS=("${ALL_INITS[@]}")
else
    ACTIVE_INITS=("$INIT_MODE_FILTER")
fi

log_msg "=============================================================================="
log_msg "    INICIO DEL BENCHMARK DE ESTRATEGIAS DE INICIALIZACIÓN (MALA vs RL-MALA)"
log_msg "=============================================================================="
log_msg "Directorio de Salida   : ${OUT_ROOT}"
log_msg "Sistema(s) Objetivo    : ${SYSTEM}"
log_msg "Inicializaciones       : ${ACTIVE_INITS[*]}"
log_msg "Réplicas por Config    : ${N_EXP} (con distintas seeds)"
log_msg "Cadenas por Corrida    : ${N_EVAL} (para barras de error SEM)"
log_msg "Pasos de Medición      : ${MEASURE_STEPS}"
log_msg "Pasos de Burn-in       : ${EVAL_BURN_IN}"
log_msg "Barridos RL TD3        : ${RL_TRAIN_STEPS}"
log_msg "Caminantes MCMC        : ${N_WALKERS}"
log_msg "Slots Concurrentes     : ${MAX_CONCURRENT_JOBS} (GPUs: ${SLOT_GPUS[*]:-CPU})"
log_msg "=============================================================================="

# ==============================================================================
# SECCIÓN 4: ORQUESTACIÓN DE LAS CORRIDAS EXPERIMENTALES
# ==============================================================================

run_benchmark_for_system() {
    local sys_tag="$1"
    local npz_path="$2"
    local n_elec="$3"
    local n_up="$4"
    local n_dn="$5"
    local omega="$6"
    local g_val="$7"
    local soft="$8"
    local mala_tau="$9"
    local rl_tau="${10}"
    local rl_amp="${11}"
    local rl_pen="${12}"
    local rl_sig="${13}"
    local sigma_env="${14}"
    local box_half="${15}"

    local sys_out="${OUT_ROOT}/${sys_tag}"
    mkdir -p "$sys_out"

    log_msg "------------------------------------------------------------------------------"
    log_msg ">>> PROCESANDO SISTEMA: ${sys_tag} (N=${n_elec}, g=${g_val}, omega=${omega})"
    log_msg "    Archivo CASSCF Referencia : ${npz_path}"
    log_msg "    MALA tau óptimo/extrap.   : ${mala_tau}"
    log_msg "    RL-MALA tau/amp/pen/sig   : tau=${rl_tau}, A=${rl_amp}, lambda=${rl_pen}, sigma=${rl_sig}"
    log_msg "    Gaussiana Envolvente sigma: ${sigma_env} Bohr | Uniforme semiancho: ${box_half} Bohr"
    log_msg "------------------------------------------------------------------------------"

    for init_mode in "${ACTIVE_INITS[@]}"; do
        local init_tag=""
        local init_cli_args=()

        case "$init_mode" in
            alternating_spin)
                init_tag="alt_spin"
                init_cli_args=(
                    --init_mode "alternating_spin"
                    --eval_init_mode "same_as_sampler"
                )
                ;;
            symmetric_gaussian)
                init_tag="sym_gauss"
                init_cli_args=(
                    --init_mode "symmetric_gaussian"
                    --init_std "$sigma_env"
                    --eval_init_mode "same_as_sampler"
                )
                ;;
            uniform)
                init_tag="uniform"
                init_cli_args=(
                    --init_mode "uniform"
                    --init_uniform_halfwidth "$box_half"
                    --eval_init_mode "same_as_sampler"
                )
                ;;
            *)
                log_msg "[AVISO] Modo de inicialización desconocido '${init_mode}'. Saltando."
                continue
                ;;
        esac

        for (( exp=1; exp<=N_EXP; exp++ )); do
            local cur_seed=$(( BASE_SEED + (exp - 1) * 1000 ))

            local common_args=(
                --n_elec "$n_elec"
                --n_up "$n_up"
                --n_dn "$n_dn"
                --omega "$omega"
                --interaction_strength "$g_val"
                --softening "$soft"
                --batch_size "$N_WALKERS"
                "${init_cli_args[@]}"
                --n_eval_chains "$N_EVAL"
                --eval_burn_in "$EVAL_BURN_IN"
                --measure_steps "$MEASURE_STEPS"
                --jit_chunk_size "$JIT_CHUNK_SIZE"
                --eloc_chunk_size "$ELOC_CHUNK_SIZE"
                --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
                --reference_casscf_npz "$npz_path"
                --master_csv "$MASTER_CSV"
                --seed "$cur_seed"
            )

            # 1. Corrida MALA Clásico (eval_only sobre CASSCF congelado)
            local mala_rname="${sys_tag}_MALA_${init_tag}_exp${exp}"
            local mala_rdir="${sys_out}/${mala_rname}"
            launch_job "$sys_tag" "$npz_path" "$mala_rname" "$mala_rdir" \
                "$PYTHON_BIN" -u cli/run_1d.py \
                "${common_args[@]}" \
                --method MALA \
                --train_mode eval_only \
                --ansatz_type casscf_npz \
                --casscf_npz_path "$npz_path" \
                --no-casscf_trainable \
                --tau "$mala_tau" \
                --pipeline_stage "init_benchmark_mala" \
                --run_name "$mala_rname" \
                --results_out_dir "$mala_rdir"

            # 2. Corrida RL-MALA (TD3 DeepSet, train_mode rl_only sobre CASSCF congelado + evaluación)
            local rl_rname="${sys_tag}_RL_MALA_${init_tag}_exp${exp}"
            local rl_rdir="${sys_out}/${rl_rname}"
            launch_job "$sys_tag" "$npz_path" "$rl_rname" "$rl_rdir" \
                "$PYTHON_BIN" -u cli/run_1d.py \
                "${common_args[@]}" \
                --method TD3 \
                --train_mode rl_only \
                --train_steps "$RL_TRAIN_STEPS" \
                --n_burn_in "$N_BURN_IN_TRAIN" \
                --ansatz_type casscf_npz \
                --casscf_npz_path "$npz_path" \
                --no-casscf_trainable \
                --tau "$rl_tau" \
                --actor_amplitude "$rl_amp" \
                --sigma_action_scale "$rl_sig" \
                --reward_type cdlb \
                --penalty_factor "$rl_pen" \
                --lr_actor 1e-5 \
                --lr_critic 3e-5 \
                --use_actor_in_eval \
                --pipeline_stage "init_benchmark_rl_mala" \
                --run_name "$rl_rname" \
                --results_out_dir "$rl_rdir"
        done
    done
}

if [[ "$POSTPROCESS_ONLY" -eq 1 ]]; then
    log_msg "[MODO POSTPROCESS-ONLY] Omitiendo despacho de simulaciones. Compilando directamente figuras y estadísticas..."
else
    SYS_LOWER="$(echo "$SYSTEM" | tr '[:upper:]' '[:lower:]')"
    case "$SYS_LOWER" in
        n4)
            run_benchmark_for_system "N4_w0.1_g1.0" "$NPZ_N4" \
                "$N4_ELEC" "$N4_UP" "$N4_DN" "$N4_OMEGA" "$N4_G" "$N4_SOFT" \
                "$N4_MALA_TAU" "$N4_RL_TAU" "$N4_RL_AMP" "$N4_RL_PEN" "$N4_RL_SIG" \
                "$N4_SIGMA_ENV" "$N4_BOX_HALF"
            ;;
        n8)
            run_benchmark_for_system "N8_w0.1_g1.0" "$NPZ_N8" \
                "$N8_ELEC" "$N8_UP" "$N8_DN" "$N8_OMEGA" "$N8_G" "$N8_SOFT" \
                "$N8_MALA_TAU" "$N8_RL_TAU" "$N8_RL_AMP" "$N8_RL_PEN" "$N8_RL_SIG" \
                "$N8_SIGMA_ENV" "$N8_BOX_HALF"
            ;;
        all|*)
            run_benchmark_for_system "N4_w0.1_g1.0" "$NPZ_N4" \
                "$N4_ELEC" "$N4_UP" "$N4_DN" "$N4_OMEGA" "$N4_G" "$N4_SOFT" \
                "$N4_MALA_TAU" "$N4_RL_TAU" "$N4_RL_AMP" "$N4_RL_PEN" "$N4_RL_SIG" \
                "$N4_SIGMA_ENV" "$N4_BOX_HALF"

            log_msg "[COLA] Esperando finalización de trabajos para N4..."
            wait_all_slots
            log_msg "[OK] Simulaciones de N4 finalizadas. Generando figuras preliminares de N4..."
            "$PYTHON_BIN" scripts/plot_initialization_benchmark.py \
                "$OUT_ROOT" \
                --system N4 \
                --out_dir "$FIG_DIR" \
                --format "$FIGURE_FORMAT" \
                --dpi "$FIGURE_DPI" \
                --font_scale "$FONT_SCALE" \
                --line_width_scale "$LINE_WIDTH_SCALE" \
                2>&1 | tee -a "$MASTER_LOG" || true

            run_benchmark_for_system "N8_w0.1_g1.0" "$NPZ_N8" \
                "$N8_ELEC" "$N8_UP" "$N8_DN" "$N8_OMEGA" "$N8_G" "$N8_SOFT" \
                "$N8_MALA_TAU" "$N8_RL_TAU" "$N8_RL_AMP" "$N8_RL_PEN" "$N8_RL_SIG" \
                "$N8_SIGMA_ENV" "$N8_BOX_HALF"
            ;;
    esac

    # Esperar finalización de todos los trabajos en cola
    log_msg "[COLA] Esperando a que todos los trabajos terminen de ejecutarse..."
    wait_all_slots
    log_msg "[COLA] Todas las simulaciones han finalizado."
fi

# ==============================================================================
# SECCIÓN 5: POST-PROCESAMIENTO Y COMPILACIÓN DE FIGURAS CON BARRAS DE ERROR
# ==============================================================================

log_msg "=============================================================================="
log_msg ">>> COMPILANDO FIGURAS DE ARTÍCULO Y TABLAS ESTADÍSTICAS CON BARRAS DE ERROR..."
log_msg "=============================================================================="

"$PYTHON_BIN" scripts/plot_initialization_benchmark.py \
    "$OUT_ROOT" \
    --out_dir "$FIG_DIR" \
    --format "$FIGURE_FORMAT" \
    --dpi "$FIGURE_DPI" \
    --font_scale "$FONT_SCALE" \
    --line_width_scale "$LINE_WIDTH_SCALE" \
    2>&1 | tee -a "$MASTER_LOG"

log_msg "=============================================================================="
log_msg "[OK] BENCHMARK DE INICIALIZACIONES COMPLETADO SATISFACTORIAMENTE."
log_msg "     Directorio de Resultados : ${OUT_ROOT}"
log_msg "     Figuras de Publicación  : ${FIG_DIR}"
log_msg "     Tabla Resumen CSV       : ${FIG_DIR}/initialization_benchmark_summary_table.csv"
log_msg "     Reporte Markdown        : ${FIG_DIR}/INITIALIZATION_BENCHMARK_REPORT.md"
log_msg "=============================================================================="
