#!/usr/bin/env bash
# ==============================================================================
# RL-VMC 1D: ORQUESTADOR DE MUESTREO PURO CASSCF Y CRUCE NODAL (AYLER & BAKER)
# ==============================================================================
# Propósito:
#   Evaluar y comparar el muestreo puro de funciones de onda CASSCF congeladas
#   entre el algoritmo MALA clásico y samplers neuronales RL-MALA (TD3 con
#   arquitectura DeepSet), enfocándose en el cruce activo de barreras nodales
#   y de repulsión de Wigner (sin entrenar el ansatz variacional).
#
# Características de Diseño:
#   1. NUEVA REGIÓN DE HIPERPARÁMETROS COMPLEMENTARIA (AYLER 3 GPUs, ~10h):
#      Basada en los hallazgos de N=4 (donde el óptimo se ubicó cerca de
#      tau=0.10, A in [0.10, 0.15], lambda=0.25 y MALA seguía bajando en tau=0.12),
#      explora una grilla más amplia y fina sin repetir ninguna corrida de Baker:
#      - MALA tau       : (0.06 0.14 0.16 0.20 0.25) -> localiza el quiebre exacto de MALA
#      - RL-MALA tau    : (0.10 0.12 0.14)           -> explora pasos base mayores
#      - RL-MALA amp (A): (0.06 0.08 0.12 0.18)      -> refina alrededor del sweet spot
#      - RL-MALA penalty: (0.20 0.30 0.40)           -> refina y extiende lambda >= 0.20
#      - RL-MALA sigma  : (1.15)                     -> expansión adaptativa de varianza
#      Total: 5 MALA + 36 RL-MALA = 41 modelos/sistema (82 modelos entre N=4 y N=6).
#      (Se puede volver a la grilla original de Baker con `--preset baker`).
#   2. MUESTREO PURO DE CASSCF CONGELADO (`--no-casscf_trainable`):
#      - Fase 1 (MALA): `--train_mode eval_only` sobre el estado CASSCF exacto.
#      - Fase 2 (RL-MALA): `--train_mode rl_only` (adapta únicamente el Actor TD3
#        sobre el CASSCF congelado) seguido de evaluación MCMC larga.
#      - No realiza entrenamiento VMC del ansatz en ningún punto.
#   3. LOGS Y COMPILACIÓN DE FIGURAS EN TIEMPO REAL:
#      - Los logs se escriben en tiempo real (`python3 -u`, `PYTHONUNBUFFERED=1`).
#      - Cada vez que converge un modelo o termina una fase/sistema, se compilan
#        en tiempo real (en CPU, sin robar VRAM) las 8 figuras estándar del sistema,
#        las figuras de cruce nodal (incluyendo trazas de 1 solo walker del Mejor MALA
#        vs Mejor RL-MALA) y el resumen global de todos los modelos/sistemas.
#   4. ESTILO VISUAL Y FORMATO DE ARTÍCULO CONFIGURABLE:
#      - Líneas discontinuas (`--`) para MALA y líneas continuas (`-`) para RL-MALA.
#      - Guardado por defecto en `.png` (ahorro de espacio), con banderas para
#        cambiar a `.pdf`, ajustar `--ppi`/`--dpi`, títulos (`--title-prefix`,
#        `--custom-suptitle`, `--no-titles`) y escalas de fuente/línea.
#   5. PROTECCIÓN ESTRICTA DE MEMORIA (VRAM + RAM HOST) EN AYLER Y BAKER:
#      - `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_ALLOCATOR=platform`.
#      - En Ayler (3 GPUs): 1 trabajo por GPU (`MAX_CONCURRENT_JOBS=3`),
#        `XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`, `ELOC_CHUNK_SIZE=2048`.
#      - Todo el post-procesamiento corre con `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`
#        y libera los caminantes `.h5` tras cada modelo (`gc.collect()`).
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# SECCIÓN 1: PARÁMETROS MODULARES Y DESCRIPTIVOS (MODIFICABLES)
# ==============================================================================

POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
SYSTEM="${SYSTEM:-all}"
GRID_PRESET="${GRID_PRESET:-ayler}"

# Formato y estilo de figuras (por defecto solo PNG para ahorrar espacio)
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TITLE_PREFIX="${TITLE_PREFIX:-}"
CUSTOM_SUPTITLE="${CUSTOM_SUPTITLE:-}"
NO_TITLES="${NO_TITLES:-0}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"
TRANSPARENT_BG="${TRANSPARENT_BG:-0}"

# Filtrado Top-K y trazas de 1 solo caminante (Mejor MALA vs Mejor RL-MALA)
TOP_K_MODELS="${TOP_K_MODELS:-2}"
WALKER_TRACE_TOP_K="${WALKER_TRACE_TOP_K:-1}"
N_EVAL_SUBBLOCKS="${N_EVAL_SUBBLOCKS:-5}"
REALTIME_PLOT_UPDATES="${REALTIME_PLOT_UPDATES:-1}"

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --postprocess-only|--postprocess_only|--only-postprocess|-p)
            POSTPROCESS_ONLY=1
            shift
            ;;
        --system|-s)
            SYSTEM="$2"
            shift 2
            ;;
        --preset|--grid-preset)
            GRID_PRESET="$2"
            shift 2
            ;;
        --top-k|--top_k_models|-k)
            TOP_K_MODELS="$2"
            shift 2
            ;;
        --walker-top-k|--walker_trace_top_k)
            WALKER_TRACE_TOP_K="$2"
            shift 2
            ;;
        --figure-format|--figure_format|--format|-f)
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
        --dpi|--ppi)
            FIGURE_DPI="$2"
            shift 2
            ;;
        --title-prefix|--title_prefix)
            TITLE_PREFIX="$2"
            shift 2
            ;;
        --custom-suptitle|--custom_suptitle)
            CUSTOM_SUPTITLE="$2"
            shift 2
            ;;
        --no-titles|--no_titles|--hide-titles)
            NO_TITLES=1
            shift
            ;;
        --font-scale|--font_scale)
            FONT_SCALE="$2"
            shift 2
            ;;
        --line-width-scale|--line_width_scale)
            LINE_WIDTH_SCALE="$2"
            shift 2
            ;;
        --transparent-bg|--transparent_bg)
            TRANSPARENT_BG=1
            shift
            ;;
        --no-realtime-plots)
            REALTIME_PLOT_UPDATES=0
            shift
            ;;
        --help|-h)
            echo "Uso: $0 [OPCIONES] [CARPETA_SALIDA]"
            echo ""
            echo "Opciones principales:"
            echo "  --preset [ayler|baker]        Selecciona la región de búsqueda de hiperparámetros:"
            echo "                                  ayler: Nueva región complementaria (41 modelos/sistema, ~10h en 3 GPUs, default)"
            echo "                                  baker: Región compacta original (16 modelos/sistema para 1 GPU)"
            echo "  --system, -s [N4|N6|all]      Sistema físico a evaluar (N4, N6, o all; default: all)"
            echo "  --postprocess-only, -p        Solo ejecuta el post-procesamiento y gráficas sobre resultados existentes"
            echo "  --top-k, -k K                 Número de mejores MALA vs mejores RL-MALA en curvas comparativas (default: 2)"
            echo "  --walker-top-k K              Número de mejores modelos por clase en trazas de 1 walker (default: 1)"
            echo ""
            echo "Opciones de formato de figuras (para espacio o publicación):"
            echo "  --figure-format, -f [png|pdf|both]  Formato de salida de gráficas (default: png)"
            echo "  --pdf                         Atajo para --figure-format pdf"
            echo "  --ppi, --dpi N                Resolución en puntos por pulgada (default: 300)"
            echo "  --title-prefix TEXTO          Prefijo para los títulos de todas las figuras"
            echo "  --custom-suptitle TEXTO       Título global personalizado"
            echo "  --no-titles                   Oculta títulos superiores (ideal para insertar en LaTeX)"
            echo "  --font-scale ESCALA           Factor de escala tipográfica (default: 1.0)"
            echo "  --line-width-scale ESCALA     Factor de grosor de línea (default: 1.0)"
            echo "  --transparent-bg              Fondo transparente al exportar figuras"
            echo "  [CARPETA_SALIDA]              Directorio raíz de salida (default: results_nodal_crossing_search)"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    OUT_ROOT="${POSITIONAL_ARGS[0]}"
else
    OUT_ROOT="${OUT_ROOT:-results_nodal_crossing_search}"
fi
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # 1 = saltar corridas ya terminadas; 0 = recalcular todo

# Construir banderas comunes de estilo de figuras para todos los scripts de post-procesamiento
PLOT_STYLE_ARGS=(
    --figure_format "$FIGURE_FORMAT"
    --dpi "$FIGURE_DPI"
    --font_scale "$FONT_SCALE"
    --line_width_scale "$LINE_WIDTH_SCALE"
)
if [[ -n "$TITLE_PREFIX" ]]; then
    PLOT_STYLE_ARGS+=(--title_prefix "$TITLE_PREFIX")
fi
if [[ -n "$CUSTOM_SUPTITLE" ]]; then
    PLOT_STYLE_ARGS+=(--custom_suptitle "$CUSTOM_SUPTITLE")
fi
if [[ "$NO_TITLES" -eq 1 || "$NO_TITLES" == "true" ]]; then
    PLOT_STYLE_ARGS+=(--no_titles)
fi
if [[ "$TRANSPARENT_BG" -eq 1 || "$TRANSPARENT_BG" == "true" ]]; then
    PLOT_STYLE_ARGS+=(--transparent_bg)
fi

# Configurar sistemas activos según $SYSTEM
SYSTEM_LOWER="$(echo "$SYSTEM" | tr '[:upper:]' '[:lower:]')"
case "$SYSTEM_LOWER" in
    n4|wigner_n4|wigner)
        RUN_SYSTEM_WIGNER_N4_G4=1
        RUN_SYSTEM_MULTIDET_N6_G1=0
        ;;
    n6|multidet_n6|multidet|b256)
        RUN_SYSTEM_WIGNER_N4_G4=0
        RUN_SYSTEM_MULTIDET_N6_G1=1
        ;;
    all)
        RUN_SYSTEM_WIGNER_N4_G4="${RUN_SYSTEM_WIGNER_N4_G4:-1}"
        RUN_SYSTEM_MULTIDET_N6_G1="${RUN_SYSTEM_MULTIDET_N6_G1:-1}"
        ;;
    *)
        echo "[AVISO] Sistema desconocido '$SYSTEM'. Opciones válidas: N4, N6, all. Usando 'all'."
        RUN_SYSTEM_WIGNER_N4_G4="${RUN_SYSTEM_WIGNER_N4_G4:-1}"
        RUN_SYSTEM_MULTIDET_N6_G1="${RUN_SYSTEM_MULTIDET_N6_G1:-1}"
        ;;
esac

# --- 1.2 Selección de Archivos CASSCF de Entrada ---
# Sistema A: Molécula de Wigner (N=4, g=4.0, omega=0.1, softening=0.1)
NPZ_WIGNER_N4_G4="${NPZ_WIGNER_N4_G4:-uhf_1d_ho_v270826_m-determinants/results_N4/g_4.0/cas_n4_g4_omega0.1_soft0.1_b32_c0-a4/casscf_state.npz}"

# Sistema B: CASSCF Multideterminante N=6 (N=6, g=1.0, omega=0.1, softening=0.1, 20 det, base=256)
NPZ_MULTIDET_N6_G1="${NPZ_MULTIDET_N6_G1:-figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz}"
if [[ ! -f "$NPZ_MULTIDET_N6_G1" ]]; then
    if [[ -f "figures_allmodels_benchmarks/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" ]]; then
        NPZ_MULTIDET_N6_G1="figures_allmodels_benchmarks/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
    elif [[ -f "${SCRIPT_DIR}/figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz" ]]; then
        NPZ_MULTIDET_N6_G1="${SCRIPT_DIR}/figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz"
    fi
fi

# --- 1.3 Grillas de Búsqueda de Hiperparámetros (MALA vs RL-MALA TD3) ---
# Selección por preset:
#   - 'ayler' (default): Nueva región complementaria (0 traslape con Baker),
#     diseñada a partir de los resultados de N=4 para 3 GPUs (~9-10 horas total).
#   - 'baker': Región original ejecutada en Baker (1 GPU).
PRESET_LOWER="$(echo "$GRID_PRESET" | tr '[:upper:]' '[:lower:]')"
if [[ "$PRESET_LOWER" == "baker" ]]; then
    MALA_TAU_GRID=(${MALA_TAU_GRID_STR:-0.04 0.08 0.10 0.12})
    TD3_TAU_GRID=(${TD3_TAU_GRID_STR:-0.08 0.10})
    ACTOR_AMP_GRID=(${ACTOR_AMP_GRID_STR:-0.10 0.15 0.25})
    PENALTY_GRID=(${PENALTY_GRID_STR:-0.15 0.25})
    SIGMA_SCALE_GRID=(${SIGMA_SCALE_GRID_STR:-1.1})
else
    # Región complementaria para Ayler (5 MALA + 36 RL-MALA = 41 modelos/sistema)
    MALA_TAU_GRID=(${MALA_TAU_GRID_STR:-0.06 0.14 0.16 0.20 0.25})
    TD3_TAU_GRID=(${TD3_TAU_GRID_STR:-0.10 0.12 0.14})
    ACTOR_AMP_GRID=(${ACTOR_AMP_GRID_STR:-0.06 0.08 0.12 0.18})
    PENALTY_GRID=(${PENALTY_GRID_STR:-0.20 0.30 0.40})
    SIGMA_SCALE_GRID=(${SIGMA_SCALE_GRID_STR:-1.15})
fi

# --- 1.4 Hiperparámetros de Adaptación del Sampler RL (TD3 sobre CASSCF congelado) ---
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-6000}"
N_WALKERS="${N_WALKERS:-128}"
N_BURN_IN_TRAIN="${N_BURN_IN_TRAIN:-2000}"
RL_LR_ACTOR="${RL_LR_ACTOR:-1e-5}"
RL_LR_CRITIC="${RL_LR_CRITIC:-3e-5}"
RL_USE_LR_SCHEDULER="${RL_USE_LR_SCHEDULER:-false}"
MAX_PHYS_JUMP="${MAX_PHYS_JUMP:-2.0}"
RL_REPLAY_CAPACITY="${RL_REPLAY_CAPACITY:-50000}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"

# --- 1.5 Configuración de Evaluación de Muestreo Puro y Cruce de Barreras Nodales ---
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"
EVAL_BURN_IN="${EVAL_BURN_IN:-200}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
N_EVAL_CHAINS="${N_EVAL_CHAINS:-3}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-2000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-128}"

SEED="${SEED:-42}"

# ==============================================================================
# SECCIÓN 2: PROTECCIÓN DE MEMORIA Y AUTO-DETECCIÓN DE GPUs (AYLER / BAKER)
# ==============================================================================

# Limitar hilos de CPU por proceso para evitar agotamiento de RAM en Ayler
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

# Desactivar pre-asignación masiva de VRAM en JAX/XLA (crítico para no colgar Ayler)
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
        # Modo BAKER (1 sola GPU)
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
        # Modo AYLER (3 GPUs): 1 trabajo por GPU por defecto (3 procesos simultáneos seguros)
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
GLOBAL_FIG_DIR="${FIG_DIR}/all_models_global"
MASTER_CSV="${CSV_SUMMARY_DIR}/all_models_master.csv"
MASTER_LOG="${LOG_DIR}/master_search.log"
PLOT_LOCK_DIR="${LOG_DIR}/.realtime_plot.lock"

mkdir -p "$LOG_DIR" "$CSV_SUMMARY_DIR" "$FIG_DIR" "$GLOBAL_FIG_DIR"
rm -rf "$PLOT_LOCK_DIR" 2>/dev/null || true
touch "$MASTER_LOG"

log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "[$ts] $*" | tee -a "$MASTER_LOG"
}

log_msg "=============================================================================="
log_msg "INICIANDO BÚSQUEDA DE PARÁMETROS DE MUESTREO PURO CASSCF (AYLER / BAKER)"
log_msg "=============================================================================="
log_msg "Directorio raíz        : ${OUT_ROOT}"
log_msg "Preset de grilla       : ${GRID_PRESET}"
log_msg "GPUs físicas detectadas: ${NUM_GPUS} (${AVAILABLE_GPUS[*]:-CPU}) | VRAM por GPU: ${TOTAL_VRAM_MB} MB"
log_msg "Trabajos por GPU       : ${JOBS_PER_GPU} | Concurrencia total: ${MAX_CONCURRENT_JOBS}"
log_msg "XLA_MEM_FRACTION       : ${XLA_PYTHON_CLIENT_MEM_FRACTION} (ELOC_CHUNK=${ELOC_CHUNK_SIZE})"
log_msg "Muestreo RL-MALA TD3   : rl_only sobre CASSCF congelado (${RL_TRAIN_STEPS} sweeps, N_walkers=${N_WALKERS})"
log_msg "Evaluación MCMC        : init=${EVAL_INIT_MODE}, chains=${N_EVAL_CHAINS}, burn_in=${EVAL_BURN_IN}, steps=${MEASURE_STEPS}"
log_msg "Formato de figuras     : ${FIGURE_FORMAT} (${FIGURE_DPI} PPI) | Top-K=${TOP_K_MODELS} | WalkerTrace Top-K=${WALKER_TRACE_TOP_K}"
log_msg "Grilla MALA (tau)      : ${MALA_TAU_GRID[*]} (${#MALA_TAU_GRID[@]} modelos)"
log_msg "Grilla TD3 (tau)       : ${TD3_TAU_GRID[*]}"
log_msg "Grilla TD3 (actor_amp) : ${ACTOR_AMP_GRID[*]}"
log_msg "Grilla TD3 (penalty)   : ${PENALTY_GRID[*]}"
log_msg "Grilla TD3 (sigma_sc)  : ${SIGMA_SCALE_GRID[*]}"
log_msg "=============================================================================="

# ==============================================================================
# SECCIÓN 3: COMPILACIÓN DE FIGURAS EN TIEMPO REAL Y COLA DE TRABAJOS GPU
# ==============================================================================

# Compila/actualiza las figuras globales de todos los sistemas/modelos en CPU
update_global_all_models_figures() {
    local nodal_csv="${CSV_SUMMARY_DIR}/master_nodal_crossing_comparison.csv"
    local param_csv="${CSV_SUMMARY_DIR}/master_param_search_summary.csv"
    local target_csv=""
    if [[ -f "$nodal_csv" ]]; then
        target_csv="$nodal_csv"
    elif [[ -f "$param_csv" ]]; then
        target_csv="$param_csv"
    elif [[ -f "$MASTER_CSV" ]]; then
        target_csv="$MASTER_CSV"
    else
        local first_nodal
        first_nodal=$(ls "${CSV_SUMMARY_DIR}"/nodal_crossing_summary_*.csv 2>/dev/null | head -n 1 || true)
        if [[ -n "$first_nodal" && -f "$first_nodal" ]]; then
            target_csv="$first_nodal"
        fi
    fi

    if [[ -n "$target_csv" ]]; then
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            python3 -u scripts/plot_param_search_results.py \
            --csv "$target_csv" \
            --out_dir "$GLOBAL_FIG_DIR" \
            "${PLOT_STYLE_ARGS[@]}" \
            >> "${LOG_DIR}/realtime_global_plots.log" 2>&1 || true
    fi
}

# Compila/actualiza las 8 figuras estándar + cruce nodal de un sistema en CPU
compile_system_figures() {
    local sys_tag="$1"
    local npz_path="$2"
    local mode="${3:-full}"  # 'incremental' o 'full'
    local sys_out="${OUT_ROOT}/${sys_tag}"
    local sys_fig="${FIG_DIR}/${sys_tag}"
    mkdir -p "$sys_fig"

    local h5_files=()
    local h5_labels=()

    for d in "${sys_out}"/*; do
        if [[ -d "$d" && -f "${d}/run_bundle.h5" && -f "${d}/summary.json" ]]; then
            h5_files+=("${d}/run_bundle.h5")
            local bname
            bname="$(basename "$d")"
            h5_labels+=("${bname#${sys_tag}_}")
        fi
    done

    if [[ "${#h5_files[@]}" -eq 0 ]]; then
        for d in "${OUT_ROOT}"/*; do
            if [[ -d "$d" && -f "${d}/run_bundle.h5" && -f "${d}/summary.json" ]]; then
                local bname
                bname="$(basename "$d")"
                h5_files+=("${d}/run_bundle.h5")
                h5_labels+=("${bname#${sys_tag}_}")
            fi
        done
    fi

    if [[ "${#h5_files[@]}" -gt 0 ]]; then
        if [[ "$mode" == "full" ]]; then
            log_msg "[${sys_tag} - POSTPROCESS] Compilando las 8 figuras estándar y métricas (${#h5_files[@]} modelos)..."
        fi
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            python3 -u cli/postprocess.py \
            --inputs "${h5_files[@]}" \
            --labels "${h5_labels[@]}" \
            --casscf_npz_path "$npz_path" \
            --out_dir "${sys_fig}/postprocess_standard" \
            --csv_collect_dir "$CSV_SUMMARY_DIR" \
            --system_tag "$sys_tag" \
            --top_k_models "$TOP_K_MODELS" \
            --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
            --n_eval_subblocks "$N_EVAL_SUBBLOCKS" \
            "${PLOT_STYLE_ARGS[@]}" \
            >> "${LOG_DIR}/postprocess_${sys_tag}.log" 2>&1 || log_msg "[AVISO] postprocess reportó advertencias en ${sys_tag}"

        if [[ "$mode" == "full" ]]; then
            # En modo final del sistema, evaluar cruce nodal completo desde .h5 (con trazas de 1 walker)
            log_msg "[${sys_tag} - NODAL CROSSING] Compilando figuras de cruce nodal y trazas de 1 walker (Top-${WALKER_TRACE_TOP_K})..."
            CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
                python3 -u cli/evaluate_nodal_crossing_1d.py \
                --inputs "${h5_files[@]}" \
                --labels "${h5_labels[@]}" \
                --casscf_npz_path "$npz_path" \
                --out_dir "$sys_fig" \
                --csv_collect_dir "$CSV_SUMMARY_DIR" \
                --system_tag "$sys_tag" \
                --top_k_models "$TOP_K_MODELS" \
                --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
                --n_eval_subblocks "$N_EVAL_SUBBLOCKS" \
                "${PLOT_STYLE_ARGS[@]}" \
                >> "${LOG_DIR}/nodal_crossing_eval_${sys_tag}.log" 2>&1 || log_msg "[AVISO] evaluate_nodal_crossing_1d reportó advertencias en ${sys_tag}"
        else
            # En modo incremental en vivo, usar --from_postprocess_dir (instantáneo, <2s en CPU)
            CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
                python3 -u cli/evaluate_nodal_crossing_1d.py \
                --from_postprocess_dir "${sys_fig}/postprocess_standard" \
                --casscf_npz_path "$npz_path" \
                --out_dir "$sys_fig" \
                --csv_collect_dir "$CSV_SUMMARY_DIR" \
                --system_tag "$sys_tag" \
                --top_k_models "$TOP_K_MODELS" \
                --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
                --n_eval_subblocks "$N_EVAL_SUBBLOCKS" \
                "${PLOT_STYLE_ARGS[@]}" \
                >> "${LOG_DIR}/nodal_crossing_eval_${sys_tag}.log" 2>&1 || true
        fi

        # Actualizar figuras globales de todos los modelos/sistemas
        update_global_all_models_figures
    elif [[ -d "${sys_fig}/postprocess_standard" ]]; then
        log_msg "[${sys_tag} - MODO CSV] Reconstruyendo figuras desde ${sys_fig}/postprocess_standard..."
        CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
            python3 -u cli/evaluate_nodal_crossing_1d.py \
            --from_postprocess_dir "${sys_fig}/postprocess_standard" \
            --casscf_npz_path "$npz_path" \
            --out_dir "$sys_fig" \
            --csv_collect_dir "$CSV_SUMMARY_DIR" \
            --system_tag "$sys_tag" \
            --top_k_models "$TOP_K_MODELS" \
            --walker_trace_top_k "$WALKER_TRACE_TOP_K" \
            --n_eval_subblocks "$N_EVAL_SUBBLOCKS" \
            "${PLOT_STYLE_ARGS[@]}" \
            >> "${LOG_DIR}/nodal_crossing_eval_${sys_tag}.log" 2>&1 || true
        update_global_all_models_figures
    else
        log_msg "[AVISO] No se encontraron archivos run_bundle.h5 ni CSVs en ${sys_out} / ${sys_fig}."
    fi
}

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
    local sys_tag="$1"
    local npz_path="$2"
    local run_name="$3"
    local run_dir="$4"
    shift 4
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
        local exit_code=$?

        if [[ "$exit_code" -eq 0 && -f "${run_dir}/run_bundle.h5" ]]; then
            # 1. Post-procesamiento individual del modelo recién terminado (en CPU)
            CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
                python3 -u cli/postprocess.py \
                --inputs "${run_dir}/run_bundle.h5" \
                --labels "${run_name#${sys_tag}_}" \
                --casscf_npz_path "$npz_path" \
                --out_dir "${run_dir}/postprocess" \
                --system_tag "$run_name" \
                --top_k_models 1 \
                --walker_trace_top_k 1 \
                "${PLOT_STYLE_ARGS[@]}" \
                >> "$log_file" 2>&1 || true

            # 2. Compilación incremental en tiempo real de las gráficas del sistema y globales
            #    (protegida con mkdir lock atómico para no duplicar procesos CPU si 2 GPUs terminan al mismo segundo)
            if [[ "$REALTIME_PLOT_UPDATES" -eq 1 ]]; then
                if mkdir "$PLOT_LOCK_DIR" 2>/dev/null; then
                    compile_system_figures "$sys_tag" "$npz_path" "incremental"
                    rmdir "$PLOT_LOCK_DIR" 2>/dev/null || true
                fi
            fi
        fi
        exit "$exit_code"
    ) &
    SLOT_PIDS[$slot]=$!
    SLOT_NAMES[$slot]="$run_name"
}

# ==============================================================================
# SECCIÓN 4: EJECUCIÓN DEL PIPELINE DE MUESTREO PURO POR SISTEMA FÍSICO
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
        if [[ -f "${SCRIPT_DIR}/${npz_path}" ]]; then
            npz_path="${SCRIPT_DIR}/${npz_path}"
        else
            log_msg "[AVISO] No se encontró el archivo CASSCF '${npz_path}' para ${sys_tag}. Saltando sistema."
            return 0
        fi
    fi

    local sys_out="${OUT_ROOT}/${sys_tag}"
    local sys_fig="${FIG_DIR}/${sys_tag}"
    mkdir -p "$sys_out" "$sys_fig"

    log_msg "------------------------------------------------------------------------------"
    log_msg "SISTEMA: ${sys_tag} (N=${n_elec}, up=${n_up}, dn=${n_dn}, g=${g_val}, omega=${omega})"
    log_msg "Archivo CASSCF (congelado): ${npz_path}"
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

    if [[ "$POSTPROCESS_ONLY" -eq 1 || "$POSTPROCESS_ONLY" == "true" ]]; then
        log_msg "[MODO POSTPROCESS-ONLY] Omitiendo simulación para ${sys_tag}. Compilando figuras directamente..."
    else
        # --------------------------------------------------------------------------
        # FASE 1: BARRIDO DE BASELINES MALA CLÁSICO (MUESTREO PURO DE CASSCF)
        # --------------------------------------------------------------------------
        log_msg "[${sys_tag} - FASE 1] Evaluando muestreo puro MALA Clásico sobre CASSCF congelado..."
        for tau in "${MALA_TAU_GRID[@]}"; do
            local rname="${sys_tag}_MALA_tau${tau}"
            local rdir="${sys_out}/${rname}"
            launch_job "$sys_tag" "$npz_path" "$rname" "$rdir" \
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
        # FASE 2: ADAPTACIÓN DEL SAMPLER RL-MALA (TD3) Y EVALUACIÓN DE MUESTREO PURO
        # (El ansatz CASSCF permanece 100% congelado con `--no-casscf_trainable`).
        # --------------------------------------------------------------------------
        log_msg "[${sys_tag} - FASE 2] Adaptando sampler RL-MALA (--train_mode rl_only) + Evaluación de Cruce Nodal..."
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
                        launch_job "$sys_tag" "$npz_path" "$rname" "$rdir" \
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

        # Esperar a que terminen todos los modelos de este sistema
        wait_all_slots
    fi

    # --------------------------------------------------------------------------
    # FASE 3: COMPILACIÓN COMPLETA DE LAS 8 FIGURAS DEL SISTEMA + CRUCE NODAL
    #         + ACTUALIZACIÓN DE LAS FIGURAS GLOBALES DE TODOS LOS MODELOS
    # --------------------------------------------------------------------------
    rm -rf "$PLOT_LOCK_DIR" 2>/dev/null || true
    log_msg "[${sys_tag} - FASE 3] Compilando post-producción completa del sistema y actualizando gráficos globales..."
    compile_system_figures "$sys_tag" "$npz_path" "full"
    log_msg "[OK] Post-producción de ${sys_tag} completada en: ${sys_fig} y ${GLOBAL_FIG_DIR}"
}

# Ejecutar los sistemas habilitados (con actualización de figuras al terminar cada sistema)
if [[ "$RUN_SYSTEM_WIGNER_N4_G4" -eq 1 ]]; then
    run_system_pipeline "Wigner_N4_g4.0" "$NPZ_WIGNER_N4_G4" 4 2 2 0.1 4.0 0.1
fi

if [[ "$RUN_SYSTEM_MULTIDET_N6_G1" -eq 1 ]]; then
    run_system_pipeline "MultiDet_N6_g1.0" "$NPZ_MULTIDET_N6_G1" 6 3 3 0.1 1.0 0.1
fi

# Compilación final global de todos los modelos
update_global_all_models_figures

log_msg "=============================================================================="
log_msg "PIPELINE DE MUESTREO PURO Y CRUCE NODAL COMPLETADO EXITOSAMENTE"
log_msg "  -> Tablas CSV concentradas en      : ${CSV_SUMMARY_DIR}"
log_msg "  -> Figuras por sistema en          : ${FIG_DIR}"
log_msg "  -> Figuras globales de modelos en  : ${GLOBAL_FIG_DIR}"
log_msg "  -> Bitácora maestra en             : ${MASTER_LOG}"
log_msg "=============================================================================="
