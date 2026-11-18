#!/usr/bin/env bash
# ==============================================================================
# BENCHMARK MODERNO DE ESTRATEGIAS DE INICIALIZACIÓN: MALA vs RL-MALA (CASSCF CONVERGIDOS)
# ==============================================================================
# Propósito:
#   Evaluar sistemáticamente la sensibilidad, resiliencia y aceleración de
#   muestreadores MALA clásico y neuronales RL-MALA (TD3 DeepSet) frente a
#   estrategias de inicialización espacial sobre estados CASSCF CONVERGIDOS
#   de espacio activo completo (CASSCF_csv/, b=256, c0-aN, converged=True):
#
#   1) alternating_spin   : "Un electrón en cada Gaussiana" (red alternada con jitter).
#   2) randomized_wigner  : Estándar moderno (red Wigner centrada con permutación aleatoria
#                           por caminante, introducida para evitar trampas nodales en N>=8).
#   3) symmetric_gaussian : "Gaussiana centrada en x=0 cubriendo la envolvente" N(0, sigma^2).
#   4) uniform            : "Uniforme en la caja [-L, L]" sin sesgo armónico inicial.
#
# Referencias CASSCF Convergidas (verificadas por metadata en CASSCF_csv/):
#   - N=4, g=1.0, omega=0.1, b=256, c0-a4 (converged=True, E = 1.94862194 Ha)
#     NPZ: CASSCF_csv/casscf_state_cas_n4_g1.0_omega0.1_soft0.1_b256_c0-a4_det6.npz
#   - N=6, g=1.0, omega=0.1, b=256, c0-a6 det-20 (converged=True, E = 4.52405213 Ha) [opcional]
#     NPZ: CASSCF_csv/casscf_state_cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_det-20.npz
#   - N=8, g=1.0, omega=0.1, b=256, c0-a8 det-70 (converged=True, E = 8.07323817 Ha)
#     NPZ: CASSCF_csv/casscf_state_cas_n8_g1.0_omega0.1_soft0.1_b256_c0-a8_det-70.npz
#
# Mejoras metodológicas incorporadas respecto a la campaña v1 (Nov 5-11):
#   * Estados CASSCF verificados (converged=True, n_core=0, b=256, g=1.0 exacto).
#   * Piso numérico exacto (tiny_eps + harm_floor_log) y factorización de determinantes
#     únicos por espín (en HEAD de rl_vmc/ansatze/ansatz_1d.py).
#   * Escalamiento dimensional de sigma_action_scale: 1.10 (N<=6) y 0.90 (N=8).
#   * Tasas RL calibradas sin decaimiento coseno (--no-use_lr_scheduler --lr_actor 1e-4 --lr_critic 3e-4).
#   * Bandera --max_phys_jump_factor 2.0, burn-in 2000, train_steps 6000, eloc_chunk_size 64.
#   * Banderas XLA estables (Triton GEMM desactivado), cuarentena automática de .h5
#     previos no convergidos, y circuit-breaker ante fallos instantáneos.
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN PREDETERMINADA Y PARÁMETROS MODIFICABLES
# ==============================================================================

OUT_ROOT="${OUT_ROOT:-results_init_benchmark_converged}"
SYSTEM="${SYSTEM:-N4 N8}"         # Por defecto N4 y N8 convergidos a g=1.0 (admite N6, all, all3)
INIT_MODE_FILTER="${INIT_MODE_FILTER:-all}"
INCLUDE_RANDOMIZED_WIGNER="${INCLUDE_RANDOMIZED_WIGNER:-1}"

N_EXP="${N_EXP:-5}"                # Réplicas independientes con distinta seed
N_EVAL="${N_EVAL:-5}"              # Cadenas de evaluación por corrida (barras de error SEM)
BASE_SEED="${BASE_SEED:-42}"

# Hiperparámetros MCMC y de Producción (calibrados para estabilidad y buena estadística)
MEASURE_STEPS="${MEASURE_STEPS:-30000}"
EVAL_BURN_IN="${EVAL_BURN_IN:-2000}"
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-6000}"
N_BURN_IN_TRAIN="${N_BURN_IN_TRAIN:-2000}"
N_WALKERS="${N_WALKERS:-128}"
JIT_CHUNK_SIZE="${JIT_CHUNK_SIZE:-1000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-64}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-2.0}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"

# Formato y estilo de figuras
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"

POSTPROCESS_ONLY=0
SKIP_EXISTING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system|-s)
            SYSTEM="$2"
            shift 2
            ;;
        --include-n6|--with-n6)
            SYSTEM="N4 N6 N8"
            shift
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
        --classic-3-inits)
            INCLUDE_RANDOMIZED_WIGNER=0
            shift
            ;;
        --with-randomized-wigner)
            INCLUDE_RANDOMIZED_WIGNER=1
            shift
            ;;
        --walkers|-w)
            N_WALKERS="$2"
            shift 2
            ;;
        --measure-steps|-m)
            MEASURE_STEPS="$2"
            shift 2
            ;;
        --eval-burn-in)
            EVAL_BURN_IN="$2"
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
        --both-formats)
            FIGURE_FORMAT="both"
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
        --force-recalculate|--force-rerun)
            SKIP_EXISTING=0
            shift
            ;;
        --help|-h)
            echo "Uso: $0 [OPCIONES] [CARPETA_SALIDA]"
            echo ""
            echo "Opciones principales:"
            echo "  --system, -s [N4|N6|N8|'N4 N8'|all|all3]  Sistema(s) CASSCF a evaluar (default: 'N4 N8')"
            echo "  --include-n6 / --with-n6                    Evalúa N4, N6 y N8 (todos a g=1.0)"
            echo "  --init-mode [all|alternating_spin|randomized_wigner|symmetric_gaussian|uniform]"
            echo "  --classic-3-inits                           Solo las 3 inicializaciones originales"
            echo "  --n-exp, -e N                               Réplicas independientes con distinta seed (default: 5)"
            echo "  --n-eval, -c N                              Cadenas MCMC de evaluación por réplica (default: 5)"
            echo "  --measure-steps, -m N                       Pasos MCMC de producción por cadena (default: 30000)"
            echo "  --eval-burn-in N                            Pasos de burn-in en evaluación (default: 2000)"
            echo "  --rl-steps, -r N                            Barridos TD3 para RL-MALA (default: 6000)"
            echo "  --walkers, -w N                             Número de caminantes (default: 128)"
            echo "  --gpu [ID|'ID1 ID2'|auto]                   Asignación de GPU(s) (default: auto)"
            echo "  --jobs-per-gpu, -j N                        Trabajos paralelos por GPU (default: 1)"
            echo "  --format, -f [png|pdf|both]                 Formato gráfico (default: png)"
            echo "  --dpi N                                     Resolución DPI (default: 300)"
            echo "  --postprocess-only, -p                      Solo recompila figuras y tablas"
            echo "  --force-recalculate / --force-rerun         Re-ejecuta corridas aunque existan"
            echo "  [CARPETA_SALIDA]                            Directorio de salida (default: results_init_benchmark_converged)"
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
CSV_COLLECT_DIR="${OUT_ROOT}/results_csv"
QUARANTINE_DIR="${OUT_ROOT}/quarantine_unconverged"
MASTER_CSV="${OUT_ROOT}/master_initialization_benchmark.csv"
MASTER_LOG="${LOG_DIR}/master_benchmark.log"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$FIG_DIR" "$CSV_COLLECT_DIR"

log_msg() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $1" | tee -a "$MASTER_LOG"
}

# ==============================================================================
# SECCIÓN 2: ENTORNO CUDA / XLA ESTABLE Y AUTO-DETECCIÓN DE GPUs
# ==============================================================================

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_MEM_FRACTION:-0.35}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export JAX_ENABLE_X64=True
export XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || echo "python3")}"

if [ "$POSTPROCESS_ONLY" -eq 0 ]; then
    if ! "$PYTHON_BIN" -c "import jax, h5py" >/dev/null 2>&1; then
        log_msg "[ERROR FATAL] '$PYTHON_BIN' no puede importar jax/h5py. Activa tu entorno conda (ej. conda activate jax-gpu)."
        exit 1
    fi
fi

# Detección dinámica de GPUs
GPU_ARG="${GPU_ARG:-auto}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

if [[ "$GPU_ARG" != "auto" && -n "$GPU_ARG" ]]; then
    GPU_ARG="${GPU_ARG//,/ }"
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
if [[ "${GPUS[0]}" == "-1" ]]; then
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
declare -a SLOT_START_TS
for (( s=0; s<MAX_CONCURRENT_JOBS; s++ )); do
    SLOT_PIDS[$s]=0
    SLOT_NAMES[$s]=""
    SLOT_RUN_DIRS[$s]=""
    SLOT_START_TS[$s]=0
done

FREE_SLOT=-1
CONSECUTIVE_FAST_FAILS=0

verify_casscf_npz_converged() {
    local npz_path="$1"
    local exp_n="$2"
    local exp_g="$3"
    "$PYTHON_BIN" - "$npz_path" "$exp_n" "$exp_g" << 'PYEOF'
import sys, json, zipfile
p, exp_n, exp_g = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
with zipfile.ZipFile(p, "r") as z:
    raw = z.read("metadata_json.npy")
    hlen = int.from_bytes(raw[8:10], "little")
    meta = json.loads(raw[10 + hlen:].decode("utf-32-le").rstrip("\x00"))
cfg = meta.get("config", {})
conv = bool(meta.get("converged", False))
n_el = int(cfg.get("n_up", 0)) + int(cfg.get("n_down", 0))
g_v = float(cfg.get("g", -1.0))
n_core = int(cfg.get("n_core", -1))
n_basis = int(cfg.get("n_basis", 0))
e_val = float(meta.get("selected_energy", meta.get("energy", float("nan"))))
grad = float(meta.get("diagnostics", {}).get("orbital_gradient_max", float("nan")))
ok = conv and (n_el == exp_n) and (abs(g_v - exp_g) < 1e-4) and (n_core == 0) and (n_basis >= 256)
print(f"    Verificado NPZ '{p}': converged={conv}, N={n_el}, g={g_v}, c{n_core}, b={n_basis}, |g_orb|={grad:.2e}, E={e_val:.8f} Ha")
if not ok:
    sys.stderr.write(f"[ERROR FATAL] NPZ '{p}' NO cumple requisitos de convergencia (esperado N={exp_n}, g={exp_g}, converged=True, c0, b>=256).\n")
    sys.exit(2)
PYEOF
}

is_h5_completed_with_converged_casscf() {
    local rdir="$1"
    local exp_n="$2"
    local exp_g="$3"
    local h5f="${rdir}/run_bundle.h5"
    if [[ "$SKIP_EXISTING" -eq 0 ]] || [[ ! -f "$h5f" ]]; then
        return 1
    fi
    local sz=0
    sz=$(wc -c < "$h5f" 2>/dev/null || echo 0)
    if [[ "$sz" -le 10000 ]]; then
        return 1
    fi
    "$PYTHON_BIN" - "$h5f" "$exp_n" "$exp_g" << 'PYEOF' >/dev/null 2>&1
import sys, json, h5py, os, zipfile
p, exp_n, exp_g = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
with h5py.File(p, "r") as f:
    if "production" not in f or "walkers" not in f["production"]:
        sys.exit(1)
    if f["production/walkers"].shape[0] < 1:
        sys.exit(1)
    if "config_json" not in f.attrs:
        sys.exit(2)
    cfg = json.loads(str(f.attrs["config_json"]))
    npz_path = (cfg.get("ansatz") or {}).get("casscf_npz_path") or ""
    if not npz_path or not os.path.exists(npz_path):
        sys.exit(2)
    with zipfile.ZipFile(npz_path, "r") as z:
        raw = z.read("metadata_json.npy")
        hlen = int.from_bytes(raw[8:10], "little")
        meta = json.loads(raw[10 + hlen:].decode("utf-32-le").rstrip("\x00"))
    m_cfg = meta.get("config", {})
    ok = (
        bool(meta.get("converged", False))
        and int(m_cfg.get("n_basis", 0)) >= 256
        and int(m_cfg.get("n_core", -1)) == 0
        and (int(m_cfg.get("n_up", 0)) + int(m_cfg.get("n_down", 0))) == exp_n
        and abs(float(m_cfg.get("g", -1.0)) - exp_g) < 1e-4
    )
    sys.exit(0 if ok else 2)
PYEOF
    local rc=$?
    if [[ "$rc" -eq 2 ]]; then
        local ts_q
        ts_q="$(date '+%Y%m%d_%H%M%S')"
        mkdir -p "$QUARANTINE_DIR"
        log_msg "[QUARANTINE] Corrida previa inválida/no-convergida en ${rdir} movida a ${QUARANTINE_DIR}/$(basename "$rdir")_${ts_q}"
        mv "$rdir" "${QUARANTINE_DIR}/$(basename "$rdir")_${ts_q}"
        return 1
    fi
    return "$rc"
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
                wait "$pid"
                local wrc=$?
                local prev_run="${SLOT_NAMES[$s]}"
                local dt=$(( $(date +%s) - SLOT_START_TS[$s] ))
                if [[ "$wrc" -ne 0 && "$dt" -lt 5 ]]; then
                    CONSECUTIVE_FAST_FAILS=$((CONSECUTIVE_FAST_FAILS + 1))
                    if [[ "$CONSECUTIVE_FAST_FAILS" -ge 3 ]]; then
                        log_msg "[CIRCUIT BREAKER] 3 fallos instantáneos consecutivos (<5s). Abortando para no desperdiciar cola."
                        exit 1
                    fi
                elif [[ "$wrc" -eq 0 ]]; then
                    CONSECUTIVE_FAST_FAILS=0
                fi
                log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]:-CPU}') liberado por '${prev_run}' (${dt}s, rc=${wrc})"
                SLOT_PIDS[$s]=0
                SLOT_NAMES[$s]=""
                SLOT_RUN_DIRS[$s]=""
                SLOT_START_TS[$s]=0
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
            local dt=$(( $(date +%s) - SLOT_START_TS[$s] ))
            log_msg "[FIN] Slot #${s} (GPU='${SLOT_GPUS[$s]:-CPU}') liberado por '${SLOT_NAMES[$s]}' (${dt}s)"
            SLOT_PIDS[$s]=0
            SLOT_NAMES[$s]=""
            SLOT_RUN_DIRS[$s]=""
            SLOT_START_TS[$s]=0
        fi
    done
}

launch_job() {
    local exp_n="$1"
    local exp_g="$2"
    local run_name="$3"
    local run_dir="$4"
    shift 4
    local cmd=("$@")

    if is_h5_completed_with_converged_casscf "$run_dir" "$exp_n" "$exp_g"; then
        log_msg "[SKIP] '${run_name}' ya completado con CASSCF convergido en ${run_dir}."
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
        exit "$exit_code"
    ) &

    SLOT_PIDS[$slot]=$!
    SLOT_NAMES[$slot]="$run_name"
    SLOT_RUN_DIRS[$slot]="$run_dir"
    SLOT_START_TS[$slot]=$(date +%s)
    sleep 1
}

# ==============================================================================
# SECCIÓN 3: REFERENCIAS CASSCF CONVERGIDAS Y PARÁMETROS CALIBRADOS
# ==============================================================================

NPZ_N4="${NPZ_N4:-CASSCF_csv/casscf_state_cas_n4_g1.0_omega0.1_soft0.1_b256_c0-a4_det6.npz}"
if [[ ! -f "$NPZ_N4" && -f "figures/casscf-state_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6.npz" ]]; then
    NPZ_N4="figures/casscf-state_n4_g1_omega0.1_soft0.1_b256_c0-a4_det6.npz"
fi
NPZ_N6="${NPZ_N6:-CASSCF_csv/casscf_state_cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_det-20.npz}"
NPZ_N8="${NPZ_N8:-CASSCF_csv/casscf_state_cas_n8_g1.0_omega0.1_soft0.1_b256_c0-a8_det-70.npz}"

# N=4, g=1.0
N4_ELEC=4;  N4_UP=2; N4_DN=2; N4_OMEGA=0.1; N4_G=1.0; N4_SOFT=0.1
N4_MALA_TAU=0.10; N4_RL_TAU=0.10; N4_RL_AMP=0.14; N4_RL_PEN=0.25; N4_RL_SIG=1.10
N4_SIGMA_ENV=6.0; N4_BOX_HALF=10.0

# N=6, g=1.0 (referencia intermedia multi-determinante CAS(6e,6o), det-20)
N6_ELEC=6;  N6_UP=3; N6_DN=3; N6_OMEGA=0.1; N6_G=1.0; N6_SOFT=0.1
N6_MALA_TAU=0.10; N6_RL_TAU=0.10; N6_RL_AMP=0.14; N6_RL_PEN=0.25; N6_RL_SIG=1.10
N6_SIGMA_ENV=7.5; N6_BOX_HALF=13.0

# N=8, g=1.0 (escala calibrada sig=0.90 en 8D según run_advisor_casscf_pipeline.sh)
N8_ELEC=8;  N8_UP=4; N8_DN=4; N8_OMEGA=0.1; N8_G=1.0; N8_SOFT=0.1
N8_MALA_TAU=0.08; N8_RL_TAU=0.08; N8_RL_AMP=0.10; N8_RL_PEN=0.25; N8_RL_SIG=0.90
N8_SIGMA_ENV=9.0; N8_BOX_HALF=16.0

if [[ "$INCLUDE_RANDOMIZED_WIGNER" -eq 1 ]]; then
    ALL_INITS=("alternating_spin" "randomized_wigner" "symmetric_gaussian" "uniform")
else
    ALL_INITS=("alternating_spin" "symmetric_gaussian" "uniform")
fi

ACTIVE_INITS=()
if [[ "$INIT_MODE_FILTER" == "all" ]]; then
    ACTIVE_INITS=("${ALL_INITS[@]}")
else
    read -r -a ACTIVE_INITS <<< "${INIT_MODE_FILTER//,/ }"
fi

log_msg "=============================================================================="
log_msg "    BENCHMARK DE INICIALIZACIONES SOBRE ESTADOS CASSCF CONVERGIDOS (c0-aN, b=256)"
log_msg "=============================================================================="
log_msg "Directorio de Salida   : ${OUT_ROOT}"
log_msg "Sistema(s) Objetivo    : ${SYSTEM}"
log_msg "Inicializaciones       : ${ACTIVE_INITS[*]}"
log_msg "Réplicas / Cadenas     : ${N_EXP} seeds x ${N_EVAL} cadenas"
log_msg "Entrenamiento RL TD3   : ${RL_TRAIN_STEPS} pasos (burn-in=${N_BURN_IN_TRAIN}, const LR=${LR_ACTOR}/${LR_CRITIC})"
log_msg "Evaluación MCMC        : ${MEASURE_STEPS} pasos (burn-in=${EVAL_BURN_IN}, walkers=${N_WALKERS}, eloc_chunk=${ELOC_CHUNK_SIZE})"
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

    if [[ ! -f "$npz_path" ]]; then
        log_msg "[CASSCF] No se encontró archivo preconfigurado '$npz_path'. Buscando referencia convergida equivalente o calculando en CPU..."
        local resolved_npz=""
        resolved_npz=$("$PYTHON_BIN" - "$n_elec" "$g_val" "$omega" "$OUT_ROOT" << 'PYEOF'
import sys, json, zipfile
from pathlib import Path

exp_n, exp_g, exp_w, out_root = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])
candidates = []
for d in [
    Path("CASSCF_csv"),
    out_root / "casscf_refs",
    Path("figures"),
    *sorted(Path(".").glob("results_production_3pipelines*/pipeline1_casscf")),
]:
    if d.is_dir():
        candidates.extend(sorted(d.rglob("*.npz")))

for p in candidates:
    try:
        with zipfile.ZipFile(p, "r") as z:
            raw = z.read("metadata_json.npy")
            hlen = int.from_bytes(raw[8:10], "little")
            meta = json.loads(raw[10 + hlen:].decode("utf-32-le").rstrip("\x00"))
        c = meta.get("config", {})
        if (
            bool(meta.get("converged", False))
            and (int(c.get("n_up", 0)) + int(c.get("n_down", 0))) == exp_n
            and abs(float(c.get("g", -1.0)) - exp_g) < 1e-4
            and abs(float(c.get("omega", -1.0)) - exp_w) < 1e-4
            and int(c.get("n_core", -1)) == 0
            and int(c.get("n_basis", 0)) >= 256
        ):
            print(str(p))
            sys.exit(0)
    except Exception:
        continue
PYEOF
        )
        if [[ -n "$resolved_npz" && -f "$resolved_npz" ]]; then
            npz_path="$resolved_npz"
            log_msg "[CASSCF] Referencia convergida localizada automáticamente: $npz_path"
        else
            local cas_ref_dir="${OUT_ROOT}/casscf_refs/${sys_tag}"
            local cas_log="${LOG_DIR}/p1_casscf_${sys_tag}.log"
            local n_sel_dets=0
            [[ "$n_elec" -eq 4 ]] && n_sel_dets=6
            [[ "$n_elec" -eq 6 ]] && n_sel_dets=20
            [[ "$n_elec" -eq 8 ]] && n_sel_dets=70
            log_msg "[CASSCF] Calculando referencia convergida en CPU (b=256, c0-a${n_elec}, quad=800) -> ${cas_ref_dir}"
            env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu "$PYTHON_BIN" -u cli/run_casscf_1d.py \
                --n_elec "$n_elec" --n_up "$n_up" --n_dn "$n_dn" \
                --omega "$omega" --g "$g_val" --softening "$soft" \
                --n_core 0 --n_active_orbitals "$n_elec" \
                --n_basis 256 --quadrature_order 800 \
                --casscf_max_iter 2000 --casscf_energy_tol 1e-9 --casscf_grad_tol 1e-5 \
                --n_final_determinants "$n_sel_dets" \
                --outdir "$cas_ref_dir" --reuse_existing > "$cas_log" 2>&1
            npz_path="${cas_ref_dir}/casscf_state.npz"
        fi
    fi

    if [[ ! -f "$npz_path" ]]; then
        log_msg "[ERROR FATAL] No existe el archivo CASSCF para ${sys_tag}: $npz_path"
        exit 1
    fi
    verify_casscf_npz_converged "$npz_path" "$n_elec" "$g_val" | tee -a "$MASTER_LOG" || exit 1

    local sys_out="${OUT_ROOT}/${sys_tag}"
    mkdir -p "$sys_out"

    log_msg "------------------------------------------------------------------------------"
    log_msg ">>> PROCESANDO SISTEMA: ${sys_tag} (N=${n_elec}, g=${g_val}, omega=${omega})"
    log_msg "    Referencia CASSCF       : ${npz_path}"
    log_msg "    MALA tau                : ${mala_tau}"
    log_msg "    RL-MALA calibrado       : tau=${rl_tau}, A=${rl_amp}, lambda=${rl_pen}, sigma_scale=${rl_sig}"
    log_msg "    Envolvente Gauss / Box  : sigma=${sigma_env} Bohr | L=${box_half} Bohr"
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
            randomized_wigner)
                init_tag="rand_wigner"
                init_cli_args=(
                    --init_mode "randomized_wigner"
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
                --save_production_energy_mode "step_mean"
                --save_production_walkers_dtype "float32"
                --save_production_stats_dtype "float32"
                --no-export_legacy_files
                --reference_casscf_npz "$npz_path"
                --master_csv "$MASTER_CSV"
                --csv_collect_dir "$CSV_COLLECT_DIR"
                --seed "$cur_seed"
            )

            # 1. MALA Clásico (eval_only sobre CASSCF congelado)
            local mala_rname="${sys_tag}_MALA_${init_tag}_exp${exp}"
            local mala_rdir="${sys_out}/${mala_rname}"
            launch_job "$n_elec" "$g_val" "$mala_rname" "$mala_rdir" \
                "$PYTHON_BIN" -u cli/run_1d.py \
                "${common_args[@]}" \
                --proposal_type mala \
                --train_mode eval_only \
                --ansatz_type casscf_npz \
                --casscf_npz_path "$npz_path" \
                --no-casscf_trainable \
                --tau "$mala_tau" \
                --pipeline_stage "init_benchmark_mala" \
                --run_name "$mala_rname" \
                --output_h5 "${mala_rdir}/run_bundle.h5"

            # 2. RL-MALA (TD3 DeepSet, calibración moderna sobre CASSCF congelado)
            local rl_rname="${sys_tag}_RL_MALA_${init_tag}_exp${exp}"
            local rl_rdir="${sys_out}/${rl_rname}"
            launch_job "$n_elec" "$g_val" "$rl_rname" "$rl_rdir" \
                "$PYTHON_BIN" -u cli/run_1d.py \
                "${common_args[@]}" \
                --proposal_type neural_deepset \
                --proposal_mode residual_mala \
                --rl_algorithm td3 \
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
                --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
                --no-use_lr_scheduler \
                --lr_actor "$LR_ACTOR" \
                --lr_critic "$LR_CRITIC" \
                --use_actor_in_eval \
                --pipeline_stage "init_benchmark_rl_mala" \
                --run_name "$rl_rname" \
                --output_h5 "${rl_rdir}/run_bundle.h5"
        done
    done
}

if [[ "$POSTPROCESS_ONLY" -eq 1 ]]; then
    log_msg "[MODO POSTPROCESS-ONLY] Omitiendo simulaciones. Compilando figuras y estadísticas..."
else
    SYS_NORM="$(echo "$SYSTEM" | tr '[:upper:],' '[:lower:] ')"
    RUN_LIST=()
    for tok in $SYS_NORM; do
        case "$tok" in
            n4)   RUN_LIST+=("N4") ;;
            n6)   RUN_LIST+=("N6") ;;
            n8)   RUN_LIST+=("N8") ;;
            all)  RUN_LIST+=("N4" "N8") ;;
            all3) RUN_LIST+=("N4" "N6" "N8") ;;
        esac
    done
    # Deduplicar preservando orden
    UNIQUE_SYS=()
    for s in "${RUN_LIST[@]}"; do
        skip=0
        for u in "${UNIQUE_SYS[@]:-}"; do [[ "$u" == "$s" ]] && skip=1; done
        [[ "$skip" -eq 0 ]] && UNIQUE_SYS+=("$s")
    done

    for idx_s in "${!UNIQUE_SYS[@]}"; do
        sys_code="${UNIQUE_SYS[$idx_s]}"
        case "$sys_code" in
            N4)
                run_benchmark_for_system "N4_w0.1_g1.0" "$NPZ_N4" \
                    "$N4_ELEC" "$N4_UP" "$N4_DN" "$N4_OMEGA" "$N4_G" "$N4_SOFT" \
                    "$N4_MALA_TAU" "$N4_RL_TAU" "$N4_RL_AMP" "$N4_RL_PEN" "$N4_RL_SIG" \
                    "$N4_SIGMA_ENV" "$N4_BOX_HALF"
                ;;
            N6)
                run_benchmark_for_system "N6_w0.1_g1.0" "$NPZ_N6" \
                    "$N6_ELEC" "$N6_UP" "$N6_DN" "$N6_OMEGA" "$N6_G" "$N6_SOFT" \
                    "$N6_MALA_TAU" "$N6_RL_TAU" "$N6_RL_AMP" "$N6_RL_PEN" "$N6_RL_SIG" \
                    "$N6_SIGMA_ENV" "$N6_BOX_HALF"
                ;;
            N8)
                run_benchmark_for_system "N8_w0.1_g1.0" "$NPZ_N8" \
                    "$N8_ELEC" "$N8_UP" "$N8_DN" "$N8_OMEGA" "$N8_G" "$N8_SOFT" \
                    "$N8_MALA_TAU" "$N8_RL_TAU" "$N8_RL_AMP" "$N8_RL_PEN" "$N8_RL_SIG" \
                    "$N8_SIGMA_ENV" "$N8_BOX_HALF"
                ;;
        esac

        # Al concluir N4 o N6, esperar y graficar preliminarmente antes del sistema mayor
        if [[ "$idx_s" -lt $(( ${#UNIQUE_SYS[@]} - 1 )) ]]; then
            log_msg "[COLA] Esperando finalización de ${sys_code} para graficar en vivo..."
            wait_all_slots
            "$PYTHON_BIN" scripts/plot_initialization_benchmark.py \
                "$OUT_ROOT" \
                --system "$sys_code" \
                --out_dir "$FIG_DIR" \
                --format "$FIGURE_FORMAT" \
                --dpi "$FIGURE_DPI" \
                --font_scale "$FONT_SCALE" \
                --line_width_scale "$LINE_WIDTH_SCALE" \
                2>&1 | tee -a "$MASTER_LOG" || true
        fi
    done

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
log_msg "     CSVs Recolectados       : ${CSV_COLLECT_DIR}"
log_msg "     Tabla Resumen CSV       : ${FIG_DIR}/initialization_benchmark_summary_table.csv"
log_msg "     Reporte Markdown        : ${FIG_DIR}/INITIALIZATION_BENCHMARK_REPORT.md"
log_msg "=============================================================================="
