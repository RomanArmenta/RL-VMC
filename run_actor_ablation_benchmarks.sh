#!/usr/bin/env bash
# ==============================================================================
# ACTOR ABLATION BENCHMARKS (B1 & B2) — CONTROL JUSTO VS MALA GRANDE Y
# DESBLOQUEO DE SENSIBILIDAD LOCAL + EQUIVARIANZA DE PARIDAD EN RL-MALA
# ==============================================================================
# Ejecuta de forma reproducible (en Ayler multi-GPU o en CPU local) los dos
# benchmarks diagnósticos sobre funciones de onda CASSCF congeladas (`CASSCF_csv/`):
#
#   BENCHMARK 1 (Control Justo vs MALA Grande y RWMH de Ancho Emparejado):
#     1. B1_MALA_base_tau0.10         : MALA estándar (tau=0.10, sigma=0.447 Bohr)
#     2. B1_MALA_large_tau0.50        : MALA grande   (tau=0.50, sigma=1.000 Bohr)
#     3. B1_MALA_matched_tau0.7389    : MALA emparejado en sigma (tau=0.7389, sigma=1.216 Bohr)
#     4. B1_RWMH_matched_sig1.2156    : Random-Walk MH sin deriva cuántica (sigma=1.216 Bohr)
#     5. B1_RL_MALA_legacy            : RL-MALA legado (--no-parity_equivariant --no-unclip_sigma_action)
#
#   BENCHMARK 2 (Equivarianza Exacta de Paridad + Desbloqueo de sigma + Amplitud Física):
#     1. B1_RL_MALA_legacy            : Referencia legada (reutilizada de B1 si existe)
#     2. B2_RL_MALA_parity_only       : Solo --parity_equivariant (mismo amp=0.15, clip legado=1.0)
#     3. B2_RL_MALA_eq_unclip_amp0.80_sig1.8 : --parity_equivariant --unclip_sigma_action amp=0.80 sig=1.8
#     4. B2_RL_MALA_eq_unclip_amp1.20_sig2.0 : --parity_equivariant --unclip_sigma_action amp=1.20 sig=2.0
#
# Uso en Ayler (3 GPUs, producción completa N=4 g=2.0 y N=6 g=1.0,2.0):
#   chmod +x run_actor_ablation_benchmarks.sh
#   nohup ./run_actor_ablation_benchmarks.sh --gpus "0 1 2" > actor_ablation_ayler.log 2>&1 &
#
# Uso selectivo (solo Benchmark 1 o solo Benchmark 2):
#   ./run_actor_ablation_benchmarks.sh --benchmark 1 --systems "4:2.0 6:1.0" --gpus "0 1 2"
#   ./run_actor_ablation_benchmarks.sh --benchmark 2 --systems "4:2.0 6:1.0" --gpus "0 1 2"
#
# Prueba rápida local en CPU:
#   ./run_actor_ablation_benchmarks.sh --cpu --systems "4:2.0" --batch-size 32 \
#       --rl-steps 400 --measure-steps 3000 --eval-chains 2 --eval-burn-in 300 --n-burn-in 300
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=1
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

# Valores por defecto (calibrados para Ayler)
BENCHMARK_MODE="${BENCHMARK_MODE:-all}"          # 1 | 2 | all
SYSTEMS_LIST="${SYSTEMS_LIST:-4:2.0 6:1.0 6:2.0}"
OMEGA="${OMEGA:-0.1}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results_actor_ablation_benchmarks}"
CASSCF_DIR="${CASSCF_DIR:-${SCRIPT_DIR}/CASSCF_csv}"
GPU_LIST="${GPU_LIST:-auto}"
USE_CPU="${USE_CPU:-0}"
CPU_WORKERS="${CPU_WORKERS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"

# Hiperparámetros de entrenamiento y evaluación
BATCH_SIZE="${BATCH_SIZE:-128}"
N_BURN_IN="${N_BURN_IN:-2000}"
RL_STEPS="${RL_STEPS:-6000}"
EVAL_CHAINS="${EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-2000}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
INIT_MODE="${INIT_MODE:-randomized_wigner}"
INIT_STD="${INIT_STD:-3.0}"
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"
REWARD_TYPE="${REWARD_TYPE:-cdlb}"
PENALTY_FACTOR="${PENALTY_FACTOR:-0.25}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-2.0}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Parseo de argumentos CLI
while [[ $# -gt 0 ]]; do
    case "$1" in
        --benchmark) BENCHMARK_MODE="$2"; shift 2 ;;
        --systems) SYSTEMS_LIST="$2"; shift 2 ;;
        --omega) OMEGA="$2"; shift 2 ;;
        --out-dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --casscf-dir) CASSCF_DIR="$2"; shift 2 ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --cpu) USE_CPU=1; shift ;;
        --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --force) FORCE_RERUN=1; shift ;;
        --postprocess-only) POSTPROCESS_ONLY=1; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --n-burn-in) N_BURN_IN="$2"; shift 2 ;;
        --rl-steps) RL_STEPS="$2"; shift 2 ;;
        --eval-chains) EVAL_CHAINS="$2"; shift 2 ;;
        --eval-burn-in) EVAL_BURN_IN="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --penalty) PENALTY_FACTOR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,35p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Opción desconocida: $1" >&2
            exit 1
            ;;
    esac
done

# Detectar intérprete Python con JAX/Flax
if [[ -z "$PYTHON_BIN" ]]; then
    for cand in \
        "$(command -v python3 2>/dev/null || true)" \
        "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12" \
        "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "$cand" ]] && "$cand" -c "import jax, flax, h5py" >/dev/null 2>&1; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "[ERROR] No se encontró un Python con jax, flax y h5py. Pasa --python /ruta/a/python3." >&2
    exit 1
fi

# Detectar GPUs disponibles si no se forzó CPU
declare -a GPUS=()
if [[ "$USE_CPU" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    export JAX_PLATFORMS="cpu"
    for ((wi=0; wi<CPU_WORKERS; wi++)); do GPUS+=(""); done
else
    if [[ "$GPU_LIST" == "auto" ]]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            N_DET="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
            if [[ "${N_DET:-0}" -gt 0 ]]; then
                for ((gi=0; gi<N_DET; gi++)); do
                    for ((ji=0; ji<JOBS_PER_GPU; ji++)); do GPUS+=("$gi"); done
                done
            else
                USE_CPU=1
                export CUDA_VISIBLE_DEVICES=""
                for ((wi=0; wi<CPU_WORKERS; wi++)); do GPUS+=(""); done
            fi
        else
            USE_CPU=1
            export CUDA_VISIBLE_DEVICES=""
            for ((wi=0; wi<CPU_WORKERS; wi++)); do GPUS+=(""); done
        fi
    else
        for g_tok in $GPU_LIST; do
            for ((ji=0; ji<JOBS_PER_GPU; ji++)); do GPUS+=("$g_tok"); done
        done
    fi
fi

mkdir -p "$BASE_OUT_DIR"

echo "=============================================================================="
echo "  RL-VMC 1D — BENCHMARKS DE ABLACIÓN DEL ACTOR (B1 & B2)"
echo "=============================================================================="
echo "  Python          : $PYTHON_BIN"
echo "  Benchmark mode  : $BENCHMARK_MODE"
echo "  Sistemas (N:g)  : $SYSTEMS_LIST (omega=$OMEGA)"
echo "  Directorio out  : $BASE_OUT_DIR"
echo "  Modo hardware   : $( [[ "$USE_CPU" == "1" ]] && echo "CPU" || echo "GPUs [${GPUS[*]}]" )"
echo "  Train / Eval    : batch=$BATCH_SIZE, rl_steps=$RL_STEPS, eval=${EVAL_CHAINS}x${MEASURE_STEPS}"
echo "=============================================================================="

# Helper: localizar el archivo CASSCF .npz del asesor para (N, g)
find_casscf_npz() {
    local n_val="$1"
    local g_val="$2"
    local g_int="${g_val%.0}"
    local found=""
    for pat in \
        "casscf_state_cas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
        "casscf_statecas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
        "casscf_state_cas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz \
        "casscf_statecas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz; do
        for f in "${CASSCF_DIR}"/$pat; do
            if [[ -f "$f" ]]; then
                found="$f"
                break 2
            fi
        done
    done
    echo "$found"
}

# Helper: verificar si un bundle HDF5 ya está completo con production/walkers
is_bundle_complete() {
    local h5_file="$1"
    [[ -f "$h5_file" ]] || return 1
    "$PYTHON_BIN" -c "
import h5py, sys
try:
    with h5py.File(sys.argv[1], 'r') as f:
        ok = ('production' in f) and ('walkers' in f['production']) and (f['production']['walkers'].size > 0)
        sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
" "$h5_file" >/dev/null 2>&1
}

# Cola de trabajos con ejecución paralela por GPU (o secuencial en CPU)
declare -a JOB_PIDS=()
declare -a JOB_NAMES=()
JOB_IDX=0

wait_for_slot() {
    local max_jobs="${#GPUS[@]}"
    while [[ "${#JOB_PIDS[@]}" -ge "$max_jobs" ]]; do
        local next_pids=()
        local next_names=()
        for i in "${!JOB_PIDS[@]}"; do
            if kill -0 "${JOB_PIDS[$i]}" 2>/dev/null; then
                next_pids+=("${JOB_PIDS[$i]}")
                next_names+=("${JOB_NAMES[$i]}")
            else
                wait "${JOB_PIDS[$i]}" || echo "  [AVISO] Trabajo ${JOB_NAMES[$i]} terminó con código distinto de 0."
            fi
        done
        JOB_PIDS=(${next_pids[@]+"${next_pids[@]}"})
        JOB_NAMES=(${next_names[@]+"${next_names[@]}"})
        if [[ "${#JOB_PIDS[@]}" -ge "$max_jobs" ]]; then
            sleep 2
        fi
    done
}

wait_all_jobs() {
    if [[ "${#JOB_PIDS[@]}" -gt 0 ]]; then
        for i in "${!JOB_PIDS[@]}"; do
            wait "${JOB_PIDS[$i]}" || echo "  [AVISO] Trabajo ${JOB_NAMES[$i]} terminó con error."
        done
    fi
    JOB_PIDS=()
    JOB_NAMES=()
}

launch_run() {
    local run_dir="$1"
    local run_name="$2"
    shift 2
    local h5_out="${run_dir}/run_bundle.h5"
    local log_out="${run_dir}/train.log"

    if [[ "$POSTPROCESS_ONLY" == "1" ]]; then
        return 0
    fi
    if [[ "$FORCE_RERUN" != "1" ]] && is_bundle_complete "$h5_out"; then
        echo "  [SKIP] Ya existe bundle íntegro: $h5_out"
        return 0
    fi

    mkdir -p "$run_dir"
    wait_for_slot
    local gpu_id="${GPUS[$((JOB_IDX % ${#GPUS[@]}))]}"
    JOB_IDX=$((JOB_IDX + 1))

    if [[ "$USE_CPU" == "1" ]]; then
        echo "  [RUN][CPU] $run_name -> $h5_out"
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/cli/run_1d.py" \
            --output_h5 "$h5_out" \
            --run_name "$run_name" \
            "$@" > "$log_out" 2>&1 &
    else
        echo "  [RUN][GPU $gpu_id] $run_name -> $h5_out"
        CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "${SCRIPT_DIR}/cli/run_1d.py" \
            --output_h5 "$h5_out" \
            --run_name "$run_name" \
            "$@" > "$log_out" 2>&1 &
    fi
    JOB_PIDS+=("$!")
    JOB_NAMES+=("$run_name")
}

# ==============================================================================
# EJECUCIÓN POR SISTEMA (N, g)
# ==============================================================================
for sys_pair in $SYSTEMS_LIST; do
    N_ELEC="${sys_pair%%:*}"
    G_VAL="${sys_pair##*:}"
    N_UP=$((N_ELEC / 2))
    N_DN=$((N_ELEC - N_UP))
    SYS_TAG="N${N_ELEC}_w${OMEGA}_g${G_VAL}"
    SYS_DIR="${BASE_OUT_DIR}/${SYS_TAG}"
    mkdir -p "$SYS_DIR"

    NPZ_PATH="$(find_casscf_npz "$N_ELEC" "$G_VAL")"
    if [[ -z "$NPZ_PATH" ]]; then
        echo "[ERROR] No se encontró archivo CASSCF .npz en $CASSCF_DIR para N=$N_ELEC, g=$G_VAL, omega=$OMEGA." >&2
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo "  SISTEMA: ${SYS_TAG} | CASSCF: $(basename "$NPZ_PATH")"
    echo "------------------------------------------------------------------------------"

    COMMON_ARGS=(
        --n_elec "$N_ELEC" --n_up "$N_UP" --n_dn "$N_DN"
        --omega "$OMEGA" --interaction_strength "$G_VAL" --softening 0.1
        --ansatz_type casscf_npz --casscf_npz_path "$NPZ_PATH" --no-casscf_trainable
        --reference_casscf_npz "$NPZ_PATH"
        --batch_size "$BATCH_SIZE"
        --n_burn_in "$N_BURN_IN"
        --init_mode "$INIT_MODE" --init_std "$INIT_STD"
        --eval_init_mode "$EVAL_INIT_MODE"
        --n_eval_chains "$EVAL_CHAINS"
        --eval_burn_in "$EVAL_BURN_IN"
        --measure_steps "$MEASURE_STEPS"
        --eloc_chunk_size "$ELOC_CHUNK_SIZE"
        --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
        --save_production_energy_mode step_mean
        --save_production_walkers_dtype float32
        --save_production_stats_dtype float32
        --seed "$SEED"
    )

    RL_COMMON_ARGS=(
        --proposal_type neural_deepset
        --proposal_mode residual_mala
        --train_mode rl_only
        --rl_algorithm td3
        --train_steps "$RL_STEPS"
        --reward_type "$REWARD_TYPE"
        --penalty_factor "$PENALTY_FACTOR"
        --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR"
        --lr_actor "$LR_ACTOR" --lr_critic "$LR_CRITIC"
        --no-use_lr_scheduler
        --use_reward_centering
        --tau 0.10
    )

    # --------------------------------------------------------------------------
    # BENCHMARK 1: Control Justo vs MALA Grande y RWMH de Ancho Emparejado
    # --------------------------------------------------------------------------
    if [[ "$BENCHMARK_MODE" == "1" || "$BENCHMARK_MODE" == "all" ]]; then
        echo "  >>> Lanzando Benchmark 1 (Control Justo vs MALA Grande / RWMH)..."

        # 1.1 MALA base (tau=0.10, sigma=0.4472 Bohr)
        launch_run "${SYS_DIR}/B1_MALA_base_tau0.10" "${SYS_TAG}_B1_MALA_base_tau0.10" \
            "${COMMON_ARGS[@]}" \
            --proposal_type mala --train_mode eval_only --tau 0.10

        # 1.2 MALA grande (tau=0.50, sigma=1.0000 Bohr)
        launch_run "${SYS_DIR}/B1_MALA_large_tau0.50" "${SYS_TAG}_B1_MALA_large_tau0.50" \
            "${COMMON_ARGS[@]}" \
            --proposal_type mala --train_mode eval_only --tau 0.50

        # 1.3 MALA emparejado exactamente al techo sigma de RL-MALA legado (tau=0.10*exp(2)=0.7389056, sigma=1.21565 Bohr)
        launch_run "${SYS_DIR}/B1_MALA_matched_tau0.7389" "${SYS_TAG}_B1_MALA_matched_tau0.7389" \
            "${COMMON_ARGS[@]}" \
            --proposal_type mala --train_mode eval_only --tau 0.7389056

        # 1.4 Random-Walk Metropolis-Hastings (deriva = 0, sigma = 1.21565 Bohr emparejado)
        launch_run "${SYS_DIR}/B1_RWMH_matched_sig1.2156" "${SYS_TAG}_B1_RWMH_matched_sig1.2156" \
            "${COMMON_ARGS[@]}" \
            --proposal_type metropolis --train_mode eval_only --rw_sigma 1.21565

        # 1.5 RL-MALA legado (banderas OFF para reproducir exactamente el actor anterior)
        launch_run "${SYS_DIR}/B1_RL_MALA_legacy" "${SYS_TAG}_B1_RL_MALA_legacy" \
            "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
            --no-parity_equivariant --no-unclip_sigma_action \
            --actor_amplitude 0.15 --sigma_action_scale 1.1 --scale_clip_max 1.0
    fi

    # --------------------------------------------------------------------------
    # BENCHMARK 2: Desbloqueo de Sensibilidad Local + Equivarianza Exacta de Paridad
    # --------------------------------------------------------------------------
    if [[ "$BENCHMARK_MODE" == "2" || "$BENCHMARK_MODE" == "all" ]]; then
        echo "  >>> Lanzando Benchmark 2 (Equivarianza de Paridad + Sigma Desbloqueada + Amplitud Física)..."

        # 2.0 Asegurar existencia del control legado si se corrió --benchmark 2 en solitario
        if [[ "$BENCHMARK_MODE" == "2" ]]; then
            launch_run "${SYS_DIR}/B1_RL_MALA_legacy" "${SYS_TAG}_B1_RL_MALA_legacy" \
                "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
                --no-parity_equivariant --no-unclip_sigma_action \
                --actor_amplitude 0.15 --sigma_action_scale 1.1 --scale_clip_max 1.0
        fi

        # 2.1 Ablación 1-factor: SOLO equivarianza exacta de paridad (amp=0.15, techo clip=1.0 legado)
        launch_run "${SYS_DIR}/B2_RL_MALA_parity_only" "${SYS_TAG}_B2_RL_MALA_parity_only" \
            "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
            --parity_equivariant --no-unclip_sigma_action \
            --actor_amplitude 0.15 --sigma_action_scale 1.1 --scale_clip_max 1.0

        # 2.2 Actor completo desbloqueado (moderado): parity ON + unclip_sigma ON + cdlb_exact_entropy ON (sig in [0.27, 2.71] Bohr) + amp=0.80 Bohr
        launch_run "${SYS_DIR}/B2_RL_MALA_eq_unclip_amp0.80_sig1.8" "${SYS_TAG}_B2_RL_MALA_eq_unclip_amp0.80_sig1.8" \
            "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
            --parity_equivariant --unclip_sigma_action --cdlb_exact_entropy \
            --actor_amplitude 0.80 --sigma_action_scale 1.8 --scale_clip_min -0.5 --scale_clip_max 1.8

        # 2.3 Actor completo desbloqueado (barrera Wigner fuerte): parity ON + unclip_sigma ON + cdlb_exact_entropy ON (sig in [0.27, 3.30] Bohr) + amp=1.20 Bohr
        launch_run "${SYS_DIR}/B2_RL_MALA_eq_unclip_amp1.20_sig2.0" "${SYS_TAG}_B2_RL_MALA_eq_unclip_amp1.20_sig2.0" \
            "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
            --parity_equivariant --unclip_sigma_action --cdlb_exact_entropy \
            --actor_amplitude 1.20 --sigma_action_scale 2.0 --scale_clip_min -0.5 --scale_clip_max 2.0
    fi

    # Esperar a que terminen todos los runs de este sistema antes de graficar
    wait_all_jobs

    # --------------------------------------------------------------------------
    # POST-PROCESAMIENTO Y DIAGNÓSTICOS ESPACIALES AUTOMÁTICOS PARA ESTE SISTEMA
    # --------------------------------------------------------------------------
    echo "  >>> Ejecutando análisis comparativo y sondas espaciales para ${SYS_TAG}..."
    FIG_DIR="${SYS_DIR}/figures_ablation"
    mkdir -p "$FIG_DIR"

    # 1) Comparativa cuantitativa B1 + B2 + superposición espacial de actores
    if [[ -f "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" ]]; then
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" \
            --sys-dir "$SYS_DIR" \
            --casscf-npz "$NPZ_PATH" \
            --out-dir "$FIG_DIR" || echo "  [AVISO] Falló plot_actor_ablation_comparison.py para ${SYS_TAG}"
    fi

    # 1b) Comparativa de trazas de caminantes y marginales de espín por sampler
    if [[ -f "${SCRIPT_DIR}/scripts/plot_ablation_traces_and_marginals.py" ]]; then
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_ablation_traces_and_marginals.py" \
            --sys-dir "$SYS_DIR" \
            --casscf-npz "$NPZ_PATH" \
            --out-dir "$FIG_DIR" \
            --trace-steps 4000 || echo "  [AVISO] Falló plot_ablation_traces_and_marginals.py para ${SYS_TAG}"
    fi

    # 2) Sonda espacial detallada de cada actor neuronal entrenado
    for rl_h5 in "${SYS_DIR}"/B*_RL_MALA_*/run_bundle.h5; do
        if [[ -f "$rl_h5" ]]; then
            CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/probe_actor_spatial_response.py" \
                --h5 "$rl_h5" \
                --casscf_npz "$NPZ_PATH" \
                --probe_spin down \
                --out_dir "${FIG_DIR}/spatial_probes" \
                --figure_format both || true
            CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_walker_flip_proposal.py" \
                --h5 "$rl_h5" \
                --casscf_npz "$NPZ_PATH" \
                --spin_channel both \
                --out_dir "${FIG_DIR}/walker_flip_proposals" \
                --figure_format both || true
        fi
    done
done

echo ""
echo "=============================================================================="
echo "  BENCHMARKS DE ABLACIÓN COMPLETADOS -> ${BASE_OUT_DIR}"
echo "=============================================================================="
