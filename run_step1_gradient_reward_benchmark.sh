#!/usr/bin/env bash
# ==============================================================================
# BENCHMARK PASO 1 (HOJA DE RUTA RL-MCMC — CERO CÓDIGO NUEVO EN rl_vmc):
# GRADIENTE EXACTO REPARAMETRIZADO (REINFORCE) vs TD3, DESACOPLAMIENTO DE DERIVA
# (full_gaussian), RECOMPENSAS DE EFICIENCIA (log_esjd / kappa_jump vs cdlb) Y
# MODO DE ACTUALIZACIÓN (all_electrons vs single_electron) EN 3 SISTEMAS CASSCF
# ==============================================================================
# Usa exclusivamente las clases y algoritmos YA implementados en el paquete
# `rl_vmc` (`DeepSetNeuralProposal1D`, `FlaxTD3Actor`, `MHSampler1D`,
# `REINFORCELearner1D`, `TD3Learner1D`, `compute_rl_reward`, `CASSCF1DAnsatz`)
# a través de `cli/run_1d.py`, evaluados sobre 3 sistemas CASSCF convergidos
# (`CASSCF_csv/`, b=256, c0):
#
#   1. N=4, g=2.0 (36 dets,  E= 2.802268 Ha, d_adj=6.18 Bohr, v/p=0.171) — Puente directo con B1/B2
#   2. N=6, g=5.0 (20 dets,  E=11.272810 Ha, d_adj=6.72 Bohr, v/p=0.040) — Quiebre de RWMH all-elec (Acc ~ 2-3%)
#   3. N=8, g=5.0 (70 dets,  E=20.297512 Ha, d_adj=5.91 Bohr, v/p=0.052) — Muro exponencial N=8 (Acc_RWMH ~ 0.6%)
#
# Matriz controlada de 10 modelos por sistema (4 clásicos + 6 RL = 2 olas exactas en 3 GPUs):
#   Bloque A — Controles Clásicos y Modo de Actualización (eval_only):
#     1. S1_MALA_all_tau0.10      : MALA clásico all_electrons (tau=0.10, sigma=0.447 Bohr)
#     2. S1_RWMH_all_sig1.22      : RWMH campeón de B1/B2 (all_electrons, sigma=1.216 Bohr)
#     3. S1_RWMH_all_sig3.00      : RWMH all_electrons ancho grande (sigma=3.00 Bohr, mide maldición (sig_w/sig)^N)
#     4. S1_RWMH_1e_sig3.00       : RWMH single_electron (sigma=3.00 Bohr, rompe parcialmente la maldición en N)
#   Bloque B — Control Off-Policy TD3 (full_gaussian + parity_eq + unclip_sigma + force_scale=1.0):
#     5. S1_TD3_fullG_cdlb        : TD3 + full_gaussian + CDLB exacto (control directo sin deriva tau*F_Q)
#     6. S1_TD3_fullG_logESJD     : TD3 + full_gaussian + log_esjd (ablación 1-factor de recompensa dentro de TD3)
#   Bloque C — Gradiente Exacto Reparametrizado On-Policy (REINFORCELearner1D, cero dilución 1/N):
#     7. S1_RF_fullG_cdlb         : REINFORCE + full_gaussian + CDLB exacto (ablación 1-factor TD3 -> REINFORCE)
#     8. S1_RF_fullG_logESJD      : REINFORCE + full_gaussian + log_esjd (candidato principal de eficiencia local)
#     9. S1_RF_fullG_kappa        : REINFORCE + full_gaussian + kappa_jump (salto + regularización kappa*alpha*(1-alpha))
#    10. S1_RF_resMALA_logESJD    : REINFORCE + residual_mala + log_esjd (ablación 1-factor full_gaussian vs residual_mala)
#   Opcional (--include-barker):
#    11. S1_RF_fullG_barker       : REINFORCE + full_gaussian + barker_jump
#
# Uso en Ayler (3 GPUs, producción completa en los 3 sistemas CASSCF):
#   chmod +x run_step1_gradient_reward_benchmark.sh
#   nohup ./run_step1_gradient_reward_benchmark.sh --gpus "0 1 2" \
#       --out-dir results_step1_gradient_reward_ayler \
#       > step1_benchmark_ayler.log 2>&1 &
#
# Uso en prueba rápida local en CPU (validación extrema a extremo):
#   ./run_step1_gradient_reward_benchmark.sh --cpu --systems "4:2.0" \
#       --batch-size 32 --rf-steps 250 --td3-steps 400 --n-burn-in 200 \
#       --eval-chains 1 --eval-burn-in 300 --measure-steps 1500 \
#       --out-dir results_step1_local_smoke
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

# Valores por defecto calibrados para producción en Ayler (3 GPUs)
SYSTEMS_LIST="${SYSTEMS_LIST:-4:2.0 6:5.0 8:5.0}"
OMEGA="${OMEGA:-0.1}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results_step1_gradient_reward}"
CASSCF_DIR="${CASSCF_DIR:-${SCRIPT_DIR}/CASSCF_csv}"
GPU_LIST="${GPU_LIST:-auto}"
USE_CPU="${USE_CPU:-0}"
CPU_WORKERS="${CPU_WORKERS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
INCLUDE_BARKER="${INCLUDE_BARKER:-0}"

# Hiperparámetros de entrenamiento y evaluación
BATCH_SIZE="${BATCH_SIZE:-128}"
N_BURN_IN="${N_BURN_IN:-1000}"
TD3_STEPS="${TD3_STEPS:-6000}"
RF_STEPS="${RF_STEPS:-2000}"
EVAL_CHAINS="${EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-2000}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
INIT_MODE="${INIT_MODE:-randomized_wigner}"
EVAL_INIT_MODE="${EVAL_INIT_MODE:-spin_ordered_offset}"

# Hiperparámetros del Actor y recompensas del Paso 1
TAU_BASE="${TAU_BASE:-0.10}"
ACTOR_AMP="${ACTOR_AMP:-2.0}"
SIGMA_ACTION_SCALE="${SIGMA_ACTION_SCALE:-2.0}"
SCALE_CLIP_MIN="${SCALE_CLIP_MIN:--0.5}"
SCALE_CLIP_MAX="${SCALE_CLIP_MAX:-2.0}"
FORCE_SCALE="${FORCE_SCALE:-1.0}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-30.0}"
PENALTY_FACTOR="${PENALTY_FACTOR:-0.25}"
KAPPA_VAL="${KAPPA_VAL:-2.5}"
BARKER_BETA="${BARKER_BETA:-2.0}"
BARKER_GAMMA="${BARKER_GAMMA:-0.5}"
LR_ACTOR_TD3="${LR_ACTOR_TD3:-1e-4}"
LR_CRITIC_TD3="${LR_CRITIC_TD3:-3e-4}"
LR_ACTOR_RF="${LR_ACTOR_RF:-2e-3}"

# Memoria y guardado compacto HDF5
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Parseo de argumentos CLI
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --include-barker) INCLUDE_BARKER=1; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --n-burn-in) N_BURN_IN="$2"; shift 2 ;;
        --td3-steps) TD3_STEPS="$2"; shift 2 ;;
        --rf-steps) RF_STEPS="$2"; shift 2 ;;
        --eval-chains) EVAL_CHAINS="$2"; shift 2 ;;
        --eval-burn-in) EVAL_BURN_IN="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --actor-amp) ACTOR_AMP="$2"; shift 2 ;;
        --force-scale) FORCE_SCALE="$2"; shift 2 ;;
        --lr-rf) LR_ACTOR_RF="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,47p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Opción desconocida: $1" >&2
            exit 1
            ;;
    esac
done

# Detectar intérprete Python con JAX/Flax/h5py
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
MASTER_CSV="${BASE_OUT_DIR}/step1_master_metrics.csv"

echo "=============================================================================="
echo "  RL-VMC 1D — BENCHMARK PASO 1 (HOJA DE RUTA: REINFORCE vs TD3 + RECOMPENSAS)"
echo "=============================================================================="
echo "  Python          : $PYTHON_BIN"
echo "  Sistemas (N:g)  : $SYSTEMS_LIST (omega=$OMEGA)"
echo "  Directorio out  : $BASE_OUT_DIR"
echo "  Master CSV      : $MASTER_CSV"
echo "  Modo hardware   : $( [[ "$USE_CPU" == "1" ]] && echo "CPU" || echo "GPUs [${GPUS[*]}]" )"
echo "  Train / Eval    : batch=$BATCH_SIZE, rf_steps=$RF_STEPS, td3_steps=$TD3_STEPS, eval=${EVAL_CHAINS}x${MEASURE_STEPS}"
echo "  Actor Paso 1    : amp=$ACTOR_AMP, sig_scale=$SIGMA_ACTION_SCALE, clip=[$SCALE_CLIP_MIN,$SCALE_CLIP_MAX], force_scale=$FORCE_SCALE"
echo "=============================================================================="

# Helper: localizar el archivo CASSCF .npz convergido para (N, g)
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

# Helper: parámetros (init_offset_base, init_offset_step) centrados en el cristal
# de Wigner exacto para que `spin_ordered_offset` inicialice 100% polarizado por
# espín (todos up a la izquierda, todos down a la derecha) SIN comprimir electrones
# fuera de distribución en [5, 6, ..., 5+N-1].
resolve_wigner_offset() {
    local npz_file="$1"
    local n_val="$2"
    local g_val="$3"
    case "${n_val}:${g_val}" in
        "4:2.0"|"4:2")   echo "-9.27 6.18" ;;
        "4:5.0"|"4:5")   echo "-12.03 8.02" ;;
        "6:1.0"|"6:1")   echo "-10.86 4.34" ;;
        "6:5.0"|"6:5")   echo "-16.80 6.72" ;;
        "8:5.0"|"8:5")   echo "-20.67 5.91" ;;
        *)
            "$PYTHON_BIN" -c "
import sys, numpy as np
sys.path.insert(0, '${SCRIPT_DIR}')
from rl_vmc.postprocess.casscf_baseline import compute_exact_casscf_1body_density
res = compute_exact_casscf_1body_density(sys.argv[1], x_min=-30.0, x_max=30.0, n_grid=2001)
x, rho = np.asarray(res['x']), np.asarray(res['rho_total'])
pks = [i for i in range(1, len(rho)-1) if rho[i]>rho[i-1] and rho[i]>rho[i+1] and rho[i]>0.1*np.max(rho)]
n = int(sys.argv[2])
if len(pks) >= 2:
    xp = x[pks]
    step = float((xp[-1] - xp[0]) / max(n - 1, 1))
    base = -0.5 * (n - 1) * step
    print(f'{base:.2f} {step:.2f}')
else:
    step = 4.0
    base = -0.5 * (n - 1) * step
    print(f'{base:.2f} {step:.2f}')
" "$npz_file" "$n_val"
            ;;
    esac
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
# EJECUCIÓN POR SISTEMA CASSCF (N, g)
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

    read -r OFF_BASE OFF_STEP <<< "$(resolve_wigner_offset "$NPZ_PATH" "$N_ELEC" "$G_VAL")"

    # Chunk seguro de energía local para N >= 8 (70 dets)
    SYS_ELOC_CHUNK="$ELOC_CHUNK_SIZE"
    if [[ "$N_ELEC" -ge 8 && "$SYS_ELOC_CHUNK" -gt 1024 ]]; then
        SYS_ELOC_CHUNK=1024
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo "  SISTEMA: ${SYS_TAG} | CASSCF: $(basename "$NPZ_PATH")"
    echo "  Init polarizado Wigner (eval): base=${OFF_BASE}, step=${OFF_STEP} Bohr"
    echo "------------------------------------------------------------------------------"

    COMMON_ARGS=(
        --n_elec "$N_ELEC" --n_up "$N_UP" --n_dn "$N_DN"
        --omega "$OMEGA" --interaction_strength "$G_VAL" --softening 0.1
        --ansatz_type casscf_npz --casscf_npz_path "$NPZ_PATH" --no-casscf_trainable
        --reference_casscf_npz "$NPZ_PATH"
        --batch_size "$BATCH_SIZE"
        --n_burn_in "$N_BURN_IN"
        --init_mode "$INIT_MODE"
        --eval_init_mode "$EVAL_INIT_MODE"
        --init_offset_base "$OFF_BASE"
        --init_offset_step "$OFF_STEP"
        --n_eval_chains "$EVAL_CHAINS"
        --eval_burn_in "$EVAL_BURN_IN"
        --measure_steps "$MEASURE_STEPS"
        --eloc_chunk_size "$SYS_ELOC_CHUNK"
        --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
        --save_production_energy_mode step_mean
        --save_production_walkers_dtype float32
        --save_production_stats_dtype float32
        --master_csv "$MASTER_CSV"
        --csv_collect_dir "${BASE_OUT_DIR}/csv_summary"
        --pipeline_stage "step1_gradient_reward_benchmark"
        --seed "$SEED"
    )

    NEURAL_BASE_ARGS=(
        --proposal_type neural_deepset
        --train_mode rl_only
        --tau "$TAU_BASE"
        --actor_amplitude "$ACTOR_AMP"
        --sigma_action_scale "$SIGMA_ACTION_SCALE"
        --scale_clip_min "$SCALE_CLIP_MIN"
        --scale_clip_max "$SCALE_CLIP_MAX"
        --force_scale "$FORCE_SCALE"
        --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR"
        --parity_equivariant
        --unclip_sigma_action
        --cdlb_exact_entropy
    )

    # --------------------------------------------------------------------------
    # BLOQUE A: Controles Clásicos (MALA + RWMH all_electrons vs single_electron)
    # --------------------------------------------------------------------------
    echo "  >>> Bloque A: Controles Clásicos (MALA, RWMH all_electrons, RWMH single_electron)..."

    # A.1 MALA estándar (all_electrons, tau=0.10, sigma=0.447 Bohr)
    launch_run "${SYS_DIR}/S1_MALA_all_tau0.10" "${SYS_TAG}_S1_MALA_all_tau0.10" \
        "${COMMON_ARGS[@]}" \
        --proposal_type mala --update_mode all_electrons --train_mode eval_only --tau "$TAU_BASE"

    # A.2 RWMH campeón B1/B2 (all_electrons, sigma=1.21565 Bohr)
    launch_run "${SYS_DIR}/S1_RWMH_all_sig1.22" "${SYS_TAG}_S1_RWMH_all_sig1.22" \
        "${COMMON_ARGS[@]}" \
        --proposal_type metropolis --update_mode all_electrons --train_mode eval_only --rw_sigma 1.21565

    # A.3 RWMH ancho grande all_electrons (sigma=3.00 Bohr — mide colapso (sig_w/sig)^N)
    launch_run "${SYS_DIR}/S1_RWMH_all_sig3.00" "${SYS_TAG}_S1_RWMH_all_sig3.00" \
        "${COMMON_ARGS[@]}" \
        --proposal_type metropolis --update_mode all_electrons --train_mode eval_only --rw_sigma 3.00

    # A.4 RWMH single_electron (sigma=3.00 Bohr — 1 electrón a la vez, MHSampler1D.step_single_electron)
    launch_run "${SYS_DIR}/S1_RWMH_1e_sig3.00" "${SYS_TAG}_S1_RWMH_1e_sig3.00" \
        "${COMMON_ARGS[@]}" \
        --proposal_type metropolis --update_mode single_electron --train_mode eval_only --rw_sigma 3.00

    # --------------------------------------------------------------------------
    # BLOQUE B: Control Off-Policy TD3 (full_gaussian + force_scale=1.0)
    # --------------------------------------------------------------------------
    echo "  >>> Bloque B: Control Off-Policy TD3 (CDLB vs log_esjd en full_gaussian)..."

    # B.1 TD3 + full_gaussian + CDLB exacto
    launch_run "${SYS_DIR}/S1_TD3_fullG_cdlb" "${SYS_TAG}_S1_TD3_fullG_cdlb" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode full_gaussian \
        --rl_algorithm td3 --train_steps "$TD3_STEPS" \
        --reward_type cdlb --penalty_factor "$PENALTY_FACTOR" \
        --lr_actor "$LR_ACTOR_TD3" --lr_critic "$LR_CRITIC_TD3" \
        --no-use_lr_scheduler --use_reward_centering

    # B.2 TD3 + full_gaussian + log_esjd
    launch_run "${SYS_DIR}/S1_TD3_fullG_logESJD" "${SYS_TAG}_S1_TD3_fullG_logESJD" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode full_gaussian \
        --rl_algorithm td3 --train_steps "$TD3_STEPS" \
        --reward_type log_esjd \
        --lr_actor "$LR_ACTOR_TD3" --lr_critic "$LR_CRITIC_TD3" \
        --no-use_lr_scheduler --use_reward_centering

    # --------------------------------------------------------------------------
    # BLOQUE C: Gradiente Exacto Reparametrizado On-Policy (REINFORCELearner1D)
    # --------------------------------------------------------------------------
    echo "  >>> Bloque C: Gradiente Exacto Reparametrizado REINFORCE (CDLB, log_esjd, kappa_jump, resMALA)..."

    # C.1 REINFORCE + full_gaussian + CDLB exacto (ablación 1-factor: TD3 -> REINFORCE)
    launch_run "${SYS_DIR}/S1_RF_fullG_cdlb" "${SYS_TAG}_S1_RF_fullG_cdlb" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode full_gaussian \
        --rl_algorithm reinforce --train_steps "$RF_STEPS" \
        --reward_type cdlb --penalty_factor "$PENALTY_FACTOR" \
        --lr_actor "$LR_ACTOR_RF" \
        --use_lr_scheduler --lr_scheduler_type cosine --lr_warmup_steps 100 --lr_min_ratio 0.05

    # C.2 REINFORCE + full_gaussian + log_esjd (candidato principal de eficiencia local)
    launch_run "${SYS_DIR}/S1_RF_fullG_logESJD" "${SYS_TAG}_S1_RF_fullG_logESJD" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode full_gaussian \
        --rl_algorithm reinforce --train_steps "$RF_STEPS" \
        --reward_type log_esjd \
        --lr_actor "$LR_ACTOR_RF" \
        --use_lr_scheduler --lr_scheduler_type cosine --lr_warmup_steps 100 --lr_min_ratio 0.05

    # C.3 REINFORCE + full_gaussian + kappa_jump (distancia esperada + regularización alpha*(1-alpha))
    launch_run "${SYS_DIR}/S1_RF_fullG_kappa" "${SYS_TAG}_S1_RF_fullG_kappa" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode full_gaussian \
        --rl_algorithm reinforce --train_steps "$RF_STEPS" \
        --reward_type kappa_jump --kappa "$KAPPA_VAL" \
        --lr_actor "$LR_ACTOR_RF" \
        --use_lr_scheduler --lr_scheduler_type cosine --lr_warmup_steps 100 --lr_min_ratio 0.05

    # C.4 REINFORCE + residual_mala + log_esjd (ablación 1-factor: full_gaussian vs residual_mala)
    launch_run "${SYS_DIR}/S1_RF_resMALA_logESJD" "${SYS_TAG}_S1_RF_resMALA_logESJD" \
        "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
        --proposal_mode residual_mala \
        --rl_algorithm reinforce --train_steps "$RF_STEPS" \
        --reward_type log_esjd \
        --lr_actor "$LR_ACTOR_RF" \
        --use_lr_scheduler --lr_scheduler_type cosine --lr_warmup_steps 100 --lr_min_ratio 0.05

    # Opcional: REINFORCE + full_gaussian + barker_jump
    if [[ "$INCLUDE_BARKER" == "1" ]]; then
        launch_run "${SYS_DIR}/S1_RF_fullG_barker" "${SYS_TAG}_S1_RF_fullG_barker" \
            "${COMMON_ARGS[@]}" "${NEURAL_BASE_ARGS[@]}" \
            --proposal_mode full_gaussian \
            --rl_algorithm reinforce --train_steps "$RF_STEPS" \
            --reward_type barker_jump --beta "$BARKER_BETA" --gamma "$BARKER_GAMMA" \
            --lr_actor "$LR_ACTOR_RF" \
            --use_lr_scheduler --lr_scheduler_type cosine --lr_warmup_steps 100 --lr_min_ratio 0.05
    fi

    # Esperar a que terminen todos los runs de este sistema antes de graficar
    wait_all_jobs

    # --------------------------------------------------------------------------
    # POST-PROCESAMIENTO AUTOMÁTICO (Reutilizando scripts existentes)
    # --------------------------------------------------------------------------
    echo "  >>> Ejecutando análisis comparativo, trazas y sondas espaciales para ${SYS_TAG}..."
    FIG_DIR="${SYS_DIR}/figures_step1"
    mkdir -p "$FIG_DIR"

    if [[ -f "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" ]]; then
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" \
            --sys-dir "$SYS_DIR" \
            --casscf-npz "$NPZ_PATH" \
            --out-dir "$FIG_DIR" || echo "  [AVISO] Falló plot_actor_ablation_comparison.py para ${SYS_TAG}"
    fi

    if [[ -f "${SCRIPT_DIR}/scripts/plot_ablation_traces_and_marginals.py" ]]; then
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_ablation_traces_and_marginals.py" \
            --sys-dir "$SYS_DIR" \
            --casscf-npz "$NPZ_PATH" \
            --out-dir "$FIG_DIR" \
            --trace-steps 4000 || echo "  [AVISO] Falló plot_ablation_traces_and_marginals.py para ${SYS_TAG}"
    fi

    for rl_h5 in "${SYS_DIR}"/S1_TD3_*/run_bundle.h5 "${SYS_DIR}"/S1_RF_*/run_bundle.h5; do
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
echo "  BENCHMARK PASO 1 COMPLETADO -> ${BASE_OUT_DIR}"
echo "  Tabla Maestra CSV           -> ${MASTER_CSV}"
echo "=============================================================================="
