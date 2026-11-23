#!/usr/bin/env bash
# ==============================================================================
# PRB RL-VMC PRODUCTION BENCHMARK — SAME 15 SYSTEMS AS results_prb_final_benchmark
# N in {4, 6, 10}, g in {0.1, 0.5, 1.0, 3.0, 5.0} (omega=0.1, soft a=0.1)
# ==============================================================================
# Benchmarks full coupled variational ground-state optimization (RL-VMC) using
# Slater-Jastrow-Backflow (SJB: backflow_ho + spin_rbf_jastrow) with long
# convergence-assured training and multi-chain production evaluation.
#
# Trainer and hyperparameter choices are grounded directly in:
#   - results_pipeline4_param_search / PIPELINE4_ONLINE_COUPLED_REPORT.md:
#       Winners #13/#17/#09 -> 4-regime online_coupled (NodalWarmup + free
#       off-policy CriticWarmup + Two-Timescale CoEvolution + TailFreeze),
#       replay_capacity=25000, walkers_to_store=64, UTD=1, penalty=0.25,
#       reward_centering=True, cosine LR decay.
#   - results_prb_final_benchmark / PRB_FINAL_BENCHMARK_REPORT.md:
#       Parity-equivariant unlocked actor with dimension-calibrated scale
#       ceiling s_sigma(N), stabilized confinement/outward-drift regularization,
#       and optimal-tau MALA / optimal-sigma RWMH strict baselines.
#
# Models executed per (N, g) system:
#   1. M1_VMC_MALA_opt              : VMC + MALA at optimal tau*(N)
#   2. M2_VMC_RWMH_opt              : VMC + RWMH at optimal sigma*(N, g)
#   3. M3_RLVMC_P3_BlockCoupled     : Legacy coupled mode (macro-cycles) + RL-MALA
#   4. M4_RLVMC_P4_Online_CDLB_Safe : Pipeline 4 (4 regimes) + parity RL-MALA (safe scale)
#   5. M5_RLVMC_P4_Online_Unclip    : Pipeline 4 (4 regimes) + unlocked parity RL-MALA (r_CDLB)
#   6. M6_RLVMC_P4_Online_Confined  : Pipeline 4 (4 regimes) + unlocked parity RL-MALA (r_CDLB + r_reg)
#   [Optional --with-base-mala adds M0_VMC_MALA_base_tau0.10]
#
# Usage on Ayler (3 GPUs, 1 job/GPU for large B=512 chains):
#   chmod +x run_prb_rl_vmc_benchmark.sh
#   nohup ./run_prb_rl_vmc_benchmark.sh --gpus "0 1 2" --jobs-per-gpu 1 > prb_rl_vmc_benchmark.log 2>&1 &
#
# Dry-run verification (prints exact planned commands without launching):
#   ./run_prb_rl_vmc_benchmark.sh --dry-run --systems "4:1.0 6:1.0 10:1.0"
# ==============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

# Exact 15-system matrix matching results_prb_final_benchmark
SYSTEMS_LIST="${SYSTEMS_LIST:-4:0.1 4:0.5 4:1.0 4:3.0 4:5.0 6:0.1 6:0.5 6:1.0 6:3.0 6:5.0 10:0.1 10:0.5 10:1.0 10:3.0 10:5.0}"
OMEGA="${OMEGA:-0.1}"
SOFTENING="${SOFTENING:-0.1}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results_prb_rl_vmc_benchmark}"
CASSCF_DIR="${CASSCF_DIR:-${SCRIPT_DIR}/CASSCF_csv}"
GEN_CASSCF_FALLBACK="${GEN_CASSCF_FALLBACK:-${SCRIPT_DIR}/results_prb_final_benchmark/generated_casscf}"

# Hardware & execution control
GPU_LIST="${GPU_LIST:-auto}"
USE_CPU="${USE_CPU:-0}"
CPU_WORKERS="${CPU_WORKERS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
DRY_RUN="${DRY_RUN:-0}"
WITH_BASE_MALA="${WITH_BASE_MALA:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Variational Ansatz (proven in Pipeline 4: beats 20-det CASSCF by -50 mHa on N=6 g=1.0)
ANSATZ_TYPE="${ANSATZ_TYPE:-slater_jastrow_1d}"
ORBITAL_MODEL="${ORBITAL_MODEL:-backflow_ho}"
CORRELATION_MODEL="${CORRELATION_MODEL:-spin_rbf_jastrow}"

# Convergence-assured training & evaluation budgets
BATCH_SIZE="${BATCH_SIZE:-512}"
SWEEPS_PER_STEP="${SWEEPS_PER_STEP:-5}"
VMC_STEPS_SMALL="${VMC_STEPS_SMALL:-12000}"     # N in {4, 6}: 12,000 VMC steps x 5 sweeps = 60,000 MCMC sweeps
VMC_STEPS_LARGE="${VMC_STEPS_LARGE:-15000}"     # N = 10     : 15,000 VMC steps x 5 sweeps = 75,000 MCMC sweeps
N_BURN_IN="${N_BURN_IN:-2000}"
INIT_MODE="${INIT_MODE:-randomized_wigner}"
INIT_STD="${INIT_STD:-3.0}"

LR_ANSATZ="${LR_ANSATZ:-1e-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-300}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.05}"

# Production evaluation (strict 3 x 40,000 chains)
EVAL_CHAINS="${EVAL_CHAINS:-3}"
EVAL_BURN_IN="${EVAL_BURN_IN:-3000}"
MEASURE_STEPS="${MEASURE_STEPS:-40000}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"

# Winning Pipeline 4 (online_coupled) 4-regime schedule & TD3 hyperparameters
# Scaled proportionally from winners #13/#17/#09 of results_pipeline4_param_search
P4_NODAL_WARMUP="${P4_NODAL_WARMUP:-1000}"
P4_CRITIC_WARMUP="${P4_CRITIC_WARMUP:-800}"
P4_FREEZE_TAIL="${P4_FREEZE_TAIL:-2000}"
RL_REPLAY_CAP="${RL_REPLAY_CAP:-25000}"
WALKERS_TO_STORE="${WALKERS_TO_STORE:-64}"
UTD_RATIO="${UTD_RATIO:-1}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-256}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
PENALTY_FACTOR="${PENALTY_FACTOR:-0.25}"
MAX_PHYS_JUMP_FACTOR="${MAX_PHYS_JUMP_FACTOR:-2.0}"
SEED="${SEED:-42}"

# Legacy Pipeline 3 (coupled by macro-cycles) matched total VMC steps
P3_CYCLES="${P3_CYCLES:-12}"                    # 12 x 1000 = 12,000 VMC steps (+ 12 x 400 = 4,800 RL sweeps)
P3_RL_SWEEPS_PER_CYCLE="${P3_RL_SWEEPS_PER_CYCLE:-400}"

# CLI parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --systems) SYSTEMS_LIST="$2"; shift 2 ;;
        --omega) OMEGA="$2"; shift 2 ;;
        --softening) SOFTENING="$2"; shift 2 ;;
        --out-dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --ansatz-type) ANSATZ_TYPE="$2"; shift 2 ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --cpu) USE_CPU=1; shift ;;
        --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --sweeps-per-step) SWEEPS_PER_STEP="$2"; shift 2 ;;
        --vmc-steps-small) VMC_STEPS_SMALL="$2"; shift 2 ;;
        --vmc-steps-large) VMC_STEPS_LARGE="$2"; shift 2 ;;
        --eval-chains) EVAL_CHAINS="$2"; shift 2 ;;
        --eval-burn-in) EVAL_BURN_IN="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --with-base-mala) WITH_BASE_MALA=1; shift ;;
        --force) FORCE_RERUN=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --postprocess-only) POSTPROCESS_ONLY=1; shift ;;
        --seed) SEED="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,39p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Auto-detect Python with JAX/Flax/h5py
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
    echo "[ERROR] No Python interpreter with jax, flax, and h5py found. Pass --python /path/to/python3." >&2
    exit 1
fi

# GPU / CPU worker pool setup
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

mkdir -p "$BASE_OUT_DIR/results_csv"
MASTER_CSV="${BASE_OUT_DIR}/results_csv/prb_rl_vmc_benchmark_master.csv"

echo "=============================================================================="
echo "  RL-VMC 1D — PRB PRODUCTION VARIATIONAL BENCHMARK (15 SYSTEMS)"
echo "=============================================================================="
echo "  Python          : $PYTHON_BIN"
echo "  Systems (N:g)   : $SYSTEMS_LIST (omega=$OMEGA, a=$SOFTENING)"
echo "  Ansatz          : $ANSATZ_TYPE ($ORBITAL_MODEL + $CORRELATION_MODEL)"
echo "  Output dir      : $BASE_OUT_DIR"
echo "  Hardware        : $( [[ "$USE_CPU" == "1" ]] && echo "CPU ($CPU_WORKERS workers)" || echo "GPUs [${GPUS[*]}]" )"
echo "  Training budget : B=$BATCH_SIZE, sweeps=$SWEEPS_PER_STEP, steps=${VMC_STEPS_SMALL}(N<=6)/${VMC_STEPS_LARGE}(N=10)"
echo "  P4 4-Regime     : Nodal=$P4_NODAL_WARMUP, Critic=$P4_CRITIC_WARMUP, Freeze=$P4_FREEZE_TAIL, buf=$RL_REPLAY_CAP, w=$WALKERS_TO_STORE, UTD=$UTD_RATIO"
echo "  Eval budget     : ${EVAL_CHAINS} chains x ${MEASURE_STEPS} steps (burn-in=${EVAL_BURN_IN})"
echo "  Dry run         : $DRY_RUN"
echo "=============================================================================="

# Locate CASSCF reference .npz for exact delta-E and D_KL^spin diagnostics
find_casscf_npz() {
    local n_val="$1"
    local g_val="$2"
    local g_int="${g_val%.0}"
    local found=""
    for search_dir in "$CASSCF_DIR" "${GEN_CASSCF_FALLBACK}/N${n_val}_w${OMEGA}_g${g_val}"; do
        [[ -d "$search_dir" ]] || continue
        for pat in \
            "casscf_state_cas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
            "casscf_statecas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
            "casscf_state_cas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz \
            "casscf_statecas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz \
            "casscf_state_N${n_val}_w${OMEGA}_g${g_val}.npz" \
            "casscf_state.npz"; do
            for f in "${search_dir}"/$pat; do
                if [[ -f "$f" ]]; then
                    found="$f"
                    break 3
                fi
            done
        done
    done
    echo "$found"
}

# Dimension-calibrated optimal classical & RL proposal parameters
# Directly calibrated from results_prb_final_benchmark & results_pipeline4_param_search
optimal_mala_tau() {
    local n_val="$1"
    if [[ "$n_val" -le 4 ]]; then
        echo "0.35"   # sigma = 0.837 Bohr: high acceptance (~75%), low tau_int(E), zero nodal explosion
    elif [[ "$n_val" -le 6 ]]; then
        echo "0.25"   # sigma = 0.707 Bohr: safely below tau=0.74 autocorrelation collapse (tau_E=253)
    else
        echo "0.15"   # sigma = 0.548 Bohr: stable Langevin drift in N=10
    fi
}

optimal_rwmh_sigma() {
    local n_val="$1"
    local g_val="$2"
    if [[ "$n_val" -le 4 ]]; then
        echo "2.7055"
    elif [[ "$n_val" -le 6 ]]; then
        # g <= 1.0: 2.00 keeps acceptance ~18-28% with fast barrier crossing; g >= 3.0: 2.71 spans wide Wigner wells
        if awk "BEGIN {exit !($g_val >= 2.5)}"; then
            echo "2.7055"
        else
            echo "2.0000"
        fi
    else
        # N = 10: sigma = 1.2156 avoids simultaneous all-electron acceptance collapse (13-29% vs 0.3-2% at 2.71)
        echo "1.2156"
    fi
}

optimal_rl_scale_params() {
    # Returns: "ACTOR_AMP SIGMA_ACTION_SCALE SCALE_CLIP_MAX CONF_WEIGHT OUT_WEIGHT"
    local n_val="$1"
    if [[ "$n_val" -le 4 ]]; then
        echo "0.80 1.80 1.80 0.25 0.25"
    elif [[ "$n_val" -le 6 ]]; then
        echo "0.80 1.80 1.80 0.20 0.30"
    else
        # N=10: cap sigma at ~1.56 Bohr (s_sigma=1.25) and use outward-only penalty (conf_w=0.0)
        # to eliminate the negative-reward variance-collapse trap diagnosed at N=10, g>=3.0!
        echo "0.50 1.25 1.25 0.00 0.30"
    fi
}

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
                wait "${JOB_PIDS[$i]}" || echo "  [WARN] Job ${JOB_NAMES[$i]} exited with non-zero status."
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
            wait "${JOB_PIDS[$i]}" || echo "  [WARN] Job ${JOB_NAMES[$i]} failed."
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
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "  [DRY-RUN] $run_name"
        echo "            $PYTHON_BIN cli/run_1d.py --output_h5 $h5_out --run_name $run_name $*"
        return 0
    fi
    if [[ "$FORCE_RERUN" != "1" ]] && is_bundle_complete "$h5_out"; then
        echo "  [SKIP] Complete bundle exists: $h5_out"
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
# EXECUTION ACROSS ALL 15 (N, g) SYSTEMS
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
    REF_ARGS=()
    if [[ -n "$NPZ_PATH" ]]; then
        REF_ARGS=(--reference_casscf_npz "$NPZ_PATH")
    fi

    if [[ "$N_ELEC" -le 6 ]]; then
        VMC_STEPS="$VMC_STEPS_SMALL"
        STEPS_FOR_BEST=3000
        EVAL_INIT_MODE="spin_ordered_offset"
    else
        VMC_STEPS="$VMC_STEPS_LARGE"
        STEPS_FOR_BEST=4000
        # Avoid frozen-walker nodal trapping observed in N=10, g>=3.0 under spin_ordered_offset
        if awk "BEGIN {exit !($G_VAL >= 2.5)}"; then
            EVAL_INIT_MODE="randomized_wigner"
        else
            EVAL_INIT_MODE="spin_ordered_offset"
        fi
    fi

    TAU_OPT="$(optimal_mala_tau "$N_ELEC")"
    SIG_OPT="$(optimal_rwmh_sigma "$N_ELEC" "$G_VAL")"
    read -r RL_AMP RL_SIG_SCALE RL_CLIP_MAX RL_CONF_W RL_OUT_W <<< "$(optimal_rl_scale_params "$N_ELEC")"
    P3_VMC_PER_CYCLE=$((VMC_STEPS / P3_CYCLES))

    echo ""
    echo "------------------------------------------------------------------------------"
    echo "  SYSTEM: ${SYS_TAG} | VMC steps=${VMC_STEPS} | MALA tau*=${TAU_OPT} | RWMH sig*=${SIG_OPT}"
    echo "  RL opt: A=${RL_AMP}, s_sig=${RL_SIG_SCALE}, conf=${RL_CONF_W}, out=${RL_OUT_W} | eval_init=${EVAL_INIT_MODE}"
    echo "  CASSCF ref: ${NPZ_PATH:-NONE}"
    echo "------------------------------------------------------------------------------"

    COMMON_VMC_ARGS=(
        --n_elec "$N_ELEC" --n_up "$N_UP" --n_dn "$N_DN"
        --omega "$OMEGA" --interaction_strength "$G_VAL" --softening "$SOFTENING"
        --ansatz_type "$ANSATZ_TYPE"
        --orbital_model "$ORBITAL_MODEL"
        --correlation_model "$CORRELATION_MODEL"
        --batch_size "$BATCH_SIZE"
        --sweeps_per_step "$SWEEPS_PER_STEP"
        --n_burn_in "$N_BURN_IN"
        --init_mode "$INIT_MODE" --init_std "$INIT_STD"
        --lr_ansatz "$LR_ANSATZ"
        --use_lr_scheduler --lr_scheduler_type cosine
        --lr_warmup_steps "$LR_WARMUP_STEPS" --lr_min_ratio "$LR_MIN_RATIO"
        --grad_clip_norm "$GRAD_CLIP_NORM"
        --steps_for_best "$STEPS_FOR_BEST"
        --eval_init_mode "$EVAL_INIT_MODE"
        --n_eval_chains "$EVAL_CHAINS"
        --eval_burn_in "$EVAL_BURN_IN"
        --measure_steps "$MEASURE_STEPS"
        --eloc_chunk_size "$ELOC_CHUNK_SIZE"
        --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH"
        --save_production_energy_mode step_mean
        --save_production_walkers_dtype float32
        --save_production_stats_dtype float32
        --master_csv "$MASTER_CSV"
        --csv_collect_dir "${BASE_OUT_DIR}/results_csv"
        --pipeline_stage "prb_rl_vmc_benchmark"
        --seed "$SEED"
        "${REF_ARGS[@]}"
    )

    RL_TD3_COMMON=(
        --proposal_type neural_deepset
        --proposal_mode residual_mala
        --rl_algorithm td3
        --reward_type cdlb
        --penalty_factor "$PENALTY_FACTOR"
        --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR"
        --lr_actor "$LR_ACTOR" --lr_critic "$LR_CRITIC"
        --rl_batch_size "$RL_BATCH_SIZE"
        --rl_replay_capacity "$RL_REPLAY_CAP"
        --use_reward_centering
        --parity_equivariant --cdlb_exact_entropy
        --tau 0.10
    )

    P4_SCHEDULE_ARGS=(
        --train_mode online_coupled
        --train_steps "$VMC_STEPS"
        --online_nodal_warmup_steps "$P4_NODAL_WARMUP"
        --online_critic_warmup_steps "$P4_CRITIC_WARMUP"
        --online_rl_freeze_tail_steps "$P4_FREEZE_TAIL"
        --online_walkers_to_store "$WALKERS_TO_STORE"
        --online_rl_update_every 1
        --online_rl_updates_per_step "$UTD_RATIO"
        --online_exp_noise 0.0
    )

    # 0. Optional baseline MALA tau=0.10
    if [[ "$WITH_BASE_MALA" == "1" ]]; then
        launch_run "${SYS_DIR}/M0_VMC_MALA_base_tau0.10" "${SYS_TAG}_M0_VMC_MALA_base_tau0.10" \
            "${COMMON_VMC_ARGS[@]}" \
            --proposal_type mala --train_mode vmc_only --train_steps "$VMC_STEPS" --tau 0.10
    fi

    # 1. VMC + Optimal-tau MALA
    launch_run "${SYS_DIR}/M1_VMC_MALA_opt_tau${TAU_OPT}" "${SYS_TAG}_M1_VMC_MALA_opt_tau${TAU_OPT}" \
        "${COMMON_VMC_ARGS[@]}" \
        --proposal_type mala --train_mode vmc_only --train_steps "$VMC_STEPS" --tau "$TAU_OPT"

    # 2. VMC + Optimal-sigma RWMH
    launch_run "${SYS_DIR}/M2_VMC_RWMH_opt_sig${SIG_OPT}" "${SYS_TAG}_M2_VMC_RWMH_opt_sig${SIG_OPT}" \
        "${COMMON_VMC_ARGS[@]}" \
        --proposal_type metropolis --train_mode vmc_only --train_steps "$VMC_STEPS" --rw_sigma "$SIG_OPT"

    # 3. Legacy Mode: Pipeline 3 Block-Coupled (macro-cycles) + Parity RL-MALA
    launch_run "${SYS_DIR}/M3_RLVMC_P3_BlockCoupled" "${SYS_TAG}_M3_RLVMC_P3_BlockCoupled" \
        "${COMMON_VMC_ARGS[@]}" "${RL_TD3_COMMON[@]}" \
        --train_mode coupled \
        --coupled_training_cycles "$P3_CYCLES" \
        --vmc_iterations_per_cycle "$P3_VMC_PER_CYCLE" \
        --rl_sweeps_per_cycle "$P3_RL_SWEEPS_PER_CYCLE" \
        --unclip_sigma_action \
        --actor_amplitude "$RL_AMP" \
        --sigma_action_scale "$RL_SIG_SCALE" \
        --scale_clip_min -0.5 --scale_clip_max "$RL_CLIP_MAX" \
        --confinement_weight "$RL_CONF_W" --outward_drift_penalty "$RL_OUT_W"

    # 4. Pipeline 4 Online Coupled (4 Regimes) — Ultra-safe scale (P4 Param Search #13/#17 champion)
    launch_run "${SYS_DIR}/M4_RLVMC_P4_Online_CDLB_Safe" "${SYS_TAG}_M4_RLVMC_P4_Online_CDLB_Safe" \
        "${COMMON_VMC_ARGS[@]}" "${RL_TD3_COMMON[@]}" "${P4_SCHEDULE_ARGS[@]}" \
        --no-unclip_sigma_action \
        --actor_amplitude 0.15 --sigma_action_scale 1.0 \
        --scale_clip_min -0.5 --scale_clip_max 1.0 \
        --confinement_weight 0.0 --outward_drift_penalty 0.0

    # 5. Pipeline 4 Online Coupled (4 Regimes) — Unlocked Parity RL-MALA (Pure r_CDLB, Case 1)
    launch_run "${SYS_DIR}/M5_RLVMC_P4_Online_Unclip_CDLB" "${SYS_TAG}_M5_RLVMC_P4_Online_Unclip_CDLB" \
        "${COMMON_VMC_ARGS[@]}" "${RL_TD3_COMMON[@]}" "${P4_SCHEDULE_ARGS[@]}" \
        --unclip_sigma_action \
        --actor_amplitude "$RL_AMP" \
        --sigma_action_scale "$RL_SIG_SCALE" \
        --scale_clip_min -0.5 --scale_clip_max "$RL_CLIP_MAX" \
        --confinement_weight 0.0 --outward_drift_penalty 0.0

    # 6. Pipeline 4 Online Coupled (4 Regimes) — Unlocked Parity RL-MALA + Stabilized Confinement (Case 2)
    launch_run "${SYS_DIR}/M6_RLVMC_P4_Online_Confined" "${SYS_TAG}_M6_RLVMC_P4_Online_Confined" \
        "${COMMON_VMC_ARGS[@]}" "${RL_TD3_COMMON[@]}" "${P4_SCHEDULE_ARGS[@]}" \
        --unclip_sigma_action \
        --actor_amplitude "$RL_AMP" \
        --sigma_action_scale "$RL_SIG_SCALE" \
        --scale_clip_min -0.5 --scale_clip_max "$RL_CLIP_MAX" \
        --confinement_weight "$RL_CONF_W" --outward_drift_penalty "$RL_OUT_W"

    wait_all_jobs
done

if [[ "$DRY_RUN" != "1" ]]; then
    echo ""
    echo ">>> Running per-system post-processing & ranked summary aggregation..."
    for sys_pair in $SYSTEMS_LIST; do
        N_ELEC="${sys_pair%%:*}"
        G_VAL="${sys_pair##*:}"
        SYS_TAG="N${N_ELEC}_w${OMEGA}_g${G_VAL}"
        SYS_DIR="${BASE_OUT_DIR}/${SYS_TAG}"
        H5_LIST=("${SYS_DIR}"/*/run_bundle.h5)
        if [[ -f "${H5_LIST[0]:-}" && -f "${SCRIPT_DIR}/cli/postprocess_1d.py" ]]; then
            CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/cli/postprocess_1d.py" \
                --inputs "${H5_LIST[@]}" \
                --out_dir "${SYS_DIR}/figures_postprocess" \
                --csv_collect_dir "${BASE_OUT_DIR}/results_csv" \
                --system_tag "PRB_RLVMC_${SYS_TAG}" \
                --figure_format both --dpi 600 || echo "  [WARN] postprocess failed for ${SYS_TAG}"
        fi
    done
fi

echo ""
echo "=============================================================================="
echo "  PRB RL-VMC BENCHMARK COMPLETE -> ${BASE_OUT_DIR}"
echo "=============================================================================="
