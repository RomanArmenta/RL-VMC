#!/usr/bin/env bash
# ==============================================================================
# PRB LETTER FINAL BENCHMARK (OPTION A) — SCALING N in {4, 6, 10},
# g in {0.1, 0.5, 1.0, 3.0, 5.0} — CURRENT vs. CONFINED RL-MALA + CONTROLS
# ==============================================================================
# Runs the complete 15-system benchmark for the Physical Review B Letter:
#   N in {4, 6, 10}, g in {0.1, 0.5, 1.0, 3.0, 5.0} (omega = 0.1, soft a = 0.1)
#
# For every (N, g) system:
#   0. Auto-locates CASSCF reference .npz in CASSCF_csv/ (or auto-generates
#      missing references such as N=10, g=0.1 and N=10, g=0.5 via cli/run_casscf_1d.py).
#   1. B1_MALA_base_tau0.10               : Baseline MALA (tau=0.10, sigma=0.447 Bohr)
#   2. B1_MALA_matched_tau0.7389          : Large-step MALA (tau=0.7389, sigma=1.216 Bohr)
#   3. B1_RWMH_matched_sig1.2156          : Matched-sigma RWMH (sigma=1.216 Bohr)
#   4. B1_RWMH_matched_sig2.7055          : Matched-sigma RWMH (sigma=2.706 Bohr)
#   5. B2_RL_MALA_eq_unclip_amp0.80_sig1.8: CASE 1 — Current parity-equivariant RL-MALA
#                                           (confinement_weight=0.0, outward_drift_penalty=0.0)
#   6. B2_RL_MALA_confined_amp0.80_sig1.8 : CASE 2 — Confinement-regularized RL-MALA
#                                           (confinement_weight=0.5, outward_drift_penalty=0.25)
#
# At completion:
#   - Aggregates per-system summaries into ${BASE_OUT_DIR}/prb_final_benchmark_master.csv
#   - Runs spatial probes on Case 1 & Case 2 actors
#   - Invokes scripts/generate_prb_letter_figures.py --benchmark-dir "${BASE_OUT_DIR}"
#     to regenerate the publication PDF/PNG figures into RL_Sampler_PRBLett/figures/.
#
# Usage on Ayler (3 GPUs, full Option A production):
#   chmod +x run_prb_final_benchmark.sh
#   nohup ./run_prb_final_benchmark.sh --gpus "0 1 2" > prb_final_benchmark.log 2>&1 &
#
# Quick local CPU smoke check:
#   ./run_prb_final_benchmark.sh --cpu --systems "4:1.0" --batch-size 32 \
#       --rl-steps 400 --measure-steps 2000 --eval-chains 1 --eval-burn-in 200 --n-burn-in 200
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

# Option A default matrix (15 systems)
SYSTEMS_LIST="${SYSTEMS_LIST:-4:0.1 4:0.5 4:1.0 4:3.0 4:5.0 6:0.1 6:0.5 6:1.0 6:3.0 6:5.0 10:0.1 10:0.5 10:1.0 10:3.0 10:5.0}"
OMEGA="${OMEGA:-0.1}"
SOFTENING="${SOFTENING:-0.1}"
BASE_OUT_DIR="${BASE_OUT_DIR:-results_prb_final_benchmark}"
CASSCF_DIR="${CASSCF_DIR:-${SCRIPT_DIR}/CASSCF_csv}"
AUTO_GENERATE_CASSCF="${AUTO_GENERATE_CASSCF:-1}"
GPU_LIST="${GPU_LIST:-auto}"
USE_CPU="${USE_CPU:-0}"
CPU_WORKERS="${CPU_WORKERS:-1}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
POSTPROCESS_ONLY="${POSTPROCESS_ONLY:-0}"
REGENERATE_PRB_FIGURES="${REGENERATE_PRB_FIGURES:-1}"

# Training & evaluation hyperparameters (calibrated for Ayler GPUs)
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
CONFINEMENT_WEIGHT="${CONFINEMENT_WEIGHT:-0.5}"
OUTWARD_DRIFT_PENALTY="${OUTWARD_DRIFT_PENALTY:-0.25}"
ACTOR_AMPLITUDE="${ACTOR_AMPLITUDE:-0.80}"
SIGMA_ACTION_SCALE="${SIGMA_ACTION_SCALE:-1.8}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-3e-4}"
ELOC_CHUNK_SIZE="${ELOC_CHUNK_SIZE:-2048}"
SAVE_WALKERS_MAX_BATCH="${SAVE_WALKERS_MAX_BATCH:-32}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-}"

# CLI argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --systems) SYSTEMS_LIST="$2"; shift 2 ;;
        --omega) OMEGA="$2"; shift 2 ;;
        --softening) SOFTENING="$2"; shift 2 ;;
        --out-dir) BASE_OUT_DIR="$2"; shift 2 ;;
        --casscf-dir) CASSCF_DIR="$2"; shift 2 ;;
        --no-auto-casscf) AUTO_GENERATE_CASSCF=0; shift ;;
        --gpus) GPU_LIST="$2"; shift 2 ;;
        --cpu) USE_CPU=1; shift ;;
        --cpu-workers) CPU_WORKERS="$2"; shift 2 ;;
        --jobs-per-gpu) JOBS_PER_GPU="$2"; shift 2 ;;
        --force) FORCE_RERUN=1; shift ;;
        --postprocess-only) POSTPROCESS_ONLY=1; shift ;;
        --no-prb-figures) REGENERATE_PRB_FIGURES=0; shift ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --n-burn-in) N_BURN_IN="$2"; shift 2 ;;
        --rl-steps) RL_STEPS="$2"; shift 2 ;;
        --eval-chains) EVAL_CHAINS="$2"; shift 2 ;;
        --eval-burn-in) EVAL_BURN_IN="$2"; shift 2 ;;
        --measure-steps) MEASURE_STEPS="$2"; shift 2 ;;
        --penalty) PENALTY_FACTOR="$2"; shift 2 ;;
        --confinement-weight) CONFINEMENT_WEIGHT="$2"; shift 2 ;;
        --outward-drift-penalty) OUTWARD_DRIFT_PENALTY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,38p' "$0"
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

# Detect available GPUs unless --cpu forced
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
GEN_CASSCF_DIR="${BASE_OUT_DIR}/generated_casscf"

echo "=============================================================================="
echo "  RL-VMC 1D — PRB LETTER FINAL BENCHMARK (OPTION A)"
echo "=============================================================================="
echo "  Python              : $PYTHON_BIN"
echo "  Systems (N:g)       : $SYSTEMS_LIST (omega=$OMEGA, a=$SOFTENING)"
echo "  Output directory    : $BASE_OUT_DIR"
echo "  Hardware            : $( [[ "$USE_CPU" == "1" ]] && echo "CPU ($CPU_WORKERS workers)" || echo "GPUs [${GPUS[*]}]" )"
echo "  Train / Eval        : batch=$BATCH_SIZE, rl_steps=$RL_STEPS, eval=${EVAL_CHAINS}x${MEASURE_STEPS}"
echo "  Case 2 confinement  : confinement_weight=$CONFINEMENT_WEIGHT, outward_drift_penalty=$OUTWARD_DRIFT_PENALTY"
echo "=============================================================================="

# Locate CASSCF .npz in CASSCF_DIR or previously auto-generated folder
find_casscf_npz() {
    local n_val="$1"
    local g_val="$2"
    local g_int="${g_val%.0}"
    local found=""
    for search_dir in "$CASSCF_DIR" "${GEN_CASSCF_DIR}/N${n_val}_w${OMEGA}_g${g_val}"; do
        [[ -d "$search_dir" ]] || continue
        for pat in \
            "casscf_state_cas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
            "casscf_statecas_n${n_val}_g${g_val}_omega${OMEGA}_"*.npz \
            "casscf_state_cas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz \
            "casscf_statecas_n${n_val}_g${g_int}_omega${OMEGA}_"*.npz \
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

ensure_casscf_npz() {
    local n_val="$1"
    local g_val="$2"
    local npz_found
    npz_found="$(find_casscf_npz "$n_val" "$g_val")"
    if [[ -n "$npz_found" ]]; then
        echo "$npz_found"
        return 0
    fi
    if [[ "$AUTO_GENERATE_CASSCF" != "1" ]]; then
        echo ""
        return 1
    fi
    local out_cas="${GEN_CASSCF_DIR}/N${n_val}_w${OMEGA}_g${g_val}"
    mkdir -p "$out_cas"
    echo "  [CASSCF] Missing reference for N=${n_val}, g=${g_val} -> generating with cli/run_casscf_1d.py (soft=${SOFTENING})..." >&2
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/cli/run_casscf_1d.py" \
        --n_elec "$n_val" \
        --omega "$OMEGA" \
        --g "$g_val" \
        --softening "$SOFTENING" \
        --outdir "$out_cas" \
        --reuse_existing \
        --csv_collect_dir "" > "${out_cas}/casscf_gen.log" 2>&1 || true
    npz_found="$(find_casscf_npz "$n_val" "$g_val")"
    echo "$npz_found"
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
# EXECUTION ACROSS ALL (N, g) SYSTEMS
# ==============================================================================
for sys_pair in $SYSTEMS_LIST; do
    N_ELEC="${sys_pair%%:*}"
    G_VAL="${sys_pair##*:}"
    N_UP=$((N_ELEC / 2))
    N_DN=$((N_ELEC - N_UP))
    SYS_TAG="N${N_ELEC}_w${OMEGA}_g${G_VAL}"
    SYS_DIR="${BASE_OUT_DIR}/${SYS_TAG}"
    mkdir -p "$SYS_DIR"

    NPZ_PATH="$(ensure_casscf_npz "$N_ELEC" "$G_VAL")"
    if [[ -z "$NPZ_PATH" ]]; then
        echo "[ERROR] Could not locate or generate CASSCF .npz for N=$N_ELEC, g=$G_VAL, omega=$OMEGA. Skipping." >&2
        continue
    fi

    echo ""
    echo "------------------------------------------------------------------------------"
    echo "  SYSTEM: ${SYS_TAG} | CASSCF: $(basename "$NPZ_PATH")"
    echo "------------------------------------------------------------------------------"

    COMMON_ARGS=(
        --n_elec "$N_ELEC" --n_up "$N_UP" --n_dn "$N_DN"
        --omega "$OMEGA" --interaction_strength "$G_VAL" --softening "$SOFTENING"
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
        --csv_collect_dir ""
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
        --parity_equivariant --unclip_sigma_action --cdlb_exact_entropy
        --actor_amplitude "$ACTOR_AMPLITUDE"
        --sigma_action_scale "$SIGMA_ACTION_SCALE"
        --scale_clip_min -0.5 --scale_clip_max "$SIGMA_ACTION_SCALE"
        --tau 0.10
    )

    # 1. Baseline MALA (tau=0.10, sigma=0.4472 Bohr)
    launch_run "${SYS_DIR}/B1_MALA_base_tau0.10" "${SYS_TAG}_B1_MALA_base_tau0.10" \
        "${COMMON_ARGS[@]}" \
        --proposal_type mala --train_mode eval_only --tau 0.10

    # 2. Large-step MALA (tau=0.7389, sigma=1.2156 Bohr)
    launch_run "${SYS_DIR}/B1_MALA_matched_tau0.7389" "${SYS_TAG}_B1_MALA_matched_tau0.7389" \
        "${COMMON_ARGS[@]}" \
        --proposal_type mala --train_mode eval_only --tau 0.7389056

    # 3. Matched-sigma RWMH (sigma=1.2156 Bohr, legacy ceiling)
    launch_run "${SYS_DIR}/B1_RWMH_matched_sig1.2156" "${SYS_TAG}_B1_RWMH_matched_sig1.2156" \
        "${COMMON_ARGS[@]}" \
        --proposal_type metropolis --train_mode eval_only --rw_sigma 1.21565

    # 4. Matched-sigma RWMH (sigma=2.7055 Bohr, exact unlocked RL-MALA width for s_sigma=1.8)
    launch_run "${SYS_DIR}/B1_RWMH_matched_sig2.7055" "${SYS_TAG}_B1_RWMH_matched_sig2.7055" \
        "${COMMON_ARGS[@]}" \
        --proposal_type metropolis --train_mode eval_only --rw_sigma 2.70548

    # 5. CASE 1: Current parity-equivariant unlocked RL-MALA (no confinement regularization)
    launch_run "${SYS_DIR}/B2_RL_MALA_eq_unclip_amp0.80_sig1.8" "${SYS_TAG}_B2_RL_MALA_eq_unclip_amp0.80_sig1.8" \
        "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
        --confinement_weight 0.0 --outward_drift_penalty 0.0

    # 6. CASE 2: Confinement-regularized parity-equivariant RL-MALA
    launch_run "${SYS_DIR}/B2_RL_MALA_confined_amp0.80_sig1.8" "${SYS_TAG}_B2_RL_MALA_confined_amp0.80_sig1.8" \
        "${COMMON_ARGS[@]}" "${RL_COMMON_ARGS[@]}" \
        --confinement_weight "$CONFINEMENT_WEIGHT" \
        --outward_drift_penalty "$OUTWARD_DRIFT_PENALTY"

    wait_all_jobs

    # Per-system post-processing & spatial probes
    echo "  >>> Post-processing & spatial probes for ${SYS_TAG}..."
    FIG_DIR="${SYS_DIR}/figures_ablation"
    mkdir -p "$FIG_DIR"

    if [[ -f "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" ]]; then
        CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/plot_actor_ablation_comparison.py" \
            --sys-dir "$SYS_DIR" \
            --casscf-npz "$NPZ_PATH" \
            --out-dir "$FIG_DIR" || echo "  [WARN] plot_actor_ablation_comparison.py failed for ${SYS_TAG}"
    fi

    for rl_h5 in "${SYS_DIR}"/B2_RL_MALA_*/run_bundle.h5; do
        if [[ -f "$rl_h5" ]]; then
            CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/probe_actor_spatial_response.py" \
                --h5 "$rl_h5" \
                --casscf_npz "$NPZ_PATH" \
                --probe_spin down \
                --out_dir "${FIG_DIR}/spatial_probes" \
                --figure_format both || true
        fi
    done
done

# ==============================================================================
# MASTER SUMMARY AGGREGATION & AUTOMATIC PRB FIGURE REGENERATION
# ==============================================================================
echo ""
echo ">>> Aggregating master CSV table across all completed systems..."
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" - "$BASE_OUT_DIR" <<'PYEOF'
import csv, glob, os, re, sys
from pathlib import Path

base_dir = Path(sys.argv[1])
rows = []
fieldnames = None

for csv_path in sorted(base_dir.glob("N*_w*_g*/figures_ablation/ablation_summary_*.csv")):
    m = re.search(r"N(\d+)_w([0-9.]+)_g([0-9.]+)", str(csv_path.parent.parent.name))
    if not m:
        continue
    n_val, w_val, g_val = int(m.group(1)), float(m.group(2)), float(m.group(3))
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = ["N", "omega", "g", "system_tag"] + list(reader.fieldnames or [])
        for r in reader:
            row = {"N": n_val, "omega": w_val, "g": g_val, "system_tag": csv_path.parent.parent.name}
            row.update(r)
            rows.append(row)

if rows and fieldnames:
    out_csv = base_dir / "prb_final_benchmark_master.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[+] Wrote master benchmark table ({len(rows)} rows): {out_csv}")
else:
    print("[WARN] No per-system ablation_summary CSVs found yet.")
PYEOF

if [[ "$REGENERATE_PRB_FIGURES" == "1" && -f "${SCRIPT_DIR}/scripts/generate_prb_letter_figures.py" ]]; then
    echo ">>> Regenerating publication-grade PRB Letter figures using ${BASE_OUT_DIR}..."
    CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" "${SCRIPT_DIR}/scripts/generate_prb_letter_figures.py" \
        --benchmark-dir "$BASE_OUT_DIR" \
        --out-dir "${SCRIPT_DIR}/RL_Sampler_PRBLett/figures" || echo "[WARN] PRB figure regeneration reported an issue."
fi

echo ""
echo "=============================================================================="
echo "  PRB FINAL BENCHMARK COMPLETE -> ${BASE_OUT_DIR}"
echo "=============================================================================="
