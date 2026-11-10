#!/usr/bin/env bash
# ==============================================================================
# ORQUESTADOR DE PRODUCCIÓN CON ESTADOS CASSCF CONVERGIDOS DEL ASESOR (N=6 y N=8)
# ==============================================================================
# Este script:
#   1. Instala los estados CASSCF de espacio activo completo (b=256, c0-a6 para N=6
#      y b=256, c0-a8 para N=8) provistos por el asesor en la carpeta `CASSCF_csv/`
#      hacia `<BASE_OUT_DIR>/pipeline1_casscf/N{N}_w0.1_g{G}/casscf_state.npz`,
#      generando sus densidades exactas de 1 cuerpo y gráficas diagnósticas.
#   2. Detecta y respalda automáticamente corridas previas de N=6, N=8 o N=10 que
#      hayan sido generadas con el CASSCF truncado no convergido (c1-a4 / c4-a4, b=48),
#      preservando intactas las corridas válidas de N=2 y N=4 (y cualquier corrida
#      de N=6, 8 que ya haya usado el estado b=256 del asesor).
#   3. Ejecuta el Pipeline 2 (MALA vs RL-MALA TD3 en muestreo puro sobre CASSCF
#      congelado desde `spin_ordered_offset`) en paralelo sobre las GPUs disponibles.
#   4. Genera todas las gráficas de post-procesamiento, cruce nodal, resumen global
#      y el análisis exhaustivo de flips (`scripts/analyze_flips_vs_g.py`) dentro de
#      `<BASE_OUT_DIR>/figures/flips_analysis`.
#
# Uso en Ayler / Baker:
#   chmod +x run_advisor_casscf_pipeline.sh
#   nohup ./run_advisor_casscf_pipeline.sh results_production_3pipelines_baker > out_advisor_casscf.log 2>&1 &
#
# Opciones adicionales:
#   ./run_advisor_casscf_pipeline.sh results_production_3pipelines_baker --gpus 0,1,2,3 --jobs-per-gpu 1
#   ./run_advisor_casscf_pipeline.sh results_production_3pipelines_baker --postprocess-only
#   ./run_advisor_casscf_pipeline.sh results_production_3pipelines_baker --force-rerun
# ==============================================================================

set -u
set -o pipefail

export JAX_ENABLE_X64=True
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
export XLA_FLAGS="--xla_gpu_autotune_level=2 --xla_gpu_enable_triton_gemm=false ${XLA_FLAGS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

BASE_OUT_DIR="results_production_3pipelines_baker"
ADVISOR_DIR="${SCRIPT_DIR}/CASSCF_csv"
POSTPROCESS_ONLY=0
FORCE_RERUN=0
KEEP_N10=0
USER_GPUS=""
JOBS_PER_GPU=1
MAX_TOTAL_JOBS=4
FIGURE_FORMAT="png"
FIGURE_DPI=300

while [ $# -gt 0 ]; do
    case "$1" in
        --postprocess-only|--plots-only)
            POSTPROCESS_ONLY=1
            ;;
        --force-rerun|--Overwrite)
            FORCE_RERUN=1
            ;;
        --keep-n10)
            KEEP_N10=1
            ;;
        --advisor-dir)
            shift
            ADVISOR_DIR="${1:-}"
            ;;
        --gpus)
            shift
            USER_GPUS="${1:-}"
            ;;
        --jobs-per-gpu)
            shift
            JOBS_PER_GPU="${1:-1}"
            ;;
        --max-jobs)
            shift
            MAX_TOTAL_JOBS="${1:-4}"
            ;;
        --figure-format)
            shift
            FIGURE_FORMAT="${1:-png}"
            ;;
        --dpi)
            shift
            FIGURE_DPI="${1:-300}"
            ;;
        -h|--help)
            echo "Uso: $0 [carpeta_salida] [--gpus 0,1,2,3] [--jobs-per-gpu 1] [--max-jobs 4] [--postprocess-only] [--force-rerun]"
            exit 0
            ;;
        -*)
            echo "[ERROR] Bandera no reconocida: $1" >&2
            exit 1
            ;;
        *)
            BASE_OUT_DIR="$1"
            ;;
    esac
    shift
done

BASE_OUT_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$BASE_OUT_DIR")"
ADVISOR_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$ADVISOR_DIR")"

if [ ! -d "$ADVISOR_DIR" ]; then
    echo "[ERROR] No existe la carpeta con los estados CASSCF del asesor: $ADVISOR_DIR" >&2
    exit 1
fi

P1_DIR="${BASE_OUT_DIR}/pipeline1_casscf"
P2_DIR="${BASE_OUT_DIR}/pipeline2_pure_sampling"
CSV_DIR="${BASE_OUT_DIR}/live_csv_tables"
CSV_COLLECT_DIR="${BASE_OUT_DIR}/results_csv"
LOG_DIR="${BASE_OUT_DIR}/logs"
FIG_ROOT_DIR="${BASE_OUT_DIR}/figures"
FIG_P1_DIR="${FIG_ROOT_DIR}/pipeline1_casscf"
FIG_P2_DIR="${FIG_ROOT_DIR}/pipeline2_systems"
FIG_ALL_MODELS_DIR="${FIG_ROOT_DIR}/all_models_vs_N_and_g"
QUARANTINE_DIR="${BASE_OUT_DIR}/quarantine_unconverged_casscf"
TRACKER_DIR="${LOG_DIR}/.gpu_tracker_advisor_$$"

mkdir -p "$P1_DIR" "$P2_DIR" "$CSV_DIR" "$CSV_COLLECT_DIR" "$LOG_DIR" \
         "$FIG_P1_DIR" "$FIG_P2_DIR" "$FIG_ALL_MODELS_DIR" "$TRACKER_DIR"

LIVE_CSV_P1="${CSV_DIR}/live_pipeline1_casscf_summary.csv"
LIVE_CSV_P2_RUNS="${CSV_DIR}/live_pipeline2_runs_raw.csv"
LIVE_CSV_P2_COMP="${CSV_DIR}/live_pipeline2_mala_vs_rl_comparison.csv"
LIVE_CSV_ALL_MODELS="${CSV_DIR}/live_all_models_master.csv"

OMEGA="0.1"
N_WALKERS=128
N_BURN_IN=2000
RL_ONLY_STEPS=6000
N_EVAL_CHAINS=3
EVAL_BURN_IN=200
MEASURE_STEPS=40000
ELOC_CHUNK_SIZE=64
SAVE_WALKERS_MAX_BATCH=32
REWARD_TYPE="cdlb"
PENALTY_FACTOR="0.25"
SIGMA_ACTION_SCALE="1.1"
MAX_PHYS_JUMP_FACTOR="2.0"
LR_ACTOR="1e-4"
LR_CRITIC="3e-4"
INIT_MODE="symmetric_gaussian"
INIT_STD="3.0"
EVAL_INIT_MODE="spin_ordered_offset"
PLOT_STYLE_ARGS=(--figure_format "$FIGURE_FORMAT" --dpi "$FIGURE_DPI")

# Detectar GPUs disponibles
AVAILABLE_GPUS=()
if [ -n "$USER_GPUS" ]; then
    IFS=',' read -r -a AVAILABLE_GPUS <<< "$USER_GPUS"
elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r gpu_id; do
        gpu_id=$(echo "$gpu_id" | tr -d '[:space:]')
        [ -n "$gpu_id" ] && AVAILABLE_GPUS+=("$gpu_id")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)
fi
if [ "${#AVAILABLE_GPUS[@]}" -eq 0 ]; then
    AVAILABLE_GPUS=("0")
fi

cleanup_on_exit() {
    rm -rf "$TRACKER_DIR" 2>/dev/null || true
}
trap cleanup_on_exit EXIT INT TERM

get_system_p2_params() {
    local N=$1
    local G=$2
    local tau="0.10"
    local actor_amp="0.15"
    local penalty="$PENALTY_FACTOR"
    local sig_scale="$SIGMA_ACTION_SCALE"
    local is_strong_g
    is_strong_g=$(awk -v g="$G" 'BEGIN { print (g >= 2.5) ? 1 : 0 }')

    if [ "$N" -le 6 ]; then
        if [ "$is_strong_g" -eq 1 ]; then
            tau="0.08"
            actor_amp="0.12"
        else
            tau="0.10"
            actor_amp="0.15"
        fi
    else
        # N=8: movimiento colectivo de 8 electrones en base b=256 (70 dets)
        if [ "$is_strong_g" -eq 1 ]; then
            tau="0.06"
            actor_amp="0.10"
        else
            tau="0.08"
            actor_amp="0.12"
        fi
    fi
    echo "$tau $actor_amp $penalty $sig_scale"
}

is_h5_completed_with_advisor_b256() {
    local h5_file="$1"
    [ -f "$h5_file" ] || return 1
    python3 - "$h5_file" << 'PYEOF' >/dev/null 2>&1
import sys, json, h5py, os, zipfile

p = sys.argv[1]
if not os.path.exists(p) or os.path.getsize(p) < 10240:
    sys.exit(1)
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
        # Intentar leer summary.json junto al .h5
        sum_p = os.path.join(os.path.dirname(p), "summary.json")
        if os.path.exists(sum_p):
            with open(sum_p, "r", encoding="utf-8") as fs:
                s = json.load(fs)
            npz_path = (s.get("casscf_comparison") or {}).get("casscf_npz_path") or ""
    if not npz_path or not os.path.exists(npz_path):
        sys.exit(2)
    with zipfile.ZipFile(npz_path, "r") as z:
        raw = z.read("metadata_json.npy")
        hlen = int.from_bytes(raw[8:10], "little")
        meta = json.loads(raw[10 + hlen:].decode("utf-32-le").rstrip("\x00"))
    m_cfg = meta.get("config", {})
    n_basis = int(m_cfg.get("n_basis", 0))
    n_core = int(m_cfg.get("n_core", -1))
    conv = bool(meta.get("converged", False))
    if conv and n_basis >= 256 and n_core == 0:
        sys.exit(0)
    sys.exit(2)
PYEOF
    return $?
}

quarantine_old_system_runs_if_needed() {
    local N="$1"
    local G="$2"
    local G_ALT
    G_ALT=$(python3 -c 'import sys; print(f"{float(sys.argv[1]):g}")' "$G")
    local tag
    for tag in "N${N}_w${OMEGA}_g${G}" "N${N}_w${OMEGA}_g${G_ALT}"; do
        local SYS_P2="${P2_DIR}/${tag}"
        [ -d "$SYS_P2" ] || continue

        local needs_quarantine=0
        for h5f in "$SYS_P2"/*/run_bundle.h5; do
            [ -e "$h5f" ] || continue
            if [ "$FORCE_RERUN" -eq 1 ] || ! is_h5_completed_with_advisor_b256 "$h5f"; then
                needs_quarantine=1
                break
            fi
        done
        # También poner en cuarentena si no hay ningún .h5 completo o si cambió el nombre del tag
        if [ "$needs_quarantine" -eq 0 ] && [ "$tag" != "N${N}_w${OMEGA}_g${G}" ]; then
            needs_quarantine=1
        fi

        if [ "$needs_quarantine" -eq 1 ]; then
            local ts
            ts=$(date "+%Y%m%d_%H%M%S")
            local q_dest="${QUARANTINE_DIR}/${tag}_${ts}"
            mkdir -p "$QUARANTINE_DIR"
            echo "  [Quarantine] Respaldando corrida previa (CASSCF antiguo c1-a4/b48 o --force-rerun) de ${tag} -> ${q_dest}"
            mv "$SYS_P2" "$q_dest"
            if [ -d "${FIG_P2_DIR}/${tag}" ]; then
                mv "${FIG_P2_DIR}/${tag}" "${q_dest}/figures_p2" 2>/dev/null || true
            fi
        fi
    done
}

# Si existen corridas de N=10 con CASSCF truncado c4-a4 que no convergió, moverlas a cuarentena
if [ "$KEEP_N10" -eq 0 ]; then
    for n10_dir in "${P2_DIR}"/N10_w*; do
        [ -d "$n10_dir" ] || continue
        n10_tag=$(basename "$n10_dir")
        ts=$(date "+%Y%m%d_%H%M%S")
        mkdir -p "$QUARANTINE_DIR"
        echo "[Quarantine] Moviendo sistema ${n10_tag} (CASSCF c4-a4 truncado) a ${QUARANTINE_DIR}/${n10_tag}_${ts}"
        mv "$n10_dir" "${QUARANTINE_DIR}/${n10_tag}_${ts}"
    done
fi

echo "=============================================================================="
echo " PIPELINE DE PRODUCCIÓN CON ESTADOS CASSCF CONVERGIDOS DEL ASESOR (b=256)"
echo " Carpeta de entrada CASSCF : $ADVISOR_DIR"
echo " Carpeta raíz de resultados: $BASE_OUT_DIR"
echo " GPUs detectadas           : ${AVAILABLE_GPUS[*]} (máx $MAX_TOTAL_JOBS trabajos simultáneos)"
echo "=============================================================================="

# Descubrir automáticamente todos los pares (N, g, npz_path) en CASSCF_csv
SYSTEM_RECORDS=()
while IFS= read -r line; do
    [ -n "$line" ] && SYSTEM_RECORDS+=("$line")
done < <(python3 - "$ADVISOR_DIR" << 'PYEOF'
import sys, glob, json, zipfile
from pathlib import Path

adv_dir = Path(sys.argv[1])
candidates = list(sorted(adv_dir.glob("*.npz")))
# Incluir también N=6, g=1.0 b=256 desde figures/ si no estuviera en CASSCF_csv
fallback_n6_g1 = Path("figures/cas_n6_g1.0_omega0.1_soft0.1_b256_c0-a6_casscf_state_det-20.npz")
if fallback_n6_g1.exists():
    candidates.append(fallback_n6_g1)

seen_ng = set()
items = []
for p in candidates:
    try:
        with zipfile.ZipFile(p, "r") as z:
            raw = z.read("metadata_json.npy")
            hlen = int.from_bytes(raw[8:10], "little")
            meta = json.loads(raw[10+hlen:].decode("utf-32-le").rstrip("\x00"))
        cfg = meta.get("config", {})
        n_elec = int(cfg.get("n_up", 0)) + int(cfg.get("n_down", 0))
        g_val = float(cfg.get("g", 0.0))
        g_str = f"{g_val:.1f}" if abs(g_val - round(g_val)) < 1e-9 else f"{g_val:g}"
        if (n_elec, round(g_val, 5)) in seen_ng:
            continue
        seen_ng.add((n_elec, round(g_val, 5)))
        items.append((n_elec, g_val, g_str, str(p.resolve())))
    except Exception as e:
        sys.stderr.write(f"[WARN] No se pudo leer {p}: {e}\n")

items.sort(key=lambda x: (x[0], x[1]))
for n_elec, _, g_str, path_str in items:
    print(f"{n_elec}|{g_str}|{path_str}")
PYEOF
)

if [ "${#SYSTEM_RECORDS[@]}" -eq 0 ]; then
    echo "[ERROR] No se encontraron archivos .npz válidos en $ADVISOR_DIR" >&2
    exit 1
fi

echo "[+] Se descubrieron ${#SYSTEM_RECORDS[@]} estados CASSCF convergidos del asesor:"
for rec in "${SYSTEM_RECORDS[@]}"; do
    IFS='|' read -r N G NPZ_PATH <<< "$rec"
    echo "    -> N=${N}, g=${G} : $(basename "$NPZ_PATH")"
done

# ==============================================================================
# ETAPA 1: INSTALACIÓN DE REFERENCIAS CASSCF CONVERGIDAS EN PIPELINE 1
# ==============================================================================
if [ "$POSTPROCESS_ONLY" -eq 0 ]; then
    echo ""
    echo "=============================================================================="
    echo " [ETAPA 1] Instalando estados CASSCF convergidos del asesor en Pipeline 1..."
    echo "=============================================================================="
    for rec in "${SYSTEM_RECORDS[@]}"; do
        IFS='|' read -r N G NPZ_PATH <<< "$rec"
        N_UP=$((N / 2))
        N_DN=$((N - N_UP))
        SYS_TAG="N${N}_w${OMEGA}_g${G}"
        OUT_CASSCF="${P1_DIR}/${SYS_TAG}"
        FIG_SYS_P1="${FIG_P1_DIR}/${SYS_TAG}"
        SYS_LOG="${LOG_DIR}/p1_advisor_casscf_${SYS_TAG}.log"

        quarantine_old_system_runs_if_needed "$N" "$G"

        echo "  -> [Pipeline 1 | ${SYS_TAG}] Instalando $(basename "$NPZ_PATH")..."
        env CUDA_VISIBLE_DEVICES="" python3 -u cli/run_casscf_1d.py \
            --n_elec "$N" \
            --n_up "$N_UP" \
            --n_dn "$N_DN" \
            --omega "$OMEGA" \
            --g "$G" \
            --advisor_npz_path "$NPZ_PATH" \
            --outdir "$OUT_CASSCF" \
            --master_csv "$LIVE_CSV_P1" \
            --csv_collect_dir "$CSV_COLLECT_DIR" > "$SYS_LOG" 2>&1
        if [ $? -ne 0 ] || [ ! -f "${OUT_CASSCF}/casscf_state.npz" ]; then
            echo "  [ERROR] Falló instalación de CASSCF para ${SYS_TAG}. Ver log: $SYS_LOG" >&2
            continue
        fi

        mkdir -p "$FIG_SYS_P1"
        for f_img in "$OUT_CASSCF"/*.png "$OUT_CASSCF"/*.pdf "$OUT_CASSCF"/summary*.json; do
            [ -e "$f_img" ] && cp -f "$f_img" "$FIG_SYS_P1/"
        done
    done
fi

# ==============================================================================
# ETAPA 2: MUESTREO PURO MALA vs RL-MALA TD3 SOBRE ESTADOS DEL ASESOR
# ==============================================================================
get_active_jobs_on_gpu() {
    local target_gpu="$1"
    local count=0
    for f in "$TRACKER_DIR"/job_*_gpu_${target_gpu}.meta; do
        [ -e "$f" ] || continue
        local pid
        pid=$(basename "$f" | cut -d'_' -f2)
        if kill -0 "$pid" 2>/dev/null; then
            count=$((count + 1))
        else
            rm -f "$f"
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
        else
            rm -f "$f"
        fi
    done
    echo "$count"
}

SELECTED_GPU="0"
wait_for_free_gpu_slot() {
    while true; do
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
                SELECTED_GPU="$best_gpu"
                return 0
            fi
        fi
        sleep 2
    done
}

run_advisor_p2_worker() {
    trap - EXIT INT TERM
    local GPU_ID="$1"
    local N="$2"
    local G="$3"
    local N_UP=$((N / 2))
    local N_DN=$((N - N_UP))
    read -r TAU ACTOR_AMP PENALTY SIG_SCALE <<< "$(get_system_p2_params "$N" "$G")"

    local SYS_TAG="N${N}_w${OMEGA}_g${G}"
    local CASSCF_NPZ="${P1_DIR}/${SYS_TAG}/casscf_state.npz"
    local SYS_OUT="${P2_DIR}/${SYS_TAG}"
    local MALA_DIR="${SYS_OUT}/${SYS_TAG}_MALA_tau${TAU}"
    local RL_DIR="${SYS_OUT}/${SYS_TAG}_RL_MALA_tau${TAU}_amp${ACTOR_AMP}_pen${PENALTY}_sig${SIG_SCALE}"
    local COMP_DIR="${SYS_OUT}/postprocess_comparison"
    local FIG_SYS_P2="${FIG_P2_DIR}/${SYS_TAG}"
    local MALA_H5="${MALA_DIR}/run_bundle.h5"
    local RL_H5="${RL_DIR}/run_bundle.h5"

    mkdir -p "$MALA_DIR" "$RL_DIR" "$COMP_DIR" "$FIG_SYS_P2"

    if [ "$POSTPROCESS_ONLY" -eq 0 ]; then
        if ! is_h5_completed_with_advisor_b256 "$MALA_H5"; then
            echo "[${SYS_TAG} | GPU ${GPU_ID}] Ejecutando MALA (tau=${TAU}) sobre CASSCF b=256..."
            env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                --run_name "P2_MALA_FixedCASSCF_${SYS_TAG}" \
                --pipeline_stage "pipeline2_pure_sampling" \
                --train_mode eval_only \
                --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                --omega "$OMEGA" --g "$G" --softening 0.1 \
                --ansatz_type casscf_npz \
                --casscf_npz_path "$CASSCF_NPZ" \
                --no-casscf_trainable \
                --proposal_type mala \
                --tau "$TAU" \
                --batch_size "$N_WALKERS" \
                --init_mode "$INIT_MODE" \
                --init_std "$INIT_STD" \
                --eval_init_mode "$EVAL_INIT_MODE" \
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                --save_production_energy_mode "step_mean" \
                --save_production_walkers_dtype "float32" \
                --save_production_stats_dtype "float32" \
                --output_h5 "$MALA_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P2_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --no-export_legacy_files || return 1
        else
            echo "[${SYS_TAG}] MALA ya completado con CASSCF b=256: $MALA_H5"
        fi

        if ! is_h5_completed_with_advisor_b256 "$RL_H5"; then
            echo "[${SYS_TAG} | GPU ${GPU_ID}] Ejecutando RL-MALA TD3 (tau=${TAU}, amp=${ACTOR_AMP}, pen=${PENALTY}) sobre CASSCF b=256..."
            env CUDA_VISIBLE_DEVICES="$GPU_ID" python3 -u cli/run_1d.py \
                --run_name "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
                --pipeline_stage "pipeline2_pure_sampling" \
                --train_mode rl_only \
                --n_elec "$N" --n_up "$N_UP" --n_dn "$N_DN" \
                --omega "$OMEGA" --g "$G" --softening 0.1 \
                --ansatz_type casscf_npz \
                --casscf_npz_path "$CASSCF_NPZ" \
                --no-casscf_trainable \
                --proposal_type neural_deepset \
                --proposal_mode residual_mala \
                --rl_algorithm td3 \
                --reward_type "$REWARD_TYPE" \
                --actor_amplitude "$ACTOR_AMP" \
                --penalty_factor "$PENALTY" \
                --sigma_action_scale "$SIG_SCALE" \
                --max_phys_jump_factor "$MAX_PHYS_JUMP_FACTOR" \
                --no-use_lr_scheduler \
                --lr_actor "$LR_ACTOR" \
                --lr_critic "$LR_CRITIC" \
                --n_ter "$N_BURN_IN" \
                --train_steps "$RL_ONLY_STEPS" \
                --tau "$TAU" \
                --batch_size "$N_WALKERS" \
                --init_mode "$INIT_MODE" \
                --init_std "$INIT_STD" \
                --eval_init_mode "$EVAL_INIT_MODE" \
                --n_eval_chains "$N_EVAL_CHAINS" \
                --eval_burn_in "$EVAL_BURN_IN" \
                --measure_steps "$MEASURE_STEPS" \
                --eloc_chunk_size "$ELOC_CHUNK_SIZE" \
                --save_walkers_max_batch "$SAVE_WALKERS_MAX_BATCH" \
                --save_production_energy_mode "step_mean" \
                --save_production_walkers_dtype "float32" \
                --save_production_stats_dtype "float32" \
                --output_h5 "$RL_H5" \
                --reference_casscf_npz "$CASSCF_NPZ" \
                --master_csv "$LIVE_CSV_P2_RUNS" \
                --csv_collect_dir "$CSV_COLLECT_DIR" \
                --no-export_legacy_files || return 1
        else
            echo "[${SYS_TAG}] RL-MALA TD3 ya completado con CASSCF b=256: $RL_H5"
        fi
    fi

    if [ -f "$MALA_H5" ] && [ -f "$RL_H5" ]; then
        echo "[${SYS_TAG}] Generando 8 figuras de postprocess + 4 figuras de cruce nodal..."
        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u cli/postprocess.py \
            --inputs "$MALA_H5" "$RL_H5" \
            --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
            --casscf_npz_path "$CASSCF_NPZ" \
            --out_dir "$COMP_DIR" \
            --system_tag "P2_${SYS_TAG}" \
            --master_comparison_csv "$LIVE_CSV_P2_COMP" \
            --csv_collect_dir "$CSV_COLLECT_DIR" \
            "${PLOT_STYLE_ARGS[@]}" || return 1

        env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u cli/evaluate_nodal_crossing_1d.py \
            --inputs "$MALA_H5" "$RL_H5" \
            --labels "P2_MALA_FixedCASSCF_${SYS_TAG}" "P2_RL_TD3_FixedCASSCF_${SYS_TAG}" \
            --casscf_npz_path "$CASSCF_NPZ" \
            --system_tag "$SYS_TAG" \
            --out_dir "$FIG_SYS_P2" \
            --csv_collect_dir "$CSV_COLLECT_DIR" \
            "${PLOT_STYLE_ARGS[@]}" || return 1

        for f_out in "$COMP_DIR"/*.png "$COMP_DIR"/*.pdf "$COMP_DIR"/*.csv "$COMP_DIR"/*.json; do
            [ -e "$f_out" ] && cp -f "$f_out" "$FIG_SYS_P2/"
        done
    fi
    return 0
}

echo ""
echo "=============================================================================="
echo " [ETAPA 2] Ejecutando Pipeline 2 (MALA vs RL-MALA) sobre estados CASSCF b=256"
echo "=============================================================================="

for rec in "${SYSTEM_RECORDS[@]}"; do
    IFS='|' read -r N G NPZ_PATH <<< "$rec"
    SYS_TAG="N${N}_w${OMEGA}_g${G}"
    SYS_LOG="${LOG_DIR}/p2_advisor_sampling_${SYS_TAG}.log"

    wait_for_free_gpu_slot
    GPU_ID="$SELECTED_GPU"

    run_advisor_p2_worker "$GPU_ID" "$N" "$G" > "$SYS_LOG" 2>&1 &
    worker_pid=$!
    echo "gpu=$GPU_ID" > "$TRACKER_DIR/job_${worker_pid}_gpu_${GPU_ID}.meta"
    echo "  >> Despachado ${SYS_TAG} en GPU ${GPU_ID} (PID=${worker_pid}) -> Log: ${SYS_LOG}"
done

echo ""
echo "[+] Esperando a que finalicen todos los trabajadores de Pipeline 2..."
while [ "$(get_total_active_jobs)" -gt 0 ]; do
    sleep 3
done

# Verificar que cada sistema despachado haya generado ambos run_bundle.h5 válidos
P2_FAILED=0
for rec in "${SYSTEM_RECORDS[@]}"; do
    IFS='|' read -r N G NPZ_PATH <<< "$rec"
    SYS_TAG="N${N}_w${OMEGA}_g${G}"
    read -r TAU ACTOR_AMP PENALTY SIG_SCALE <<< "$(get_system_p2_params "$N" "$G")"
    M_H5="${P2_DIR}/${SYS_TAG}/${SYS_TAG}_MALA_tau${TAU}/run_bundle.h5"
    R_H5="${P2_DIR}/${SYS_TAG}/${SYS_TAG}_RL_MALA_tau${TAU}_amp${ACTOR_AMP}_pen${PENALTY}_sig${SIG_SCALE}/run_bundle.h5"
    if [ ! -f "$M_H5" ] || [ ! -f "$R_H5" ]; then
        echo "  [WARN] ${SYS_TAG} no completó ambos run_bundle.h5. Revisa: ${LOG_DIR}/p2_advisor_sampling_${SYS_TAG}.log" >&2
        P2_FAILED=$((P2_FAILED + 1))
    fi
done

# ==============================================================================
# ETAPA 3: RESUMEN GLOBAL Y GRÁFICAS DE FLIPS VS g (N=2, 4, 6, 8)
# ==============================================================================
echo ""
echo "=============================================================================="
echo " [ETAPA 3] Generando gráficas globales y análisis de flips vs g (N=2, 4, 6, 8)"
echo "=============================================================================="

python3 - "$LIVE_CSV_P2_COMP" "$LIVE_CSV_ALL_MODELS" "$CSV_COLLECT_DIR" "$KEEP_N10" << 'PYEOF' || true
import os, sys, pandas as pd
from rl_vmc.core.io_bundle import atomic_write_csv, atomic_copy_file
p2_csv, out_csv, collect_dir, keep_n10 = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
if os.path.exists(p2_csv) and os.path.getsize(p2_csv) > 0:
    df = pd.read_csv(p2_csv)
    if "label" in df.columns:
        df = df.drop_duplicates(subset=["label"], keep="last").reset_index(drop=True)
    if keep_n10 == 0 and "N" in df.columns:
        df = df[pd.to_numeric(df["N"], errors="coerce") != 10].reset_index(drop=True)
    atomic_write_csv(df, out_csv, index=False)
    if os.path.isdir(collect_dir):
        atomic_copy_file(out_csv, os.path.join(collect_dir, os.path.basename(out_csv)))
PYEOF

if [ -f "$LIVE_CSV_ALL_MODELS" ] && [ -s "$LIVE_CSV_ALL_MODELS" ]; then
    env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u scripts/plot_benchmark_all_models_summary.py \
        --csv "$LIVE_CSV_ALL_MODELS" \
        --out_dir "$FIG_ALL_MODELS_DIR" \
        "${PLOT_STYLE_ARGS[@]}" || true
fi

FLIPS_FILTER_ARGS=(--n_filter 2 4 6 8)
if [ "$KEEP_N10" -eq 1 ]; then
    FLIPS_FILTER_ARGS=()
fi

env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu python3 -u scripts/analyze_flips_vs_g.py \
    --root_dir "$BASE_OUT_DIR" \
    --out_dir "${FIG_ROOT_DIR}/flips_analysis" \
    "${FLIPS_FILTER_ARGS[@]}" \
    "${PLOT_STYLE_ARGS[@]}"

echo ""
echo "=============================================================================="
echo " EJECUCIÓN COMPLETADA EXITOSAMENTE"
echo " Figuras de flips vs g en : ${FIG_ROOT_DIR}/flips_analysis"
echo " Resumen CSV de flips en  : ${FIG_ROOT_DIR}/flips_analysis/flips_vs_g_summary.csv"
echo " Figuras por sistema en   : ${FIG_P2_DIR}"
echo "=============================================================================="
