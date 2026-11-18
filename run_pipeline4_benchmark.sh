#!/usr/bin/env bash
# ==============================================================================
# PIPELINE 4 BENCHMARK: ENTRENAMIENTO VARIACIONAL ACOPLADO ONLINE (4 REGÍMENES)
# ==============================================================================
# Orquestador dedicado para recorrer múltiples sistemas físicos (N, g) con el
# Pipeline 4 (`--train_mode online_coupled` en `OnlineCoupledTrainer1D`),
# comparando:
#   - Baseline VMC + MALA (`vmc_only`)
#   - (Opcional con `--with-p3`) Pipeline 3 Acoplado por Bloques (`coupled`)
#   - Pipeline 4 Acoplado Online Paso a Paso en 4 Regímenes (`online_coupled`)
#
# Por defecto recorre un grupo piloto de 3 sistemas representativos:
#   N=4 (g=2.0), N=4 (g=4.0, Cristal de Wigner) y N=6 (g=1.0, Punto Cuántico 6e)
# o toda la grilla (N in {2,4,6,10}, g in {0.1..8.0}) pasando `--full-grid`.
#
# Uso en Ayler (3 GPUs, 2 trabajos/GPU = 6 trabajos paralelos):
#   chmod +x run_pipeline4_benchmark.sh
#   nohup ./run_pipeline4_benchmark.sh --gpus "0 1 2" --jobs-per-gpu 2 > master_p4_benchmark.log 2>&1 &
#
# Comparar Pipeline 3 vs Pipeline 4 en los 3 sistemas piloto:
#   nohup ./run_pipeline4_benchmark.sh --with-p3 --gpus "0 1 2" --jobs-per-gpu 2 > master_p3_vs_p4.log 2>&1 &
#
# Regenerar únicamente figuras y CSVs (ej. en PDF a 600 DPI):
#   ./run_pipeline4_benchmark.sh --postprocess-only --pdf --dpi 600
# ==============================================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export BASE_OUT_DIR="${BASE_OUT_DIR:-results_pipeline4_benchmark}"
export PIPELINES_TO_RUN="${PIPELINES_TO_RUN:-4}"
# Por defecto probamos un grupo pequeño de 3 sistemas representativos (N4:2.0, N4:4.0, N6:1.0)
export SYSTEMS_FILTER="${SYSTEMS_FILTER:-N4:2.0 N4:4.0 N6:1.0}"
export N_LIST="${N_LIST:-4 6}"
export G_LIST="${G_LIST:-1.0 2.0 4.0}"
export GPU_LIST="${GPU_LIST:-auto}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-auto}"

FORWARD_ARGS=()
USER_SET_SYSTEMS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full-grid|--all-systems)
            SYSTEMS_FILTER=""
            N_LIST="2 4 6 10"
            G_LIST="0.1 0.5 1.0 2.0 3.0 5.0 8.0"
            USER_SET_SYSTEMS=1
            shift
            ;;
        --with-p3|--compare-p3)
            PIPELINES_TO_RUN="3 4"
            shift
            ;;
        --with-p1-p2|--all-pipelines)
            PIPELINES_TO_RUN="1 2 3 4"
            shift
            ;;
        --systems|--system-filter)
            SYSTEMS_FILTER="$2"
            USER_SET_SYSTEMS=1
            shift 2
            ;;
        --n-list|--n_list|--n-filter)
            N_LIST="$2"
            if [[ "$USER_SET_SYSTEMS" -eq 0 ]]; then
                SYSTEMS_FILTER=""
            fi
            USER_SET_SYSTEMS=1
            shift 2
            ;;
        --g-list|--g_list|--g-filter)
            G_LIST="$2"
            if [[ "$USER_SET_SYSTEMS" -eq 0 ]]; then
                SYSTEMS_FILTER=""
            fi
            USER_SET_SYSTEMS=1
            shift 2
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

ARGS_TO_PASS=(
    "$BASE_OUT_DIR"
    --pipelines "$PIPELINES_TO_RUN"
    --gpus "$GPU_LIST"
    --jobs-per-gpu "$JOBS_PER_GPU"
    --max-parallel-jobs "$MAX_PARALLEL_JOBS"
    --n-list "$N_LIST"
    --g-list "$G_LIST"
)

if [[ -n "$SYSTEMS_FILTER" ]]; then
    ARGS_TO_PASS+=(--systems "$SYSTEMS_FILTER")
fi

if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    ARGS_TO_PASS+=("${FORWARD_ARGS[@]}")
fi

exec "${SCRIPT_DIR}/run_production_benchmark_3pipelines.sh" "${ARGS_TO_PASS[@]}"
