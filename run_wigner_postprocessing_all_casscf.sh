#!/usr/bin/env bash
# ==============================================================================
# run_wigner_postprocessing_all_casscf.sh
# ==============================================================================
# Revisa todos los estados CASSCF generados por el productor de 3 pipelines
# y ejecuta el post-procesamiento de Wigner del asesor para extraer las densidades
# analíticas exactas completas.
# ==============================================================================

set -u
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

LOG_DIR="logs_wigner_post"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/master_wigner_postprocessing.log"

echo "=============================================================================="
echo "    INICIANDO POST-PROCESAMIENTO DE WIGNER CASSCF EXACTO"
echo "    Fecha y Hora: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "    Python Bin  : ${PYTHON_BIN}"
echo "    Log Maestro : ${MASTER_LOG}"
echo "=============================================================================="

START_EPOCH=$(date +%s)

${PYTHON_BIN} -u scripts/run_wigner_postprocessing_all_casscf.py \
    --search_dirs \
        results_production_3pipelines_b2/pipeline1_casscf \
        results_production_3pipelines/pipeline1_casscf \
        uhf_1d_ho_v270826_m-determinants \
        figures \
    --out_dir results_casscf_wigner_densities \
    --csv_collect_dir results_csv_b2 \
    --x_min -15.0 \
    --x_max 15.0 \
    --n_grid 500 2>&1 | tee -a "${MASTER_LOG}"

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))

echo "=============================================================================="
echo "    POST-PROCESAMIENTO DE WIGNER FINALIZADO"
echo "    Duración Total: ${ELAPSED}s"
echo "    Fecha y Hora  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "==============================================================================" | tee -a "${MASTER_LOG}"
