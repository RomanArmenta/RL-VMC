#!/usr/bin/env bash
# ==============================================================================
# run_wigner_postprocessing_all_casscf.sh
# ==============================================================================
# Revisa todos los estados CASSCF generados por el productor de 3 pipelines
# y ejecuta el post-procesamiento de Wigner del asesor para extraer las densidades
# analíticas exactas completas.
# Soporta banderas de formato de figura (PNG por defecto, PDF, PPI, títulos, etc.).
# ==============================================================================

set -u
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FIGURE_DPI="${FIGURE_DPI:-300}"
TITLE_PREFIX="${TITLE_PREFIX:-}"
CUSTOM_SUPTITLE="${CUSTOM_SUPTITLE:-}"
NO_TITLES="${NO_TITLES:-0}"
FONT_SCALE="${FONT_SCALE:-1.0}"
LINE_WIDTH_SCALE="${LINE_WIDTH_SCALE:-1.0}"
TRANSPARENT_BG="${TRANSPARENT_BG:-0}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --figure-format|--figure_format|--format|-f)
            FIGURE_FORMAT="$2"; shift 2 ;;
        --pdf)
            FIGURE_FORMAT="pdf"; shift ;;
        --png)
            FIGURE_FORMAT="png"; shift ;;
        --dpi|--ppi)
            FIGURE_DPI="$2"; shift 2 ;;
        --title-prefix|--title_prefix)
            TITLE_PREFIX="$2"; shift 2 ;;
        --custom-suptitle|--custom_suptitle)
            CUSTOM_SUPTITLE="$2"; shift 2 ;;
        --no-titles|--no_titles|--hide-titles)
            NO_TITLES=1; shift ;;
        --font-scale|--font_scale)
            FONT_SCALE="$2"; shift 2 ;;
        --line-width-scale|--line_width_scale)
            LINE_WIDTH_SCALE="$2"; shift 2 ;;
        --transparent-bg|--transparent_bg)
            TRANSPARENT_BG=1; shift ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

PLOT_STYLE_ARGS=(
    --figure_format "$FIGURE_FORMAT"
    --dpi "$FIGURE_DPI"
    --font_scale "$FONT_SCALE"
    --line_width_scale "$LINE_WIDTH_SCALE"
)
[[ -n "$TITLE_PREFIX" ]] && PLOT_STYLE_ARGS+=(--title_prefix "$TITLE_PREFIX")
[[ -n "$CUSTOM_SUPTITLE" ]] && PLOT_STYLE_ARGS+=(--custom_suptitle "$CUSTOM_SUPTITLE")
[[ "$NO_TITLES" -eq 1 || "$NO_TITLES" == "true" ]] && PLOT_STYLE_ARGS+=(--no_titles)
[[ "$TRANSPARENT_BG" -eq 1 || "$TRANSPARENT_BG" == "true" ]] && PLOT_STYLE_ARGS+=(--transparent_bg)

LOG_DIR="logs_wigner_post"
mkdir -p "$LOG_DIR"
MASTER_LOG="${LOG_DIR}/master_wigner_postprocessing.log"

echo "=============================================================================="
echo "    INICIANDO POST-PROCESAMIENTO DE WIGNER CASSCF EXACTO"
echo "    Fecha y Hora  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "    Python Bin    : ${PYTHON_BIN}"
echo "    Formato / PPI : ${FIGURE_FORMAT} (${FIGURE_DPI} PPI)"
echo "    Log Maestro   : ${MASTER_LOG}"
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
    --n_grid 500 \
    "${PLOT_STYLE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "${MASTER_LOG}"

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))

echo "=============================================================================="
echo "    POST-PROCESAMIENTO DE WIGNER FINALIZADO"
echo "    Duración Total: ${ELAPSED}s"
echo "    Fecha y Hora  : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "==============================================================================" | tee -a "${MASTER_LOG}"
