#!/bin/bash

# ==============================================================================
# PIPELINE DE EVALUACIÓN DEL GRID SEARCH (HYPERPARAMETER SWEEP)
# ==============================================================================

# Cambia esto si nombraste el directorio de forma diferente en tu benchmark .sh
BASE_DIR="results_hyperparam_sweep_N10_g5.0"

echo "[$(date +'%H:%M:%S')] Iniciando evaluación del Hyperparameter Grid Search en: $BASE_DIR"

if [ ! -d "$BASE_DIR/MALA_Baselines" ]; then
    echo "ERROR: No se encontró el directorio de baselines de MALA."
    exit 1
fi

# 1. Ejecutar el escaneo estricto de eficiencia sobre todas las configuraciones
echo "-------------------------------------------------------------------"
echo "[$(date +'%H:%M:%S')] Analizando eficiencia G_RL y ESS/s para todas las configuraciones..."
echo "-------------------------------------------------------------------"
python3 scripts/evaluate_grid_search.py \
    --base_dir "$BASE_DIR" \
    --n_samples 20000 \
    --n_evaluations 3 \
    --n_ter 2000

# 2. Leer la configuración ganadora
BEST_TD3_DIR=$(cat "${BASE_DIR}/best_td3_model.txt")
echo "[$(date +'%H:%M:%S')] -> La configuración ganadora es: $BEST_TD3_DIR"

# 3. Preparar entorno para escaneo de observables (ACF, Varianza, Densidad)
echo "-------------------------------------------------------------------"
echo "[$(date +'%H:%M:%S')] Generando Escaneo Profundo de Observables contra Baseline..."
echo "-------------------------------------------------------------------"

# Creamos un enlace simbólico temporal del mejor TD3 dentro de MALA_Baselines
# para engañar a generate_observables_analysis.py y que analice solo esta pareja.
ln -sfn "../${BEST_TD3_DIR}" "${BASE_DIR}/MALA_Baselines/TD3"

# Ejecutamos el análisis de observables sobre el entorno virtual acoplado
python3 scripts/generate_observables_analysis.py \
    --sys_dir "${BASE_DIR}/MALA_Baselines" \
    --n_samples 20000 \
    --n_ter 2000

# Limpiamos el enlace simbólico para mantener limpio el entorno y movemos el PDF al raíz
rm "${BASE_DIR}/MALA_Baselines/TD3"
mv "${BASE_DIR}/MALA_Baselines/Observables_Deep_Scan.pdf" "${BASE_DIR}/Winning_Config_Observables.pdf"

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] TODAS LAS EVALUACIONES DEL GRID SEARCH COMPLETADAS."
echo "Archivos principales generados:"
echo " - ${BASE_DIR}/Hyperparam_Sweep_Dashboard.pdf (Ranking de configuraciones)"
echo " - ${BASE_DIR}/Winning_Config_Observables.pdf (Física del mejor modelo)"
echo " - ${BASE_DIR}/hyperparam_sweep_results.csv (Datos tabulares en crudo)"
echo "==================================================================="