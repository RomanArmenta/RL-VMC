#!/bin/bash

BASE_DIR="results_hyperparam_sweep_N10_g5.0"

# Leemos la configuración que ganó de tu archivo de texto
BEST_TD3_DIR=$(cat "${BASE_DIR}/best_td3_model.txt")

# Basados en tu log, sabemos que el tau de 0.01 fue el baseline óptimo
BEST_MALA_DIR="MALA_Baselines/MALA_tau0.01"

echo "Generando Escaneo Profundo de Observables contra Baseline..."

python3 scripts/generate_observables_analysis.py \
    --mala_dir "${BASE_DIR}/${BEST_MALA_DIR}" \
    --td3_dir "${BASE_DIR}/${BEST_TD3_DIR}" \
    --out_dir "${BASE_DIR}" \
    --n_samples 20000 \
    --n_ter 2000

echo "¡Evaluación completada con éxito!"