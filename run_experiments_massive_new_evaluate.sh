#!/bin/bash

# ==============================================================================
# PIPELINE DE EVALUACIÓN COMPLETA PARA EXPERIMENTOS RL-VMC
# ==============================================================================

# Cambia BASE_DIR al directorio donde realizaste tus barridos (ej. results_massive_7)
BASE_DIR="results_massive_7"

# Define la lista de sistemas N y las interacciones g exploradas
NS=(4 10)
GS=(0.1 1.0 3.0 5.0)
OMEGA=0.1
N_EVALUATIONS=5

echo "[$(date +'%H:%M:%S')] Iniciando pipeline de evaluación para el directorio: $BASE_DIR"

# 1. Ejecutar evaluaciones y gráficos por cada sub-sistema
for n in "${NS[@]}"; do
    for g in "${GS[@]}"; do
        SYS_DIR="${BASE_DIR}/results_N${n}_g${g}"
        
        if [ ! -d "$SYS_DIR" ]; then
            continue
        fi
        
        echo "-------------------------------------------------------------------"
        echo "[$(date +'%H:%M:%S')] Procesando sistema: N=$n, g=$g ($SYS_DIR)"
        echo "-------------------------------------------------------------------"
        
        # A. Generar Dashboard de Entrenamiento
        echo "  -> Generando Dashboard de Métricas de Entrenamiento..."
        python3 scripts/plot_training_metrics_all.py --sys_dir "$SYS_DIR"
        
        # B. Generar Figura 4 (Eficiencia Computacional y Estadística stats.json)
        # Esto es obligatorio antes de los observables porque identifica el mejor Tau de MALA
        echo "  -> Evaluando eficiencia y generando Figura 4..."
        python3 scripts/generate_figure4_dynamic.py --sys_dir "$SYS_DIR" --n_evaluations $N_EVALUATIONS
        
        # C. Generar Escaneo de Observables (ACF, Densidad, Pares, Varianza)
        echo "  -> Extrayendo observables físicos y estadísticos..."
        python3 scripts/generate_observables_analysis.py --sys_dir "$SYS_DIR"
        
    done

    # 2. Generar Figura 3 agregada para el número de electrones N
    echo "-------------------------------------------------------------------"
    echo "[$(date +'%H:%M:%S')] Agregando datos y generando Figura 3 para N=$n..."
    echo "-------------------------------------------------------------------"
    python3 scripts/generate_figure3_dynamic.py --base_dir "$BASE_DIR" --n_elec $n --omega $OMEGA

done

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] TODAS LAS EVALUACIONES Y GRÁFICOS HAN SIDO GENERADOS."
echo "==================================================================="