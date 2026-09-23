#!/bin/bash

# ==============================================================================
# EVALUACIÓN MASIVA: ESTADÍSTICA DE BARRIDO DE MALA VS TD3
# Este script NO entrena modelos. Solo evalúa modelos existentes en BASE_DIR.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. HIPERPARÁMETROS Y RUTAS CONFIGURABLES
# ------------------------------------------------------------------------------
BASE_DIR="results_massive"

# Espacio físico evaluado (Debe coincidir con las carpetas existentes)
NS=(2 4 6 10)
GS=(0.01 0.1 0.5 1.0 2.0 3.0 4.0 5.0)
OMEGA=0.1

# Parámetros estrictos de la evaluación (MCMC)
N_EVALUATIONS=5      # Número de evaluaciones independientes por modelo (semillas distintas)
N_SAMPLES=100_000      # Longitud de la cadena de Markov (producción)
N_TER=0           # Pasos de termalización (burn-in) descartados antes de medir

# Recursos computacionales
GPUS=(0)       # Lista de GPUs disponibles

# ------------------------------------------------------------------------------
# 2. SISTEMA DE GESTIÓN DE RECURSOS (GPU)
# ------------------------------------------------------------------------------
get_free_gpu(){
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_python=$(nvidia-smi -i $gpu --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)
            if [ -z "$process_python" ]; then
                echo "$gpu"
                return
            fi
        done
        sleep 15 # Revisión frecuente ya que las evaluaciones son más rápidas
    done
}

# ------------------------------------------------------------------------------
# 3. CICLO DE EVALUACIÓN Y RENDERIZADO
# ------------------------------------------------------------------------------
echo "==================================================================="
echo "[$(date +'%H:%M:%S')] INICIANDO PROTOCOLO DE EVALUACIÓN MASIVA"
echo "Directorio Base : $BASE_DIR"
echo "Evaluaciones    : $N_EVALUATIONS"
echo "Pasos (Producción): $N_SAMPLES | Pasos (Burn-in): $N_TER"
echo "==================================================================="

# Restringir la preasignación masiva de memoria de JAX para poder correr en paralelo
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

for n in "${NS[@]}"; do
    
    PIDS_FIG4=()
    
    for g in "${GS[@]}"; do
        SYS_DIR="${BASE_DIR}/results_N${n}_g${g}"
        
        # Validación de existencia del modelo
        if [ ! -d "$SYS_DIR" ]; then
            echo "[$(date +'%H:%M:%S')] [SKIPPED] No se encontró el directorio: $SYS_DIR"
            continue
        fi

        echo "[$(date +'%H:%M:%S')] -> Programando Evaluación (Fig 4) para N=${n}, g=${g}..."
        
        FREE_GPU=$(get_free_gpu)
        echo "           -> GPU $FREE_GPU asignada a N=${n}, g=${g}"
        
        # Ejecución paralela de la evaluación para cada 'g'
        CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 scripts/generate_figure4_dynamic.py \
            --sys_dir "$SYS_DIR" \
            --n_evaluations $N_EVALUATIONS \
            --n_samples $N_SAMPLES \
            --n_ter $N_TER > "${SYS_DIR}/eval_figure4.log" 2>&1 &
            
        PIDS_FIG4+=($!)
        
        # Breve pausa para asegurar que JAX inicialice correctamente en la GPU
        sleep 5
    done
    
    # Sincronización: Esperar a que terminen TODAS las evaluaciones (todos los g) para este N
    if [ ${#PIDS_FIG4[@]} -gt 0 ]; then
        echo "[$(date +'%H:%M:%S')] Esperando a que finalicen las evaluaciones para N=${n}..."
        wait "${PIDS_FIG4[@]}"
        echo "[$(date +'%H:%M:%S')] Evaluaciones para N=${n} completadas."
    fi

    # Generación de la Figura 3 al confirmar que todos los stats.json de este N existen
    echo "[$(date +'%H:%M:%S')] Renderizando Figura 3 (Barrido de Lambda) para N=${n}..."
    CUDA_VISIBLE_DEVICES="" python3 scripts/generate_figure3_dynamic.py \
        --base_dir "$BASE_DIR" \
        --n_elec $n \
        --omega $OMEGA > "${BASE_DIR}/eval_figure3_N${n}.log" 2>&1

    echo "-------------------------------------------------------------------"
done

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] PROTOCOLO DE EVALUACIÓN COMPLETADO"
echo "==================================================================="