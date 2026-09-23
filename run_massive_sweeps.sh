#!/bin/bash

# ==============================================================================
# BARRIDO MASIVO: MALA OPTIMIZADO VS TD3 + GRÁFICAS ESTADÍSTICAS
# ==============================================================================
BASE_DIR_NAME="results_massive"
mkdir -p "$BASE_DIR_NAME"

NS=(2 4 6 10)
GS=(0.01 0.1 0.5 1.0 2.0 3.0 4.0 5.0)
TAUS_MALA=(0.001 0.005 0.01 0.02 0.05)

# Parámetros físicos e hiperparámetros estáticos
OMEGA=0.1
AMP=0.01
N_EVALUATIONS=3 # Número de evaluaciones independientes por modelo
TRAIN_STEPS=50000

GPUS=(0) # Ajusta según tus GPUs disponibles

get_free_gpu(){
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_python=$(nvidia-smi -i $gpu --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)
            if [ -z "$process_python" ]; then
                echo "$gpu"
                return
            fi
        done
        sleep 30 
    done
}

for n in "${NS[@]}"; do
    
    # Calcular splits de spin
    N_UP=$((n / 2 + n % 2))
    N_DN=$((n / 2))

    for g in "${GS[@]}"; do
        SYS_DIR="${BASE_DIR_NAME}/results_N${n}_g${g}"
        mkdir -p "$SYS_DIR"
        
        echo "==================================================================="
        echo "[$(date +'%H:%M:%S')] INICIANDO SISTEMA N=${n}, g=${g}"
        echo "==================================================================="

        # 1. BARRIDO DE MALA (Optimización de Tau)
        MALA_PIDS=()
        for tau in "${TAUS_MALA[@]}"; do
            SEED=$RANDOM
            OUT_DIR="${SYS_DIR}/MALA_tau${tau}"
            FREE_GPU=$(get_free_gpu)
            
            echo "[$(date +'%H:%M:%S')] -> Entrenando MALA (Tau=${tau}) en GPU ${FREE_GPU} con Seed=${SEED}"
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_2.py \
                --method="MALA" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
                --interaction_strength=$g --omega=$OMEGA --tau=$tau --seed=$SEED \
                --train_steps=$TRAIN_STEPS \
                --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
                
            MALA_PIDS+=($!)
            sleep 10
        done
        
        echo "Esperando a que terminen todos los benchmarks MALA..."
        wait "${MALA_PIDS[@]}"

        # 2. ENTRENAMIENTO TD3
        SEED=$RANDOM
        OUT_DIR="${SYS_DIR}/TD3"
        FREE_GPU=$(get_free_gpu)
        
        echo "[$(date +'%H:%M:%S')] -> Entrenando TD3_Full en GPU ${FREE_GPU} con Seed=${SEED}"
        CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_2.py \
            --method="TD3" --actor_type="full" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
            --interaction_strength=$g --omega=$OMEGA --actor_amplitude=$AMP \
            --tau=0.01 --seed=$SEED \
            --coupled_training_cycles=8 --vmc_iterations_per_cycle=12500 --rl_sweeps_per_cycle=2500 \
            --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
            
        TD3_PID=$!
        wait $TD3_PID

        # 3. EVALUACIÓN Y FIGURA 4
        echo "[$(date +'%H:%M:%S')] Generando Figura 4 (Estadística de ${N_EVALUATIONS} corridas)..."
        FREE_GPU=$(get_free_gpu)
        CUDA_VISIBLE_DEVICES="$FREE_GPU" python3 scripts/generate_figure4_dynamic.py \
            --sys_dir "$SYS_DIR" --n_evaluations $N_EVALUATIONS

    done

    # 4. FIGURA 3 AL TERMINAR EL BARRIDO DE 'g' PARA UN 'N'
    echo "[$(date +'%H:%M:%S')] Generando Figura 3 para N=${n}..."
    python3 scripts/generate_figure3_dynamic.py \
        --base_dir "$BASE_DIR_NAME" --n_elec $n --omega $OMEGA

done
echo "[$(date +'%H:%M:%S')] BARRIDO TOTAL COMPLETADO."