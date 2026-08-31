#!/bin/bash

# Directorios necesarios
mkdir -p logs results params_tests

GPUS=(2)
today=$(date +%F)

echo "========================================================"
echo " Iniciando Administrador de Colas VMC "
echo " Archivos de configuración encontrados: $(ls params/*.conf 2>/dev/null | wc -l)"
echo " GPUs detectadas: ${#GPUS[@]}"
echo "========================================================"
echo "Test: No inicializamos en cero al actor sino con ruido para romper simetrías y así"

# Función para encontrar una GPU libre
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

# Iterar sobre todos los archivos .conf en la carpeta params/
for param_file in params_tests/*.conf; do

    if [ ! -f "$param_file" ]; then
        echo "No se encontraron archivos .conf en la carpeta params/"
        break
    fi

    # Cargar las variables del archivo
    source "$param_file"

    # Preparar directorios y nombres basados en el archivo
    experiment_name="$(basename "$param_file" .conf)"
    
    # El out_dir ahora incluye el experiment_name para diferenciar sd_sj, md_sjbf, etc.
    out_dir="./results/${experiment_name}_${today}_NewActor_test"
    log_name="${experiment_name}_${today}_NewActor_test.txt"

    # Esperar hasta obtener una GPU libre
    FREE_GPU=$(get_free_gpu)

    echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU libre. Ejecutando experimento: $experiment_name"

    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment.py \
            --system="$SYSTEM" \
            --config_name="$CONFIG_NAME" \
            --n_determinants=$N_DETERMINANTS \
            --backflow_transform="$BACKFLOW_TRANSFORM" \
            --batch_size=$BATCH_SIZE \
            --tau=$TAU \
            --lr_ansatz=$LR_ANSATZ \
            --lr_actor=$LR_ACTOR \
            --train_steps=$TRAIN_STEPS \
            --warmup_steps=$WARMUP_STEPS \
            --measure_steps=$MEASURE_STEPS \
            --gamma=$GAMMA \
            --beta=$BETA \
            --seed=$SEED \
            --trans_steps_sched_ansatz=$TRANS_STEPS_SCHED_ANSATZ \
            --decay_rate_ansatz=$DECAY_RATE_ANSATZ \
            --trans_steps_sched_actor=$TRANS_STEPS_SCHED_ACTOR \
            --decay_rate_actor=$DECAY_RATE_ACTOR \
            --min_clip_grads_ansatz=$MIN_CLIP_GRADS_ANSATZ \
            --max_clip_grads_ansatz=$MAX_CLIP_GRADS_ANSATZ \
            --n_print=$N_PRINT \
            --n_log=$N_LOG \
            --steps_for_best=$STEPS_FOR_BEST \
            --min_clip_log_ratio=$MIN_CLIP_LOG_RATIO \
            --max_clip_log_ratio=$MAX_CLIP_LOG_RATIO \
            --nan_clip_log_ratio=$NAN_CLIP_LOG_RATIO \
            --neginf_clip_log_ratio=$NEGINF_CLIP_LOG_RATIO \
            --posinf_clip_log_ratio=$POSINF_CLIP_LOG_RATIO \
            --scale_clip_min=$SCALE_CLIP_MIN \
            --scale_clip_max=$SCALE_CLIP_MAX \
            --sigma_energy_clip=$SIGMA_ENERGY_CLIP \
            --energy_clip_min=$ENERGY_CLIP_MIN \
            --energy_clip_max=$ENERGY_CLIP_MAX \
            --actor_amplitude=$ACTOR_AMPLITUDE \
            --results_out_dir="$out_dir" \
            --penalty_factor=$PENALTY_FACTOR \
            --actor_warmup_steps=$ACTOR_WARMUP_STEPS \
            --max_phys_jump_std=$MAX_PHYS_JUMP_STD \
            --kappa=$KAPPA \
            --actor_name=$ACTOR_NAME \
            > "./logs/$log_name" 2>&1 &

    sleep 10
done

echo "========================================================"
echo "Todos los experimentos han sido enviados a la cola."
wait 
echo "[$(date +'%H:%M:%S')] Todas las simulaciones han finalizado."