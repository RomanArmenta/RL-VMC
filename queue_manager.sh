#!/bin/bash

# directory and resources
mkdir -p logs

GPUS=(0 1 2)
# SYSTEMS=("H2" "LiH" "Be" "B" "Ne") 
#SYSTEMS=("H2" "He" "LiH") # little thinguis (tau = 0.01)
#SYSTEMS=("Be" "B" "C" "Ne") # larger thinguis  (tau = 0.002)
SYSTEMS=("cyclobutadiene_square")

# ==========================================
# Hyperparameters (Globales para la cola)
# ==========================================
CONFIG_NAME="default"
BATCH_SIZE=1024
TAU=0.005
LR_ANSATZ=5e-4
LR_ACTOR=1e-4
TRAIN_STEPS=15000
WARMUP_STEPS=2000
MEASURE_STEPS=5000
GAMMA=0.1
BETA=10.0
SEED=42
TRANS_STEPS_SCHED_ANSATZ=2000
DECAY_RATE_ANSATZ=0.5
TRANS_STEPS_SCHED_ACTOR=2000
DECAY_RATE_ACTOR=0.5
MIN_CLIP_GRADS_ANSATZ=-200.
MAX_CLIP_GRADS_ANSATZ=200.
N_PRINT=500
N_LOG=50
STEPS_FOR_BEST=500
MIN_CLIP_LOG_RATIO=-20.
MAX_CLIP_LOG_RATIO=20.
NAN_CLIP_LOG_RATIO=-20.
NEGINF_CLIP_LOG_RATIO=-20.
POSINF_CLIP_LOG_RATIO=20.
SCALE_CLIP_MIN=-5.0
SCALE_CLIP_MAX=1.0
SIGMA_ENERGY_CLIP=5.0
ENERGY_CLIP_MIN=-50.
ENERGY_CLIP_MAX=0.

echo "=========================="
echo " Beginning VMC experiments "
echo " Systems to compute: ${#SYSTEMS[@]}"
echo " Detected GPUs: ${#GPUS[@]}"
echo "=========================="

# find a free gpu
get_free_gpu(){
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_python=$(nvidia-smi -i $gpu --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)

            # if variable empty (-z) means that there is no python running in that gpu
            if [ -z "$process_python" ]; then
                echo "$gpu" # we return the GPUs ID
                return
            fi
        done

        # if reach this point, all GPUS are occupied
        # sleep 30 seconds and ask again (mejor 30s que 300s para no perder tiempo de GPU)
        sleep 30 
    done
}

for system in "${SYSTEMS[@]}"; do

    # the script will pause here until get_free_gpu finds space
    FREE_GPU=$(get_free_gpu)

    echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU is free. Starting training for system: $system"

    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment.py \
            --system="$system" \
            --config_name="$CONFIG_NAME" \
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
            > "./logs/${system}_log.txt" 2>&1 &

    sleep 5

done

echo "=========================="
echo "All experiments are in queue"

# this command waits for all the "&" commands are done
wait 

echo "[$(date +'%H:%M:%S')] simulations complete."