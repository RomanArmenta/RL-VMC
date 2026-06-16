#!/bin/bash

# directory and resourses
mkdir -p logs
GPUS=(0 1 2)
SYSTEMS=("H2" "LiH" "Be" "B" "Ne")

echo "=========================="
echo " Beggining VMC experiments "
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
        # sleep 300 secongs and asks again
        sleep 300
    done
}

for system in "${SYSTEMS[@]}"; do

    # the script will pause here until get_free_gpu finds space

    FREE_GPU=$(get_free_gpu)

    echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU is free. Starting training for system: $system"

    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u run_experiment.py --system "$system" > "./logs/${system}_log.txt" 2>&1 &

    sleep 5

done

echo "=========================="
echo "All experiments are in queue"

# this command waits for all the "&" commands are done
wait 

echo "[$(date +'%H:%M:%S')] ¡Simulaciones completadas!"