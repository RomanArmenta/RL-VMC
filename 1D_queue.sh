#!/bin/bash

OMEGAS=(0.1)
ACTOR_AMPLITUDES=(0.01 0.05 0.1)
GS=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)
METHODS=("TD3" "REINFORCE")

# Otros parámetros fijos
N_ELEC=4
N_UP=2
N_DN=2
TAU=0.01
TRAIN_STEPS=20_000
COUPLED_CYCLES=20
VMC_ITERS=1_000 # COUPLED_CYCLES * VMC_ITERS = TRAIN_STEPS para ser justos
RL_SWEEPS=1_000

GPUS=(0)
today=$(date +%F)
BASE_RESULTS_DIR="./results_1D"

mkdir -p logs "$BASE_RESULTS_DIR"

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

for omega in "${OMEGAS[@]}"; do
    for amp in "${ACTOR_AMPLITUDES[@]}"; do
        for int_st in "${GS[@]}"; do
            for method in "${METHODS[@]}"; do
                experiment_name="1D__${method}_g${int_st}_Om${omega}_Amp${amp}"
                
                base_out_dir="${BASE_RESULTS_DIR}/${today}_${experiment_name}"
                count=1
                out_dir="${base_out_dir}_${count}"
                
                while [ -d "$out_dir" ]; do
                    count=$((count+1))
                    out_dir="${base_out_dir}_${count}"
                done

                log_name="${today}_${experiment_name}_${count}.txt"

                FREE_GPU=$(get_free_gpu)

                echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU libre. Ejecutando: $experiment_name"

                CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D.py \
                        --rl_method="$method" \
                        --n_elec=$N_ELEC \
                        --n_up=$N_UP \
                        --n_dn=$N_DN \
                        --omega=$omega \
                        --actor_amplitude=$amp \
                        --interaction_strength=$int_st \
                        --tau=$TAU \
                        --train_steps=$TRAIN_STEPS \
                        --coupled_training_cycles=$COUPLED_CYCLES \
                        --vmc_iterations_per_cycle=$VMC_ITERS \
                        --rl_sweeps_per_cycle=$RL_SWEEPS \
                        --results_out_dir="$out_dir" \
                        > "./logs/$log_name" 2>&1 &

                sleep 10
            done # Cierra el for method
        done # Cierra el for int_st
    done # Cierra el for amp
done # Cierra el for omega

wait
echo "[$(date +'%H:%M:%S')] Todas las combinaciones han finalizado."