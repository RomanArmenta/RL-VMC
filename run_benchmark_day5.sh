#!/bin/bash
BASE_DIR="results_Ablation_N10_g5_MyReward_0tertrain"
BASE_LOG="logs_Ablation_N10_g5_MyReward_0tertrain"
mkdir -p $BASE_DIR $BASE_LOG

N_ELEC=10; N_UP=5; N_DN=5; G=5.0

ACTORS=("full" "scale_only" "flat")

echo "Entrenando MALA baseline..."
CUDA_VISIBLE_DEVICES="0" python3 -u scripts/run_experiment_1D_2.py \
    --method="MALA" --n_elec=$N_ELEC --n_up=$N_UP --n_dn=$N_DN --interaction_strength=$G \
    --train_steps=100000 --n_ter=0 --results_out_dir="$BASE_DIR/MALA" > "$BASE_LOG/train_MALA.txt" 2>&1

for actor in "${ACTORS[@]}"; do
    echo "Entrenando TD3 variante: $actor..."
    CUDA_VISIBLE_DEVICES="0" python3 -u scripts/run_experiment_1D_2.py \
        --method="TD3" --actor_type="$actor" --n_elec=$N_ELEC --n_up=$N_UP --n_dn=$N_DN \
        --interaction_strength=$G \
        --coupled_training_cycles=8 \
        --vmc_iterations_per_cycle=12500 \
        --rl_sweeps_per_cycle=2500 --n_ter=0 \
        --results_out_dir="$BASE_DIR/TD3_$actor" > "$BASE_LOG/train_TD3_$actor.txt" 2>&1
done

echo "¡Entrenamientos de Ablación Finalizados!"