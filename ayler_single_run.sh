#!/bin/bash

mkdir -p logs

GPU=1
SYSTEM="LiH"
CONFIG_NAME="default"
BATCH_SIZE=2048
TAU=0.05
LR_ANSATZ=5e-4
LR_ACTOR=1e-4
TRAIN_STEPS=5_000
WARMUP_STEPS=2_000
MEASURE_STEPS=1_000
GAMMA=0.1
BETA=10.0
SEED=42
TRANS_STEPS_SCHED_ANSATZ=2_000
DECAY_RATE_ANSATZ=0.1
TRANS_STEPS_SCHED_ACTOR=2_000
DECAY_RATE_ACTOR=0.1
MIN_CLIP_GRADS_ANSATZ=-20.
MAX_CLIP_GRADS_ANSATZ=20.
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
ACTOR_AMPLITUDE=0.05


echo "================================================"
echo " Launching benchmark "
echo " System: $SYSTEM "
echo "================================================"

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
            --actor_amplitude=$ACTOR_AMPLITUDE \
            > "./logs/${system}_log.txt" 2>&1 &
    
echo "Working in Background. Check logs in: logs/${SYSTEM}_log.txt"