#!/bin/bash

# ==============================================================================
# PIPELINE MASIVO DE RL-VMC: OPTIMIZACIÓN CRUZADA Y EVALUACIÓN DINÁMICA
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. BARRIDOS Y PARÁMETROS FÍSICOS
# ------------------------------------------------------------------------------
BASE_DIR="results_massive_2"
mkdir -p "$BASE_DIR"

#NS=(2 4 6 10)
NS=(4 10)
#GS=(0.01 0.1 0.5 1.0 2.0 3.0 4.0 5.0)
GS=(0.1 1.0 3.0 5.0)
TAUS_MALA=(0.001 0.005 0.01 0.02 0.05)
GPUS=(0) # GPUs disponibles en tu servidor

# ------------------------------------------------------------------------------
# 2. HIPERPARÁMETROS DEL EXPERIMENTO (Totalmente Configurables)
# ------------------------------------------------------------------------------
# Sistema
OMEGA=0.1
SOFTENING=0.1
WF_MODEL="slater_jastrow"
ORBITAL_MODEL="backflow_ho"
CORR_MODEL="spin_rbf_jastrow"

# Redes Ansatz (Backflow)
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0

# Entrenamiento VMC / RL
BATCH_SIZE=1024
ACTOR_AMPLITUDE=0.01
LR_ANSATZ=5e-5
LR_ACTOR=5e-5
LR_CRITIC=5e-4
TRAIN_STEPS=100000
MEASURE_STEPS=5000
N_TER_TRAIN=0

# Ciclos Acoplados TD3 (Staggered Transfer)
COUPLED_CYCLES=8
VMC_ITERS=12500
RL_SWEEPS=2500
RL_REPLAY_CAP=10000
RL_GAMMA_TD3=0.99

# Parámetros de la Recompensa CDLB
GAMMA_REWARD=0.1
BETA=10.0
KAPPA=0.5
PENALTY_FACTOR=0.1
MAX_PHYS_JUMP=5.0

# Clipping y Estabilidad
E_CLIP_MIN=-500.0
E_CLIP_MAX=500.0
SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-500.0
MAX_CLIP_GRADS=500.0
MIN_CLIP_LOG=-50.0
MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0
SCALE_MAX=1.0

# Logging e I/O
N_PRINT=100
N_LOG=50
STEPS_FOR_BEST=500
N_EVALUATIONS_FIG=5 # Evaluaciones para error estadístico en la Fig 4

# ------------------------------------------------------------------------------
# 3. FUNCIONES AUXILIARES
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
        sleep 20 
    done
}

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 4. BUCLE MAESTRO DE EJECUCIÓN
# ------------------------------------------------------------------------------
for n in "${NS[@]}"; do
    N_UP=$((n / 2 + n % 2))
    N_DN=$((n / 2))

    for g in "${GS[@]}"; do
        SYS_DIR="${BASE_DIR}/results_N${n}_g${g}"
        mkdir -p "$SYS_DIR"
        
        echo "==================================================================="
        echo "[$(date +'%H:%M:%S')] INICIANDO SISTEMA N=${n}, g=${g}"
        echo "==================================================================="

        # FASE 1: BARRIDO DE MALA (Benchmarking)
        MALA_PIDS=()
        for tau in "${TAUS_MALA[@]}"; do
            SEED=$RANDOM
            OUT_DIR="${SYS_DIR}/MALA_tau${tau}"
            FREE_GPU=$(get_free_gpu)
            
            echo "[$(date +'%H:%M:%S')] -> MALA (Tau=${tau}) en GPU ${FREE_GPU} [Seed=${SEED}]"
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_3.py \
                --method="MALA" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
                --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
                --wf_model=$WF_MODEL --orbital_model=$ORBITAL_MODEL --correlation_model=$CORR_MODEL \
                --backflow_components=$BF_COMPONENTS --backflow_r_max=$BF_RMAX --backflow_width=$BF_WIDTH --backflow_envelope_length=$BF_ENV_LEN \
                --batch_size=$BATCH_SIZE --tau=$tau --lr_ansatz=$LR_ANSATZ \
                --train_steps=$TRAIN_STEPS --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
                --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
                --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
                --seed=$SEED --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
                
            MALA_PIDS+=($!)
            sleep 15
        done
        
        echo "Esperando a que terminen todos los benchmarks MALA..."
        wait "${MALA_PIDS[@]}"

        # FASE 2: EXTRACCIÓN DEL MEJOR TAU MALA
        echo "[$(date +'%H:%M:%S')] -> Evaluando el mejor MALA para asignarlo a TD3..."
        BEST_TAU=$(python3 scripts/get_best_mala_tau.py --sys_dir "$SYS_DIR")
        echo "           -> Mejor Tau encontrado: $BEST_TAU"

        # FASE 3: ENTRENAMIENTO TD3 (Heredando el Tau óptimo)
        SEED=$RANDOM
        OUT_DIR="${SYS_DIR}/TD3"
        FREE_GPU=$(get_free_gpu)
        
        echo "[$(date +'%H:%M:%S')] -> TD3_Full (Tau=${BEST_TAU}) en GPU ${FREE_GPU} [Seed=${SEED}]"
        CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_3.py \
            --method="TD3" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
            --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
            --wf_model=$WF_MODEL --orbital_model=$ORBITAL_MODEL --correlation_model=$CORR_MODEL \
            --backflow_components=$BF_COMPONENTS --backflow_r_max=$BF_RMAX --backflow_width=$BF_WIDTH --backflow_envelope_length=$BF_ENV_LEN \
            --batch_size=$BATCH_SIZE --tau=$BEST_TAU --actor_amplitude=$ACTOR_AMPLITUDE \
            --gamma=$GAMMA_REWARD --beta=$BETA --kappa=$KAPPA --penalty_factor=$PENALTY_FACTOR --max_phys_jump_std=$MAX_PHYS_JUMP \
            --lr_ansatz=$LR_ANSATZ --lr_actor=$LR_ACTOR --lr_critic=$LR_CRITIC \
            --coupled_training_cycles=$COUPLED_CYCLES --vmc_iterations_per_cycle=$VMC_ITERS \
            --rl_sweeps_per_cycle=$RL_SWEEPS --rl_replay_capacity=$RL_REPLAY_CAP --rl_gamma_td3=$RL_GAMMA_TD3 \
            --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
            --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
            --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
            --min_clip_log_ratio=$MIN_CLIP_LOG --max_clip_log_ratio=$MAX_CLIP_LOG \
            --nan_clip_log_ratio=$MIN_CLIP_LOG --posinf_clip_log_ratio=$MAX_CLIP_LOG --neginf_clip_log_ratio=$MIN_CLIP_LOG \
            --scale_clip_min=$SCALE_MIN --scale_clip_max=$SCALE_MAX \
            --seed=$SEED --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
            --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
            
        TD3_PID=$!
        wait $TD3_PID

        # FASE 4: PLOTEO DE MÉTRICAS (DASHBOARD)
        echo "[$(date +'%H:%M:%S')] Generando Dashboard de Entrenamiento..."
        python3 scripts/plot_training_metrics.py --sys_dir "$SYS_DIR"

        # FASE 5: EVALUACIÓN RIGUROSA Y FIGURA 4
        echo "[$(date +'%H:%M:%S')] Generando Figura 4 (Estadística de ${N_EVALUATIONS_FIG} corridas)..."
        FREE_GPU=$(get_free_gpu)
        CUDA_VISIBLE_DEVICES="$FREE_GPU" python3 scripts/generate_figure4_dynamic.py \
            --sys_dir "$SYS_DIR" --n_evaluations $N_EVALUATIONS_FIG

    done

    # FASE 6: FIGURA 3 (Al terminar el barrido de 'g' para un 'N' específico)
    echo "[$(date +'%H:%M:%S')] Generando Figura 3 para N=${n}..."
    python3 scripts/generate_figure3_dynamic.py --base_dir "$BASE_DIR" --n_elec $n --omega $OMEGA

done
echo "==================================================================="
echo "[$(date +'%H:%M:%S')] BARRIDO MASIVO TOTAL COMPLETADO."
echo "==================================================================="