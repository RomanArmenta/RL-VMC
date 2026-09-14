#!/bin/bash

# ==============================================================================
# BARRIDO DE BENCHMARK FÍSICO (g=0)
# ==============================================================================
MODE="train"
BASE_DIR_NAME="Day2_1D_benchmark_3"

# Solo evaluamos g=0 (gas ideal de fermiones)
GS=(0.0)
omega=0.1
amp=0.01
METHODS=("MALA" "TD3")

# Pares de configuraciones: "N_ELEC N_UP"
CONFIGS=("4 1" "4 2" "6 3" "10 3" "10 5")

# ==============================================================================
# PARÁMETROS ESTÁTICOS Y DE PROTECCIÓN (NODE-CROSSING)
# ==============================================================================
SOFTENING=0.1
WF_MODEL="slater_jastrow"
ORBITAL_MODEL="backflow_ho"
CORR_MODEL="spin_rbf_jastrow"
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0
BATCH_SIZE=1024
GAMMA=0.1
BETA=10.0
KAPPA=0.5
PENALTY_FACTOR=0.1

# Ajustes críticos para convergencia de N=10
TAU=0.005                 # Paso de tiempo reducido
MAX_PHYS_JUMP=2.0         # Evita que atraviesen los nodos violentamente
MIN_CLIP_GRADS=-20.0      # Evita explosiones de fuerza cuántica
MAX_CLIP_GRADS=20.0
LR_ANSATZ=1e-4            # Learning rate suave para el gas ideal

LR_ACTOR=5e-5
LR_CRITIC=5e-4
TRAIN_STEPS=150000
MEASURE_STEPS=5000
COUPLED_CYCLES=8
VMC_ITERS=12500
RL_SWEEPS=2500
RL_REPLAY_CAP=10000
N_TER_TRAIN=1000

# ==============================================================================
# NUEVO: PARÁMETROS DE EVALUACIÓN FÍSICA
# ==============================================================================
EVAL_SAMPLES=100000       # Muestras totales a evaluar por el sampler
EVAL_SUBSAMPLES=10        # Guardar 1 de cada 10 para reducir autocorrelación
EVAL_TER=5000             # Pasos de termalización pura (burn-in) antes de medir

BASE_RESULTS_DIR="./results_$BASE_DIR_NAME"
BASE_LOGS_DIR="./logs_$BASE_DIR_NAME"
mkdir -p "$BASE_LOGS_DIR" "$BASE_RESULTS_DIR"

GPUS=(0)
get_free_gpu(){
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_python=$(nvidia-smi -i $gpu --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)
            if [ -z "$process_python" ]; then echo "$gpu"; return; fi
        done
        sleep 30 
    done
}

if [ "$MODE" == "train" ]; then
    for conf in "${CONFIGS[@]}"; do
        read -r N_ELEC N_UP <<< "$conf"
        N_DN=$((N_ELEC - N_UP))
        int_st=0.0
        
        PERM_DIR="${BASE_RESULTS_DIR}/Perm_Ne${N_ELEC}_Nup${N_UP}_g${int_st}_Om${omega}"
        mkdir -p "$PERM_DIR"

        PIDS=()
        echo "==================================================================="
        echo "[$(date +'%H:%M:%S')] ENTRENANDO BENCHMARK: N=${N_ELEC}, g=${int_st}"
        
        for method in "${METHODS[@]}"; do
            out_dir="${PERM_DIR}/${method}"
            log_name="train_${method}_Ne${N_ELEC}_Nup${N_UP}_g${int_st}.txt"
            
            FREE_GPU=$(get_free_gpu)
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_2.py \
                --method="$method" --n_elec=$N_ELEC --n_up=$N_UP --n_dn=$N_DN --omega=$omega \
                --interaction_strength=$int_st --softening=$SOFTENING --wf_model="$WF_MODEL" \
                --orbital_model="$ORBITAL_MODEL" --correlation_model="$CORR_MODEL" \
                --backflow_components=$BF_COMPONENTS --backflow_r_max=$BF_RMAX --backflow_width=$BF_WIDTH \
                --backflow_envelope_length=$BF_ENV_LEN --batch_size=$BATCH_SIZE --tau=$TAU \
                --actor_amplitude=$amp --gamma=$GAMMA --beta=$BETA --kappa=$KAPPA --penalty_factor=$PENALTY_FACTOR \
                --max_phys_jump_std=$MAX_PHYS_JUMP --lr_ansatz=$LR_ANSATZ --lr_actor=$LR_ACTOR \
                --lr_critic=$LR_CRITIC --train_steps=$TRAIN_STEPS --measure_steps=$MEASURE_STEPS \
                --coupled_training_cycles=$COUPLED_CYCLES --vmc_iterations_per_cycle=$VMC_ITERS \
                --rl_sweeps_per_cycle=$RL_SWEEPS --rl_replay_capacity=$RL_REPLAY_CAP \
                --min_clip_grads_ansatz=$MIN_CLIP_GRADS \
                --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
                --n_ter=$N_TER_TRAIN \
                --results_out_dir="$out_dir" > "${BASE_LOGS_DIR}/${log_name}" 2>&1 &
            
            PID=$!
            PIDS+=($PID)
            echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU asignada a $method (PID: $PID)"
            sleep 10
        done 
        wait "${PIDS[@]}"
    done
fi

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] EVALUANDO FÍSICA Y GENERANDO GRÁFICAS DE CREDIBILIDAD..."
for conf in "${CONFIGS[@]}"; do
    read -r N_ELEC N_UP <<< "$conf"
    log_name="validate_Ne${N_ELEC}_Nup${N_UP}.txt"
    FREE_GPU=$(get_free_gpu)
    
    # Pasamos las configuraciones clave de protección al validador para que no explote evaluando
    CUDA_VISIBLE_DEVICES="$FREE_GPU" python3 scripts/Day2_validate_physics.py \
        --base_dir "$BASE_RESULTS_DIR" \
        --n_elec "$N_ELEC" \
        --n_up "$N_UP" \
        --omega "$omega" \
        --g "0.0" \
        --tau "$TAU" \
        --min_clip_grads "$MIN_CLIP_GRADS" \
        --max_clip_grads "$MAX_CLIP_GRADS" \
        --n_samples "$EVAL_SAMPLES" \
        --n_subsamples "$EVAL_SUBSAMPLES" \
        --n_ter_eval "$EVAL_TER" > "${BASE_LOGS_DIR}/${log_name}" 2>&1 &
    
    # Evaluar uno por uno de manera segura para no desbordar memoria VRAM
    wait $!
done