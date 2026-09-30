#!/bin/bash

# ==============================================================================
# PIPELINE DE HYPERPARAMETER SWEEP RL-VMC: RÉGIMEN DE FUERTE CORRELACIÓN
# Sistema Objetivo: Cristalización de Wigner (N=10, g=5.0)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. DIRECTORIOS Y RECURSOS
# ------------------------------------------------------------------------------
BASE_DIR="results_hyperparam_sweep_N10_g5.0"
mkdir -p "$BASE_DIR"

GPUS=(0 1 2) # GPUs disponibles en el servidor local
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 2. DEFINICIÓN DEL SISTEMA FÍSICO Y ANSATZ (Fijos)
# ------------------------------------------------------------------------------
N_ELEC=10
N_UP=5
N_DN=5
OMEGA=0.1
G_VAL=5.0
SOFTENING=0.1

WF_MODEL="slater_jastrow"
ORBITAL_MODEL="backflow_ho"
CORR_MODEL="spin_rbf_jastrow"
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0

# ------------------------------------------------------------------------------
# 3. ESPACIO DE BÚSQUEDA DEL GRID (Hyperparameters RL)
# ------------------------------------------------------------------------------
# a) Amplitud máxima de la red (Radio de la "Trust Region" espacial)
ACTOR_AMPLITUDES=(0.05 0.15)

# b) Coeficiente lambda (Penalización entrópica de la recompensa CDLB)
PENALTY_FACTORS=(0.1 0.3)

# c) Escala máxima de alteración de la covarianza térmica
SIGMA_SCALES=(0.2 0.5)

# d) Factor de restricción para el radio del salto físico evaluable
PHYS_JUMPS=(3.0 5.0)

# ------------------------------------------------------------------------------
# 4. HIPERPARÁMETROS DE ENTRENAMIENTO Y MUESTREO (Estáticos)
# ------------------------------------------------------------------------------
BATCH_SIZE=1024
LR_ANSATZ=1e-5
LR_ACTOR=1e-5
LR_CRITIC=1e-4

TRAIN_STEPS=100000
MEASURE_STEPS=5000
N_TER_TRAIN=2000

COUPLED_CYCLES=8
VMC_ITERS=12500
RL_SWEEPS=2500
RL_REPLAY_CAP=10000
RL_GAMMA_TD3=0.99

# Estabilidad RL y Clipping Numérico
GAMMA_REWARD=0.1
BETA=10.0
KAPPA=0.5
FORCE_SCALE=5.0
N_SCALE=10.0
R_SCALE=5.0
MAX_DRIFT_CLIP=10.0
ACTION_REV_LIMIT=1.0

# Clipping de la Cadena de Markov y Energía
E_CLIP_MIN=-500.0
E_CLIP_MAX=500.0
SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-500.0
MAX_CLIP_GRADS=500.0
MIN_CLIP_LOG=-50.0
MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0
SCALE_MAX=1.0

N_PRINT=100
N_LOG=50
STEPS_FOR_BEST=500

# ------------------------------------------------------------------------------
# FUNCIONES AUXILIARES
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

echo "[$(date +'%H:%M:%S')] Solicitando acceso a la GPU (Flock)..."
exec 9> /tmp/gpu_rl_vmc_sweep.lock
flock 9
echo "[$(date +'%H:%M:%S')] Turno adquirido. Iniciando Sweep Masivo para N=${N_ELEC}, g=${G_VAL}"

# ==============================================================================
# FASE 1: OBTENCIÓN DE LÍNEA BASE (Optimización de Tau en MALA Clásico)
# ==============================================================================
TAUS_MALA=(0.001 0.01 0.1)
MALA_PIDS=()
MALA_BASE_DIR="${BASE_DIR}/MALA_Baselines"
mkdir -p "$MALA_BASE_DIR"

echo "-------------------------------------------------------------------"
echo " FASE 1: Búsqueda del Tau óptimo (Langevin puro)"
echo "-------------------------------------------------------------------"

for tau in "${TAUS_MALA[@]}"; do
    OUT_DIR="${MALA_BASE_DIR}/MALA_tau${tau}"
    FREE_GPU=$(get_free_gpu)
    
    echo "[$(date +'%H:%M:%S')] Ejecutando MALA (Tau=${tau}) en GPU ${FREE_GPU}..."
    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
        --method="MALA" --n_elec=$N_ELEC --n_up=$N_UP --n_dn=$N_DN \
        --omega=$OMEGA --interaction_strength=$G_VAL --softening=$SOFTENING \
        --wf_model=$WF_MODEL --orbital_model=$ORBITAL_MODEL --correlation_model=$CORR_MODEL \
        --backflow_components=$BF_COMPONENTS --backflow_r_max=$BF_RMAX --backflow_width=$BF_WIDTH --backflow_envelope_length=$BF_ENV_LEN \
        --batch_size=$BATCH_SIZE --tau=$tau --lr_ansatz=$LR_ANSATZ \
        --train_steps=$TRAIN_STEPS --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
        --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
        --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
        --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
        --max_phys_jump_factor=${PHYS_JUMPS[0]} \
        --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
        
    MALA_PIDS+=($!)
    sleep 15
done

echo "Esperando finalización de benchmarks de MALA..."
wait "${MALA_PIDS[@]}"

echo "[$(date +'%H:%M:%S')] Evaluando el mejor MALA de la línea base..."
BEST_TAU=$(python3 scripts/get_best_mala_tau.py --sys_dir "$MALA_BASE_DIR")
echo "-> Mejor Tau establecido para RL-MALA: $BEST_TAU"

# ==============================================================================
# FASE 2: GRID SEARCH SOBRE TD3 ACTOR (RL-MALA)
# ==============================================================================
echo "-------------------------------------------------------------------"
echo " FASE 2: Hyperparameter Sweep (RL-MALA)"
echo "-------------------------------------------------------------------"

for amp in "${ACTOR_AMPLITUDES[@]}"; do
    for pen in "${PENALTY_FACTORS[@]}"; do
        for sig in "${SIGMA_SCALES[@]}"; do
            for jump in "${PHYS_JUMPS[@]}"; do
                
                EXP_NAME="TD3_amp${amp}_pen${pen}_sig${sig}_jmp${jump}"
                OUT_DIR="${BASE_DIR}/${EXP_NAME}"
                mkdir -p "$OUT_DIR"
                FREE_GPU=$(get_free_gpu)
                
                echo "[$(date +'%H:%M:%S')] Iniciando $EXP_NAME en GPU ${FREE_GPU}..."
                
                CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
                    --method="TD3" --n_elec=$N_ELEC --n_up=$N_UP --n_dn=$N_DN \
                    --omega=$OMEGA --interaction_strength=$G_VAL --softening=$SOFTENING \
                    --wf_model=$WF_MODEL --orbital_model=$ORBITAL_MODEL --correlation_model=$CORR_MODEL \
                    --backflow_components=$BF_COMPONENTS --backflow_r_max=$BF_RMAX --backflow_width=$BF_WIDTH --backflow_envelope_length=$BF_ENV_LEN \
                    --batch_size=$BATCH_SIZE --tau=$BEST_TAU --actor_amplitude=$amp \
                    --penalty_factor=$pen --sigma_action_scale=$sig --max_phys_jump_factor=$jump \
                    --gamma=$GAMMA_REWARD --beta=$BETA --kappa=$KAPPA \
                    --lr_ansatz=$LR_ANSATZ --lr_actor=$LR_ACTOR --lr_critic=$LR_CRITIC \
                    --coupled_training_cycles=$COUPLED_CYCLES --vmc_iterations_per_cycle=$VMC_ITERS \
                    --rl_sweeps_per_cycle=$RL_SWEEPS --rl_replay_capacity=$RL_REPLAY_CAP --rl_gamma_td3=$RL_GAMMA_TD3 \
                    --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
                    --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
                    --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
                    --min_clip_log_ratio=$MIN_CLIP_LOG --max_clip_log_ratio=$MAX_CLIP_LOG \
                    --nan_clip_log_ratio=$MIN_CLIP_LOG --posinf_clip_log_ratio=$MAX_CLIP_LOG --neginf_clip_log_ratio=$MIN_CLIP_LOG \
                    --scale_clip_min=$SCALE_MIN --scale_clip_max=$SCALE_MAX \
                    --force_scale=$FORCE_SCALE --n_scale=$N_SCALE --r_scale=$R_SCALE \
                    --max_drift_clip=$MAX_DRIFT_CLIP \
                    --action_rev_limit=$ACTION_REV_LIMIT \
                    --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                    --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
                
                # Desplazamiento de tiempo leve para evitar colisiones de memoria en inicialización JAX
                sleep 20
                
            done
        done
    done
done

echo "[$(date +'%H:%M:%S')] Todos los experimentos en cola. Monitorea mediante nvidia-smi."