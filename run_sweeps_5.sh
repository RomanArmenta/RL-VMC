#!/bin/bash

# Staggered transfer learning

# ==============================================================================
# 0.5 SISTEMA DE COLA PERFECTA (FILE LOCK)
# ==============================================================================
# echo "[$(date +'%H:%M:%S')] Solicitando acceso a la cola de ejecución..."
# exec 9> /tmp/run_sweeps.lock
# flock 9
# echo "[$(date +'%H:%M:%S')] ¡Turno adquirido! Iniciando ejecución."

# echo "==================================================================="
# echo "MODO DE EJECUCIÓN: $MODE"
# echo "CARPETA BASE: $BASE_DIR_NAME"
# echo "==================================================================="

# ==============================================================================
# 1. PARÁMETROS DEL BARRIDO
# ==============================================================================
omega=0.1
amp=0.01
GS=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)
METHODS=("MALA" "TD3")

# ==============================================================================
# 2. PARÁMETROS ESTÁTICOS
# ==============================================================================
N_ELEC=10
N_UP=5
N_DN=5
SOFTENING=0.1
WF_MODEL="slater_jastrow"
ORBITAL_MODEL="backflow_ho"
CORR_MODEL="spin_rbf_jastrow"
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0
BATCH_SIZE=1024
TAU=0.01
GAMMA=0.1
BETA=10.0
KAPPA=0.5
PENALTY_FACTOR=0.1
MAX_PHYS_JUMP=5.0
LR_ANSATZ=5e-5
LR_ACTOR=5e-5
LR_CRITIC=5e-4
TRAIN_STEPS=100_000
MEASURE_STEPS=10_000
COUPLED_CYCLES=8
VMC_ITERS=12_500
RL_SWEEPS=2_500
RL_REPLAY_CAP=10_000
RL_GAMMA_TD3=0.99
E_CLIP_MIN=-500.0
E_CLIP_MAX=500.0
SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-500.0
MAX_CLIP_GRADS=500.0
MIN_CLIP_LOG=-50.0
MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0
SCALE_MAX=1.0
SEED=42
N_PRINT=100
N_LOG=50
STEPS_FOR_BEST=500

N_TER_TRAIN=1000

# ==============================================================================
# 0. PARSEO DE ARGUMENTOS
# ==============================================================================
MODE="train" # Por defecto
BASE_DIR_NAME="1D_N${N_ELEC}_sweep_g_100kTrain_StagTransLearning"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --base_dir_name) BASE_DIR_NAME="$2"; shift ;;
        *) echo "Parámetro desconocido: $1"; exit 1 ;;
    esac
    shift
done

# Parámetros de evaluación
N_SAMPLES_EVALUATIONS=(5000 10000 100000)
N_SUBSAMPLES_EVALUATIONS=(1 10 100 200)
N_TER_EVALUATIONS=(0 500 1000 2000)

MALA_DIR="None" # Forzamos a que Python use la carpeta local de la permutación

# ==============================================================================
# 3. CONFIGURACIÓN DEL ENTORNO
# ==============================================================================
GPUS=(0)
today=$(date +%F)
BASE_RESULTS_DIR="./results_$BASE_DIR_NAME"
BASE_LOGS_DIR="./logs_$BASE_DIR_NAME"

mkdir -p "$BASE_LOGS_DIR" "$BASE_RESULTS_DIR"

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

# ==============================================================================
# 4. ENTRENAMIENTO
# ==============================================================================
if [ "$MODE" == "train" ]; then
    for int_st in "${GS[@]}"; do
        
        # Determinar RUN_ID buscando carpetas maestras
        count=1
        while [ -d "${BASE_RESULTS_DIR}/Perm_g${int_st}_Om${omega}_Amp${amp}_Run${count}" ]; do
            count=$((count+1))
        done
        RUN_ID=$count
        
        # Crear la carpeta maestra para esta permutación
        PERM_DIR="${BASE_RESULTS_DIR}/Perm_g${int_st}_Om${omega}_Amp${amp}_Run${RUN_ID}"
        mkdir -p "$PERM_DIR"

        PIDS=()
        echo "==================================================================="
        echo "[$(date +'%H:%M:%S')] ENTRENANDO PERMUTACIÓN: g=${int_st}, Amp=${amp}, ID=${RUN_ID}"
        
        for method in "${METHODS[@]}"; do
            out_dir="${PERM_DIR}/${method}"
            log_name="train_${method}_g${int_st}_Om${omega}_Amp${amp}_Run${RUN_ID}.txt"
            log_dir="${BASE_LOGS_DIR}/${log_name}"

            FREE_GPU=$(get_free_gpu)
            
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_3.py \
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
                --rl_gamma_td3=$RL_GAMMA_TD3 --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX \
                --sigma_energy_clip=$SIG_E_CLIP --min_clip_grads_ansatz=$MIN_CLIP_GRADS \
                --max_clip_grads_ansatz=$MAX_CLIP_GRADS --min_clip_log_ratio=$MIN_CLIP_LOG \
                --max_clip_log_ratio=$MAX_CLIP_LOG --nan_clip_log_ratio=$MIN_CLIP_LOG \
                --posinf_clip_log_ratio=$MAX_CLIP_LOG --neginf_clip_log_ratio=$MIN_CLIP_LOG \
                --scale_clip_min=$SCALE_MIN --scale_clip_max=$SCALE_MAX --seed=$SEED \
                --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                --n_ter=$N_TER_TRAIN \
                --results_out_dir="$out_dir" > "$log_dir" 2>&1 &
            
            # Capturamos el PID del subproceso en background
            PID=$!
            PIDS+=($PID)
            echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU asignada a $method (PID: $PID)"
            
            # Pausa breve para que jax reserve la memoria y la GPU no parezca "libre" al siguiente
            sleep 10
        done 
        
        echo "[$(date +'%H:%M:%S')] Entrenamientos lanzados. Esperando a que terminen..."
        wait "${PIDS[@]}"
    done
fi

# ==============================================================================
# 5. EVALUACIÓN DE HIPERPARÁMETROS
# ==============================================================================
for N_SAMPLES_EVALUATION in "${N_SAMPLES_EVALUATIONS[@]}"; do
    for N_SUBSAMPLES_EVALUATION in "${N_SUBSAMPLES_EVALUATIONS[@]}"; do
        for N_TER_EVALUATION in "${N_TER_EVALUATIONS[@]}"; do
            
            CURRENT_LOGS_DIR="${BASE_LOGS_DIR}/eval_${N_SAMPLES_EVALUATION}s_${N_SUBSAMPLES_EVALUATION}sub_${N_TER_EVALUATION}ter"
            mkdir -p "$CURRENT_LOGS_DIR"

            for int_st in "${GS[@]}"; do
                
                # Buscar el RUN_ID existente más alto usando sort -V
                folder=$(ls -d ${BASE_RESULTS_DIR}/Perm_g${int_st}_Om${omega}_Amp${amp}_Run* 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$folder" ]; then
                    RUN_ID=$(echo "$folder" | grep -o 'Run[0-9]*' | sed 's/Run//')
                else
                    RUN_ID=1
                fi

                echo "==================================================================="
                echo "[$(date +'%H:%M:%S')] EVALUANDO: g=${int_st}, Samples=${N_SAMPLES_EVALUATION}, Sub=${N_SUBSAMPLES_EVALUATION}, Ter=${N_TER_EVALUATION}"
                
                log_name="eval_g${int_st}_Om${omega}_Amp${amp}_Run${RUN_ID}.txt"

                echo "[$(date +'%H:%M:%S')] -> Ejecutando gráficas de barrido en CPU..."
                CUDA_VISIBLE_DEVICES="" python3 scripts/plot_sweep.py \
                --mala_dir "$MALA_DIR" --base_dir "$BASE_RESULTS_DIR" --g "$int_st" --omega "$omega" \
                --amp "$amp" --run_id "$RUN_ID" > "${CURRENT_LOGS_DIR}/out_sweep_${log_name}" 2>&1

                FREE_GPU=$(get_free_gpu)
                export XLA_PYTHON_CLIENT_PREALLOCATE=false
                
                CUDA_VISIBLE_DEVICES="$FREE_GPU" python3 scripts/evaluate_1D_models.py \
                --mala_dir "$MALA_DIR" --base_dir "$BASE_RESULTS_DIR" \
                --n_samples "$N_SAMPLES_EVALUATION" \
                --subsample "$N_SUBSAMPLES_EVALUATION" \
                --n_ter "$N_TER_EVALUATION" \
                --g "$int_st" --omega "$omega" --amp "$amp" \
                --run_id "$RUN_ID" > "${CURRENT_LOGS_DIR}/out_evaluate_${log_name}" 2>&1 &
                
                EVAL_PID=$!
                echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU asignada a Evaluación (PID: $EVAL_PID)"
                wait $EVAL_PID

            done 

            echo "==================================================================="
            echo "[$(date +'%H:%M:%S')] GENERANDO GRÁFICA Y TABLA PARA ${N_SAMPLES_EVALUATION}s ${N_SUBSAMPLES_EVALUATION}sub ${N_TER_EVALUATION}ter..."
            CUDA_VISIBLE_DEVICES="" python3 scripts/plot_summary.py \
                --logs_dir "$CURRENT_LOGS_DIR" \
                --out_dir "$BASE_RESULTS_DIR" \
                --n_elec "$N_ELEC" \
                --n_samples "$N_SAMPLES_EVALUATION" \
                --subsample "$N_SUBSAMPLES_EVALUATION" \
                --n_ter "$N_TER_EVALUATION"
                
        done
    done 
done 

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] BARRIDO COMPLETADO EXITOSAMENTE."