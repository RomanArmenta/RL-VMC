#!/bin/bash

# ==============================================================================
# 0. PARSEO DE ARGUMENTOS
# ==============================================================================
MODE="train" # Por defecto
BASE_DIR_NAME="1D_N4_sweep_g"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift ;;
        --base_dir_name) BASE_DIR_NAME="$2"; shift ;;
        *) echo "Parámetro desconocido: $1"; exit 1 ;;
    esac
    shift
done

echo "==================================================================="
echo "MODO DE EJECUCIÓN: $MODE"
echo "CARPETA BASE: $BASE_DIR_NAME"
echo "==================================================================="

# ==============================================================================
# 1. PARÁMETROS DEL BARRIDO (PERMUTACIONES)
# ==============================================================================
omega=0.1
amp=0.01
GS=(0.1 0.5 1.0 2.0 3.0 4.0 5.0)
METHODS=("MALA" "TD3")

# ==============================================================================
# 2. PARÁMETROS ESTÁTICOS
# ==============================================================================
N_ELEC=4
N_UP=2
N_DN=2
SOFTENING=0.1
WF_MODEL="slater_jastrow"
ORBITAL_MODEL="backflow_ho"
CORR_MODEL="spin_rbf_jastrow"
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0
BATCH_SIZE=256
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
RL_SWEEPS=1_000
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
N_SAMPLES_EVALUATIONS=(1000 10000 100000)
N_SUBSAMPLES_EVALUATIONS=(1 5 10 20 30 40 50 100)

# MALA_DIR="./results_1D_new_N10_g5.0_actamp_sweep_5_100k_huziel_reward/2026-09-03_1D__MALA_g5.0_Om0.1_Amp0.1_1"
# MALA_DIR=None

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
# 4. BUCLE DE EJECUCIÓN
# ==============================================================================
for N_SAMPLES_EVALUATION in "${N_SAMPLES_EVALUATIONS[@]}"; do
    for N_SUBSAMPLES_EVALUATION in "${N_SUBSAMPLES_EVALUATIONS[@]}"; do
        
        # CREAMOS UNA SUBCARPETA ÚNICA PARA ESTA COMBINACIÓN DE SAMPLES
        CURRENT_LOGS_DIR="${BASE_LOGS_DIR}/eval_${N_SAMPLES_EVALUATION}s_${N_SUBSAMPLES_EVALUATION}sub"
        mkdir -p "$CURRENT_LOGS_DIR"

        for int_st in "${GS[@]}"; do
            
            if [ "$MODE" == "train" ]; then
                count=1
                experiment_base="1D__MALA_g${int_st}_Om${omega}_Amp${amp}"
                while [ -d "${BASE_RESULTS_DIR}/${today}_${experiment_base}_${count}" ]; do
                    count=$((count+1))
                done
                RUN_ID=$count
            else
                folder=$(ls -d ${BASE_RESULTS_DIR}/*_1D__TD3_g${int_st}_Om${omega}_Amp${amp}_* 2>/dev/null | head -n 1)
                if [ -n "$folder" ]; then
                    RUN_ID=$(echo "$folder" | awk -F'_' '{print $NF}')
                else
                    RUN_ID=1
                fi
            fi

            PIDS=()
            echo "==================================================================="
            echo "[$(date +'%H:%M:%S')] PERMUTACIÓN: g=${int_st}, Samples=${N_SAMPLES_EVALUATION}, Sub=${N_SUBSAMPLES_EVALUATION}"
            
            for method in "${METHODS[@]}"; do
                experiment_name="1D__${method}_g${int_st}_Om${omega}_Amp${amp}"
                out_dir="${BASE_RESULTS_DIR}/${today}_${experiment_name}_${RUN_ID}"
                
                # OJO: Los logs de entrenamiento siguen yendo a la carpeta base
                log_name="${today}_${experiment_name}_${RUN_ID}.txt"
                log_dir="${BASE_LOGS_DIR}/${log_name}"

                if [ "$MODE" == "train" ]; then
                    FREE_GPU=$(get_free_gpu)
                    echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU libre. Entrenando $method..."

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
                        --rl_gamma_td3=$RL_GAMMA_TD3 --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX \
                        --sigma_energy_clip=$SIG_E_CLIP --min_clip_grads_ansatz=$MIN_CLIP_GRADS \
                        --max_clip_grads_ansatz=$MAX_CLIP_GRADS --min_clip_log_ratio=$MIN_CLIP_LOG \
                        --max_clip_log_ratio=$MAX_CLIP_LOG --nan_clip_log_ratio=$MIN_CLIP_LOG \
                        --posinf_clip_log_ratio=$MAX_CLIP_LOG --neginf_clip_log_ratio=$MIN_CLIP_LOG \
                        --scale_clip_min=$SCALE_MIN --scale_clip_max=$SCALE_MAX --seed=$SEED \
                        --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                        --results_out_dir="$out_dir" > "$log_dir" 2>&1 &
                    
                    PIDS+=($!)
                    sleep 10
                fi
            done 
            
            if [ "$MODE" == "train" ]; then
                echo "[$(date +'%H:%M:%S')] Entrenamientos lanzados. Esperando..."
                wait "${PIDS[@]}"
            fi

            echo "[$(date +'%H:%M:%S')] -> Ejecutando gráficas de barrido en CPU..."
            CUDA_VISIBLE_DEVICES="" python3 scripts/plot_sweep.py \
            --mala_dir "$MALA_DIR" --base_dir "$BASE_RESULTS_DIR" --g "$int_st" --omega "$omega" \
            --amp "$amp" --run_id "$RUN_ID" > "${CURRENT_LOGS_DIR}/out_sweep_${log_name}" 2>&1

            FREE_GPU=$(get_free_gpu)
            echo "[$(date +'%H:%M:%S')] -> GPU $FREE_GPU libre. Ejecutando evaluación MALA vs TD3..."
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            
            # REDIRIGIMOS EL OUTPUT A LA SUBCARPETA ACTUAL
            CUDA_VISIBLE_DEVICES="$FREE_GPU" python3 scripts/evaluate_1D_models.py \
            --mala_dir "$MALA_DIR" --base_dir "$BASE_RESULTS_DIR" --n_samples "$N_SAMPLES_EVALUATION" \
            --subsample "$N_SUBSAMPLES_EVALUATION" --g "$int_st" --omega "$omega" --amp "$amp" \
            --run_id "$RUN_ID" > "${CURRENT_LOGS_DIR}/out_evaluate_${log_name}" 2>&1

        done 

        # LLAMAMOS AL SUMMARY AL TERMINAR TODOS LOS 'G' PARA ESTA COMBINACIÓN ESPECÍFICA
        echo "==================================================================="
        echo "[$(date +'%H:%M:%S')] GENERANDO GRÁFICA Y TABLA PARA ${N_SAMPLES_EVALUATION}s ${N_SUBSAMPLES_EVALUATION}sub..."
        CUDA_VISIBLE_DEVICES="" python3 scripts/plot_summary.py \
            --logs_dir "$CURRENT_LOGS_DIR" \
            --out_dir "$BASE_RESULTS_DIR" \
            --n_samples "$N_SAMPLES_EVALUATION" \
            --subsample "$N_SUBSAMPLES_EVALUATION"
            
    done 
done 

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] BARRIDO COMPLETADO EXITOSAMENTE."