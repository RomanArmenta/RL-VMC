#!/bin/bash

# ==============================================================================
# PIPELINE MASIVO [VERSIÓN DIAGNÓSTICO PROFUNDO]: WIGNER CRYSTAL
# ==============================================================================

BASE_DIR="results_phase_diagram_5"
mkdir -p "$BASE_DIR"

GPUS=(0 1 2) 
#export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 1. RÉGIMENES FÍSICOS OBJETIVO
# ------------------------------------------------------------------------------
NS=(10)
GS=(0.1 1.0 10.0 20.0) 
OMEGA=0.1
SOFTENING=0.1

# ------------------------------------------------------------------------------
# 2. DISTINTOS MODELOS DE RED NEURONAL (Ablación del Ansatz)
# ------------------------------------------------------------------------------
WF_MODELS=("slater_jastrow" "slater_jastrow")
ORB_MODELS=("ho_variational" "backflow_ho")
CORR_MODELS=("spin_rbf" "spin_rbf_jastrow")

# ------------------------------------------------------------------------------
# 3. ESPACIO DE BÚSQUEDA RL 
# ------------------------------------------------------------------------------
ACTOR_AMPLITUDES=(0.1 0.01 0.001)    
PENALTY_FACTORS=(0.1 0.5 1.0)       
SIGMA_SCALES=(0.01 0.1 0.5) # 0.5 1.0)          
PHYS_JUMPS=(3.0)

# ------------------------------------------------------------------------------
# 4. HIPERPARÁMETROS ESTÁTICOS Y ESTABILIDAD NUMÉRICA
# ------------------------------------------------------------------------------
BATCH_SIZE=1024
TAUS_MALA=(0.0005 0.005 0.05 0.0001 0.001 0.01) 
LR_ANSATZ=1e-4
LR_ACTOR=1e-4
LR_CRITIC=1e-4

TRAIN_STEPS=200000
MEASURE_STEPS=20000
N_TER_TRAIN=5000              

COUPLED_CYCLES=20
VMC_ITERS=10000
RL_SWEEPS=5000
RL_REPLAY_CAP=10000
RL_GAMMA_TD3=0.99

GAMMA_REWARD=0.1
BETA=10.0
KAPPA=0.5
FORCE_SCALE=10.0
N_SCALE=20.0                  
R_SCALE=5.0
MAX_DRIFT_CLIP=15.0           
ACTION_REV_LIMIT=1.0

# Límites de Clipping Extremos
E_CLIP_MIN=-5000.0            
E_CLIP_MAX=5000.0
SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-2000.0
MAX_CLIP_GRADS=2000.0
MIN_CLIP_LOG=-50.0
MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0
SCALE_MAX=1.0

N_PRINT=100
N_LOG=50
STEPS_FOR_BEST=1000
N_EVAL_FIGS=3

# ------------------------------------------------------------------------------
# DISPATCHER DE GPU CON DIAGNÓSTICO
# ------------------------------------------------------------------------------
get_free_gpu(){
    local waited=0
    while true; do
        for gpu in "${GPUS[@]}"; do
            process_python=$(nvidia-smi -i $gpu --query-compute-apps=pid,process_name --format=csv,noheader | grep -i python)
            if [ -z "$process_python" ]; then
                echo "$gpu"
                return
            fi
        done
        
        if [ "$waited" -eq 0 ]; then
            # Imprimimos al STDERR para que Bash no lo asigne a la variable FREE_GPU
            echo "[DEBUG]  Todas las GPUs están ocupadas. Esperando a que se liberen..." >&2
            waited=1
        fi
        sleep 20 
    done
}

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] INICIALIZANDO PIPELINE DE DIAGNÓSTICO EN ${#GPUS[@]} GPUs"
echo "==================================================================="
exec 9> /tmp/gpu_rl_vmc_phase_diag.lock
flock 9

# ==============================================================================
# BUCLE MAESTRO DE EJECUCIÓN
# ==============================================================================
for m in "${!ORB_MODELS[@]}"; do
    WF_M=${WF_MODELS[$m]}
    ORB_M=${ORB_MODELS[$m]}
    CORR_M=${CORR_MODELS[$m]}
    
    MODEL_DIR="${BASE_DIR}/Model_${ORB_M}_${CORR_M}"
    mkdir -p "$MODEL_DIR"
    echo "[DEBUG] Procesando Modelo: ${ORB_M} + ${CORR_M}"

    for n in "${NS[@]}"; do
        N_UP=$((n / 2))
        N_DN=$((n / 2))

        for g in "${GS[@]}"; do
            
            SYS_DIR="${MODEL_DIR}/results_N${n}_g${g}"
            mkdir -p "$SYS_DIR"
            
            echo "-------------------------------------------------------------------"
            echo "[$(date +'%H:%M:%S')] SISTEMA ACTUAL -> N=${n} | g=${g} | Modelo: ${ORB_M}"
            echo "-------------------------------------------------------------------"

            # ----------------------------------------------------------------------
            # FASE 1: LÍNEA BASE MALA
            # ----------------------------------------------------------------------
            MALA_BASE_DIR="${SYS_DIR}/MALA_Baselines"
            mkdir -p "$MALA_BASE_DIR"
            MALA_PIDS=()
            
            echo "[DEBUG] FASE 1: Lanzando baselines MALA..."
            for tau in "${TAUS_MALA[@]}"; do
                OUT_DIR="${MALA_BASE_DIR}/MALA_tau${tau}"
                FREE_GPU=$(get_free_gpu)
                
                echo "[$(date +'%H:%M:%S')] -> Asignando MALA (Tau=${tau}) a GPU ${FREE_GPU}..."
                CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_6.py \
                    --method="MALA" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
                    --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
                    --wf_model=$WF_M --orbital_model=$ORB_M --correlation_model=$CORR_M \
                    --batch_size=$BATCH_SIZE --tau=$tau --lr_ansatz=$LR_ANSATZ \
                    --train_steps=$TRAIN_STEPS --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
                    --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
                    --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
                    --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                    --max_phys_jump_factor=${PHYS_JUMPS[0]} \
                    --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
                    
                MALA_PID=$!
                MALA_PIDS+=($MALA_PID)
                echo "[DEBUG] PID MALA (Tau=${tau}): $MALA_PID"
                sleep 15
            done
            
            echo "[DEBUG] Esperando explícitamente a los procesos MALA con PIDs: ${MALA_PIDS[*]}"
            wait "${MALA_PIDS[@]}"
            echo "[DEBUG] Todos los procesos MALA terminaron correctamente."

            echo "[DEBUG] FASE 1.5: Extrayendo el mejor Tau..."
            BEST_TAU=$(python3 scripts/get_best_mala_tau.py --sys_dir "$MALA_BASE_DIR")
            echo "-> Mejor Tau para RL-MALA: $BEST_TAU"

            # ----------------------------------------------------------------------
            # FASE 2: BARRIDO DE HIPERPARÁMETROS RL-MALA (TD3)
            # ----------------------------------------------------------------------
            echo "[DEBUG] FASE 2: Lanzando Grid Search de TD3..."
            TD3_PIDS=()

            for amp in "${ACTOR_AMPLITUDES[@]}"; do
                for pen in "${PENALTY_FACTORS[@]}"; do
                    for sig in "${SIGMA_SCALES[@]}"; do
                        for jump in "${PHYS_JUMPS[@]}"; do
                            
                            EXP_NAME="TD3_amp${amp}_pen${pen}_sig${sig}_jmp${jump}"
                            OUT_DIR="${SYS_DIR}/${EXP_NAME}"
                            mkdir -p "$OUT_DIR"
                            FREE_GPU=$(get_free_gpu)
                            
                            echo "[$(date +'%H:%M:%S')] -> Asignando TD3 ($EXP_NAME) a GPU ${FREE_GPU}..."
                            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_6.py \
                                --method="TD3" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN \
                                --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
                                --wf_model=$WF_M --orbital_model=$ORB_M --correlation_model=$CORR_M \
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
                            
                            TD3_PID=$!
                            TD3_PIDS+=($TD3_PID)
                            # echo "[DEBUG] PID TD3 registrado: $TD3_PID"
                            sleep 15
                        done
                    done
                done
            done
            
            # Barrera de sincronización estricta por PID (Inmune a otros procesos)
            echo "[DEBUG] FASE 2.5: Esperando explícitamente a que terminen los procesos TD3: ${TD3_PIDS[*]}"
            wait "${TD3_PIDS[@]}"
            echo "[DEBUG] 🟢 Todos los entrenamientos TD3 han concluido."

            # ----------------------------------------------------------------------
            # FASE 3: EVALUACIÓN AUTOMÁTICA
            # ----------------------------------------------------------------------
            echo "[DEBUG] FASE 3: Evaluando el Grid Search..."
            python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 20000 --n_evaluations $N_EVAL_FIGS --n_ter $N_TER_TRAIN
            echo "[DEBUG] Evaluador completado."
            
            BEST_TD3_DIR=$(cat "${SYS_DIR}/best_td3_model.txt")
            echo "[DEBUG] Configuración TD3 ganadora: $BEST_TD3_DIR"
            
            echo "[DEBUG] Creando Symlinks..."
            ln -sfn "MALA_Baselines/MALA_tau${BEST_TAU}" "${SYS_DIR}/MALA_tau${BEST_TAU}"
            ln -sfn "${BEST_TD3_DIR}" "${SYS_DIR}/TD3"
            
            echo "[DEBUG] FASE 3.1: Generando Figura 4 (Eficiencia Computacional)..."
            python3 scripts/generate_figure4_dynamic.py --sys_dir "$SYS_DIR" --n_evaluations $N_EVAL_FIGS
            echo "[DEBUG] Figura 4 terminada."
            
            echo "[DEBUG] FASE 3.2: Generando Escaneo de Observables..."
            python3 scripts/generate_observables_analysis.py \
                --mala_dir "${SYS_DIR}/MALA_tau${BEST_TAU}" \
                --td3_dir "${SYS_DIR}/TD3" \
                --out_dir "${SYS_DIR}" \
                --n_samples 20000 \
                --n_ter $N_TER_TRAIN
            echo "[DEBUG] Observables terminados."

            echo "[DEBUG] Sistema g=${g} completado exitosamente."

        done # Fin del ciclo g
        
        # ----------------------------------------------------------------------
        # FASE 4: DIAGRAMA DE FASES (FIGURA 3)
        # ----------------------------------------------------------------------
        echo "[DEBUG] FASE 4: Generando Figura 3 agregada para N=${n}..."
        python3 scripts/generate_figure3_dynamic.py --base_dir "$MODEL_DIR" --n_elec $n --omega $OMEGA
        echo "[DEBUG] Figura 3 terminada."

    done # Fin del ciclo N
done # Fin del ciclo de Modelos

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] DIAGRAMA DE FASES MASIVO TOTALMENTE COMPLETADO."
echo "==================================================================="