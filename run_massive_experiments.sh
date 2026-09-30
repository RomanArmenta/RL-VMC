#!/bin/bash

# ==============================================================================
# PIPELINE MASIVO: DIAGRAMA DE FASES (WIGNER CRYSTAL) Y ABLACIÓN DE RED NEURONAL
# ==============================================================================

BASE_DIR="results_phase_diagram_1"
mkdir -p "$BASE_DIR"

# Aprovechamiento de las 3 GPUs en paralelo
GPUS=(0 1 2) 
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 1. RÉGIMENES FÍSICOS OBJETIVO
# ------------------------------------------------------------------------------
NS=(10 14 16 20)
#GS=(0.1 1.0 10.0 20.0 50.0)
GS=(1.0 10.0 20.0 50.0) 
OMEGA=0.1
SOFTENING=0.1

# ------------------------------------------------------------------------------
# 2. DISTINTOS MODELOS DE RED NEURONAL (Ablación del Ansatz)
# ------------------------------------------------------------------------------
# Modelo 1: Semi-flexible (Permite evaluar cómo RL ayuda cuando el Ansatz no es perfecto)
# Modelo 2: Full Backflow (El modelo más expresivo)
WF_MODELS=("slater_jastrow" "slater_jastrow")
ORB_MODELS=("ho_variational" "backflow_ho")
CORR_MODELS=("spin_rbf" "spin_rbf_jastrow")

# ------------------------------------------------------------------------------
# 3. ESPACIO DE BÚSQUEDA RL (Adaptado a N=20 y g=50)
# ------------------------------------------------------------------------------
ACTOR_AMPLITUDES=(0.02 0.08)    # Amplitudes pequeñas requeridas por N alto y densidades críticas
PENALTY_FACTORS=(0.1 0.5)       
SIGMA_SCALES=(0.1 0.3)          
PHYS_JUMPS=(2.0 4.0)

# ------------------------------------------------------------------------------
# 4. HIPERPARÁMETROS ESTÁTICOS Y ESTABILIDAD NUMÉRICA
# ------------------------------------------------------------------------------
# Reducción de Batch Size para evitar Out-Of-Memory (OOM) en N=20
BATCH_SIZE=512
TAUS_MALA=(0.0005 0.005 0.05) 
LR_ANSATZ=1e-5
LR_ACTOR=1e-5
LR_CRITIC=1e-4

TRAIN_STEPS=100000
MEASURE_STEPS=5000
N_TER_TRAIN=3000              

COUPLED_CYCLES=8
VMC_ITERS=12500
RL_SWEEPS=2500
RL_REPLAY_CAP=10000
RL_GAMMA_TD3=0.99

GAMMA_REWARD=0.1
BETA=10.0
KAPPA=0.5
FORCE_SCALE=10.0
N_SCALE=20.0                  # Escalado ajustado al nuevo máximo de electrones
R_SCALE=5.0
MAX_DRIFT_CLIP=15.0           
ACTION_REV_LIMIT=1.0

# Límites de Clipping Extremos para soportar la repulsión de N=20 y g=50
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
# DISPATCHER DE GPU
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

echo "[$(date +'%H:%M:%S')] Inicializando Pipeline en ${#GPUS[@]} GPUs..."
exec 9> /tmp/gpu_rl_vmc_phase.lock
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

    for n in "${NS[@]}"; do
        # Asegurar paridad de espín exacta
        N_UP=$((n / 2))
        N_DN=$((n / 2))

        for g in "${GS[@]}"; do
            
            SYS_DIR="${MODEL_DIR}/results_N${n}_g${g}"
            mkdir -p "$SYS_DIR"
            
            echo "==================================================================="
            echo "[$(date +'%H:%M:%S')] SISTEMA: N=${n} | g=${g} | Modelo: ${ORB_M}"
            echo "==================================================================="

            # ----------------------------------------------------------------------
            # FASE 1: LÍNEA BASE MALA
            # ----------------------------------------------------------------------
            MALA_BASE_DIR="${SYS_DIR}/MALA_Baselines"
            mkdir -p "$MALA_BASE_DIR"
            MALA_PIDS=()
            
            for tau in "${TAUS_MALA[@]}"; do
                OUT_DIR="${MALA_BASE_DIR}/MALA_tau${tau}"
                FREE_GPU=$(get_free_gpu)
                
                echo "[$(date +'%H:%M:%S')] -> MALA (Tau=${tau}) en GPU ${FREE_GPU}..."
                CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
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
                    
                MALA_PIDS+=($!)
                sleep 15
            done
            
            echo "Esperando a que las líneas base MALA terminen..."
            wait "${MALA_PIDS[@]}"

            echo "[$(date +'%H:%M:%S')] Extrayendo el mejor Tau..."
            BEST_TAU=$(python3 scripts/get_best_mala_tau.py --sys_dir "$MALA_BASE_DIR")
            echo "-> Mejor Tau para RL-MALA: $BEST_TAU"

            # ----------------------------------------------------------------------
            # FASE 2: BARRIDO DE HIPERPARÁMETROS RL-MALA (TD3)
            # ----------------------------------------------------------------------
            for amp in "${ACTOR_AMPLITUDES[@]}"; do
                for pen in "${PENALTY_FACTORS[@]}"; do
                    for sig in "${SIGMA_SCALES[@]}"; do
                        for jump in "${PHYS_JUMPS[@]}"; do
                            
                            EXP_NAME="TD3_amp${amp}_pen${pen}_sig${sig}_jmp${jump}"
                            OUT_DIR="${SYS_DIR}/${EXP_NAME}"
                            mkdir -p "$OUT_DIR"
                            FREE_GPU=$(get_free_gpu)
                            
                            echo "[$(date +'%H:%M:%S')] -> TD3 $EXP_NAME en GPU ${FREE_GPU}..."
                            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
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
                            
                            sleep 20
                        done
                    done
                done
            done
            
            # Barrera de sincronización: Esperar que terminen todos los TD3 del sistema actual
            while [ $(jobs -p | wc -l) -ge 1 ]; do sleep 30; done

            # ----------------------------------------------------------------------
            # FASE 3: EVALUACIÓN Y GENERACIÓN DE OBSERVABLES
            # ----------------------------------------------------------------------
            echo "[$(date +'%H:%M:%S')] Ejecutando Evaluador de Grid Search..."
            python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 20000 --n_evaluations $N_EVAL_FIGS --n_ter $N_TER_TRAIN
            
            BEST_TD3_DIR=$(cat "${SYS_DIR}/best_td3_model.txt")
            
            # Crear Symlinks para unificar la estructura requerida por los scripts gráficos
            ln -sfn "MALA_Baselines/MALA_tau${BEST_TAU}" "${SYS_DIR}/MALA_tau${BEST_TAU}"
            ln -sfn "${BEST_TD3_DIR}" "${SYS_DIR}/TD3"
            
            echo "[$(date +'%H:%M:%S')] Generando Figura 4 (MALA vs RL-MALA)..."
            python3 scripts/generate_figure4_dynamic.py --sys_dir "$SYS_DIR" --n_evaluations $N_EVAL_FIGS
            
            echo "[$(date +'%H:%M:%S')] Generando Dashboard de Observables de la Configuración Ganadora..."
            python3 scripts/generate_observables_analysis.py \
                --mala_dir "${SYS_DIR}/MALA_tau${BEST_TAU}" \
                --td3_dir "${SYS_DIR}/TD3" \
                --out_dir "${SYS_DIR}" \
                --n_samples 20000 \
                --n_ter $N_TER_TRAIN

        done # Fin del ciclo g
        
        # ----------------------------------------------------------------------
        # FASE 4: DIAGRAMA DE FASES (FIGURA 3) POR NÚMERO DE ELECTRONES
        # ----------------------------------------------------------------------
        echo "[$(date +'%H:%M:%S')] Agregando datos y generando Figura 3 para N=${n}..."
        python3 scripts/generate_figure3_dynamic.py --base_dir "$MODEL_DIR" --n_elec $n --omega $OMEGA

    done # Fin del ciclo N
done # Fin del ciclo de Modelos

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] DIAGRAMA DE FASES MASIVO TOTALMENTE COMPLETADO."
echo "==================================================================="