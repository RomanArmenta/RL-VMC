#!/bin/bash

# ==============================================================================
# PIPELINE MASIVO: HYPERPARAMETER LINEAR SEARCH (COORDINATE DESCENT)
# Sistemas: N={10, 14} | g={1.0, 10.0, 20.0, 50.0}
# ==============================================================================

BASE_DIR="results_linear_search_4"
mkdir -p "$BASE_DIR"

GPUS=(0 1 2) 
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 1. PARÁMETROS FÍSICOS Y ANSATZ
# ------------------------------------------------------------------------------
NS=(10 14)
GS=(1.0 10.0 20.0 50.0)
OMEGA=0.1
SOFTENING=0.1

WF_MODEL="slater_jastrow"
ORB_MODEL="ho_variational"
CORR_MODEL="spin_rbf_jastrow"
BF_COMPONENTS=8
BF_RMAX=5.0
BF_WIDTH=0.7
BF_ENV_LEN=3.0

# ------------------------------------------------------------------------------
# 2. ESPACIO DE BÚSQUEDA LINEAL (Vectores 1D)
# ------------------------------------------------------------------------------
AMPS=(0.02 0.05 0.10)
PENS=(0.1 0.3 0.5)
SIGS=(0.1 0.2 0.4)
JMPS=(2.0 3.0 5.0)

# Valores ancla iniciales (Medias)
DEF_AMP=0.05
DEF_PEN=0.3
DEF_SIG=0.2
DEF_JMP=3.0

# ------------------------------------------------------------------------------
# 3. CONFIGURACIÓN ESTÁTICA
# ------------------------------------------------------------------------------
BATCH_SIZE=512
TAUS_MALA=(0.0005 0.005 0.05)
LR_ANSATZ=1e-5; LR_ACTOR=1e-5; LR_CRITIC=1e-4
TRAIN_STEPS=200000; MEASURE_STEPS=10000; N_TER_TRAIN=10000
COUPLED_CYCLES=16; VMC_ITERS=12500; RL_SWEEPS=5000
RL_REPLAY_CAP=10000; RL_GAMMA_TD3=0.99
GAMMA_REWARD=0.1; BETA=10.0; KAPPA=0.5
FORCE_SCALE=10.0; N_SCALE=20.0; R_SCALE=5.0
MAX_DRIFT_CLIP=15.0; ACTION_REV_LIMIT=1.0

# Estabilidad
E_CLIP_MIN=-5000.0; E_CLIP_MAX=5000.0; SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-2000.0; MAX_CLIP_GRADS=2000.0
MIN_CLIP_LOG=-50.0; MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0; SCALE_MAX=1.0

N_PRINT=100; N_LOG=50; STEPS_FOR_BEST=1000; N_EVAL_FIGS=3

# ------------------------------------------------------------------------------
# HELPER: Python en línea para extraer el mejor modelo tras cada evaluación
# ------------------------------------------------------------------------------
cat << 'EOF' > scripts/extract_best_linear.py
import pandas as pd
import sys
try:
    df = pd.read_csv(sys.argv[1])
    best = df.sort_values("G_RL", ascending=False).iloc[0]
    print(f"{best['actor_amplitude']},{best['penalty_factor']},{best['sigma_scale']},{best['max_phys_jump']}")
except Exception as e:
    sys.exit(1)
EOF

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

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] INICIALIZANDO PIPELINE DE BÚSQUEDA LINEAL..."
echo "==================================================================="
exec 9> /tmp/gpu_rl_linear_search.lock
flock 9

for n in "${NS[@]}"; do
    N_UP=$((n / 2))
    N_DN=$((n / 2))

    for g in "${GS[@]}"; do
        
        SYS_DIR="${BASE_DIR}/results_N${n}_g${g}"
        mkdir -p "$SYS_DIR"
        echo "-------------------------------------------------------------------"
        echo "[$(date +'%H:%M:%S')] SISTEMA: N=${n} | g=${g}"
        echo "-------------------------------------------------------------------"

        # ======================================================================
        # FASE 1: MALA BASELINE (FORZADO A EJECUTARSE SIEMPRE)
        # ======================================================================
        MALA_BASE_DIR="${SYS_DIR}/MALA_Baselines"
        
        # ELIMINAMOS cualquier corrida previa para forzar el entrenamiento desde cero
        rm -rf "$MALA_BASE_DIR"
        mkdir -p "$MALA_BASE_DIR"
        MALA_PIDS=()
        
        for tau in "${TAUS_MALA[@]}"; do
            OUT_DIR="${MALA_BASE_DIR}/MALA_tau${tau}"
            FREE_GPU=$(get_free_gpu)
            echo "[$(date +'%H:%M:%S')] -> MALA (Tau=${tau}) en GPU ${FREE_GPU}..."
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
                --method="MALA" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
                --wf_model=$WF_MODEL --orbital_model=$ORB_MODEL --correlation_model=$CORR_MODEL \
                --batch_size=$BATCH_SIZE --tau=$tau --lr_ansatz=$LR_ANSATZ \
                --train_steps=$TRAIN_STEPS --measure_steps=$MEASURE_STEPS --n_ter=$N_TER_TRAIN \
                --energy_clip_min=$E_CLIP_MIN --energy_clip_max=$E_CLIP_MAX --sigma_energy_clip=$SIG_E_CLIP \
                --min_clip_grads_ansatz=$MIN_CLIP_GRADS --max_clip_grads_ansatz=$MAX_CLIP_GRADS \
                --max_phys_jump_factor=$DEF_JMP --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
            MALA_PIDS+=($!)
            sleep 10
        done
        wait "${MALA_PIDS[@]}"
        
        BEST_TAU=$(python3 scripts/get_best_mala_tau.py --sys_dir "$MALA_BASE_DIR")
        
        # CHEQUEO DE SEGURIDAD EXTREMA
        if [ -z "$BEST_TAU" ]; then
            echo "[ERROR CRÍTICO] get_best_mala_tau.py devolvió vacío. Abortando sistema N=${n} g=${g}."
            continue
        fi
        
        echo "[$(date +'%H:%M:%S')] -> Mejor Tau MALA extraído exitosamente: $BEST_TAU"

        # ======================================================================
        # FASE 2: BÚSQUEDA LINEAL DINÁMICA (COORDINATE DESCENT)
        # ======================================================================
        CURR_AMP=$DEF_AMP
        CURR_PEN=$DEF_PEN
        CURR_SIG=$DEF_SIG
        CURR_JMP=$DEF_JMP

        run_linear_step() {
            local step_name=$1
            shift
            local pids=()
            
            for param_str in "$@"; do
                eval $param_str 
                EXP_NAME="TD3_amp${A}_pen${P}_sig${S}_jmp${J}"
                OUT_DIR="${SYS_DIR}/${EXP_NAME}"
                
                if [ ! -f "$OUT_DIR/TD3_actor_weights.pkl" ]; then
                    FREE_GPU=$(get_free_gpu)
                    echo "[$(date +'%H:%M:%S')]    -> Lanzando $EXP_NAME en GPU ${FREE_GPU}..."
                    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
                        --method="TD3" --n_elec=$n --n_up=$N_UP --n_dn=$N_DN --omega=$OMEGA --interaction_strength=$g --softening=$SOFTENING \
                        --wf_model=$WF_MODEL --orbital_model=$ORB_MODEL --correlation_model=$CORR_MODEL \
                        --batch_size=$BATCH_SIZE --tau=$BEST_TAU \
                        --actor_amplitude=$A --penalty_factor=$P --sigma_action_scale=$S --max_phys_jump_factor=$J \
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
                        --max_drift_clip=$MAX_DRIFT_CLIP --action_rev_limit=$ACTION_REV_LIMIT \
                        --n_print=$N_PRINT --n_log=$N_LOG --steps_for_best=$STEPS_FOR_BEST \
                        --results_out_dir="$OUT_DIR" > "${OUT_DIR}_train.log" 2>&1 &
                    pids+=($!)
                    sleep 15
                fi
            done
            wait "${pids[@]}"

            # Evaluar y extraer
            python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 10000 --n_evaluations 2 --n_ter $N_TER_TRAIN > /dev/null
            
            WINNER_PARAMS=$(python3 scripts/extract_best_linear.py "$SYS_DIR/hyperparam_sweep_results.csv")
            if [ $? -eq 0 ] && [ ! -z "$WINNER_PARAMS" ]; then
                CURR_AMP=$(echo $WINNER_PARAMS | cut -d',' -f1)
                CURR_PEN=$(echo $WINNER_PARAMS | cut -d',' -f2)
                CURR_SIG=$(echo $WINNER_PARAMS | cut -d',' -f3)
                CURR_JMP=$(echo $WINNER_PARAMS | cut -d',' -f4)
                echo "[$(date +'%H:%M:%S')]    ✅ Fin del Paso ($step_name). Mejor -> Amp:$CURR_AMP, Pen:$CURR_PEN, Sig:$CURR_SIG, Jmp:$CURR_JMP"
                
                ln -sfn "TD3_amp${CURR_AMP}_pen${CURR_PEN}_sig${CURR_SIG}_jmp${CURR_JMP}" "${SYS_DIR}/TD3"
                python3 scripts/plot_training_metrics_all.py --sys_dir "$SYS_DIR" > /dev/null
            else
                echo "[ERROR CRÍTICO] La evaluación falló en la fase: $step_name. Abortando búsqueda lineal para este g."
                exit 1
            fi
        }

        # --- Paso A: Variar Amplitud ---
        echo " -> Optimizando: Amplitud"
        configs=()
        for a in "${AMPS[@]}"; do configs+=("A=$a P=$CURR_PEN S=$CURR_SIG J=$CURR_JMP"); done
        run_linear_step "Amplitud" "${configs[@]}"

        # --- Paso B: Variar Penalización ---
        echo " -> Optimizando: Penalización (CDLB Lambda)"
        configs=()
        for p in "${PENS[@]}"; do configs+=("A=$CURR_AMP P=$p S=$CURR_SIG J=$CURR_JMP"); done
        run_linear_step "Penalización" "${configs[@]}"

        # --- Paso C: Variar Escala de Varianza ---
        echo " -> Optimizando: Escala Sigma"
        configs=()
        for s in "${SIGS[@]}"; do configs+=("A=$CURR_AMP P=$CURR_PEN S=$s J=$CURR_JMP"); done
        run_linear_step "Sigma" "${configs[@]}"

        # --- Paso D: Variar Salto Físico ---
        echo " -> Optimizando: Salto Físico"
        configs=()
        for j in "${JMPS[@]}"; do configs+=("A=$CURR_AMP P=$CURR_PEN S=$CURR_SIG J=$j"); done
        run_linear_step "Salto" "${configs[@]}"

        # ======================================================================
        # FASE 3: EVALUACIÓN FINAL DE ALTA PRECISIÓN Y OBSERVABLES
        # ======================================================================
        echo "[$(date +'%H:%M:%S')] Generando Dashboard Final (Alta precisión)..."
        python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 20000 --n_evaluations $N_EVAL_FIGS --n_ter $N_TER_TRAIN
        
        BEST_TD3_DIR=$(cat "${SYS_DIR}/best_td3_model.txt")
        ln -sfn "MALA_Baselines/MALA_tau${BEST_TAU}" "${SYS_DIR}/MALA_tau${BEST_TAU}"
        ln -sfn "${BEST_TD3_DIR}" "${SYS_DIR}/TD3"
        
        echo "[$(date +'%H:%M:%S')] Creando Figura 4 y Observables..."
        python3 scripts/generate_figure4_dynamic.py --sys_dir "$SYS_DIR" --n_evaluations $N_EVAL_FIGS
        python3 scripts/generate_observables_analysis.py \
            --mala_dir "${SYS_DIR}/MALA_tau${BEST_TAU}" \
            --td3_dir "${SYS_DIR}/TD3" \
            --out_dir "${SYS_DIR}" --n_samples 20000 --n_ter $N_TER_TRAIN

    done
    
    echo "[$(date +'%H:%M:%S')] Agregando datos de N=${n} para Figura 3..."
    python3 scripts/generate_figure3_dynamic.py --base_dir "$BASE_DIR" --n_elec $n --omega $OMEGA

done

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] PIPELINE DE BÚSQUEDA LINEAL COMPLETADO."
echo "==================================================================="