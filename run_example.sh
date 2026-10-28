#!/bin/bash

# ==============================================================================
# PIPELINE FOCALIZADO: N=10, g=20.0 (Con Búsqueda Independiente de Tau para TD3)
# ==============================================================================

BASE_DIR="results_example_N10_g20_7"
mkdir -p "$BASE_DIR"

GPUS=(0) 
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ------------------------------------------------------------------------------
# 1. PARÁMETROS FÍSICOS Y ANSATZ
# ------------------------------------------------------------------------------
N=10
G=20.0
OMEGA=0.1
SOFTENING=0.1

WF_MODEL="slater_jastrow"
ORB_MODEL="ho_variational"     # Sin Backflow para mantener nodos fijos
CORR_MODEL="spin_rbf_jastrow"

# ------------------------------------------------------------------------------
# 2. ESPACIO DE BÚSQUEDA LINEAL 
# ------------------------------------------------------------------------------
# MALA usará estos para encontrar su línea base estable
TAUS_MALA=(0.0001 0.0005 0.001 0.005)

# TD3 buscará su propio Tau, pudiendo potencialmente usar uno mayor que MALA
#TAUS_TD3=(0.001 0.005 0.01) #0.0001 0.0005 0.001)
TAUS_TD3=(0.00005 0.0005 0.005 0.05 0.5)
AMPS=(0.025 0.02 0.15 0.25 0.35 0.55) # 0.02 0.05 0.10)
PENS=(0.015 0.055 0.15 0.0055 0.25 0.35) # 0.1 0.3 0.5)
SIGS=(0.45 0.65 0.85 0.15 0.25 0.55 1.5) # 0.1 0.2 0.4)
JMPS=(2.0 3.0 5.0 10.0 1.0 0.5)

# Valores ancla iniciales para TD3
DEF_AMP=0.05
DEF_PEN=0.3
DEF_SIG=0.2
DEF_JMP=3.0

# ------------------------------------------------------------------------------
# 3. CONFIGURACIÓN ESTÁTICA
# ------------------------------------------------------------------------------
BATCH_SIZE=512
LR_ANSATZ=1e-4; LR_ACTOR=1e-4; LR_CRITIC=1e-4
TRAIN_STEPS=200000; MEASURE_STEPS=20000; N_TER_TRAIN=5000
COUPLED_CYCLES=20; VMC_ITERS=10000; RL_SWEEPS=5000
RL_REPLAY_CAP=10000; RL_GAMMA_TD3=0.99
GAMMA_REWARD=0.1; BETA=10.0; KAPPA=0.5
FORCE_SCALE=10.0; N_SCALE=20.0; R_SCALE=5.0
MAX_DRIFT_CLIP=15.0; ACTION_REV_LIMIT=1.0

E_CLIP_MIN=-5000.0; E_CLIP_MAX=5000.0; SIG_E_CLIP=5.0
MIN_CLIP_GRADS=-2000.0; MAX_CLIP_GRADS=2000.0
MIN_CLIP_LOG=-50.0; MAX_CLIP_LOG=50.0
SCALE_MIN=-5.0; SCALE_MAX=1.0

N_PRINT=100; N_LOG=50; STEPS_FOR_BEST=1000; N_EVAL_FIGS=3

# ------------------------------------------------------------------------------
# HELPER: Python para extraer parámetros del ganador desde el nombre de carpeta
# ------------------------------------------------------------------------------
cat << 'EOF' > scripts/extract_best_focused.py
import sys, re
try:
    with open(sys.argv[1], 'r') as f: dirname = f.read().strip()
    m = re.search(r'tau([0-9.]+)_amp([0-9.]+)_pen([0-9.]+)_sig([0-9.]+)_jmp([0-9.]+)', dirname)
    print(f"{m.group(1)},{m.group(2)},{m.group(3)},{m.group(4)},{m.group(5)}")
except Exception:
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
echo "[$(date +'%H:%M:%S')] INICIANDO RUN FOCALIZADO N=${N} g=${G}"
echo "==================================================================="
exec 9> /tmp/gpu_rl_focused.lock
flock 9

N_UP=$((N / 2))
N_DN=$((N / 2))
SYS_DIR="${BASE_DIR}/results_N${N}_g${G}"
mkdir -p "$SYS_DIR"

# ======================================================================
# FASE 1: MALA BASELINE 
# ======================================================================
MALA_BASE_DIR="${SYS_DIR}/MALA_Baselines"
rm -rf "$MALA_BASE_DIR"
mkdir -p "$MALA_BASE_DIR"
MALA_PIDS=()

for tau in "${TAUS_MALA[@]}"; do
    OUT_DIR="${MALA_BASE_DIR}/MALA_tau${tau}"
    FREE_GPU=$(get_free_gpu)
    echo "[$(date +'%H:%M:%S')] -> MALA (Tau=${tau}) en GPU ${FREE_GPU}..."
    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/run_experiment_1D_5.py \
        --method="MALA" --n_elec=$N --n_up=$N_UP --n_dn=$N_DN --omega=$OMEGA --interaction_strength=$G --softening=$SOFTENING \
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

# Usamos el script corregido que creaste
BEST_TAU_MALA=$(python3 scripts/get_best_mala_tau.py --sys_dir "$MALA_BASE_DIR")

if [ -z "$BEST_TAU_MALA" ]; then
    echo "[ERROR CRÍTICO] get_best_mala_tau.py devolvió vacío. Abortando."
    exit 1
fi
echo "[$(date +'%H:%M:%S')] -> Mejor Tau MALA extraído exitosamente: $BEST_TAU_MALA"

# ======================================================================
# FASE 2: BÚSQUEDA LINEAL DINÁMICA DE TD3 (INCLUYE TAU)
# ======================================================================
# TD3 inicia su búsqueda asumiendo el mejor Tau de MALA
CURR_TAU=$BEST_TAU_MALA  
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
        EXP_NAME="TD3_tau${T}_amp${A}_pen${P}_sig${S}_jmp${J}"
        OUT_DIR="${SYS_DIR}/${EXP_NAME}"
        
        if [ ! -f "$OUT_DIR/TD3_actor_weights.pkl" ]; then
            FREE_GPU=$(get_free_gpu)
            echo "[$(date +'%H:%M:%S')]    -> Lanzando $EXP_NAME en GPU ${FREE_GPU}..."
            
            # EL TRUCO: Usamos bash -c para encadenar el entrenamiento (&&) con el graficado en el mismo background worker
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup bash -c "python3 -u scripts/run_experiment_1D_6.py \
                --method=\"TD3\" --n_elec=$N --n_up=$N_UP --n_dn=$N_DN --omega=$OMEGA --interaction_strength=$G --softening=$SOFTENING \
                --wf_model=$WF_MODEL --orbital_model=$ORB_MODEL --correlation_model=$CORR_MODEL \
                --batch_size=$BATCH_SIZE --tau=$T \
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
                --results_out_dir=\"$OUT_DIR\" \
                && python3 scripts/plot_individual_training.py --td3_dir \"$OUT_DIR\" --mala_dir \"${SYS_DIR}/MALA_Baselines/MALA_tau${BEST_TAU_MALA}\"" > "${OUT_DIR}_train.log" 2>&1 &
            
            pids+=($!)
            sleep 15
        fi
    done
    wait "${pids[@]}"
    
    # ... (El resto del código sigue igual, evaluate_grid_search, extract_best, etc.)

    # Usamos evaluate_grid_search_2.py asegurándonos de que sea el script blindado
    python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 10000 --n_evaluations 2 --n_ter $N_TER_TRAIN > /dev/null
    
    WINNER_PARAMS=$(python3 scripts/extract_best_focused.py "${SYS_DIR}/best_td3_model.txt")
    if [ $? -eq 0 ] && [ ! -z "$WINNER_PARAMS" ]; then
        CURR_TAU=$(echo $WINNER_PARAMS | cut -d',' -f1)
        CURR_AMP=$(echo $WINNER_PARAMS | cut -d',' -f2)
        CURR_PEN=$(echo $WINNER_PARAMS | cut -d',' -f3)
        CURR_SIG=$(echo $WINNER_PARAMS | cut -d',' -f4)
        CURR_JMP=$(echo $WINNER_PARAMS | cut -d',' -f5)
        echo "[$(date +'%H:%M:%S')]    ✅ Fin del Paso ($step_name). Mejor -> Tau:$CURR_TAU, Amp:$CURR_AMP, Pen:$CURR_PEN, Sig:$CURR_SIG, Jmp:$CURR_JMP"
        
        # Symlink y plot de entrenamiento
        ln -sfn "TD3_tau${CURR_TAU}_amp${CURR_AMP}_pen${CURR_PEN}_sig${CURR_SIG}_jmp${CURR_JMP}" "${SYS_DIR}/TD3"
        ln -sfn "MALA_Baselines/MALA_tau${BEST_TAU_MALA}" "${SYS_DIR}/MALA_tau${BEST_TAU_MALA}"
        python3 scripts/plot_training_metrics_all.py --sys_dir "$SYS_DIR" > /dev/null
    else
        echo "[ERROR CRÍTICO] La evaluación falló en la fase: $step_name."
        exit 1
    fi
}

# --- Paso 0: Optimizar TAU para TD3 ---
echo " -> Optimizando: Tau de Integración (TD3)"
configs=()
for t in "${TAUS_TD3[@]}"; do configs+=("T=$t A=$CURR_AMP P=$CURR_PEN S=$CURR_SIG J=$CURR_JMP"); done
run_linear_step "Tau" "${configs[@]}"

# --- Paso A: Variar Amplitud ---
echo " -> Optimizando: Amplitud"
configs=()
for a in "${AMPS[@]}"; do configs+=("T=$CURR_TAU A=$a P=$CURR_PEN S=$CURR_SIG J=$CURR_JMP"); done
run_linear_step "Amplitud" "${configs[@]}"

# --- Paso B: Variar Penalización ---
echo " -> Optimizando: Penalización (CDLB Lambda)"
configs=()
for p in "${PENS[@]}"; do configs+=("T=$CURR_TAU A=$CURR_AMP P=$p S=$CURR_SIG J=$CURR_JMP"); done
run_linear_step "Penalización" "${configs[@]}"

# --- Paso C: Variar Escala de Varianza ---
echo " -> Optimizando: Escala Sigma"
configs=()
for s in "${SIGS[@]}"; do configs+=("T=$CURR_TAU A=$CURR_AMP P=$CURR_PEN S=$s J=$CURR_JMP"); done
run_linear_step "Sigma" "${configs[@]}"

# --- Paso D: Variar Salto Físico ---
echo " -> Optimizando: Salto Físico"
configs=()
for j in "${JMPS[@]}"; do configs+=("T=$CURR_TAU A=$CURR_AMP P=$CURR_PEN S=$CURR_SIG J=$j"); done
run_linear_step "Salto" "${configs[@]}"

# ======================================================================
# FASE 3: EVALUACIÓN FINAL DE ALTA PRECISIÓN Y OBSERVABLES
# ======================================================================
echo "[$(date +'%H:%M:%S')] Generando Dashboard Final (Alta precisión)..."
python3 scripts/evaluate_grid_search.py --base_dir "$SYS_DIR" --n_samples 20000 --n_evaluations $N_EVAL_FIGS --n_ter $N_TER_TRAIN

# Asegurar symlinks definitivos
BEST_TD3_DIR=$(cat "${SYS_DIR}/best_td3_model.txt")
ln -sfn "MALA_Baselines/MALA_tau${BEST_TAU_MALA}" "${SYS_DIR}/MALA_tau${BEST_TAU_MALA}"
ln -sfn "${BEST_TD3_DIR}" "${SYS_DIR}/TD3"

echo "[$(date +'%H:%M:%S')] Creando Figura 4 y Observables..."
python3 scripts/generate_figure4_dynamic.py --sys_dir "$SYS_DIR" --n_evaluations $N_EVAL_FIGS
python3 scripts/generate_observables_analysis.py \
    --mala_dir "${SYS_DIR}/MALA_tau${BEST_TAU_MALA}" \
    --td3_dir "${SYS_DIR}/TD3" \
    --out_dir "${SYS_DIR}" --n_samples 20000 --n_ter $N_TER_TRAIN

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] PIPELINE COMPLETADO PARA N=${N} g=${G}."
echo "==================================================================="