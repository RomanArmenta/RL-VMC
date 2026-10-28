#!/bin/bash

# ==============================================================================
# PIPELINE DE GRID SEARCH FUNCIONAL CON GESTIÓN DE COLAS Y GPUs
# ==============================================================================

BASE_DIR="results_sampling_N10_g10"
mkdir -p "$BASE_DIR"

# Ajusta esta ruta a tu archivo .npz
# N=4 g=4:
NPZ_PATH="/home/romanarmenta/Documents/projects/uhf_1d_ho_v270826_m-determinants/results_N4/g_4.0/cas_n4_g4_omega0.1_soft0.1_b32_c0-a4/casscf_state.npz"

# N=10 g=10:
# NPZ_PATH="/home/romanarmenta/Documents/projects/uhf_1d_ho_v270826_m-determinants/results_N10/cas_g10_n10_b40_c4-a2/casscf_state.npz"


GPUS=(0 1 2) 

# Parámetros Base
N_WALKERS=100
N_STEPS_EVAL=500000
N_SWEEPS_TRAIN=20000

# Grids
TAUS=(0.01 0.05 0.1 0.25)
ACTOR_AMPLITUDES=(0.1 0.15 0.25)
PENALTY_FACTORS=(0.1 0.25)
SIGMA_SCALE=1.1
MAX_JUMP=2.0

# ------------------------------------------------------------------------------
# DISPATCHER DE GPU CON DIAGNÓSTICO (ESPERA DE GPUs LIBRES)
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
            echo "[COLA] Todas las GPUs ocupadas. Esperando a que se liberen..." >&2
            waited=1
        fi
        sleep 10 
    done
}

echo "==================================================================="
echo "[$(date +'%H:%M:%S')] INICIANDO GRID SEARCH CON COLAS"
echo "==================================================================="
exec 9> /tmp/gpu_grid_sampling.lock
flock 9

# ------------------------------------------------------------------------------
# 1. EVALUACIÓN DE MALA PURO
# ------------------------------------------------------------------------------
echo ">>> FASE 1: Lanzando baselines MALA..."
MALA_PIDS=()

for tau in "${TAUS[@]}"; do
    OUT_DIR="${BASE_DIR}/MALA_tau${tau}"
    mkdir -p "$OUT_DIR" # Se crea el directorio antes para poder guardar el .txt
    FREE_GPU=$(get_free_gpu)
    
    echo " -> Lanzando MALA (Tau=${tau}) en GPU ${FREE_GPU}..."
    CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/evaluar_wigner_casscf_2.py \
        --npz_path "$NPZ_PATH" \
        --out_dir "$OUT_DIR" \
        --n_walkers "$N_WALKERS" \
        --n_steps_eval "$N_STEPS_EVAL" \
        --n_sweeps_train 0 \
        --tau_base "$tau" \
        --actor_amp 0.0 > "${OUT_DIR}/terminal_output.txt" 2>&1 &
        
    MALA_PID=$!
    MALA_PIDS+=($MALA_PID)
    sleep 5
done

echo "Esperando que terminen los baselines de MALA: ${MALA_PIDS[*]}"
wait "${MALA_PIDS[@]}"
echo "Baselines completados."

# ------------------------------------------------------------------------------
# 2. EVALUACIÓN DE RL-MALA
# ------------------------------------------------------------------------------
echo ">>> FASE 2: Lanzando exploraciones RL-MALA..."
RL_PIDS=()

for tau in "${TAUS[@]}"; do
    for amp in "${ACTOR_AMPLITUDES[@]}"; do
        for pen in "${PENALTY_FACTORS[@]}"; do
            OUT_DIR="${BASE_DIR}/RLMALA_tau${tau}_amp${amp}_pen${pen}"
            mkdir -p "$OUT_DIR" # Se crea el directorio antes para poder guardar el .txt
            FREE_GPU=$(get_free_gpu)
            
            echo " -> Lanzando RL-MALA (Tau=${tau}, Amp=${amp}, Pen=${pen}) en GPU ${FREE_GPU}..."
            CUDA_VISIBLE_DEVICES="$FREE_GPU" nohup python3 -u scripts/evaluar_wigner_casscf_2.py \
                --npz_path "$NPZ_PATH" \
                --out_dir "$OUT_DIR" \
                --n_walkers "$N_WALKERS" \
                --n_steps_eval "$N_STEPS_EVAL" \
                --n_sweeps_train "$N_SWEEPS_TRAIN" \
                --tau_base "$tau" \
                --actor_amp "$amp" \
                --penalty "$pen" \
                --sigma_scale "$SIGMA_SCALE" \
                --max_jump "$MAX_JUMP" > "${OUT_DIR}/terminal_output.txt" 2>&1 &
                
            RL_PID=$!
            RL_PIDS+=($RL_PID)
            sleep 5
        done
    done
done

echo "Esperando a que terminen todos los procesos de RL-MALA: ${RL_PIDS[*]}"
wait "${RL_PIDS[@]}"
echo "Fase RL-MALA completada."

# ------------------------------------------------------------------------------
# 3. RECOPILACIÓN DE RESULTADOS
# ------------------------------------------------------------------------------
echo "==================================================================="
echo "COMPILANDO RESULTADOS MAESTROS..."
SUMMARY_MASTER="${BASE_DIR}/master_summary.csv"

awk '(NR == 1) || (FNR > 1)' ${BASE_DIR}/*/summary.csv > "$SUMMARY_MASTER"
echo "Grid Search finalizado. Master CSV generado en $SUMMARY_MASTER"