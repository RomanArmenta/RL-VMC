#!/bin/bash

mkdir -p logs

GPU=1
SYSTEM="bicyclobutane"

echo "================================================"
echo " Launching benchmark "
echo " System: $SYSTEM "
echo "================================================"

CUDA_VISIBLE_DEVICES="$GPU" nohup python3 -u run_experiment.py --system "$SYSTEM" --batch_size 1024 > "./logs/${SYSTEM}_log.txt" 2>&1 &

echo "Working in Background. Check logs in: logs/${SYSTEM}_log.txt"