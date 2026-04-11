#!/bin/bash
# Usage: bash slurm/run_pipeline.sh
# Submits train (all 3 archs) → predict → accuracy as a SLURM dependency chain.
# Logs are written to logs/pipeline/run_<timestamp>.log

set -e

PROJECT_ROOT=/d/hpc/home/jn16867/ris
cd "$PROJECT_ROOT"

mkdir -p logs/pipeline logs/train_efficient_net_b2 logs/train_efficient_net_b4 logs/train_convnext_tiny logs/train_vit_b_16 logs/predict

RUN_ID=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="logs/pipeline/run_${RUN_ID}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$PIPELINE_LOG"; }

log "=== Starting pipeline run $RUN_ID ==="

# --- Submit training jobs (independent, run in parallel) ---
JID_ENB=$(sbatch --parsable slurm/train_efficient_net_b2.slurm)
log "Submitted EfficientNet-B2 training: job $JID_ENB"

JID_ENB4=$(sbatch --parsable slurm/train_efficient_net_b4.slurm)
log "Submitted EfficientNet-B4 training: job $JID_ENB4"

JID_CNX=$(sbatch --parsable slurm/train_convnext_tiny.slurm)
log "Submitted ConvNeXt-Tiny training:   job $JID_CNX"

JID_VIT=$(sbatch --parsable slurm/train_vit_b_16.slurm)
log "Submitted ViT-B/16 training:        job $JID_VIT"

# --- Submit predict job (waits for all 3 training jobs) ---
JID_PRED=$(sbatch --parsable \
    --dependency=afterok:${JID_ENB}:${JID_ENB4}:${JID_CNX}:${JID_VIT} \
    slurm/predict.slurm)
log "Submitted predict:                  job $JID_PRED (depends on $JID_ENB,$JID_ENB4,$JID_CNX,$JID_VIT)"

# --- Submit accuracy job (waits for predict) ---
JID_ACC=$(sbatch --parsable \
    --dependency=afterok:${JID_PRED} \
    --job-name=ris-accuracy \
    --partition=gpu \
    --gres=gpu:0 \
    --time=0:05:00 \
    --mem=2G \
    --cpus-per-task=1 \
    --output="logs/pipeline/accuracy_${RUN_ID}_%j.out" \
    --error="logs/pipeline/accuracy_${RUN_ID}_%j.err" \
    --wrap="
        cd $PROJECT_ROOT
        source ~/.bashrc
        conda activate ris
        python scripts/calculate_accuracy.py \
            --predictions Jur.txt \
            --ground-truth data/testni_set.csv \
            --run-id $RUN_ID \
            --log logs/pipeline/results.log
    ")
log "Submitted accuracy:                 job $JID_ACC (depends on $JID_PRED)"

# Write run ID to sentinel file so the polling cron job knows what to wait for
echo "$RUN_ID" > logs/pipeline/.pending_run_id

log ""
log "Pipeline submitted. Monitor with:"
log "  squeue -u \$USER"
log "  tail -f $PIPELINE_LOG"
log "  tail -f logs/pipeline/results.log"
