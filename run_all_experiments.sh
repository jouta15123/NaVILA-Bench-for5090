#!/bin/bash
set -e

cd /home/jouta/NaVILA-Bench
PYTHON=/home/jouta/venvs/motionclip/bin/python

# W&B settings (override via env if needed)
WANDB_PROJECT=${WANDB_PROJECT:-supcon_hoyo_main}
WANDB_GROUP=${WANDB_GROUP:-all_experiments_freeze}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
BEST_METRIC=${BEST_METRIC:-avg_r@1}

echo "=========================================="
echo "Starting all 4 experiments..."
echo "=========================================="

# Check if sarashina_fine is already running or completed
if [ -f "hoyo_v1_1/joint_training_results/sarashina_fine/latent_snapshot_final.npz" ]; then
    echo "[1/4] Sarashina + Fine: Already completed, skipping..."
else
    echo "[1/4] Running Sarashina + Fine..."
    $PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
        --sem-encoder=sarashina --label-mode=fine --steps=5000 --run-name=sarashina_fine --stage=freeze \
        --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" --best-metric="$BEST_METRIC" --early-stop-patience=0 \
        --wandb --wandb-project="$WANDB_PROJECT" --wandb-group="$WANDB_GROUP"
fi

echo "[2/4] Running Sarashina + Coarse..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=sarashina --label-mode=coarse --steps=5000 --run-name=sarashina_coarse --stage=freeze \
    --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" --best-metric="$BEST_METRIC" --early-stop-patience=0 \
    --wandb --wandb-project="$WANDB_PROJECT" --wandb-group="$WANDB_GROUP"

echo "[3/4] Running SigLIP + Fine..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=siglip --label-mode=fine --steps=5000 --run-name=siglip_fine --stage=freeze \
    --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" --best-metric="$BEST_METRIC" --early-stop-patience=0 \
    --wandb --wandb-project="$WANDB_PROJECT" --wandb-group="$WANDB_GROUP"

echo "[4/4] Running SigLIP + Coarse..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=siglip --label-mode=coarse --steps=5000 --run-name=siglip_coarse --stage=freeze \
    --log-interval="$LOG_INTERVAL" --eval-interval="$EVAL_INTERVAL" --best-metric="$BEST_METRIC" --early-stop-patience=0 \
    --wandb --wandb-project="$WANDB_PROJECT" --wandb-group="$WANDB_GROUP"

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
