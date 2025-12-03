#!/bin/bash
set -e

cd /home/jouta/NaVILA-Bench
PYTHON=/home/jouta/venvs/motionclip/bin/python

# 前回うまくいった設定
COMMON_ARGS="--stage=full --steps=5000 --lr=5e-05 --lr-encoder=2e-05 --lr-decoder=2e-05 --lambda-vae=1.0 --lambda-contrastive=0.5"

echo "=========================================="
echo "Starting all 4 experiments with stage=full"
echo "=========================================="

echo "[1/4] Running Sarashina + Fine..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=sarashina --label-mode=fine --run-name=sarashina_fine_full $COMMON_ARGS

echo "[2/4] Running Sarashina + Coarse..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=sarashina --label-mode=coarse --run-name=sarashina_coarse_full $COMMON_ARGS

echo "[3/4] Running SigLIP + Fine..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=siglip --label-mode=fine --run-name=siglip_fine_full $COMMON_ARGS

echo "[4/4] Running SigLIP + Coarse..."
$PYTHON hoyo_v1_1/models/train_motionclip_joint.py \
    --sem-encoder=siglip --label-mode=coarse --run-name=siglip_coarse_full $COMMON_ARGS

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
