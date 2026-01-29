#!/usr/bin/env bash
set -euo pipefail

cd /home/jouta/NaVILA-Bench

PYTHON=/home/jouta/venvs/motionclip/bin/python

# Prefer cached models to avoid network dependency during long runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/numba

RUN_PREFIX=${RUN_PREFIX:-20260127_prod_supcon_v1}
RESULTS_DIR="hoyo_v1_1/joint_training_results"

# W&B settings (override via env if needed)
WANDB_PROJECT=${WANDB_PROJECT:-supcon_hoyo_main}
WANDB_GROUP=${WANDB_GROUP:-production_supcon_v1}
SEEDS=${SEEDS:-42}
SPLIT_RATIO=${SPLIT_RATIO:-0.8}
SCHEDULER=${SCHEDULER:-plateau}
PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-3}
PLATEAU_FACTOR=${PLATEAU_FACTOR:-0.5}

# Traceability metadata
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_DATE=$(date -Iseconds 2>/dev/null || date)
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"

COMMON_ARGS=(
  --sem-encoder sarashina
  --label-mode fine
  --contrastive-mode supcon
  --batch-size 32
  --lr 7.73e-05
  --lr-encoder 7.73e-05
  --lr-decoder 7.73e-05
  --weight-decay 6.8e-04
  --log-interval 100
  --eval-interval 200
  --best-metric avg_r@1
  --scheduler "${SCHEDULER}"
  --plateau-patience "${PLATEAU_PATIENCE}"
  --plateau-factor "${PLATEAU_FACTOR}"
  --early-stop-patience 0
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
)

run_case() {
  local suffix="$1"
  local lambda_c="$2"
  local lambda_v="$3"
  local temp="$4"
  local freeze_steps="${5:-2000}"
  local full_steps="${6:-8000}"
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    local base_run_name="${RUN_PREFIX}_${suffix}_s${seed}"
    local freeze_run_name="${base_run_name}_freeze"
    local full_run_name="${base_run_name}_full"
    local freeze_run_dir="${RESULTS_DIR}/${freeze_run_name}"
    local full_run_dir="${RESULTS_DIR}/${full_run_name}"

    mkdir -p "${freeze_run_dir}" "${full_run_dir}"

    cat > "${freeze_run_dir}/run_config.json" <<JSON
{
  "run_name": "${freeze_run_name}",
  "run_name_base": "${base_run_name}",
  "stage": "freeze",
  "linked_full_run_name": "${full_run_name}",
  "run_prefix": "${RUN_PREFIX}",
  "run_date": "${RUN_DATE}",
  "git_commit": "${GIT_COMMIT}",
  "seed": ${seed},
  "split_ratio": ${SPLIT_RATIO},
  "suffix": "${suffix}",
  "freeze_steps": ${freeze_steps},
  "full_steps": ${full_steps},
  "lambda_contrastive": ${lambda_c},
  "lambda_vae": ${lambda_v},
  "temp": ${temp},
  "batch_size": 32,
  "lr": 7.73e-05,
  "lr_encoder": 7.73e-05,
  "lr_decoder": 7.73e-05,
  "weight_decay": 6.8e-04,
  "log_interval": 100,
  "eval_interval": 200,
  "early_stop_patience": 0,
  "sem_encoder": "sarashina",
  "label_mode": "fine",
  "contrastive_mode": "supcon",
  "best_metric": "avg_r@1",
  "scheduler": "${SCHEDULER}",
  "plateau_patience": ${PLATEAU_PATIENCE},
  "plateau_factor": ${PLATEAU_FACTOR},
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${WANDB_GROUP}"
}
JSON

    cat > "${full_run_dir}/run_config.json" <<JSON
{
  "run_name": "${full_run_name}",
  "run_name_base": "${base_run_name}",
  "stage": "full",
  "init_from_run": "${freeze_run_name}",
  "run_prefix": "${RUN_PREFIX}",
  "run_date": "${RUN_DATE}",
  "git_commit": "${GIT_COMMIT}",
  "seed": ${seed},
  "split_ratio": ${SPLIT_RATIO},
  "suffix": "${suffix}",
  "freeze_steps": ${freeze_steps},
  "full_steps": ${full_steps},
  "lambda_contrastive": ${lambda_c},
  "lambda_vae": ${lambda_v},
  "temp": ${temp},
  "batch_size": 32,
  "lr": 7.73e-05,
  "lr_encoder": 7.73e-05,
  "lr_decoder": 7.73e-05,
  "weight_decay": 6.8e-04,
  "log_interval": 100,
  "eval_interval": 200,
  "early_stop_patience": 0,
  "sem_encoder": "sarashina",
  "label_mode": "fine",
  "contrastive_mode": "supcon",
  "best_metric": "avg_r@1",
  "scheduler": "${SCHEDULER}",
  "plateau_patience": ${PLATEAU_PATIENCE},
  "plateau_factor": ${PLATEAU_FACTOR},
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${WANDB_GROUP}"
}
JSON

    echo "=========================================="
    echo "Run base: ${base_run_name}"
    echo "  freeze_run=${freeze_run_name}"
    echo "  full_run=${full_run_name} (init_from=${freeze_run_name})"
    echo "  lambda_c=${lambda_c} lambda_v=${lambda_v} temp=${temp} seed=${seed}"
    echo "  freeze_steps=${freeze_steps} full_steps=${full_steps}"
    echo "  git_commit=${GIT_COMMIT}"
    echo "  freeze_config: ${freeze_run_dir}/run_config.json"
    echo "  full_config:   ${full_run_dir}/run_config.json"
    echo "=========================================="

    "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
      --stage freeze \
      --steps "${freeze_steps}" \
      --seed "${seed}" \
      --lambda-contrastive "${lambda_c}" \
      --lambda-vae "${lambda_v}" \
      --temp "${temp}" \
      --run-name "${freeze_run_name}" \
      "${COMMON_ARGS[@]}"

    "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
      --stage full \
      --steps "${full_steps}" \
      --seed "${seed}" \
      --init-from-run "${freeze_run_name}" \
      --lambda-contrastive "${lambda_c}" \
      --lambda-vae "${lambda_v}" \
      --temp "${temp}" \
      --run-name "${full_run_name}" \
      "${COMMON_ARGS[@]}"
  done
}

# 1) Optuna best (extreme SupCon)
run_case best_extreme 2.3604 0.0572 0.02097

# 2) RL-friendly: keep contrastive strong but restore some VAE pressure
run_case rl_mid 2.0 0.12 0.02097

# 3) RL-friendly: more balanced VAE
run_case rl_balanced 1.5 0.25 0.02097

# 4) Slightly higher temperature to test smoother geometry
run_case temp_high 2.0 0.15 0.035
