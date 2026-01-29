#!/usr/bin/env bash
set -euo pipefail

cd /home/jouta/NaVILA-Bench
PYTHON=/home/jouta/venvs/motionclip/bin/python
RESULTS_DIR=hoyo_v1_1/joint_training_results
RUN_PREFIX=${RUN_PREFIX:-20260127_supcon_main}

# W&B settings (override via env if needed)
WANDB_PROJECT=${WANDB_PROJECT:-supcon_hoyo_main}
WANDB_GROUP=${WANDB_GROUP:-supcon_main_staged}

# Fixed protocol settings
FREEZE_STEPS=${FREEZE_STEPS:-2000}
FULL_STEPS=${FULL_STEPS:-8000}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
BEST_METRIC=${BEST_METRIC:-avg_r@1}
LABEL_MODE=${LABEL_MODE:-fine}
SEEDS=${SEEDS:-42}
SPLIT_RATIO=${SPLIT_RATIO:-0.8}
SCHEDULER=${SCHEDULER:-plateau}
PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-3}
PLATEAU_FACTOR=${PLATEAU_FACTOR:-0.5}
TEMP=${TEMP:-0.05}
LAMBDA_CONTRASTIVE=${LAMBDA_CONTRASTIVE:-1.0}
LAMBDA_VAE=${LAMBDA_VAE:-0.3}

# Traceability metadata
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_DATE=$(date -Iseconds 2>/dev/null || date)

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"

COMMON_ARGS=(
  --lr 5e-05
  --lr-encoder 2e-05
  --lr-decoder 2e-05
  --temp "${TEMP}"
  --lambda-contrastive "${LAMBDA_CONTRASTIVE}"
  --lambda-vae "${LAMBDA_VAE}"
  --best-metric "${BEST_METRIC}"
  --log-interval "${LOG_INTERVAL}"
  --eval-interval "${EVAL_INTERVAL}"
  --scheduler "${SCHEDULER}"
  --plateau-patience "${PLATEAU_PATIENCE}"
  --plateau-factor "${PLATEAU_FACTOR}"
  --early-stop-patience 0
  --wandb
  --wandb-project "${WANDB_PROJECT}"
  --wandb-group "${WANDB_GROUP}"
)

run_case() {
  local sem_encoder="$1"
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    local base_run_name="${RUN_PREFIX}_${sem_encoder}_${LABEL_MODE}_temp${TEMP}_lc${LAMBDA_CONTRASTIVE}_lv${LAMBDA_VAE}_s${seed}_f${FREEZE_STEPS}_full${FULL_STEPS}"
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
  "sem_encoder": "${sem_encoder}",
  "label_mode": "${LABEL_MODE}",
  "split_ratio": ${SPLIT_RATIO},
  "freeze_steps": ${FREEZE_STEPS},
  "full_steps": ${FULL_STEPS},
  "temp": ${TEMP},
  "lr": 5e-05,
  "lr_encoder": 2e-05,
  "lr_decoder": 2e-05,
  "lambda_contrastive": ${LAMBDA_CONTRASTIVE},
  "lambda_vae": ${LAMBDA_VAE},
  "best_metric": "${BEST_METRIC}",
  "log_interval": ${LOG_INTERVAL},
  "eval_interval": ${EVAL_INTERVAL},
  "scheduler": "${SCHEDULER}",
  "plateau_patience": ${PLATEAU_PATIENCE},
  "plateau_factor": ${PLATEAU_FACTOR},
  "early_stop_patience": 0,
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
  "sem_encoder": "${sem_encoder}",
  "label_mode": "${LABEL_MODE}",
  "split_ratio": ${SPLIT_RATIO},
  "freeze_steps": ${FREEZE_STEPS},
  "full_steps": ${FULL_STEPS},
  "temp": ${TEMP},
  "lr": 5e-05,
  "lr_encoder": 2e-05,
  "lr_decoder": 2e-05,
  "lambda_contrastive": ${LAMBDA_CONTRASTIVE},
  "lambda_vae": ${LAMBDA_VAE},
  "best_metric": "${BEST_METRIC}",
  "log_interval": ${LOG_INTERVAL},
  "eval_interval": ${EVAL_INTERVAL},
  "scheduler": "${SCHEDULER}",
  "plateau_patience": ${PLATEAU_PATIENCE},
  "plateau_factor": ${PLATEAU_FACTOR},
  "early_stop_patience": 0,
  "wandb_project": "${WANDB_PROJECT}",
  "wandb_group": "${WANDB_GROUP}"
}
JSON

    echo "=========================================="
    echo "Run base: ${base_run_name}"
    echo "  freeze_run=${freeze_run_name}"
    echo "  full_run=${full_run_name} (init_from=${freeze_run_name})"
    echo "  sem_encoder=${sem_encoder} label_mode=${LABEL_MODE} seed=${seed}"
    echo "  temp=${TEMP} lambda_c=${LAMBDA_CONTRASTIVE} lambda_vae=${LAMBDA_VAE}"
    echo "  freeze_steps=${FREEZE_STEPS} full_steps=${FULL_STEPS}"
    echo "  git_commit=${GIT_COMMIT}"
    echo "  freeze_config: ${freeze_run_dir}/run_config.json"
    echo "  full_config:   ${full_run_dir}/run_config.json"
    echo "=========================================="

    "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
      --stage freeze \
      --steps "${FREEZE_STEPS}" \
      --seed "${seed}" \
      --sem-encoder "${sem_encoder}" \
      --label-mode "${LABEL_MODE}" \
      --run-name "${freeze_run_name}" \
      "${COMMON_ARGS[@]}"

    "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
      --stage full \
      --steps "${FULL_STEPS}" \
      --seed "${seed}" \
      --init-from-run "${freeze_run_name}" \
      --sem-encoder "${sem_encoder}" \
      --label-mode "${LABEL_MODE}" \
      --run-name "${full_run_name}" \
      "${COMMON_ARGS[@]}"
  done
}

echo "=========================================="
echo "Starting staged SupCon-main experiments"
echo "  run_prefix=${RUN_PREFIX}"
echo "  freeze=${FREEZE_STEPS} -> full=${FULL_STEPS}"
echo "  label_mode=${LABEL_MODE} (default: coarse)"
echo "  seeds=${SEEDS}"
echo "  temp=${TEMP} lambda_c=${LAMBDA_CONTRASTIVE} lambda_vae=${LAMBDA_VAE}"
echo "  wandb project=${WANDB_PROJECT} group=${WANDB_GROUP}"
echo "=========================================="

run_case sarashina
run_case siglip

echo "=========================================="
echo "All staged SupCon-main experiments completed!"
echo "=========================================="
