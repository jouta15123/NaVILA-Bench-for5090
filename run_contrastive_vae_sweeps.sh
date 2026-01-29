#!/usr/bin/env bash
set -euo pipefail

cd /home/jouta/NaVILA-Bench
PYTHON=/home/jouta/venvs/motionclip/bin/python
RESULTS_DIR=hoyo_v1_1/joint_training_results

# Traceability
RUN_PREFIX=${RUN_PREFIX:-20260127_contrastive_vae}
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
RUN_DATE=$(date -Iseconds 2>/dev/null || date)

# Fixed protocol
LABEL_MODE=${LABEL_MODE:-fine}
FREEZE_STEPS=${FREEZE_STEPS:-2000}
FULL_STEPS=${FULL_STEPS:-8000}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-200}
BEST_METRIC=${BEST_METRIC:-avg_r@1}
SCHEDULER=${SCHEDULER:-plateau}
PLATEAU_PATIENCE=${PLATEAU_PATIENCE:-3}
PLATEAU_FACTOR=${PLATEAU_FACTOR:-0.5}
SEEDS=${SEEDS:-42}
SPLIT_RATIO=${SPLIT_RATIO:-0.8}

# W&B
WANDB_PROJECT=${WANDB_PROJECT:-supcon_hoyo_main}
WANDB_GROUP=${WANDB_GROUP:-contrastive_vae_sweeps}

# Sweep anchors (override if needed)
BASE_TEMP=${BASE_TEMP:-0.05}
BASE_LAMBDA_VAE=${BASE_LAMBDA_VAE:-0.06}
BASE_LAMBDA_CONT=${BASE_LAMBDA_CONT:-2.0}

# Sweep grids (comma-separated)
TEMP_SWEEP=${TEMP_SWEEP:-0.02,0.03,0.05,0.07,0.10}
LAMBDA_VAE_SWEEP=${LAMBDA_VAE_SWEEP:-0.03,0.06,0.12}
LAMBDA_CONT_SWEEP=${LAMBDA_CONT_SWEEP:-1.0,2.0,3.0}

# Encoders (comma-separated: sarashina,siglip)
SEM_ENCODERS=${SEM_ENCODERS:-sarashina}

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
IFS=',' read -r -a TEMP_LIST <<< "${TEMP_SWEEP}"
IFS=',' read -r -a LAMBDA_VAE_LIST <<< "${LAMBDA_VAE_SWEEP}"
IFS=',' read -r -a LAMBDA_CONT_LIST <<< "${LAMBDA_CONT_SWEEP}"
IFS=',' read -r -a ENCODER_LIST <<< "${SEM_ENCODERS}"

COMMON_ARGS=(
  --label-mode "${LABEL_MODE}"
  --lr 7.73e-05
  --lr-encoder 7.73e-05
  --lr-decoder 7.73e-05
  --batch-size 32
  --weight-decay 6.8e-04
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

write_run_config() {
  local run_dir="$1"
  local run_name="$2"
  local stage="$3"
  local sem_encoder="$4"
  local seed="$5"
  local sweep_name="$6"
  local sweep_param="$7"
  local sweep_value="$8"
  local temp="$9"
  local lambda_c="${10}"
  local lambda_v="${11}"
  local init_from_run="${12}"

  mkdir -p "${run_dir}"

  cat > "${run_dir}/run_config.json" <<JSON
{
  "run_name": "${run_name}",
  "run_prefix": "${RUN_PREFIX}",
  "run_date": "${RUN_DATE}",
  "git_commit": "${GIT_COMMIT}",
  "stage": "${stage}",
  "init_from_run": "${init_from_run}",
  "seed": ${seed},
  "split_ratio": ${SPLIT_RATIO},
  "sem_encoder": "${sem_encoder}",
  "label_mode": "${LABEL_MODE}",
  "sweep_name": "${sweep_name}",
  "sweep_param": "${sweep_param}",
  "sweep_value": ${sweep_value},
  "freeze_steps": ${FREEZE_STEPS},
  "full_steps": ${FULL_STEPS},
  "temp": ${temp},
  "lambda_contrastive": ${lambda_c},
  "lambda_vae": ${lambda_v},
  "lr": 7.73e-05,
  "lr_encoder": 7.73e-05,
  "lr_decoder": 7.73e-05,
  "batch_size": 32,
  "weight_decay": 6.8e-04,
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
}

run_pair() {
  local sem_encoder="$1"
  local seed="$2"
  local sweep_name="$3"
  local sweep_param="$4"
  local sweep_value="$5"
  local temp="$6"
  local lambda_c="$7"
  local lambda_v="$8"

  sem_encoder="${sem_encoder//[[:space:]]/}"
  seed="${seed//[[:space:]]/}"
  sweep_value="${sweep_value//[[:space:]]/}"

  local base_run_name="${RUN_PREFIX}_${sweep_name}_${sem_encoder}_${LABEL_MODE}_${sweep_param}${sweep_value}_temp${temp}_lc${lambda_c}_lv${lambda_v}_s${seed}_f${FREEZE_STEPS}_full${FULL_STEPS}"
  local freeze_run_name="${base_run_name}_freeze"
  local full_run_name="${base_run_name}_full"

  local freeze_run_dir="${RESULTS_DIR}/${freeze_run_name}"
  local full_run_dir="${RESULTS_DIR}/${full_run_name}"

  write_run_config "${freeze_run_dir}" "${freeze_run_name}" "freeze" "${sem_encoder}" "${seed}" "${sweep_name}" "${sweep_param}" "${sweep_value}" "${temp}" "${lambda_c}" "${lambda_v}" ""
  write_run_config "${full_run_dir}" "${full_run_name}" "full" "${sem_encoder}" "${seed}" "${sweep_name}" "${sweep_param}" "${sweep_value}" "${temp}" "${lambda_c}" "${lambda_v}" "${freeze_run_name}"

  echo "=========================================="
  echo "Run base: ${base_run_name}"
  echo "  sem_encoder=${sem_encoder} seed=${seed} label_mode=${LABEL_MODE}"
  echo "  sweep=${sweep_name} param=${sweep_param} value=${sweep_value}"
  echo "  temp=${temp} lambda_c=${lambda_c} lambda_vae=${lambda_v}"
  echo "  freeze=${FREEZE_STEPS} -> full=${FULL_STEPS} (init_from=${freeze_run_name})"
  echo "  git_commit=${GIT_COMMIT}"
  echo "=========================================="

  "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
    --stage freeze \
    --steps "${FREEZE_STEPS}" \
    --seed "${seed}" \
    --sem-encoder "${sem_encoder}" \
    --temp "${temp}" \
    --lambda-contrastive "${lambda_c}" \
    --lambda-vae "${lambda_v}" \
    --run-name "${freeze_run_name}" \
    "${COMMON_ARGS[@]}"

  "${PYTHON}" hoyo_v1_1/models/train_motionclip_joint.py \
    --stage full \
    --steps "${FULL_STEPS}" \
    --seed "${seed}" \
    --init-from-run "${freeze_run_name}" \
    --sem-encoder "${sem_encoder}" \
    --temp "${temp}" \
    --lambda-contrastive "${lambda_c}" \
    --lambda-vae "${lambda_v}" \
    --run-name "${full_run_name}" \
    "${COMMON_ARGS[@]}"
}

echo "=========================================="
echo "Starting contrastive+VAE sweeps"
echo "  run_prefix=${RUN_PREFIX} label_mode=${LABEL_MODE}"
echo "  freeze=${FREEZE_STEPS} -> full=${FULL_STEPS}"
echo "  seeds=${SEEDS} encoders=${SEM_ENCODERS}"
echo "  base: temp=${BASE_TEMP} lc=${BASE_LAMBDA_CONT} lv=${BASE_LAMBDA_VAE}"
echo "=========================================="

# 1) Temperature sweep (others fixed)
for sem_encoder in "${ENCODER_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for temp in "${TEMP_LIST[@]}"; do
      temp="${temp//[[:space:]]/}"
      run_pair "${sem_encoder}" "${seed}" "temp" "temp" "${temp}" "${temp}" "${BASE_LAMBDA_CONT}" "${BASE_LAMBDA_VAE}"
    done
  done
done

# 2) Lambda VAE sweep (temp fixed near best band)
for sem_encoder in "${ENCODER_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for lambda_v in "${LAMBDA_VAE_LIST[@]}"; do
      lambda_v="${lambda_v//[[:space:]]/}"
      run_pair "${sem_encoder}" "${seed}" "lambda_vae" "lv" "${lambda_v}" "${BASE_TEMP}" "${BASE_LAMBDA_CONT}" "${lambda_v}"
    done
  done
done

# 3) Lambda contrastive sweep (temp/lambda_vae fixed)
for sem_encoder in "${ENCODER_LIST[@]}"; do
  for seed in "${SEED_LIST[@]}"; do
    for lambda_c in "${LAMBDA_CONT_LIST[@]}"; do
      lambda_c="${lambda_c//[[:space:]]/}"
      run_pair "${sem_encoder}" "${seed}" "lambda_cont" "lc" "${lambda_c}" "${BASE_TEMP}" "${lambda_c}" "${BASE_LAMBDA_VAE}"
    done
  done
done

echo "=========================================="
echo "All sweeps completed!"
echo "=========================================="
