#!/usr/bin/env bash
set -euo pipefail

# Runbook executor for:
# - Stage0 preflight (20 iters)
# - Stage1 train (1000 iters) and eval (model_999.pt)
# - Stage2 train (3000 iters) and eval (model_2000.pt, model_2999.pt)
#
# Usage:
#   scripts/experiments/run_h1_style_separation.sh preflight
#   scripts/experiments/run_h1_style_separation.sh stage1-train
#   scripts/experiments/run_h1_style_separation.sh stage1-eval
#   STAGE2_ARMS=armB,armC scripts/experiments/run_h1_style_separation.sh stage2-train
#   STAGE2_ARMS=armB,armC scripts/experiments/run_h1_style_separation.sh stage2-eval

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 {preflight|stage1-train|stage1-eval|stage2-train|stage2-eval}"
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-/workspace/NaVILA-Bench}"
ISAACLAB_SH="${ISAACLAB_SH:-/workspace/IsaacLab/isaaclab.sh}"
TRAIN_PY="${REPO_ROOT}/legged-loco/scripts/train.py"
EVAL_PY="${REPO_ROOT}/legged-loco/scripts/eval_motion.py"

SEED="${SEED:-42}"
HISTORY_LENGTH="${HISTORY_LENGTH:-9}"
TERRAIN="${TERRAIN:-flat}"
STYLE_WEIGHT="${STYLE_WEIGHT:-6.0}"
STYLE_CENTROID_MODE="${STYLE_CENTROID_MODE:-centroid}"
STYLE_NEG_WEIGHT="${STYLE_NEG_WEIGHT:-0.3}"
STYLE_NEG_MARGIN="${STYLE_NEG_MARGIN:-0.05}"
LOG_PROJECT_NAME="${LOG_PROJECT_NAME:-StyleWalker_RL_fixed}"
LOGGER="${LOGGER:-wandb}"
HEADLESS="${HEADLESS:-1}"

EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-16}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_LIN_VEL_X="${EVAL_LIN_VEL_X:-0.3}"
EVAL_LIN_VEL_Y="${EVAL_LIN_VEL_Y:-0.0}"
EVAL_ANG_VEL_Z="${EVAL_ANG_VEL_Z:-0.0}"

PRECHECK_ITERS="${PRECHECK_ITERS:-20}"
STAGE1_ITERS="${STAGE1_ITERS:-1000}"
STAGE2_ITERS="${STAGE2_ITERS:-3000}"
STAGE1_CKPT="${STAGE1_CKPT:-model_999.pt}"
STAGE2_CKPTS="${STAGE2_CKPTS:-model_2000.pt,model_2999.pt}"

# arm definitions (fixed by plan)
declare -a ALL_ARMS=("armA" "armB" "armC")
STAGE2_ARMS="${STAGE2_ARMS:-armB,armC}"

task_for_arm() {
  local arm="$1"
  case "${arm}" in
    armA|armB)
      echo "h1_vision_without_speedinput_exp_c_fixed03_res08_newenc"
      ;;
    armC)
      echo "h1_vision_without_speedinput_exp_c_fixed03_fullft_newenc"
      ;;
    *)
      echo "Unknown arm: ${arm}" >&2
      exit 1
      ;;
  esac
}

style_mode_for_arm() {
  local arm="$1"
  case "${arm}" in
    armA) echo "legacy" ;;
    armB|armC) echo "hardneg" ;;
    *)
      echo "Unknown arm: ${arm}" >&2
      exit 1
      ;;
  esac
}

run_name_for_stage() {
  local stage="$1"
  local arm="$2"
  case "${stage}" in
    preflight) echo "pf_${arm}_seed${SEED}" ;;
    s1) echo "s1_${arm}_seed${SEED}" ;;
    s2) echo "s2_${arm}_seed${SEED}" ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      exit 1
      ;;
  esac
}

append_headless() {
  local -n cmd_ref=$1
  if [[ "${HEADLESS}" == "1" ]]; then
    cmd_ref+=("--headless")
  fi
}

run_train_arm() {
  local arm="$1"
  local iterations="$2"
  local run_name="$3"

  local task
  task="$(task_for_arm "${arm}")"
  local style_mode
  style_mode="$(style_mode_for_arm "${arm}")"

  local cmd=(
    "${ISAACLAB_SH}" -p "${TRAIN_PY}"
    --task "${task}"
    --max_iterations "${iterations}"
    --seed "${SEED}"
    --terrain "${TERRAIN}"
    --history_length "${HISTORY_LENGTH}"
    --style_weight "${STYLE_WEIGHT}"
    --style_centroid_mode "${STYLE_CENTROID_MODE}"
    --style_reward_mode "${style_mode}"
    --logger "${LOGGER}"
    --log_project_name "${LOG_PROJECT_NAME}"
    --run_name "${run_name}"
  )
  if [[ "${style_mode}" == "hardneg" ]]; then
    cmd+=(--style_neg_weight "${STYLE_NEG_WEIGHT}" --style_neg_margin "${STYLE_NEG_MARGIN}")
  fi
  append_headless cmd

  echo "[train] arm=${arm} task=${task} run_name=${run_name} iters=${iterations}"
  "${cmd[@]}"
}

run_eval_arm() {
  local arm="$1"
  local run_name="$2"
  local checkpoint="$3"
  local output_dir="$4"

  local task
  task="$(task_for_arm "${arm}")"
  local cmd=(
    "${ISAACLAB_SH}" -p "${EVAL_PY}"
    --task "${task}"
    --load_run "${run_name}"
    --checkpoint "${checkpoint}"
    --num_envs "${EVAL_NUM_ENVS}"
    --eval_steps "${EVAL_STEPS}"
    --terrain "${TERRAIN}"
    --history_length "${HISTORY_LENGTH}"
    --lin_vel_x "${EVAL_LIN_VEL_X}"
    --lin_vel_y "${EVAL_LIN_VEL_Y}"
    --ang_vel_z "${EVAL_ANG_VEL_Z}"
    --output_dir "${output_dir}"
  )
  append_headless cmd

  echo "[eval] arm=${arm} task=${task} run_name=${run_name} checkpoint=${checkpoint}"
  "${cmd[@]}"
}

csv_to_arms() {
  local csv="$1"
  local -a out=()
  local IFS=','
  read -r -a raw <<< "${csv}"
  for arm in "${raw[@]}"; do
    arm="${arm//[[:space:]]/}"
    [[ -n "${arm}" ]] && out+=("${arm}")
  done
  echo "${out[*]}"
}

cd "${REPO_ROOT}"

case "${MODE}" in
  preflight)
    for arm in "${ALL_ARMS[@]}"; do
      run_train_arm "${arm}" "${PRECHECK_ITERS}" "$(run_name_for_stage preflight "${arm}")"
    done
    ;;
  stage1-train)
    for arm in "${ALL_ARMS[@]}"; do
      run_train_arm "${arm}" "${STAGE1_ITERS}" "$(run_name_for_stage s1 "${arm}")"
    done
    ;;
  stage1-eval)
    for arm in "${ALL_ARMS[@]}"; do
      run_eval_arm \
        "${arm}" \
        "$(run_name_for_stage s1 "${arm}")" \
        "${STAGE1_CKPT}" \
        "eval_results/motion/style_sep_stage1/${arm}/${STAGE1_CKPT%.pt}"
    done
    ;;
  stage2-train)
    read -r -a arms <<< "$(csv_to_arms "${STAGE2_ARMS}")"
    for arm in "${arms[@]}"; do
      run_train_arm "${arm}" "${STAGE2_ITERS}" "$(run_name_for_stage s2 "${arm}")"
    done
    ;;
  stage2-eval)
    read -r -a arms <<< "$(csv_to_arms "${STAGE2_ARMS}")"
    read -r -a checkpoints <<< "$(echo "${STAGE2_CKPTS}" | tr ',' ' ')"
    for arm in "${arms[@]}"; do
      for ckpt in "${checkpoints[@]}"; do
        ckpt="${ckpt//[[:space:]]/}"
        [[ -z "${ckpt}" ]] && continue
        run_eval_arm \
          "${arm}" \
          "$(run_name_for_stage s2 "${arm}")" \
          "${ckpt}" \
          "eval_results/motion/style_sep_stage2/${arm}/${ckpt%.pt}"
      done
    done
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    echo "Usage: $0 {preflight|stage1-train|stage1-eval|stage2-train|stage2-eval}" >&2
    exit 1
    ;;
esac

