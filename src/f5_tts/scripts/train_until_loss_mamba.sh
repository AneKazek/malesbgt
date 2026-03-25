#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/f5_tts/scripts/train_until_loss_mamba.sh [target_loss]
# Example:
#   bash src/f5_tts/scripts/train_until_loss_mamba.sh 3.0

TARGET_LOSS="${1:-3.0}"
MAX_EPOCHS="${MAX_EPOCHS:-999}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-kcv-tts-mamba}"
WANDB_RUN_PREFIX="${WANDB_RUN_PREFIX:-train-until-loss}"
WANDB_MODE_VALUE="${WANDB_MODE_VALUE:-online}"

# Adjust these if needed.
VENV_PY="./.venv/bin/python"
DATASET_NAME="datasetku"
MODEL_NAME="F5TTS_MAMBA_datasetku"
TOKENIZER_NAME="pinyin"
MEL_SPEC_NAME="vocos"
LOG_DIR="./ckpts/train_until_logs"
CHECKPOINT_DIR="./ckpts/${MODEL_NAME}_${MEL_SPEC_NAME}_${TOKENIZER_NAME}_${DATASET_NAME}"
BEST_DIR="${CHECKPOINT_DIR}/best"

mkdir -p "$LOG_DIR"
mkdir -p "$BEST_DIR"

echo "Target loss      : $TARGET_LOSS"
echo "Max epochs       : $MAX_EPOCHS"
echo "Use W&B          : $USE_WANDB"
echo "W&B project      : $WANDB_PROJECT_NAME"
echo "W&B mode         : $WANDB_MODE_VALUE"
echo "Dataset          : $DATASET_NAME"
echo "Model name       : $MODEL_NAME"
echo "Loop logs        : $LOG_DIR"
echo "Checkpoint dir   : $CHECKPOINT_DIR"
echo "Best ckpt dir    : $BEST_DIR"
echo "Will auto-resume : yes (same ckpt save_dir)"

best_loss="9999"
round=0

if [[ "$USE_WANDB" == "1" ]]; then
  export WANDB_MODE="$WANDB_MODE_VALUE"
  if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is empty while WANDB_MODE=online."
    echo "Set WANDB_API_KEY or use WANDB_MODE_VALUE=offline."
  fi
fi

while true; do
  round=$((round + 1))
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="$LOG_DIR/round_${round}_${ts}.log"

  echo
  echo "=== ROUND $round START ($(date)) ==="
  echo "Log file: $log_file"

  run_name="${WANDB_RUN_PREFIX}-${MODEL_NAME}-r${round}-${ts}"

  if [[ "$USE_WANDB" == "1" ]]; then
    logger_override="++ckpts.logger=wandb"
    wandb_project_override="++ckpts.wandb_project=${WANDB_PROJECT_NAME}"
    wandb_run_override="++ckpts.wandb_run_name=${run_name}"
  else
    logger_override="++ckpts.logger=null"
    wandb_project_override="++ckpts.wandb_project=${WANDB_PROJECT_NAME}"
    wandb_run_override="++ckpts.wandb_run_name=${run_name}"
  fi

  # Note:
  # - Same model.name + datasets.name means Trainer.load_checkpoint() can resume.
  # - Keep settings conservative for low VRAM.
  set +e
  CUDA_VISIBLE_DEVICES=0 "$VENV_PY" -m accelerate.commands.launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    src/f5_tts/train/train.py \
    --config-name F5TTS_SANITY_HYBRID_1K.yaml \
    ++datasets.name="$DATASET_NAME" \
    ++datasets.batch_size_per_gpu=256 \
    ++datasets.batch_size_type=frame \
    ++datasets.max_samples=1 \
    ++datasets.num_workers=1 \
    ++optim.epochs="$MAX_EPOCHS" \
    ++optim.learning_rate=1.0e-5 \
    ++optim.num_warmup_updates=20 \
    ++optim.grad_accumulation_steps=4 \
    ++optim.max_grad_norm=1.0 \
    "$logger_override" \
    "$wandb_project_override" \
    "$wandb_run_override" \
    ++ckpts.log_samples=False \
    ++ckpts.save_per_updates=50 \
    ++ckpts.last_per_updates=25 \
    ++ckpts.keep_last_n_checkpoints=2 \
    ++model.name="$MODEL_NAME" \
    ++model.arch.dim=256 \
    ++model.arch.depth=6 \
    ++model.arch.heads=4 \
    ++model.arch.text_dim=256 \
    ++model.arch.checkpoint_activations=True \
    ++model.arch.use_mamba=true \
    ++model.arch.mamba_layers=[1,3] \
    2>&1 | tee "$log_file"
  cmd_status=${PIPESTATUS[0]}
  set -e

  if [[ "$cmd_status" -ne 0 ]]; then
    echo "Training command exited with code $cmd_status."
    echo "Cooldown 10s then retrying..."
    sleep 10
    continue
  fi

  # Extract last seen loss from the round log.
  # Supports formats like: loss=3.21, loss=3, loss=1.23e-4
  last_loss="$(grep -oE 'loss=[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file" | tail -n 1 | cut -d= -f2 || true)"

  # Fallback: if no progress bar loss token appears, try explicit trailing logs.
  if [[ -z "$last_loss" ]]; then
    last_loss="$(grep -oE 'loss[^0-9]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file" | tail -n 1 | grep -oE '[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' || true)"
  fi

  if [[ -z "$last_loss" ]]; then
    if grep -q 'Saved last checkpoint at update' "$log_file"; then
      echo "No loss token found, but checkpoint update exists."
      echo "Likely resumed near epoch boundary or non-progress logging format."
    fi
    echo "Could not parse loss from $log_file"
    echo "Retrying next round in 10s..."
    sleep 10
    continue
  fi

  is_improved="$("$VENV_PY" - <<PY
best = float("$best_loss")
cur = float("$last_loss")
print("yes" if cur < best else "no")
PY
)"

  if [[ "$is_improved" == "yes" ]]; then
    best_loss="$last_loss"

    src_ckpt="${CHECKPOINT_DIR}/model_last.pt"
    if [[ -f "$src_ckpt" ]]; then
      loss_tag="$(echo "$best_loss" | tr '.' '_')"
      best_ckpt_file="${BEST_DIR}/model_best_loss_${loss_tag}_round_${round}_${ts}.pt"
      cp -f "$src_ckpt" "$best_ckpt_file"
      cp -f "$src_ckpt" "${BEST_DIR}/model_best.pt"
      echo "New best loss found: $best_loss"
      echo "Saved best checkpoint: $best_ckpt_file"
      echo "Updated latest best  : ${BEST_DIR}/model_best.pt"
    else
      echo "New best loss found ($best_loss), but source checkpoint missing: $src_ckpt"
    fi
  fi

  echo "Round $round last_loss = $last_loss | best_loss = $best_loss"

  below_target="$("$VENV_PY" - <<PY
cur = float("$last_loss")
tgt = float("$TARGET_LOSS")
print("yes" if cur < tgt else "no")
PY
)"

  if [[ "$below_target" == "yes" ]]; then
    echo "Target reached: last_loss ($last_loss) < target ($TARGET_LOSS)"
    echo "Stopping loop."
    break
  fi

  echo "Target not reached yet. Continue training in 5s..."
  sleep 5
done

echo "Done."
