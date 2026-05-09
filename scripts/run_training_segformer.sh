#!/usr/bin/env bash
# ============================================================
# scripts/run_training_segformer.sh
#
# Usage:
#   bash scripts/run_training_segformer.sh           # fresh run
#   bash scripts/run_training_segformer.sh --resume  # resume from checkpoint
#
# Logs go to runs/segformer_finetune/
# Best checkpoint saved to checkpoints/segformer_nuscenes.pth
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

RESUME=""
if [[ "$1" == "--resume" ]]; then
  RESUME="--resume"
  echo "Resuming from checkpoint..."
fi

echo "Starting SegFormer fine-tuning..."
echo "Logs: runs/segformer_finetune/"
echo "Checkpoint: checkpoints/segformer_nuscenes.pth"
echo ""

python src/train/train_segformer.py $RESUME

echo ""
echo "Done. Run tensorboard with:"
echo "  tensorboard --logdir runs/segformer_finetune"