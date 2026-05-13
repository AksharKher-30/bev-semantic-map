#!/usr/bin/env bash
# ============================================================
# scripts/run_training_lss.sh
#
# Usage:
#   bash scripts/run_training_lss.sh           # fresh run
#   bash scripts/run_training_lss.sh --resume  # resume from checkpoint
#
# Logs  → runs/lss_nuscenes/
# Ckpt  → checkpoints/lss_best.pth
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

echo "Starting LSS training..."
echo "Logs : runs/lss_nuscenes/"
echo "Ckpt : checkpoints/lss_best.pth"
echo ""

python src/train/train_lss.py $RESUME

echo ""
echo "Done. Monitor with:"
echo "  tensorboard --logdir runs/lss_nuscenes"