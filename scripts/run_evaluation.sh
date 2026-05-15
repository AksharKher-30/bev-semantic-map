#!/usr/bin/env bash
# ============================================================
# scripts/run_evaluation.sh
#
# Evaluates both LSS and IPM on the val split.
# Saves results to outputs/results/evaluation_results.csv
#
# Usage:
#   bash scripts/run_evaluation.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Running Phase 6 evaluation..."
echo "Results will be saved to outputs/results/"
echo ""

python - << 'PYEOF'
import sys
sys.path.insert(0, "src")

from utils.device import get_device
from models.lss.lss_model import LSSModel
from models.segformer.model import build_segformer_zero_shot
from train.checkpointing import load_lss
from eval.evaluate import evaluate_both
from eval.terrain_analysis import evaluate_by_terrain, print_terrain_table
from eval.results_table import print_results_table, save_results_csv

device = get_device()

print("Loading LSS checkpoint...")
lss_model = LSSModel().to(device)
epoch, miou = load_lss(lss_model, device=device)
print(f"LSS checkpoint: epoch={epoch}, best_mIoU={miou:.4f}")

print("\nLoading zero-shot SegFormer...")
seg_model = build_segformer_zero_shot().to(device)

print("\n── Overall evaluation ───────────────────────────────")
results = evaluate_both(lss_model, seg_model, split="val", device=device)
print_results_table(results["lss"], results["ipm"])

print("\n── Terrain-stratified evaluation ────────────────────")
lss_terrain = evaluate_by_terrain(lss_model, model_type="lss", split="val", device=device)
ipm_terrain = evaluate_by_terrain(None, model_type="ipm", seg_model=seg_model,
                                   split="val", device=device)
print_terrain_table(lss_terrain, ipm_terrain)

save_results_csv(results["lss"], results["ipm"], lss_terrain, ipm_terrain)
print("\nEvaluation complete.")
PYEOF