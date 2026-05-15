#!/usr/bin/env python
# ============================================================
# scripts/verify_phase6.py
# Run AFTER Phase 6 evaluation completes.
# Run BEFORE starting Phase 7 (video pipeline).
#
# Usage: python scripts/verify_phase6.py
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"
failures = []

def check(label, fn):
    try:
        result = fn()
        tag = FAIL if result is False else PASS
        if result is False:
            failures.append(label)
        print(f"{tag}  {label}")
    except Exception as e:
        print(f"{FAIL}  {label}  →  {e}")
        failures.append(label)


# ── imports ───────────────────────────────────────────────────
print("\n── Module imports ───────────────────────────────────────")

def _imports():
    from eval.iou_metric     import make_metric, compute_iou_from_logits
    from eval.evaluate       import evaluate_lss, evaluate_ipm, evaluate_both
    from eval.terrain_analysis import evaluate_by_terrain, print_terrain_table
    from eval.error_viz      import make_error_map, make_all_class_error_maps
    from eval.results_table  import print_results_table, save_results_csv
check("all eval modules importable", _imports)


# ── BEVIoUMetric correctness ──────────────────────────────────
print("\n── IoU metric correctness ───────────────────────────────")

from utils.config import CLASSES, BEV

def _make(v, C=None, sz=None):
    C  = C  or CLASSES["num_classes"]
    sz = sz or BEV["size"]
    return torch.full((1, C, sz, sz), float(v))

def _perfect():
    from train.losses import BEVIoUMetric
    m = BEVIoUMetric()
    m.update(_make(10.0), _make(1.0))
    assert abs(m.compute()["mIoU"] - 1.0) < 1e-3
check("perfect prediction → mIoU = 1.0", _perfect)

def _all_miss():
    from train.losses import BEVIoUMetric
    m = BEVIoUMetric()
    m.update(_make(-10.0), _make(1.0))
    assert abs(m.compute()["mIoU"] - 0.0) < 1e-3
check("all-miss prediction → mIoU = 0.0", _all_miss)

def _has_all_keys():
    from train.losses import BEVIoUMetric
    m = BEVIoUMetric()
    m.update(_make(0.0), _make(0.0))
    r = m.compute()
    assert "mIoU" in r
    for name in CLASSES["names"]:
        assert name in r
check("metric result contains all class keys + mIoU", _has_all_keys)

def _accumulation():
    from train.losses import BEVIoUMetric
    m = BEVIoUMetric()
    for _ in range(3):
        m.update(_make(10.0), _make(1.0))
    assert abs(m.compute()["mIoU"] - 1.0) < 1e-3
check("accumulation across 3 batches: still mIoU=1.0", _accumulation)


# ── error visualisation ───────────────────────────────────────
print("\n── Error visualisation ──────────────────────────────────")

def _err_map_tp():
    from eval.error_viz import make_error_map, TP_COLOR
    logits = _make(10.0).squeeze(0)
    gt     = _make(1.0).squeeze(0)
    err    = make_error_map(logits, gt, 0)
    assert err.shape == (BEV["size"], BEV["size"], 3)
    assert (err == TP_COLOR).all(), "All-TP case should be green"
check("error map: all TP → all green", _err_map_tp)

def _err_map_fp():
    from eval.error_viz import make_error_map, FP_COLOR
    logits = _make(10.0).squeeze(0)
    gt     = _make(0.0).squeeze(0)
    err    = make_error_map(logits, gt, 0)
    assert (err == FP_COLOR).all()
check("error map: all FP → all red", _err_map_fp)

def _err_map_fn():
    from eval.error_viz import make_error_map, FN_COLOR
    logits = _make(-10.0).squeeze(0)
    gt     = _make(1.0).squeeze(0)
    err    = make_error_map(logits, gt, 0)
    assert (err == FN_COLOR).all()
check("error map: all FN → all blue", _err_map_fn)


# ── results table ─────────────────────────────────────────────
print("\n── Results table ────────────────────────────────────────")

def _table_smoke():
    from eval.results_table import print_results_table, save_results_csv
    import tempfile
    from utils.config import PATHS
    fake_lss = {n: 0.4 for n in CLASSES["names"]}; fake_lss["mIoU"] = 0.4
    fake_ipm = {n: 0.2 for n in CLASSES["names"]}; fake_ipm["mIoU"] = 0.2
    print_results_table(fake_lss, fake_ipm)

    orig = PATHS["results"]
    with tempfile.TemporaryDirectory() as tmp:
        PATHS["results"] = Path(tmp)
        p = save_results_csv(fake_lss, fake_ipm)
        assert p.exists() and p.stat().st_size > 0
    PATHS["results"] = orig
check("print_results_table + save_results_csv smoke test", _table_smoke)


# ── pytest ────────────────────────────────────────────────────
print("\n── Pytest ───────────────────────────────────────────────")

def _run_pytest():
    import subprocess
    root   = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_iou_metric.py",
         "-v", "--tb=short", "-q"],
        cwd=str(root), capture_output=True, text=True
    )
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)
    if result.returncode != 0:
        print(result.stderr[-500:])
        raise AssertionError("pytest failures")
check("pytest test_iou_metric all pass", _run_pytest)


# ── evaluation CSV (post-eval) ────────────────────────────────
print("\n── Evaluation output ────────────────────────────────────")

from utils.config import PATHS

def _csv_exists():
    csv_path = PATHS["results"] / "evaluation_results.csv"
    if not csv_path.exists():
        print(f"{WARN}  No results CSV yet.")
        print(f"       Run:  bash scripts/run_evaluation.sh")
        return True
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "mIoU" in [r["metric"] for r in rows]
    miou_row = next(r for r in rows if r["metric"] == "mIoU")
    print(f"       LSS mIoU={miou_row['lss']}  IPM mIoU={miou_row['ipm']}")
check("evaluation_results.csv exists with mIoU row (warn-only)", _csv_exists)


# ── summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED — {len(failures)} check(s):")
    for f in failures:
        print(f"  • {f}")
    print("Fix before proceeding to Phase 7.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED — Phase 6 complete. Safe to start Phase 7.\n")
    sys.exit(0)