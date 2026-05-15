import csv
from pathlib import Path
from utils.config import CLASSES, PATHS


def print_results_table(lss_results, ipm_results):
    """Print a formatted comparison table to stdout."""
    names = CLASSES["names"] + ["mIoU"]

    print("\n" + "="*52)
    print(f"{'Metric':<20} {'IPM (baseline)':>15} {'LSS (ours)':>12}")
    print("="*52)
    for name in names:
        ipm_val = ipm_results.get(name, 0.0)
        lss_val = lss_results.get(name, 0.0)
        delta   = lss_val - ipm_val
        marker  = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "~")
        print(f"{name:<20} {ipm_val:>15.4f} {lss_val:>12.4f}  {marker}{abs(delta):.4f}")
    print("="*52)
    print(f"{'LSS improvement':<20} {'':>15} {'':>12}  Δ mIoU = "
          f"{lss_results.get('mIoU',0)-ipm_results.get('mIoU',0):+.4f}\n")


def save_results_csv(lss_results, ipm_results,
                     lss_terrain=None, ipm_terrain=None,
                     filename="evaluation_results.csv"):
    """
    Save full evaluation results to CSV.

    Columns: metric, ipm, lss, delta, lss_boston, lss_sg, ipm_boston, ipm_sg
    """
    out_path = PATHS["results"] / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = CLASSES["names"] + ["mIoU"]
    rows  = []

    for name in names:
        row = {
            "metric"  : name,
            "ipm"     : round(ipm_results.get(name, 0.0), 6),
            "lss"     : round(lss_results.get(name, 0.0), 6),
            "delta"   : round(lss_results.get(name, 0.0) -
                              ipm_results.get(name, 0.0), 6),
        }
        if lss_terrain and ipm_terrain:
            row["lss_boston"]    = round(lss_terrain.get("boston",    {}).get(name, 0.0), 6)
            row["lss_singapore"] = round(lss_terrain.get("singapore", {}).get(name, 0.0), 6)
            row["ipm_boston"]    = round(ipm_terrain.get("boston",    {}).get(name, 0.0), 6)
            row["ipm_singapore"] = round(ipm_terrain.get("singapore", {}).get(name, 0.0), 6)
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved → {out_path}")
    return out_path