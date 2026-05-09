#!/usr/bin/env python
# ============================================================
# scripts/verify_env.py  - Phase 0 integrity check
# Run: python scripts/verify_env.py
# Exit 0 = all pass. Exit 1 = fix required.
# ============================================================

import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


# ── 1. Python version ─────────────────────────────────────────
print("\n── Python ──────────────────────────────────────────────")

def _python_version():
    v = sys.version_info
    assert v.major == 3 and v.minor == 10, \
        f"Need Python 3.10, got {v.major}.{v.minor}"

check("Python 3.10", _python_version)


# ── 2. Package imports ────────────────────────────────────────
print("\n── Imports ─────────────────────────────────────────────")

required_packages = [
    ("torch",        "2.3"),
    ("torchvision",  "0.18"),
    ("transformers", "4.41"),
    ("timm",         "0.9"),
    ("nuscenes",     "1.2"),   # __version__=="0.0" - uses importlib.metadata
    ("cv2",          None),
    ("numpy",        "1.26"),
    ("PIL",          None),
    ("einops",       None),
    ("pyquaternion", None),
    ("tensorboard",  None),
    ("yaml",         None),
    ("scipy",        None),
    ("tqdm",         None),
    ("matplotlib",   "3.6"),
]

_DIST_NAME = {
    "nuscenes" : "nuscenes-devkit",
    "cv2"      : "opencv-python",
    "PIL"      : "Pillow",
    "yaml"     : "PyYAML",
}

def _pkg_version(import_name):
    """importlib.metadata is accurate; __version__ lies for nuscenes-devkit."""
    import importlib.metadata as meta
    for dist in [_DIST_NAME.get(import_name), import_name]:
        if dist is None:
            continue
        try:
            return meta.version(dist)
        except meta.PackageNotFoundError:
            pass
    mod = importlib.import_module(import_name)
    return getattr(mod, "__version__", "0.0")

for pkg_name, min_ver in required_packages:
    def _import_check(name=pkg_name, ver=min_ver):
        importlib.import_module(name)
        if ver:
            installed = _pkg_version(name)
            inst = [int(x) for x in installed.split(".")[:2]]
            need = [int(x) for x in ver.split(".")[:2]]
            assert inst >= need, f"Need >= {ver}, got {installed}"
    check(f"import {pkg_name}" + (f" >= {min_ver}" if min_ver else ""), _import_check)


# ── 3. Device ─────────────────────────────────────────────────
print("\n── Device ──────────────────────────────────────────────")

def _mps_available():
    import torch
    if not torch.backends.mps.is_available():
        print(f"{WARN}  MPS not available - will fall back to CPU")
        return True
    t = torch.ones(4, 4, device="mps")
    assert t.sum().item() == 16.0
check("MPS backend + tensor creation", _mps_available)

def _device_util():
    from utils.device import get_device
    assert get_device(verbose=False) is not None
check("get_device() returns device", _device_util)


# ── 4. Config consistency ─────────────────────────────────────
print("\n── Config consistency ──────────────────────────────────")

def _config_import():
    from utils.config import PROJECT_ROOT, PATHS, NUSCENES, BEV, CLASSES, LSS, TRAIN, EVAL, VIDEO
    assert PROJECT_ROOT.exists()
check("config.py importable + PROJECT_ROOT exists", _config_import)

def _bev_derived():
    from utils.config import BEV
    assert BEV["range_m"] == BEV["size"] * BEV["resolution"]
check("BEV['range_m'] == size x resolution", _bev_derived)

def _class_count():
    from utils.config import CLASSES
    assert len(CLASSES["names"]) == CLASSES["num_classes"]
check("CLASSES num_classes matches len(names)", _class_count)

def _cityscapes_map():
    from utils.config import CITYSCAPES_TO_BEV
    assert len(CITYSCAPES_TO_BEV) == 19
    assert set(CITYSCAPES_TO_BEV.values()).issubset({0, 1, 2, 3, 4})
check("CITYSCAPES_TO_BEV covers all 19 classes", _cityscapes_map)

def _pos_weights():
    from utils.config import TRAIN, CLASSES
    assert len(TRAIN["lss"]["pos_weights"]) == CLASSES["num_classes"]
check("TRAIN pos_weights length == num_classes", _pos_weights)

def _paths_keys():
    from utils.config import PATHS
    missing = [k for k in ["dataroot","checkpoints","outputs","runs","videos","results"]
               if k not in PATHS]
    assert not missing, f"Missing: {missing}"
check("PATHS has all required keys", _paths_keys)


# ── 5. Directory structure ─────────────────────────────────────
print("\n── Directory structure ─────────────────────────────────")

from utils.config import PROJECT_ROOT

required_dirs = [
    "src/utils", "src/data", "src/models/segformer", "src/models/lss",
    "src/ipm", "src/train", "src/eval", "src/viz",
    "scripts", "notebooks", "checkpoints",
    "outputs/videos", "outputs/results", "tests",
]

for d in required_dirs:
    def _dir_exists(path=d):
        full = PROJECT_ROOT / path
        assert full.exists() and full.is_dir(), f"Missing: {full}"
    check(f"dir exists: {d}", _dir_exists)


# ── 6. nuScenes dataset ────────────────────────────────────────
print("\n── nuScenes dataset ────────────────────────────────────")

def _nuscenes_layout():
    from utils.config import PATHS, NUSCENES
    dataroot    = PATHS["dataroot"]          # .../data/nuscenes/
    version     = NUSCENES["version"]        # "v1.0-mini"
    version_dir = dataroot / version         # .../data/nuscenes/v1.0-mini/
    json_file   = version_dir / "scene.json" # devkit canary

    if not dataroot.exists():
        print(f"{WARN}  Dataroot missing: {dataroot}")
        print(f"       Download from https://www.nuscenes.org/download")
        return True

    # ── Detect wrong double-nesting (common tar extraction mistake) ──────
    # Wrong: data/nuscenes/v1.0-mini/v1.0-mini/scene.json
    # Right: data/nuscenes/v1.0-mini/scene.json
    nested_json = version_dir / version / "scene.json"
    if not json_file.exists() and nested_json.exists():
        print(f"{WARN}  WRONG LAYOUT - tarballs extracted one level too deep.")
        print(f"       JSON files are at:  .../{version}/{version}/")
        print(f"       Devkit needs them at: .../{version}/")
        print()
        print(f"       Run these commands inside your data/nuscenes/ folder:")
        print(f"       ───────────────────────────────────────────────────────")
        print(f"       cd \"{dataroot}\"")
        print(f"       mv {version}/{version}/*.json  {version}/")
        print(f"       for d in maps samples sweeps; do")
        print(f"         mv {version}/{version}/$d   {version}/  2>/dev/null || true")
        print(f"       done")
        print(f"       rmdir {version}/{version}")
        print(f"       ───────────────────────────────────────────────────────")
        return True   # warn, don't hard-fail

    # ── Map expansion check ──────────────────────────────────────────────
    # devkit looks for maps at:  dataroot/maps/  (NOT inside version_dir)
    maps_dir = dataroot / "maps"
    boston_tile = maps_dir / "36092f0b03a857c6a3403e25b4b7aab3.png"
    if not boston_tile.exists():
        print(f"{WARN}  Map expansion not found.")
        print(f"       nuScenes-map-expansion-v1.3/ contents must go into:")
        print(f"       {maps_dir}/")
        print(f"       Command:")
        print(f"       cp -r nuScenes-map-expansion-v1.3/maps/*  \"{maps_dir}/\"")
    else:
        print(f"{PASS}  Map expansion found at {maps_dir}")

    return True

check("nuScenes layout check (warn-only)", _nuscenes_layout)

def _nuscenes_devkit_load():
    from utils.config import PATHS, NUSCENES
    dataroot    = PATHS["dataroot"]
    version_dir = dataroot / NUSCENES["version"]
    json_file   = version_dir / "scene.json"

    if not json_file.exists():
        print(f"{WARN}  Skipping devkit load - JSON not at expected path: {json_file}")
        return True

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=NUSCENES["version"], dataroot=str(dataroot), verbose=False)
    assert len(nusc.scene) == 10,  f"Expected 10 scenes, got {len(nusc.scene)}"
    assert len(nusc.sample) == 404, f"Expected 404 samples, got {len(nusc.sample)}"
    assert len(nusc.calibrated_sensor) > 0
    print(f"       scenes={len(nusc.scene)}  samples={len(nusc.sample)}  "
          f"cal_sensors={len(nusc.calibrated_sensor)}")

check("nuScenes devkit load + record counts", _nuscenes_devkit_load)


# ── Summary ───────────────────────────────────────────────────
print("\n────────────────────────────────────────────────────────")
if failures:
    print(f"FAILED - {len(failures)} check(s) did not pass:")
    for f in failures:
        print(f"  • {f}")
    print("Fix the above before proceeding to Phase 1.\n")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED - Phase 0 complete. Safe to start Phase 1.\n")
    sys.exit(0)