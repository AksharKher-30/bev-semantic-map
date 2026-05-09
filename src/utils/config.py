from pathlib import Path

# ------------------------------------------------------------
# Project root - everything is relative to this
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # bev-semantic-map/

# ------------------------------------------------------------
# Paths  (change DATAROOT if your nuScenes lives elsewhere)
# ------------------------------------------------------------
PATHS = {
    "dataroot"      : PROJECT_ROOT / "data" / "nuscenes",
    "checkpoints"   : PROJECT_ROOT / "checkpoints",
    "outputs"       : PROJECT_ROOT / "outputs",
    "runs"          : PROJECT_ROOT / "runs",
    "videos"        : PROJECT_ROOT / "outputs" / "videos",
    "results"       : PROJECT_ROOT / "outputs" / "results",
}

# ------------------------------------------------------------
# nuScenes dataset config
# ------------------------------------------------------------
NUSCENES = {
    "version"       : "v1.0-mini",          # swap → "v1.0-trainval" for full dataset
    "verbose"       : True,
    "camera"        : "CAM_FRONT",          # primary camera used throughout
    "map_names"     : ["boston-seaport",
                       "singapore-onenorth",
                       "singapore-hollandvillage",
                       "singapore-queenstown"],
}

# ------------------------------------------------------------
# BEV grid parameters  (Phase 1, 3, 4, 6)
# Changing BEV_SIZE or BEV_RESOLUTION here propagates to:
#   data/bev_gt_generator.py, ipm/homography.py,
#   models/lss/splat.py, eval/iou_metric.py
# ------------------------------------------------------------
BEV = {
    "size"          : 200,      # pixels (200×200 grid)
    "resolution"    : 0.5,      # metres per pixel  →  100m × 100m coverage
    "d_min"         : 1.0,      # nearest depth bin (metres)  - LSS
    "d_max"         : 60.0,     # furthest depth bin (metres) - LSS
    "d_bins"        : 41,       # number of discrete depth bins - LSS
    "ego_x_offset"  : 0,        # lateral ego shift from grid centre (px)
    "ego_y_offset"  : 0,        # longitudinal ego shift (px)
    "camera_height" : 1.5,      # approximate camera height above ground (metres) - IPM
}

# BEV coverage in metres (derived - do not edit these)
BEV["range_m"] = BEV["size"] * BEV["resolution"]   # 100 m

# ------------------------------------------------------------
# Semantic classes  (Phase 1, 2, 3, 4, 6)
# Add/remove a class here → changes propagate everywhere.
# class index = position in this list (0-indexed)
# ------------------------------------------------------------
CLASSES = {
    "names"  : ["drivable_area", "vehicle", "pedestrian"],
    "colors" : {                    # BGR for OpenCV; RGB for matplotlib
        "drivable_area" : (100, 160, 100),
        "vehicle"       : (200, 100,  50),
        "pedestrian"    : ( 50, 120, 220),
        "background"    : ( 30,  30,  30),
    },
    # nuScenes map layers that map to BEV gt  (Phase 1)
    "map_layers" : ["drivable_area", "road_divider", "lane_divider", "ped_crossing"],
    # nuScenes annotation categories → our class index
    "category_map" : {
        "drivable_area" : 0,
        "road_divider"  : 0,
        "lane_divider"  : 0,
        "ped_crossing"  : 0,
    },
    "vehicle_keywords"    : ["vehicle"],       # substring match on nuScenes category_name
    "pedestrian_keywords" : ["pedestrian"],
    "num_classes"         : 3,
}

# ------------------------------------------------------------
# SegFormer config  (Phase 2)
# ------------------------------------------------------------
SEGFORMER = {
    "pretrained_ckpt" : "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "img_h"           : 512,   # input resolution during training (resize from 900)
    "img_w"           : 1024,
    "num_classes"     : 5,     # road, vehicle, pedestrian, sky, background (front-view)
    "ignore_index"    : 255,   # pseudo-label ignore value
    "lr"              : 6e-5,
    "epochs"          : 30,
    "batch_size"      : 2,     # M3 16GB comfortable batch
    "class_weights"   : [0.1, 0.5, 1.5, 2.0, 0.3],  # road,veh,ped,sky,bg
}

# Cityscapes 19-class → SegFormer 5-class mapping  (Phase 2)
CITYSCAPES_TO_BEV = {
    0 : 0,   # road      → road
    1 : 4,   # sidewalk  → background
    2 : 4,   # building  → background
    3 : 4,   # wall      → background
    4 : 4,   # fence     → background
    5 : 4,   # pole      → background
    6 : 4,   # traffic light → background
    7 : 4,   # sign      → background
    8 : 4,   # vegetation → background
    9 : 0,   # terrain   → road
    10: 3,   # sky       → sky
    11: 2,   # person    → pedestrian
    12: 2,   # rider     → pedestrian
    13: 1,   # car       → vehicle
    14: 1,   # truck     → vehicle
    15: 1,   # bus       → vehicle
    16: 1,   # train     → vehicle
    17: 1,   # motorcycle → vehicle
    18: 1,   # bicycle   → vehicle
}

# ------------------------------------------------------------
# LSS model config  (Phase 4 & 5)
# ------------------------------------------------------------
LSS = {
    "feature_channels" : 64,    # C - context feature dim per pixel
    "backbone"         : "efficientnet_b0",
    "backbone_out_ch"  : 1280,  # EfficientNet-B0 final feature channels
    "reduce_channels"  : 512,   # intermediate reduction channels
    "bev_encoder_ch"   : [128, 256],  # shoot module channel progression
}

# ------------------------------------------------------------
# Training config  (Phase 5)
# ------------------------------------------------------------
TRAIN = {
    "lss": {
        "epochs"         : 50,
        "batch_size"     : 2,
        "backbone_lr"    : 2e-5,   # lower LR for pre-trained backbone
        "head_lr"        : 2e-4,
        "weight_decay"   : 1e-4,
        "eta_min"        : 1e-6,   # CosineAnnealingLR floor
        "grad_clip"      : 1.0,    # max gradient norm
        "pos_weights"    : [0.5, 8.0, 20.0],  # BEV class pos_weight (road,veh,ped)
        "val_every"      : 1,      # validate every N epochs
    },
    "segformer": {
        "epochs"         : 30,
        "batch_size"     : 2,
        "lr"             : 6e-5,
        "weight_decay"   : 0.01,
        "class_weights"  : [0.1, 0.5, 1.5, 2.0, 0.3],
    },
    "num_workers"        : 0,      # set 0 for MPS (multiprocessing issues on Mac)
    "pin_memory"         : False,  # False for MPS
    "seed"               : 42,
}

# ------------------------------------------------------------
# Evaluation config  (Phase 6)
# ------------------------------------------------------------
EVAL = {
    "iou_threshold"  : 0.5,    # sigmoid threshold for binary prediction
    "class_names"    : CLASSES["names"],
    "save_error_maps": True,
}

# ------------------------------------------------------------
# Video output config  (Phase 7)
# ------------------------------------------------------------
VIDEO = {
    "fps"            : 2,       # nuScenes keyframe rate
    "width"          : 1280,    # output frame width (split L+R = 640 each)
    "height"         : 480,
    "fourcc"         : "mp4v",
    "ego_marker_r"   : 5,       # ego dot radius in BEV
}