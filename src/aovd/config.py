"""Dataset definitions and fixed (a-priori) experimental constants.

The text prompts below were fixed before any experiment was run and were never
tuned on validation data.
"""
from pathlib import Path

DATA_ROOT = Path("/tmp/claude-0/dl/ds")   # overridden with --data-root
WEIGHTS_DIR = Path("/tmp/claude-0/work")  # contains yoloe-11s-seg.pt, yolo11n.pt, mobileclip_blt.ts

DATASETS = {
    "homeobjects": dict(
        root="homeobjects-3K",
        eval_splits=["val"],
        names=["bed", "sofa", "chair", "table", "lamp", "tv", "laptop",
               "wardrobe", "window", "door", "potted plant", "photo frame"],
        prompts=["bed", "sofa", "chair", "table", "lamp", "television", "laptop",
                 "wardrobe", "window", "door", "potted plant", "photo frame"],
    ),
    "ppe": dict(
        root="construction-ppe",
        eval_splits=["val", "test"],
        names=["helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
               "no_helmet", "no_goggle", "no_gloves", "no_boots"],
        prompts=["hard hat", "gloves", "safety vest", "boots", "safety goggles",
                 "person without protective equipment", "person",
                 "head without helmet", "face without goggles",
                 "bare hand", "bare foot"],
    ),
}

BUDGETS = [10, 25, 50, 100, 200, 400]   # labeled images
POOL_SIZE = 400                          # unlabeled pool per seed (labeled images are drawn from it)
STUDENT_WEIGHTS = "yolo11n.pt"
TEACHER_WEIGHTS = "yoloe-11s-seg.pt"
STUDENT_IMGSZ = 320
TEACHER_IMGSZ = 640
PSEUDO_CONF = 0.25
PSEUDO_GRID = [0.10, 0.25, 0.40]
MIN_VISITS = 4800        # minimum image-visits (epochs x images) per training run
