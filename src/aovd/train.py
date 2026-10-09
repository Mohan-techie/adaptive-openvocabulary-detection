"""Student training (fine-tuning on gold labels or on gold+pseudo labels)."""
import math, os, shutil, time
from pathlib import Path
from . import config as C
from . import data as D


def train_student(ds, train_list, n_images, seed, tag, workdir, val_list):
    from ultralytics import YOLO
    visits = max(C.MIN_VISITS, 12 * n_images)
    epochs = math.ceil(visits / n_images)
    yaml = Path(workdir) / f"{tag}.yaml"
    D.write_yaml(yaml, ds, train_list, val_list)
    model = YOLO(str(C.WEIGHTS_DIR / C.STUDENT_WEIGHTS))
    t0 = time.time()
    model.train(data=str(yaml), epochs=epochs, imgsz=C.STUDENT_IMGSZ, batch=16, seed=seed, device="cpu",
                workers=1, cache="ram", val=False, plots=False, project=str(Path(workdir) / "runs"), name=tag,
                exist_ok=True, warmup_epochs=1, close_mosaic=max(1, int(0.15 * epochs)), patience=0,
                verbose=False, amp=False, deterministic=False)
    secs = time.time() - t0
    w = Path(workdir) / "runs" / tag / "weights" / "last.pt"
    return str(w), secs, epochs
