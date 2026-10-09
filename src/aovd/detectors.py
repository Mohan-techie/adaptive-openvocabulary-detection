"""Teacher (zero-shot YOLOE) and student (YOLO11n) inference with on-disk caching."""
import os, pickle, time, warnings
from pathlib import Path
import numpy as np
from . import config as C
from . import data as D

warnings.filterwarnings("ignore")


def _chdir_weights():
    os.chdir(C.WEIGHTS_DIR)   # ultralytics resolves mobileclip_blt.ts relative to cwd / weights dir


class Teacher:
    def __init__(self, ds):
        _chdir_weights()
        from ultralytics import YOLOE
        self.ds = ds
        self.m = YOLOE(str(C.WEIGHTS_DIR / C.TEACHER_WEIGHTS))
        prompts = C.DATASETS[ds]["prompts"]
        self.m.set_classes(prompts, self.m.get_text_pe(prompts))
        self.cache_path = Path(__file__).resolve().parents[2] / "results" / "cache" / f"teacher_{ds}.pkl"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = pickle.load(open(self.cache_path, "rb")) if self.cache_path.exists() else {}

    def predict(self, files, bs=16):
        todo = [f for f in files if f not in self.cache]
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            for f, r in zip(chunk, self.m.predict(chunk, imgsz=C.TEACHER_IMGSZ, conf=0.001, iou=0.7, max_det=300,
                                                  device="cpu", verbose=False)):
                self.cache[f] = (r.boxes.data.cpu().numpy().astype(np.float32), tuple(r.orig_shape))
        if todo:
            pickle.dump(self.cache, open(self.cache_path, "wb"))
        return {f: self.cache[f] for f in files}


def pseudo_labels(teacher_preds, files, thr):
    """teacher_preds: img -> ((n,6) xyxy,conf,cls ; (h,w)). Returns img -> (k,5) cls,xc,yc,w,h normalised."""
    out = {}
    for f in files:
        p, (h, w) = teacher_preds[f]
        p = p[p[:, 4] >= thr]
        if len(p) == 0:
            out[f] = np.zeros((0, 5), dtype=np.float32)
            continue
        xc = (p[:, 0] + p[:, 2]) / 2 / w
        yc = (p[:, 1] + p[:, 3]) / 2 / h
        out[f] = np.stack([p[:, 5], xc, yc, (p[:, 2] - p[:, 0]) / w, (p[:, 3] - p[:, 1]) / h], 1).astype(np.float32)
    return out


def student_predict(weights, files, bs=16, imgsz=None):
    from ultralytics import YOLO
    m = YOLO(weights)
    out = {}
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        for f, r in zip(chunk, m.predict(chunk, imgsz=imgsz or C.STUDENT_IMGSZ, conf=0.001, iou=0.7, max_det=300,
                                         device="cpu", verbose=False)):
            out[f] = (r.boxes.data.cpu().numpy().astype(np.float32), tuple(r.orig_shape))
    return out


def stats_for(preds, files):
    from .evalmap import image_stats
    return [image_stats(preds[f][0], D.read_labels(f), preds[f][1]) for f in files]


def calibrate_threshold(teacher_preds, gold_files):
    """Pick pseudo-label confidence threshold from PSEUDO_GRID maximising micro-F1@IoU0.5 on the gold images."""
    from .evalmap import image_stats
    best, best_f1 = C.PSEUDO_CONF, -1.0
    for thr in C.PSEUDO_GRID:
        tp = npred = ngt = 0
        for f in gold_files:
            p, hw = teacher_preds[f]
            p = p[p[:, 4] >= thr]
            s = image_stats(p, D.read_labels(f), hw)
            tp += int(s["tp"][:, 0].sum()); npred += len(p); ngt += len(s["tcls"])
        prec = tp / max(npred, 1); rec = tp / max(ngt, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1 + 1e-9:
            best, best_f1 = thr, f1
    return best
