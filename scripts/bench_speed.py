#!/usr/bin/env python
"""CPU inference latency of teacher (YOLOE-11s @640) vs student (YOLO11n @320 / @640), batch size 1."""
import json, os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np, torch
torch.set_num_threads(4)
from aovd import config as C, data as D, detectors as DET
os.chdir(C.WEIGHTS_DIR)
from ultralytics import YOLO
out = {}
for ds in C.DATASETS:
    files = D.eval_images(ds)[:40]
    t = DET.Teacher.__new__(DET.Teacher)
    from ultralytics import YOLOE
    m = YOLOE(str(C.WEIGHTS_DIR / C.TEACHER_WEIGHTS)); pr = C.DATASETS[ds]["prompts"]
    m.set_classes(pr, m.get_text_pe(pr))
    def bench(model, imgsz):
        model.predict(files[0], imgsz=imgsz, device="cpu", verbose=False)   # warm-up
        ts = []
        for f in files:
            t0 = time.perf_counter(); model.predict(f, imgsz=imgsz, device="cpu", verbose=False); ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1000)
    s = YOLO(str(C.WEIGHTS_DIR / f"student_ft400_{ds}.pt"))
    out[ds] = dict(teacher_640_ms=bench(m, 640), student_320_ms=bench(s, C.STUDENT_IMGSZ), student_640_ms=bench(s, 640))
    print(ds, out[ds], flush=True)
(REPO / "results" / "speed.json").write_text(json.dumps(out, indent=1))
