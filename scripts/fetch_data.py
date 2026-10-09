#!/usr/bin/env python
"""Download datasets and model weights used in the paper (all from public GitHub release assets).

    python scripts/fetch_data.py --data-root /path/datasets --weights-dir /path/weights
"""
import argparse, urllib.request, zipfile
from pathlib import Path

BASE = "https://github.com/ultralytics/assets/releases/download"
DATASETS = {"homeobjects-3K": f"{BASE}/v0.0.0/homeobjects-3K.zip", "construction-ppe": f"{BASE}/v0.0.0/construction-ppe.zip"}
WEIGHTS = ["yoloe-11s-seg.pt", "yolo11n.pt", "mobileclip_blt.ts"]

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="datasets")
ap.add_argument("--weights-dir", default="models")
a = ap.parse_args()
for name, url in DATASETS.items():
    dst = Path(a.data_root) / name
    if dst.exists():
        continue
    dst.mkdir(parents=True)
    z = dst.parent / f"{name}.zip"
    urllib.request.urlretrieve(url, z)
    zipfile.ZipFile(z).extractall(dst)
    z.unlink()
Path(a.weights_dir).mkdir(parents=True, exist_ok=True)
for w in WEIGHTS:
    f = Path(a.weights_dir) / w
    if not f.exists():
        urllib.request.urlretrieve(f"{BASE}/v8.3.0/{w}", f)
print("done")
