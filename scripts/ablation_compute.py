#!/usr/bin/env python
"""Compute-sensitivity ablation: fine-tune on N=400 labeled images (seed 0) with 4x training visits and/or 640 px.
Usage: python scripts/ablation_compute.py --dataset homeobjects --mult 4 --imgsz 320 --threads 4
"""
import argparse, json, os, shutil, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--mult", type=int, default=4)
ap.add_argument("--imgsz", type=int, default=320); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--N", type=int, default=400); ap.add_argument("--workdir", default="/tmp/claude-0/work/abl")
a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads)
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO / "src"))
import torch; torch.set_num_threads(a.threads)
from aovd import config as C, data as D, detectors as DET, evalmap as EM, train as TR
os.chdir(REPO)
ds = a.dataset; work = Path(a.workdir) / ds; work.mkdir(parents=True, exist_ok=True)
ev = D.eval_images(ds); (work / "val.txt").write_text("\n".join(ev) + "\n")
pool = D.pool_for_seed(ds, 0)[:a.N]
tag = f"abl_ft_N{a.N}_x{a.mult}_sz{a.imgsz}"
lst = D.build_mixed_dataset(work / tag, ds, pool, a.N, {})
w, secs, ep = TR.train_student(ds, lst, a.N, 0, tag, work, str(work / "val.txt"), visits_mult=a.mult, imgsz=a.imgsz)
st = DET.stats_for(DET.student_predict(w, ev, imgsz=a.imgsz), ev)
m = dict(ds=ds, method="ft_abl", N=a.N, seed=0, mult=a.mult, imgsz=a.imgsz, train_s=secs, epochs=ep, **EM.score(st))
out = REPO / "results" / "ablation"; out.mkdir(exist_ok=True)
(out / f"{ds}_{tag}.json").write_text(json.dumps(m, indent=1)); print(m)
shutil.rmtree(work / "runs" / tag, ignore_errors=True)
