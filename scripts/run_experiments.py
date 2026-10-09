#!/usr/bin/env python
"""Run all strategies (ZS / FT / TS / TS-cal) over annotation budgets, seeds and one dataset.

Resumable: finished runs (json present) are skipped.
Usage: python scripts/run_experiments.py --dataset ppe --seeds 0 1 2 --threads 2
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--data-root", default=None)
ap.add_argument("--weights-dir", default=None)
ap.add_argument("--workdir", default="/tmp/claude-0/work/exp")
ap.add_argument("--budgets", type=int, nargs="+", default=None)
args = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(args.threads)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import torch
torch.set_num_threads(args.threads)
from aovd import config as C, data as D, detectors as DET, evalmap as EM, train as TR
if args.data_root: C.DATA_ROOT = Path(args.data_root)
if args.weights_dir: C.WEIGHTS_DIR = Path(args.weights_dir)
os.chdir(REPO)

ds = args.dataset
budgets = args.budgets or C.BUDGETS
raw = REPO / "results" / "raw" / ds
raw.mkdir(parents=True, exist_ok=True)
work = Path(args.workdir) / ds
work.mkdir(parents=True, exist_ok=True)
eval_files = D.eval_images(ds)
val_list = work / "val.txt"
val_list.write_text("\n".join(eval_files) + "\n")


def save(name, meta, stats):
    meta.update(EM.score(stats))
    np.savez_compressed(raw / f"{name}.npz", **EM.pack(stats))
    (raw / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[{ds}] {name}: mAP50={meta['map50']:.3f} mAP50-95={meta['map5095']:.3f} "
          f"train_s={meta.get('train_s', 0):.0f}", flush=True)


teacher = DET.Teacher(ds)
t0 = time.time()
tp_eval = teacher.predict(eval_files)
print(f"[{ds}] teacher on eval set done ({time.time() - t0:.0f}s)", flush=True)
if not (raw / "zs.json").exists():
    save("zs", dict(ds=ds, method="zs", N=0, seed=-1, n_eval=len(eval_files)), DET.stats_for(tp_eval, eval_files))


def run_student(method, N, seed, thr, pool, pseudo, name):
    n_gold = N if method == "ft" else N
    if method == "ft":
        files = pool[:N]
        lst = D.build_mixed_dataset(work / name, ds, files, N, {})
    else:
        lst = D.build_mixed_dataset(work / name, ds, pool, N, pseudo)
    n_img = len(files) if method == "ft" else len(pool)
    w, secs, epochs = TR.train_student(ds, lst, n_img, seed, name, work, str(val_list))
    preds = DET.student_predict(w, eval_files)
    stats = DET.stats_for(preds, eval_files)
    if name == "ft_N400_s0":
        shutil.copy(w, C.WEIGHTS_DIR / f"student_ft400_{ds}.pt")
    shutil.rmtree(work / "runs" / name, ignore_errors=True)
    shutil.rmtree(work / name, ignore_errors=True)
    return stats, secs, epochs


for seed in args.seeds:
    pool = D.pool_for_seed(ds, seed)
    t0 = time.time()
    tp_pool = teacher.predict(pool)
    teacher_s = time.time() - t0
    print(f"[{ds}] seed {seed}: teacher on pool done ({teacher_s:.0f}s if uncached)", flush=True)
    for N in [0] + budgets:
        # ---- fine-tuning on N gold images (undefined for N=0)
        if N > 0 and not (raw / f"ft_N{N}_s{seed}.json").exists():
            stats, secs, ep = run_student("ft", N, seed, None, pool, None, f"ft_N{N}_s{seed}")
            save(f"ft_N{N}_s{seed}", dict(ds=ds, method="ft", N=N, seed=seed, train_s=secs, epochs=ep, n_train=N), stats)
        # ---- teacher-student with fixed threshold
        thr = C.PSEUDO_CONF
        if not (raw / f"ts_N{N}_s{seed}.json").exists():
            pseudo = DET.pseudo_labels(tp_pool, pool, thr)
            stats, secs, ep = run_student("ts", N, seed, thr, pool, pseudo, f"ts_N{N}_s{seed}")
            save(f"ts_N{N}_s{seed}", dict(ds=ds, method="ts", N=N, seed=seed, thr=thr, train_s=secs, epochs=ep,
                                          n_train=len(pool)), stats)
        # ---- teacher-student with threshold calibrated on the N gold images
        if N > 0 and not (raw / f"tsc_N{N}_s{seed}.json").exists():
            thr_c = DET.calibrate_threshold(tp_pool, pool[:N])
            if thr_c == thr:
                meta = json.loads((raw / f"ts_N{N}_s{seed}.json").read_text())
                meta.update(method="tsc", thr=thr_c, copied_from="ts")
                shutil.copy(raw / f"ts_N{N}_s{seed}.npz", raw / f"tsc_N{N}_s{seed}.npz")
                (raw / f"tsc_N{N}_s{seed}.json").write_text(json.dumps(meta, indent=1))
                print(f"[{ds}] tsc_N{N}_s{seed}: thr*={thr_c} == default, reused ts run", flush=True)
            else:
                pseudo = DET.pseudo_labels(tp_pool, pool, thr_c)
                stats, secs, ep = run_student("ts", N, seed, thr_c, pool, pseudo, f"tsc_N{N}_s{seed}")
                save(f"tsc_N{N}_s{seed}", dict(ds=ds, method="tsc", N=N, seed=seed, thr=thr_c, train_s=secs,
                                               epochs=ep, n_train=len(pool)), stats)
print(f"[{ds}] ALL DONE", flush=True)
