#!/usr/bin/env python
"""Aggregate raw results -> tables (LaTeX), figures (PDF), summary.json."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from aovd import config as C, data as D, evalmap as EM, selector as SEL, cost as COST

RAW = REPO / "results" / "raw"
OUT = REPO / "paper"
(OUT / "figs").mkdir(parents=True, exist_ok=True)
DSN = {"homeobjects": "HomeObjects-3K", "ppe": "Construction-PPE"}
LAB = {"zs": "Zero-shot (YOLOE-11s)", "ts": "TS (fixed $\\tau$)", "tsc": "TS-cal", "ft": "Fine-tune (YOLO11n)"}
COL = {"zs": "#7f7f7f", "ts": "#d95f02", "tsc": "#1b9e77", "ft": "#1f4e9c"}
MK = {"zs": "", "ts": "s", "tsc": "o", "ft": "^"}
rng = np.random.RandomState(0)


def load():
    R = defaultdict(dict)        # R[ds][(method,N,seed)] = meta
    for ds in DSN:
        for p in sorted((RAW / ds).glob("*.json")):
            m = json.loads(p.read_text())
            R[ds][(m["method"], m["N"], m["seed"])] = m
    return R


R = load()
summary = {}
BPI = {ds: D.avg_boxes_per_image(ds) for ds in DSN if ds in R and R[ds]}
summary["boxes_per_image"] = BPI


def curve(ds, method, key="map50", Ns=None):
    Ns = Ns or [0] + C.BUDGETS
    xs, mu, sd, n = [], [], [], []
    for N in Ns:
        v = [m[key] for (me, nn, s), m in R[ds].items() if me == method and nn == N]
        if v:
            xs.append(N); mu.append(np.mean(v)); sd.append(np.std(v)); n.append(len(v))
    return np.array(xs), np.array(mu), np.array(sd), np.array(n)


def zs_val(ds, key="map50"):
    m = R[ds].get(("zs", 0, -1))
    return m[key] if m else float("nan")


# ------------------------------------------------------------------ Table: main results
rows = []
for ds in DSN:
    if not R[ds]:
        continue
    summary[ds] = {"zs": {k: zs_val(ds, k) for k in ("map50", "map5095")}}
    for key in ("map50", "map5095"):
        pass
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{DSN[ds]}}} (zero-shot: mAP$_{{50}}$ {100*zs_val(ds):.1f}, "
                f"mAP$_{{50:95}}$ {100*zs_val(ds,'map5095'):.1f})}} \\\\")
    for N in [0] + C.BUDGETS:
        cells = []
        for me in ("ft", "ts", "tsc"):
            x, mu, sd, n = curve(ds, me, "map50", [N])
            x2, mu2, sd2, _ = curve(ds, me, "map5095", [N])
            cells.append("--" if len(x) == 0 else f"{100*mu[0]:.1f}$\\pm${100*sd[0]:.1f} / {100*mu2[0]:.1f}")
        if all(c == "--" for c in cells):
            continue
        ann = COST.annotation_hours(N, BPI[ds])
        rows.append(f"{N} & {ann:.2f} & " + " & ".join(cells) + " & " +
                    str(max([len([1 for (me, nn, s) in R[ds] if nn == N and me == m_]) for m_ in ("ft", "ts", "tsc")])) + " \\\\")
    rows.append("\\hline")
(OUT / "auto_main_table.tex").write_text("\n".join(rows) + "\n")

# ------------------------------------------------------------------ Figure: learning curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
for ax, ds in zip(axes, DSN):
    if not R[ds]:
        continue
    ax.axhline(100 * zs_val(ds), color=COL["zs"], ls="--", lw=1.2, label=LAB["zs"].replace("$", ""))
    for me in ("ft", "ts", "tsc"):
        x, mu, sd, n = curve(ds, me)
        if len(x) == 0:
            continue
        xp = np.where(x == 0, 5, x)
        ax.errorbar(xp, 100 * mu, yerr=100 * sd, color=COL[me], marker=MK[me], ms=4, lw=1.2, capsize=2,
                    label=LAB[me].replace("$", "").replace("\\tau", "τ"))
    ax.set_xscale("log"); ax.set_xticks([5, 10, 25, 50, 100, 200, 400]); ax.set_xticklabels(["0", "10", "25", "50", "100", "200", "400"])
    ax.set_xlabel("labeled images $N$"); ax.set_title(DSN[ds], fontsize=9); ax.grid(alpha=.3)
axes[0].set_ylabel("mAP$_{50}$ (%)")
axes[1].legend(fontsize=6.5, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "figs" / "learning_curves.pdf"); plt.close()

# ------------------------------------------------------------------ Crossover points
cross = {}
for ds in DSN:
    if not R[ds]:
        continue
    zs = zs_val(ds)
    res = {}
    for me in ("ft", "ts", "tsc"):
        x, mu, sd, n = curve(ds, me)
        above = [int(a) for a, b in zip(x, mu) if a > 0 and b >= zs]
        res[f"{me}_beats_zs_from"] = above[0] if above else None
    xf, muf, _, _ = curve(ds, "ft"); xt, mut, _, _ = curve(ds, "tsc")
    common = [int(n) for n in xf if n in xt and n > 0]
    res["ft_beats_tsc_from"] = next((n for n in common if muf[list(xf).index(n)] >= mut[list(xt).index(n)]), None)
    cross[ds] = res
summary["crossover"] = cross

# ------------------------------------------------------------------ Accuracy vs annotation cost (sensitivity over t_box)
def hours_to_reach(ds, target, method, t_box):
    x, mu, _, _ = curve(ds, method)
    for a, b in zip(x, mu):
        if b >= target:
            return COST.annotation_hours(a, BPI[ds], t_box=t_box) if method in ("ft",) else COST.annotation_hours(a, BPI[ds], t_box=t_box)
    return None

cost_tab = {}
for ds in DSN:
    if not R[ds]:
        continue
    ft = curve(ds, "ft")
    out = {}
    for frac in (0.5, 0.75, 0.9):   # fractions of the best FT mAP50 observed
        tgt = frac * ft[1].max() if len(ft[1]) else None
        if tgt is None:
            continue
        entry = {"target_map50": tgt}
        for me in ("zs", "ts", "tsc", "ft"):
            if me == "zs":
                entry[me] = {tb: (0.0 if zs_val(ds) >= tgt else None) for tb in (5, 10, 35)}
            else:
                entry[me] = {tb: hours_to_reach(ds, tgt, me, tb) for tb in (5, 10, 35)}
        out[str(frac)] = entry
    cost_tab[ds] = out
summary["hours_to_target"] = cost_tab

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, ds in zip(axes, DSN):
    if not R[ds]:
        continue
    ax.scatter([0], [100 * zs_val(ds)], color=COL["zs"], marker="*", s=60, label="Zero-shot", zorder=3)
    for me in ("ft", "ts", "tsc"):
        x, mu, _, _ = curve(ds, me)
        h = [COST.annotation_hours(a, BPI[ds]) for a in x]
        ax.plot(h, 100 * mu, color=COL[me], marker=MK[me], ms=4, lw=1.2, label=LAB[me].replace("$", "").replace("\\tau", "τ"))
    ax.set_xlabel("annotation time (h), $t_{box}$=10 s"); ax.set_title(DSN[ds], fontsize=9); ax.grid(alpha=.3)
    ax.set_xscale("symlog", linthresh=0.05)
axes[0].set_ylabel("mAP$_{50}$ (%)"); axes[1].legend(fontsize=6.5, loc="lower right")
plt.tight_layout(); plt.savefig(OUT / "figs" / "cost_accuracy.pdf"); plt.close()

# ------------------------------------------------------------------ Training / compute cost
comp = {}
for ds in DSN:
    for me in ("ft", "ts", "tsc"):
        v = [m["train_s"] for (mm, N, s), m in R[ds].items() if mm == me and "train_s" in m and "copied_from" not in m]
        if v:
            comp[f"{ds}/{me}"] = dict(mean_train_s=float(np.mean(v)), n=len(v))
summary["compute"] = comp

# ------------------------------------------------------------------ Selector evaluation (extrapolation from small budgets)
FIT_N = [10, 25, 50]
TEST_N = [100, 200, 400]
sel_rows = []
for kind in ("loglin", "power"):
    for allowed_name, allowed in (("zs/ts/ft", ["zs", "ts", "ft"]), ("zs/tsc/ft", ["zs", "tsc", "ft"])):
        recs = []
        for ds in DSN:
            seeds = sorted({s for (me, N, s) in R[ds] if me == "ft"})
            for s in seeds:
                if not all(("ft", n, s) in R[ds] and ("ts", n, s) in R[ds] for n in FIT_N):
                    continue
                curves = {"zs": ("const", zs_val(ds))}
                for me in ("ts", "tsc", "ft"):
                    ns = [n for n in FIT_N if (me, n, s) in R[ds]]
                    curves[me] = SEL.fit_curve(ns, [R[ds][(me, n, s)]["map50"] for n in ns], kind)
                for N in TEST_N:
                    if not all((me, N, s) in R[ds] for me in ("ft", "ts", "tsc")):
                        continue
                    true = {me: (zs_val(ds) if me == "zs" else R[ds][(me, N, s)]["map50"]) for me in ["zs", "ts", "tsc", "ft"]}
                    best = max(allowed, key=lambda k: true[k])
                    pick, _ = SEL.recommend(curves, N, allowed=allowed)
                    heur = "ft" if N >= 100 else "zs"
                    recs.append(dict(ds=ds, s=s, N=N, pick=pick, best=best, regret=true[best] - true[pick],
                                     reg_zs=true[best] - true["zs"], reg_ts=true[best] - true.get("ts", 0) if "ts" in allowed else true[best] - true["tsc"],
                                     reg_ft=true[best] - true["ft"], reg_heur=true[best] - true[heur],
                                     hit=float(true[pick] >= true[best] - 0.005)))
        if recs:
            sel_rows.append(dict(curve=kind, allowed=allowed_name, n_cases=len(recs),
                                 hit=float(np.mean([r["hit"] for r in recs])),
                                 regret=float(np.mean([r["regret"] for r in recs])),
                                 reg_zs=float(np.mean([r["reg_zs"] for r in recs])),
                                 reg_ts=float(np.mean([r["reg_ts"] for r in recs])),
                                 reg_ft=float(np.mean([r["reg_ft"] for r in recs])),
                                 reg_heur=float(np.mean([r["reg_heur"] for r in recs]))))
summary["selector"] = sel_rows
with open(OUT / "auto_selector_table.tex", "w") as f:
    for r in sel_rows:
        f.write(f"{r['curve']} & {r['allowed']} & {r['n_cases']} & {100*r['hit']:.0f} & {100*r['regret']:.2f} & "
                f"{100*r['reg_zs']:.2f} & {100*r['reg_ts']:.2f} & {100*r['reg_ft']:.2f} & {100*r['reg_heur']:.2f} \\\\\n")

# ------------------------------------------------------------------ Pilot-set reliability of zero-shot estimate
def load_stats(ds, name):
    z = np.load(RAW / ds / f"{name}.npz")
    return EM.unpack({k: z[k] for k in z.files})

pilot = {}
for ds in DSN:
    if ("zs", 0, -1) not in R[ds]:
        continue
    st = load_stats(ds, "zs")
    full = EM.score(st)["map50"]
    ft_sets = [(s, load_stats(ds, f"ft_N100_s{s}")) for s in (0, 1, 2) if ("ft", 100, s) in R[ds]]
    out = {}
    for n in (10, 25, 50, 100):
        errs, agree = [], []
        for _ in range(200):
            idx = rng.choice(len(st), n, replace=False)
            e = EM.score([st[i] for i in idx])["map50"]
            errs.append(e - full)
            for s, fs in ft_sets[:1]:
                fz = EM.score([fs[i] for i in idx])["map50"]
                agree.append(float((e > fz) == (full > EM.score(fs)["map50"])))
        out[n] = dict(mae=float(np.mean(np.abs(errs))), bias=float(np.mean(errs)),
                      hw90=float(np.percentile(np.abs(errs), 90)), order_agree=float(np.mean(agree)) if agree else None)
    pilot[ds] = out
summary["pilot"] = pilot
Path(REPO / "results" / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: summary[k] for k in ("crossover", "selector", "pilot")}, indent=1, default=str))
