#!/usr/bin/env python
"""Generate LaTeX tables (pilot reliability, compute ablation, speed) from results/ JSON files."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1]
S = json.load(open(R / "results/summary.json"))
N = {"homeobjects": "HomeObjects-3K", "ppe": "Construction-PPE"}

# pilot reliability: MAE (mAP50 points) of zero-shot estimate on n labeled images
with open(R / "paper/auto_pilot_table.tex", "w") as f:
    for ds in N:
        p = S["pilot"][ds]
        f.write(N[ds] + " & " + " & ".join(f"{100*p[str(n)]['mae']:.1f} / {100*p[str(n)]['bias']:+.1f}" for n in (10, 25, 50, 100)) + " \\\\\n")

# compute ablation
ab = {}
for p in sorted((R / "results/ablation").glob("*.json")):
    m = json.load(open(p)); ab[(m["ds"], m["mult"], m["imgsz"])] = m
main = {}
for ds in N:
    v = [json.load(open(p)) for p in (R / f"results/raw/{ds}").glob("ft_N400_s*.json")]
    main[ds] = (sum(x["map50"] for x in v) / len(v), sum(x["map5095"] for x in v) / len(v))
with open(R / "paper/auto_ablation_table.tex", "w") as f:
    for ds in N:
        zs = json.load(open(R / f"results/raw/{ds}/zs.json"))
        cells = [f"{100*main[ds][0]:.1f}"]
        for key in ((ds, 4, 320), (ds, 1, 640)):
            cells.append(f"{100*ab[key]['map50']:.1f}" if key in ab else "--")
        f.write(N[ds] + " & " + " & ".join(cells) + f" & {100*zs['map50']:.1f} \\\\\n")

sp = json.load(open(R / "results/speed.json"))
with open(R / "paper/auto_speed_table.tex", "w") as f:
    for ds in N:
        s = sp[ds]
        f.write(f"{N[ds]} & {s['teacher_640_ms']:.0f} & {s['student_640_ms']:.0f} & {s['student_320_ms']:.0f} & "
                f"{s['teacher_640_ms']/s['student_320_ms']:.1f}$\\times$ \\\\\n")
print("ok")
