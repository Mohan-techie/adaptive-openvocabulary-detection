# Adaptive Open-Vocabulary Detection

## Overview

This repository contains the research implementation for an adaptive hybrid open-vocabulary object detection framework under label scarcity.

The project investigates when zero-shot open-vocabulary detectors can replace fine-tuned detectors and explores adaptive strategies that balance annotation cost, accuracy, and deployment efficiency.

---

## Research Objectives

- Study open-vocabulary object detection under low-label settings.
- Compare zero-shot and fine-tuned detectors.
- Design an adaptive detection framework.
- Reduce annotation cost while maintaining high detection performance.
- Benchmark across multiple datasets and annotation budgets.

---

## Results (summary)

Budget-controlled comparison of zero-shot YOLOE-11s, teacher--student pseudo-labeling and fine-tuning of YOLO11n, on HomeObjects-3K and Construction-PPE, labeled budgets N = 0-400 images, 2 seeds, CPU-only students (see `paper/paper.pdf` for numbers, caveats and limitations).

- HomeObjects: zero-shot (55.3 mAP50) was not beaten by any trained student up to N = 400 (28.4); a 4x-compute ablation narrows but does not close the gap (41.5).
- Construction-PPE: zero-shot 23.6; fine-tuning on 10 labels already 27.4, 41.1 at N = 400.
- Pseudo-label students at best match the teacher; mixing noisy pseudo-labels with scarce gold labels hurt.
- Gold-calibrated pseudo-label threshold: no measurable effect. Learning-curve selector: correct in all 12 (low-difficulty) cases. 50 labeled images estimate zero-shot mAP50 to about +-3 points.

Not done: Grounding DINO / OWLv2 / YOLO-World teachers and the RPC retail dataset (not reachable from the sandbox used), a third seed, converged students.

## Reproduce

```
pip install -r requirements.txt
python scripts/fetch_data.py --data-root datasets --weights-dir models   # edit DATA_ROOT/WEIGHTS_DIR in src/aovd/config.py or pass --data-root/--weights-dir
python scripts/run_experiments.py --dataset ppe --seeds 0 1 --threads 2
python scripts/run_experiments.py --dataset homeobjects --seeds 0 1 --threads 2
python scripts/ablation_compute.py --dataset homeobjects --mult 4 --imgsz 320
python scripts/bench_speed.py
python scripts/analyze.py && python scripts/make_tables.py && bash paper/build.sh
```

Raw per-run results are in `results/raw/` (json + per-image match statistics), ablations in `results/ablation/`, summary in `results/summary.json`.

## Repository Structure

```
src/aovd/   core library (data, evaluator, detectors, training, cost model, selector)
scripts/    experiment, analysis, table and benchmark scripts
results/    raw results, cache of teacher predictions, summary
paper/      IEEEtran LaTeX source (build.sh) and compiled paper.pdf
docs/       original planning/literature notes
```

## Status

Paper draft complete (see limitations); results are small-scale.
