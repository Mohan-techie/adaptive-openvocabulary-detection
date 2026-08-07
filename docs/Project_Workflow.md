# Project Workflow

## Project Objective

Develop an annotation-budget-aware decision framework for object detection under limited labeled data by evaluating Zero-Shot, Teacher–Student, and Fine-Tuning strategies.

---

# Team Structure

| Member | Responsibility | Deliverables |
|---------|----------------|--------------|
| Member A | Dataset Pipeline | Processed dataset, annotation files, dataset configuration |
| Member B | Model Pipeline | Zero-shot models, fine-tuned models, trained weights |
| Member C | Experiment Pipeline | Evaluation, benchmarking, plots, comparison tables |
| Project Lead | Integration & Research | GitHub, methodology, analysis, paper writing, experiment design |

---

# Overall Pipeline

```

Raw Dataset
↓

Dataset Preparation
↓

Training / Zero-Shot Models
↓

Inference
↓

Evaluation
↓

Comparison
↓

Decision Framework
↓

IEEE Paper

```

---

# Module 1 — Dataset Pipeline

### Responsibilities

- Download datasets
- Organize folder structure
- Convert annotations
- Train/Validation/Test split
- Generate dataset.yaml

### Output

```

datasets/
├── raw/
├── processed/
├── annotations/
└── dataset.yaml

```

### Used By

- Model Pipeline
- Experiment Pipeline

---

# Module 2 — Model Pipeline

### Responsibilities

- Install models
- Train YOLOv8
- Run Grounding DINO
- Run YOLO-World
- Export trained weights

### Input

```

datasets/

```

### Output

```

models/
weights/
runs/

```

### Used By

- Experiment Pipeline

---

# Module 3 — Experiment Pipeline

### Responsibilities

- Run evaluation
- Compute metrics
- Generate plots
- Compare models

### Inputs

```

datasets/
weights/

```

### Output

```

results/
├── metrics.csv
├── confusion_matrix.png
├── plots/
└── benchmark_tables/

```

### Used By

- Research & Paper

---

# Module 4 — Research & Integration

### Responsibilities

- Literature review
- Novelty verification
- Methodology
- Analysis
- IEEE Paper
- GitHub management

### Inputs

```

results/
plots/
metrics/

```

### Output

```

IEEE Paper

Presentation

Research Figures

```

---

# Dependency Graph

```

Dataset Pipeline
↓

Model Pipeline
↓

Experiment Pipeline
↓

Research Analysis
↓

IEEE Paper

```

---

# GitHub Folder Structure

```

adaptive-openvocabulary-detection/

├── datasets/
├── docs/
├── experiments/
├── models/
├── notebooks/
├── results/
├── scripts/
├── src/
├── README.md
└── requirements.txt

```

---

# Git Workflow

Every member follows the same workflow.

```

Pull latest changes

↓

Create or modify files

↓

Test locally

↓

Commit with meaningful message

↓

Push to GitHub

↓

Project Lead reviews

↓

Merge

```

---

# Expected Deliverables

## Dataset Engineer

- Dataset ready
- Annotation files
- Dataset configuration

## Model Engineer

- YOLOv8 baseline
- Grounding DINO
- YOLO-World
- Trained weights

## Experiment Engineer

- Metrics
- Benchmark tables
- Figures
- Performance comparison

## Project Lead

- Research methodology
- Decision framework
- Analysis
- IEEE paper
