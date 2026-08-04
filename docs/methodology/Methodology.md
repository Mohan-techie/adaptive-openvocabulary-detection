# Methodology

## Objective

Develop an annotation-budget-aware framework that recommends the most suitable object detection strategy under different labeling constraints.

---

# Overall Pipeline

Dataset
↓

Annotation Budget Selection
↓

Learning Strategy Recommendation
↓

Object Detection Model

↓

Evaluation

---

# Stage 1: Dataset Preparation

Datasets considered:

- COCO
- Pascal VOC

Different annotation budgets will be simulated.

Example:

- 0 labels/class
- 10 labels/class
- 25 labels/class
- 50 labels/class
- 100 labels/class

---

# Stage 2: Strategy Selection

Depending on the annotation budget, different learning strategies are evaluated.

Possible strategies:

- Zero-shot Open Vocabulary Detection
- Teacher–Student Learning
- Semi-supervised Learning
- Supervised Fine-tuning

---

# Stage 3: Object Detection

Representative models include:

Zero-shot

- YOLO-World
- Grounding DINO
- OWLv2

Teacher–Student

- Unbiased Teacher
- Soft Teacher

Fine-tuning

- YOLOv8
- RT-DETR

---

# Stage 4: Evaluation Metrics

Detection Performance

- mAP@50
- mAP@50:95

Efficiency

- Annotation Cost
- Annotation Time

Resource Usage

- GPU Time
- Training Time

---

# Stage 5: Comparative Analysis

Compare

- Detection Accuracy
- Annotation Cost
- Training Time
- Cost vs Accuracy Tradeoff

under every annotation budget.

---

# Expected Output

Instead of proposing another detector,

the framework provides

- the best strategy
- expected performance
- annotation cost

for a given annotation budget.

---

# Research Contribution

A unified benchmark for identifying the transition point where

Zero-shot

↓

Teacher–Student

↓

Supervised Fine-tuning

becomes the optimal learning strategy.
