# Experimental Protocol

## Objective

Evaluate which object detection strategy provides the best trade-off between detection performance and annotation effort under different labeling budgets.

---

## Dataset

**Candidate:** RPC (Retail Product Checkout Dataset)

Reason:
- Retail shelf images
- Widely used benchmark
- Suitable for open-vocabulary and fine-tuning experiments

---

## Detection Strategies

### Strategy A

Zero-Shot Detection

Models:
- YOLO-World
- Grounding DINO

Label Budget:
0 labeled images

---

### Strategy B

Teacher–Student

Teacher:
Grounding DINO

Student:
YOLOv8

Pseudo-label generation followed by supervised training.

---

### Strategy C

Fine-Tuned Detector

Model:
YOLOv8

Training using manually labeled subsets.

---

### Strategy D (Proposed)

Budget-Aware Adaptive Strategy

Automatically recommends:

- Zero-shot
- Teacher–Student
- Fine-tuning

based on the available annotation budget.

---

## Annotation Budgets

- 0 images
- 10 images/class
- 25 images/class
- 50 images/class
- 100 images/class
- 250 images/class
- Full dataset

---

## Evaluation Metrics

Detection Performance

- mAP@50
- mAP@50:95
- Precision
- Recall

Efficiency

- Annotation Time
- Training Time
- Inference Speed (FPS)

---

## Expected Outcome

Identify the annotation budget at which each detection strategy becomes the most effective.

Develop a practical recommendation framework for real-world deployment.
