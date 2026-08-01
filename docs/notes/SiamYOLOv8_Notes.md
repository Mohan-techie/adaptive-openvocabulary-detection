# SiamYOLOv8 (Applied Intelligence, 2025)

## Paper Information

**Title:** SiamYOLOv8: A Rapid Conditional Detection Framework for One-Shot Object Detection

**Authors:** Matthieu Desmarescaux, Wissam Kaddah, Ayman Alfalou, Isabelle Badoc

**Journal:** Applied Intelligence

**Year:** 2025

---

## Problem

Deep object detectors typically require large amounts of labeled data. In many retail and warehouse applications, new products appear frequently and may have only one labeled example, making conventional supervised object detection impractical. The paper addresses **One-Shot Object Detection (OSOD)** and **Conditional Detection**, where the goal is to detect a specific object instance given only a single reference example. SiamYOLOv8.pdf

---

## Why Previous Methods Were Not Enough?

- Conventional YOLO models require many labeled training examples.
- Standard object detectors recognize predefined classes rather than individual product instances.
- Existing one-shot approaches often sacrifice real-time performance.
- Retail environments frequently introduce new products that have little or no training data. SiamYOLOv8.pdf

---

## Proposed Method

The authors extend YOLOv8 by integrating:

- A **Siamese Network** to compare a reference image with the scene.
- A **matching module** that learns similarity between the reference object and candidate detections.
- A conditional detection pipeline capable of detecting novel product instances using only one labeled example.

The method is evaluated on retail-oriented datasets.

---

## Main Contributions

1. Proposes SiamYOLOv8 for one-shot conditional object detection.
2. Integrates a Siamese architecture into YOLOv8.
3. Introduces an evaluation protocol using RPC and Grozi-3.2k retail datasets.
4. Demonstrates significant improvements over previous one-shot detection methods.
5. Targets practical retail scenarios where new products have very limited labeled data. SiamYOLOv8.pdf

---

## Experimental Results

Evaluation datasets:

- RPC (Retail Product Checkout)
- Grozi-3.2k

Reported improvements:

- **+20.33% Average Precision (+12.41 AP)** on Grozi-3.2k.
- **+25.68% Average Precision (+17.37 AP)** on RPC.

The paper reports outperforming previous state-of-the-art one-shot object detection methods on these datasets. SiamYOLOv8.pdf

---

## Strengths

- Specifically designed for retail products.
- Excellent performance with extremely limited labeled data.
- Maintains YOLO's efficient detection pipeline.
- Practical for rapidly changing product catalogs.
- Strong benchmark improvements on retail datasets.

---

## Limitations

- Solves **one-shot conditional detection**, not open-vocabulary detection.
- Requires a reference image for the target object.
- Does not use language prompts.
- Does not compare zero-shot open-vocabulary detectors with fine-tuned detectors.
- Does not investigate annotation budget optimization.
- Does not propose an adaptive hybrid framework.

---

## Important Notes

### Similarity to Our Research

**Medium**

Although the paper operates in a low-label retail setting, it studies **one-shot conditional detection**, whereas our research focuses on **open-vocabulary detection, annotation budgets, and adaptive selection between zero-shot and fine-tuned models**.

---

### Key Takeaway

This paper demonstrates that strong performance can be achieved with extremely limited labeled data in retail environments. However, it does **not** answer our main research question:

- When is zero-shot detection sufficient?
- When should fine-tuning begin?
- Can an adaptive framework choose the best strategy based on annotation cost and expected performance?

Therefore, SiamYOLOv8 is an important related work but does not directly overlap with our proposed contribution.