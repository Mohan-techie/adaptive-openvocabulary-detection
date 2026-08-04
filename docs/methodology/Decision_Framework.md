# Decision Framework

## Motivation

Different annotation budgets require different learning strategies.

Instead of always training the same detector, this framework recommends the most suitable strategy based on the available labeling budget.

---

# Input

- Annotation Budget
- Number of Classes
- Dataset Size
- Available Computing Resources

---

# Candidate Strategies

## Strategy 1

Zero-shot Open-Vocabulary Detection

Models:

- YOLO-World
- Grounding DINO
- OWLv2

Advantages

- No annotation required
- Immediate deployment

Limitations

- Lower accuracy on domain-specific objects

---

## Strategy 2

Teacher–Student Learning

Teacher

- Grounding DINO

Student

- YOLOv8

Advantages

- Reduces manual labeling
- Better accuracy than pure zero-shot

Limitations

- Quality depends on pseudo-labels

---

## Strategy 3

Supervised Fine-Tuning

Models

- YOLOv8
- RT-DETR

Advantages

- Highest performance with sufficient labels

Limitations

- Expensive annotation cost

---

# Decision Logic

For every annotation budget,

evaluate

- Detection Accuracy
- Annotation Cost
- Training Cost

Recommend the strategy that maximizes performance while minimizing annotation effort.

---

# Expected Output

Input

↓

Recommended Learning Strategy

↓

Expected Accuracy

↓

Expected Annotation Cost

↓

Expected Training Time

---

# Contribution

Rather than introducing another detector architecture,

this work provides a practical decision framework for selecting the most annotation-efficient object detection strategy.
