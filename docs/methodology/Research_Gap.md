# Research Gap

## Existing Work
Open-vocabulary detectors (YOLO-World, Grounding DINO, OWLv2, YOLOE), teacher--student pseudo-labeling (e.g., Grounding DINO -> YOLO), and fine-tuning are each studied in isolation, usually at one fixed level of supervision.

## Common Limitations
Few studies vary the manual-label budget to locate where each strategy wins, control training compute, or give a rule for choosing among strategies before annotation money is spent.

## Opportunity
Budget-indexed comparison + a small selector + a measure of how many labeled pilot images are needed to evaluate zero-shot reliably.

## Proposed Research Direction
See `paper/paper.pdf` and `README.md`. Honest status: two datasets, two seeds, CPU-limited students; larger-scale validation (stronger teachers, retail data such as RPC) remains open.
