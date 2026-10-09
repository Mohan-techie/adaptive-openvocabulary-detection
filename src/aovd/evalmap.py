"""COCO-style box mAP (101-point AP, IoU 0.50:0.95) shared by every strategy.

Per-image match statistics are kept so that any subset of images (e.g. a small
labeled pilot set) can be re-scored without re-running a detector.
"""
import numpy as np
from ultralytics.utils.metrics import ap_per_class

IOUV = np.linspace(0.5, 0.95, 10)


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    ab = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def gt_xyxy(lab, hw):
    """lab: (n,5) cls,xc,yc,w,h normalised -> cls (n,), boxes xyxy in pixels."""
    h, w = hw
    if len(lab) == 0:
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    xc, yc, bw, bh = (lab[:, 1] * w, lab[:, 2] * h, lab[:, 3] * w, lab[:, 4] * h)
    box = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], 1)
    return lab[:, 0].astype(int), box


def match(pred, gcls, gbox):
    """pred: (n,6) xyxy,conf,cls -> tp (n,10) bool (Ultralytics matching rule)."""
    n = len(pred)
    correct = np.zeros((n, len(IOUV)), dtype=bool)
    if n == 0 or len(gcls) == 0:
        return correct
    iou = _iou(gbox, pred[:, :4]) * (gcls[:, None] == pred[None, :, 5].astype(int))
    for i, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], i] = True
    return correct


def image_stats(pred, lab, hw):
    gcls, gbox = gt_xyxy(lab, hw)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 6)
    return dict(tp=match(pred, gcls, gbox), conf=pred[:, 4], pcls=pred[:, 5].astype(int), tcls=gcls)


def score(stats):
    """Aggregate a list of per-image stat dicts -> dict of metrics."""
    tcls = np.concatenate([s["tcls"] for s in stats]) if stats else np.zeros(0)
    if len(tcls) == 0:
        return dict(map50=float("nan"), map5095=float("nan"), p=float("nan"), r=float("nan"))
    tp = np.concatenate([s["tp"] for s in stats])
    conf = np.concatenate([s["conf"] for s in stats])
    pcls = np.concatenate([s["pcls"] for s in stats])
    if len(conf) == 0:
        return dict(map50=0.0, map5095=0.0, p=0.0, r=0.0)
    res = ap_per_class(tp, conf, pcls, tcls)
    p, r, ap = res[2], res[3], res[5]
    return dict(map50=float(ap[:, 0].mean()), map5095=float(ap.mean()), p=float(p.mean()), r=float(r.mean()))


def pack(stats):
    """Compress a list of per-image stats into flat arrays (for np.savez_compressed)."""
    return dict(
        tp=np.concatenate([s["tp"] for s in stats]) if stats else np.zeros((0, 10), bool),
        conf=np.concatenate([s["conf"] for s in stats]).astype(np.float16),
        pcls=np.concatenate([s["pcls"] for s in stats]).astype(np.int8),
        tcls=np.concatenate([s["tcls"] for s in stats]).astype(np.int8),
        n_pred=np.array([len(s["conf"]) for s in stats]),
        n_gt=np.array([len(s["tcls"]) for s in stats]),
    )


def unpack(d):
    out, po, go = [], 0, 0
    for npd, ngt in zip(d["n_pred"], d["n_gt"]):
        out.append(dict(tp=d["tp"][po:po + npd], conf=d["conf"][po:po + npd].astype(np.float32),
                        pcls=d["pcls"][po:po + npd].astype(int), tcls=d["tcls"][go:go + ngt].astype(int)))
        po += npd
        go += ngt
    return out
