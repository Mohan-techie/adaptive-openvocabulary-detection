"""Dataset access, nested budget sampling, and pseudo-label dataset construction."""
import os
from pathlib import Path
import numpy as np
from . import config as C


def list_images(ds, split):
    d = C.DATA_ROOT / C.DATASETS[ds]["root"] / "images" / split
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def label_path(img):
    return str(Path(img.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt"))


def read_labels(img):
    p = label_path(img)
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    a = np.loadtxt(p, dtype=np.float32, ndmin=2)
    return a[:, :5]


def eval_images(ds):
    out = []
    for s in C.DATASETS[ds]["eval_splits"]:
        out += list_images(ds, s)
    return out


def pool_for_seed(ds, seed):
    """Seeded permutation of the training split; first POOL_SIZE form the unlabeled pool.
    Labeled subsets of size N are the first N images (nested across budgets)."""
    imgs = list_images(ds, "train")
    rng = np.random.RandomState(1000 + seed)
    perm = rng.permutation(len(imgs))
    return [imgs[i] for i in perm[:C.POOL_SIZE]]


def write_yaml(path, ds, train_list_file, val_list_file):
    names = C.DATASETS[ds]["names"]
    lines = [f"path: {Path(path).parent}", f"train: {train_list_file}", f"val: {val_list_file}", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    Path(path).write_text("\n".join(lines) + "\n")


def build_mixed_dataset(root, ds, pool, n_gold, pseudo):
    """Create root/images/train (symlinks) + root/labels/train.
    Gold labels for pool[:n_gold]; teacher pseudo-labels (dict img -> (k,5) cls,xc,yc,w,h) for the rest.
    Returns path of train list file."""
    root = Path(root)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    files = []
    for i, img in enumerate(pool):
        link = root / "images" / "train" / Path(img).name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(img)
        lab = read_labels(img) if i < n_gold else pseudo[img]
        txt = root / "labels" / "train" / (Path(img).stem + ".txt")
        txt.write_text("".join(f"{int(r[0])} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n" for r in lab))
        files.append(str(link))
    lst = root / "train.txt"
    lst.write_text("\n".join(files) + "\n")
    return str(lst)


def avg_boxes_per_image(ds):
    imgs = list_images(ds, "train")
    return float(np.mean([len(read_labels(i)) for i in imgs]))
