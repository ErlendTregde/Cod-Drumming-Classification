"""Train the domain-adapted head on detector-crops and evaluate on held-out files.

Loads the detector-crop training set (`data/domain_train.npz`, built by
build_domain_dataset.py), trains the same PyTorch head, and scores it on the three
held-out test recordings — comparing correct-classification against the existing
clean-clip baseline (`models/classifier_logistic.pt`). Torch only (cached
embeddings), no TensorFlow.

The test files must already have single-pass detection caches (run detect.py on
them first). Output model: `models/classifier_domain.pt`.

    uv run python -m src.data.analysis.train_domain
"""
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.data.config import CLASSES, MODEL_DIR
from src.inference.evaluate_detection import find_selection_table, load_selection_table, score
from src.model.classifier import load_model, save_model
from src.model.perch import detection_cache_path
from src.training.train import label_names, predict, train_classifier

_enc = LabelEncoder().fit(CLASSES)
TEST = [
    Path("data/unannotated/01-220412_1221_Ch6_all.wav"),  # quiet vocal (baseline 29%)
    Path("data/unannotated/01-220301_1434_Ch6_all.wav"),  # vocal-heavy (baseline 64%)
    Path("data/unannotated/01-220224_1200_Ch4_all.wav"),  # click-heavy (baseline 59%)
]


def eval_on_file(model, wav):
    """Classify a file's detected events and score vs its table (greedy 1-to-1)."""
    d = np.load(detection_cache_path(wav))
    emb, starts, ends = d["embeddings"], d["starts"], d["ends"]
    gt = load_selection_table(find_selection_table(wav))
    labels = label_names()
    pred = predict(model, emb)
    dets = [(float(s), float(e), labels[c]) for s, e, c in zip(starts, ends, pred)]
    r = score(gt, dets, 0.5)
    return len(gt), r["n_correct"], r["confusion"]


def main():
    data = np.load("data/domain_train.npz", allow_pickle=True)
    X, ystr = data["X"], data["y"]
    y = _enc.transform(ystr)
    print("domain train set:", X.shape, dict(Counter(ystr)))

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    nval = max(1, int(0.15 * len(X)))
    vi, ti = idx[:nval], idx[nval:]
    model = train_classifier(X[ti], y[ti], X[vi], y[vi], "logistic")
    save_model(model, "logistic", X.shape[1], len(CLASSES), MODEL_DIR / "classifier_domain.pt")

    baseline = load_model(MODEL_DIR / "classifier_logistic.pt")
    print("\n================ HELD-OUT TEST: domain vs clean-clip baseline ================")
    print(f"{'file':28} {'true':>5} {'clean-clip':>12} {'domain':>12}")
    for wav in TEST:
        n, nc_d, conf_d = eval_on_file(model, wav)
        _, nc_b, conf_b = eval_on_file(baseline, wav)
        print(f"{wav.stem:28} {n:>5} {nc_b:>4}/{n:<3} ({100*nc_b/n:>3.0f}%) "
              f"{nc_d:>4}/{n:<3} ({100*nc_d/n:>3.0f}%)")
        for cls in ("vocal", "click", "water", "other", "silence"):
            cb, cd = conf_b.get((cls, cls), 0), conf_d.get((cls, cls), 0)
            tb = sum(v for (t, _), v in conf_b.items() if t == cls)
            if tb:
                print(f"    {cls:8} correct→{cls}:  clean {cb}/{tb}  →  domain {cd}/{tb}")


if __name__ == "__main__":
    main()
