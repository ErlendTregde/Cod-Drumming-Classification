"""Experiment: does augmenting Perch embeddings with cheap signal features help?

The data analysis (run.py) showed click vs vocal separate at AUC 0.98 by spectral
flatness and by pulse structure — features Perch's mean-pool partly discards. This
script tests, on the annotated clips, whether *appending* those width-independent
signal features to the 1536-d embedding improves classification (especially the
click/vocal pair) over the embedding alone.

Three arms, same training loop (src/training/train.train_classifier):
  1. embedding-only   (current production baseline)
  2. features-only    (how much do the ~8 hand features capture by themselves?)
  3. embedding + features (the augmentation)

Torch only — uses CACHED embeddings (numpy) + pure-numpy signal features, never
imports TensorFlow, so it's segfault-safe.

    uv run python -m src.data.analysis.augment_experiment
    uv run python -m src.data.analysis.augment_experiment --classifier mlp
"""
import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from src.data.analysis import signal_features as sfeat
from src.data.config import CACHE_DIR, CLASSES, DATA_DIR, RESULTS_DIR
from src.data.loader import load_dataset
from src.model.perch import load_cached_embeddings
from src.training.train import label_names, predict, train_classifier

OUT = RESULTS_DIR / "analysis" / "augmentation"
FEAT_CACHE = Path("data/analysis_feature_cache.npz")
FEATURE_NAMES = ["flatness", "pulse_strength", "log_n_pulses", "log_dur_ms",
                 "centroid_hz", "bandwidth_hz", "dominant_hz", "pulse_rate_hz"]
_enc = LabelEncoder().fit(CLASSES)  # same encoding as train.py (alphabetical)


def _vec(f: dict) -> np.ndarray:
    return np.array([
        f["flatness"], f["pulse_strength"], np.log1p(f["n_pulses"]),
        np.log1p(f["sound_duration_ms"]), f["centroid_hz"], f["bandwidth_hz"],
        f["dominant_hz"], f["pulse_rate_hz"],
    ], dtype=np.float32)


def _feat_worker(path: str) -> tuple[str, np.ndarray]:
    """Compute one clip's feature vector (top-level so it's picklable for Pool)."""
    a, sr = sfeat.load_clip(path)
    if len(a) > 6 * sr:
        a = a[: 6 * sr]
    return path, _vec(sfeat.features_from_audio(a, sr))


def compute_features(paths: list[str], workers: int | None = None) -> dict[str, np.ndarray]:
    """Signal-feature vector per clip, computed in parallel across CPU cores.

    Pure numpy/scipy (FFT pulse + spectral features) — not a GPU workload, but
    embarrassingly parallel, so we fan it out over the machine's cores instead of
    running single-threaded. Results cached to one npz (long clips truncated to 6s).
    """
    cached = {}
    if FEAT_CACHE.exists():
        d = np.load(FEAT_CACHE, allow_pickle=True)
        cached = {str(p): x for p, x in zip(d["paths"], d["X"])}
    missing = [p for p in paths if p not in cached]
    if workers is None:
        workers = min(64, os.cpu_count() or 8)
    print(f"  signal features: {len(cached)} cached, {len(missing)} to compute on {workers} cores")
    if missing:
        with Pool(workers) as pool:
            for i, (p, v) in enumerate(pool.imap_unordered(_feat_worker, missing, chunksize=16)):
                cached[p] = v
                if i % 1000 == 0 and i:
                    print(f"    {i}/{len(missing)}")
        ps = list(cached)
        np.savez(FEAT_CACHE, paths=np.array(ps), X=np.stack([cached[p] for p in ps]))
    return {p: cached[p] for p in paths}


def split_arrays(samples, embeddings, feats, split):
    ss = [s for s in samples if s.split == split]
    Xe = np.stack([embeddings[str(s.path)] for s in ss])
    Xf = np.stack([feats[str(s.path)] for s in ss])
    y = _enc.transform([s.label for s in ss])
    return Xe.astype(np.float32), Xf.astype(np.float32), y


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), label_names(), rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), label_names())
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def evaluate_arm(name, Xtr, ytr, Xva, yva, Xte, yte, classifier):
    print(f"\n===== ARM: {name} ({classifier}, dim={Xtr.shape[1]}) =====")
    model = train_classifier(Xtr, ytr, Xva, yva, classifier)
    pred_te = predict(model, Xte)
    pred_va = predict(model, Xva)
    acc_te = float((pred_te == yte).mean())
    acc_va = float((pred_va == yva).mean())
    labels = label_names()
    f1s = f1_score(yte, pred_te, labels=range(len(CLASSES)), average=None, zero_division=0)
    cm = confusion_matrix(yte, pred_te, labels=range(len(CLASSES)))
    print(classification_report(yte, pred_te, labels=range(len(CLASSES)),
                                target_names=labels, zero_division=0))
    plot_confusion(cm, f"{name} ({classifier}) — test", OUT / f"cm_{classifier}_{name}.png")
    ci, vi = labels.index("click"), labels.index("vocal")
    return dict(name=name, classifier=classifier, val_acc=acc_va, test_acc=acc_te,
                click_f1=float(f1s[ci]), vocal_f1=float(f1s[vi]),
                vocal_as_click=int(cm[vi, ci]), click_as_vocal=int(cm[ci, vi]))


def main():
    ap = argparse.ArgumentParser(description="Signal-feature augmentation experiment")
    ap.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading dataset + cached embeddings...")
    samples = load_dataset(DATA_DIR)
    embeddings = load_cached_embeddings(samples, CACHE_DIR)
    samples = [s for s in samples if str(s.path) in embeddings]
    feats = compute_features([str(s.path) for s in samples])

    Xe_tr, Xf_tr, ytr = split_arrays(samples, embeddings, feats, "train")
    Xe_va, Xf_va, yva = split_arrays(samples, embeddings, feats, "val")
    Xe_te, Xf_te, yte = split_arrays(samples, embeddings, feats, "test")
    print(f"  train {len(ytr)}  val {len(yva)}  test {len(yte)}  | emb dim {Xe_tr.shape[1]}, feat dim {Xf_tr.shape[1]}")

    # standardize signal features with TRAIN stats
    mu, sd = Xf_tr.mean(0), Xf_tr.std(0) + 1e-6
    Zf_tr, Zf_va, Zf_te = (Xf_tr - mu) / sd, (Xf_va - mu) / sd, (Xf_te - mu) / sd

    arms = {
        "embedding": (Xe_tr, Xe_va, Xe_te),
        "features":  (Zf_tr, Zf_va, Zf_te),
        "embedding+features": (np.hstack([Xe_tr, Zf_tr]),
                               np.hstack([Xe_va, Zf_va]),
                               np.hstack([Xe_te, Zf_te])),
    }
    results = [evaluate_arm(name, tr, ytr, va, yva, te, yte, args.classifier)
               for name, (tr, va, te) in arms.items()]

    # summary table
    lines = [f"# Signal-feature augmentation — {args.classifier}\n",
             "Test set; click/vocal are the pair the analysis flagged. "
             "`vocal→click` = true vocals misclassified as click (and vice-versa).\n",
             "| arm | val acc | test acc | click F1 | vocal F1 | vocal→click | click→vocal |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['val_acc']:.3f} | {r['test_acc']:.3f} | "
                     f"{r['click_f1']:.3f} | {r['vocal_f1']:.3f} | {r['vocal_as_click']} | {r['click_as_vocal']} |")
    report = "\n".join(lines)
    (OUT / f"report_{args.classifier}.md").write_text(report + "\n")
    print("\n" + report)
    print(f"\nSaved → {OUT}/report_{args.classifier}.md")


if __name__ == "__main__":
    main()
