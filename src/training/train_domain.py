"""Train the domain-adapted classifier on detector-cropped events from long files.

The long-file counterpart to `main.py`. Where `main.py` trains on the
clean clips, this trains on events the *detector* cut from the long `_all`
recordings (labelled from their selection tables), which matches what the model sees
at inference. Same clean pipeline shape:

    extract (subprocess, TF) -> build arrays -> train -> save -> evaluate (held-out)

    uv run python -m src.training.train_domain                # 5-class -> models/classifier_domain.pt
    uv run python -m src.training.train_domain --background   # + reject -> models/classifier_domain_bg.pt
    uv run python -m src.training.train_domain --eval-only    # just score the saved head on the val files

Embedding (Perch/TensorFlow) runs in a subprocess; this process is PyTorch-only, so
the two never share a kernel (they segfault together).
"""
import argparse
import subprocess
import sys
from collections import Counter

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.data.config import CLASSES, DOMAIN_DATASET, DOMAIN_VAL_DIR, MODEL_DIR
from src.model.classifier import load_model, save_model
from src.training.evaluate import evaluate_long_files
from src.training.train import label_names, train_classifier


def ensure_domain_dataset(force: bool) -> None:
    """Build the detector-crop dataset in a subprocess (TensorFlow) unless cached."""
    if force or not DOMAIN_DATASET.exists():
        print("Building detector-crop dataset in a separate process (TensorFlow)...")
        subprocess.run([sys.executable, "-m", "src.model.extract_domain"], check=True)


def build_arrays(background: bool):
    """Load the cached detector-crops → (X, y_int, classes). Adds background if asked."""
    data = np.load(DOMAIN_DATASET, allow_pickle=True)
    X, ystr = data["X"].astype(np.float32), data["y"]
    classes = list(CLASSES)
    if background:
        X_bg = data["X_bg"]
        X = np.vstack([X, X_bg]).astype(np.float32)
        ystr = np.concatenate([ystr, np.array(["background"] * len(X_bg))])
        classes = CLASSES + ["background"]
    enc = LabelEncoder().fit(classes)
    print("domain train set:", X.shape, dict(Counter(ystr)))
    return X, enc.transform(ystr), list(enc.classes_)


def main():
    parser = argparse.ArgumentParser(description="Train the domain-adapted head on detector crops")
    parser.add_argument("--background", action="store_true",
                        help="add a 6th `background` class that rejects over-detections")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training; just evaluate the saved head on the val files")
    parser.add_argument("--force-recompute", action="store_true",
                        help="rebuild the detector-crop dataset even if cached")
    args = parser.parse_args()

    name = "domain_bg" if args.background else "domain"
    model_path = MODEL_DIR / f"classifier_{name}.pt"

    if args.eval_only:
        model = load_model(model_path)
        labels = CLASSES + ["background"] if args.background else list(CLASSES)
    else:
        ensure_domain_dataset(args.force_recompute)
        X, y, labels = build_arrays(args.background)

        # hold out 15% of the crops for early stopping (mirrors main.py's val split)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(X))
        n_val = max(1, int(0.15 * len(X)))
        vi, ti = idx[:n_val], idx[n_val:]

        print(f"\nTraining {name} classifier ({len(labels)} classes)...")
        model = train_classifier(X[ti], y[ti], X[vi], y[vi], "logistic", n_classes=len(labels))
        save_model(model, "logistic", X.shape[1], len(labels), model_path)
        print(f"Saved trained model → {model_path}")

    baseline = load_model(MODEL_DIR / "classifier_logistic.pt")
    evaluate_long_files(model, labels, DOMAIN_VAL_DIR,
                        baseline=baseline, baseline_labels=label_names())


if __name__ == "__main__":
    main()
