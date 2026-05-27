"""Generate all visualizations for the cod drumming dataset.

Run main.py first so embeddings are cached, then:
    uv run src/visualize/run.py
    uv run src/visualize/run.py --split val
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.data.config import CACHE_DIR, DATA_DIR, MODEL_DIR
from src.data.loader import load_dataset
from src.training.train import build_arrays, label_names, train_classifier
from src.visualize.metrics import plot_class_metrics
from src.visualize.spectrograms import plot_spectrograms
from src.visualize.tsne import plot_tsne
from src.visualize.waveforms import plot_waveforms


def _load_cached_embeddings(samples, cache_dir):
    results = {}
    missing = 0
    for s in samples:
        key = f"{s.split}_{s.label}_{s.path.stem}"
        p = cache_dir / f"{key}.npy"
        if p.exists():
            results[str(s.path)] = np.load(p)
        else:
            missing += 1
    if missing:
        print(f"  Warning: {missing} embeddings not in cache — run main.py first.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for cod drumming data")
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="Split to use for waveform/spectrogram examples (default: train)"
    )
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    args = parser.parse_args()

    print("Loading dataset...")
    samples = load_dataset(DATA_DIR)

    print("Loading cached embeddings...")
    embeddings = _load_cached_embeddings(samples, CACHE_DIR)
    have_embeddings = len(embeddings) > 0

    print(f"\nPlotting waveforms ({args.split} split)...")
    plot_waveforms(DATA_DIR, split=args.split)

    print(f"Plotting spectrograms ({args.split} split)...")
    plot_spectrograms(DATA_DIR, split=args.split)

    if have_embeddings:
        print("Plotting t-SNE of Perch embeddings...")
        plot_tsne(samples, embeddings)

        print("Training classifier for metrics plot...")
        X_train, y_train = build_arrays(samples, embeddings, "train")
        X_val,   y_val   = build_arrays(samples, embeddings, "val")
        X_test,  y_test  = build_arrays(samples, embeddings, "test")
        clf = train_classifier(X_train, y_train, args.classifier)
        plot_class_metrics(clf, X_val, y_val, X_test, y_test, label_names())

    print(f"\nAll figures saved to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
