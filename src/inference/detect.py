"""Detect and classify cod sound events in a long recording (detector-first).

Single-pass baseline — the supported pipeline for long files. An energy/onset
detector finds the short events in the raw waveform (cod clicks/vocals/water are
only milliseconds long), each event is embedded ONCE at its native detected extent
(+ a small pad) with Perch, and classified with the trained model. This matches the
conditions the classifier was trained under (a short event zero-padded to 5s)
instead of blindly tiling the file into 5s windows (see src/inference/infer.py),
where millisecond events are averaged away by Perch's pooling.

An earlier variant embedded each event at several widths (10/50/200 ms) and kept
the most-confident scale. The data analysis in `src/data/analysis/` showed the
10 ms crop collapses *every* class into `click` (and the `max` rule then always
picks it), which both misclassified vocals and over-called clicks — so that
multi-scale path is kept only as a documented experiment, not the supported one.
See CLAUDE.md "Data analysis".

    uv run python -m src.inference.detect data/unannotated/01-220213_1505_Ch4_all.wav
    uv run python -m src.inference.detect <wav> --classifier mlp

Run `main.py` first so a trained model exists.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.config import MODEL_DIR, RESULTS_DIR
from src.model.classifier import load_model
from src.model.perch import detection_cache_path
from src.training.train import label_names, predict_proba


def ensure_detection(wav: Path, force: bool):
    """Detect + embed events in a subprocess (TensorFlow) unless already cached.

    Returns (embeddings (N, 1536), starts (N,), ends (N,)). A legacy multi-scale
    cache (N, scales, 1536) is detected and transparently recomputed single-pass.
    """
    cache_path = detection_cache_path(wav)
    need = force or not cache_path.exists()
    if not need:
        data = np.load(cache_path)
        if data["embeddings"].ndim != 2:
            print("Cache is multi-scale (legacy) — recomputing single-pass...")
            need = True
    if need:
        print("Detecting + embedding in a separate process (TensorFlow)...")
        subprocess.run([sys.executable, "-m", "src.model.detect_long", str(wav)], check=True)
        data = np.load(cache_path)
    return data["embeddings"], data["starts"], data["ends"]


def classify(model, embeddings):
    """Argmax class index + confidence per event from (N, 1536) embeddings."""
    proba = predict_proba(model, embeddings)
    return proba.argmax(axis=1), proba.max(axis=1)


def build_rows(class_idx, conf, starts, ends, labels):
    """One row per detected event: start_s, end_s, duration_s, class, confidence."""
    rows = []
    for ci, cf, start, end in zip(class_idx, conf, starts, ends):
        rows.append({
            "start_s": round(float(start), 3),
            "end_s": round(float(end), 3),
            "duration_s": round(float(end - start), 3),
            "class": labels[ci],
            "confidence": round(float(cf), 4),
        })
    return rows


def write_csv(rows, path, fields):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def plot_timeline(rows, labels, wav_name, path):
    """Scatter each detected event at its time, colored by confidence."""
    idx = {name: i for i, name in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(14, 4))
    if rows:
        t = [r["start_s"] / 60 for r in rows]  # minutes
        y = [idx[r["class"]] for r in rows]
        c = [r["confidence"] for r in rows]
        sc = ax.scatter(t, y, c=c, cmap="viridis", s=25, vmin=0, vmax=1)
        fig.colorbar(sc, ax=ax, label="confidence")
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="time (minutes)",
        ylabel="predicted class",
        title=f"Detected events — {wav_name}",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved timeline → {path}")


def main():
    parser = argparse.ArgumentParser(description="Detect + classify events in a long recording")
    parser.add_argument("wav", type=Path, help="Path to the long WAV file")
    parser.add_argument("--classifier", choices=["logistic", "mlp", "domain"], default="logistic",
                        help="head to use: logistic/mlp (clean-clip) or domain (detector-crop, best)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-detect + re-embed even if cached")
    args = parser.parse_args()

    print(f"Preparing detections for {args.wav.name}...")
    embeddings, starts, ends = ensure_detection(args.wav, args.force_recompute)
    print(f"  {len(embeddings)} events embedded")

    out_dir = RESULTS_DIR / "detection" / args.wav.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = label_names()
    fields = ["start_s", "end_s", "duration_s", "class", "confidence"]
    if len(embeddings) == 0:
        print("No events detected — try lowering DETECT_THRESHOLD_K in config.py.")
        write_csv([], out_dir / "events.csv", fields)
        plot_timeline([], labels, args.wav.name, out_dir / "timeline.png")
        return

    model_path = MODEL_DIR / f"classifier_{args.classifier}.pt"
    print(f"Loading classifier from {model_path}...")
    model = load_model(model_path)

    class_idx, conf = classify(model, embeddings)
    rows = build_rows(class_idx, conf, starts, ends, labels)

    write_csv(rows, out_dir / "events.csv", fields)
    plot_timeline(rows, labels, args.wav.name, out_dir / "timeline.png")

    counts = {lbl: sum(r["class"] == lbl for r in rows) for lbl in labels}
    print("\nDetected events by class:")
    for lbl in labels:
        print(f"  {lbl:8s} {counts[lbl]}")
    print(f"\n{len(rows)} total detected events")
    print(f"All outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
