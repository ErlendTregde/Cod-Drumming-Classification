"""Run the trained classifier over a long unannotated recording.

Slides a 5s window across the file, embeds each window with Perch (in a
subprocess — see src/model/embed_long.py), classifies every window, and writes
a timeline of predictions. No manual sound extraction needed: the sliding
window + classifier do the detection automatically.

    uv run python -m src.inference.infer data/unannotated/01-220213_1505_Ch2.wav
    uv run python -m src.inference.infer <wav> --classifier mlp --hop 80000   # 2.5s overlap
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.config import (
    DEFAULT_HOP_SAMPLES,
    MODEL_DIR,
    PERCH_SAMPLE_RATE,
    PERCH_WINDOW_SAMPLES,
    RESULTS_DIR,
)
from src.model.classifier import load_model
from src.model.perch import inference_cache_path
from src.training.train import label_names, predict_proba

WINDOW_SECONDS = PERCH_WINDOW_SAMPLES / PERCH_SAMPLE_RATE


def ensure_embeddings(wav: Path, hop: int, force: bool) -> np.ndarray:
    """Embed the long file in a subprocess (TensorFlow) unless already cached."""
    cache_path = inference_cache_path(wav, hop)
    if force or not cache_path.exists():
        print("Embedding file in a separate process (TensorFlow)...")
        cmd = [sys.executable, "-m", "src.model.embed_long", str(wav), "--hop", str(hop)]
        subprocess.run(cmd, check=True)
    return np.load(cache_path)


def build_rows(probs: np.ndarray, hop: int, labels: list[str]) -> list[dict]:
    """Per-window predictions: start_s, end_s, class, confidence."""
    hop_seconds = hop / PERCH_SAMPLE_RATE
    classes = probs.argmax(1)
    confidences = probs.max(1)
    rows = []
    for i, (cls, conf) in enumerate(zip(classes, confidences)):
        start = i * hop_seconds
        rows.append({
            "start_s": round(start, 2),
            "end_s": round(start + WINDOW_SECONDS, 2),
            "class": labels[cls],
            "confidence": round(float(conf), 4),
        })
    return rows


def merge_events(rows: list[dict]) -> list[dict]:
    """Collapse consecutive same-class windows into events with mean confidence."""
    events = []
    for row in rows:
        if events and events[-1]["class"] == row["class"]:
            ev = events[-1]
            ev["end_s"] = row["end_s"]
            ev["_confs"].append(row["confidence"])
        else:
            events.append({
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "class": row["class"],
                "_confs": [row["confidence"]],
            })
    for ev in events:
        ev["mean_confidence"] = round(sum(ev["_confs"]) / len(ev["_confs"]), 4)
        del ev["_confs"]
    return events


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


def plot_timeline(rows: list[dict], labels: list[str], wav_name: str, path: Path) -> None:
    """Step plot of predicted class vs time across the file."""
    idx = {name: i for i, name in enumerate(labels)}
    t = [r["start_s"] / 60 for r in rows]  # minutes
    y = [idx[r["class"]] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(t, y, where="post", color="steelblue", linewidth=1)
    ax.scatter(t, y, s=8, color="steelblue")
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="time (minutes)",
        ylabel="predicted class",
        title=f"Prediction timeline — {wav_name}",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved timeline → {path}")


def main():
    parser = argparse.ArgumentParser(description="Classify a long unannotated recording")
    parser.add_argument("wav", type=Path, help="Path to the long unannotated WAV file")
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP_SAMPLES,
                        help=f"Window step in samples (default {DEFAULT_HOP_SAMPLES} = 5s, no overlap)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-embed the file even if cached")
    args = parser.parse_args()

    print(f"Preparing embeddings for {args.wav.name}...")
    embeddings = ensure_embeddings(args.wav, args.hop, args.force_recompute)
    print(f"  {embeddings.shape[0]} windows embedded")

    model_path = MODEL_DIR / f"classifier_{args.classifier}.pt"
    print(f"Loading classifier from {model_path}...")
    model = load_model(model_path)
    labels = label_names()

    probs = predict_proba(model, embeddings)
    rows = build_rows(probs, args.hop, labels)
    events = merge_events(rows)

    out_dir = RESULTS_DIR / "inference" / args.wav.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "windows.csv", ["start_s", "end_s", "class", "confidence"])
    write_csv(events, out_dir / "events.csv", ["start_s", "end_s", "class", "mean_confidence"])
    plot_timeline(rows, labels, args.wav.name, out_dir / "timeline.png")

    counts = {lbl: sum(r["class"] == lbl for r in rows) for lbl in labels}
    print("\nWindow counts by class:")
    for lbl, n in counts.items():
        print(f"  {lbl:8s} {n}")
    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
