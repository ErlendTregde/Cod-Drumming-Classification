"""Embed a long unannotated WAV by sliding a 5s window across it.

Runs as its own process (TensorFlow only — no PyTorch import) for the same
reason as `src/model/extract.py`: TF and PyTorch segfault when co-resident.
`infer.py` invokes this via subprocess when the cache is missing; you can also
run it directly:

    uv run python -m src.model.embed_long path/to/long.wav
    uv run python -m src.model.embed_long path/to/long.wav --hop 80000
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.config import DEFAULT_HOP_SAMPLES, INFERENCE_CACHE_DIR
from src.data.preprocess import load_windows
from src.model.perch import embed_arrays, inference_cache_path, load_perch_model


def main():
    parser = argparse.ArgumentParser(description="Embed a long WAV with a sliding 5s window")
    parser.add_argument("wav", type=Path, help="Path to the long unannotated WAV file")
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP_SAMPLES,
                        help=f"Window step in samples (default {DEFAULT_HOP_SAMPLES} = 5s, no overlap)")
    args = parser.parse_args()

    print(f"Windowing {args.wav.name} (hop={args.hop} samples)...")
    windows = load_windows(args.wav, hop_samples=args.hop)
    print(f"  {len(windows)} windows")

    model = load_perch_model()
    embeddings = embed_arrays(model, windows)

    INFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = inference_cache_path(args.wav, args.hop)
    np.save(out_path, embeddings)
    print(f"Saved {embeddings.shape} embeddings → {out_path}")


if __name__ == "__main__":
    main()
