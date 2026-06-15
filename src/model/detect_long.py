"""Detect short events in a long WAV and embed each one with Perch.

Detector-first counterpart to `src/model/embed_long.py`. Instead of blindly
tiling the file into 5s windows (which dilutes millisecond-long events past the
point Perch can see them), this runs an energy/onset detector to find the short
candidate events, cuts each one, zero-pads it to 5s exactly like the training
clips, and embeds it. Runs as its own process (TensorFlow only — no PyTorch) for
the same segfault-avoidance reason as `extract.py`.

`detect.py` invokes this via subprocess when the cache is missing; you can also
run it directly:

    uv run python -m src.model.detect_long path/to/long.wav
"""
import argparse
from pathlib import Path

import numpy as np

from src.data.config import DETECTION_CACHE_DIR, PERCH_SAMPLE_RATE
from src.data.preprocess import detect_events, extract_event_windows, load_mono_resampled
from src.model.perch import detection_cache_path, embed_arrays, load_perch_model


def main():
    parser = argparse.ArgumentParser(description="Detect + embed events in a long WAV")
    parser.add_argument("wav", type=Path, help="Path to the long WAV file")
    args = parser.parse_args()

    print(f"Loading {args.wav.name} (resampling to {PERCH_SAMPLE_RATE} Hz)...")
    audio = load_mono_resampled(args.wav)

    print("Detecting events...")
    events = detect_events(audio)
    print(f"  {len(events)} events detected")

    starts = np.array([s / PERCH_SAMPLE_RATE for s, _ in events], dtype=np.float32)
    ends = np.array([e / PERCH_SAMPLE_RATE for _, e in events], dtype=np.float32)

    if events:
        # Single-pass: embed each event ONCE at its native detected extent (+ a
        # small pad), so each sound is judged at its own measured duration. (An
        # earlier multi-scale variant embedded at 10/50/200 ms and took the most
        # confident scale; src/data/analysis showed the 10 ms crop collapses every
        # class to `click`, so this single-pass path is the supported one.)
        # Stream in chunks: a 5s window is 640 KB, so materializing every event
        # window at once blows up RAM on dense (click) files. Embeddings are tiny.
        model = load_perch_model()
        chunk = 512
        parts = []
        for i in range(0, len(events), chunk):
            print(f"  events {i}/{len(events)}")
            w = extract_event_windows(audio, events[i : i + chunk])  # (c, samples)
            parts.append(embed_arrays(model, w))
        embeddings = np.concatenate(parts)
    else:
        embeddings = np.empty((0, 1536), dtype=np.float32)

    DETECTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = detection_cache_path(args.wav)
    np.savez(out_path, embeddings=embeddings, starts=starts, ends=ends)
    print(f"Saved {embeddings.shape} embeddings (events × dim) + times → {out_path}")


if __name__ == "__main__":
    main()
