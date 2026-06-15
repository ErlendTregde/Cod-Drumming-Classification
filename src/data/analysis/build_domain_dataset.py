"""Build a domain-adaptation training set from the train `_all` files.

Instead of the biologists' clean isolated clips, this builds training data that
matches what the model sees at inference: events cut by the **detector** from the
continuous recording, each labelled exactly from its selection table.

Per file: detect events → greedy-match each detection to a table annotation (so it
gets an exact label AND the detector's boundary) → embed only the matched crops →
pool. Output `data/domain_train.npz` (X embeddings, y label strings).

TensorFlow only (Perch) — never imports PyTorch (segfault-safe). The held-out test
recordings are skipped so the train/test split is clean.

    uv run python -m src.data.analysis.build_domain_dataset
"""
import random
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.config import PERCH_SAMPLE_RATE
from src.data.preprocess import detect_events, extract_event_windows, load_mono_resampled
from src.inference.evaluate_detection import (
    find_selection_table, greedy_match, load_selection_table, overlaps)
from src.model.perch import embed_arrays, load_perch_model

TRAIN_DIR = Path("data/unannotated/train")
TEST_RECS = {"01-220224_1200_Ch4", "01-220301_1434_Ch6", "01-220412_1221_Ch6"}
OUT = Path("data/domain_train.npz")
TOL = 0.5         # detector-event ↔ table-annotation match tolerance (s)
BG_PER_FILE = 50  # background crops sampled per file (detections overlapping NO annotation)


def main():
    wavs = sorted(TRAIN_DIR.glob("*_all.wav"))
    print(f"{len(wavs)} _all files in {TRAIN_DIR}")
    model = load_perch_model()

    X_parts, y_all, Xbg_parts, n_det_total, n_match_total = [], [], [], 0, 0
    for wi, wav in enumerate(wavs, 1):
        rec = wav.name[:-len("_all.wav")]
        if rec in TEST_RECS:
            print(f"[{wi}/{len(wavs)}] SKIP {rec} (held-out test)")
            continue
        table = find_selection_table(wav)
        if not table:
            print(f"[{wi}/{len(wavs)}] SKIP {rec} (no selection table)")
            continue

        audio = load_mono_resampled(wav)
        events = detect_events(audio)
        n_det_total += len(events)
        if not events:
            print(f"[{wi}/{len(wavs)}] {rec}: 0 events detected")
            continue

        starts = [s / PERCH_SAMPLE_RATE for s, _ in events]
        ends = [e / PERCH_SAMPLE_RATE for _, e in events]
        gt = load_selection_table(table)
        dets = list(zip(starts, ends, [None] * len(events)))
        pairs, _ = greedy_match(gt, dets, TOL)  # [(gt_idx, det_idx), ...]
        if not pairs:
            print(f"[{wi}/{len(wavs)}] {rec}: detected {len(events)}, matched 0")
            continue

        matched_events = [events[di] for _, di in pairs]
        labels = [gt[gi][2] for gi, _ in pairs]
        windows = extract_event_windows(audio, matched_events)  # (M, samples)
        emb = embed_arrays(model, windows)
        X_parts.append(emb)
        y_all.extend(labels)
        n_match_total += len(pairs)

        # background = detections overlapping NO table annotation (false positives)
        gtbe = [(b, e) for b, e, _ in gt]
        order = list(range(len(events)))
        random.Random(wi).shuffle(order)
        bg_idx = []
        for di in order:
            if not any(overlaps(b, e, starts[di], ends[di], TOL) for b, e in gtbe):
                bg_idx.append(di)
                if len(bg_idx) >= BG_PER_FILE:
                    break
        if bg_idx:
            bg_win = extract_event_windows(audio, [events[i] for i in bg_idx])
            Xbg_parts.append(embed_arrays(model, bg_win))

        print(f"[{wi}/{len(wavs)}] {rec}: detected {len(events)}, "
              f"labelled {len(pairs)} + {len(bg_idx)} bg  {dict(Counter(labels))}")

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.array(y_all)
    X_bg = np.concatenate(Xbg_parts).astype(np.float32) if Xbg_parts else np.zeros((0, 1536), np.float32)
    np.savez(OUT, X=X, y=y, X_bg=X_bg)
    print(f"\nDetected {n_det_total} events total; labelled (matched to table) {n_match_total}; "
          f"background sampled {len(X_bg)}.")
    print(f"Saved {X.shape} positives + {X_bg.shape} background → {OUT}")
    print("label counts:", dict(Counter(y)))


if __name__ == "__main__":
    main()
