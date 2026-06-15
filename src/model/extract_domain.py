"""Build the domain-adaptation training set (detector crops) — TensorFlow subprocess.

The domain analog of `src/model/extract.py`: instead of embedding the biologists'
clean clips, this builds training data that matches what the model sees at inference
— events cut by the **detector** from the long `_all` recordings, each labelled from
its selection table.

Per file in `DOMAIN_TRAIN_DIR`: detect events → greedy-match each to a table
annotation (so each crop gets an exact label AND the detector's boundary) → embed
the matched crops. It also samples `DOMAIN_BG_PER_FILE` "background" crops per file
(detections overlapping no annotation = the over-detection) so the head can learn a
reject class. Held-out recordings are skipped. Output → `DOMAIN_DATASET` (X, y, X_bg).

TensorFlow only (Perch) — never imports PyTorch, so it stays segfault-safe. Run via
`train_domain.py` (which calls it in a subprocess) or directly:

    uv run python -m src.model.extract_domain
"""
import random
from collections import Counter

import numpy as np

from src.data.config import (DOMAIN_BG_PER_FILE, DOMAIN_DATASET, DOMAIN_HELDOUT_RECS,
                             DOMAIN_TRAIN_DIR, PERCH_SAMPLE_RATE)
from src.data.preprocess import detect_events, extract_event_windows, load_mono_resampled
from src.inference.evaluate_detection import (
    find_selection_table, greedy_match, load_selection_table, overlaps)
from src.model.perch import embed_arrays, load_perch_model

TOL = 0.5  # detector-event ↔ table-annotation match tolerance (s)


def main():
    wavs = sorted(DOMAIN_TRAIN_DIR.glob("*_all.wav"))
    print(f"{len(wavs)} _all files in {DOMAIN_TRAIN_DIR}")
    model = load_perch_model()

    X_parts, y_all, Xbg_parts, n_det_total, n_match_total = [], [], [], 0, 0
    for wi, wav in enumerate(wavs, 1):
        rec = wav.name[:-len("_all.wav")]
        if rec in DOMAIN_HELDOUT_RECS:
            print(f"[{wi}/{len(wavs)}] SKIP {rec} (held-out)")
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

        # positives: detector crop at its own boundary + the table's exact label
        matched_events = [events[di] for _, di in pairs]
        labels = [gt[gi][2] for gi, _ in pairs]
        X_parts.append(embed_arrays(model, extract_event_windows(audio, matched_events)))
        y_all.extend(labels)
        n_match_total += len(pairs)

        # background: detections overlapping NO table annotation (the false positives)
        gtbe = [(b, e) for b, e, _ in gt]
        order = list(range(len(events)))
        random.Random(wi).shuffle(order)
        bg_idx = []
        for di in order:
            if not any(overlaps(b, e, starts[di], ends[di], TOL) for b, e in gtbe):
                bg_idx.append(di)
                if len(bg_idx) >= DOMAIN_BG_PER_FILE:
                    break
        if bg_idx:
            bg_win = extract_event_windows(audio, [events[i] for i in bg_idx])
            Xbg_parts.append(embed_arrays(model, bg_win))

        print(f"[{wi}/{len(wavs)}] {rec}: detected {len(events)}, "
              f"labelled {len(pairs)} + {len(bg_idx)} bg  {dict(Counter(labels))}")

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.array(y_all)
    X_bg = np.concatenate(Xbg_parts).astype(np.float32) if Xbg_parts else np.zeros((0, 1536), np.float32)
    DOMAIN_DATASET.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DOMAIN_DATASET, X=X, y=y, X_bg=X_bg)
    print(f"\nDetected {n_det_total} events; labelled {n_match_total}; background {len(X_bg)}.")
    print(f"Saved {X.shape} positives + {X_bg.shape} background → {DOMAIN_DATASET}")
    print("label counts:", dict(Counter(y)))


if __name__ == "__main__":
    main()
