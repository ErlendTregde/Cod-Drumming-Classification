"""Decisive test: do width-independent signal features help on the LONG-FILE events?

The augmentation gave no lift on annotated clips because those are native-width,
where Perch's embedding already separates click/vocal. The problem lives in the
long-file *detector crops*. This script trains two heads on the annotated clips —
embedding-only and embedding+features — then applies BOTH to the real detected
events of each `_all` file and scores correct-classification + vocal<->click
confusion against the Raven selection tables (honest greedy 1-to-1).

Reuses the single-pass detection caches (run detect.py on each file first).
Torch + numpy/scipy only; no TensorFlow.

    uv run python -m src.data.analysis.longfile_feature_test
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.analysis import signal_features as sfeat
from src.data.analysis.augment_experiment import (
    _enc, _vec, compute_features, split_arrays)
from src.data.config import CACHE_DIR, CLASSES, DATA_DIR
from src.data.loader import load_dataset
from src.model.perch import detection_cache_path, load_cached_embeddings
from src.inference.evaluate_detection import (
    find_selection_table, load_selection_table, score)
from src.training.train import label_names, predict, train_classifier

ALL_FILES = [
    Path("data/unannotated/01-220412_1221_Ch6_all.wav"),
    Path("data/unannotated/01-220301_1434_Ch6_all.wav"),
    Path("data/unannotated/01-220224_1200_Ch4_all.wav"),
]


def _audio_feat_worker(arg):
    audio, sr = arg
    return _vec(sfeat.features_from_audio(audio, sr))


def event_audios(wav: Path, starts, ends):
    """Read each detected event's native-rate audio slice (+2 ms pad)."""
    audios = []
    with sf.SoundFile(str(wav)) as f:
        sr, total = f.samplerate, len(f)
        pad = int(sr * 0.002)
        for b, e in zip(starts, ends):
            s = max(0, int(b * sr) - pad)
            st = min(total, int(e * sr) + pad)
            f.seek(s)
            seg = f.read(st - s, dtype="float32", always_2d=False)
            if seg.ndim > 1:
                seg = seg.mean(axis=1)
            audios.append((seg, sr))
    return audios


def score_predictions(gt, starts, ends, class_idx):
    labels = label_names()
    dets = [(float(s), float(e), labels[c]) for s, e, c in zip(starts, ends, class_idx)]
    res = score(gt, dets, tol=0.5)
    conf = res["confusion"]
    return dict(
        n_correct=res["n_correct"], n_matched=res["n_matched"],
        vocal_correct=conf.get(("vocal", "vocal"), 0),
        vocal_as_click=conf.get(("vocal", "click"), 0),
        click_correct=conf.get(("click", "click"), 0),
        click_as_vocal=conf.get(("click", "vocal"), 0),
    )


def main():
    print("Training embedding-only and embedding+features heads on annotated clips...")
    samples = load_dataset(DATA_DIR)
    embeddings = load_cached_embeddings(samples, CACHE_DIR)
    samples = [s for s in samples if str(s.path) in embeddings]
    feats = compute_features([str(s.path) for s in samples])
    Xe_tr, Xf_tr, ytr = split_arrays(samples, embeddings, feats, "train")
    Xe_va, Xf_va, yva = split_arrays(samples, embeddings, feats, "val")
    mu, sd = Xf_tr.mean(0), Xf_tr.std(0) + 1e-6

    model_emb = train_classifier(Xe_tr, ytr, Xe_va, yva, "logistic")
    model_aug = train_classifier(np.hstack([Xe_tr, (Xf_tr - mu) / sd]), ytr,
                                 np.hstack([Xe_va, (Xf_va - mu) / sd]), yva, "logistic")

    rows = []
    for wav in ALL_FILES:
        cache = detection_cache_path(wav)
        table = find_selection_table(wav)
        if not cache.exists() or not table:
            print(f"  skip {wav.name} (need detect.py output + selection table)")
            continue
        d = np.load(cache)
        emb, starts, ends = d["embeddings"], d["starts"], d["ends"]
        if emb.ndim != 2:
            print(f"  skip {wav.name}: cache is multi-scale — re-run detect.py")
            continue
        gt = load_selection_table(table)
        n_true = len(gt)
        print(f"\n{wav.stem}: {len(emb)} events, {n_true} ground-truth")

        # per-event signal features (parallel), standardized with train stats
        audios = event_audios(wav, starts, ends)
        with Pool(min(64, len(audios) or 1)) as pool:
            Xf = np.stack(pool.map(_audio_feat_worker, audios, chunksize=32))
        Xf_std = (Xf - mu) / sd

        emb_pred = predict(model_emb, emb)
        aug_pred = predict(model_aug, np.hstack([emb, Xf_std]))

        r_emb = score_predictions(gt, starts, ends, emb_pred)
        r_aug = score_predictions(gt, starts, ends, aug_pred)
        for tag, r in [("embedding", r_emb), ("embedding+features", r_aug)]:
            rows.append(dict(file=wav.stem, model=tag, n_true=n_true, **r))
            print(f"  {tag:18}: correct {r['n_correct']}/{n_true} "
                  f"({100*r['n_correct']/n_true:.0f}%)  "
                  f"vocal→vocal {r['vocal_correct']}, vocal→click {r['vocal_as_click']}; "
                  f"click→click {r['click_correct']}, click→vocal {r['click_as_vocal']}")

    print("\n================ SUMMARY (correct-classification of true events) ================")
    print(f"{'file':28} {'model':18} {'correct':>10} {'vocal→click':>12} {'click→vocal':>12}")
    for r in rows:
        print(f"{r['file']:28} {r['model']:18} {r['n_correct']:>4}/{r['n_true']:<5} "
              f"{r['vocal_as_click']:>12} {r['click_as_vocal']:>12}")


if __name__ == "__main__":
    main()
