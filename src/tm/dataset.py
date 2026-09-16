"""Turn data/annotated/ clips into feature matrices for the TM track.

Cached to .npz per feature so a rerun is fast and byte-identical — the
expensive part is decoding and resampling ~7000 WAVs, not training the TM.
"""
from pathlib import Path

import numpy as np

from src.data.config import CLASSES, DATA_DIR
from src.data.loader import AudioSample, load_dataset
from src.data.preprocess import load_mono_resampled
from src.tm.config import TM_CACHE_DIR, TM_SAMPLE_RATE
from src.tm.features import FEATURES

SPLITS = ("train", "val", "test")


def build_split(
    feature_name: str, split: str, samples: list[AudioSample]
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrix and uint32 labels for one split.

    Labels are CLASSES.index(...), never a fitted LabelEncoder — CLAUDE.md
    records a past bug where the encoder's internal sorting silently
    renumbered the classes.
    """
    fn = FEATURES[feature_name]
    rows, labels = [], []
    for s in samples:
        if s.split != split:
            continue
        audio = load_mono_resampled(s.path, target_sr=TM_SAMPLE_RATE)
        rows.append(fn(audio))
        labels.append(CLASSES.index(s.label))
    if not rows:
        raise ValueError(f"no samples for split {split!r}")
    return np.stack(rows).astype(np.float32), np.array(labels, dtype=np.uint32)


def load_or_build(
    feature_name: str, data_dir: Path = DATA_DIR, cache_dir: Path = TM_CACHE_DIR
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """All three splits for one feature, using the cache when present."""
    cache = cache_dir / f"{feature_name}.npz"
    if cache.exists():
        d = np.load(cache)
        return {s: (d[f"X_{s}"], d[f"y_{s}"]) for s in SPLITS}

    samples = load_dataset(data_dir)
    print(f"building {feature_name} features for {len(samples)} clips ...")
    out = {s: build_split(feature_name, s, samples) for s in SPLITS}

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        **{f"X_{s}": out[s][0] for s in SPLITS},
        **{f"y_{s}": out[s][1] for s in SPLITS},
    )
    print(f"cached {cache}")
    return out
