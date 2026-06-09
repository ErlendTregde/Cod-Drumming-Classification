from pathlib import Path
from typing import Callable

import numpy as np

from src.data.config import CACHE_DIR, PERCH_MODEL_NAME
from src.data.loader import AudioSample
from src.data.preprocess import load_and_preprocess


def load_perch_model(model_name: str = PERCH_MODEL_NAME):
    """Load Perch 2.0. Downloads weights automatically on first call.

    Importing TensorFlow has to stay out of any process that also runs PyTorch
    training — the two segfault when both are loaded. Extraction therefore runs
    in its own process (see `src/model/extract.py`); training never calls this.
    """
    from perch_hoplite.zoo import model_configs
    return model_configs.load_model_by_name(model_name)


def _cache_key(sample: AudioSample) -> str:
    return f"{sample.split}_{sample.label}_{sample.path.stem}"


def load_cached_embeddings(
    samples: list[AudioSample], cache_dir: Path = CACHE_DIR
) -> dict[str, np.ndarray]:
    """Load already-cached embeddings from disk (no model needed).

    Returns a dict mapping str(sample.path) → (1536,) embedding, skipping any
    samples whose embedding is not yet cached.
    """
    results = {}
    for s in samples:
        cache_path = cache_dir / f"{_cache_key(s)}.npy"
        if cache_path.exists():
            results[str(s.path)] = np.load(cache_path)
    return results


def missing_embeddings(
    samples: list[AudioSample], cache_dir: Path = CACHE_DIR
) -> list[AudioSample]:
    """Return samples that do not yet have a cached embedding."""
    return [s for s in samples if not (cache_dir / f"{_cache_key(s)}.npy").exists()]


def get_embedding(model, audio: np.ndarray) -> np.ndarray:
    """Run Perch on a single (PERCH_WINDOW_SAMPLES,) float32 array.

    Returns a (1536,) embedding averaged over time frames.
    """
    import tensorflow as tf
    outputs = model.embed(tf.constant(audio[np.newaxis], dtype=tf.float32))
    emb = np.array(outputs.embeddings)
    return emb.mean(axis=1).squeeze(0)


def extract_and_cache(
    samples: list[AudioSample],
    model_loader: Callable[[], object],
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Extract Perch embeddings for all samples, caching each to disk.

    `model_loader` is called lazily — the (heavy, TensorFlow) Perch model is only
    loaded if at least one sample actually needs computing, so a fully-cached run
    never touches TensorFlow. Returns a dict mapping str(sample.path) → embedding.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    model = None

    for i, sample in enumerate(samples, 1):
        cache_path = cache_dir / f"{_cache_key(sample)}.npy"

        if cache_path.exists() and not force:
            emb = np.load(cache_path)
        else:
            if model is None:
                model = model_loader()
            print(f"  [{i}/{len(samples)}] embedding {sample.split}/{sample.label}/{sample.path.name}")
            audio = load_and_preprocess(sample.path)
            emb = get_embedding(model, audio)
            np.save(cache_path, emb)

        results[str(sample.path)] = emb

    return results
