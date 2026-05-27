from pathlib import Path

import numpy as np

from src.data.config import CACHE_DIR, PERCH_MODEL_NAME
from src.data.loader import AudioSample
from src.data.preprocess import load_and_preprocess


def load_perch_model(model_name: str = PERCH_MODEL_NAME):
    """Load Perch 2.0. Downloads weights automatically on first call."""
    from perch_hoplite.zoo import model_configs
    return model_configs.load_model_by_name(model_name)


def get_embedding(model, audio: np.ndarray) -> np.ndarray:
    """Run Perch on a single (PERCH_WINDOW_SAMPLES,) float32 array.

    Returns a (1280,) embedding averaged over time frames.
    """
    import tensorflow as tf
    outputs = model.embed(tf.constant(audio[np.newaxis], dtype=tf.float32))
    emb = np.array(outputs.embeddings)
    return emb.mean(axis=1).squeeze(0)


def extract_and_cache(
    samples: list[AudioSample],
    model,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Extract Perch embeddings for all samples, caching each to disk.

    Returns a dict mapping str(sample.path) → (1280,) embedding.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for i, sample in enumerate(samples, 1):
        key = f"{sample.split}_{sample.label}_{sample.path.stem}"
        cache_path = cache_dir / f"{key}.npy"

        if cache_path.exists() and not force:
            emb = np.load(cache_path)
        else:
            print(f"  [{i}/{len(samples)}] embedding {sample.split}/{sample.label}/{sample.path.name}")
            audio = load_and_preprocess(sample.path)
            emb = get_embedding(model, audio)
            np.save(cache_path, emb)

        results[str(sample.path)] = emb

    return results
