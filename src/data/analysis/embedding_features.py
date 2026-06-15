"""Perch-embedding-domain analysis — TensorFlow only, never imports PyTorch.

Embeds clips (and tight crops of them) with Perch and measures, in the 1536-d
mean-pooled embedding space:

  - separability (H3): which class pairs sit closest? is click<->vocal the worst?
  - scale-migration (H4): when a vocal is cropped tighter and tighter (centered on
    its energy peak), does its embedding drift toward the *click* centroid? This is
    the load-bearing claim behind "a tight vocal pulse IS a click to Perch."

All distances are cosine on the raw Perch vectors.
"""
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.data.config import PERCH_SAMPLE_RATE, PERCH_WINDOW_SAMPLES
from src.data.preprocess import _pad_or_trim, load_and_preprocess
from src.model.perch import embed_arrays


# --------------------------------------------------------------------- loading
def to_32k(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample a mono signal to Perch's 32 kHz."""
    if sr == PERCH_SAMPLE_RATE:
        return audio.astype(np.float32)
    g = gcd(sr, PERCH_SAMPLE_RATE)
    return resample_poly(audio, PERCH_SAMPLE_RATE // g, sr // g).astype(np.float32)


def load_32k(path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return to_32k(audio, sr)


def crop_window(audio32k: np.ndarray, width_ms: float) -> np.ndarray:
    """Symmetric `width_ms` crop centered on the energy peak, padded to 5 s.

    Mirrors the detector's multi-scale extraction (preprocess.extract_multiscale_
    windows) so the migration test reproduces exactly what detect.py feeds Perch.
    """
    if len(audio32k) == 0:
        return np.zeros(PERCH_WINDOW_SAMPLES, dtype=np.float32)
    width = max(1, int(PERCH_SAMPLE_RATE * width_ms / 1000))
    peak = int(np.argmax(np.abs(audio32k)))
    half = width // 2
    seg = audio32k[max(0, peak - half): peak + half]
    return _pad_or_trim(seg, PERCH_WINDOW_SAMPLES)


# ------------------------------------------------------------------- embedding
def embed_paths(model, paths, chunk: int = 256) -> np.ndarray:
    """Embed clips exactly like training (resample 32k, center-pad to 5 s)."""
    embs = []
    for i in range(0, len(paths), chunk):
        windows = np.stack([load_and_preprocess(p) for p in paths[i:i + chunk]])
        embs.append(embed_arrays(model, windows))
    return np.concatenate(embs) if embs else np.zeros((0, 1536), np.float32)


def embed_crops(model, audios32k: list[np.ndarray], width_ms: float) -> np.ndarray:
    """Embed a peak-centered `width_ms` crop of each (already-32k) signal."""
    if not audios32k:
        return np.zeros((0, 1536), np.float32)
    windows = np.stack([crop_window(a, width_ms) for a in audios32k])
    return embed_arrays(model, windows)


# -------------------------------------------------------------- geometry / dist
def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def class_centroids(emb: np.ndarray, labels, classes) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    return {c: emb[labels == c].mean(axis=0) for c in classes if (labels == c).any()}


def cosine_dist_to(emb: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Cosine distance (1 - cos sim) from each row of `emb` to one centroid."""
    return 1.0 - l2norm(emb) @ (centroid / (np.linalg.norm(centroid) + 1e-12))


def centroid_distance_matrix(centroids: dict[str, np.ndarray], classes) -> np.ndarray:
    present = [c for c in classes if c in centroids]
    M = np.zeros((len(present), len(present)), dtype=np.float32)
    for i, a in enumerate(present):
        for j, b in enumerate(present):
            M[i, j] = cosine_dist_to(centroids[a][None], centroids[b])[0]
    return M, present


def nearest_centroid_pred(emb: np.ndarray, centroids: dict[str, np.ndarray], classes):
    """Predicted class = nearest (cosine) centroid, for each row of `emb`."""
    present = [c for c in classes if c in centroids]
    dists = np.stack([cosine_dist_to(emb, centroids[c]) for c in present], axis=1)
    idx = dists.argmin(axis=1)
    return np.array([present[i] for i in idx]), present, dists
