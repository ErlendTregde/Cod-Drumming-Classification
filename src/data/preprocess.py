from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from src.data.config import PERCH_SAMPLE_RATE, PERCH_WINDOW_SAMPLES


def load_and_preprocess(
    path: Path,
    target_sr: int = PERCH_SAMPLE_RATE,
    target_samples: int = PERCH_WINDOW_SAMPLES,
) -> np.ndarray:
    """Load a WAV file, resample to target_sr, and center-pad or trim to target_samples.

    Returns a float32 array of shape (target_samples,).
    """
    audio, src_sr = sf.read(path, dtype="float32", always_2d=False)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if src_sr != target_sr:
        audio = _resample(audio, src_sr, target_sr)

    return _pad_or_trim(audio, target_samples)


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    g = gcd(src_sr, dst_sr)
    return resample_poly(audio, dst_sr // g, src_sr // g).astype(np.float32)


def _pad_or_trim(audio: np.ndarray, n: int) -> np.ndarray:
    if len(audio) >= n:
        start = (len(audio) - n) // 2
        return audio[start : start + n]
    deficit = n - len(audio)
    left = deficit // 2
    right = deficit - left
    return np.pad(audio, (left, right), mode="constant").astype(np.float32)
