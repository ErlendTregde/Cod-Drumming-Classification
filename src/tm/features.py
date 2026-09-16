"""Feature functions for the Tsetlin Machine track.

Every feature function takes a mono float32 waveform already resampled to
TM_SAMPLE_RATE and returns a fixed-shape float array. They all share one
signature, so the training path never needs to know which one it is using and
M2/M3 representations drop in without touching it.
"""
import numpy as np

from src.tm.config import TM_ENVELOPE_FRAME_SAMPLES, TM_WINDOW_SAMPLES


def fixed_window(audio: np.ndarray, n: int = TM_WINDOW_SAMPLES) -> np.ndarray:
    """Left-align the clip at t=0: truncate if longer, zero-pad the tail if shorter.

    Left-aligned rather than centred because a standard TM has no translation
    invariance, so every clip must put its event at the same offset. This is
    why src/data/preprocess.py:load_and_preprocess, which CENTRE-pads, is not
    reused here.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) >= n:
        return audio[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(audio)] = audio
    return out


def raw(audio: np.ndarray) -> np.ndarray:
    """The waveform itself, no feature extraction — M1's control arm (TM-H2)."""
    return fixed_window(audio)


def envelope(audio: np.ndarray, frame: int = TM_ENVELOPE_FRAME_SAMPLES) -> np.ndarray:
    """RMS amplitude per short frame — the pulse-train structure from H1.

    RMS squares the signal before averaging, so this is already sign-independent:
    no separate rectification step is needed.
    """
    window = fixed_window(audio)
    frames = window.reshape(-1, frame)
    return np.sqrt((frames ** 2).mean(axis=1)).astype(np.float32)


#: Name -> feature function. Adding an M2/M3 representation means adding a
#: function here, not changing the training path.
FEATURES = {"raw": raw, "envelope": envelope}
