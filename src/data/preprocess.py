from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from src.data.config import (
    DEFAULT_HOP_SAMPLES,
    DETECT_BANDPASS_HZ,
    DETECT_FRAME_MS,
    DETECT_MAX_EVENT_MS,
    DETECT_MERGE_GAP_MS,
    DETECT_MIN_EVENT_MS,
    DETECT_PAD_MS,
    DETECT_SCALES_MS,
    DETECT_THRESHOLD_K,
    PERCH_SAMPLE_RATE,
    PERCH_WINDOW_SAMPLES,
)


def load_mono_resampled(path: Path, target_sr: int = PERCH_SAMPLE_RATE) -> np.ndarray:
    """Load a WAV file as a mono float32 array resampled to target_sr."""
    audio, src_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if src_sr != target_sr:
        audio = _resample(audio, src_sr, target_sr)
    return audio


def load_and_preprocess(
    path: Path,
    target_sr: int = PERCH_SAMPLE_RATE,
    target_samples: int = PERCH_WINDOW_SAMPLES,
) -> np.ndarray:
    """Load a WAV file, resample to target_sr, and center-pad or trim to target_samples.

    Returns a float32 array of shape (target_samples,).
    """
    return _pad_or_trim(load_mono_resampled(path, target_sr), target_samples)


def load_windows(
    path: Path,
    hop_samples: int = DEFAULT_HOP_SAMPLES,
    target_sr: int = PERCH_SAMPLE_RATE,
    window_samples: int = PERCH_WINDOW_SAMPLES,
) -> np.ndarray:
    """Slice a long WAV into consecutive 5s windows for Perch.

    Resamples to target_sr, then steps a `window_samples`-wide window by
    `hop_samples`. The final partial window is padded. Returns a float32 array
    of shape (n_windows, window_samples).
    """
    audio = load_mono_resampled(path, target_sr)

    windows = []
    for start in range(0, max(len(audio), 1), hop_samples):
        windows.append(_pad_or_trim(audio[start : start + window_samples], window_samples))
        if start + window_samples >= len(audio):
            break

    return np.stack(windows)


def detect_events(
    audio: np.ndarray,
    sr: int = PERCH_SAMPLE_RATE,
    bandpass_hz: tuple[float, float] = DETECT_BANDPASS_HZ,
    frame_ms: float = DETECT_FRAME_MS,
    threshold_k: float = DETECT_THRESHOLD_K,
    min_event_ms: float = DETECT_MIN_EVENT_MS,
    merge_gap_ms: float = DETECT_MERGE_GAP_MS,
    max_event_ms: float = DETECT_MAX_EVENT_MS,
) -> list[tuple[int, int]]:
    """Find short transient events in a mono signal via band-passed energy.

    Band-passes the signal to the cod frequency range, computes a short-frame RMS
    envelope, thresholds it at ``median + threshold_k * MAD`` (robust to the few
    loud events), then groups above-threshold frames into events, merging those
    closer than ``merge_gap_ms`` and dropping those shorter than ``min_event_ms``.
    Events longer than ``max_event_ms`` are split into chunks so a rapid train of
    sounds is not collapsed into one long "noise" blob.

    Returns a list of ``(start_sample, end_sample)`` tuples in `audio` coordinates.
    """
    lo, hi = bandpass_hz
    hi = min(hi, sr / 2 - 1)  # keep below Nyquist
    sos = butter(4, (lo, hi), btype="bandpass", fs=sr, output="sos")
    filtered = sosfiltfilt(sos, audio).astype(np.float32)

    hop = max(1, int(sr * frame_ms / 1000))
    n_frames = len(filtered) // hop
    if n_frames == 0:
        return []
    frames = filtered[: n_frames * hop].reshape(n_frames, hop)
    env = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)

    med = np.median(env)
    mad = np.median(np.abs(env - med)) + 1e-12
    above = env > (med + threshold_k * mad)

    merge_gap_frames = int(merge_gap_ms / frame_ms)
    min_event_frames = max(1, int(min_event_ms / frame_ms))

    # collect raw runs of above-threshold frames (end is exclusive)
    runs: list[tuple[int, int]] = []
    f = 0
    while f < n_frames:
        if not above[f]:
            f += 1
            continue
        start = f
        while f < n_frames and above[f]:
            f += 1
        runs.append((start, f))

    # merge runs separated by a gap below merge_gap_frames, then drop short ones
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= merge_gap_frames:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    max_event_samples = int(sr * max_event_ms / 1000)
    events = []
    for s, e in merged:
        if (e - s) < min_event_frames:
            continue
        start, end = s * hop, min(e * hop, len(audio))
        # split overly long runs (e.g. a rapid click train) into chunks
        for chunk in range(start, end, max_event_samples):
            events.append((chunk, min(chunk + max_event_samples, end)))
    return events


def extract_event_windows(
    audio: np.ndarray,
    events: list[tuple[int, int]],
    sr: int = PERCH_SAMPLE_RATE,
    pad_ms: float = DETECT_PAD_MS,
    window_samples: int = PERCH_WINDOW_SAMPLES,
) -> np.ndarray:
    """Cut each detected event (plus context) and center-pad it to a 5s Perch window.

    Mirrors the training-clip preprocessing (a short event zero-padded to 5s), so
    detected events are embedded under the same conditions the classifier saw.
    Returns a float32 array of shape (len(events), window_samples).
    """
    pad = int(sr * pad_ms / 1000)
    windows = []
    for start, end in events:
        seg = audio[max(0, start - pad) : min(len(audio), end + pad)]
        windows.append(_pad_or_trim(seg, window_samples))
    return np.stack(windows) if windows else np.empty((0, window_samples), dtype=np.float32)


def extract_multiscale_windows(
    audio: np.ndarray,
    events: list[tuple[int, int]],
    scales_ms: tuple[float, ...] = DETECT_SCALES_MS,
    sr: int = PERCH_SAMPLE_RATE,
    window_samples: int = PERCH_WINDOW_SAMPLES,
) -> np.ndarray:
    """Cut each event at several widths, centered on its peak, padded to 5s.

    For each event we find the highest-energy sample and take symmetric windows
    of every width in `scales_ms` around it. The caller embeds all of them and
    keeps the most confident (width, class) — this lets clicks (~10ms) and vocals
    (~50ms) each be classified at the width that matches their native duration.
    Returns a float32 array of shape (len(events), len(scales_ms), window_samples).
    """
    scales = [max(1, int(sr * ms / 1000)) for ms in scales_ms]
    out = np.empty((len(events), len(scales), window_samples), dtype=np.float32)
    for i, (start, end) in enumerate(events):
        peak = start + int(np.argmax(np.abs(audio[start:end]))) if end > start else start
        for j, width in enumerate(scales):
            half = width // 2
            seg = audio[max(0, peak - half) : peak + half]
            out[i, j] = _pad_or_trim(seg, window_samples)
    return out


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
