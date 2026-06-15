"""Signal-domain characterization of cod sounds — pure numpy/scipy, no Perch.

These features test, straight from the raw waveform, what distinguishes the five
classes *before* any embedding model is involved:

  - temporal structure (H1): is a vocal a *pulse train* (several regular pulses)
    and a click a *single transient*?  -> `detect_pulses`, `pulse_periodicity`
  - spectral shape (H2): do click and vocal differ in frequency content, or only
    in timing?  -> `spectral_features` (flatness = tonal vs broadband)

`features_from_audio` bundles them into one flat dict per sound; `clip_features`
is the convenience wrapper that loads a WAV first.
"""
import numpy as np
import soundfile as sf
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from src.data.config import DETECT_BANDPASS_HZ


def load_clip(path) -> tuple[np.ndarray, int]:
    """Load a WAV at its native sample rate as mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def bandpass(audio: np.ndarray, sr: int, band=DETECT_BANDPASS_HZ) -> np.ndarray:
    """Band-pass to the cod frequency range (same band the detector uses).

    `sosfiltfilt` needs the signal longer than its edge pad; a handful of clips are
    only a few samples long (sub-millisecond cuts), so for those we return the raw
    signal rather than crash. Pad length scales with filter order (4 sections).
    """
    lo, hi = band
    hi = min(hi, sr / 2 - 1)
    sos = butter(4, (lo, hi), btype="bandpass", fs=sr, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(audio) <= padlen:
        return audio.astype(np.float32)
    return sosfiltfilt(sos, audio).astype(np.float32)


def amplitude_envelope(audio: np.ndarray, sr: int, smooth_ms: float = 1.0) -> np.ndarray:
    """Hilbert amplitude envelope, lightly moving-average smoothed."""
    if len(audio) < 2:
        return np.abs(audio)
    env = np.abs(hilbert(audio))
    w = max(1, int(sr * smooth_ms / 1000))
    if w > 1:
        env = np.convolve(env, np.ones(w) / w, mode="same")
    return env.astype(np.float32)


def sound_extent(env: np.ndarray, sr: int, rel_thresh: float = 0.1) -> tuple[int, int, float]:
    """Contiguous span above rel_thresh*max(env) — the actual sound within a clip.

    Returns (start_idx, end_idx, duration_s). Handles clips that are mostly the
    biologist's tight cut as well as long silence/other clips with quiet tails.
    """
    if env.max() <= 0:
        return 0, len(env), len(env) / sr
    above = np.where(env > rel_thresh * env.max())[0]
    if len(above) == 0:
        return 0, len(env), len(env) / sr
    s, e = int(above[0]), int(above[-1]) + 1
    return s, e, (e - s) / sr


def detect_pulses(env: np.ndarray, sr: int, min_interval_ms: float = 3.0,
                  rel_height: float = 0.25) -> tuple[int, np.ndarray]:
    """Count envelope peaks (pulses). A click ~1; a cod grunt (pulse train) many.

    `min_interval_ms` stops one transient being counted twice; `rel_height`
    ignores ripple below a fraction of the loudest pulse. Returns (n, peak_times_s).
    """
    if env.max() <= 0:
        return 0, np.array([])
    distance = max(1, int(sr * min_interval_ms / 1000))
    peaks, _ = find_peaks(env, height=rel_height * env.max(), distance=distance)
    return len(peaks), peaks / sr


def pulse_periodicity(env: np.ndarray, sr: int, min_rate_hz: float = 5.0,
                      max_rate_hz: float = 400.0, work_sr: int = 2000) -> tuple[float, float]:
    """Dominant pulse rate + regularity via envelope autocorrelation.

    Strength in [0,1] is the first autocorrelation peak inside the plausible
    pulse-rate band: high = a regular pulse train, low = a single transient or
    noise. Returns (rate_hz, strength).

    The envelope is first decimated to ~`work_sr` (pulse rates of interest are
    <=max_rate_hz, so a few kHz is ample) and the autocorrelation is computed via
    FFT — both keep this O(n log n) instead of O(n^2) on long clips.
    """
    factor = max(1, sr // work_sr)
    e = env[::factor].astype(np.float64)
    fs = sr / factor
    e = e - e.mean()
    n = len(e)
    if n < 4 or np.allclose(e, 0):
        return 0.0, 0.0
    # FFT autocorrelation (zero-padded to avoid circular wrap)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(e, m)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (ac[0] + 1e-12)
    lo_lag = max(1, int(fs / max_rate_hz))
    hi_lag = min(n - 1, int(fs / min_rate_hz))
    if hi_lag <= lo_lag:
        return 0.0, 0.0
    band = ac[lo_lag:hi_lag]
    k = int(np.argmax(band)) + lo_lag
    return fs / k, float(ac[k])


def spectral_features(audio: np.ndarray, sr: int, band=DETECT_BANDPASS_HZ) -> dict:
    """Spectral centroid, bandwidth, dominant frequency, and flatness.

    Flatness (Wiener entropy, 0 = tonal/harmonic, 1 = flat/broadband-noise) is the
    decisive click-vs-vocal *spectral* discriminator if H2 holds: a click is a
    broadband transient (flat), a vocal is tonal/harmonic (peaky).
    """
    if len(audio) < 4:
        return dict(centroid_hz=0.0, bandwidth_hz=0.0, dominant_hz=0.0, flatness=0.0)
    x = audio * np.hanning(len(audio))
    spec = np.abs(np.fft.rfft(x)) + 1e-12
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    lo, hi = band
    m = (freqs >= lo) & (freqs <= min(hi, sr / 2))
    f, p = freqs[m], spec[m]
    if len(p) == 0 or p.sum() <= 0:
        return dict(centroid_hz=0.0, bandwidth_hz=0.0, dominant_hz=0.0, flatness=0.0)
    pn = p / p.sum()
    centroid = float((f * pn).sum())
    bandwidth = float(np.sqrt(((f - centroid) ** 2 * pn).sum()))
    dominant = float(f[np.argmax(p)])
    power = p ** 2
    flatness = float(np.exp(np.mean(np.log(power))) / (np.mean(power) + 1e-12))
    return dict(centroid_hz=centroid, bandwidth_hz=bandwidth,
                dominant_hz=dominant, flatness=flatness)


def features_from_audio(audio: np.ndarray, sr: int) -> dict:
    """All signal-domain features for one already-loaded sound → flat dict.

    Band-passes once, trims to the actual sound, then measures temporal (pulse)
    and spectral structure on that core.
    """
    filt = bandpass(audio, sr)
    env_full = amplitude_envelope(filt, sr, smooth_ms=2.0)
    s, e, dur = sound_extent(env_full, sr)
    core = filt[s:e] if e > s else filt
    core_env = amplitude_envelope(core, sr, smooth_ms=1.0)
    n_pulses, _ = detect_pulses(core_env, sr)
    rate, strength = pulse_periodicity(core_env, sr)
    spec = spectral_features(core, sr)
    return dict(
        full_duration_ms=round(1000 * len(audio) / sr, 2),
        sound_duration_ms=round(1000 * dur, 2),
        n_pulses=int(n_pulses),
        pulse_rate_hz=round(rate, 1),
        pulse_strength=round(strength, 3),
        rms=round(float(np.sqrt((core ** 2).mean() + 1e-12)), 5),
        **{k: round(v, 2) for k, v in spec.items()},
    )


def clip_features(path) -> dict:
    """`features_from_audio` for a WAV file path (adds `path`)."""
    audio, sr = load_clip(path)
    return {"path": str(path), **features_from_audio(audio, sr)}
