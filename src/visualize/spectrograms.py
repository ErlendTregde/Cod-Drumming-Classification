from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram as compute_spectrogram

from src.data.config import CLASSES, DATA_DIR, MODEL_DIR

_MAX_FREQ_HZ = 5000  # cod drumming sits below 5kHz


def plot_spectrograms(data_dir: Path = DATA_DIR, split: str = "train", out_path: Path | None = None) -> None:
    """Plot a spectrogram for one example clip per class (0–5 kHz range)."""
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 12))

    for ax, cls in zip(axes, CLASSES):
        cls_dir = data_dir / split / cls
        files = sorted(cls_dir.glob("*.wav"))
        if not files:
            ax.set_ylabel(cls)
            continue

        audio, sr = sf.read(files[0], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # nperseg must be smaller than signal length; floor to power of 2
        max_seg = max(4, len(audio) // 4)
        nperseg = min(512, 2 ** int(np.log2(max_seg)))
        noverlap = nperseg // 2

        freqs, times, Sxx = compute_spectrogram(
            audio, fs=sr, nperseg=nperseg, noverlap=noverlap
        )

        freq_mask = freqs <= _MAX_FREQ_HZ
        Sxx_db = 10 * np.log10(Sxx[freq_mask] + 1e-10)

        ax.pcolormesh(times, freqs[freq_mask], Sxx_db, shading="auto", cmap="inferno")
        ax.set_ylabel(f"{cls}\n(Hz)", fontsize=9)
        duration_ms = len(audio) / sr * 1000
        ax.set_title(f"{files[0].name}  ({duration_ms:.1f} ms)", fontsize=8, loc="right", color="gray")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Spectrograms by class — {split} split  (0–{_MAX_FREQ_HZ // 1000} kHz)", fontsize=13)
    fig.tight_layout()

    out = out_path or (MODEL_DIR / "spectrograms.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved spectrograms → {out}")
