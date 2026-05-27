from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from src.data.config import CLASSES, DATA_DIR, MODEL_DIR


def plot_waveforms(data_dir: Path = DATA_DIR, split: str = "train", out_path: Path | None = None) -> None:
    """Plot raw amplitude waveform for one example clip per class."""
    fig, axes = plt.subplots(len(CLASSES), 1, figsize=(12, 10))

    for ax, cls in zip(axes, CLASSES):
        cls_dir = data_dir / split / cls
        files = sorted(cls_dir.glob("*.wav"))
        if not files:
            ax.set_ylabel(cls)
            continue

        audio, sr = sf.read(files[0], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        time = np.linspace(0, len(audio) / sr, len(audio))
        ax.plot(time, audio, linewidth=0.6, color="steelblue")
        ax.set_ylabel(cls, fontsize=10)
        ax.set_xlim(0, time[-1])
        ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
        duration_ms = len(audio) / sr * 1000
        ax.set_title(f"{files[0].name}  ({duration_ms:.1f} ms)", fontsize=8, loc="right", color="gray")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Waveforms by class — {split} split", fontsize=13)
    fig.tight_layout()

    out = out_path or (MODEL_DIR / "waveforms.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved waveforms → {out}")
