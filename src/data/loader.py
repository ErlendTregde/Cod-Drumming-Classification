from dataclasses import dataclass
from pathlib import Path

from src.data.config import CLASSES


@dataclass
class AudioSample:
    path: Path
    label: str
    split: str


def load_dataset(data_dir: Path) -> list[AudioSample]:
    samples = []
    skipped = 0

    for wav_path in sorted(data_dir.glob("*/*/*.wav")):
        split = wav_path.parts[-3]
        label = wav_path.parts[-2]

        if label not in CLASSES:
            skipped += 1
            continue

        samples.append(AudioSample(path=wav_path, label=label, split=split))

    if skipped:
        print(f"Skipped {skipped} files with unlisted labels (e.g. NA)")

    return samples
