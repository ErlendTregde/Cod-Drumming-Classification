from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.data.config import CLASSES, MODEL_DIR
from src.data.loader import AudioSample

_SPLIT_COLORS = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}


def plot_tsne(
    samples: list[AudioSample],
    embeddings: dict[str, np.ndarray],
    out_path: Path | None = None,
) -> None:
    """Project Perch embeddings to 2D with t-SNE and plot colored by class and split."""
    valid = [(s, embeddings[str(s.path)]) for s in samples if str(s.path) in embeddings]
    if not valid:
        print("No embeddings found — run main.py first.")
        return

    X = np.stack([e for _, e in valid])
    labels = [s.label for s, _ in valid]
    splits = [s.split for s, _ in valid]

    perplexity = min(30, len(valid) - 1)
    X_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(X)

    fig, (ax_class, ax_split) = plt.subplots(1, 2, figsize=(14, 6))

    # — colored by class —
    class_colors = plt.cm.tab10(np.linspace(0, 0.9, len(CLASSES)))
    for cls, color in zip(CLASSES, class_colors):
        mask = np.array([l == cls for l in labels])
        ax_class.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=cls, alpha=0.75, s=35)
    ax_class.legend(title="class", fontsize=9)
    ax_class.set_title("Colored by class")
    ax_class.axis("off")

    # — colored by split —
    for split, color in _SPLIT_COLORS.items():
        mask = np.array([s == split for s in splits])
        ax_split.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=split, alpha=0.75, s=35)
    ax_split.legend(title="split", fontsize=9)
    ax_split.set_title("Colored by split")
    ax_split.axis("off")

    fig.suptitle("t-SNE of Perch embeddings (1280 → 2D)", fontsize=13)
    fig.tight_layout()

    out = out_path or (MODEL_DIR / "tsne.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved t-SNE → {out}")
