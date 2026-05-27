from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.data.config import MODEL_DIR


def evaluate(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    split_name: str = "test",
) -> None:
    """Print accuracy and per-class metrics, save confusion matrix PNG."""
    y_pred = clf.predict(X)
    accuracy = (y_pred == y).mean()

    print(f"\n=== {split_name.upper()} ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(classification_report(y, y_pred, target_names=label_names, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    out_path = MODEL_DIR / f"confusion_{split_name}.png"
    save_confusion_matrix(cm, label_names, out_path)
    print(f"Confusion matrix saved to {out_path}")


def save_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title=f"Confusion matrix — {out_path.stem}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
