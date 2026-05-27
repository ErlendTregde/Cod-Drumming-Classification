from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

from src.data.config import MODEL_DIR


def plot_class_metrics(
    clf,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    out_path: Path | None = None,
) -> None:
    """Grouped bar chart of per-class F1 score on val and test splits."""
    report_val = classification_report(
        y_val, clf.predict(X_val), target_names=label_names, output_dict=True, zero_division=0
    )
    report_test = classification_report(
        y_test, clf.predict(X_test), target_names=label_names, output_dict=True, zero_division=0
    )

    f1_val  = [report_val[cls]["f1-score"]  for cls in label_names]
    f1_test = [report_test[cls]["f1-score"] for cls in label_names]

    x = np.arange(len(label_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, f1_val,  width, label="val",  color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, f1_test, width, label="test", color="coral",     alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 score — val vs test")
    ax.legend()
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="0.8 target")
    ax.axhline(y=1.0, color="lightgray", linestyle="-", linewidth=0.5)

    for i, (v, t) in enumerate(zip(f1_val, f1_test)):
        ax.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color="steelblue")
        ax.text(i + width / 2, t + 0.02, f"{t:.2f}", ha="center", fontsize=8, color="coral")

    fig.tight_layout()

    out = out_path or (MODEL_DIR / "class_metrics.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved class metrics → {out}")
