from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from src.data.config import MODEL_DIR
from src.training.train import predict


def evaluate_long_files(model, labels, wav_dir, baseline=None, baseline_labels=None, tol=0.5):
    """Score a head on held-out long recordings (detector caches + selection tables).

    For each `*_all.wav` in `wav_dir` with a selection table: classify its detected
    events, score correct-classification of the true events (greedy 1-to-1), and —
    if `labels` includes a `background` class — report how many detections were
    rejected (over-detection reduction). Optionally compares against a `baseline`
    head. Mirrors `evaluate()` but for the detector-first long-file pipeline.

    Imports are local so this module stays importable without TensorFlow.
    """
    from src.inference.detect import ensure_detection
    from src.inference.evaluate_detection import (
        find_selection_table, load_selection_table, score)

    has_bg = "background" in labels
    wavs = sorted(Path(wav_dir).glob("*_all.wav"))
    print(f"\n=== HELD-OUT LONG FILES ({wav_dir}) ===")
    hdr = f"{'file':30} {'true':>5} {'correct':>10}"
    if baseline is not None:
        hdr += f" {'baseline':>10}"
    if has_bg:
        hdr += f" {'kept/det':>14}"
    print(hdr)

    tot_true = tot_correct = tot_base = tot_kept = tot_det = 0
    for wav in wavs:
        table = find_selection_table(wav)
        if not table:
            continue
        emb, starts, ends = ensure_detection(wav, force=False)
        gt = load_selection_table(table)
        n = len(gt)
        if n == 0:
            continue
        pred = [labels[c] for c in predict(model, emb)]
        dets = [(float(s), float(e), p) for s, e, p in zip(starts, ends, pred)]
        nc = score(gt, dets, tol)["n_correct"]
        kept = sum(p != "background" for p in pred) if has_bg else len(emb)
        tot_true += n; tot_correct += nc; tot_kept += kept; tot_det += len(emb)

        line = f"{wav.stem:30} {n:>5} {nc:>4}/{n:<4}({100*nc/n:>3.0f}%)"
        if baseline is not None:
            bl = baseline_labels or labels
            bpred = [bl[c] for c in predict(baseline, emb)]
            bdets = [(float(s), float(e), p) for s, e, p in zip(starts, ends, bpred)]
            nbc = score(gt, bdets, tol)["n_correct"]
            tot_base += nbc
            line += f" {nbc:>4}/{n:<4}({100*nbc/n:>3.0f}%)"
        if has_bg:
            line += f" {kept:>6}/{len(emb):<7}"
        print(line)

    if not tot_true:
        print("(no scorable files found)")
        return
    print(f"\n{'TOTAL':30} {tot_true:>5} {tot_correct:>4}/{tot_true:<4}({100*tot_correct/tot_true:>3.0f}%)"
          + (f" {tot_base:>4}/{tot_true:<4}({100*tot_base/tot_true:>3.0f}%)" if baseline is not None else ""))
    if has_bg:
        print(f"  detections kept: {tot_kept}/{tot_det} "
              f"({100*tot_kept/tot_det:.0f}% — {100*(1-tot_kept/tot_det):.0f}% rejected as background)")


def evaluate(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    split_name: str = "test",
) -> None:
    """Print accuracy and per-class metrics, save confusion matrix PNG."""
    y_pred = predict(model, X)
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
