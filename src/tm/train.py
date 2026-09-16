"""Train a standard Tsetlin Machine on one (feature, cut-scheme) arm."""
import argparse

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from src.data.config import CLASSES
from src.tm.booleanize import bit_labels, describe, fit_cuts, transform
from src.tm.clauses import format_clauses, top_clauses
from src.tm.config import (
    TM_CLAUSES,
    TM_EPOCHS,
    TM_MAX_INCLUDED_LITERALS,
    TM_S,
    TM_SEED,
    TM_T,
)
from src.tm.dataset import load_or_build


def epoch_summary(history: list[float], last: int = 10) -> tuple[float, float]:
    """Best and mean accuracy over the final `last` epochs.

    TM test scores drift between epochs, so a single final-epoch number is
    misleading — and selecting on the best epoch while reporting the final one
    produces a gap that looks exactly like an encoding ceiling. Reporting both
    is the convention the MixCTME paper uses.
    """
    tail = history[-last:]
    return float(np.max(tail)), float(np.mean(tail))


def train_arm(feature_name: str, scheme: str, epochs: int = TM_EPOCHS) -> dict:
    # Imported here so the test suite and the feature/booleanize modules never
    # pay TMU's import cost (or its pycuda warning noise).
    from tmu.models.classification.vanilla_classifier import TMClassifier

    data = load_or_build(feature_name)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = data["train"], data["val"], data["test"]

    cuts = fit_cuts(X_tr, scheme=scheme)            # TRAIN ONLY — never val/test
    B_tr, B_va, B_te = (transform(X, cuts) for X in (X_tr, X_va, X_te))
    print(describe(cuts, B_tr))

    tm = TMClassifier(
        number_of_clauses=TM_CLAUSES,
        T=TM_T,
        s=TM_S,
        max_included_literals=TM_MAX_INCLUDED_LITERALS,
        platform="CPU",
        weighted_clauses=False,   # unweighted +-1 votes keep clauses readable as rules
        seed=TM_SEED,             # never 0: seed=0 hangs fit() forever
    )

    val_hist, test_hist = [], []
    for e in range(epochs):
        tm.fit(B_tr, y_tr.astype(np.uint32))        # fit() has no epochs argument
        val_hist.append(float((tm.predict(B_va) == y_va).mean()))
        test_hist.append(float((tm.predict(B_te) == y_te).mean()))
        if (e + 1) % 10 == 0 or e == 0:
            print(f"  epoch {e+1:3d}  val {val_hist[-1]:.4f}  test {test_hist[-1]:.4f}")

    pred_te = tm.predict(B_te)
    val_best, val_mean = epoch_summary(val_hist)
    test_best, test_mean = epoch_summary(test_hist)
    return {
        "feature": feature_name,
        "scheme": scheme,
        "n_features": int(B_tr.shape[1]),
        "val_best10": val_best,
        "val_mean10": val_mean,
        "test_best10": test_best,
        "test_mean10": test_mean,
        "test_f1_per_class": dict(
            zip(CLASSES, f1_score(y_te, pred_te, average=None, labels=list(range(len(CLASSES)))))
        ),
        "confusion": confusion_matrix(y_te, pred_te, labels=list(range(len(CLASSES)))).tolist(),
        "history": {"val": val_hist, "test": test_hist},
        "tm": tm,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Train a standard TM on one arm.")
    p.add_argument("--features", required=True, choices=["raw", "envelope"])
    p.add_argument("--cuts", default="shared", choices=["shared", "per-column"])
    p.add_argument("--epochs", type=int, default=TM_EPOCHS)
    a = p.parse_args()

    r = train_arm(a.features, a.cuts, a.epochs)
    print(
        f"\n{r['feature']} / {r['scheme']}: "
        f"test best-of-last-10 {r['test_best10']:.4f}, mean {r['test_mean10']:.4f}"
    )
    print("per-class F1:", {c: round(float(v), 3) for c, v in r["test_f1_per_class"].items()})

    data = load_or_build(a.features)
    cuts = fit_cuts(data["train"][0], scheme=a.cuts)
    labels = bit_labels(cuts, prefix="frame" if a.features == "envelope" else "s")
    B_te, y_te = transform(data["test"][0], cuts), data["test"][1]
    for c, name in enumerate(CLASSES):
        print(f"\ntop clauses for {name}:")
        print(format_clauses(top_clauses(r["tm"], c, B_te, y_te, r["n_features"], labels=labels)))


if __name__ == "__main__":
    main()
