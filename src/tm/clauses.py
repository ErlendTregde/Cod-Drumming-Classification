"""Read learned rules out of a trained Tsetlin Machine.

Interpretability is the reason this track exists, so no TM result ships
without its clauses.

The API order is a trap, and it differs between the two calls:
    get_ta_action(clause, ta, the_class=, polarity=)   <- clause FIRST
    clause_precision(the_class, polarity, X, Y)        <- class FIRST
Neither raises when swapped; you just get a different clause, quietly.
Verified against the installed package 2026-09-16.
"""
import numpy as np


def literal_name(k: int, n_features: int, labels: list[str] | None = None) -> str:
    """TMU literal indexing: k < n_features is bit k, k >= n_features is NOT bit k-n_features."""
    negated = k >= n_features
    idx = k - n_features if negated else k
    name = labels[idx] if labels else f"b{idx}"
    return f"NOT {name}" if negated else name


def top_clauses(
    tm,
    the_class: int,
    B: np.ndarray,
    y: np.ndarray,
    n_features: int,
    top: int = 5,
    polarity: int = 0,
    labels: list[str] | None = None,
) -> list[dict]:
    """The `top` highest-precision clauses voting for `the_class`.

    Ranked by precision rather than dumped wholesale: a domain expert wants the
    few rules carrying the decision, and precision makes that choice defensible
    instead of arbitrary. clause_precision returns number_of_clauses // 2
    entries, which is why the loop is bounded that way.
    """
    precision = tm.clause_precision(the_class, polarity, B, y)
    recall = tm.clause_recall(the_class, polarity, B, y)

    rows = []
    for j in range(tm.number_of_clauses // 2):
        literals = [
            literal_name(k, n_features, labels)
            for k in range(n_features * 2)
            if tm.get_ta_action(j, k, the_class=the_class, polarity=polarity)
        ]
        rows.append(
            {
                "clause": j,
                "precision": float(np.nan_to_num(precision[j])),
                "recall": float(np.nan_to_num(recall[j])),
                "literals": literals,
            }
        )
    rows.sort(key=lambda r: r["precision"], reverse=True)
    return rows[:top]


def format_clauses(rows: list[dict], max_literals: int = 12) -> str:
    out = []
    for r in rows:
        lits = r["literals"]
        if not lits:
            body = "(empty)"
        elif len(lits) > max_literals:
            body = " AND ".join(lits[:max_literals]) + f" ... (+{len(lits)-max_literals} more)"
        else:
            body = " AND ".join(lits)
        out.append(f"  clause {r['clause']:4d}  P={r['precision']:.2f} R={r['recall']:.2f}  {body}")
    return "\n".join(out)
