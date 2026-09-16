"""Thermometer booleanization for the Tsetlin Machine track.

Thermometer rather than one-hot because the bits preserve ordering: once
"energy > q40" is true, "energy > q20" is also true, so a clause including two
bits of one value reads back as an interval a biologist can interpret.

Quantile cuts (rather than evenly spaced min-to-max) matter here specifically:
these hydrophone recordings peak at ~1e-5 to 1e-4, so any fixed or
evenly-spaced threshold collapses every clip to the same bit pattern.

Adapted from skills/tsetlin-machine/scripts/booleanize.py, which fits cuts per
column. We add a `shared` scheme (one threshold set pooled across all columns)
because per-column cuts rescale each frame independently and destroy absolute
loudness — the main silence-vs-other cue, and a ~20x difference between classes
in this data. TM-H3 measures which wins.

Fit on TRAIN ONLY, then transform every split, or the thresholds leak test data.
"""
import numpy as np

from src.tm.config import TM_BITS

SCHEMES = ("shared", "per-column")


def fit_cuts(X: np.ndarray, bits: int = TM_BITS, scheme: str = "shared") -> list[np.ndarray]:
    """Thermometer thresholds, returned as one array per column in both schemes.

    `shared`     — quantiles pooled over every value in X, reused for all columns.
    `per-column` — quantiles computed independently for each column.

    Returning the same structure either way keeps `transform` scheme-agnostic.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {SCHEMES}")
    X = np.asarray(X, dtype=float)
    qs = np.linspace(0, 100, bits + 2)[1:-1]
    if scheme == "shared":
        shared = np.unique(np.percentile(X, qs))
        return [shared] * X.shape[1]
    return [np.unique(np.percentile(X[:, j], qs)) for j in range(X.shape[1])]


def transform(X: np.ndarray, cuts: list[np.ndarray]) -> np.ndarray:
    """Apply fitted cuts. Returns uint32, which is what TMU requires."""
    X = np.asarray(X, dtype=float)
    blocks = [(X[:, j, None] > c[None, :]) for j, c in enumerate(cuts) if c.size]
    if not blocks:
        raise ValueError("every column produced zero cuts — is X constant?")
    return np.hstack(blocks).astype(np.uint32)


def describe(cuts: list[np.ndarray], B: np.ndarray) -> str:
    """Human-readable encoding summary.

    Print this on every run: a degenerate encoding (all zeros, all ones, or a
    pile of constant bits) is obvious here and invisible in an accuracy score.
    """
    constant = int(((B == 0).all(axis=0) | (B == 1).all(axis=0)).sum())
    return (
        f"booleanized: {B.shape[0]} rows x {B.shape[1]} features "
        f"({len(cuts)} columns), bit density {B.mean():.3f}, "
        f"{constant} constant bits"
    )


def bit_labels(cuts: list[np.ndarray], prefix: str = "f") -> list[str]:
    """One human-readable name per boolean feature, e.g. "f12>4.1e-05".

    `transform` lays bits out column-major (all of column 0's cuts, then all of
    column 1's), so this must iterate in the same order to stay aligned.
    """
    return [f"{prefix}{j}>{c:.3g}" for j, col in enumerate(cuts) for c in col]
