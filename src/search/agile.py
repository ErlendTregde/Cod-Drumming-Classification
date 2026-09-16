"""Agile vector search over a long file's detected events (Perch as intended).

Ranks the detected events of a recording by **embedding similarity to a query
example** — a class prototype built from the annotated clips, or a single example
clip — so a human can review the top matches. This is the agile-modeling workflow
Perch is built for (perch-hoplite vector search): give an example → retrieve the
most similar events → verify the top-K. No trained classifier needed.

Scoring uses perch_hoplite's cosine score (`score_functions.numpy_cos`). Pure numpy
(reads cached embeddings) — no TensorFlow/PyTorch, so it's fast and segfault-safe.

    # rank a file's detected events by similarity to the 'vocal' prototype
    uv run python -m src.search.agile data/unannotated/01-220412_1221_Ch6_all.wav --query vocal
    # ...or to a single example clip; add --evaluate to score precision@k vs the table
    uv run python -m src.search.agile <wav> --query vocal --evaluate

Run `src/inference/detect.py` on the file first (this reads its detection cache).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from perch_hoplite.db import score_functions

from src.data.config import CACHE_DIR, CLASSES, DATA_DIR, RESULTS_DIR
from src.data.loader import load_dataset
from src.model.perch import detection_cache_path, load_cached_embeddings


def load_corpus(wav: Path):
    """Detected-event embeddings + times for a long file (from detect.py's cache)."""
    cache = detection_cache_path(wav)
    if not cache.exists():
        raise SystemExit(f"No detection cache for {wav.name} — run "
                         f"`python -m src.inference.detect {wav}` first.")
    d = np.load(cache)
    emb = d["embeddings"]
    if emb.ndim != 2:
        raise SystemExit("Detection cache is multi-scale (legacy) — re-run detect.py.")
    return emb.astype(np.float32), d["starts"], d["ends"]


DOMAIN_DATASET = Path("data/domain_train.npz")


def class_prototype(label: str, source: str = "auto") -> np.ndarray:
    """Query = mean embedding of class `label` (the 'example' to search for).

    - ``source="domain"``: mean of the **detector-crop** embeddings (data/domain_train.npz).
      These are IN-DISTRIBUTION with the corpus (also detector crops), so the search
      ranks markedly better — the prototype isn't fighting the clean-clip↔detector gap.
    - ``source="clip"``: mean of the biologists' clean annotated clips.
    - ``source="auto"``: domain crops if available, else clean clips.
    """
    if source in ("auto", "domain") and DOMAIN_DATASET.exists():
        d = np.load(DOMAIN_DATASET, allow_pickle=True)
        vecs = d["X"][d["y"] == label]
        if len(vecs):
            return vecs.mean(axis=0).astype(np.float32)
        if source == "domain":
            raise SystemExit(f"No '{label}' detector-crops in {DOMAIN_DATASET}.")

    samples = [s for s in load_dataset(DATA_DIR) if s.label == label]
    embs = load_cached_embeddings(samples, CACHE_DIR)
    vecs = [embs[str(s.path)] for s in samples if str(s.path) in embs]
    if not vecs:
        raise SystemExit(f"No cached embeddings for class '{label}' — run main.py first.")
    return np.mean(np.stack(vecs), axis=0).astype(np.float32)


def clip_query(path: Path) -> np.ndarray:
    """Query = embedding of a single example clip (must already be cached)."""
    path = path.resolve()
    samples = [s for s in load_dataset(DATA_DIR) if s.path.resolve() == path]
    embs = load_cached_embeddings(samples, CACHE_DIR)
    if not samples or str(samples[0].path) not in embs:
        raise SystemExit(f"No cached embedding for {path}. Use a clip under data/annotated/ "
                         f"(run main.py to cache it), or pass a class name.")
    return embs[str(samples[0].path)].astype(np.float32)


def rank_by_similarity(corpus: np.ndarray, query: np.ndarray):
    """Cosine similarity of every corpus embedding to the query (perch_hoplite scoring).

    Returns (order, scores) where order indexes corpus from most to least similar.
    """
    scores = np.asarray(score_functions.numpy_cos(corpus, query), dtype=np.float32).reshape(-1)
    return np.argsort(-scores), scores


def precision_at_k(starts, ends, order, query_class, table, ks=(10, 25, 50, 100)):
    """Fraction of the top-K ranked events that land on a true `query_class` event.

    A ranking-quality check: if the search is good, true events of the query class
    cluster at the top. Tables are partial, so this is a lower bound.
    """
    from src.inference.evaluate_detection import load_selection_table, overlaps
    gt = [(b, e) for b, e, l in load_selection_table(table) if l == query_class]
    hit = np.array([
        any(overlaps(b, e, float(starts[i]), float(ends[i]), 0.5) for b, e in gt)
        for i in order
    ])
    return {k: float(hit[:k].sum()) / k for k in ks if k <= len(hit)}, len(gt)


def main():
    ap = argparse.ArgumentParser(description="Agile vector search over a file's detected events")
    ap.add_argument("wav", type=Path, help="long recording (needs a detection cache)")
    ap.add_argument("--query", required=True,
                    help=f"class name ({'/'.join(CLASSES)}) or path to an example clip")
    ap.add_argument("--prototype", choices=["auto", "domain", "clip"], default="auto",
                    help="for a class query: build the prototype from detector-crops (domain, "
                         "in-distribution, default) or clean clips")
    ap.add_argument("--top-k", type=int, default=25, help="how many top matches to print")
    ap.add_argument("--evaluate", action="store_true",
                    help="score precision@k vs the selection table (query must be a class)")
    args = ap.parse_args()

    corpus, starts, ends = load_corpus(args.wav)
    if args.query in CLASSES:
        query, qname = class_prototype(args.query, args.prototype), args.query
    else:
        qpath = Path(args.query)
        query, qname = clip_query(qpath), qpath.stem
    order, scores = rank_by_similarity(corpus, query)

    out_dir = RESULTS_DIR / "search" / args.wav.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ranked_{qname}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "start_s", "end_s", "duration_s", "similarity"])
        for rank, i in enumerate(order, 1):
            w.writerow([rank, round(float(starts[i]), 3), round(float(ends[i]), 3),
                        round(float(ends[i] - starts[i]), 3), round(float(scores[i]), 4)])
    print(f"Ranked {len(order)} detected events by similarity to '{qname}' → {out}")

    if args.evaluate and args.query in CLASSES:
        from src.inference.evaluate_detection import find_selection_table
        table = find_selection_table(args.wav)
        if table:
            prec, n_true = precision_at_k(starts, ends, order, args.query, table)
            print(f"\nprecision@k vs table ({n_true} true '{args.query}' events, partial → lower bound):")
            for k, p in prec.items():
                print(f"  top-{k:<4}: {p:.0%} land on a true {args.query}")

    print(f"\nTop {min(args.top_k, len(order))} matches (review these first):")
    print(f"{'rank':>4} {'start_s':>9} {'dur_s':>7} {'similarity':>11}")
    for rank, i in enumerate(order[:args.top_k], 1):
        print(f"{rank:>4} {float(starts[i]):>9.2f} {float(ends[i]-starts[i]):>7.3f} {float(scores[i]):>11.3f}")


if __name__ == "__main__":
    main()
