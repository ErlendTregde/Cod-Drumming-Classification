"""Score detect.py against ground truth for a long `_all` recording.

Ground truth is the Raven selection table (`selection_<stem>.txt`, tab-separated)
sitting next to the WAV — exact `Begin Time (s)`, `End Time (s)`, and label
(`Species`) for every annotation.

Run src/inference/detect.py on the same file first (this reads its events.csv).

    uv run python -m src.inference.evaluate_detection data/unannotated/01-220301_1434_Ch6_all.wav
    uv run python -m src.inference.evaluate_detection <wav> --truth path/to/selection_...txt
"""
import argparse
import csv
import glob
from collections import Counter
from pathlib import Path

from src.data.config import CLASSES, RESULTS_DIR


# --------------------------------------------------------------------------- IO
def find_selection_table(wav: Path) -> Path | None:
    """Look for selection_<stem>.txt next to the wav (or given via --truth)."""
    for cand in [wav.with_name(f"selection_{wav.stem}.txt"),
                 wav.with_name(f"selection_{wav.stem}.wav.txt")]:
        if cand.exists():
            return cand
    hits = glob.glob(str(wav.parent / f"selection_*{wav.stem}*.txt"))
    return Path(hits[0]) if hits else None


def load_selection_table(path: Path) -> list[tuple[float, float, str]]:
    """Parse a Raven selection table → [(begin_s, end_s, label), ...].

    Columns are matched by name so order doesn't matter. Label comes from the
    'Species' column; rows with a label outside CLASSES are skipped (with a note).
    """
    events, skipped = [], Counter()
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        begin = cols.get("begin time (s)")
        end = cols.get("end time (s)")
        label_col = cols.get("species") or cols.get("label")
        if not (begin and end and label_col):
            raise SystemExit(f"Could not find Begin/End/Species columns in {path}")
        for row in reader:
            label = (row.get(label_col) or "").strip().lower()
            try:
                b, e = float(row[begin]), float(row[end])
            except (ValueError, TypeError):
                continue
            if label in CLASSES:
                events.append((b, e, label))
            elif label:
                skipped[label] += 1
    if skipped:
        print(f"  (skipped labels not in CLASSES: {dict(skipped)})")
    return events


def load_detections(wav: Path) -> list[tuple[float, float, str]]:
    path = RESULTS_DIR / "detection" / wav.stem / "events.csv"
    if not path.exists():
        raise SystemExit(f"No detect.py output at {path} — run src/inference/detect.py first.")
    with open(path) as f:
        return [(float(r["start_s"]), float(r["end_s"]), r["class"]) for r in csv.DictReader(f)]


# --------------------------------------------------------------------- scoring
def overlaps(a0, a1, b0, b1, tol) -> bool:
    return a0 - tol <= b1 and b0 - tol <= a1


def _mid(a0, a1):
    return 0.5 * (a0 + a1)


def greedy_match(gt, detections, tol):
    """One-to-one greedy match of true events to detections.

    Each true event claims at most one detection and each detection is claimed by
    at most one true event — closest (by midpoint) overlapping pairs first. This
    is what makes recall honest: with the old "any detection within ±tol counts"
    rule, a detector that fires 35x too often scores ~100% recall purely from
    density (every true event has several stray detections sitting near it). Here
    a true event can only be credited to a single detection, so spamming
    detections no longer inflates recall — and every unclaimed detection is a
    countable false positive.

    Returns (pairs, true_to_det) where pairs is [(gt_idx, det_idx), ...] and
    true_to_det maps gt_idx -> det_idx for matched events.
    """
    cands = []
    for gi, (b, e, _) in enumerate(gt):
        gm = _mid(b, e)
        for di, (ds, de, _) in enumerate(detections):
            if overlaps(b, e, ds, de, tol):
                cands.append((abs(gm - _mid(ds, de)), gi, di))
    cands.sort()
    true_to_det, used_det, pairs = {}, set(), []
    for _, gi, di in cands:
        if gi in true_to_det or di in used_det:
            continue
        true_to_det[gi] = di
        used_det.add(di)
        pairs.append((gi, di))
    return pairs, true_to_det


def score(gt, detections, tol):
    """Greedy 1-to-1 scoring → per-class recall, precision, confusion + totals.

    Returns a dict with Counters (total, found, confusion, prec_hit, prec_total)
    plus n_matched (true events located) and n_correct (located AND right label).
    Precision denominators are ALL detections of a class, so unmatched detections
    and matched-but-misclassified ones both count against precision.
    """
    pairs, true_to_det = greedy_match(gt, detections, tol)
    det_to_true_label = {di: gt[gi][2] for gi, di in pairs}

    total, found, confusion = Counter(), Counter(), Counter()
    for gi, (b, e, label) in enumerate(gt):
        total[label] += 1
        if gi in true_to_det:
            found[label] += 1
            confusion[(label, detections[true_to_det[gi]][2])] += 1
        else:
            confusion[(label, "MISS")] += 1

    prec_hit, prec_total = Counter(), Counter()
    for di, (ds, de, pred) in enumerate(detections):
        prec_total[pred] += 1
        if det_to_true_label.get(di) == pred:
            prec_hit[pred] += 1

    return dict(total=total, found=found, confusion=confusion,
                prec_hit=prec_hit, prec_total=prec_total,
                n_matched=len(pairs), n_correct=sum(prec_hit.values()))


def main():
    parser = argparse.ArgumentParser(description="Score detect.py against ground truth")
    parser.add_argument("wav", type=Path)
    parser.add_argument("--truth", type=Path, help="Selection table (.txt); else auto-detected")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Overlap slack in seconds")
    args = parser.parse_args()

    table = args.truth or find_selection_table(args.wav)
    if not table:
        raise SystemExit(
            f"No selection table found for {args.wav.name} — expected "
            f"selection_{args.wav.stem}.txt next to it, or pass --truth path/to/table.txt")
    print(f"Ground truth: selection table {table.name}")
    gt = load_selection_table(table)
    print(f"{len(gt)} ground-truth events")

    detections = load_detections(args.wav)
    print(f"{len(detections)} detected events\n")

    out_dir = RESULTS_DIR / "detection" / args.wav.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ground_truth.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["start_s", "end_s", "label"])
        for b, e, l in sorted(gt):
            w.writerow([round(b, 3), round(e, 3), l])

    tol = args.tolerance
    res = score(gt, detections, tol)
    total, found, confusion = res["total"], res["found"], res["confusion"]
    prec_hit, prec_total = res["prec_hit"], res["prec_total"]

    # ---- over-detection headline: how much of the output is false positives?
    n_true, n_det = len(gt), len(detections)
    n_matched, n_correct = res["n_matched"], res["n_correct"]
    n_fp = n_det - n_matched
    ratio = n_det / n_true if n_true else float("nan")
    print("Detection volume (1-to-1 greedy match, tolerance ±%.2fs):" % tol)
    print(f"  ground-truth events : {n_true}")
    print(f"  detected events     : {n_det}   ({ratio:.1f} detections per true event)")
    print(f"  located a true event: {n_matched}   (recall {n_matched}/{n_true} = {100*n_matched/n_true:.1f}%)" if n_true else "")
    print(f"    ...correct label  : {n_correct}   ({100*n_correct/n_true:.1f}% of true events)" if n_true else "")
    print(f"  false positives     : {n_fp}   ({100*n_fp/n_det:.1f}% of detections land on no annotation)" if n_det else "")
    print("  NOTE: tables are only partially annotated, so some false positives are")
    print("        real-but-unmarked sounds — precision is a pessimistic lower bound.\n")

    print("Per-class results:")
    print(f"{'class':8} {'detect-recall':>13} {'precision':>12}   confusion (true -> pred)")
    for c in CLASSES:
        if not total[c] and not prec_total[c]:
            continue
        rec = f"{found[c]}/{total[c]}" if total[c] else "—"
        prec = f"{prec_hit[c]}/{prec_total[c]}" if prec_total[c] else "—"
        preds = {p: n for (l, p), n in confusion.items() if l == c}
        conf = ", ".join(f"{p}:{n}" for p, n in sorted(preds.items(), key=lambda x: -x[1]))
        print(f"{c:8} {rec:>13} {prec:>12}   {conf}")
    print(f"\nWrote ground_truth.csv → {out_dir}/")


if __name__ == "__main__":
    main()
