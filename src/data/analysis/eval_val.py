"""Evaluate the domain-adapted head on the held-out VALIDATION recordings.

`data/unannotated/val/` holds recordings NOT seen during training. For each, detect
events single-pass (embedding runs in a TF subprocess if not cached), classify with
both the domain-adapted head and the clean-clip baseline, and score against the
selection table (greedy 1-to-1). Confirms the domain-adaptation gain generalizes to
fresh data. Torch-only in this process; Perch runs in the subprocess.

    uv run python -m src.data.analysis.eval_val
"""
from pathlib import Path

from src.data.config import MODEL_DIR
from src.inference.detect import ensure_detection
from src.inference.evaluate_detection import find_selection_table, load_selection_table, score
from src.model.classifier import load_model
from src.training.train import label_names, predict

VAL_DIR = Path("data/unannotated/val")
TEST_RECS = {"01-220224_1200_Ch4", "01-220301_1434_Ch6", "01-220412_1221_Ch6"}


def eval_model(model, emb, starts, ends, gt):
    labels = label_names()
    pred = predict(model, emb)
    dets = [(float(s), float(e), labels[c]) for s, e, c in zip(starts, ends, pred)]
    r = score(gt, dets, 0.5)
    return r["n_correct"], r["confusion"]


def main():
    domain = load_model(MODEL_DIR / "classifier_domain.pt")
    base = load_model(MODEL_DIR / "classifier_logistic.pt")
    wavs = sorted(VAL_DIR.glob("*_all.wav"))
    print(f"{len(wavs)} validation files\n")
    print(f"{'file':30} {'true':>5} {'clean-clip':>13} {'domain':>13}")

    tot_n = tot_b = tot_d = 0
    conf_b_all, conf_d_all = {}, {}
    for wav in wavs:
        table = find_selection_table(wav)
        if not table:
            print(f"{wav.stem}: no table — skip")
            continue
        emb, starts, ends = ensure_detection(wav, force=False)
        gt = load_selection_table(table)
        n = len(gt)
        if n == 0:
            continue
        nb, cb = eval_model(base, emb, starts, ends, gt)
        nd, cd = eval_model(domain, emb, starts, ends, gt)
        tot_n += n; tot_b += nb; tot_d += nd
        for k, v in cb.items(): conf_b_all[k] = conf_b_all.get(k, 0) + v
        for k, v in cd.items(): conf_d_all[k] = conf_d_all.get(k, 0) + v
        flag = "  (also a test file)" if wav.name[:-len("_all.wav")] in TEST_RECS else ""
        print(f"{wav.stem:30} {n:>5} {nb:>4}/{n:<4}({100*nb/n:>3.0f}%) "
              f"{nd:>4}/{n:<4}({100*nd/n:>3.0f}%){flag}")

    print(f"\n{'TOTAL':30} {tot_n:>5} {tot_b:>4}/{tot_n:<4}({100*tot_b/tot_n:>3.0f}%) "
          f"{tot_d:>4}/{tot_n:<4}({100*tot_d/tot_n:>3.0f}%)")
    print("\nper-class correct (clean → domain), aggregated over val:")
    for cls in ("vocal", "click", "water", "other", "silence"):
        tb = sum(v for (t, _), v in conf_b_all.items() if t == cls)
        if not tb:
            continue
        cb = conf_b_all.get((cls, cls), 0); cd = conf_d_all.get((cls, cls), 0)
        print(f"  {cls:8} {cb:>4}/{tb:<4} → {cd:>4}/{tb:<4}")


if __name__ == "__main__":
    main()
