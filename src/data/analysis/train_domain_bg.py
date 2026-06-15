"""Domain-adapted head WITH a `background` class, evaluated on the held-out val set.

Adds a 6th class — `background`, sampled from detector events that overlap no table
annotation (the false positives) — so the model can *reject* over-detections instead
of forcing every detection into a cod-sound class. Trains a 6-class head on the
detector-crops + background (`data/domain_train.npz`), then scores the 10 validation
recordings, reporting BOTH:

  - correct-classification of true events (a true event predicted `background` = miss), and
  - over-detection: how many detections survive as a real class vs are rejected.

Compared side-by-side against the 5-class domain head. Torch only.

    uv run python -m src.data.analysis.train_domain_bg
"""
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.config import (BATCH_SIZE, CLASSES, EARLY_STOP_PATIENCE, LEARNING_RATE,
                             MAX_EPOCHS, MODEL_DIR, TORCH_SEED, WEIGHT_DECAY)
from src.inference.detect import ensure_detection
from src.inference.evaluate_detection import find_selection_table, load_selection_table, score
from src.model.classifier import build_classifier, load_model, save_model
from src.training.train import label_names, predict

VAL_DIR = Path("data/unannotated/val")
LABELS6 = CLASSES + ["background"]
_enc6 = LabelEncoder().fit(LABELS6)
_dev = torch.device("cpu")


def train_head(Xtr, ytr, Xva, yva, n_classes, name="logistic"):
    torch.manual_seed(TORCH_SEED)
    model = build_classifier(name, Xtr.shape[1], n_classes).to(_dev)
    t = lambda a, d=torch.float32: torch.tensor(a, dtype=d, device=_dev)
    Xt, yt, Xv, yv = t(Xtr), t(ytr, torch.long), t(Xva), t(yva, torch.long)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(MAX_EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


def predict6(model, X):
    """Predicted label strings — decode through the FITTED encoder (sorted), not
    the raw LABELS6 list, or the class order is scrambled."""
    model.eval()
    with torch.no_grad():
        idx = model(torch.tensor(X, dtype=torch.float32)).argmax(1).cpu().numpy()
    return _enc6.inverse_transform(idx)


def main():
    data = np.load("data/domain_train.npz", allow_pickle=True)
    Xpos, ypos = data["X"], data["y"]
    Xbg = data["X_bg"] if "X_bg" in data else np.zeros((0, Xpos.shape[1]), np.float32)
    print(f"positives {Xpos.shape}, background {Xbg.shape}")

    X = np.vstack([Xpos, Xbg]).astype(np.float32)
    ystr = np.concatenate([ypos, np.array(["background"] * len(Xbg))])
    y = _enc6.transform(ystr)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); nval = max(1, int(0.15 * len(X)))
    vi, ti = idx[:nval], idx[nval:]
    model_bg = train_head(X[ti], y[ti], X[vi], y[vi], len(LABELS6))
    save_model(model_bg, "logistic", X.shape[1], len(LABELS6), MODEL_DIR / "classifier_domain_bg.pt")

    model_d = load_model(MODEL_DIR / "classifier_domain.pt")  # 5-class domain head
    labels5 = label_names()

    wavs = sorted(VAL_DIR.glob("*_all.wav"))
    print(f"\n{'file':30} {'true':>5}  {'domain(5)':>20}  {'domain+bg(6)':>26}")
    T = dict(n=0, det=0, cd=0, kd=0, cb=0, kb=0)  # totals
    for wav in wavs:
        table = find_selection_table(wav)
        if not table:
            continue
        emb, starts, ends = ensure_detection(wav, force=False)
        gt = load_selection_table(table); n = len(gt)
        if n == 0:
            continue
        # 5-class domain
        p5 = predict(model_d, emb)
        d5 = [(float(s), float(e), labels5[c]) for s, e, c in zip(starts, ends, p5)]
        c5 = score(gt, d5, 0.5)["n_correct"]
        kept5 = len(emb)  # 5-class never rejects
        # 6-class domain+bg
        p6 = predict6(model_bg, emb)
        d6 = [(float(s), float(e), p) for s, e, p in zip(starts, ends, p6)]
        c6 = score(gt, d6, 0.5)["n_correct"]
        kept6 = int((p6 != "background").sum())

        T["n"] += n; T["det"] += len(emb)
        T["cd"] += c5; T["kd"] += kept5; T["cb"] += c6; T["kb"] += kept6
        print(f"{wav.stem:30} {n:>5}  {c5:>4}/{n:<3}({100*c5/n:>3.0f}%) det {kept5:>5}  "
              f"{c6:>4}/{n:<3}({100*c6/n:>3.0f}%) kept {kept6:>5}/{len(emb)}")

    print(f"\n{'TOTAL':30} {T['n']:>5}")
    print(f"  domain (5-class):   correct {T['cd']}/{T['n']} ({100*T['cd']/T['n']:.0f}%)   "
          f"detections kept {T['kd']}/{T['det']} (100%)")
    print(f"  domain+bg (6-class): correct {T['cb']}/{T['n']} ({100*T['cb']/T['n']:.0f}%)   "
          f"detections kept {T['kb']}/{T['det']} ({100*T['kb']/T['det']:.0f}%  → "
          f"{100*(1-T['kb']/T['det']):.0f}% rejected as background)")


if __name__ == "__main__":
    main()
