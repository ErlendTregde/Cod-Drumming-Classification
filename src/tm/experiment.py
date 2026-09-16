"""M1: the four-arm controlled comparison (raw|envelope x shared|per-column).

Everything except the variable under test is held fixed — same clips, splits,
window, resampling, bit budget, s/T/clauses, seed and epoch-selection policy —
so a difference between arms is attributable to the arm.

Reproducibility contract: fixed seed, cached features, and a manifest recording
the exact config, library versions and git commit that produced the numbers.
Rerunning `uv run python -m src.tm.experiment` reproduces results.csv.
"""
import csv
import json
import subprocess

import numpy as np

from src.data.config import CLASSES
from src.tm import config as cfg
from src.tm.booleanize import bit_labels, fit_cuts, transform
from src.tm.clauses import format_clauses, top_clauses
from src.tm.dataset import load_or_build
from src.tm.train import train_arm

ARMS = [(f, s) for f in ("raw", "envelope") for s in ("shared", "per-column")]
OUT = cfg.TM_RESULTS_DIR / "m1"

FIELDS = [
    "feature", "scheme", "n_features",
    "val_best10", "val_mean10", "test_best10", "test_mean10",
] + [f"f1_{c}" for c in CLASSES]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _manifest() -> dict:
    import scipy
    import tmu
    return {
        "git_commit": _git_commit(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        # tmu self-reports 0.8.3 even when installed from git; the pin that
        # actually matters lives in pyproject.toml [tool.uv.sources].
        "tmu_reported_version": getattr(tmu, "__version__", "unknown"),
        "tmu_source": "git+https://github.com/cair/tmu.git (see pyproject [tool.uv.sources])",
        "config": {k: str(v) for k, v in vars(cfg).items() if k.startswith("TM_")},
        "classes": CLASSES,
        "arms": [list(a) for a in ARMS],
    }


def run_m1() -> list[dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for feature, scheme in ARMS:
        print(f"\n{'=' * 64}\n{feature} / {scheme}\n{'=' * 64}")
        r = train_arm(feature, scheme)

        row = {k: r[k] for k in FIELDS if k in r}
        row.update({f"f1_{c}": round(float(r["test_f1_per_class"][c]), 4) for c in CLASSES})
        results.append(row)

        np.savetxt(
            OUT / f"confusion_{feature}_{scheme}.csv",
            np.array(r["confusion"]), fmt="%d", delimiter=",",
            header=",".join(CLASSES), comments="",
        )

        data = load_or_build(feature)
        cuts = fit_cuts(data["train"][0], scheme=scheme)
        labels = bit_labels(cuts, prefix="frame" if feature == "envelope" else "s")
        B_te, y_te = transform(data["test"][0], cuts), data["test"][1]
        text = "\n\n".join(
            f"top clauses for {name}:\n"
            + format_clauses(
                top_clauses(r["tm"], c, B_te, y_te, r["n_features"], labels=labels)
            )
            for c, name in enumerate(CLASSES)
        )
        (OUT / f"clauses_{feature}_{scheme}.txt").write_text(text)

        with open(OUT / f"history_{feature}_{scheme}.json", "w") as fh:
            json.dump(r["history"], fh)

    with open(OUT / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)
    (OUT / "manifest.json").write_text(json.dumps(_manifest(), indent=2))

    print(f"\n{'feature':10s} {'scheme':12s} {'feats':>6s} {'val':>7s} {'test':>7s}")
    for r in results:
        print(
            f"{r['feature']:10s} {r['scheme']:12s} {r['n_features']:6d} "
            f"{r['val_best10']:7.4f} {r['test_best10']:7.4f}"
        )
    print(f"\nwrote {OUT}/results.csv")
    return results


if __name__ == "__main__":
    run_m1()
