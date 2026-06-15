# Cod Drumming Classification

Classifying underwater cod sounds using Google AI's [Perch 2.0](https://github.com/google-research/perch) model.

The pipeline extracts 1536-dimensional audio embeddings from annotated WAV clips using Perch, then trains a lightweight **PyTorch** classifier on top (a linear "logistic" model or an MLP).

**Classes:** `click` · `vocal` · `water` · `silence` · `other`

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then:

```bash
uv sync
```

**GPU:** Perch embedding uses GPU TensorFlow (`tensorflow[and-cuda]`) when an NVIDIA GPU is
present — about 25× faster than CPU. It needs a CUDA 12.x-capable driver (the pinned TF build
is CUDA 12, matching common A100 drivers). Without a GPU it falls back to CPU automatically.

## Usage

```bash
uv run main.py                        # linear "logistic" model (default)
uv run main.py --classifier mlp       # MLP classifier
uv run main.py --force-recompute      # re-extract embeddings (skip cache)
```

Both classifiers are PyTorch models trained with early stopping on the validation set;
a training curve is saved to `models/training_curve.png` and the trained model to
`models/classifier_<name>.pt`. Embeddings are cached after the first run — subsequent runs
are fast.

### Detecting events in long recordings

To find and classify cod sounds in a long (5–45 min) **unannotated** recording, use
`src/inference/detect.py`. An energy/onset detector finds the short events (a click is ~a few ms,
a vocal ~50 ms), then each is cut, zero-padded to 5s like the training clips, embedded with Perch,
and classified:

```bash
uv run python -m src.inference.detect <wav>                      # clean-clip head (baseline)
uv run python -m src.inference.detect <wav> --classifier domain  # domain-adapted head (recommended)
```

> Blind 5-second tiling does **not** work here — Perch mean-pools over its 5s window, so a
> millisecond event is averaged away (`src/inference/infer.py` is kept only to demonstrate this).
> The detector-first approach matches the conditions the classifier was trained under.

Outputs → `results/detection/<filename>/`: `events.csv` (one row per event:
`start_s, end_s, duration_s, class, confidence`) and `timeline.png`. Score a run against a Raven
selection table with `src/inference/evaluate_detection.py`. Detector settings live in
`src/data/config.py`.

Run `main.py` first so the baseline model exists; the domain-adapted head is produced by the
analysis pipeline in `src/data/analysis/` (see the report linked under **Results**).

## Data

Place annotated WAV clips under `data/annotated/` with this structure:

```
data/annotated/
  train/  click/  vocal/  water/  silence/  other/
  val/    click/  vocal/  water/  silence/  other/
  test/   click/  vocal/  water/  silence/  other/  NA/
```


## Structure

```
main.py                        — train + evaluate on annotated clips
src/
  data/                        — config, dataset loader, audio preprocessing (load_windows, detect_events)
  data/analysis/               — class-separability analysis + domain-adaptation training/eval
  model/perch.py               — Perch embedding extraction + cache (TensorFlow)
  model/classifier.py          — PyTorch model architectures + save/load (LinearHead, MLPHead)
  model/extract.py             — subprocess: embed the annotated dataset (TF-only)
  model/embed_long.py          — subprocess: embed a long file with a sliding window (TF-only)
  model/detect_long.py         — subprocess: detect events in a long file + embed each (TF-only)
  inference/detect.py          — detect + classify events in a long recording (recommended)
  inference/evaluate_detection.py — score detect.py against Raven selection tables
  inference/infer.py           — blind 5s sliding window over a long file (diagnostic only)
  training/                    — training loop (train.py) and evaluation (evaluate.py)
  visualize/                   — waveforms, spectrograms, t-SNE, metrics plots
notebooks/                     — Jupyter notebooks for exploration
results/                       — saved experiment outputs (metrics, figures, detection timelines)
scripts/                       — helper scripts (data transfer)
data/                          — local data files (not tracked in git)
```

## Results

**Annotated clips** (Perch embedding + linear/MLP head) — test accuracy ~**86%**, all classes
incl. click/vocal cleanly separated:

| Experiment | Val acc | Test acc |
|---|---|---|
| Perch v2 + Logistic Regression (n=500/class) | 70% | 86% |
| Perch v2 + MLP (n=500/class) | 70% | 84% |

**Long unannotated files** (detect → classify → score vs Raven tables, held-out recordings):

- **Detection** finds 86–99% of annotated events (but over-detects 5–35× — it casts a wide net).
- The **domain-adapted** head (retrained on detector-cropped events, not the clean clips) reaches
  **66% correct on a held-out 10-file set** (vocals **88%**), up from 48% with the clean-clip head.
- An optional **`background` class** rejects ~94% of detections, cutting over-detection 52.8×→2.9×.

Full methodology, per-file numbers, and figures: **[`results/detection/REPORT.md`](results/detection/REPORT.md)**.
Confusion matrices, t-SNE, and class-separability plots are under [`results/`](results/).
