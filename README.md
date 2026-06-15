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

To find and classify cod sounds in a long (5–45 min) recording, use `src/inference/detect.py`:

```bash
uv run python -m src.inference.detect data/unannotated/01-220213_1505_Ch4_all.wav   # uses models/classifier_logistic.pt
uv run python -m src.inference.detect <wav> --classifier mlp                        # use the MLP model
```

The annotated events are only milliseconds long (a click is ~a few ms; a vocal ~50 ms), so
`src/inference/detect.py` first runs an energy/onset detector over the raw waveform to find the
short candidate events, then cuts each one, zero-pads it to 5s exactly like the training clips,
and classifies it with Perch + the trained model. This matches the conditions the classifier was
trained under. Detector settings (band-pass, threshold, min duration) live in
`src/data/config.py`.

Outputs are written to `results/detection/<filename>/`:
- `events.csv` — one row per detected event (`start_s, end_s, duration_s, class, confidence`)
- `timeline.png` — detected events over time, colored by confidence

Run `main.py` first so a trained model exists.

#### Diagnostic: blind sliding window (`src/inference/infer.py`)

`src/inference/infer.py` tiles the file into fixed 5s windows and classifies every window.
**This does not work for this data** — Perch mean-pools over each 5s window, so a
millisecond-long event is averaged away and the model only ever sees background (it returns
`silence`/`other` for the whole file, even on clips the trained model scores 95% F1 on in
isolation). It is kept only for comparison; use `src/inference/detect.py` for real results.

```bash
uv run python -m src.inference.infer <wav>             # blind 5s tiling — diagnostic only
uv run python -m src.inference.infer <wav> --hop 80000 # 2.5s overlap
```

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

See [`results/`](results/) for confusion matrices, per-class metrics, and t-SNE plots.

| Experiment | Val accuracy | Test accuracy |
|---|---|---|
| Perch v2 + Logistic Regression (n=20/class) | 80% | 81% |
| Perch v2 + MLP (n=20/class) | 84% | 80% |
| Perch v2 + Logistic Regression (n=500/class) | 70% | 86% |
| Perch v2 + MLP (n=500/class) | 70% | 84% |

The val/test gap at n=500 reflects domain shift between splits (likely different recording sessions). Test accuracy is the more reliable indicator. Full per-class metrics and confusion matrices are in [`results/`](results/).
