# Cod Drumming Classification

Classifying underwater cod sounds using Google AI's [Perch 2.0](https://github.com/google-research/perch) model.

The pipeline extracts 1536-dimensional audio embeddings from annotated WAV clips using Perch, then trains a lightweight **PyTorch** classifier on top (a linear "logistic" model or an MLP).

**Classes:** `click` · `vocal` · `water` · `silence` · `other`

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then:

```bash
uv sync
```

## Usage

```bash
uv run main.py                        # linear "logistic" model (default)
uv run main.py --classifier mlp       # MLP classifier
uv run main.py --force-recompute      # re-extract embeddings (skip cache)
```

Both classifiers are PyTorch models trained with early stopping on the validation set;
a training curve is saved to `models/training_curve.png`. Embeddings are cached after the
first run — subsequent runs are fast.

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
main.py                  — entry point, CLI arguments
src/
  data/                  — config and dataset loader
  model/perch.py         — Perch embedding extraction + cache (TensorFlow)
  model/classifier.py    — PyTorch model architectures (LinearHead, MLPHead)
  model/extract.py       — subprocess entry point for TF-only embedding extraction
  training/              — training loop (train.py) and evaluation (evaluate.py)
  visualize/             — waveforms, spectrograms, t-SNE, metrics plots
notebooks/               — Jupyter notebooks for exploration
results/                 — saved experiment outputs (metrics, figures)
scripts/                 — helper scripts (data download)
data/                    — local data files (not tracked in git)
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
