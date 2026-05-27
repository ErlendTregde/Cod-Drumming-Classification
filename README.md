# Cod Drumming Classification

Classifying underwater cod sounds using Google AI's [Perch 2.0](https://github.com/google-research/perch) model. A collaboration between CAIR and UiA bioacoustic biologists.

The pipeline extracts 1280-dimensional audio embeddings from annotated WAV clips using Perch, then trains a lightweight classifier on top.

**Classes:** `click` · `vocal` · `water` · `silence` · `other`

## Setup

Install [uv](https://docs.astral.sh/uv/) if you don't have it, then:

```bash
uv sync
```

## Usage

```bash
uv run main.py                        # logistic regression (default)
uv run main.py --classifier mlp       # MLP classifier
uv run main.py --force-recompute      # re-extract embeddings (skip cache)
```

Embeddings are cached after the first run — subsequent runs are fast.

## Data

Place annotated WAV clips under `data/annotated/` with this structure:

```
data/annotated/
  train/  click/  vocal/  water/  silence/  other/
  val/    click/  vocal/  water/  silence/  other/
  test/   click/  vocal/  water/  silence/  other/  NA/
```

The full dataset (~11 000 files, 48.9 GB) lives on `\\fil01\share\Naturlyder\CodPond2\RAW_Audio\annotated_files\annotated`. Use `scripts/download_sample.ps1` (Windows) to copy a small sample locally.

## Structure

```
main.py                  — entry point, CLI arguments
src/
  data/                  — config and dataset loader
  model/perch.py         — Perch embedding extraction + cache
  training/              — classifier training and evaluation
  visualize/             — waveforms, spectrograms, t-SNE, metrics plots
notebooks/               — Jupyter notebooks for exploration
results/                 — saved experiment outputs (metrics, figures)
scripts/                 — helper scripts (data download)
data/                    — local data files (not tracked in git)
```

## Results

See [`results/`](results/) for confusion matrices, per-class metrics, and t-SNE plots.

| Experiment | Test accuracy |
|---|---|
| Perch v2 + Logistic Regression (n=20) | 81% |
| Perch v2 + MLP (n=20) | 80% |
