# Experiment: Perch v2 + MLP — 500 samples per class

**Date:** 2026-05-28

## What we did

Same pipeline as previous experiments, scaled up to 500 samples per class. Used Google AI's Perch 2.0 model to extract 1280-dim audio embeddings, then trained a two-layer MLP `(256, 64)` classifier on top.

**Pipeline:**
1. Load annotated WAV clips (96 kHz) from train/val/test folders
2. Resample to 32 kHz and center-pad/trim to 5 seconds (Perch's required input format)
3. Extract a 1280-dimensional embedding per clip using Perch v2
4. Train an MLP classifier `(hidden_layer_sizes=(256, 64))` on the train embeddings
5. Evaluate on val and test sets

## Dataset

| Split | Samples | Notes |
|-------|---------|-------|
| train | 2 500   | 500 per class × 5 classes |
| val   | 2 490   | 500 per class (water: 490) |
| test  | 2 166   | 500 per class (water: 166) |

**Classes:** click · other · silence · vocal · water
**Excluded:** NA (1 file in test only, not used)

## Results

### Validation (70.0% accuracy)

| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| click   | 0.75      | 1.00   | 0.85 | 500     |
| other   | 0.79      | 0.75   | 0.77 | 500     |
| silence | 0.87      | 0.85   | 0.86 | 500     |
| vocal   | 0.56      | 0.63   | 0.59 | 500     |
| water   | 0.43      | 0.26   | 0.32 | 490     |

### Test (84.2% accuracy)

| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| click   | 0.92      | 0.99   | 0.95 | 500     |
| other   | 0.87      | 0.63   | 0.73 | 500     |
| silence | 0.80      | 0.93   | 0.86 | 500     |
| vocal   | 0.84      | 0.83   | 0.84 | 500     |
| water   | 0.69      | 0.80   | 0.74 | 166     |

## Comparison with n=20

| | n=20 val | n=500 val | n=20 test | n=500 test |
|---|---|---|---|---|
| Accuracy | 84% | **70%** | 80% | **84%** |

- **Test improved** (+4) as expected with 25× more training data.
- **Val dropped sharply** (−14) — driven mostly by *water* (F1 0.85 → 0.32) and *vocal* (F1 0.80 → 0.59).
- The val and test sets now show very different difficulty per class, suggesting the splits draw from different recording sessions / channels and aren't fully interchangeable.

## Observations

- **click** classification is near-perfect (F1 0.95 on test) — the transient is highly distinctive in Perch's embedding space.
- **water** is the weakest class on val (F1 0.32) but reasonable on test (F1 0.74). Strong indication of a domain shift between val and test for water.
- **other** has high precision but low recall on test — the model is conservative about predicting *other* and misses many true instances.
- The gap between val and test (14 percentage points) is a finding in itself: the split design needs review.

## Figures

| Figure | Description |
|--------|-------------|
| [waveforms.png](waveforms.png) | Raw amplitude per class |
| [spectrograms.png](spectrograms.png) | Frequency content over time per class |
| [tsne.png](tsne.png) | Perch embeddings projected to 2D |
| [class_metrics.png](class_metrics.png) | Per-class F1, val vs test |
| [confusion_val.png](confusion_val.png) | Confusion matrix — validation |
| [confusion_test.png](confusion_test.png) | Confusion matrix — test |

## Confusion matrix analysis (test set)

| True class | Correctly classified |
|---|---|
| click   | 495 / 500 (99%) |
| silence | 464 / 500 (93%) |
| vocal   | 416 / 500 (83%) |
| water   | 132 / 166 (80%) |
| **other** | **317 / 500 (63%)** |

**Key confusion patterns:**
- `other → vocal`: 71 errors
- `vocal → silence`: 61 errors (much worse than logistic, which had only 22)
- `other → silence`: 52 errors
- `other → water`: 43 errors
- `water → other`: 27 errors (asymmetric)
- `silence → click`: 19 errors

**Interpretation — `other` is a semantic dumping ground.** By definition it contains "everything that isn't click/vocal/water/silence", so it has no consistent acoustic signature for Perch's embeddings to capture. The model spreads its `other` predictions across all four other classes.

**MLP-specific weakness:** the MLP confuses `vocal` as `silence` 61 times vs only 22 for logistic regression — this is why the MLP underperforms overall despite having more capacity. The MLP appears to over-fit to low-energy patterns shared between vocal and silence.

Two ways to address the `other` bottleneck:

1. **Sub-label `other`** — split it into meaningful sub-categories (motor noise, unknown fish, etc.) so each has a learnable signature
2. **Treat `other` as a reject class** — exclude it from training and treat low-confidence predictions on the other 4 classes as `other` automatically. Simpler, and would likely push 4-class accuracy above 90% immediately.

## Next steps

- Investigate the val/test gap — check whether files in val and test come from different recording sessions or channels (`Ch4` vs `Ch6`)
- Apply the model to a full `_all.wav` recording using a sliding 5-second window to evaluate end-to-end performance
- Compare predictions on long files against the biologist's selection tables as ground truth
