# Experiment: Perch v2 + Logistic Regression — 500 samples per class

**Date:** 2026-05-28

## What we did

Same pipeline as previous experiments, scaled up to 500 samples per class. Used Google AI's Perch 2.0 model to extract 1280-dim audio embeddings, then trained a logistic regression classifier on top.

**Pipeline:**
1. Load annotated WAV clips (96 kHz) from train/val/test folders
2. Resample to 32 kHz and center-pad/trim to 5 seconds (Perch's required input format)
3. Extract a 1280-dimensional embedding per clip using Perch v2
4. Train a logistic regression classifier on the train embeddings
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

### Validation (69.7% accuracy)

| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| click   | 0.72      | 1.00   | 0.84 | 500     |
| other   | 0.80      | 0.75   | 0.77 | 500     |
| silence | 0.89      | 0.84   | 0.86 | 500     |
| vocal   | 0.56      | 0.65   | 0.60 | 500     |
| water   | 0.42      | 0.24   | 0.31 | 490     |

### Test (85.8% accuracy)

| Class   | Precision | Recall | F1   | Support |
|---------|-----------|--------|------|---------|
| click   | 0.92      | 0.99   | 0.95 | 500     |
| other   | 0.88      | 0.61   | 0.72 | 500     |
| silence | 0.89      | 0.92   | 0.90 | 500     |
| vocal   | 0.82      | 0.92   | 0.86 | 500     |
| water   | 0.69      | 0.83   | 0.75 | 166     |

## Comparison with n=20

| | n=20 val | n=500 val | n=20 test | n=500 test |
|---|---|---|---|---|
| Accuracy | 80% | **70%** | 81% | **86%** |

- **Test improved** (+5) with 25× more training data — better generalization across recording sessions.
- **Val dropped** (−10), again driven mostly by *water* (F1 0.83 → 0.31) and *vocal* (F1 0.75 → 0.60).
- Same val/test gap as the MLP run — confirms the issue is in the data splits, not the model.

## Comparison with MLP (n=500)

| | Logistic test | MLP test | Winner |
|---|---|---|---|
| Overall accuracy | **85.8%** | 84.2% | Logistic |
| click F1 | 0.95 | 0.95 | Tie |
| other F1 | 0.72 | 0.73 | MLP |
| silence F1 | **0.90** | 0.86 | Logistic |
| vocal F1 | **0.86** | 0.84 | Logistic |
| water F1 | **0.75** | 0.74 | Logistic |

Logistic regression still edges out the MLP, even at 25× more data. The MLP is likely under-tuned (default sklearn settings, no regularisation sweep, no early stopping on val).

## Observations

- **click** classification is near-perfect (F1 0.95) — the transient is highly distinctive in Perch's embedding space.
- **water** is the weakest class on val (F1 0.31) but reasonable on test (F1 0.75). Strong indication of a domain shift between val and test for water.
- **other** has high precision (0.88) but lower recall (0.61) on test — the model is conservative about predicting *other* and misses many true instances.
- The 16-point gap between val and test accuracy is consistent across both classifiers — a property of the data splits, not the model choice.

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
| click   | 497 / 500 (99%) |
| silence | 459 / 500 (92%) |
| vocal   | 459 / 500 (92%) |
| water   | 137 / 166 (83%) |
| **other** | **307 / 500 (61%)** |

**Key confusion patterns:**
- `other → vocal`: 92 errors (biggest single confusion)
- `other → water`: 44 errors
- `other → silence`: 35 errors
- `other → click`: 22 errors
- `water → other`: 25 errors (asymmetric — `other → water` is bigger)
- `silence → click`: 19 errors

**Interpretation — `other` is a semantic dumping ground.** By definition it contains "everything that isn't click/vocal/water/silence", so it has no consistent acoustic signature for Perch's embeddings to capture. The model spreads its `other` predictions across all four other classes.

This is the main bottleneck holding overall accuracy below ~90%. Two ways to address it:

1. **Sub-label `other`** — split it into meaningful sub-categories (motor noise, unknown fish, etc.) so each has a learnable signature
2. **Treat `other` as a reject class** — exclude it from training and treat low-confidence predictions on the other 4 classes as `other` automatically. Simpler, and would likely push 4-class accuracy above 90% immediately.

## Next steps

- Investigate the val/test gap — check whether files in val and test come from different recording sessions or channels (`Ch4` vs `Ch6`)
- Apply the model to a full `_all.wav` recording using a sliding 5-second window to evaluate end-to-end performance
- Compare predictions on long files against the biologist's selection tables as ground truth
- Tune the MLP (regularisation, early stopping) to see if it can overtake logistic regression
