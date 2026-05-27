# Experiment: Perch v2 + MLP — 20 samples per class

**Date:** 2026-05-27

## What we did

Same pipeline as `perch_v2_logistic_n20`, but replaced the logistic regression classifier with a two-layer MLP (256 → 64 hidden units).

**Pipeline:**
1. Load annotated WAV clips (96 kHz) from train/val/test folders
2. Resample to 32 kHz and center-pad/trim to 5 seconds (Perch's required input format)
3. Extract a 1280-dimensional embedding per clip using Perch v2
4. Train an MLP classifier `(hidden_layer_sizes=(256, 64))` on the train embeddings
5. Evaluate on val and test sets

## Dataset

| Split | Samples per class | Total |
|-------|-------------------|-------|
| train | 20 | 100 |
| val   | 20 | 100 |
| test  | 20 | 100 |

**Classes:** click · other · silence · vocal · water  
**Excluded:** NA (1 file in test only, not used)

## Results

### Validation (84% accuracy)

| Class   | Precision | Recall | F1   |
|---------|-----------|--------|------|
| click   | 1.00      | 1.00   | 1.00 |
| other   | 0.75      | 0.45   | 0.56 |
| silence | 1.00      | 0.90   | 0.95 |
| vocal   | 0.67      | 1.00   | 0.80 |
| water   | 0.85      | 0.85   | 0.85 |

### Test (80% accuracy)

| Class   | Precision | Recall | F1   |
|---------|-----------|--------|------|
| click   | 1.00      | 1.00   | 1.00 |
| other   | 0.62      | 0.50   | 0.56 |
| silence | 1.00      | 0.75   | 0.86 |
| vocal   | 0.68      | 0.85   | 0.76 |
| water   | 0.75      | 0.90   | 0.82 |

## Comparison with logistic regression

| | Logistic (test) | MLP (test) | Winner |
|---|---|---|---|
| Overall accuracy | **81%** | 80% | Logistic |
| click F1 | 0.95 | **1.00** | MLP |
| other F1 | 0.55 | 0.56 | Tie |
| silence F1 | **0.89** | 0.86 | Logistic |
| vocal F1 | 0.77 | 0.76 | Logistic |
| water F1 | **0.84** | 0.82 | Logistic |

MLP scores higher on val (84% vs 80%) but lower on test (80% vs 81%) — a sign of mild overfitting with only 100 training samples. Logistic regression generalises better at this data size. MLP is expected to outperform once more training data is available.

## Figures

| Figure | Description |
|--------|-------------|
| [waveforms.png](waveforms.png) | Raw amplitude per class |
| [spectrograms.png](spectrograms.png) | Frequency content over time per class |
| [tsne.png](tsne.png) | Perch embeddings projected to 2D |
| [class_metrics.png](class_metrics.png) | Per-class F1, val vs test |
| [confusion_val.png](confusion_val.png) | Confusion matrix — validation |
| [confusion_test.png](confusion_test.png) | Confusion matrix — test |

## Next steps

- Increase training data — MLP should benefit more than logistic regression from more samples
- Re-run both classifiers at n=100+ to see if MLP overtakes logistic
