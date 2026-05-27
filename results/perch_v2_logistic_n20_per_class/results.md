# Experiment: Perch v2 + Logistic Regression — 20 samples per class

**Date:** 2026-05-27

## What we did

Used Google AI's Perch 2.0 model to extract audio embeddings from short annotated underwater recordings of cod, then trained a logistic regression classifier on top of those embeddings.

**Pipeline:**
1. Load annotated WAV clips (96 kHz) from train/val/test folders
2. Resample to 32 kHz and center-pad/trim to 5 seconds (Perch's required input format)
3. Extract a 1280-dimensional embedding per clip using Perch v2
4. Train a logistic regression classifier on the train embeddings
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

### Validation (80% accuracy)

| Class   | Precision | Recall | F1   |
|---------|-----------|--------|------|
| click   | 1.00      | 1.00   | 1.00 |
| other   | 0.60      | 0.45   | 0.51 |
| silence | 1.00      | 0.80   | 0.89 |
| vocal   | 0.61      | 1.00   | 0.75 |
| water   | 0.94      | 0.75   | 0.83 |

### Test (81% accuracy)

| Class   | Precision | Recall | F1   |
|---------|-----------|--------|------|
| click   | 0.95      | 0.95   | 0.95 |
| other   | 0.69      | 0.45   | 0.55 |
| silence | 0.94      | 0.85   | 0.89 |
| vocal   | 0.71      | 0.85   | 0.77 |
| water   | 0.76      | 0.95   | 0.84 |

## Observations

- **click** is nearly perfectly classified — the short transient is very distinctive to Perch
- **other** is the weakest class (F1 0.55) — it is a catch-all category with highly variable content
- **silence** and **water** are well separated from biological sounds
- **vocal** confusion with **other** is expected given some overlap in frequency content

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

- Increase training data (full dataset has ~11 000 files)
- Try MLP classifier: `uv run main.py --classifier mlp`
- Investigate `other` class confusions — may need sub-labelling
