# Evaluating Perch 2.0 for Cod Drumming Detection & Classification

**Project:** Cod Drumming Classification
**Date:** 2026-06-10
**Scope of this report:** an end-to-end, results-based evaluation of Google AI's **Perch 2.0**
embedding model for (1) classifying annotated cod-sound clips and (2) detecting + classifying cod
sounds in long, unannotated recordings("apply the
model to a longer 5–45 min file and have it extract and classify the sounds, without
pre-extracting").

> **Goal framing.** The assignment is to *evaluate how Perch
> performs on cod-drumming data and document it* — not to ship a production detector. An honest,
> well-evidenced characterization of where Perch works and where it breaks is the deliverable. All
> numbers below come from measured results in `results/`

**Classes:** `click` · `vocal` · `water` · `silence` · `other`

---

## 1. Executive summary

1. **Perch is strong for *pre-segmented* cod sounds.** A linear head on 1536-d Perch embeddings
   reaches **86% test accuracy** (n=500/class); all classes, including the hard click/vocal pair,
   separate cleanly at their native width.
2. **Blind 5-second tiling of long files completely fails.** Perch mean-pools over its 5 s window,
   so a 2 ms click is averaged into ~5 s of background and vanishes — long files return only
   `silence`/`other`, **zero** cod sounds, even on the channel the model was trained on most.
3. **A detector-first pipeline fixes *detection*.** An energy detector finds **86–99%** of the
   annotated events. But it **over-detects by 5–35×**, and classification of the detected events is
   only **29–66%** correct.
4. **We proved *why* classification is hard — and corrected an earlier wrong assumption.** Click and
   vocal are *not* inherently similar to Perch (they are the **best-separated** class pair, and
   differ at **AUC 0.98** by spectral flatness). The confusion was an artifact of a multi-scale
   10 ms crop that collapses *every* class into `click`. Removing it (single-pass) fixed the worst
   case.
5. **The limit was a train/serve mismatch — and fixing it works.** Appending width-independent
   signal features gave only +2–4 pp; but **domain-adaptation retraining on detector-cropped events
   (labelled from the selection tables) lifted held-out correct-classification by +13 to +40 pp**
   on all three files (§4.7). The classifier had been trained on clean isolated clips but applied to
   detector crops embedded in real background; training it on the detector's own output closes most
   of the gap.

---

## 2. Data

- **Annotated clips:** WAV clips, 96 kHz mono, split train/val/test. The
  stored clips are (a click is 1–6 ms; a vocal grunt 90–680 ms). Test set:
  500 each of click/other/silence/vocal, 166 water.
- **Long files (`_all`):** the full recordings the clips were cut from, with Raven **selection
  tables** (`selection_<stem>.txt`) giving exact `(begin, end, label)` ground truth. Three were
  used here:
  - `01-220412_1221_Ch6_all` — 21.6 min, quiet vocals (83 annotated events)
  - `01-220301_1434_Ch6_all` — vocal-heavy (384 annotated events)
  - `01-220224_1200_Ch4_all` — click-heavy (385 annotated events)
- **Compute:** Perch embedding runs on an **NVIDIA A100** (MIG 1g.20gb slice) via GPU TensorFlow
  (~25× CPU). The PyTorch head and signal-feature extraction run on CPU.

---

## 3. Method

Three stages, isolated to avoid a TensorFlow/PyTorch segfault (Perch embedding always runs in a
subprocess; training/inference processes load only PyTorch):

1. **Annotated-clip classification** (`main.py`): Perch embedding → linear or MLP head.
2. **Long-file detection** (`src/inference/detect.py`): energy/onset detector finds short events →
   embed each **once at its native detected extent** → classify with the trained head → score
   against the selection table (`src/inference/evaluate_detection.py`).
3. **Diagnostic class analysis** (`src/data/analysis/`): characterizes each class from the raw
   waveform (pulse structure, spectral flatness) and from Perch embeddings (separability,
   scale-migration), to explain the classification behavior.

### Honest scoring (greedy 1-to-1 matching)

An earlier metric counted a true event as "found" if *any* detection fell within ±1 s. Because the
detector fires 5–35× more often than there are annotations, that scored ~100% recall purely from
**density** (random timestamps would score the same). All numbers in this report use **greedy
1-to-1 matching** (each true event claims at most one detection; each detection at most one true
event) at ±0.5 s tolerance — so recall is honest and every unclaimed detection is a counted false
positive.

---

## 4. Results

### 4.1 Annotated-clip classification (Perch works on segmented sounds)

| Model (n=500/class) | Val accuracy | Test accuracy |
|---|---|---|
| Perch v2 + Logistic Regression | 70% | **86%** |
| Perch v2 + MLP | 70% | 84% |

Within the augmentation experiment (same train loop, embedding-only arm): test accuracy 0.838
(logistic) / 0.845 (MLP); **click F1 0.94–0.95, vocal F1 0.86**, with only **2–3 of 500** vocals
misclassified as click. The val/test gap reflects domain shift between recording sessions; test is
the reliable indicator. *Click and vocal are already well-separated here* — a key fact for §4.4.

Detail: `results/perch_v2_logistic_n500_per_class/`, `results/perch_v2_MLP_n500_per_class/`.

### 4.2 Long files via blind 5 s tiling — fails (diagnostic only)

Tiling `01-220213_1505` into fixed 5 s windows (`src/inference/infer.py`) returns only
`silence`/`other`, **0** click/vocal/water — even on Ch4, the most-trained channel (502 silence, 17
other). **Root cause: temporal-scale mismatch.** Perch mean-pools over 5 s; a millisecond event is
averaged into background. This is why a detector-first approach is required.

Detail: `results/inference/`.

### 4.3 Detector-first: detection works, but over-detects

Single-pass detector-first pipeline, scored vs the selection tables (greedy 1-to-1):

| file (dominant class) | detect-recall | detections / true event | false positives |
|---|---|---|---|
| `01-220412` (quiet vocal) | **99%** (82/83) | 34.8× | 97% |
| `01-220301` (vocal-heavy) | **86%** (330/384) | 5.4× | 84% |
| `01-220224` (click-heavy) | **96%** (371/385) | 25.5× | 96% |

**Detection recall is high (86–99%)**, but the detector emits **5–35× more events than are
annotated**; 84–97% of detections land on no annotation. The selection tables are only *partially*
annotated, so some "false positives" are real-but-unmarked sounds — **recall is trustworthy,
precision is a pessimistic lower bound.** Even so, the detector genuinely over-fires
(`DETECT_THRESHOLD_K=3` was lowered to catch quiet vocals). **Precision, not recall, is the metric
that matters here.**

Per-file outputs: `results/detection/<stem>/events.csv`, `ground_truth.csv`, `timeline.png`.

### 4.4 Class-separability analysis — *why* classification is hard (key scientific result)

`src/data/analysis/` tested five hypotheses on the test set + all three long files. It **corrects a
load-bearing assumption** (that click ≈ a single vocal pulse to Perch).

Acoustic medians by class (from raw waveforms):

| class | sound dur (ms) | n_pulses | pulse_strength | flatness | centroid (Hz) |
|---|---|---|---|---|---|
| click | 2.1 | **1** | **0.00** | **0.63** (broadband) | 966 |
| vocal | 61.5 | **13** | **0.48** (pulse train) | **0.07** (tonal) | 559 |
| water | 44.1 | 9.5 | 0.32 | 0.17 | 823 |
| other | 216 | 33.5 | 0.45 | 0.10 | 734 |
| silence | 3993 | 876 | 0.42 | 0.15 | 876 |

**Verdicts:**

- **H1 — vocal = pulse train, click = single transient: TRUE.** (vocal 13 pulses vs click 1;
  strength 0.48 vs 0.00). See `example_vocal.png`, `example_click.png`.
- **H2 — click & vocal differ *spectrally*, not just temporally: TRUE, strongly.** Spectral
  flatness separates them at **AUC 0.98** (click broadband 0.63 vs vocal tonal 0.07).
- **H3 — click↔vocal is the most-confused pair in Perch space: FALSE.** It is the **best-separated**
  pair (centroid cosine distance 0.285); the genuinely close pairs are other↔water (0.125),
  vocal↔water (0.137). Silhouette 0.249. See `centroid_distance.png`, `tsne.png`.
- **H4 — a tight crop migrates vocals → click: TRUE — this is the whole problem.** At a **10 ms**
  crop, **100% of vocals (and water, and other) embed nearest the `click` centroid**; at 50 ms only
  3% of vocals do. A 10 ms window is too short to hold a pulse train or stable spectrum, so Perch's
  mean-pool dumps every class into click-land. See `scale_migration.png`, `scale_migration_fracs.png`.
- **H5 — real `_all` events reproduce this: TRUE** on all three files (real vocals → click
  100% @10 ms vs ~0–18% @50 ms; real vocal median pulses 120–142).

**The width tension is real and opposite** (measured on the click file): clicks read correctly only
at **≤25 ms** (at 50 ms → vocal, at 200 ms → other); vocals only at **≥50 ms** (≤10 ms → click). The
two classes need opposite widths and swap identities at the crossover — so **no fixed scale and no
max/mean/vote aggregation rule can resolve it.**

Detail: `results/analysis/analysis_report.md` + figures.

### 4.5 Multi-scale vs single-pass

An earlier multi-scale variant embedded each event at 10/50/200 ms and kept the most-confident
scale. §4.4 shows this is exactly what manufactures the confusion (the 10 ms scale + `max` rule
collapses everything to `click`). We reverted to **single-pass native-extent** embedding.
Correct-classification (% of true events labelled right):

| file (dominant class) | multi-scale (old, gamed) | **single-pass (supported)** |
|---|---|---|
| `01-220412` (quiet vocal) | 17% | **29%** |
| `01-220301` (vocal-heavy) | 81% | **64%** |
| `01-220224` (click-heavy) | 71% | **59%** |

Multi-scale's `max` rule *gamed* single-class-dominated files (it rides the dominant class) but
**collapsed on the mixed file** (vocals → click). Single-pass is more uniform and honest, and fixes
the worst case (`220412` vocal→click: 34 → 4). **Neither is good** — the Perch single-embedding
ceiling.

### 4.6 Signal-feature augmentation — marginal (the "good results" attempt)

Since click/vocal separate at AUC 0.98 by *width-independent* signal features, we tested appending
them (flatness, pulse count/strength, centroid, bandwidth, duration) to the embedding.

**On annotated clips (`augment_experiment.py`) — no lift:**

| arm (logistic) | test acc | click F1 | vocal F1 |
|---|---|---|---|
| embedding only | 0.838 | 0.941 | 0.857 |
| features only (8 features) | 0.711 | 0.924 | 0.771 |
| embedding + features | 0.834 | 0.947 | 0.853 |

(MLP: 0.845 → 0.817, slightly worse.) The clips are native-width, where the embedding already
separates the classes — there is no problem here for features to fix.

**On the real long-file detector events (`longfile_feature_test.py`) — marginal:**

| file | embedding-only | + signal features |
|---|---|---|
| `01-220412` | 24/83 (29%) | 27/83 (33%) |
| `01-220301` | 245/384 (64%) | 252/384 (66%) |
| `01-220224` | 227/385 (59%) | 229/385 (59%) |

**+2–4 pp.** The main residual error (vocal→click 55 on `220301`) is unchanged. The remaining errors
are **diffuse** (vocal→water/other, click→other/water), i.e. a **train/serve mismatch**, not a
width or feature problem.

### 4.7 Domain-adaptation retraining — the fix that works (key positive result)

If the bottleneck is the clean-clip↔detector-crop mismatch, the fix is to **train on detector
crops**. We transferred **91 additional `_all` recordings + their selection tables** and built a
training set from them (`build_domain_dataset.py`): run the detector on each, **greedy-match each
detection to a table annotation** (so each crop gets an exact label *and* the detector's boundary),
embed the matched crops, and pool — **3,542 labelled detector-crops** across 90 recordings (vocal
1589, click 724, silence 708, other 452, water 69). The three evaluation files were **excluded**
from training. We then trained the same logistic head on these crops and re-scored the three
held-out files (`train_domain.py`):

| held-out file | clean-clip baseline | **domain-adapted** | gain |
|---|---|---|---|
| `01-220412` (quiet vocal) | 24/83 (29%) | **57/83 (69%)** | **+40 pp** |
| `01-220301` (vocal-heavy) | 245/384 (64%) | **298/384 (78%)** | **+14 pp** |
| `01-220224` (click-heavy) | 227/385 (59%) | **278/385 (72%)** | **+13 pp** |

Per-class, the gains land exactly on the classes that were failing: `220412` vocal **22→43** of 46;
`220301` vocal **243→295** of 357; `220224` click **211→257** of 349 (and silence 0→5). The
improvement holds **despite** the baseline having seen the test recordings' clips during its own
training (a conservative comparison).

**Leakage control:** training excluded all three test recordings. `220412` and `220224` have **no
same-datetime sibling** in training (fully clean). `220301`'s same-datetime *Ch4* sibling (a
different hydrophone of the same events) is in training, so its +14 pp is mildly optimistic — but
the two fully-clean files (+40, +13) establish the effect independently.

**Confirmed on a fresh validation set** (`eval_val.py`, 10 recordings in `data/unannotated/val/`,
**914 events, none in training**, no recording overlap or same-datetime sibling leak):

| metric (val, 914 events) | clean-clip | **domain-adapted** |
|---|---|---|
| overall correct | 442/914 (**48%**) | 605/914 (**66%**) |
| vocal | 63/146 (43%) | **129/146 (88%)** |
| click | 356/663 (54%) | **445/663 (67%)** |
| silence | 4/51 (8%) | 17/51 (33%) |
| other | 17/51 (33%) | 14/51 (27%) |

So the gain generalizes to unseen recordings: **+18 pp overall, +45 pp on vocals, +13 pp on clicks**
(`other` dips slightly; water is noise at n=3). Per-file gains are consistent (`220228_1025`
41→90%, `220303_1159_Ch4` 45→75%, `220304_1252` 52→77%). One outlier — `220213_1505_Ch4` scores 5%
for *both* models (the dense-click-train file neither detector can segment), not a domain-adaptation
failure.

**Caveats:** (1) **Over-detection is unchanged** (5–35×) — we trained only on matched positives with
no `background` class, so this improved *classification of detected events*, not *precision*; that
remains a separate detector-threshold problem. (2) **Water stays weak** (69 training examples).
(3) Numbers are correct-classification of detected events, not a fully clean labelled list.

### 4.8 A `background` class to cut over-detection (precision/recall knob)

Domain adaptation fixed *classification* but not *over-detection* (5–35× too many events). To attack
that, we added a 6th class — **`background`**, sampled from detector events that overlap **no** table
annotation (the false positives, 50/file ≈ 4,488 crops) — so the model can *reject* a detection
instead of forcing it into a cod-sound class (`build_domain_dataset.py` + `train_domain_bg.py`).
Evaluated on the same 10 val files (914 events):

| model | correct-classification | detections kept | over-detection |
|---|---|---|---|
| domain (5-class) | 605/914 (66%) | 48,293 (100%) | 52.8× |
| **domain + background (6-class)** | 519/914 (57%) | **2,671 (6%)** | **2.9×** |

The `background` class **rejects 94% of detections** — over-detection drops **52.8× → 2.9×** (≈18×
less output), lifting precision roughly **~1.3% → ~19%** — at a cost of **−9 pp** correct-classification
(some true events are wrongly rejected). This is the first mechanism that meaningfully improves
*precision*, turning the output from "mostly noise" into "~3× the annotations, mostly real." It works
**despite** the background crops being sampled from only-partially-annotated recordings (the noise
distribution dominates). It is a **tunable knob** (`BG_PER_FILE`): fewer background examples → less
aggressive rejection, trading precision back for recall.

> **Process note:** the first run looked like a catastrophic failure (17% correct, 0% rejected). That
> was a label-decoding bug — `LabelEncoder` sorts classes, so the 6 outputs had to be decoded through
> the fitted encoder, not the raw class list. Fixed; the numbers above are correct.

**Why reject downstream instead of tightening the detector?** A `DETECT_THRESHOLD_K` sweep on val
(detector only, no classifier) shows detector recall and over-detection are *tightly coupled*:

| `k` | detector-recall | over-detection |
|---|---|---|
| 3 (current) | 88% | 63.8× |
| 4 | 70% | 20.1× |
| 5 | 56% | 5.3× |
| 6 | 47% | 2.9× |

To reach the background class's ~2.9× over-detection *via the detector* (k=6), recall collapses to
**47%** — so a tight detector caps correct-classification at **≤47%**, versus **57%** for the loose
detector (k=3) + background reject at the *same* output volume. The lesson: **a missed detection is
unrecoverable, a false detection is rejectable** — keep the detector loose (high recall) and reject
downstream. Higher `k` only saves embedding *compute*, at a recall cost; it does not improve
precision-for-recall.

---

## 5. What we learned

1. **Perch is excellent for pre-segmented cod sounds** (86% test; click/vocal cleanly separated).
2. **Blind 5 s tiling cannot work** — temporal-scale mismatch (ms events vs 5 s mean-pool).
3. **Detection (energy detector) works** (86–99% recall) but over-detects 5–35×.
4. **Classification of detected events is the bottleneck** (29–66% correct).
5. **Click ≠ vocal** — they are the *best-separated* pair (AUC 0.98 spectrally). The earlier
   "click ≈ vocal to Perch" claim was wrong; the confusion was an artifact of the 10 ms multi-scale
   crop + `max` rule, which collapses every class into `click`.
6. **The width tension is real and opposite** (clicks ≤25 ms, vocals ≥50 ms) — no single scale or
   aggregation rule can serve both.
7. **The limit was a domain gap, and domain adaptation fixes most of it.** Training on
   detector-cropped events (labelled from tables) instead of clean clips lifted held-out
   correct-classification by **+13 to +40 pp** (29→69%, 64→78%, 59→72%). Feature augmentation helped
   only marginally; *changing the training distribution* helped a lot.
8. **Over-detection is addressable with a `background` class.** Training a reject class from the
   unmatched (false-positive) detections cuts output ~18× (over-detection 52.8×→2.9×, precision
   ~1.3%→~19%) for a −9 pp recall cost — a tunable precision/recall knob (§4.8). It works even though
   the background crops come from only-partially-annotated recordings.
9. **Honest evaluation matters** — a density-inflated recall metric and a "gamed" multi-scale peak
   both masked the real behavior; greedy 1-to-1 matching and single-pass exposed it; a held-out
   train/test split kept the domain-adaptation result trustworthy.

---

## 6. Limitations

- **Precision is a pessimistic lower bound** — the selection tables only partially annotate the
  dense recordings, so detections in unmarked gaps count as false positives even when plausibly
  real. A single *fully-annotated* file would let us measure precision honestly.
- **Three long files only**, each single-class-dominated. A class-balanced or fully-annotated file
  is needed to measure mixed-class precision/recall properly.
- The PyTorch head runs on CPU (torch is built for CUDA 13 vs the driver's CUDA 12.8); harmless, as
  the head is tiny, but the GPU is used only for Perch (TensorFlow).

---

## 7. Recommendations / next steps

1. **Domain-adaptation retraining — DONE, and it works (§4.7).** Now the supported direction:
   re-train the production head on detector-crops pooled from more recordings. Next refinements:
   add more **water** examples (only 69), and add a **`background`/`none` class** from unmatched
   detections (sampled from clearly-silent regions) to *also* attack over-detection.
2. **Tune the `background` class (now a working lever, §4.8).** It already cuts over-detection ~18×
   for −9 pp recall. Sweep `BG_PER_FILE` for a gentler operating point; consider cleaner background
   sampling (e.g. confidently-silent regions) to reduce pollution from unmarked-real events.
3. **Obtain one fully-annotated file** to measure precision honestly (currently a lower bound,
   since the tables only partially annotate the dense recordings).
4. Scale the domain set further (more recordings, balanced classes) and consider fine-tuning rather
   than retraining from scratch.

---

## 8. Reproducibility

```bash
uv run main.py                                                  # train + evaluate the clip classifier
uv run python -m src.inference.detect data/unannotated/<file>_all.wav        # detect + classify a long file
uv run python -m src.inference.evaluate_detection data/unannotated/<file>_all.wav  # honest score vs selection table
uv run python -m src.data.analysis.run                         # class-separability analysis (H1–H5)
uv run python -m src.data.analysis.augment_experiment          # signal-feature augmentation (clips)
uv run python -m src.data.analysis.longfile_feature_test       # augmentation on long-file events
uv run python -m src.data.analysis.build_domain_dataset        # build detector-crop training set (TF)
uv run python -m src.data.analysis.train_domain                # train + held-out domain-adaptation eval
```

## 9. Figure / artifact index

- **Class analysis:** `results/analysis/analysis_report.md`, `example_{class}.png`,
  `centroid_distance.png`, `tsne.png`, `scale_migration.png`, `scale_migration_fracs.png`,
  `acoustic_summary.csv`
- **Augmentation:** `results/analysis/augmentation/report_{logistic,mlp}.md`, `cm_*.png`
- **Long-file detection:** `results/detection/<stem>/{events.csv, ground_truth.csv, timeline.png}`
- **Clip classification:** `results/perch_v2_{logistic,MLP}_n500_per_class/`
- **Blind-tiling diagnostic:** `results/inference/`
- **Full project notes:** `CLAUDE.md` ("Data analysis", "Single-pass baseline", "Augmentation
  experiment"), `notebooks/detect.ipynb`
