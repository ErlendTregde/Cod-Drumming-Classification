from pathlib import Path

DATA_DIR = Path("data/annotated")
CACHE_DIR = Path("data/embeddings_cache")
INFERENCE_CACHE_DIR = Path("data/inference_cache")  # blind-window embeddings (infer.py)
DETECTION_CACHE_DIR = Path("data/detection_cache")  # detected-event embeddings (detect.py)
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

PERCH_MODEL_NAME = "perch_v2"
PERCH_SAMPLE_RATE = 32_000
PERCH_WINDOW_SAMPLES = 160_000  # 5s × 32kHz
DEFAULT_HOP_SAMPLES = PERCH_WINDOW_SAMPLES  # sliding-window step (no overlap by default)

# Event detection (detect.py) — find short transient events in a long recording.
# The annotated events are milliseconds long, so we detect them at their own
# timescale (energy/onset on a band-passed envelope) instead of blind 5s tiling.
DETECT_BANDPASS_HZ = (50.0, 2000.0)  # cod sounds are low-frequency; band-pass before energy
DETECT_FRAME_MS = 5.0                # energy-envelope frame/hop in ms
DETECT_THRESHOLD_K = 3.0             # threshold = median + k * MAD of the envelope (lower = more sensitive)
DETECT_MIN_EVENT_MS = 5.0           # discard events shorter than this
DETECT_MERGE_GAP_MS = 40.0          # merge gap: coalesce pulsed vocal grunts into one event
                                    # (too small fragments vocals into click-like pulses)
DETECT_MAX_EVENT_MS = 500.0         # split longer runs so click trains aren't one "noise" blob
DETECT_PAD_MS = 2.0                 # tight: Perch's 5s pooling makes the label width-sensitive
                                    # (same click reads click @<=10ms, vocal @25-50ms, other @>=100ms)
# Multi-scale extraction — DEPRECATED EXPERIMENT (reverted 2026-06-10). Each event
# was embedded at several widths and the most-confident (width, class) won. The data
# analysis (src/data/analysis/) showed the 10ms crop collapses EVERY class to `click`
# and the `max` rule then over-calls click; the supported path is now single-pass
# native-extent (extract_event_windows). Kept only for `extract_multiscale_windows`
# / the write-up. See CLAUDE.md "Data analysis" / "Single-pass baseline".
DETECT_SCALES_MS = (10.0, 50.0, 200.0)

CLASSES = ["click", "other", "silence", "vocal", "water"]  # NA excluded

# Classifier training (PyTorch)
TORCH_SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MLP_HIDDEN = (256, 64)
MLP_DROPOUT = 0.3
