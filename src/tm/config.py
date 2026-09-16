"""Configuration for the Tsetlin Machine track.

Deliberately separate from src/data/config.py: the TM arm is an independent
track (course IKT464-G) and must not entangle with the Perch pipeline's
constants. RESULTS_DIR is reused so both tracks agree on output layout, and
labels come from src/data/config.CLASSES so both tracks agree on classes.
"""
from pathlib import Path

from src.data.config import RESULTS_DIR

# --- Audio windowing -------------------------------------------------------
TM_SAMPLE_RATE = 8_000          # cod energy is 50-1000 Hz; a 4 kHz Nyquist is ample
TM_WINDOW_MS = 150.0            # keeps 100% of clicks and 97% of vocals whole
TM_WINDOW_SAMPLES = 1_200       # 150 ms x 8 kHz

# --- Envelope feature ------------------------------------------------------
TM_ENVELOPE_FRAME_MS = 2.0
TM_ENVELOPE_FRAME_SAMPLES = 16  # 2 ms x 8 kHz
TM_ENVELOPE_FRAMES = 75         # 1200 / 16

# --- Booleanization --------------------------------------------------------
TM_BITS = 4                     # thermometer bits per value

# --- Tsetlin Machine -------------------------------------------------------
# Starting values from the MixCTME grid search (s 2-15, T 100-400,
# clauses 200-1000). Starting points, not results.
TM_CLAUSES = 500
TM_T = 300
TM_S = 10.0
TM_MAX_INCLUDED_LITERALS = 32
TM_EPOCHS = 100
TM_SEED = 42                    # MUST NOT be 0: seed=0 hangs TMClassifier.fit()

# --- Paths -----------------------------------------------------------------
TM_CACHE_DIR = Path("data/tm_cache")
TM_RESULTS_DIR = RESULTS_DIR / "tm"
