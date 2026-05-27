from pathlib import Path

DATA_DIR = Path("data/annotated")
CACHE_DIR = Path("data/embeddings_cache")
MODEL_DIR = Path("models")

PERCH_MODEL_NAME = "perch_v2"
PERCH_SAMPLE_RATE = 32_000
PERCH_WINDOW_SAMPLES = 160_000  # 5s × 32kHz

CLASSES = ["click", "other", "silence", "vocal", "water"]  # NA excluded
