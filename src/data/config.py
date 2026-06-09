from pathlib import Path

DATA_DIR = Path("data/annotated")
CACHE_DIR = Path("data/embeddings_cache")
MODEL_DIR = Path("models")

PERCH_MODEL_NAME = "perch_v2"
PERCH_SAMPLE_RATE = 32_000
PERCH_WINDOW_SAMPLES = 160_000  # 5s × 32kHz

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
