"""Extract and cache Perch embeddings for the whole dataset.

Runs as its own process (TensorFlow only — no PyTorch import) because loading
TensorFlow and running PyTorch training in the same process segfaults. `main.py`
invokes this via subprocess when any embedding is missing; you can also run it
directly:

    uv run python -m src.model.extract
    uv run python -m src.model.extract --force-recompute
"""
import argparse

from src.data.config import CACHE_DIR, DATA_DIR
from src.data.loader import load_dataset
from src.model.perch import extract_and_cache, load_perch_model


def main():
    parser = argparse.ArgumentParser(description="Extract and cache Perch embeddings")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    samples = load_dataset(DATA_DIR)
    print(f"Extracting embeddings for {len(samples)} samples (cached after first run)...")
    extract_and_cache(samples, load_perch_model, CACHE_DIR, force=args.force_recompute)
    print("Done.")


if __name__ == "__main__":
    main()
