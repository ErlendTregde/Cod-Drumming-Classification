import argparse
import subprocess
import sys

from src.data.config import CACHE_DIR, CLASSES, DATA_DIR
from src.data.loader import load_dataset
from src.model.perch import load_cached_embeddings, missing_embeddings
from src.training.evaluate import evaluate
from src.training.train import build_arrays, label_names, train_classifier


def ensure_embeddings(samples, force: bool) -> None:
    """Make sure every sample has a cached embedding.

    Extraction runs in a separate process: it loads TensorFlow (Perch), which
    segfaults if co-resident with PyTorch training in this process.
    """
    if not force and not missing_embeddings(samples, CACHE_DIR):
        print("All embeddings cached.")
        return

    print("Extracting embeddings in a separate process (TensorFlow)...")
    cmd = [sys.executable, "-m", "src.model.extract"]
    if force:
        cmd.append("--force-recompute")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Cod drumming classifier using Perch 2.0")
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    print("Loading dataset...")
    samples = load_dataset(DATA_DIR)
    print(f"  {len(samples)} samples across {len(CLASSES)} classes")

    print("\nPreparing embeddings...")
    ensure_embeddings(samples, force=args.force_recompute)
    embeddings = load_cached_embeddings(samples, CACHE_DIR)

    print("\nBuilding feature arrays...")
    X_train, y_train = build_arrays(samples, embeddings, "train")
    X_val,   y_val   = build_arrays(samples, embeddings, "val")
    X_test,  y_test  = build_arrays(samples, embeddings, "test")
    print(f"  train: {X_train.shape[0]}  val: {X_val.shape[0]}  test: {X_test.shape[0]}")

    print(f"\nTraining {args.classifier} classifier...")
    model = train_classifier(X_train, y_train, X_val, y_val, args.classifier)

    evaluate(model, X_val,  y_val,  label_names(), "val")
    evaluate(model, X_test, y_test, label_names(), "test")


if __name__ == "__main__":
    main()
