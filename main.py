import argparse

from src.data.config import CACHE_DIR, CLASSES, DATA_DIR, PERCH_MODEL_NAME
from src.data.loader import load_dataset
from src.model.perch import extract_and_cache, load_perch_model
from src.training.evaluate import evaluate
from src.training.train import build_arrays, label_names, train_classifier


def main():
    parser = argparse.ArgumentParser(description="Cod drumming classifier using Perch 2.0")
    parser.add_argument("--classifier", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute embeddings even if cache exists")
    args = parser.parse_args()

    print("Loading dataset...")
    samples = load_dataset(DATA_DIR)
    print(f"  {len(samples)} samples across {len(CLASSES)} classes")

    print(f"\nLoading Perch model ({PERCH_MODEL_NAME})...")
    model = load_perch_model(PERCH_MODEL_NAME)

    print("\nExtracting embeddings (cached after first run)...")
    embeddings = extract_and_cache(samples, model, CACHE_DIR, force=args.force_recompute)

    print("\nBuilding feature arrays...")
    X_train, y_train = build_arrays(samples, embeddings, "train")
    X_val,   y_val   = build_arrays(samples, embeddings, "val")
    X_test,  y_test  = build_arrays(samples, embeddings, "test")
    print(f"  train: {X_train.shape[0]}  val: {X_val.shape[0]}  test: {X_test.shape[0]}")

    print(f"\nTraining {args.classifier} classifier...")
    clf = train_classifier(X_train, y_train, args.classifier)

    evaluate(clf, X_val,  y_val,  label_names(), "val")
    evaluate(clf, X_test, y_test, label_names(), "test")


if __name__ == "__main__":
    main()
