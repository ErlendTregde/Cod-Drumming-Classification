import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.config import (
    BATCH_SIZE,
    CLASSES,
    EARLY_STOP_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    MODEL_DIR,
    TORCH_SEED,
    WEIGHT_DECAY,
)
from src.data.loader import AudioSample
from src.model.classifier import build_classifier

_label_encoder = LabelEncoder().fit(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_arrays(
    samples: list[AudioSample],
    embeddings: dict[str, np.ndarray],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack embeddings and encode labels for a given split.

    Returns X of shape (N, embedding_dim) and y of shape (N,).
    """
    split_samples = [s for s in samples if s.split == split]
    X = np.stack([embeddings[str(s.path)] for s in split_samples])
    y = _label_encoder.transform([s.label for s in split_samples])
    return X, y


def _tensor(X: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(X, dtype=dtype, device=DEVICE)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    classifier: str = "logistic",
) -> nn.Module:
    """Train a PyTorch classifier on embedding features.

    Standard training loop: Adam + cross-entropy over mini-batches, with per-epoch
    logging and early stopping on validation loss (best weights restored). Saves a
    training-curve figure to MODEL_DIR/training_curve.png and returns the trained model.
    """
    torch.manual_seed(TORCH_SEED)

    model = build_classifier(classifier, X_train.shape[1], len(CLASSES)).to(DEVICE)

    Xtr, ytr = _tensor(X_train), _tensor(y_train, torch.long)
    Xva, yva = _tensor(X_val), _tensor(y_val, torch.long)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(Xtr)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_loss = criterion(val_logits, yva).item()
            val_acc = (val_logits.argmax(1) == yva).float().mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"  epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOP_PATIENCE:
                print(f"  early stopping at epoch {epoch} (no val improvement)")
                break

    model.load_state_dict(best_state)
    _save_training_curve(history, MODEL_DIR / "training_curve.png", classifier)
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Class-index predictions for X via a forward pass. Returns a numpy array."""
    model.eval()
    return model(_tensor(X)).argmax(1).cpu().numpy()


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Softmax class probabilities for X. Returns a (N, n_classes) numpy array."""
    model.eval()
    return model(_tensor(X)).softmax(1).cpu().numpy()


def _save_training_curve(history: dict[str, list[float]], out_path: Path, classifier: str) -> None:
    """Two-panel figure: loss (train+val) and val accuracy vs epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train", color="steelblue")
    ax_loss.plot(epochs, history["val_loss"], label="val", color="coral")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["val_acc"], color="seagreen")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy", ylim=(0, 1.05))

    fig.suptitle(f"Training curve — {classifier}", fontsize=13)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curve → {out_path}")


def label_names() -> list[str]:
    return list(_label_encoder.classes_)
