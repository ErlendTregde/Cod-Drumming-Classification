import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from src.data.config import CLASSES
from src.data.loader import AudioSample

_label_encoder = LabelEncoder().fit(CLASSES)


def build_arrays(
    samples: list[AudioSample],
    embeddings: dict[str, np.ndarray],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack embeddings and encode labels for a given split.

    Returns X of shape (N, 1280) and y of shape (N,).
    """
    split_samples = [s for s in samples if s.split == split]
    X = np.stack([embeddings[str(s.path)] for s in split_samples])
    y = _label_encoder.transform([s.label for s in split_samples])
    return X, y


def train_classifier(X: np.ndarray, y: np.ndarray, classifier: str = "logistic"):
    """Fit and return a trained sklearn classifier on embedding features."""
    if classifier == "logistic":
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500)
    clf.fit(X, y)
    return clf


def label_names() -> list[str]:
    return list(_label_encoder.classes_)
