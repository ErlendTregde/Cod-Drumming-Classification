from pathlib import Path

import torch
from torch import nn

from src.data.config import MLP_DROPOUT, MLP_HIDDEN


class LinearHead(nn.Module):
    """Logistic-regression equivalent: a single linear layer."""

    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    """Feed-forward MLP: stacked Linear -> ReLU -> Dropout blocks."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_classifier(name: str, in_dim: int, n_classes: int) -> nn.Module:
    """Construct a classifier model by name (`logistic` or `mlp`)."""
    if name == "mlp":
        return MLPHead(in_dim, MLP_HIDDEN, n_classes, MLP_DROPOUT)
    return LinearHead(in_dim, n_classes)


def save_model(model: nn.Module, name: str, in_dim: int, n_classes: int, path: Path) -> None:
    """Persist a trained classifier (weights + the args needed to rebuild it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"name": name, "in_dim": in_dim, "n_classes": n_classes, "state_dict": model.state_dict()},
        path,
    )


def load_model(path: Path) -> nn.Module:
    """Rebuild a classifier from a checkpoint and load its weights (eval mode)."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_classifier(ckpt["name"], ckpt["in_dim"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
