"""One reproducible feedforward binary classifier and its training routines."""

import copy
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class RetentionANN(nn.Module):
    """Feedforward network returning one raw logit per customer."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(size < 1 for size in hidden_dims):
            raise ValueError("Network dimensions must be positive.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        layers: list[nn.Module] = []
        width = input_dim
        for size in hidden_dims:
            layers.extend([nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)])
            width = size
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X).squeeze(-1)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch for repeatable CPU experiments."""
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    """Use CPU by default, or explicitly requested available Apple MPS."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    raise ValueError(f"Requested device is unavailable or unsupported: {requested}")


def positive_weight(y: np.ndarray) -> float:
    """Compute negative/positive ratio from the labels used in this fit only."""
    labels = np.asarray(y, dtype=int)
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training requires both classes.")
    return negative / positive


def predict_probabilities(
    model: RetentionANN, X: np.ndarray, device: torch.device
) -> np.ndarray:
    """Return finite positive-class probabilities with model in eval mode."""
    model.eval()
    with torch.inference_mode():
        tensor = torch.as_tensor(np.asarray(X, dtype=np.float32), device=device)
        probabilities = torch.sigmoid(model(tensor)).cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("ANN produced nonfinite probabilities.")
    return probabilities


def train_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    settings: dict,
    device: torch.device,
    seed: int,
    weighted: bool,
    epochs: int,
    X_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
) -> tuple[RetentionANN, list[dict], int, float | None]:
    """Train with optional inner-validation AP stopping; restore best state."""
    if (X_valid is None) != (y_valid is None):
        raise ValueError("Validation features and labels must be supplied together.")
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    set_seed(seed)
    model = RetentionANN(
        X_train.shape[1], tuple(settings["hidden_dims"]), settings["dropout"]
    ).to(device)
    weight = positive_weight(y_train) if weighted else None
    weight_tensor = (
        torch.tensor(weight, dtype=torch.float32, device=device)
        if weight is not None
        else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    if settings["optimizer"] != "AdamW":
        raise ValueError("Only AdamW is supported.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    features = torch.as_tensor(np.asarray(X_train, dtype=np.float32))
    targets = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(features, targets),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    validation = X_valid is not None
    if validation:
        valid_features = torch.as_tensor(
            np.asarray(X_valid, dtype=np.float32), device=device
        )
        valid_targets = torch.as_tensor(
            np.asarray(y_valid, dtype=np.float32), device=device
        )
    history: list[dict] = []
    best_ap = -1.0
    best_epoch = epochs
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            if not torch.isfinite(loss):
                raise ValueError("ANN training loss is nonfinite.")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
        row = {"epoch": epoch, "train_loss": total_loss / len(targets)}
        if validation:
            model.eval()
            with torch.inference_mode():
                logits = model(valid_features)
                valid_loss = criterion(logits, valid_targets)
                probability = torch.sigmoid(logits).cpu().numpy()
            if not torch.isfinite(valid_loss) or not np.isfinite(probability).all():
                raise ValueError("ANN validation loss or probability is nonfinite.")
            ap = float(average_precision_score(y_valid, probability))
            row.update(valid_loss=float(valid_loss.item()), valid_ap=ap)
            if ap > best_ap + 1e-8:
                best_ap, best_epoch, stale = ap, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(row)
        if validation and stale >= settings["early_stopping_patience"]:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch, weight
