"""Training loop for the FeatureMLP parameter predictor.

Trains FeatureMLP on accumulated experience data.
Supports weighted per-tier MSE loss, early stopping, and model checkpointing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainResult:
    """Result of a training run."""

    version_id: str
    n_training_samples: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    epochs_trained: int
    checkpoint_path: str


def _get_device() -> torch.device:
    """Get best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_tier_weights(
    param_names: list[str],
    tier1_weight: float = 3.0,
    tier2_weight: float = 1.5,
    tier3_weight: float = 1.0,
) -> np.ndarray:
    """Build per-parameter weights based on tier classification."""
    from synth2surge.optimizer.loop import classify_parameter_tier

    weights = np.ones(len(param_names), dtype=np.float32)
    for i, name in enumerate(param_names):
        tier = classify_parameter_tier(name)
        if tier == 1:
            weights[i] = tier1_weight
        elif tier == 2:
            weights[i] = tier2_weight
        else:
            weights[i] = tier3_weight
    return weights


def train_predictor(
    store_path: Path,
    models_dir: Path,
    *,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.15,
) -> TrainResult | None:
    """Train a FeatureMLP predictor on accumulated experience data.

    Args:
        store_path: Path to the experience store database.
        models_dir: Directory to save model checkpoints.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Training batch size.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        val_fraction: Fraction of data for validation.

    Returns:
        TrainResult with metrics, or None if not enough data.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: uv pip install 'synth2surge[ml]'")

    from synth2surge.ml.experience_store import ExperienceStore
    from synth2surge.ml.predictor import FeatureMLP

    store = ExperienceStore(store_path)

    # Prefer ground truth data (self-play), fall back to best-params
    features, params, param_names = store.get_ground_truth_data()
    if features.shape[0] == 0:
        features, params, param_names = store.get_training_data()

    if features.shape[0] < 10:
        logger.warning(f"Only {features.shape[0]} training samples, need at least 10")
        store.close()
        return None

    n_params = params.shape[1]
    feature_dim = features.shape[1]
    logger.info(
        f"Training on {features.shape[0]} samples, {n_params} parameters, "
        f"{feature_dim}-dim features"
    )

    # Build tier weights
    tier_weights = _build_tier_weights(param_names)
    tier_weights_tensor = torch.tensor(tier_weights, dtype=torch.float32)

    # Train/val split
    n_total = features.shape[0]
    n_val = max(1, int(n_total * val_fraction))
    n_train = n_total - n_val

    indices = np.random.RandomState(42).permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    x_train = torch.tensor(features[train_idx], dtype=torch.float32)
    y_train = torch.tensor(params[train_idx], dtype=torch.float32)
    x_val = torch.tensor(features[val_idx], dtype=torch.float32)
    y_val = torch.tensor(params[val_idx], dtype=torch.float32)

    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_ds, batch_size=min(batch_size, n_train), shuffle=True, drop_last=False
    )

    # Create model
    device = _get_device()
    model = FeatureMLP(n_params, feature_dim=feature_dim).to(device)
    tier_weights_tensor = tier_weights_tensor.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(max_epochs):
        # Train
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = model(batch_x)
            loss = (tier_weights_tensor * (pred - batch_y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val.to(device))
            val_loss = (tier_weights_tensor * (val_pred - y_val.to(device)) ** 2).mean().item()

        avg_train_loss = sum(train_losses) / len(train_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Log every 5 epochs (or every epoch for small runs)
        log_interval = 5 if max_epochs > 20 else 1
        if epoch % log_interval == 0 or epoch == max_epochs - 1:
            marker = " *" if val_loss <= best_val_loss else ""
            logger.info(
                f"Epoch {epoch:>4d}/{max_epochs}  "
                f"train={avg_train_loss:.6f}  val={val_loss:.6f}  "
                f"best={best_val_loss:.6f}{marker}"
            )

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Save best model
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    version_id = f"v{store.count()}"
    checkpoint_dir = models_dir / f"predictor_{version_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(best_state, checkpoint_path)

    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps({
        "architecture": "FeatureMLP",
        "n_params": n_params,
        "feature_dim": feature_dim,
        "n_training_samples": n_total,
        "best_val_loss": best_val_loss,
        "param_names": param_names,
    }))

    # Log to store
    store.log_model_version(
        version_id=version_id,
        n_training_runs=n_total,
        train_loss=avg_train_loss,
        val_loss=best_val_loss,
        checkpoint_path=str(checkpoint_path),
    )
    store.close()

    result = TrainResult(
        version_id=version_id,
        n_training_samples=n_total,
        final_train_loss=avg_train_loss,
        final_val_loss=val_loss,
        best_val_loss=best_val_loss,
        epochs_trained=epoch + 1,
        checkpoint_path=str(checkpoint_path),
    )
    logger.info(f"Training complete: {result}")
    return result
