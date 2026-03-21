"""CMA-ES warm-start integration using ML parameter predictions.

Uses the trained predictor to generate initial parameter guesses,
with confidence-based sigma adjustment. Falls back to default
CMA-ES initialization when confidence is low.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WarmStarter:
    """Provides ML-based warm-start parameters for CMA-ES optimization."""

    def __init__(
        self,
        store_path: Path,
        models_dir: Path,
        *,
        confidence_threshold: float = 0.3,
        n_mc_samples: int = 10,
    ) -> None:
        self._store_path = Path(store_path)
        self._models_dir = Path(models_dir)
        self._confidence_threshold = confidence_threshold
        self._n_mc_samples = n_mc_samples
        self._model = None
        self._param_names: list[str] = []
        self._loaded = False

    def _load_model(self) -> bool:
        """Lazy-load the best available model checkpoint.

        Priority order:
        1. Latest trained model (from experience store)
        2. Downloaded pretrained model (from GitHub releases)
        """
        if not TORCH_AVAILABLE:
            return False

        from synth2surge.ml.predictor import FeatureMLP
        from synth2surge.ml.pretrained import find_pretrained

        checkpoint_dir = None

        # Try 1: latest trained model from experience store
        if self._store_path.exists():
            from synth2surge.ml.experience_store import ExperienceStore

            store = ExperienceStore(self._store_path)
            version = store.latest_model_version()
            store.close()

            if version is not None:
                candidate = self._models_dir / f"predictor_{version}"
                if (candidate / "model.pt").exists():
                    checkpoint_dir = candidate
                    logger.info(f"Using trained model {version}")

        # Try 2: downloaded pretrained model
        if checkpoint_dir is None:
            pretrained = find_pretrained(self._models_dir)
            if pretrained is not None:
                checkpoint_dir = pretrained
                logger.info("Using downloaded pretrained model")

        if checkpoint_dir is None:
            return False

        config_path = checkpoint_dir / "config.json"
        model_path = checkpoint_dir / "model.pt"

        if not model_path.exists():
            return False

        config = json.loads(config_path.read_text())
        self._param_names = config["param_names"]
        n_params = config["n_params"]

        self._model = FeatureMLP(n_params)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
        self._loaded = True

        logger.info(f"Loaded predictor model from {checkpoint_dir.name} ({n_params} params)")
        return True

    def predict(
        self,
        target_features: np.ndarray,
        active_param_names: list[str] | None = None,
    ) -> tuple[dict[str, float] | None, float]:
        """Predict initial parameters with confidence estimation.

        Args:
            target_features: 512-dim audio feature vector.
            active_param_names: If provided, only return predictions for these params.

        Returns:
            (x0, sigma0) tuple. x0 is a dict of {param_name: value} or None if
            confidence is too low. sigma0 is the recommended CMA-ES step size.
        """
        if not self._loaded and not self._load_model():
            return None, 0.4  # Default: no warm-start

        features_tensor = torch.tensor(target_features, dtype=torch.float32).unsqueeze(0)

        # MC Dropout: run multiple forward passes with dropout enabled
        self._model.train()  # Enable dropout
        predictions = []
        with torch.no_grad():
            for _ in range(self._n_mc_samples):
                pred = self._model(features_tensor)
                predictions.append(pred.squeeze(0).numpy())

        self._model.eval()  # Restore eval mode

        predictions_np = np.stack(predictions)
        mean_pred = predictions_np.mean(axis=0)
        std_pred = predictions_np.std(axis=0)

        # Confidence: 1 - mean uncertainty (std across MC samples)
        confidence = float(1.0 - np.mean(std_pred))
        confidence = max(0.0, min(1.0, confidence))

        if confidence < self._confidence_threshold:
            logger.info(f"Low confidence ({confidence:.3f}), skipping warm-start")
            return None, 0.4

        # Build x0 dict
        x0 = {}
        for i, name in enumerate(self._param_names):
            if active_param_names is None or name in active_param_names:
                x0[name] = float(np.clip(mean_pred[i], 0.0, 1.0))

        # Adjust sigma based on confidence
        if confidence > 0.6:
            sigma0 = 0.15
        elif confidence > 0.3:
            sigma0 = 0.3
        else:
            sigma0 = 0.4

        logger.info(
            f"Warm-start: confidence={confidence:.3f}, sigma0={sigma0}, "
            f"predicting {len(x0)} params"
        )
        return x0, sigma0
