"""Tests for the training loop — skips if PyTorch not installed."""

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.unit
class TestTrainer:
    def _populate_store(self, store, n=50, n_params=20):
        """Add synthetic training data to the store."""
        param_names = [f"param_{i}" for i in range(n_params)]
        for _ in range(n):
            features = np.random.randn(512).astype(np.float32)
            params = np.random.uniform(0, 1, n_params).astype(np.float32)
            store.log_run(
                run_id=store.new_run_id(),
                target_features=features,
                best_params=params,
                param_names=param_names,
                best_loss=float(np.random.uniform(0.1, 5.0)),
                total_trials=0,
                ground_truth_params=params,
                generation_mode="random",
            )

    def test_train_predictor(self, tmp_path):
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        db_path = tmp_path / "train_test.db"
        store = ExperienceStore(db_path)
        self._populate_store(store, n=30, n_params=20)
        store.close()

        result = train_predictor(
            store_path=db_path,
            models_dir=tmp_path / "models",
            max_epochs=5,
            patience=3,
        )

        assert result is not None
        assert result.n_training_samples == 30
        assert result.epochs_trained > 0
        assert result.best_val_loss < float("inf")
        assert (tmp_path / "models" / f"predictor_{result.version_id}" / "model.pt").exists()

    def test_train_not_enough_data(self, tmp_path):
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        db_path = tmp_path / "small.db"
        store = ExperienceStore(db_path)
        self._populate_store(store, n=5, n_params=10)
        store.close()

        result = train_predictor(
            store_path=db_path,
            models_dir=tmp_path / "models",
            max_epochs=5,
        )
        assert result is None

    def test_tier_weights(self):
        from synth2surge.ml.trainer import _build_tier_weights

        names = [
            "a_osc_1_type",      # tier 1
            "a_filter_1_cutoff", # tier 1
            "a_osc_1_pitch",     # tier 2
            "a_lfo_1_rate",      # tier 2
            "some_random_param", # tier 3
        ]
        weights = _build_tier_weights(names)
        assert weights[0] == 3.0  # tier 1
        assert weights[1] == 3.0  # tier 1
        assert weights[2] == 1.5  # tier 2
        assert weights[3] == 1.5  # tier 2
        assert weights[4] == 1.0  # tier 3
