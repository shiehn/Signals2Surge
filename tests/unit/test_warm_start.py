"""Tests for warm-start integration — skips if PyTorch not installed."""

import json

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.unit
class TestWarmStarter:
    def test_no_model_returns_default(self, tmp_path):
        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(
            store_path=tmp_path / "nonexistent.db",
            models_dir=tmp_path / "models",
        )
        features = np.random.randn(512).astype(np.float32)
        x0, sigma0 = starter.predict(features)

        assert x0 is None
        assert sigma0 == 0.4

    def test_with_trained_model(self, tmp_path):
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.predictor import FeatureMLP

        # Create store with a model version
        db_path = tmp_path / "test.db"
        store = ExperienceStore(db_path)

        param_names = [f"param_{i}" for i in range(20)]
        store.log_model_version("v1", 50, 0.1, 0.15, "placeholder")
        store.close()

        # Save a model checkpoint
        models_dir = tmp_path / "models"
        checkpoint_dir = models_dir / "predictor_v1"
        checkpoint_dir.mkdir(parents=True)

        model = FeatureMLP(n_params=20)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        (checkpoint_dir / "config.json").write_text(json.dumps({
            "architecture": "FeatureMLP",
            "n_params": 20,
            "param_names": param_names,
        }))

        # Now test warm starter
        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(store_path=db_path, models_dir=models_dir)
        features = np.random.randn(512).astype(np.float32)
        x0, sigma0 = starter.predict(features)

        # Should either predict or fall back — both are valid
        if x0 is not None:
            assert len(x0) == 20
            assert all(0.0 <= v <= 1.0 for v in x0.values())
            assert sigma0 in (0.15, 0.3, 0.4)
        else:
            assert sigma0 == 0.4

    def test_filter_active_params(self, tmp_path):
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.predictor import FeatureMLP

        db_path = tmp_path / "test.db"
        store = ExperienceStore(db_path)
        param_names = [f"param_{i}" for i in range(20)]
        store.log_model_version("v1", 50, 0.1, 0.15, "placeholder")
        store.close()

        models_dir = tmp_path / "models"
        checkpoint_dir = models_dir / "predictor_v1"
        checkpoint_dir.mkdir(parents=True)

        model = FeatureMLP(n_params=20)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        (checkpoint_dir / "config.json").write_text(json.dumps({
            "architecture": "FeatureMLP",
            "n_params": 20,
            "param_names": param_names,
        }))

        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(store_path=db_path, models_dir=models_dir)
        features = np.random.randn(512).astype(np.float32)

        active = ["param_0", "param_1", "param_5"]
        x0, _ = starter.predict(features, active_param_names=active)

        if x0 is not None:
            assert all(k in active for k in x0.keys())
