"""E2E test: multi-probe training pipeline with real Surge XT.

Verifies the complete 3072-dim multi-probe pipeline:
1. Render 6 standardized MIDI probes through Surge XT
2. Extract per-segment features → 3072-dim vector
3. Generate training data with multi-probe features
4. Train a model on 3072-dim features
5. Load and use the model for prediction

Requires Surge XT and PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = [
    pytest.mark.requires_surge,
    pytest.mark.e2e,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed"),
]


class TestMultiProbeRendering:
    """Verify standardized probes render correctly through Surge XT."""

    def test_standard_probes_produce_6_segments(self):
        from synth2surge.audio.engine import PluginHost
        from synth2surge.audio.standard_probes import (
            N_STANDARD_PROBES,
            render_standard_features,
        )

        host = PluginHost(SURGE_VST3)
        features, segments = render_standard_features(host)

        assert len(segments) == N_STANDARD_PROBES
        # At least some segments should be non-silent
        audible = sum(
            1 for s in segments if np.sqrt(np.mean(s**2)) > 1e-6
        )
        assert audible >= 3, f"Only {audible}/6 probes produced sound"

    def test_features_are_3072_dim(self):
        from synth2surge.audio.engine import PluginHost
        from synth2surge.audio.standard_probes import (
            STANDARD_FEATURE_DIM,
            render_standard_features,
        )

        host = PluginHost(SURGE_VST3)
        features, _ = render_standard_features(host)

        assert features.shape == (STANDARD_FEATURE_DIM,)
        assert abs(np.linalg.norm(features) - 1.0) < 0.01

    def test_different_patches_different_features(self):
        from synth2surge.audio.engine import PluginHost
        from synth2surge.audio.standard_probes import render_standard_features

        host = PluginHost(SURGE_VST3)

        # Default patch
        feat1, _ = render_standard_features(host)

        # Modify params
        host.set_raw_values({"a_osc_1_type": 0.3, "a_filter_1_cutoff": 0.8})
        host.reset()
        feat2, _ = render_standard_features(host)

        cosine_sim = float(np.dot(feat1, feat2))
        assert cosine_sim < 0.99, f"Different patches should differ, cosine={cosine_sim:.4f}"


class TestMultiProbeDataGeneration:
    """Test training data generation with multi-probe features."""

    def test_generate_produces_3072_dim_features(self, tmp_path):
        from synth2surge.ml.data_generator import generate_render_only
        from synth2surge.ml.experience_store import ExperienceStore

        store = ExperienceStore(tmp_path / "test.db")

        # Generate with multiple seeds to get enough patches
        total = 0
        for seed in [42, 100, 200, 300, 400]:
            n = generate_render_only(
                SURGE_VST3, store, count=5, seed=seed, resume=False,
            )
            total += n
            if total >= 3:
                break

        assert total >= 1, "Should generate at least 1 patch"

        features, params, names = store.get_ground_truth_data()
        assert features.shape[1] == 3072, f"Expected 3072-dim, got {features.shape[1]}"
        assert params.shape[1] == 775
        store.close()


class TestMultiProbeTraining:
    """Test model training on 3072-dim features."""

    def test_train_and_predict_3072(self, tmp_path):
        """Full pipeline: generate → train → predict with 3072-dim features."""
        from synth2surge.ml.data_generator import generate_render_only
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.predictor import FeatureMLP
        from synth2surge.ml.trainer import train_predictor

        db_path = tmp_path / "test.db"
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Generate enough data
        store = ExperienceStore(db_path)
        total = 0
        for seed in range(1, 30):
            n = generate_render_only(
                SURGE_VST3, store, count=5, seed=seed * 100, resume=False,
            )
            total += n
            if total >= 12:
                break
        store.close()

        assert total >= 10, f"Need 10+ patches, got {total}"

        # Train
        result = train_predictor(
            store_path=db_path,
            models_dir=models_dir,
            max_epochs=10,
            patience=5,
        )
        assert result is not None
        assert result.best_val_loss < float("inf")

        # Verify checkpoint has correct feature_dim
        config = json.loads(
            (models_dir / f"predictor_{result.version_id}" / "config.json").read_text()
        )
        assert config["feature_dim"] == 3072

        # Load and predict
        model = FeatureMLP(config["n_params"], feature_dim=config["feature_dim"])
        state = torch.load(
            models_dir / f"predictor_{result.version_id}" / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()

        test_input = torch.randn(1, 3072)
        with torch.no_grad():
            pred = model(test_input)

        assert pred.shape[1] == config["n_params"]
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

        print("\n  Multi-probe pipeline verified:")
        print(f"    Training patches: {total}")
        print(f"    Feature dim:      {config['feature_dim']}")
        print(f"    Model params:     {config['n_params']}")
        print(f"    Val loss:         {result.best_val_loss:.6f}")
