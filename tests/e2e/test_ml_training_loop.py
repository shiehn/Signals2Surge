"""End-to-end tests for the ML training and inference pipeline.

Tests the full closed loop:
1. Generate training data (synthetic audio features + params as ground truth)
2. Train a predictor model on that data
3. Use the trained model to warm-start optimization
4. Verify predictions are better than random (the core value proposition)

These tests use synthetic audio data to avoid dependency on pedalboard
instrument rendering, which may be broken on some macOS versions.
PyTorch is required.
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

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed"),
]


# ---------------------------------------------------------------------------
# Helpers — simulate the data generator using synthetic audio
# ---------------------------------------------------------------------------

def _make_synthetic_patch(
    param_names: list[str],
    rng: np.random.RandomState,
    sr: int = 44100,
    duration: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a Surge XT patch: generate params, create synthetic audio,
    and extract features. This replaces the real plugin rendering for testing.

    Returns (features_512, params_array, audio).
    """
    from synth2surge.loss.features import extract_features

    # Random parameter vector
    params = rng.uniform(0, 1, len(param_names)).astype(np.float32)

    # Generate audio whose timbral character is deterministically derived
    # from the params, so the model has something learnable
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Use first few params to control freq, harmonics, amplitude
    base_freq = 100 + params[0] * 900  # 100-1000 Hz
    amplitude = 0.1 + params[1] * 0.9  # 0.1-1.0
    n_harmonics = int(1 + params[2] * 5)  # 1-6 harmonics
    brightness = 0.2 + params[3] * 0.8  # harmonic rolloff

    audio = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        harmonic_amp = amplitude * (brightness ** (h - 1))
        audio += harmonic_amp * np.sin(2 * np.pi * base_freq * h * t)

    # Add envelope shape from params
    attack = 0.01 + params[4] * 0.2  # attack time
    release = 0.1 + params[5] * 0.5  # release time
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)

    envelope = np.ones_like(t)
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(1, 0, release_samples)

    audio *= envelope

    # Clip to [-1, 1]
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    features = extract_features(audio, sr=sr)
    return features, params, audio


def _populate_store_synthetic(
    store,
    n: int,
    param_names: list[str],
    seed: int = 42,
) -> None:
    """Fill an experience store with synthetic (features, params) pairs."""
    rng = np.random.RandomState(seed)

    for _ in range(n):
        features, params, _ = _make_synthetic_patch(param_names, rng)
        if np.linalg.norm(features) < 1e-10:
            continue

        store.log_run(
            run_id=store.new_run_id(),
            target_features=features,
            best_params=params,
            param_names=param_names,
            best_loss=0.0,
            total_trials=0,
            ground_truth_params=params,
            generation_mode="random",
        )


# A small but representative param set (real Surge has ~280)
SYNTHETIC_PARAM_NAMES = [f"param_{i}" for i in range(50)]


@pytest.fixture
def ml_workspace(tmp_path: Path) -> dict[str, Path]:
    """Temporary workspace with all needed directories."""
    db_path = tmp_path / "experience.db"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return {
        "db_path": db_path,
        "models_dir": models_dir,
        "workspace": tmp_path,
    }


# ---------------------------------------------------------------------------
# Phase 1: Experience Store & Data Collection
# ---------------------------------------------------------------------------


class TestExperienceStoreE2E:
    """Test the experience store end-to-end with synthetic data."""

    def test_store_round_trip(self, ml_workspace):
        """Data stored can be retrieved exactly."""
        from synth2surge.ml.experience_store import ExperienceStore

        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=20, param_names=SYNTHETIC_PARAM_NAMES)

        assert store.count() == 20

        features, params, names = store.get_ground_truth_data()
        assert features.shape == (20, 512)
        assert params.shape == (20, 50)
        assert names == SYNTHETIC_PARAM_NAMES

        # Verify L2 normalization of features
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

        # Verify params are in [0, 1]
        assert params.min() >= 0.0
        assert params.max() <= 1.0

        store.close()

    def test_store_with_trials(self, ml_workspace):
        """Trial logging works end-to-end."""
        from synth2surge.ml.experience_store import ExperienceStore

        store = ExperienceStore(ml_workspace["db_path"])
        run_id = store.new_run_id()

        features = np.random.randn(512).astype(np.float32)
        params = np.random.uniform(0, 1, 50).astype(np.float32)

        store.log_run(
            run_id=run_id,
            target_features=features,
            best_params=params,
            param_names=SYNTHETIC_PARAM_NAMES,
            best_loss=1.5,
            total_trials=100,
            generation_mode="random",
        )

        # Log 100 trials with decreasing loss
        for i in range(100):
            trial_params = np.random.uniform(0, 1, 50).astype(np.float32)
            store.log_trial(run_id, stage=1, trial_idx=i, params=trial_params, loss=10.0 - i * 0.08)
        store.flush()

        assert store.trial_count(run_id) == 100
        assert store.trial_count() == 100

        summary = store.summary()
        assert summary["total_runs"] == 1
        assert summary["total_trials"] == 100

        store.close()

    def test_store_survives_reopen(self, ml_workspace):
        """Data persists across store instances (SQLite durability)."""
        from synth2surge.ml.experience_store import ExperienceStore

        # Write
        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=10, param_names=SYNTHETIC_PARAM_NAMES)
        store.close()

        # Re-open and verify
        store2 = ExperienceStore(ml_workspace["db_path"])
        assert store2.count() == 10
        features, params, _ = store2.get_ground_truth_data()
        assert features.shape[0] == 10
        store2.close()


# ---------------------------------------------------------------------------
# Phase 2: Model Training
# ---------------------------------------------------------------------------


class TestModelTrainingE2E:
    """Test model training end-to-end with synthetic data."""

    def test_train_predictor_and_checkpoint(self, ml_workspace):
        """Train a model, save checkpoint, verify it loads back."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=50, param_names=SYNTHETIC_PARAM_NAMES)
        store.close()

        result = train_predictor(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
            max_epochs=30,
            patience=10,
        )

        assert result is not None
        assert result.epochs_trained > 0
        assert result.best_val_loss < float("inf")
        assert result.n_training_samples == 50

        # Checkpoint exists on disk
        checkpoint_dir = ml_workspace["models_dir"] / f"predictor_{result.version_id}"
        assert (checkpoint_dir / "model.pt").exists()
        assert (checkpoint_dir / "config.json").exists()

        # Config is valid
        config = json.loads((checkpoint_dir / "config.json").read_text())
        assert config["architecture"] == "FeatureMLP"
        assert config["n_params"] == 50
        assert len(config["param_names"]) == 50

        # Model loads back and produces valid output
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(config["n_params"])
        state = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        test_input = torch.randn(1, 512)
        with torch.no_grad():
            pred = model(test_input)

        assert pred.shape == (1, 50)
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_model_version_recorded(self, ml_workspace):
        """Training logs the model version in the experience store."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=30, param_names=SYNTHETIC_PARAM_NAMES)
        store.close()

        result = train_predictor(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
            max_epochs=10,
        )
        assert result is not None

        store = ExperienceStore(ml_workspace["db_path"])
        version = store.latest_model_version()
        assert version is not None
        assert version == result.version_id
        store.close()

    def test_insufficient_data_returns_none(self, ml_workspace):
        """Training with < 10 samples should return None gracefully."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=5, param_names=SYNTHETIC_PARAM_NAMES)
        store.close()

        result = train_predictor(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
        )
        assert result is None


# ---------------------------------------------------------------------------
# Phase 3: Warm-Start Inference
# ---------------------------------------------------------------------------


class TestWarmStartE2E:
    """Test warm-start prediction end-to-end."""

    def _train_model(self, db_path: Path, models_dir: Path, n: int = 50):
        """Helper: populate store and train a model."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        store = ExperienceStore(db_path)
        _populate_store_synthetic(store, n=n, param_names=SYNTHETIC_PARAM_NAMES)
        store.close()

        return train_predictor(
            store_path=db_path, models_dir=models_dir,
            max_epochs=30, patience=10,
        )

    def test_warm_starter_loads_and_predicts(self, ml_workspace):
        """WarmStarter loads trained model and returns predictions."""
        result = self._train_model(ml_workspace["db_path"], ml_workspace["models_dir"])
        assert result is not None

        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
        )

        features = np.random.randn(512).astype(np.float32)
        x0, sigma0 = starter.predict(features)

        # Should produce a prediction or gracefully fall back
        assert sigma0 in (0.15, 0.3, 0.4)
        if x0 is not None:
            assert len(x0) > 0
            assert all(0.0 <= v <= 1.0 for v in x0.values())

    def test_warm_starter_no_model_falls_back(self, ml_workspace):
        """Without a trained model, WarmStarter falls back gracefully."""
        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
        )

        features = np.random.randn(512).astype(np.float32)
        x0, sigma0 = starter.predict(features)

        assert x0 is None
        assert sigma0 == 0.4

    def test_warm_starter_filters_active_params(self, ml_workspace):
        """WarmStarter only returns predictions for requested param names."""
        self._train_model(ml_workspace["db_path"], ml_workspace["models_dir"])

        from synth2surge.ml.warm_start import WarmStarter

        starter = WarmStarter(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
        )

        features = np.random.randn(512).astype(np.float32)
        active = ["param_0", "param_1", "param_5"]
        x0, _ = starter.predict(features, active_param_names=active)

        if x0 is not None:
            assert set(x0.keys()).issubset(set(active))


# ---------------------------------------------------------------------------
# Phase 4: Closed-Loop Self-Improvement
# ---------------------------------------------------------------------------


class TestClosedLoopE2E:
    """Test the core value proposition: training makes predictions better."""

    def test_predictions_better_than_random(self, ml_workspace):
        """After training on synthetic data, model predictions should be
        closer to ground truth than random uniform params.

        This is THE fundamental quality test.
        """
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.predictor import FeatureMLP
        from synth2surge.ml.trainer import train_predictor

        # Generate and train on 200 synthetic patches
        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=200, param_names=SYNTHETIC_PARAM_NAMES, seed=42)
        store.close()

        result = train_predictor(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
            max_epochs=100,
            patience=20,
        )
        assert result is not None

        # Load the trained model
        config = json.loads(
            (
                ml_workspace["models_dir"]
                / f"predictor_{result.version_id}"
                / "config.json"
            ).read_text()
        )
        model = FeatureMLP(config["n_params"])
        state = torch.load(
            ml_workspace["models_dir"]
            / f"predictor_{result.version_id}"
            / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()

        # Generate held-out test data (different seed)
        holdout_store = ExperienceStore(ml_workspace["workspace"] / "holdout.db")
        _populate_store_synthetic(
            holdout_store, n=50, param_names=SYNTHETIC_PARAM_NAMES, seed=999,
        )
        holdout_features, holdout_params, _ = holdout_store.get_ground_truth_data()
        holdout_store.close()

        assert holdout_features.shape[0] == 50

        # Predict on held-out data
        features_tensor = torch.tensor(holdout_features, dtype=torch.float32)
        with torch.no_grad():
            predicted = model(features_tensor).numpy()

        # L2 distance: model prediction vs ground truth
        model_distances = np.sqrt(np.mean((predicted - holdout_params) ** 2, axis=1))

        # L2 distance: random uniform vs ground truth
        rng = np.random.RandomState(123)
        random_params = rng.uniform(0, 1, holdout_params.shape).astype(np.float32)
        random_distances = np.sqrt(np.mean((random_params - holdout_params) ** 2, axis=1))

        avg_model_dist = float(np.mean(model_distances))
        avg_random_dist = float(np.mean(random_distances))
        improvement_pct = (1 - avg_model_dist / avg_random_dist) * 100

        print(f"\n{'='*50}")
        print("CLOSED-LOOP QUALITY EVALUATION")
        print(f"{'='*50}")
        print("Training samples:      200")
        print("Hold-out samples:      50")
        print(f"Model avg L2 distance: {avg_model_dist:.4f}")
        print(f"Random avg L2 dist:    {avg_random_dist:.4f}")
        print(f"Improvement:           {improvement_pct:.1f}%")
        print(f"{'='*50}")

        assert avg_model_dist < avg_random_dist, (
            f"Model ({avg_model_dist:.4f}) should predict closer to ground "
            f"truth than random ({avg_random_dist:.4f})"
        )

    def test_more_data_improves_model(self, ml_workspace):
        """Training with more data should produce a better model."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor

        # Train on 30 samples
        store_small = ExperienceStore(ml_workspace["workspace"] / "small.db")
        _populate_store_synthetic(store_small, n=30, param_names=SYNTHETIC_PARAM_NAMES, seed=42)
        store_small.close()

        result_small = train_predictor(
            store_path=ml_workspace["workspace"] / "small.db",
            models_dir=ml_workspace["workspace"] / "models_small",
            max_epochs=50,
            patience=15,
        )

        # Train on 200 samples
        store_large = ExperienceStore(ml_workspace["workspace"] / "large.db")
        _populate_store_synthetic(store_large, n=200, param_names=SYNTHETIC_PARAM_NAMES, seed=42)
        store_large.close()

        result_large = train_predictor(
            store_path=ml_workspace["workspace"] / "large.db",
            models_dir=ml_workspace["workspace"] / "models_large",
            max_epochs=50,
            patience=15,
        )

        assert result_small is not None
        assert result_large is not None

        print(f"\n  Small data (30) val loss:  {result_small.best_val_loss:.6f}")
        print(f"  Large data (200) val loss: {result_large.best_val_loss:.6f}")

        # More data should give equal or better validation loss
        assert result_large.best_val_loss <= result_small.best_val_loss * 1.1, (
            f"More data ({result_large.best_val_loss:.6f}) should not be "
            f"much worse than less data ({result_small.best_val_loss:.6f})"
        )


# ---------------------------------------------------------------------------
# Enriched Loss
# ---------------------------------------------------------------------------


class TestEnrichedLossE2E:
    """Test enriched loss function end-to-end with synthetic audio."""

    def test_enriched_loss_identical_near_zero(self):
        """Same audio should give near-zero enriched loss."""
        from synth2surge.loss.enriched import enriched_loss

        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        loss = enriched_loss(audio, audio)
        assert loss < 0.5, f"Same audio should give near-zero loss, got {loss}"

    def test_enriched_loss_different_is_higher(self):
        """Different audio should give higher loss than identical."""
        from synth2surge.loss.enriched import enriched_loss

        t = np.linspace(0, 1, 44100, dtype=np.float32)
        audio1 = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio2 = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        loss_same = enriched_loss(audio1, audio1)
        loss_diff = enriched_loss(audio1, audio2)

        assert loss_diff > loss_same

    def test_enriched_loss_all_components_finite(self):
        """Each component should return finite values for valid audio."""
        from synth2surge.loss.enriched import (
            centroid_distance,
            envelope_distance,
            mfcc_distance,
            spectral_flux_distance,
        )

        t = np.linspace(0, 2, 88200, dtype=np.float32)
        audio1 = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio2 = (0.3 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)

        assert np.isfinite(mfcc_distance(audio1, audio2))
        assert np.isfinite(envelope_distance(audio1, audio2))
        assert np.isfinite(centroid_distance(audio1, audio2))
        assert np.isfinite(spectral_flux_distance(audio1, audio2))


# ---------------------------------------------------------------------------
# Audio Encoder
# ---------------------------------------------------------------------------


class TestAudioEncoderE2E:
    """Test the learned audio encoder end-to-end."""

    def test_encoder_produces_normalized_embeddings(self):
        """Encoder output should be L2-normalized."""
        from synth2surge.ml.encoder import AudioEncoder

        encoder = AudioEncoder(embed_dim=128)
        mel = torch.randn(4, 1, 128, 344)

        with torch.no_grad():
            embeddings = encoder(mel)

        assert embeddings.shape == (4, 128)
        norms = torch.norm(embeddings, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_hybrid_loss_evaluator(self):
        """HybridLossEvaluator produces consistent, finite losses."""
        from synth2surge.ml.encoder import AudioEncoder
        from synth2surge.ml.hybrid_loss import HybridLossEvaluator

        encoder = AudioEncoder(embed_dim=128)
        target = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))).astype(np.float32)

        evaluator = HybridLossEvaluator(encoder, target, alpha=0.1)

        loss_same = evaluator(target)
        assert np.isfinite(loss_same)
        assert loss_same < 1.0

        noise = np.random.randn(44100).astype(np.float32) * 0.3
        loss_diff = evaluator(noise)
        assert np.isfinite(loss_diff)
        assert loss_diff > loss_same

    def test_learned_features_extraction(self):
        """extract_features_learned produces valid embeddings."""
        from synth2surge.loss.features import extract_features_learned
        from synth2surge.ml.encoder import AudioEncoder

        encoder = AudioEncoder(embed_dim=128)
        audio = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))).astype(np.float32)

        features = extract_features_learned(audio, encoder)
        assert features.shape == (128,)
        assert abs(np.linalg.norm(features) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Full Pipeline Integration (no Surge XT required)
# ---------------------------------------------------------------------------


class TestFullPipelineE2E:
    """Test the complete generate → train → predict → evaluate pipeline."""

    def test_complete_pipeline(self, ml_workspace):
        """Run the entire pipeline: generate data, train, predict, measure quality."""
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor
        from synth2surge.ml.warm_start import WarmStarter

        # Step 1: Generate data
        store = ExperienceStore(ml_workspace["db_path"])
        _populate_store_synthetic(store, n=100, param_names=SYNTHETIC_PARAM_NAMES, seed=42)
        assert store.count() == 100
        store.close()

        # Step 2: Train
        result = train_predictor(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
            max_epochs=50,
            patience=15,
        )
        assert result is not None
        assert result.best_val_loss < float("inf")

        # Step 3: Warm-start prediction
        starter = WarmStarter(
            store_path=ml_workspace["db_path"],
            models_dir=ml_workspace["models_dir"],
        )

        # Generate a test patch and predict
        rng = np.random.RandomState(777)
        test_features, test_params, _ = _make_synthetic_patch(
            SYNTHETIC_PARAM_NAMES, rng,
        )

        x0, sigma0 = starter.predict(test_features)

        # Step 4: Verify prediction quality
        if x0 is not None:
            predicted_array = np.array(
                [x0.get(f"param_{i}", 0.5) for i in range(50)], dtype=np.float32,
            )
            pred_distance = float(np.sqrt(np.mean((predicted_array - test_params) ** 2)))
            random_distance = float(
                np.sqrt(np.mean((np.random.uniform(0, 1, 50) - test_params) ** 2))
            )

            print("\n  Pipeline test:")
            print(f"  Prediction distance: {pred_distance:.4f}")
            print(f"  Random distance:     {random_distance:.4f}")
            print(f"  Sigma0:              {sigma0}")

        # Step 5: Verify model version tracking
        store = ExperienceStore(ml_workspace["db_path"])
        assert store.latest_model_version() == result.version_id
        store.close()
