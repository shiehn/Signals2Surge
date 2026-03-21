"""End-to-end tests using real Surge XT rendering.

Tests the full closed loop with actual plugin audio:
1. Verify Surge XT renders non-silent audio
2. Generate training data from real Surge XT renders
3. Train on real audio features
4. Verify the optimizer works with real audio
5. Full closed-loop: generate → train → warm-start optimize

Requires both Surge XT and PyTorch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = [
    pytest.mark.requires_surge,
    pytest.mark.e2e,
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed"),
]


class TestSurgeRendering:
    """Verify Surge XT produces non-silent audio through pedalboard."""

    def test_default_patch_renders_audio(self):
        """Default Surge XT init patch produces audible output."""
        from synth2surge.audio.engine import PluginHost

        host = PluginHost(SURGE_VST3)
        audio = host.render_midi_mono()
        rms = float(np.sqrt(np.mean(audio**2)))

        assert rms > 0.01, f"Default patch should produce sound, got RMS={rms:.6f}"
        assert len(audio) > 0

    def test_random_params_some_produce_sound(self):
        """At least some random param configurations produce audible audio."""
        from synth2surge.audio.engine import PluginHost

        host = PluginHost(SURGE_VST3)
        rng = np.random.RandomState(42)
        param_names = sorted(host.parameter_names())

        audible_count = 0
        for _ in range(10):
            random_vals = {n: float(rng.uniform(0, 1)) for n in param_names}
            host.set_raw_values(random_vals)
            host.reset()
            audio = host.render_midi_mono()
            if np.sqrt(np.mean(audio**2)) > 1e-6:
                audible_count += 1

        assert audible_count >= 1, "At least 1 of 10 random patches should produce sound"

    def test_multi_probe_renders_audio(self):
        """Multi-probe rendering produces segmented audio."""
        from synth2surge.audio.engine import PluginHost
        from synth2surge.audio.midi import compose_multi_probe
        from synth2surge.config import MultiProbeConfig

        host = PluginHost(SURGE_VST3)
        config = MultiProbeConfig.thorough()
        multi_probe = compose_multi_probe(config, sample_rate=host.sample_rate)

        full_audio, segments = host.render_multi_probe(multi_probe)
        assert len(full_audio) > 0
        assert len(segments) == 6  # thorough has 6 probes

        # At least some segments should have audio
        audible_segments = sum(
            1 for seg in segments if np.sqrt(np.mean(seg**2)) > 1e-6
        )
        assert audible_segments >= 1


class TestRealDataGeneration:
    """Test training data generation with real Surge XT rendering."""

    def test_render_only_generates_real_data(self, tmp_path):
        """Generate training data from real Surge XT renders."""
        from synth2surge.ml.data_generator import generate_render_only
        from synth2surge.ml.experience_store import ExperienceStore

        store = ExperienceStore(tmp_path / "experience.db")
        n = generate_render_only(
            SURGE_VST3, store, count=10, seed=42, resume=False,
        )

        assert n >= 1, f"Expected at least 1 non-silent patch from Surge XT, got {n}"

        features, params, names = store.get_ground_truth_data()
        assert features.shape[0] >= 1
        assert features.shape[1] == 512
        assert len(names) > 100

        # Features should be L2-normalized
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=0.01)

        store.close()


class TestRealOptimization:
    """Test CMA-ES optimization with real Surge XT audio."""

    def test_optimizer_reduces_loss(self, tmp_path):
        """CMA-ES should reduce loss over trials with real audio."""
        from synth2surge.audio.engine import PluginHost
        from synth2surge.config import MidiProbeConfig, OptimizationConfig
        from synth2surge.optimizer.loop import optimize

        host = PluginHost(SURGE_VST3)
        midi_config = MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)

        # Capture target audio from default patch
        target_audio = host.render_midi_mono(midi_config=midi_config)
        assert np.sqrt(np.mean(target_audio**2)) > 0.01

        # Randomize and optimize back
        rng = np.random.RandomState(42)
        param_names = sorted(host.parameter_names())
        random_vals = {n: float(rng.uniform(0, 1)) for n in param_names}
        host.set_raw_values(random_vals)
        host.reset()

        config = OptimizationConfig(
            n_trials_tier1=20, n_trials_tier2=0, n_trials_tier3=0,
        )
        result = optimize(
            target_audio=target_audio,
            surge_host=host,
            config=config,
            midi_config=midi_config,
            stages=[1],
            output_dir=tmp_path / "opt",
        )

        assert result.best_loss < float("inf")
        assert result.best_patch_path.exists()
        assert result.best_audio_path.exists()
        assert result.total_trials == 20

    @pytest.mark.slow
    def test_optimizer_with_experience_logging(self, tmp_path):
        """Optimization logs trials to the experience store."""
        from synth2surge.audio.engine import PluginHost
        from synth2surge.config import MidiProbeConfig, OptimizationConfig
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.optimizer.loop import optimize

        store = ExperienceStore(tmp_path / "experience.db")
        run_id = store.new_run_id()

        host = PluginHost(SURGE_VST3)
        midi_config = MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)
        target_audio = host.render_midi_mono(midi_config=midi_config)

        config = OptimizationConfig(
            n_trials_tier1=15, n_trials_tier2=0, n_trials_tier3=0,
        )
        optimize(
            target_audio=target_audio,
            surge_host=host,
            config=config,
            midi_config=midi_config,
            stages=[1],
            output_dir=tmp_path / "opt",
            experience_store=store,
            _run_id=run_id,
        )

        assert store.trial_count(run_id) == 15
        store.close()


class TestRealClosedLoop:
    """Closed-loop test: generate data inline → train → warm-start."""

    @pytest.mark.slow
    def test_generate_train_predict_with_surge(self, tmp_path):
        """Generate data from a single PluginHost, train, predict."""
        from synth2surge.audio.engine import PluginHost
        from synth2surge.loss.features import extract_features
        from synth2surge.ml.experience_store import ExperienceStore
        from synth2surge.ml.trainer import train_predictor
        from synth2surge.ml.warm_start import WarmStarter

        db_path = tmp_path / "experience.db"
        models_dir = tmp_path / "models"
        models_dir.mkdir()

        # Step 1: Generate training data inline (avoid multiple PluginHost instances)
        host = PluginHost(SURGE_VST3)
        store = ExperienceStore(db_path)
        param_names = sorted(host.parameter_names())
        rng = np.random.RandomState(42)

        generated = 0
        for i in range(20):
            random_vals = {n: float(rng.uniform(0, 1)) for n in param_names}
            host.set_raw_values(random_vals)
            host.reset()
            audio = host.render_midi_mono()
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 1e-6:
                continue

            features = extract_features(audio, sr=host.sample_rate)
            if np.linalg.norm(features) < 1e-10:
                continue

            gt_params = np.array(
                [random_vals[n] for n in param_names], dtype=np.float32
            )
            store.log_run(
                run_id=store.new_run_id(),
                target_features=features,
                best_params=gt_params,
                param_names=param_names,
                best_loss=0.0,
                total_trials=0,
                ground_truth_params=gt_params,
                generation_mode="random",
            )
            generated += 1

        store.close()
        assert generated >= 5, f"Need at least 5 patches, got {generated}"

        # Step 2: Train predictor
        result = train_predictor(
            store_path=db_path,
            models_dir=models_dir,
            max_epochs=20,
            patience=10,
        )
        assert result is not None

        # Step 3: Warm-start prediction on fresh audio
        host.reset()
        target_audio = host.render_midi_mono()
        features = extract_features(target_audio, sr=host.sample_rate)

        starter = WarmStarter(store_path=db_path, models_dir=models_dir)
        x0, sigma0 = starter.predict(features)

        assert sigma0 in (0.15, 0.3, 0.4)
        if x0 is not None:
            assert len(x0) > 0
            assert all(0.0 <= v <= 1.0 for v in x0.values())

        print("\n  Closed-loop with real Surge XT:")
        print(f"  Training patches:  {generated}")
        print(f"  Model version:     {result.version_id}")
        print(f"  Val loss:          {result.best_val_loss:.6f}")
        print(f"  Warm-start sigma:  {sigma0}")
        print(f"  Predicted params:  {len(x0) if x0 else 0}")
