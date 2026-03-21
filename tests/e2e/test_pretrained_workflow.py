"""E2E test: download pretrained model from GitHub and use it to generate a preset.

Follows the README Quick Start workflow:
1. synth2surge train download
2. synth2surge capture --plugin ... --no-gui
3. synth2surge optimize --target ... --warm-start

Requires Surge XT, PyTorch, and internet access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

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


class TestPretrainedWorkflow:
    """Follow the README Quick Start exactly: download → capture → optimize."""

    def test_download_capture_optimize(self, tmp_path: Path):
        """Full workflow: download pretrained model, capture audio, optimize with warm-start."""

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        models_dir = workspace / "models"

        # ──────────────────────────────────────────────────
        # Step 1: Download pretrained model from GitHub
        # ──────────────────────────────────────────────────
        from synth2surge.ml.pretrained import download_pretrained

        checkpoint_dir = download_pretrained(models_dir)
        assert checkpoint_dir is not None, "Failed to download pretrained model from GitHub"
        assert (checkpoint_dir / "model.pt").exists()
        assert (checkpoint_dir / "config.json").exists()

        import json

        config = json.loads((checkpoint_dir / "config.json").read_text())
        assert config["n_params"] > 100
        print("\n  Downloaded pretrained model:")
        print(f"    Params:     {config['n_params']}")
        print(f"    Trained on: {config.get('n_training_samples', '?')} samples")

        # ──────────────────────────────────────────────────
        # Step 2: Capture target audio from Surge XT
        # ──────────────────────────────────────────────────
        from synth2surge.audio.engine import PluginHost
        from synth2surge.config import MidiProbeConfig

        host = PluginHost(SURGE_VST3)
        midi_config = MidiProbeConfig(sustain_seconds=2.0, release_seconds=1.0)
        target_audio = host.render_midi_mono(midi_config=midi_config)

        rms = float(np.sqrt(np.mean(target_audio**2)))
        assert rms > 0.01, f"Target audio is silent (RMS={rms:.6f})"

        # Save to WAV (as the README workflow would)
        target_path = workspace / "target_audio.wav"
        sf.write(str(target_path), target_audio, host.sample_rate)
        assert target_path.exists()
        print(f"    Target audio RMS: {rms:.4f}")

        # ──────────────────────────────────────────────────
        # Step 3: Warm-start prediction from pretrained model
        # ──────────────────────────────────────────────────
        from synth2surge.loss.features import extract_features
        from synth2surge.ml.warm_start import WarmStarter

        features = extract_features(target_audio, sr=host.sample_rate)

        # WarmStarter should find the pretrained model (no experience store)
        starter = WarmStarter(
            store_path=workspace / "experience.db",  # doesn't exist — that's fine
            models_dir=models_dir,
        )
        x0, sigma0 = starter.predict(features)

        assert x0 is not None, "Pretrained model should produce a prediction"
        assert len(x0) > 100, f"Expected 100+ param predictions, got {len(x0)}"
        assert sigma0 in (0.15, 0.3), f"Expected high/medium confidence, got sigma={sigma0}"
        assert all(0.0 <= v <= 1.0 for v in x0.values())
        print(f"    Warm-start: {len(x0)} params, sigma={sigma0}")

        # ──────────────────────────────────────────────────
        # Step 4: Optimize with warm-start
        # ──────────────────────────────────────────────────
        from synth2surge.config import OptimizationConfig
        from synth2surge.optimizer.loop import optimize

        # Apply warm-start params
        host.set_raw_values(x0)
        host.reset()

        opt_config = OptimizationConfig(
            n_trials_tier1=20, n_trials_tier2=0, n_trials_tier3=0,
        )
        output_dir = workspace / "output"

        result = optimize(
            target_audio=target_audio,
            surge_host=host,
            config=opt_config,
            midi_config=midi_config,
            stages=[1],
            output_dir=output_dir,
        )

        # ──────────────────────────────────────────────────
        # Step 5: Verify outputs
        # ──────────────────────────────────────────────────
        assert result.best_loss < float("inf"), "Optimization should produce a finite loss"
        assert result.best_patch_path.exists(), "best_patch.bin should exist"
        assert result.best_audio_path.exists(), "best_audio.wav should exist"
        assert result.total_trials == 20

        # The .fxp should also exist (DAW-loadable preset)
        assert result.fxp_path is not None
        assert result.fxp_path.exists(), "best_patch.fxp should exist"

        # Best audio should be non-silent
        best_audio, _ = sf.read(str(result.best_audio_path), dtype="float32")
        best_rms = float(np.sqrt(np.mean(best_audio**2)))
        assert best_rms > 0.001, f"Best audio is silent (RMS={best_rms:.6f})"

        print("\n  Optimization result:")
        print(f"    Best loss:    {result.best_loss:.6f}")
        print(f"    Trials:       {result.total_trials}")
        print(f"    Patch:        {result.fxp_path}")
        print(f"    Audio RMS:    {best_rms:.4f}")
        print("\n  README workflow complete!")
