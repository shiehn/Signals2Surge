"""End-to-end pipeline test: capture -> optimize -> verify."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synth2surge.audio.engine import PluginHost
from synth2surge.capture.workflow import capture_headless
from synth2surge.config import MidiProbeConfig, OptimizationConfig
from synth2surge.optimizer.loop import optimize

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = [pytest.mark.requires_surge, pytest.mark.slow, pytest.mark.e2e]


class TestFullPipeline:
    def test_capture_then_optimize(self, tmp_path: Path):
        """Full pipeline: capture from Surge XT, optimize back to Surge XT."""
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")

        midi_config = MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)

        # Step 1: Capture target audio from Surge XT
        capture_dir = tmp_path / "capture"
        capture_result = capture_headless(
            plugin_path=SURGE_VST3,
            output_dir=capture_dir,
            midi_config=midi_config,
        )

        assert capture_result.audio_path.exists()
        target_audio, sr = sf.read(str(capture_result.audio_path), dtype="float32")
        if target_audio.ndim > 1:
            target_audio = np.mean(target_audio, axis=1)

        # Step 2: Optimize a Surge XT patch to match
        host = PluginHost(SURGE_VST3, sample_rate=sr)

        config = OptimizationConfig(
            n_trials_tier1=15,
            n_trials_tier2=0,
            n_trials_tier3=0,
        )

        optimize_dir = tmp_path / "optimize"
        result = optimize(
            target_audio=target_audio,
            surge_host=host,
            config=config,
            midi_config=midi_config,
            stages=[1],
            output_dir=optimize_dir,
        )

        # Step 3: Verify outputs exist
        assert result.best_patch_path.exists()
        assert result.best_audio_path.exists()
        assert result.best_loss < float("inf")
        assert result.total_trials == 15

        # Step 4: Verify the best audio file is valid
        best_audio, _ = sf.read(str(result.best_audio_path), dtype="float32")
        assert len(best_audio) > 0
