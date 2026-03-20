"""Acceptance tests: self-translation (Surge -> Surge) quality validation.

These tests verify that the optimization loop can find a Surge XT patch
that closely matches another Surge XT patch — the gold standard test since
the target is already expressible in the same synth.
"""

from pathlib import Path

import numpy as np
import pytest

from synth2surge.audio.engine import PluginHost
from synth2surge.config import MidiProbeConfig, OptimizationConfig
from synth2surge.loss.mr_stft import mr_stft_loss
from synth2surge.optimizer.loop import optimize

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = [pytest.mark.requires_surge, pytest.mark.slow, pytest.mark.acceptance]


@pytest.fixture(scope="module")
def surge_host() -> PluginHost:
    if not SURGE_VST3.exists():
        pytest.skip("Surge XT not installed")
    return PluginHost(SURGE_VST3)


class TestSelfTranslation:
    def test_optimization_reduces_loss_vs_random(
        self, surge_host: PluginHost, tmp_path: Path
    ):
        """Optimizer should achieve lower loss than a random parameter setting."""
        midi_config = MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)

        # Render target from current state
        surge_host.reset()
        target_audio = surge_host.render_midi_mono(midi_config=midi_config)

        # Compute loss of a random parameterization
        rng = np.random.default_rng(42)
        random_values = {
            name: rng.random()
            for name in surge_host.parameter_names()
            if not name.startswith("b_")
        }
        surge_host.set_raw_values(random_values)
        surge_host.reset()
        random_audio = surge_host.render_midi_mono(midi_config=midi_config)
        random_loss = mr_stft_loss(target_audio, random_audio)

        # Run short optimization
        config = OptimizationConfig(
            n_trials_tier1=20,
            n_trials_tier2=0,
            n_trials_tier3=0,
        )

        result = optimize(
            target_audio=target_audio,
            surge_host=surge_host,
            config=config,
            midi_config=midi_config,
            stages=[1],
            output_dir=tmp_path,
        )

        # Optimized loss should be better than random
        assert result.best_loss < random_loss, (
            f"Optimized loss ({result.best_loss:.4f}) should be lower "
            f"than random loss ({random_loss:.4f})"
        )
