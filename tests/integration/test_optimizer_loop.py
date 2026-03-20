"""Integration tests for the optimization loop — requires Surge XT."""

from pathlib import Path

import pytest

from synth2surge.audio.engine import PluginHost
from synth2surge.config import MidiProbeConfig, OptimizationConfig
from synth2surge.optimizer.loop import get_optimizable_params, optimize

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = [pytest.mark.requires_surge, pytest.mark.slow]


@pytest.fixture(scope="module")
def surge_host() -> PluginHost:
    if not SURGE_VST3.exists():
        pytest.skip("Surge XT not installed")
    return PluginHost(SURGE_VST3)


class TestGetOptimizableParams:
    def test_returns_three_tiers(self, surge_host: PluginHost):
        tiers = get_optimizable_params(surge_host, scene="a")
        assert 1 in tiers
        assert 2 in tiers
        assert 3 in tiers

    def test_tier1_manageable_size(self, surge_host: PluginHost):
        tiers = get_optimizable_params(surge_host, scene="a")
        assert 5 <= len(tiers[1]) <= 100

    def test_excludes_scene_b(self, surge_host: PluginHost):
        tiers = get_optimizable_params(surge_host, scene="a")
        all_names = tiers[1] + tiers[2] + tiers[3]
        assert not any(n.startswith("b_") for n in all_names)

    def test_total_params(self, surge_host: PluginHost):
        tiers = get_optimizable_params(surge_host, scene="a")
        total = sum(len(v) for v in tiers.values())
        assert total > 100  # Should have substantial params


class TestOptimize:
    def test_short_optimization_reduces_loss(self, surge_host: PluginHost, tmp_path: Path):
        """Run a very short optimization and verify loss decreases."""
        # Render target audio from current state
        surge_host.reset()
        target_audio = surge_host.render_midi_mono(
            midi_config=MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)
        )

        # Slightly perturb the plugin so optimizer has something to find
        raw = surge_host.get_raw_values()
        perturbed = {k: min(1.0, max(0.0, v + 0.05)) for k, v in raw.items()}
        surge_host.set_raw_values(perturbed)

        config = OptimizationConfig(
            n_trials_tier1=10,
            n_trials_tier2=0,
            n_trials_tier3=0,
        )

        losses = []

        def track_progress(p):
            losses.append(p.current_loss)

        result = optimize(
            target_audio=target_audio,
            surge_host=surge_host,
            config=config,
            midi_config=MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5),
            progress_callback=track_progress,
            stages=[1],
            output_dir=tmp_path,
        )

        assert result.best_patch_path.exists()
        assert result.best_audio_path.exists()
        assert result.best_loss < float("inf")
        assert result.total_trials == 10
        assert len(losses) == 10

    def test_output_files_created(self, surge_host: PluginHost, tmp_path: Path):
        """Verify output files are created."""
        surge_host.reset()
        target = surge_host.render_midi_mono(
            midi_config=MidiProbeConfig(sustain_seconds=0.5, release_seconds=0.3)
        )

        config = OptimizationConfig(
            n_trials_tier1=5,
            n_trials_tier2=0,
            n_trials_tier3=0,
        )

        result = optimize(
            target_audio=target,
            surge_host=surge_host,
            config=config,
            midi_config=MidiProbeConfig(sustain_seconds=0.5, release_seconds=0.3),
            stages=[1],
            output_dir=tmp_path,
        )

        assert result.best_patch_path.exists()
        assert result.best_patch_path.stat().st_size > 0
        assert result.best_audio_path.exists()
        assert result.best_audio_path.stat().st_size > 0
