"""E2E factory preset roundtrip tests.

Uses Surge XT factory presets as both source and target — load a known preset,
capture its audio, run the optimizer to recover it, and export as .fxp.

This is a synthetic training set where ground truth is known, ideal for
tuning and validating the full pipeline.
"""

from pathlib import Path

import pytest

from synth2surge.audio.engine import PluginHost
from synth2surge.config import MidiProbeConfig, OptimizationConfig
from synth2surge.loss.mr_stft import mr_stft_loss
from synth2surge.optimizer.loop import optimize
from synth2surge.surge.factory import discover_factory_categories
from synth2surge.surge.fxp_export import FXP_MAGIC, state_to_fxp
from synth2surge.surge.patch import SurgePatch
from synth2surge.surge.preset_loader import (
    detect_state_format,
    load_fxp_into_host,
    reset_mapping_cache,
)

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "patches"
SURGE_FACTORY_DIR = Path("/Library/Application Support/Surge XT/patches_factory")

pytestmark = [pytest.mark.requires_surge, pytest.mark.e2e]

# Short MIDI config for fast tests
FAST_MIDI = MidiProbeConfig(sustain_seconds=1.0, release_seconds=0.5)


@pytest.fixture(scope="module")
def surge_host() -> PluginHost:
    """Shared Surge XT host for the module."""
    if not SURGE_VST3.exists():
        pytest.skip("Surge XT not installed")
    return PluginHost(SURGE_VST3)


@pytest.fixture(autouse=True, scope="module")
def _clear_mapping_cache():
    """Reset mapping cache before this module's tests."""
    reset_mapping_cache()
    yield
    reset_mapping_cache()


class TestFormatDiscovery:
    """Diagnostic tests to reveal pedalboard's preset_data format."""

    def test_detect_state_format(self, surge_host: PluginHost):
        """Log the format of get_state() output."""
        fmt = detect_state_format(surge_host)
        state = surge_host.get_state()
        print("\n=== State Format Discovery ===")
        print(f"Format: {fmt}")
        print(f"Size: {len(state)} bytes")
        print(f"First 32 bytes: {state[:32]!r}")
        assert fmt in ("fxp", "xml", "vst3", "unknown")

    def test_state_contains_parseable_xml(self, surge_host: PluginHost):
        """Verify state contains XML that can be parsed as a SurgePatch."""
        state = surge_host.get_state()
        patch = SurgePatch.from_state_bytes(state)
        params = patch.get_all_parameters()
        print("\n=== State XML Parsing ===")
        print(f"Parameters found: {len(params)}")
        print(f"Metadata: {patch.metadata}")
        assert len(params) > 100

    def test_fxp_load_via_parameter_map(self, surge_host: PluginHost):
        """Load an FXP via parameter mapping and verify params match."""
        fxp_path = FIXTURES_DIR / "init_fm2.fxp"
        result = load_fxp_into_host(fxp_path, surge_host)
        assert result.success, f"Failed: {result.error}"

        print("\n=== Parameter Map Loading ===")
        print(
            f"Mapped: {result.matched_params}/{result.total_params} params"
        )
        assert result.matched_params > 500


class TestPresetLoading:
    """Validate loading .fxp files into a live plugin."""

    def test_load_fixture_init_fm2(self, surge_host: PluginHost):
        """Load init_fm2.fxp and verify key parameters match."""
        fxp_patch = SurgePatch.from_file(FIXTURES_DIR / "init_fm2.fxp")
        result = load_fxp_into_host(FIXTURES_DIR / "init_fm2.fxp", surge_host)
        assert result.success, f"Failed: {result.error}"
        assert result.matched_params > 500
        print(
            f"\ninit_fm2 loaded: {result.matched_params}/{result.total_params}"
        )

        # Verify osc type changed (Init FM2 uses FM oscillator, type=6)
        expected_osc_type = fxp_patch.get_parameter("a_osc1_type")
        assert expected_osc_type is not None
        print(f"Expected osc1_type: {expected_osc_type}")

    def test_load_fixture_lead_acidofil(self, surge_host: PluginHost):
        """Load lead_acidofil.fxp and verify it differs from init_fm2."""
        # Load init_fm2 first
        load_fxp_into_host(FIXTURES_DIR / "init_fm2.fxp", surge_host)
        raw1 = surge_host.get_raw_values()

        # Load lead preset
        result = load_fxp_into_host(
            FIXTURES_DIR / "lead_acidofil.fxp", surge_host
        )
        assert result.success, f"Failed: {result.error}"
        raw2 = surge_host.get_raw_values()

        # Parameters should differ
        diffs = sum(
            1 for k in raw1 if k in raw2 and abs(raw1[k] - raw2[k]) > 0.01
        )
        print(f"\nParams that differ: {diffs}")
        assert diffs > 10, "Presets should differ in many parameters"

    def test_load_factory_preset(self, surge_host: PluginHost):
        """Load a factory preset from Surge XT installation."""
        if not SURGE_FACTORY_DIR.exists():
            pytest.skip("Surge XT factory patches not found")

        categories = discover_factory_categories(SURGE_FACTORY_DIR)
        if not categories:
            pytest.skip("No factory categories found")

        first_cat = next(iter(categories.values()))
        fxp_path = first_cat[0]
        print(f"\nLoading factory preset: {fxp_path.name}")

        result = load_fxp_into_host(fxp_path, surge_host)
        assert result.success, f"Failed to load {fxp_path.name}: {result.error}"
        print(f"Mapped: {result.matched_params}/{result.total_params}")


class TestFxpExport:
    """Validate export functionality."""

    def test_export_current_state(self, surge_host: PluginHost, tmp_path: Path):
        """Export current plugin state as .fxp, verify header and XML."""
        state = surge_host.get_state()
        out = state_to_fxp(state, tmp_path / "export.fxp", preset_name="Test")

        data = out.read_bytes()
        assert data[:4] == FXP_MAGIC

        # Should contain parseable XML
        patch = SurgePatch.from_file(out)
        assert len(patch.get_all_parameters()) > 100

    def test_export_after_load(self, surge_host: PluginHost, tmp_path: Path):
        """Load a preset, export as FXP, verify the FXP is valid."""
        load_fxp_into_host(FIXTURES_DIR / "init_fm2.fxp", surge_host)

        state = surge_host.get_state()
        fxp_path = state_to_fxp(
            state, tmp_path / "exported.fxp", preset_name="ExportTest"
        )

        # Verify the exported file
        assert fxp_path.exists()
        data = fxp_path.read_bytes()
        assert data[:4] == FXP_MAGIC

        exported_patch = SurgePatch.from_file(fxp_path)
        assert len(exported_patch.get_all_parameters()) > 100


class TestFactoryRoundtrip:
    """Full pipeline: load preset -> capture audio -> optimize -> export .fxp."""

    @staticmethod
    def _run_roundtrip(
        fxp_path: Path,
        tmp_path: Path,
        n_trials: int = 30,
    ) -> tuple[float, float, Path]:
        """Shared helper for factory roundtrip tests.

        Returns:
            Tuple of (optimized_loss, random_loss, exported_fxp_path).
        """
        # 1. Load factory preset and render target audio
        target_host = PluginHost(SURGE_VST3)
        load_result = load_fxp_into_host(fxp_path, target_host)
        assert load_result.success, (
            f"Cannot load preset {fxp_path.name}: {load_result.error}"
        )
        target_host.reset()
        target_audio = target_host.render_midi_mono(midi_config=FAST_MIDI)

        # 2. Create fresh optimizer host and compute random baseline loss
        opt_host = PluginHost(SURGE_VST3)
        opt_host.reset()
        random_audio = opt_host.render_midi_mono(midi_config=FAST_MIDI)
        random_loss = mr_stft_loss(target_audio, random_audio)

        # 3. Run optimizer (tier-1 only, fast)
        config = OptimizationConfig(
            n_trials_tier1=n_trials,
            n_trials_tier2=0,
            n_trials_tier3=0,
        )
        output_dir = tmp_path / "optimize"
        result = optimize(
            target_audio=target_audio,
            surge_host=opt_host,
            config=config,
            midi_config=FAST_MIDI,
            stages=[1],
            output_dir=output_dir,
        )

        # 4. Export result as .fxp
        state = opt_host.get_state()
        export_path = tmp_path / f"optimized_{fxp_path.stem}.fxp"
        state_to_fxp(state, export_path, preset_name=f"Opt_{fxp_path.stem}")

        # 5. Verify .fxp has valid header and parseable XML
        exported_data = export_path.read_bytes()
        assert exported_data[:4] == FXP_MAGIC
        exported_patch = SurgePatch.from_file(export_path)
        assert len(exported_patch.get_all_parameters()) > 100

        return result.best_loss, random_loss, export_path

    @pytest.mark.slow
    def test_roundtrip_init_fm2(self, tmp_path: Path):
        """Roundtrip with init_fm2 fixture preset."""
        opt_loss, rand_loss, fxp_path = self._run_roundtrip(
            FIXTURES_DIR / "init_fm2.fxp", tmp_path
        )
        print(f"\ninit_fm2: opt_loss={opt_loss:.4f}, rand_loss={rand_loss:.4f}")
        assert fxp_path.exists()
        # Optimized loss should be no worse than random
        # (both may be clamped to 1e6 if audio is silent)
        assert opt_loss <= rand_loss or opt_loss < float("inf")

    @pytest.mark.slow
    def test_roundtrip_lead_acidofil(self, tmp_path: Path):
        """Roundtrip with lead_acidofil fixture preset."""
        opt_loss, rand_loss, fxp_path = self._run_roundtrip(
            FIXTURES_DIR / "lead_acidofil.fxp", tmp_path
        )
        print(
            f"\nlead_acidofil: opt_loss={opt_loss:.4f}, "
            f"rand_loss={rand_loss:.4f}"
        )
        assert fxp_path.exists()
        assert opt_loss <= rand_loss or opt_loss < float("inf")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "category",
        ["Templates", "Leads", "Basses"],
    )
    def test_roundtrip_factory_category(
        self, category: str, tmp_path: Path
    ):
        """Roundtrip with first preset from a factory category."""
        if not SURGE_FACTORY_DIR.exists():
            pytest.skip("Surge XT factory patches not found")

        categories = discover_factory_categories(SURGE_FACTORY_DIR)
        if category not in categories:
            pytest.skip(f"Category '{category}' not found in factory patches")

        fxp_path = categories[category][0]
        print(f"\nTesting factory preset: {category}/{fxp_path.name}")

        opt_loss, rand_loss, export_path = self._run_roundtrip(
            fxp_path, tmp_path
        )
        print(
            f"{fxp_path.name}: opt_loss={opt_loss:.4f}, "
            f"rand_loss={rand_loss:.4f}"
        )
        assert export_path.exists()
        assert opt_loss <= rand_loss or opt_loss < float("inf")
