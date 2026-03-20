"""E2E test for the CLI workflow."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from synth2surge.cli.main import app

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

runner = CliRunner()

pytestmark = [pytest.mark.requires_surge, pytest.mark.slow, pytest.mark.e2e]


class TestCLIWorkflow:
    def test_capture_then_optimize_via_cli(self, tmp_path: Path):
        """Test the full capture -> optimize workflow via CLI."""
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")

        capture_dir = tmp_path / "capture"

        # Step 1: Capture
        result = runner.invoke(
            app,
            [
                "capture",
                "--plugin", str(SURGE_VST3),
                "--output-dir", str(capture_dir),
                "--no-gui",
                "--duration", "1.5",
            ],
        )
        assert result.exit_code == 0, f"Capture failed: {result.stdout}"
        assert (capture_dir / "target_audio.wav").exists()

        # Step 2: Optimize (very short)
        optimize_dir = tmp_path / "optimize"
        result = runner.invoke(
            app,
            [
                "optimize",
                "--target", str(capture_dir / "target_audio.wav"),
                "--output-dir", str(optimize_dir),
                "--surge-plugin", str(SURGE_VST3),
                "--trials-t1", "5",
                "--trials-t2", "0",
                "--trials-t3", "0",
                "--stages", "1",
            ],
        )
        assert result.exit_code == 0, f"Optimize failed: {result.stdout}"

    def test_inspect_command(self):
        """Test the inspect command on a fixture patch."""
        fixture = Path(__file__).parent.parent / "fixtures" / "patches" / "init_fm2.fxp"
        result = runner.invoke(app, ["inspect", "--patch", str(fixture)])
        assert result.exit_code == 0
        assert "Init FM2" in result.stdout
