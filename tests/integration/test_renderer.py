"""Integration tests for the capture workflow — requires Surge XT."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synth2surge.capture.workflow import capture_from_state_file, capture_headless

SURGE_VST3 = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")

pytestmark = pytest.mark.requires_surge


class TestCaptureHeadless:
    def test_capture_produces_files(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        result = capture_headless(SURGE_VST3, tmp_path)
        assert result.audio_path.exists()
        assert result.state_path.exists()

    def test_audio_file_valid_wav(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        result = capture_headless(SURGE_VST3, tmp_path)
        data, sr = sf.read(str(result.audio_path))
        assert sr == 44100
        assert len(data) > 0

    def test_state_file_nonempty(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        result = capture_headless(SURGE_VST3, tmp_path)
        assert result.state_path.stat().st_size > 0

    def test_parameters_extracted(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        result = capture_headless(SURGE_VST3, tmp_path)
        assert isinstance(result.parameters, dict)

    def test_audio_array_attached(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        result = capture_headless(SURGE_VST3, tmp_path)
        assert result.audio is not None
        assert result.audio.dtype == np.float32


class TestCaptureFromStateFile:
    def test_roundtrip_via_state_file(self, tmp_path: Path):
        if not SURGE_VST3.exists():
            pytest.skip("Surge XT not installed")
        # First capture to get a state file
        result1 = capture_headless(SURGE_VST3, tmp_path / "first")
        # Then reload from that state file
        result2 = capture_from_state_file(
            SURGE_VST3,
            result1.state_path,
            tmp_path / "second",
        )
        assert result2.audio_path.exists()
        assert result2.state_path.exists()
