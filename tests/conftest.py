"""Shared test fixtures and configuration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PATCHES_DIR = FIXTURES_DIR / "patches"
AUDIO_DIR = FIXTURES_DIR / "audio"

SURGE_VST3_PATH = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")
SURGE_FACTORY_DIR = Path("/Library/Application Support/Surge XT/patches_factory")


def surge_xt_available() -> bool:
    """Check if Surge XT VST3 is installed."""
    return SURGE_VST3_PATH.exists()


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    # Markers are registered in pyproject.toml, but this ensures runtime access.


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip tests that require unavailable resources."""
    if not surge_xt_available():
        skip_surge = pytest.mark.skip(reason="Surge XT VST3 not installed")
        for item in items:
            if "requires_surge" in item.keywords:
                item.add_marker(skip_surge)


@pytest.fixture
def sample_rate() -> int:
    return 44100


@pytest.fixture
def sine_440(sample_rate: int) -> np.ndarray:
    """Generate a 2-second 440Hz sine wave."""
    from tests.factories import make_sine_wave

    return make_sine_wave(440.0, duration=2.0, sr=sample_rate)


@pytest.fixture
def sine_441(sample_rate: int) -> np.ndarray:
    """Generate a 2-second 441Hz sine wave."""
    from tests.factories import make_sine_wave

    return make_sine_wave(441.0, duration=2.0, sr=sample_rate)


@pytest.fixture
def white_noise(sample_rate: int) -> np.ndarray:
    """Generate 2 seconds of white noise."""
    from tests.factories import make_noise

    return make_noise(duration=2.0, sr=sample_rate)


@pytest.fixture
def silence(sample_rate: int) -> np.ndarray:
    """Generate 2 seconds of silence."""
    return np.zeros(2 * sample_rate, dtype=np.float32)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    """Create a temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws
