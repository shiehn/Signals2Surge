"""Test data factories for generating synthetic audio and test fixtures."""

from __future__ import annotations

import numpy as np


def make_sine_wave(
    frequency: float,
    duration: float = 2.0,
    sr: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def make_noise(
    duration: float = 2.0,
    sr: int = 44100,
    amplitude: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate white noise (deterministic with seed)."""
    rng = np.random.default_rng(seed)
    n_samples = int(sr * duration)
    return (amplitude * rng.standard_normal(n_samples)).astype(np.float32)


def make_chirp(
    f_start: float = 200.0,
    f_end: float = 2000.0,
    duration: float = 2.0,
    sr: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a linear frequency chirp (sweep)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t**2 / (2 * duration))
    return (amplitude * np.sin(phase)).astype(np.float32)


def make_silence(duration: float = 2.0, sr: int = 44100) -> np.ndarray:
    """Generate silence."""
    return np.zeros(int(sr * duration), dtype=np.float32)
