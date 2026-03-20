"""Centralized configuration for Synth2Surge."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AudioConfig(BaseSettings):
    """Audio rendering configuration."""

    sample_rate: int = 44100
    bit_depth: int = 16


class MidiProbeConfig(BaseSettings):
    """MIDI probe configuration for consistent audio comparison."""

    note: int = 60  # C4
    velocity: int = 100
    sustain_seconds: float = 3.0
    release_seconds: float = 1.0

    @property
    def total_duration(self) -> float:
        return self.sustain_seconds + self.release_seconds


class LossConfig(BaseSettings):
    """MR-STFT loss function configuration."""

    fft_sizes: list[int] = Field(default_factory=lambda: [2048, 1024, 512])
    hop_divisor: int = 4
    magnitude_alpha: float = 1.0
    log_epsilon: float = 1e-7


class SurgeConfig(BaseSettings):
    """Surge XT paths and configuration."""

    vst3_path: Path = Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3")
    factory_patches_dir: Path = Path("/Library/Application Support/Surge XT/patches_factory")
    user_patches_dir: Path = Path.home() / "Documents" / "Surge XT"


class OptimizationConfig(BaseSettings):
    """Optimization loop configuration."""

    n_trials_tier1: int = 300
    n_trials_tier2: int = 300
    n_trials_tier3: int = 200
    n_seeds: int = 3
    storage_path: Path = Path("workspace/optuna.db")


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    midi_probe: MidiProbeConfig = Field(default_factory=MidiProbeConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    surge: SurgeConfig = Field(default_factory=SurgeConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    workspace_dir: Path = Path("workspace")

    def ensure_workspace(self) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir
