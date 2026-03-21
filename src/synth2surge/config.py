"""Centralized configuration for Synth2Surge."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class ProbeType(Enum):
    """Type of MIDI probe for multi-probe system."""

    SUSTAINED = "sustained"
    VELOCITY = "velocity"
    STACCATO = "staccato"
    CHORD = "chord"
    SEQUENCE = "sequence"
    BASS_LINE = "bass_line"
    MELODY = "melody"
    CHORD_PROGRESSION = "chord_progression"
    COMPING = "comping"
    PERCUSSIVE = "percussive"
    LEGATO = "legato"
    INTERVAL_JUMP = "interval_jump"
    SUB_BASS = "sub_bass"


@dataclass
class ProbeDefinition:
    """Definition of a single probe within a multi-probe configuration."""

    probe_type: ProbeType
    notes: list[int]
    velocities: list[int]
    sustain_seconds: float
    release_seconds: float
    weight: float
    gap_seconds: float = 0.3


@dataclass
class MultiProbeConfig:
    """Configuration for multi-probe MIDI rendering."""

    mode: Literal["single", "thorough", "full"]
    probes: list[ProbeDefinition] = field(default_factory=list)

    @classmethod
    def single(cls) -> MultiProbeConfig:
        """Legacy single-probe mode: one C4 note, velocity 100, 3s sustain + 1s release."""
        return cls(
            mode="single",
            probes=[
                ProbeDefinition(
                    probe_type=ProbeType.SUSTAINED,
                    notes=[60],
                    velocities=[100],
                    sustain_seconds=3.0,
                    release_seconds=1.0,
                    weight=1.0,
                    gap_seconds=0.0,
                ),
            ],
        )

    @classmethod
    def thorough(cls) -> MultiProbeConfig:
        """6 synth-response probes testing pitch, velocity, articulation, and polyphony."""
        return cls(
            mode="thorough",
            probes=[
                ProbeDefinition(
                    probe_type=ProbeType.SUSTAINED,
                    notes=[36],
                    velocities=[100],
                    sustain_seconds=2.0,
                    release_seconds=0.8,
                    weight=0.20,
                ),
                ProbeDefinition(
                    probe_type=ProbeType.SUSTAINED,
                    notes=[60],
                    velocities=[100],
                    sustain_seconds=2.0,
                    release_seconds=0.8,
                    weight=0.30,
                ),
                ProbeDefinition(
                    probe_type=ProbeType.SUSTAINED,
                    notes=[84],
                    velocities=[100],
                    sustain_seconds=1.5,
                    release_seconds=0.5,
                    weight=0.15,
                ),
                ProbeDefinition(
                    probe_type=ProbeType.VELOCITY,
                    notes=[60],
                    velocities=[30, 80, 127],
                    sustain_seconds=1.5,
                    release_seconds=0.5,
                    weight=0.15,
                ),
                ProbeDefinition(
                    probe_type=ProbeType.STACCATO,
                    notes=[60],
                    velocities=[100],
                    sustain_seconds=0.1,
                    release_seconds=0.3,
                    weight=0.10,
                ),
                ProbeDefinition(
                    probe_type=ProbeType.CHORD,
                    notes=[60, 64, 67],
                    velocities=[90],
                    sustain_seconds=1.5,
                    release_seconds=0.8,
                    weight=0.10,
                ),
            ],
        )

    @classmethod
    def full(cls) -> MultiProbeConfig:
        """14 probes: 6 synth-response + 8 musical context probes."""
        thorough = cls.thorough()
        # Re-weight thorough probes to sum to 0.60
        thorough_total = sum(p.weight for p in thorough.probes)
        for p in thorough.probes:
            p.weight = p.weight / thorough_total * 0.60

        musical_probes = [
            ProbeDefinition(
                probe_type=ProbeType.SUB_BASS,
                notes=[24],
                velocities=[100],
                sustain_seconds=3.0,
                release_seconds=1.0,
                weight=0.06,
            ),
            ProbeDefinition(
                probe_type=ProbeType.BASS_LINE,
                notes=[28, 31, 33, 35],
                velocities=[90],
                sustain_seconds=0.4,
                release_seconds=0.0,
                weight=0.06,
            ),
            ProbeDefinition(
                probe_type=ProbeType.MELODY,
                notes=[60, 62, 64, 65, 67, 69, 71, 72],
                velocities=[80],
                sustain_seconds=0.2,
                release_seconds=0.0,
                weight=0.06,
            ),
            ProbeDefinition(
                probe_type=ProbeType.CHORD_PROGRESSION,
                notes=[60, 64, 67, 65, 69, 72, 67, 71, 74, 60, 64, 67],
                velocities=[85],
                sustain_seconds=1.0,
                release_seconds=0.3,
                weight=0.06,
            ),
            ProbeDefinition(
                probe_type=ProbeType.COMPING,
                notes=[60, 64, 67],
                velocities=[70],
                sustain_seconds=0.1,
                release_seconds=0.3,
                weight=0.04,
            ),
            ProbeDefinition(
                probe_type=ProbeType.PERCUSSIVE,
                notes=[60],
                velocities=[127],
                sustain_seconds=0.05,
                release_seconds=1.0,
                weight=0.04,
            ),
            ProbeDefinition(
                probe_type=ProbeType.LEGATO,
                notes=[60, 62, 64, 65, 67],
                velocities=[80],
                sustain_seconds=0.4,
                release_seconds=0.1,
                weight=0.04,
            ),
            ProbeDefinition(
                probe_type=ProbeType.INTERVAL_JUMP,
                notes=[36, 60, 84, 60],
                velocities=[100],
                sustain_seconds=0.8,
                release_seconds=0.2,
                weight=0.04,
            ),
        ]

        return cls(
            mode="full",
            probes=thorough.probes + musical_probes,
        )


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


class MLConfig(BaseSettings):
    """Machine learning configuration."""

    store_path: Path = Path("workspace/experience.db")
    models_dir: Path = Path("workspace/models")
    retrain_interval: int = 25  # retrain after every N new runs
    min_runs_for_mlp: int = 50  # minimum runs before MLP warm-start
    min_runs_for_cnn: int = 200  # minimum runs before CNN warm-start
    confidence_threshold: float = 0.3  # below this, fall back to pure CMA-ES
    warm_start_enabled: bool = True


class EnrichedLossConfig(BaseSettings):
    """Enriched loss function weights (Phase 3)."""

    w_stft: float = 0.6
    w_mfcc: float = 0.15
    w_envelope: float = 0.1
    w_centroid: float = 0.1
    w_flux: float = 0.05


class LearnedLossConfig(BaseSettings):
    """Learned audio encoder configuration (Phase 3)."""

    alpha: float = 0.0  # blend weight for learned similarity (0 = pure MR-STFT)
    max_alpha: float = 0.3
    encoder_checkpoint: Path | None = None
    min_runs_for_encoder: int = 200


class AppConfig(BaseSettings):
    """Top-level application configuration."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    midi_probe: MidiProbeConfig = Field(default_factory=MidiProbeConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    surge: SurgeConfig = Field(default_factory=SurgeConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    enriched_loss: EnrichedLossConfig = Field(default_factory=EnrichedLossConfig)
    learned_loss: LearnedLossConfig = Field(default_factory=LearnedLossConfig)
    workspace_dir: Path = Path("workspace")

    def ensure_workspace(self) -> Path:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir
