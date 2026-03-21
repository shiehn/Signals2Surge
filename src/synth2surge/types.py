"""Shared data types used across modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CaptureResult:
    """Result of capturing a source plugin preset."""

    audio_path: Path
    state_path: Path
    parameters: dict[str, float]
    audio: np.ndarray | None = field(default=None, repr=False)
    audio_segments: list[np.ndarray] | None = field(default=None, repr=False)


@dataclass
class RenderResult:
    """Result of rendering audio through a plugin."""

    audio: np.ndarray
    sample_rate: int
    duration: float


@dataclass
class PatchMetadata:
    """Metadata from a Surge XT patch."""

    name: str = ""
    category: str = ""
    author: str = ""
    comment: str = ""


@dataclass
class OptimizationProgress:
    """Progress snapshot of an optimization run."""

    current_trial: int
    total_trials: int
    best_loss: float
    current_loss: float
    stage: int
    loss_history: list[float] = field(default_factory=list)


@dataclass
class MLPrediction:
    """Result of an ML parameter prediction."""

    predicted_params: dict[str, float]
    confidence: float
    sigma0: float
    model_version: str | None = None


@dataclass
class OptimizationResult:
    """Final result of an optimization run."""

    best_patch_path: Path
    best_loss: float
    best_audio_path: Path
    total_trials: int
    stages_completed: int
    fxp_path: Path | None = None
    warm_started: bool = False
    model_version: str | None = None
