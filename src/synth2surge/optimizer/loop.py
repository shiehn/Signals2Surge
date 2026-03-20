"""Core CMA-ES optimization loop.

Iteratively mutates a Surge XT plugin's parameters to minimize MR-STFT loss
against a target audio signal, using Optuna's CMA-ES sampler.

Works directly with pedalboard's raw_value API (all params normalized to [0,1]),
bypassing XML patch manipulation entirely during optimization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler

from synth2surge.audio.engine import PluginHost
from synth2surge.config import MidiProbeConfig, OptimizationConfig
from synth2surge.loss.mr_stft import mr_stft_loss
from synth2surge.types import OptimizationProgress, OptimizationResult

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Default tier classification for pedalboard parameter names
_TIER1_PATTERNS = [
    "_osc_1_type", "_osc_2_type", "_osc_3_type",
    "_filter_1_type", "_filter_2_type",
    "_filter_1_cutoff", "_filter_2_cutoff",
    "_filter_1_resonance", "_filter_2_resonance",
    "_amp_eg_attack", "_amp_eg_decay", "_amp_eg_sustain", "_amp_eg_release",
    "_filter_eg_attack", "_filter_eg_decay", "_filter_eg_sustain", "_filter_eg_release",
    "_osc_1_level", "_osc_2_level", "_osc_3_level",
    "_noise_level", "_ring_mod_",
    "_fm_depth", "_fm_routing",
    "scene_a_volume", "scene_b_volume",
    "volume",
]

_TIER2_PATTERNS = [
    "_osc_1_pitch", "_osc_2_pitch", "_osc_3_pitch",
    "_osc_1_octave", "_osc_2_octave", "_osc_3_octave",
    "_osc_1_param_", "_osc_2_param_", "_osc_3_param_",
    "_filter_1_subtype", "_filter_2_subtype",
    "_filter_balance", "_filter_configuration",
    "_feedback", "_highpass",
    "_lfo_", "_drift",
    "_width", "_pan",
]


def classify_parameter_tier(name: str) -> int:
    """Classify a pedalboard parameter name into optimization tier (1, 2, or 3)."""
    for pattern in _TIER1_PATTERNS:
        if pattern in name:
            return 1
    for pattern in _TIER2_PATTERNS:
        if pattern in name:
            return 2
    return 3


def get_optimizable_params(
    host: PluginHost,
    scene: str = "a",
) -> dict[int, list[str]]:
    """Get parameter names organized by optimization tier.

    Only includes parameters for the specified scene plus globals.
    Excludes scene B if scene is 'a', and vice versa.
    """
    all_names = host.parameter_names()
    exclude_prefix = "b_" if scene == "a" else "a_"

    tiers: dict[int, list[str]] = {1: [], 2: [], 3: []}
    for name in all_names:
        # Skip the other scene's parameters
        if name.startswith(exclude_prefix):
            continue
        tier = classify_parameter_tier(name)
        tiers[tier].append(name)

    return tiers


def optimize(
    target_audio: np.ndarray,
    surge_host: PluginHost,
    config: OptimizationConfig | None = None,
    midi_config: MidiProbeConfig | None = None,
    progress_callback: Callable[[OptimizationProgress], None] | None = None,
    stages: list[int] | None = None,
    scene: str = "a",
    output_dir: Path | None = None,
) -> OptimizationResult:
    """Run multi-stage CMA-ES optimization to match target audio.

    Works directly with pedalboard's raw_value API — all parameter values
    are in the [0, 1] range, mapped internally by the plugin.

    Args:
        target_audio: Reference audio signal (1-D float32 array).
        surge_host: Loaded PluginHost with Surge XT.
        config: Optimization configuration.
        midi_config: MIDI probe configuration.
        progress_callback: Called after each trial with progress info.
        stages: Which stages to run (default [1, 2, 3]).
        scene: Which scene to optimize ('a' or 'b').
        output_dir: Directory to save results (default: workspace/).

    Returns:
        OptimizationResult with best patch path, loss, and metadata.
    """
    if config is None:
        config = OptimizationConfig()
    if stages is None:
        stages = [1, 2, 3]
    if output_dir is None:
        output_dir = config.storage_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parameter tiers
    param_tiers = get_optimizable_params(surge_host, scene=scene)

    stage_trials = {
        1: config.n_trials_tier1,
        2: config.n_trials_tier2,
        3: config.n_trials_tier3,
    }



    best_loss = float("inf")
    best_raw_values: dict[str, float] = {}
    all_losses: list[float] = []
    total_trials = sum(stage_trials.get(s, 0) for s in stages)
    completed_trials = 0
    frozen_values: dict[str, float] = {}

    for stage in stages:
        n_trials = stage_trials.get(stage, 100)
        active_param_names = param_tiers.get(stage, [])

        if not active_param_names:
            logger.info(f"Stage {stage}: no active parameters, skipping")
            continue

        logger.info(
            f"Stage {stage}: optimizing {len(active_param_names)} parameters "
            f"for {n_trials} trials"
        )

        sampler = CmaEsSampler(
            seed=42 + stage,
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_loss, best_raw_values, completed_trials

            # Suggest values for active parameters
            suggestions = {}
            for name in active_param_names:
                suggestions[name] = trial.suggest_float(name, 0.0, 1.0)

            # Merge with frozen values from previous stages
            all_values = {**frozen_values, **suggestions}

            # Apply to plugin
            surge_host.set_raw_values(all_values)
            surge_host.reset()

            # Render
            candidate_audio = surge_host.render_midi_mono(midi_config=midi_config)

            # Compute loss
            loss = mr_stft_loss(target_audio, candidate_audio)

            # Handle inf/nan
            if not np.isfinite(loss):
                loss = 1e6

            # Track best
            if loss < best_loss:
                best_loss = loss
                best_raw_values = dict(all_values)

            all_losses.append(loss)
            completed_trials += 1

            if progress_callback is not None:
                progress_callback(
                    OptimizationProgress(
                        current_trial=completed_trials,
                        total_trials=total_trials,
                        best_loss=best_loss,
                        current_loss=loss,
                        stage=stage,
                        loss_history=list(all_losses),
                    )
                )

            return loss

        study.optimize(objective, n_trials=n_trials)

        # Freeze best values from this stage for subsequent stages
        if study.best_trial is not None:
            for name in active_param_names:
                if name in study.best_trial.params:
                    frozen_values[name] = study.best_trial.params[name]

    # Apply best values and save result
    if best_raw_values:
        surge_host.set_raw_values(best_raw_values)

    # Save best state
    best_state = surge_host.get_state()
    best_patch_path = output_dir / "best_patch.bin"
    best_patch_path.write_bytes(best_state)

    # Render and save best audio
    surge_host.reset()
    best_audio = surge_host.render_midi_mono(midi_config=midi_config)
    best_audio_path = output_dir / "best_audio.wav"
    import soundfile as sf

    sf.write(str(best_audio_path), best_audio, surge_host.sample_rate)

    return OptimizationResult(
        best_patch_path=best_patch_path,
        best_loss=best_loss,
        best_audio_path=best_audio_path,
        total_trials=completed_trials,
        stages_completed=len(stages),
    )
