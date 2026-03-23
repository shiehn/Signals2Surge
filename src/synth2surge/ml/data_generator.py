"""Autonomous self-play data generation for ML training.

Generates unlimited labeled (audio_features, parameters) pairs by:
1. Randomizing Surge XT parameters to uniform [0,1]
2. Rendering audio with semi-random MIDI probes
3. Extracting features and storing as ground truth

Can optionally run CMA-ES optimization to generate trial-level data.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Callable

import numpy as np

from synth2surge.audio.standard_probes import render_standard_features
from synth2surge.config import MidiProbeConfig
from synth2surge.loss.features import extract_features
from synth2surge.ml.experience_store import ExperienceStore

logger = logging.getLogger(__name__)

# Parameters that crash Surge XT when randomized (SIGSEGV).
# Identified by binary search: trigger_mode and play_mode params cause
# crashes when set to arbitrary values.  FX type params (fx_*_fx_type)
# also crash because each FX type expects specific param semantics.
_CRASH_PATTERNS = ("trigger_mode", "play_mode")


def _safe_param_names(all_names: list[str]) -> list[str]:
    """Filter parameter names to only those safe to randomize.

    Excludes:
    - Scene B params (b_*) — we only use scene A
    - FX / global params (fx_*, active_scene, bypass, etc.) — FX type randomization crashes
    - trigger_mode, play_mode — crash Surge XT with arbitrary values
    """
    safe = []
    for name in all_names:
        # Only scene A params
        if not name.startswith("a_"):
            continue
        # Skip params known to crash
        if any(pattern in name for pattern in _CRASH_PATTERNS):
            continue
        safe.append(name)
    return safe


def _random_midi_config(rng: random.Random) -> MidiProbeConfig:
    """Generate a semi-random MIDI probe configuration."""
    return MidiProbeConfig(
        note=rng.randint(36, 84),
        velocity=rng.randint(60, 127),
        sustain_seconds=rng.uniform(1.5, 4.0),
        release_seconds=rng.uniform(0.5, 2.0),
    )


def generate_render_only(
    surge_plugin_path: str | Path,
    store: ExperienceStore,
    count: int = 100,
    *,
    sample_rate: int = 44100,
    seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
    resume: bool = True,
    feature_backend: str = "clap",
    probe_mode: str = "full",
) -> int:
    """Generate training data by randomizing params and rendering audio.

    This is the fastest mode (~2s per patch). No optimization is run.
    Each patch produces a clean (features, ground_truth_params) pair.

    Args:
        surge_plugin_path: Path to Surge XT VST3.
        store: ExperienceStore to write data to.
        count: Number of random patches to generate.
        sample_rate: Audio sample rate.
        seed: Random seed for reproducibility.
        progress_callback: Called with (completed, total) after each patch.
        resume: If True, skip patches if store already has enough data.

    Returns:
        Number of patches successfully generated.
    """
    from synth2surge.audio.engine import PluginHost

    if resume:
        existing = store.count()
        if existing >= count:
            logger.info(f"Store already has {existing} runs, skipping generation")
            return 0
        # Adjust seed to avoid re-generating same patches
        seed += existing

    host = PluginHost(surge_plugin_path, sample_rate=sample_rate)
    all_param_names = sorted(host.parameter_names())
    safe_names = _safe_param_names(all_param_names)
    default_values = host.get_raw_values()
    np_rng = np.random.RandomState(seed)
    logger.info(
        f"Randomizing {len(safe_names)}/{len(all_param_names)} safe params "
        f"(skipping {len(all_param_names) - len(safe_names)} scene-B/control params)"
    )

    generated = 0
    for i in range(count):
        run_id = store.new_run_id()
        try:
            # 1. Randomize only safe parameters (scene A + safe globals)
            random_values = {name: float(np_rng.uniform(0.0, 1.0)) for name in safe_names}
            # Build full param vector: random for safe params, defaults for the rest
            full_values = {n: random_values.get(n, default_values.get(n, 0.5))
                           for n in all_param_names}
            host.set_raw_values(random_values)
            host.reset()

            # 2. Render standardized multi-probe set
            features, segments = render_standard_features(
                host, sr=sample_rate,
                feature_backend=feature_backend, probe_mode=probe_mode,
            )

            # 3. Skip silent patches (check if any segment has audio)
            total_rms = max(
                float(np.sqrt(np.mean(seg**2))) for seg in segments if len(seg) > 0
            ) if segments else 0.0
            if total_rms < 1e-6:
                logger.debug(f"Patch {i} is silent across all probes, skipping")
                continue

            # 4. Skip if features are all zeros
            if np.linalg.norm(features) < 1e-10:
                continue

            # 5. Store ground truth (full param vector for consistent shape)
            gt_params = np.array([full_values[n] for n in all_param_names], dtype=np.float32)

            # 6. Concatenate audio segments for round-trip validation (float16 to save space)
            audio_concat = np.concatenate(
                [seg.astype(np.float16) for seg in segments if len(seg) > 0]
            )

            store.log_run(
                run_id=run_id,
                target_features=features,
                best_params=gt_params,
                param_names=all_param_names,
                best_loss=0.0,
                total_trials=0,
                ground_truth_params=gt_params,
                probe_mode=probe_mode,
                generation_mode="random",
                feature_backend=feature_backend,
                target_audio=audio_concat,
                midi_config_json=json.dumps({"probe_mode": probe_mode}),
            )
            generated += 1

        except Exception:
            logger.exception(f"Error generating patch {i}")
            continue

        if progress_callback is not None:
            progress_callback(i + 1, count)

    return generated


def generate_with_optimization(
    surge_plugin_path: str | Path,
    store: ExperienceStore,
    count: int = 50,
    *,
    trials_per_run: int = 200,
    sample_rate: int = 44100,
    seed: int = 42,
    progress_callback: Callable[[int, int], None] | None = None,
    feature_backend: str = "clap",
    probe_mode: str = "full",
) -> int:
    """Generate training data with CMA-ES optimization for richer trial-level data.

    Slower than render-only (~3-5 min per patch) but produces intermediate
    (params, loss) pairs from each of the ~N trials.

    Args:
        surge_plugin_path: Path to Surge XT VST3.
        store: ExperienceStore to write data to.
        count: Number of random patches to generate and optimize.
        trials_per_run: CMA-ES trials per optimization run.
        sample_rate: Audio sample rate.
        seed: Random seed.
        progress_callback: Called with (completed, total) after each patch.

    Returns:
        Number of patches successfully generated.
    """
    from synth2surge.audio.engine import PluginHost
    from synth2surge.config import OptimizationConfig
    from synth2surge.optimizer.loop import optimize

    host = PluginHost(surge_plugin_path, sample_rate=sample_rate)
    all_param_names = sorted(host.parameter_names())
    safe_names = _safe_param_names(all_param_names)
    default_values = host.get_raw_values()
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Use reduced trials for data generation
    trials_per_tier = trials_per_run // 3
    config = OptimizationConfig(
        n_trials_tier1=trials_per_tier,
        n_trials_tier2=trials_per_tier,
        n_trials_tier3=trials_per_run - 2 * trials_per_tier,
    )

    generated = 0
    for i in range(count):
        run_id = store.new_run_id()
        try:
            # 1. Randomize only safe parameters
            random_values = {name: float(np_rng.uniform(0.0, 1.0)) for name in safe_names}
            full_values = {n: random_values.get(n, default_values.get(n, 0.5))
                           for n in all_param_names}
            host.set_raw_values(random_values)
            host.reset()

            # 2. Render standardized multi-probe features
            features, segments = render_standard_features(
                host, sr=sample_rate,
                feature_backend=feature_backend, probe_mode=probe_mode,
            )

            # Check for silence
            total_rms = max(
                float(np.sqrt(np.mean(seg**2))) for seg in segments if len(seg) > 0
            ) if segments else 0.0
            if total_rms < 1e-6:
                continue

            if np.linalg.norm(features) < 1e-10:
                continue

            gt_params = np.array([full_values[n] for n in all_param_names], dtype=np.float32)

            # 3. Also render single-probe target audio for CMA-ES optimization
            midi_config = _random_midi_config(rng)
            target_audio = host.render_midi_mono(midi_config=midi_config)

            # 4. Reset host to default and run optimization to recover params
            host.set_raw_values({n: 0.5 for n in safe_names})
            host.reset()

            result = optimize(
                target_audio=target_audio,
                surge_host=host,
                config=config,
                midi_config=midi_config,
                experience_store=store,
                _run_id=run_id,
            )

            # 5. Log the run with ground truth
            store.log_run(
                run_id=run_id,
                target_features=features,
                best_params=np.array(
                    [host.get_raw_values().get(n, 0.5) for n in all_param_names], dtype=np.float32
                ),
                param_names=all_param_names,
                best_loss=result.best_loss,
                total_trials=result.total_trials,
                ground_truth_params=gt_params,
                probe_mode=probe_mode,
                generation_mode="optimize",
                feature_backend=feature_backend,
                midi_config_json=json.dumps({
                    "note": midi_config.note,
                    "velocity": midi_config.velocity,
                    "sustain": midi_config.sustain_seconds,
                    "release": midi_config.release_seconds,
                }),
            )
            generated += 1

        except Exception:
            logger.exception(f"Error generating/optimizing patch {i}")
            continue

        if progress_callback is not None:
            progress_callback(i + 1, count)

    return generated


def generate_from_factory(
    surge_plugin_path: str | Path,
    store: ExperienceStore,
    factory_dir: str | Path | None = None,
    max_patches: int = 50,
    *,
    sample_rate: int = 44100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Generate training data from Surge XT factory presets.

    Uses known factory presets as ground truth — these are real,
    musically useful patches (not random noise).

    Args:
        surge_plugin_path: Path to Surge XT VST3.
        store: ExperienceStore to write data to.
        factory_dir: Path to factory patches directory.
        max_patches: Max number of factory patches to process.
        sample_rate: Audio sample rate.
        progress_callback: Called with (completed, total).

    Returns:
        Number of patches successfully generated.
    """
    from synth2surge.audio.engine import PluginHost
    from synth2surge.config import SurgeConfig
    from synth2surge.surge.factory import discover_factory_patches

    if factory_dir is None:
        factory_dir = SurgeConfig().factory_patches_dir

    patches = discover_factory_patches(Path(factory_dir))[:max_patches]
    if not patches:
        logger.warning(f"No factory patches found in {factory_dir}")
        return 0

    host = PluginHost(surge_plugin_path, sample_rate=sample_rate)
    param_names = sorted(host.parameter_names())

    generated = 0
    for i, patch_path in enumerate(patches):
        run_id = store.new_run_id()
        try:
            from synth2surge.surge.preset_loader import load_fxp_into_host

            load_fxp_into_host(patch_path, host)
            host.reset()

            # Record ground truth params
            raw_values = host.get_raw_values()
            gt_params = np.array([raw_values.get(n, 0.0) for n in param_names], dtype=np.float32)

            # Render and extract features
            audio = host.render_midi_mono()
            rms = float(np.sqrt(np.mean(audio**2)))
            if rms < 1e-6:
                continue

            features = extract_features(audio, sr=sample_rate)
            if np.linalg.norm(features) < 1e-10:
                continue

            store.log_run(
                run_id=run_id,
                target_features=features,
                best_params=gt_params,
                param_names=param_names,
                best_loss=0.0,
                total_trials=0,
                ground_truth_params=gt_params,
                probe_mode="single",
                generation_mode="factory",
                midi_config_json=json.dumps({"preset": str(patch_path)}),
            )
            generated += 1

        except Exception:
            logger.exception(f"Error processing factory patch {patch_path.name}")
            continue

        if progress_callback is not None:
            progress_callback(i + 1, len(patches))

    return generated
