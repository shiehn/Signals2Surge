"""Autonomous self-improvement training loop.

Runs the full cycle unattended:
1. Generate random patches → render audio → store ground truth
2. Optionally run CMA-ES optimization on a subset for trial-level data
3. Train/retrain the predictor model on all accumulated data
4. Evaluate: A/B compare warm-started vs cold CMA-ES on held-out patches
5. Repeat for N cycles

Architecture: Subprocesses handle crash-prone Surge XT rendering.
The parent process handles CLAP feature extraction (loaded once, reused).
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def _render_audio_worker(
    surge_plugin_path: str,
    output_dir: str,
    count: int,
    seed: int,
    probe_mode: str = "full",
) -> None:
    """Subprocess target: render audio only (no feature extraction).

    Saves raw audio segments + params to .npz files in output_dir.
    CLAP feature extraction happens in the parent process.
    """
    from synth2surge.audio.engine import PluginHost
    from synth2surge.audio.midi import compose_multi_probe
    from synth2surge.audio.standard_probes import get_standard_probe_config

    host = PluginHost(surge_plugin_path, sample_rate=44100)
    all_param_names = sorted(host.parameter_names())

    # Filter to safe params (scene A only, no crash-prone params)
    safe_names = []
    crash_patterns = ("trigger_mode", "play_mode")
    for name in all_param_names:
        if not name.startswith("a_"):
            continue
        if any(p in name for p in crash_patterns):
            continue
        safe_names.append(name)

    default_values = host.get_raw_values()
    np_rng = np.random.RandomState(seed)

    logger.info(
        f"Randomizing {len(safe_names)}/{len(all_param_names)} safe params "
        f"(skipping {len(all_param_names) - len(safe_names)} scene-B/control params)"
    )

    config = get_standard_probe_config(mode=probe_mode)
    multi_probe = compose_multi_probe(config, sample_rate=44100)

    generated = 0
    for i in range(count):
        try:
            # Randomize safe parameters
            random_values = {name: float(np_rng.uniform(0.0, 1.0)) for name in safe_names}
            full_values = {n: random_values.get(n, default_values.get(n, 0.5))
                           for n in all_param_names}
            host.set_raw_values(random_values)
            host.reset()

            # Render multi-probe audio
            _, segments = host.render_multi_probe(multi_probe)

            # Skip silent patches
            total_rms = max(
                float(np.sqrt(np.mean(seg**2))) for seg in segments if len(seg) > 0
            ) if segments else 0.0
            if total_rms < 1e-6:
                continue

            # Save audio + params to npz file (parent will extract features)
            gt_params = np.array([full_values[n] for n in all_param_names], dtype=np.float32)
            out_path = Path(output_dir) / f"patch_{seed}_{i}.npz"
            np.savez_compressed(
                out_path,
                params=gt_params,
                param_names=np.array(all_param_names),
                **{f"seg_{j}": seg.astype(np.float32) for j, seg in enumerate(segments)},
                n_segments=np.array([len(segments)]),
            )
            generated += 1

        except Exception:
            logger.exception(f"Error rendering patch {i}")
            continue

    logger.info(f"Render worker generated {generated} audio patches")


def _optimize_worker(
    surge_plugin_path: str,
    store_path: str,
    count: int,
    trials_per_run: int,
    seed: int,
    feature_backend: str = "clap",
    probe_mode: str = "full",
) -> None:
    """Subprocess target: generate optimization training data.

    Optimization workers still do their own feature extraction since they
    need features for the CMA-ES loss function internally.
    """
    from synth2surge.ml.data_generator import generate_with_optimization
    from synth2surge.ml.experience_store import ExperienceStore

    store = ExperienceStore(Path(store_path))
    try:
        generated = generate_with_optimization(
            surge_plugin_path, store, count,
            trials_per_run=trials_per_run, seed=seed,
            feature_backend=feature_backend, probe_mode=probe_mode,
        )
        logger.info(f"Optimization worker generated {generated} patches")
    finally:
        store.close()


@dataclass
class CycleResult:
    """Result of a single training cycle."""

    cycle: int
    patches_generated: int
    train_loss: float | None
    val_loss: float | None
    model_version: str | None
    total_runs_in_store: int


@dataclass
class LoopResult:
    """Result of the complete training loop."""

    cycles_completed: int
    total_patches_generated: int
    final_model_version: str | None
    cycle_results: list[CycleResult]


def _extract_features_from_npz(
    npz_dir: Path,
    store,
    feature_backend: str,
    probe_mode: str,
) -> int:
    """Extract CLAP features from rendered audio in parent process and store results.

    This runs in the parent process where CLAP is loaded once.

    Returns:
        Number of patches successfully processed.
    """
    from synth2surge.audio.standard_probes import (
        extract_multi_probe_features,
        get_probe_count,
    )

    n_probes = get_probe_count(probe_mode)
    npz_files = sorted(npz_dir.glob("patch_*.npz"))
    stored = 0

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
            params = data["params"]
            param_names = list(data["param_names"])
            n_segments = int(data["n_segments"][0])

            segments = [data[f"seg_{j}"] for j in range(n_segments)]

            # Extract features in parent process (CLAP loaded once)
            features = extract_multi_probe_features(
                segments, sr=44100, feature_backend=feature_backend, n_probes=n_probes,
            )

            if np.linalg.norm(features) < 1e-10:
                continue

            # Concatenate audio for round-trip storage
            audio_concat = np.concatenate(
                [seg.astype(np.float16) for seg in segments if len(seg) > 0]
            )

            run_id = store.new_run_id()
            store.log_run(
                run_id=run_id,
                target_features=features,
                best_params=params,
                param_names=param_names,
                best_loss=0.0,
                total_trials=0,
                ground_truth_params=params,
                probe_mode=probe_mode,
                generation_mode="random",
                feature_backend=feature_backend,
                target_audio=audio_concat,
                midi_config_json=json.dumps({"probe_mode": probe_mode}),
            )
            stored += 1

        except Exception:
            logger.exception(f"Error extracting features from {npz_path.name}")
            continue

    return stored


def run_training_loop(
    surge_plugin_path: str | Path,
    store_path: Path,
    models_dir: Path,
    *,
    n_cycles: int = 10,
    patches_per_cycle: int = 100,
    optimize_fraction: float = 0.1,
    trials_per_optimize: int = 200,
    max_train_epochs: int = 200,
    seed: int = 42,
    progress_callback: Callable[[int, int, str], None] | None = None,
    feature_backend: str = "clap",
    probe_mode: str = "full",
    hidden_dims: list[int] | None = None,
) -> LoopResult:
    """Run the autonomous self-improvement loop.

    Architecture:
    - Subprocesses: Surge XT rendering only (crash-isolated)
    - Parent process: CLAP feature extraction (loaded once), training

    Args:
        surge_plugin_path: Path to Surge XT VST3.
        store_path: Path to experience store database.
        models_dir: Directory for model checkpoints.
        n_cycles: Number of generate→train cycles.
        patches_per_cycle: Random patches to generate per cycle.
        optimize_fraction: Fraction of patches to run optimization on (for trial data).
        trials_per_optimize: CMA-ES trials when running optimization.
        max_train_epochs: Max epochs for model training.
        seed: Random seed base.
        progress_callback: Called with (cycle, total_cycles, status_message).

    Returns:
        LoopResult with metrics for each cycle.
    """
    from synth2surge.ml.experience_store import ExperienceStore

    store = ExperienceStore(store_path)
    cycle_results: list[CycleResult] = []
    total_generated = 0

    ctx = mp.get_context("spawn")

    for cycle in range(n_cycles):
        cycle_seed = seed + cycle * 10000
        count_before = store.count()

        if progress_callback:
            progress_callback(cycle + 1, n_cycles, "Rendering audio (subprocess)...")

        # Step 1: Render audio in sub-batches (subprocess, crash-isolated).
        # Audio is saved to temp .npz files; features extracted in parent.
        n_render = int(patches_per_cycle * (1.0 - optimize_fraction))
        sub_batch_size = 10

        with tempfile.TemporaryDirectory(prefix="s2s_render_") as tmp_dir:
            for batch_start in range(0, n_render, sub_batch_size):
                batch_count = min(sub_batch_size, n_render - batch_start)
                batch_seed = cycle_seed + batch_start
                proc = ctx.Process(
                    target=_render_audio_worker,
                    args=(str(surge_plugin_path), tmp_dir, batch_count, batch_seed,
                          probe_mode),
                )
                proc.start()
                proc.join(timeout=120)  # 2 min per sub-batch of 10
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                elif proc.exitcode != 0:
                    logger.warning(
                        f"Cycle {cycle + 1}: render sub-batch crashed "
                        f"(exit={proc.exitcode}), continuing"
                    )

            # Extract features in parent process (CLAP loaded once here)
            if progress_callback:
                progress_callback(cycle + 1, n_cycles, "Extracting CLAP features...")

            stored = _extract_features_from_npz(
                Path(tmp_dir), store, feature_backend, probe_mode,
            )
            logger.info(f"Cycle {cycle + 1}: extracted features for {stored} patches")

        # Step 2: Generate optimization data in isolated subprocess
        n_optimize = int(patches_per_cycle * optimize_fraction)
        if n_optimize > 0:
            if progress_callback:
                progress_callback(cycle + 1, n_cycles, "Running optimization for trial data...")

            proc = ctx.Process(
                target=_optimize_worker,
                args=(
                    str(surge_plugin_path), str(store_path),
                    n_optimize, trials_per_optimize, cycle_seed + 5000,
                    feature_backend, probe_mode,
                ),
            )
            proc.start()
            proc.join(timeout=1800)  # 30 min timeout
            if proc.is_alive():
                logger.warning(f"Cycle {cycle + 1}: optimize worker timed out, killing")
                proc.kill()
                proc.join()
            elif proc.exitcode != 0:
                logger.warning(
                    f"Cycle {cycle + 1}: optimize worker crashed (exit={proc.exitcode}), continuing"
                )

        # Re-read count to see what was produced
        count_after = store.count()
        gen_count = count_after - count_before
        total_generated += gen_count

        # Step 3: Train model (in parent process — PyTorch is stable)
        train_loss = None
        val_loss = None
        model_version = None

        if count_after >= 10:
            if progress_callback:
                progress_callback(cycle + 1, n_cycles, "Training predictor model...")

            try:
                from synth2surge.audio.standard_probes import get_feature_dim_for_mode
                from synth2surge.ml.trainer import train_predictor

                expected_dim = get_feature_dim_for_mode(probe_mode)
                result = train_predictor(
                    store_path=store_path,
                    models_dir=models_dir,
                    max_epochs=max_train_epochs,
                    feature_backend=feature_backend,
                    probe_mode=probe_mode,
                    hidden_dims=hidden_dims,
                    feature_dim=expected_dim,
                )
                if result is not None:
                    train_loss = result.final_train_loss
                    val_loss = result.best_val_loss
                    model_version = result.version_id
                    logger.info(
                        f"Cycle {cycle + 1}: trained {model_version}, "
                        f"val_loss={val_loss:.6f}"
                    )
            except ImportError:
                logger.warning("PyTorch not installed, skipping model training")
            except Exception:
                logger.exception(f"Training failed in cycle {cycle + 1}")

        cycle_result = CycleResult(
            cycle=cycle + 1,
            patches_generated=gen_count,
            train_loss=train_loss,
            val_loss=val_loss,
            model_version=model_version,
            total_runs_in_store=count_after,
        )
        cycle_results.append(cycle_result)

        if progress_callback:
            progress_callback(
                cycle + 1, n_cycles,
                f"Cycle {cycle + 1} complete: {count_after} total runs, "
                f"+{gen_count} this cycle"
            )

    store.close()

    return LoopResult(
        cycles_completed=len(cycle_results),
        total_patches_generated=total_generated,
        final_model_version=cycle_results[-1].model_version if cycle_results else None,
        cycle_results=cycle_results,
    )
