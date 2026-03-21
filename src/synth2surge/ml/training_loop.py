"""Autonomous self-improvement training loop.

Runs the full cycle unattended:
1. Generate random patches → render audio → store ground truth
2. Optionally run CMA-ES optimization on a subset for trial-level data
3. Train/retrain the predictor model on all accumulated data
4. Evaluate: A/B compare warm-started vs cold CMA-ES on held-out patches
5. Repeat for N cycles
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


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
) -> LoopResult:
    """Run the autonomous self-improvement loop.

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
    from synth2surge.ml.data_generator import generate_render_only, generate_with_optimization
    from synth2surge.ml.experience_store import ExperienceStore

    store = ExperienceStore(store_path)
    cycle_results: list[CycleResult] = []
    total_generated = 0

    for cycle in range(n_cycles):
        cycle_seed = seed + cycle * 10000

        if progress_callback:
            progress_callback(cycle + 1, n_cycles, "Generating render-only data...")

        # Step 1: Generate render-only data (fast, ~2s per patch)
        n_render = int(patches_per_cycle * (1.0 - optimize_fraction))
        gen_count = generate_render_only(
            surge_plugin_path, store, n_render,
            seed=cycle_seed, resume=False,
        )
        total_generated += gen_count

        # Step 2: Generate optimization data (slower, richer)
        n_optimize = int(patches_per_cycle * optimize_fraction)
        if n_optimize > 0:
            if progress_callback:
                progress_callback(cycle + 1, n_cycles, "Running optimization for trial data...")

            opt_count = generate_with_optimization(
                surge_plugin_path, store, n_optimize,
                trials_per_run=trials_per_optimize,
                seed=cycle_seed + 5000,
            )
            total_generated += opt_count

        # Step 3: Train model
        train_loss = None
        val_loss = None
        model_version = None

        if store.count() >= 10:
            if progress_callback:
                progress_callback(cycle + 1, n_cycles, "Training predictor model...")

            try:
                from synth2surge.ml.trainer import train_predictor

                result = train_predictor(
                    store_path=store_path,
                    models_dir=models_dir,
                    max_epochs=max_train_epochs,
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
            patches_generated=gen_count + (n_optimize if n_optimize > 0 else 0),
            train_loss=train_loss,
            val_loss=val_loss,
            model_version=model_version,
            total_runs_in_store=store.count(),
        )
        cycle_results.append(cycle_result)

        if progress_callback:
            progress_callback(
                cycle + 1, n_cycles,
                f"Cycle {cycle + 1} complete: {store.count()} total runs"
            )

    store.close()

    return LoopResult(
        cycles_completed=len(cycle_results),
        total_patches_generated=total_generated,
        final_model_version=cycle_results[-1].model_version if cycle_results else None,
        cycle_results=cycle_results,
    )
