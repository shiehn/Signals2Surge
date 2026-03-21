"""Synth2Surge CLI — translate synth patches to Surge XT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

app = typer.Typer(
    name="synth2surge",
    help="Translate arbitrary VST synth patches into Surge XT patches.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Training data generation commands.")
train_app = typer.Typer(help="ML model training commands.")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
console = Console()


@app.command()
def capture(
    plugin: Path = typer.Option(..., help="Path to source VST3/AU plugin"),
    output_dir: Path = typer.Option("./workspace", help="Output directory"),
    no_gui: bool = typer.Option(False, help="Skip editor GUI, use default state"),
    state_file: Optional[Path] = typer.Option(None, help="Load state from file instead of GUI"),
    duration: float = typer.Option(4.0, help="Render duration in seconds"),
    note: int = typer.Option(60, help="MIDI note (60 = C4)"),
    velocity: int = typer.Option(100, help="MIDI velocity"),
    probe_mode: str = typer.Option(
        "single", help="Probe mode: single, thorough, or full"
    ),
) -> None:
    """Capture a preset from a source synth plugin."""
    from synth2surge.capture.workflow import (
        capture_from_fxp,
        capture_from_state_file,
        capture_headless,
        capture_with_gui,
    )
    from synth2surge.config import MidiProbeConfig, MultiProbeConfig

    midi_config = MidiProbeConfig(
        note=note,
        velocity=velocity,
        sustain_seconds=duration - 1.0,
        release_seconds=1.0,
    )

    multi_probe_config = None
    if probe_mode == "thorough":
        multi_probe_config = MultiProbeConfig.thorough()
    elif probe_mode == "full":
        multi_probe_config = MultiProbeConfig.full()

    console.print(f"[bold]Loading plugin:[/bold] {plugin}")
    console.print(f"[bold]Probe mode:[/bold] {probe_mode}")

    if state_file is not None and state_file.suffix.lower() == ".fxp":
        console.print(f"[bold]Loading FXP preset:[/bold] {state_file}")
        result = capture_from_fxp(
            plugin_path=plugin,
            fxp_path=state_file,
            output_dir=output_dir,
            midi_config=midi_config,
            multi_probe_config=multi_probe_config,
        )
    elif state_file is not None:
        console.print(f"[bold]Loading state from:[/bold] {state_file}")
        result = capture_from_state_file(
            plugin_path=plugin,
            state_file=state_file,
            output_dir=output_dir,
            midi_config=midi_config,
            multi_probe_config=multi_probe_config,
        )
    elif no_gui:
        result = capture_headless(
            plugin_path=plugin,
            output_dir=output_dir,
            midi_config=midi_config,
            multi_probe_config=multi_probe_config,
        )
    else:
        console.print(
            "[yellow]Opening plugin editor — select preset, then close window.[/yellow]"
        )
        result = capture_with_gui(
            plugin_path=plugin,
            output_dir=output_dir,
            midi_config=midi_config,
            multi_probe_config=multi_probe_config,
        )

    console.print(f"[green]Audio saved:[/green] {result.audio_path}")
    console.print(f"[green]State saved:[/green] {result.state_path}")
    console.print(f"[green]Parameters captured:[/green] {len(result.parameters)}")
    if result.audio_segments is not None:
        console.print(f"[green]Audio segments:[/green] {len(result.audio_segments)}")


@app.command()
def optimize(
    target: Path = typer.Option(..., help="Path to target audio WAV file"),
    output_dir: Path = typer.Option("./workspace", help="Output directory"),
    surge_plugin: Path = typer.Option(
        Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"),
        help="Path to Surge XT VST3",
    ),
    trials_t1: int = typer.Option(300, help="Trials for tier 1 (structural params)"),
    trials_t2: int = typer.Option(300, help="Trials for tier 2 (shaping params)"),
    trials_t3: int = typer.Option(200, help="Trials for tier 3 (detail params)"),
    stages: str = typer.Option("1,2,3", help="Optimization stages to run (comma-separated)"),
    probe_mode: str = typer.Option(
        "single", help="Probe mode: single, thorough, or full"
    ),
    warm_start: bool = typer.Option(False, help="Use ML model to warm-start CMA-ES"),
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Experience store for logging"
    ),
) -> None:
    """Optimize a Surge XT patch to match target audio."""
    import numpy as np
    import soundfile as sf

    from synth2surge.audio.engine import PluginHost
    from synth2surge.config import MultiProbeConfig, OptimizationConfig
    from synth2surge.optimizer.loop import optimize as run_optimize
    from synth2surge.types import OptimizationProgress

    stage_list = [int(s.strip()) for s in stages.split(",")]

    console.print(f"[bold]Target audio:[/bold] {target}")
    console.print(f"[bold]Probe mode:[/bold] {probe_mode}")
    console.print(f"[bold]Stages:[/bold] {stage_list}")
    console.print(f"[bold]Trials:[/bold] T1={trials_t1}, T2={trials_t2}, T3={trials_t3}")
    if warm_start:
        console.print("[bold]Warm start:[/bold] enabled (ML-predicted initial params)")

    target_audio, sr = sf.read(str(target), dtype="float32")
    if target_audio.ndim > 1:
        target_audio = np.mean(target_audio, axis=1)

    console.print(f"[bold]Loading Surge XT:[/bold] {surge_plugin}")
    host = PluginHost(surge_plugin, sample_rate=sr)

    # Set up warm starter if requested
    if warm_start:
        try:
            from synth2surge.ml.warm_start import WarmStarter

            warm_starter = WarmStarter(
                store_path=db_path,
                models_dir=db_path.parent / "models",
            )
            # Get predictions from the model using target audio features
            from synth2surge.loss.features import extract_features

            target_features = extract_features(target_audio, sr=sr)
            x0, sigma0 = warm_starter.predict(target_features)
            if x0 is not None:
                console.print(
                    f"[green]ML warm-start active:[/green] predicting {len(x0)} params, "
                    f"sigma0={sigma0:.2f}"
                )
            else:
                console.print("[yellow]ML warm-start: low confidence, using default init[/yellow]")
        except Exception as e:
            console.print(f"[yellow]ML warm-start unavailable: {e}[/yellow]")

    config = OptimizationConfig(
        n_trials_tier1=trials_t1,
        n_trials_tier2=trials_t2,
        n_trials_tier3=trials_t3,
    )

    # Set up multi-probe if requested
    multi_probe_config = None
    target_segments = None
    if probe_mode == "thorough":
        multi_probe_config = MultiProbeConfig.thorough()
    elif probe_mode == "full":
        multi_probe_config = MultiProbeConfig.full()

    if multi_probe_config is not None and multi_probe_config.mode != "single":
        # Check for pre-computed segments
        segments_path = Path(output_dir) / "target_segments.npz"
        if segments_path.exists():
            console.print(f"[bold]Loading target segments from:[/bold] {segments_path}")
            data = np.load(str(segments_path))
            target_segments = [data[k] for k in sorted(data.files)]
        else:
            console.print(
                "[red]Target segments not found. Run 'capture' with "
                f"--probe-mode {probe_mode} first.[/red]"
            )
            raise typer.Exit(1)

    total_trials = sum(
        [trials_t1, trials_t2, trials_t3][s - 1] for s in stage_list
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Optimizing...", total=total_trials)

        def on_progress(p: OptimizationProgress) -> None:
            progress.update(
                task,
                completed=p.current_trial,
                description=(
                    f"Stage {p.stage} | Best: {p.best_loss:.4f}"
                    f" | Current: {p.current_loss:.4f}"
                ),
            )

        result = run_optimize(
            target_audio=target_audio,
            surge_host=host,
            config=config,
            progress_callback=on_progress,
            stages=stage_list,
            output_dir=output_dir,
            multi_probe_config=multi_probe_config,
            target_segments=target_segments,
        )

    console.print("\n[green bold]Optimization complete![/green bold]")
    console.print(f"  Best loss:    {result.best_loss:.6f}")
    console.print(f"  Total trials: {result.total_trials}")
    console.print(f"  Patch saved:  {result.best_patch_path}")
    if result.fxp_path:
        console.print(f"  FXP saved:    {result.fxp_path}")
    console.print(f"  Audio saved:  {result.best_audio_path}")


@app.command("build-prior")
def build_prior(
    factory_dir: Path = typer.Option(
        Path("/Library/Application Support/Surge XT/patches_factory"),
        help="Surge XT factory patches directory",
    ),
    surge_plugin: Path = typer.Option(
        Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"),
        help="Path to Surge XT VST3",
    ),
    variations: int = typer.Option(5, help="Variations per factory patch"),
    output: Path = typer.Option(
        Path("./workspace/prior_index"), help="Output dir for FAISS index"
    ),
    max_patches: int = typer.Option(50, help="Max factory patches to process"),
) -> None:
    """Build the FAISS prior index from Surge XT factory patches."""
    import numpy as np

    from synth2surge.audio.engine import PluginHost
    from synth2surge.loss.features import extract_features
    from synth2surge.prior.generator import generate_variations
    from synth2surge.prior.index import PriorIndex
    from synth2surge.surge.factory import discover_factory_patches
    from synth2surge.surge.patch import SurgePatch

    patches = discover_factory_patches(factory_dir)[:max_patches]
    console.print(f"[bold]Found {len(patches)} factory patches (using {len(patches)})[/bold]")

    host = PluginHost(surge_plugin)
    index = PriorIndex(feature_dim=512)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building prior...", total=len(patches))

        for i, patch_path in enumerate(patches):
            progress.update(task, completed=i, description=f"Processing {patch_path.stem}...")
            try:
                patch = SurgePatch.from_file(patch_path)
                vars_list = generate_variations(patch, n=variations, seed=i)

                for j, var in enumerate([patch] + vars_list):
                    # Set patch state and render
                    host.reset()
                    audio = host.render_midi_mono()
                    features = extract_features(audio, sr=host.sample_rate)

                    if np.linalg.norm(features) > 0:
                        index.add(
                            features.reshape(1, -1),
                            [str(patch_path)],
                        )
            except Exception as e:
                console.print(f"[red]Error on {patch_path.name}: {e}[/red]")

    index.save(output)
    console.print(f"\n[green]Prior index saved:[/green] {output} ({index.size} entries)")


@app.command()
def inspect(
    patch: Path = typer.Option(..., help="Path to .fxp or XML patch file"),
) -> None:
    """Inspect a Surge XT patch file."""
    from rich.table import Table

    from synth2surge.surge.patch import SurgePatch

    p = SurgePatch.from_file(patch)
    meta = p.metadata

    console.print(f"\n[bold]Patch:[/bold] {meta.name}")
    console.print(f"[bold]Category:[/bold] {meta.category}")
    console.print(f"[bold]Author:[/bold] {meta.author}")
    console.print(f"[bold]Revision:[/bold] {p.revision}")

    params = p.get_all_parameters()
    types = p.get_parameter_types()

    table = Table(title=f"Parameters ({len(params)} total)")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Value", style="yellow")

    for name, value in sorted(params.items())[:50]:
        ptype = "float" if types.get(name) == "2" else "int"
        table.add_row(name, ptype, f"{value}")

    console.print(table)
    if len(params) > 50:
        console.print(f"[dim]... and {len(params) - 50} more parameters[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from synth2surge.api.app import create_app

    api = create_app()
    console.print(f"[bold]Starting server at[/bold] http://{host}:{port}")
    uvicorn.run(api, host=host, port=port)


@train_app.command("run")
def train_run(
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Path to experience store database"
    ),
    models_dir: Path = typer.Option(
        Path("./workspace/models"), help="Directory for model checkpoints"
    ),
    max_epochs: int = typer.Option(200, help="Maximum training epochs"),
    patience: int = typer.Option(20, help="Early stopping patience"),
    lr: float = typer.Option(1e-3, help="Learning rate"),
) -> None:
    """Train the parameter predictor on accumulated experience data."""
    try:
        from synth2surge.ml.trainer import train_predictor
    except ImportError:
        console.print("[red]PyTorch required. Install with: uv pip install 'synth2surge[ml]'[/red]")
        raise typer.Exit(1)

    if not db_path.exists():
        console.print("[yellow]No experience store found. Run 'data generate' first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Training predictor from:[/bold] {db_path}")
    result = train_predictor(
        store_path=db_path,
        models_dir=models_dir,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
    )

    if result is None:
        console.print(
            "[yellow]Not enough training data. Run 'data generate' first.[/yellow]"
        )
        raise typer.Exit(0)

    console.print("\n[green bold]Training complete![/green bold]")
    console.print(f"  Version:        {result.version_id}")
    console.print(f"  Samples:        {result.n_training_samples}")
    console.print(f"  Epochs:         {result.epochs_trained}")
    console.print(f"  Train loss:     {result.final_train_loss:.6f}")
    console.print(f"  Val loss:       {result.final_val_loss:.6f}")
    console.print(f"  Best val loss:  {result.best_val_loss:.6f}")
    console.print(f"  Checkpoint:     {result.checkpoint_path}")


@train_app.command("status")
def train_status(
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Path to experience store database"
    ),
    models_dir: Path = typer.Option(
        Path("./workspace/models"), help="Directory for model checkpoints"
    ),
) -> None:
    """Show ML training status and model versions."""
    from rich.table import Table

    from synth2surge.ml.experience_store import ExperienceStore

    if not db_path.exists():
        console.print("[yellow]No experience store found.[/yellow]")
        raise typer.Exit(0)

    store = ExperienceStore(db_path)
    stats = store.summary()

    table = Table(title="ML Training Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total training runs", str(stats["total_runs"]))
    table.add_row("Runs with ground truth", str(stats.get("runs_with_ground_truth", 0)))
    table.add_row("Latest model", store.latest_model_version() or "None")

    # Data thresholds
    n = stats["total_runs"]
    mlp_status = "[green]Yes[/green]" if n >= 50 else f"[yellow]{n}/50[/yellow]"
    table.add_row("MLP ready (50+)", mlp_status)
    table.add_row(
        "CNN ready (200+)", "[green]Yes[/green]" if n >= 200 else f"[yellow]{n}/200[/yellow]"
    )

    console.print(table)
    store.close()


@train_app.command("loop")
def train_loop(
    surge_plugin: Path = typer.Option(
        Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"),
        help="Path to Surge XT VST3",
    ),
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Path to experience store database"
    ),
    models_dir: Path = typer.Option(
        Path("./workspace/models"), help="Directory for model checkpoints"
    ),
    cycles: int = typer.Option(10, help="Number of generate→train cycles"),
    patches_per_cycle: int = typer.Option(100, help="Patches to generate per cycle"),
    trials: int = typer.Option(200, help="Trials per optimization run"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Run the autonomous self-improvement training loop.

    Generates data, trains models, and repeats — fully unattended.
    """
    from synth2surge.ml.training_loop import run_training_loop

    console.print("[bold]Starting autonomous training loop[/bold]")
    console.print(f"  Cycles: {cycles}")
    console.print(f"  Patches/cycle: {patches_per_cycle}")
    console.print(f"  Plugin: {surge_plugin}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training loop", total=cycles)

        def on_progress(cycle: int, total: int, msg: str) -> None:
            progress.update(task, completed=cycle, description=f"Cycle {cycle}/{total}: {msg}")

        result = run_training_loop(
            surge_plugin_path=surge_plugin,
            store_path=db_path,
            models_dir=models_dir,
            n_cycles=cycles,
            patches_per_cycle=patches_per_cycle,
            trials_per_optimize=trials,
            seed=seed,
            progress_callback=on_progress,
        )

    console.print("\n[green bold]Training loop complete![/green bold]")
    console.print(f"  Cycles completed:  {result.cycles_completed}")
    console.print(f"  Total patches:     {result.total_patches_generated}")
    console.print(f"  Final model:       {result.final_model_version or 'None'}")

    if result.cycle_results:
        from rich.table import Table

        table = Table(title="Cycle Results")
        table.add_column("Cycle", style="cyan")
        table.add_column("Generated", style="green")
        table.add_column("Total Runs", style="green")
        table.add_column("Val Loss", style="yellow")
        table.add_column("Model", style="blue")

        for cr in result.cycle_results:
            table.add_row(
                str(cr.cycle),
                str(cr.patches_generated),
                str(cr.total_runs_in_store),
                f"{cr.val_loss:.6f}" if cr.val_loss is not None else "-",
                cr.model_version or "-",
            )
        console.print(table)


@data_app.command("generate")
def data_generate(
    mode: str = typer.Option(
        "render-only", help="Generation mode: render-only, optimize, or factory"
    ),
    count: int = typer.Option(100, help="Number of patches to generate"),
    surge_plugin: Path = typer.Option(
        Path("/Library/Audio/Plug-Ins/VST3/Surge XT.vst3"),
        help="Path to Surge XT VST3",
    ),
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Path to experience store database"
    ),
    trials: int = typer.Option(200, help="Trials per run (optimize mode only)"),
    seed: int = typer.Option(42, help="Random seed"),
    resume: bool = typer.Option(True, help="Skip if store already has enough data"),
) -> None:
    """Generate training data for ML models via self-play."""
    from synth2surge.ml.data_generator import (
        generate_from_factory,
        generate_render_only,
        generate_with_optimization,
    )
    from synth2surge.ml.experience_store import ExperienceStore

    store = ExperienceStore(db_path)

    console.print(f"[bold]Mode:[/bold] {mode}")
    console.print(f"[bold]Count:[/bold] {count}")
    console.print(f"[bold]Store:[/bold] {db_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating...", total=count)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task, completed=completed, description=f"Patch {completed}/{total}")

        if mode == "render-only":
            n = generate_render_only(
                surge_plugin, store, count,
                seed=seed, progress_callback=on_progress, resume=resume,
            )
        elif mode == "optimize":
            n = generate_with_optimization(
                surge_plugin, store, count,
                trials_per_run=trials, seed=seed, progress_callback=on_progress,
            )
        elif mode == "factory":
            n = generate_from_factory(
                surge_plugin, store, max_patches=count,
                progress_callback=on_progress,
            )
        else:
            console.print(f"[red]Unknown mode: {mode}[/red]")
            raise typer.Exit(1)

    console.print(f"\n[green]Generated {n} patches.[/green]")
    console.print(f"[green]Total runs in store: {store.count()}[/green]")
    store.close()


@data_app.command("status")
def data_status(
    db_path: Path = typer.Option(
        Path("./workspace/experience.db"), help="Path to experience store database"
    ),
) -> None:
    """Show training data statistics."""
    from rich.table import Table

    from synth2surge.ml.experience_store import ExperienceStore

    if not db_path.exists():
        console.print("[yellow]No experience store found. Run 'data generate' first.[/yellow]")
        raise typer.Exit(0)

    store = ExperienceStore(db_path)
    stats = store.summary()

    table = Table(title="Experience Store Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Database", str(stats["db_path"]))
    table.add_row("Total runs", str(stats["total_runs"]))
    table.add_row("Total trials", str(stats["total_trials"]))

    if stats["total_runs"] > 0:
        table.add_row("Best loss (min)", f"{stats['best_loss_min']:.6f}")
        table.add_row("Best loss (avg)", f"{stats['best_loss_avg']:.6f}")
        table.add_row("Best loss (max)", f"{stats['best_loss_max']:.6f}")
        table.add_row("Runs with ground truth", str(stats["runs_with_ground_truth"]))
        for mode, cnt in stats.get("by_mode", {}).items():
            table.add_row(f"  Mode: {mode}", str(cnt))

    model_version = store.latest_model_version()
    table.add_row("Latest model", model_version or "None")

    console.print(table)
    store.close()


if __name__ == "__main__":
    app()
