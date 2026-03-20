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
) -> None:
    """Capture a preset from a source synth plugin."""
    from synth2surge.capture.workflow import (
        capture_from_state_file,
        capture_headless,
        capture_with_gui,
    )
    from synth2surge.config import MidiProbeConfig

    midi_config = MidiProbeConfig(
        note=note,
        velocity=velocity,
        sustain_seconds=duration - 1.0,
        release_seconds=1.0,
    )

    console.print(f"[bold]Loading plugin:[/bold] {plugin}")

    if state_file is not None:
        console.print(f"[bold]Loading state from:[/bold] {state_file}")
        result = capture_from_state_file(
            plugin_path=plugin,
            state_file=state_file,
            output_dir=output_dir,
            midi_config=midi_config,
        )
    elif no_gui:
        result = capture_headless(
            plugin_path=plugin,
            output_dir=output_dir,
            midi_config=midi_config,
        )
    else:
        console.print(
            "[yellow]Opening plugin editor — select preset, then close window.[/yellow]"
        )
        result = capture_with_gui(
            plugin_path=plugin,
            output_dir=output_dir,
            midi_config=midi_config,
        )

    console.print(f"[green]Audio saved:[/green] {result.audio_path}")
    console.print(f"[green]State saved:[/green] {result.state_path}")
    console.print(f"[green]Parameters captured:[/green] {len(result.parameters)}")


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
) -> None:
    """Optimize a Surge XT patch to match target audio."""
    import numpy as np
    import soundfile as sf

    from synth2surge.audio.engine import PluginHost
    from synth2surge.config import OptimizationConfig
    from synth2surge.optimizer.loop import optimize as run_optimize
    from synth2surge.types import OptimizationProgress

    stage_list = [int(s.strip()) for s in stages.split(",")]

    console.print(f"[bold]Target audio:[/bold] {target}")
    console.print(f"[bold]Stages:[/bold] {stage_list}")
    console.print(f"[bold]Trials:[/bold] T1={trials_t1}, T2={trials_t2}, T3={trials_t3}")

    target_audio, sr = sf.read(str(target), dtype="float32")
    if target_audio.ndim > 1:
        target_audio = np.mean(target_audio, axis=1)

    console.print(f"[bold]Loading Surge XT:[/bold] {surge_plugin}")
    host = PluginHost(surge_plugin, sample_rate=sr)

    config = OptimizationConfig(
        n_trials_tier1=trials_t1,
        n_trials_tier2=trials_t2,
        n_trials_tier3=trials_t3,
    )

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
        )

    console.print("\n[green bold]Optimization complete![/green bold]")
    console.print(f"  Best loss:    {result.best_loss:.6f}")
    console.print(f"  Total trials: {result.total_trials}")
    console.print(f"  Patch saved:  {result.best_patch_path}")
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


if __name__ == "__main__":
    app()
