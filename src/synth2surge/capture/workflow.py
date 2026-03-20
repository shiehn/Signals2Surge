"""Source plugin capture workflow.

Handles loading a source synth plugin, optionally showing its GUI for preset
selection, extracting state, and rendering target audio.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from synth2surge.audio.engine import PluginHost
from synth2surge.audio.midi import compose_multi_probe
from synth2surge.config import AudioConfig, MidiProbeConfig, MultiProbeConfig
from synth2surge.types import CaptureResult


def capture_headless(
    plugin_path: str | Path,
    output_dir: str | Path,
    state_data: bytes | None = None,
    midi_config: MidiProbeConfig | None = None,
    sample_rate: int | None = None,
    multi_probe_config: MultiProbeConfig | None = None,
) -> CaptureResult:
    """Capture a preset without GUI — uses current or provided state.

    Args:
        plugin_path: Path to VST3/AU plugin.
        output_dir: Directory to save target_audio.wav and target_state.bin.
        state_data: Optional preset state bytes to load before rendering.
        midi_config: MIDI probe configuration.
        sample_rate: Audio sample rate.
        multi_probe_config: Optional multi-probe config for thorough/full capture.

    Returns:
        CaptureResult with paths to saved files and audio data.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or AudioConfig().sample_rate
    host = PluginHost(plugin_path, sample_rate=sr)

    if state_data is not None:
        host.set_state(state_data)

    # Extract state
    state = host.get_state()
    state_path = output_dir / "target_state.bin"
    state_path.write_bytes(state)

    # Render audio
    host.reset()
    audio_segments: list[np.ndarray] | None = None

    if multi_probe_config is not None and multi_probe_config.mode != "single":
        multi_probe = compose_multi_probe(multi_probe_config, sample_rate=sr)
        audio, audio_segments = host.render_multi_probe(multi_probe)

        # Save segments as .npz
        segments_path = output_dir / "target_segments.npz"
        np.savez(
            str(segments_path),
            **{f"segment_{i}": seg for i, seg in enumerate(audio_segments)},
        )
    else:
        audio = host.render_midi_mono(midi_config=midi_config)

    audio_path = output_dir / "target_audio.wav"
    sf.write(str(audio_path), audio, sr)

    # Get parameters
    params = host.get_parameters()
    float_params = {k: v for k, v in params.items() if isinstance(v, (int, float))}

    return CaptureResult(
        audio_path=audio_path,
        state_path=state_path,
        parameters=float_params,
        audio=audio,
        audio_segments=audio_segments,
    )


def capture_from_state_file(
    plugin_path: str | Path,
    state_file: str | Path,
    output_dir: str | Path,
    midi_config: MidiProbeConfig | None = None,
    sample_rate: int | None = None,
    multi_probe_config: MultiProbeConfig | None = None,
) -> CaptureResult:
    """Capture a preset by loading a binary state file."""
    state_data = Path(state_file).read_bytes()
    return capture_headless(
        plugin_path=plugin_path,
        output_dir=output_dir,
        state_data=state_data,
        midi_config=midi_config,
        sample_rate=sample_rate,
        multi_probe_config=multi_probe_config,
    )


def capture_with_gui(
    plugin_path: str | Path,
    output_dir: str | Path,
    midi_config: MidiProbeConfig | None = None,
    sample_rate: int | None = None,
    multi_probe_config: MultiProbeConfig | None = None,
) -> CaptureResult:
    """Capture a preset via the plugin's native GUI.

    Shows the plugin editor window. The user selects their desired preset
    and closes the window. After the window closes, the state is captured
    and audio is rendered.

    WARNING: This blocks the calling thread until the editor window is closed.
    Must be called from the main thread on macOS.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or AudioConfig().sample_rate
    host = PluginHost(plugin_path, sample_rate=sr)

    # Show the plugin's native GUI — blocks until user closes window
    host._plugin.show_editor()

    # After editor closes, capture the state
    return capture_headless(
        plugin_path=plugin_path,
        output_dir=output_dir,
        state_data=host.get_state(),
        midi_config=midi_config,
        sample_rate=sr,
        multi_probe_config=multi_probe_config,
    )
