"""High-level render pipeline for producing audio from plugin + patch combinations."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

from synth2surge.audio.engine import PluginHost
from synth2surge.audio.midi import probe_duration
from synth2surge.config import MidiProbeConfig
from synth2surge.types import RenderResult


def render_patch(
    host: PluginHost,
    state: bytes | None = None,
    midi_config: MidiProbeConfig | None = None,
) -> RenderResult:
    """Render the current (or specified) patch state through a plugin.

    Args:
        host: A loaded PluginHost instance.
        state: Optional binary state to inject before rendering.
        midi_config: MIDI probe configuration.

    Returns:
        RenderResult with audio, sample rate, and duration.
    """
    if state is not None:
        host.set_state(state)

    host.reset()
    duration = probe_duration(midi_config)
    audio = host.render_midi_mono(midi_config=midi_config)

    return RenderResult(
        audio=audio,
        sample_rate=host.sample_rate,
        duration=duration,
    )


def render_and_save(
    host: PluginHost,
    output_path: str | Path,
    state: bytes | None = None,
    midi_config: MidiProbeConfig | None = None,
) -> RenderResult:
    """Render a patch and save the audio to a WAV file."""
    result = render_patch(host, state=state, midi_config=midi_config)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), result.audio, result.sample_rate)
    return result
