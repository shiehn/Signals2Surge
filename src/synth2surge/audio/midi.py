"""MIDI probe generation for consistent audio rendering.

The MIDI probe is a fixed sequence of MIDI messages used to render audio
through both the source and target plugins. Using the same probe ensures
that spectral differences reflect timbre, not performance differences.
"""

from __future__ import annotations

from synth2surge.config import MidiProbeConfig


def create_probe(
    config: MidiProbeConfig | None = None,
) -> list[tuple[float, int, int, int]]:
    """Create a MIDI probe as a list of (time_seconds, status, data1, data2) tuples.

    Pedalboard expects MIDI messages as a list of tuples:
    (time_in_seconds, status_byte, data1, data2)

    Returns a note-on/note-off pair with the configured timing.
    """
    if config is None:
        config = MidiProbeConfig()

    note_on_status = 0x90  # Note On, channel 1
    note_off_status = 0x80  # Note Off, channel 1

    messages = [
        (0.0, note_on_status, config.note, config.velocity),
        (config.sustain_seconds, note_off_status, config.note, 0),
    ]
    return messages


def probe_duration(config: MidiProbeConfig | None = None) -> float:
    """Total duration needed to capture the full probe including release tail."""
    if config is None:
        config = MidiProbeConfig()
    return config.total_duration
