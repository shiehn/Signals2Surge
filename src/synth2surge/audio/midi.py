"""MIDI probe generation for consistent audio rendering.

The MIDI probe is a fixed sequence of MIDI messages used to render audio
through both the source and target plugins. Using the same probe ensures
that spectral differences reflect timbre, not performance differences.

Pedalboard expects MIDI messages as a list of ([status, data1, data2], time)
tuples, where the first element is the MIDI bytes and the second is the
timestamp in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from synth2surge.config import MidiProbeConfig, MultiProbeConfig, ProbeType


def create_probe(
    config: MidiProbeConfig | None = None,
) -> MidiMessages:
    """Create a MIDI probe as a list of (midi_bytes, time) tuples.

    Returns a note-on/note-off pair with the configured timing.
    """
    if config is None:
        config = MidiProbeConfig()

    messages: MidiMessages = [
        ([NOTE_ON, config.note, config.velocity], 0.0),
        ([NOTE_OFF, config.note, 0], config.sustain_seconds),
    ]
    return messages


def probe_duration(config: MidiProbeConfig | None = None) -> float:
    """Total duration needed to capture the full probe including release tail."""
    if config is None:
        config = MidiProbeConfig()
    return config.total_duration


# ---------------------------------------------------------------------------
# Multi-probe generators
# ---------------------------------------------------------------------------

NOTE_ON = 0x90
NOTE_OFF = 0x80

# Each message is ([status_byte, data1, data2], timestamp_seconds)
MidiMessages = list[tuple[list[int], float]]


def create_sustained_probe(
    notes: list[int],
    velocity: int,
    sustain_s: float,
    release_s: float,
) -> MidiMessages:
    """Sequential sustained notes at different pitches."""
    msgs: MidiMessages = []
    t = 0.0
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + sustain_s))
        t += sustain_s + release_s
    return msgs


def create_velocity_probe(
    note: int,
    velocities: list[int],
    sustain_s: float,
    release_s: float,
) -> MidiMessages:
    """Same note at different velocities, sequential."""
    msgs: MidiMessages = []
    t = 0.0
    for vel in velocities:
        msgs.append(([NOTE_ON, note, vel], t))
        msgs.append(([NOTE_OFF, note, 0], t + sustain_s))
        t += sustain_s + release_s
    return msgs


def create_staccato_probe(
    note: int,
    velocity: int,
    n_hits: int = 3,
    hit_duration: float = 0.1,
    gap: float = 0.3,
) -> MidiMessages:
    """Short repeated notes with gaps."""
    msgs: MidiMessages = []
    t = 0.0
    for _ in range(n_hits):
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + hit_duration))
        t += hit_duration + gap
    return msgs


def create_chord_probe(
    notes: list[int],
    velocity: int,
    sustain_s: float,
    release_s: float,
) -> MidiMessages:
    """All note-ons simultaneous, all note-offs simultaneous."""
    msgs: MidiMessages = []
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], 0.0))
    for note in notes:
        msgs.append(([NOTE_OFF, note, 0], sustain_s))
    return msgs


def create_bass_line_probe(
    notes: list[int],
    velocity: int,
    note_duration: float,
) -> MidiMessages:
    """Walking bass pattern — sequential notes in low register."""
    msgs: MidiMessages = []
    t = 0.0
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + note_duration))
        t += note_duration
    return msgs


def create_melody_probe(
    notes: list[int],
    velocity: int,
    note_duration: float,
) -> MidiMessages:
    """Ascending/descending scale run."""
    msgs: MidiMessages = []
    t = 0.0
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + note_duration))
        t += note_duration
    return msgs


def create_chord_progression_probe(
    chords: list[list[int]],
    velocity: int,
    chord_duration: float,
    release_s: float,
) -> MidiMessages:
    """Sequence of block chords (each chord = list of simultaneous notes)."""
    msgs: MidiMessages = []
    t = 0.0
    for chord in chords:
        for note in chord:
            msgs.append(([NOTE_ON, note, velocity], t))
        for note in chord:
            msgs.append(([NOTE_OFF, note, 0], t + chord_duration))
        t += chord_duration + release_s
    return msgs


def create_comping_probe(
    chord_notes: list[int],
    velocity: int,
    n_hits: int = 4,
    hit_duration: float = 0.1,
    gap: float = 0.3,
) -> MidiMessages:
    """Rhythmic staccato chords."""
    msgs: MidiMessages = []
    t = 0.0
    for _ in range(n_hits):
        for note in chord_notes:
            msgs.append(([NOTE_ON, note, velocity], t))
        for note in chord_notes:
            msgs.append(([NOTE_OFF, note, 0], t + hit_duration))
        t += hit_duration + gap
    return msgs


def create_percussive_probe(
    note: int,
    velocity: int,
    n_hits: int = 3,
    decay_s: float = 1.0,
) -> MidiMessages:
    """Very short note-on followed by long decay."""
    hit_duration = 0.05
    msgs: MidiMessages = []
    t = 0.0
    for _ in range(n_hits):
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + hit_duration))
        t += hit_duration + decay_s
    return msgs


def create_legato_probe(
    notes: list[int],
    velocity: int,
    note_duration: float,
    overlap: float = 0.05,
) -> MidiMessages:
    """Overlapping notes for portamento/legato testing."""
    msgs: MidiMessages = []
    t = 0.0
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + note_duration))
        t += note_duration - overlap
    return msgs


def create_interval_jump_probe(
    notes: list[int],
    velocity: int,
    note_duration: float,
) -> MidiMessages:
    """Wide pitch jumps (octave leaps)."""
    msgs: MidiMessages = []
    t = 0.0
    for note in notes:
        msgs.append(([NOTE_ON, note, velocity], t))
        msgs.append(([NOTE_OFF, note, 0], t + note_duration))
        t += note_duration
    return msgs


# ---------------------------------------------------------------------------
# Probe segment tracking and composition
# ---------------------------------------------------------------------------


@dataclass
class ProbeSegment:
    """Tracks a single probe's position in the concatenated audio."""

    probe_type: ProbeType
    start_sample: int
    end_sample: int
    weight: float


@dataclass
class MultiProbeResult:
    """Result of composing multiple probes into a single MIDI message list."""

    midi_messages: MidiMessages
    total_duration: float
    segments: list[ProbeSegment] = field(default_factory=list)


def _probe_duration_from_messages(msgs: MidiMessages) -> float:
    """Duration of a probe based on its last message timestamp."""
    if not msgs:
        return 0.0
    return max(t for _, t in msgs)


def _generate_probe_messages(probe) -> tuple[MidiMessages, float]:
    """Generate MIDI messages for a single ProbeDefinition.

    Returns (messages, total_duration_before_release) where total_duration
    includes the release tail.
    """
    pt = probe.probe_type

    if pt == ProbeType.SUSTAINED or pt == ProbeType.SUB_BASS:
        msgs = create_sustained_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds, probe.release_seconds
        )
    elif pt == ProbeType.VELOCITY:
        msgs = create_velocity_probe(
            probe.notes[0], probe.velocities, probe.sustain_seconds, probe.release_seconds
        )
    elif pt == ProbeType.STACCATO:
        msgs = create_staccato_probe(
            probe.notes[0],
            probe.velocities[0],
            n_hits=3,
            hit_duration=probe.sustain_seconds,
            gap=probe.release_seconds,
        )
    elif pt == ProbeType.CHORD:
        msgs = create_chord_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds, probe.release_seconds
        )
    elif pt == ProbeType.BASS_LINE:
        msgs = create_bass_line_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds
        )
    elif pt == ProbeType.MELODY:
        msgs = create_melody_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds
        )
    elif pt == ProbeType.CHORD_PROGRESSION:
        # notes are flat: every 3 notes form a chord
        chord_size = 3
        chords = [
            probe.notes[i : i + chord_size]
            for i in range(0, len(probe.notes), chord_size)
        ]
        msgs = create_chord_progression_probe(
            chords, probe.velocities[0], probe.sustain_seconds, probe.release_seconds
        )
    elif pt == ProbeType.COMPING:
        msgs = create_comping_probe(
            probe.notes,
            probe.velocities[0],
            n_hits=4,
            hit_duration=probe.sustain_seconds,
            gap=probe.release_seconds,
        )
    elif pt == ProbeType.PERCUSSIVE:
        msgs = create_percussive_probe(
            probe.notes[0],
            probe.velocities[0],
            n_hits=3,
            decay_s=probe.release_seconds,
        )
    elif pt == ProbeType.LEGATO:
        msgs = create_legato_probe(
            probe.notes,
            probe.velocities[0],
            note_duration=probe.sustain_seconds,
            overlap=0.05,
        )
    elif pt == ProbeType.INTERVAL_JUMP:
        msgs = create_interval_jump_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds
        )
    else:
        # Fallback: single sustained note
        msgs = create_sustained_probe(
            probe.notes, probe.velocities[0], probe.sustain_seconds, probe.release_seconds
        )

    # Compute the duration: last message time + release tail
    last_msg_time = _probe_duration_from_messages(msgs)
    duration = last_msg_time + probe.release_seconds

    return msgs, duration


def compose_multi_probe(config: MultiProbeConfig, sample_rate: int = 44100) -> MultiProbeResult:
    """Compose multiple probes into a single concatenated MIDI message list.

    Offsets each probe's timestamps, inserts silence gaps between probes,
    and records segment boundaries for later slicing.
    """
    all_messages: MidiMessages = []
    segments: list[ProbeSegment] = []
    current_time = 0.0

    for i, probe_def in enumerate(config.probes):
        msgs, duration = _generate_probe_messages(probe_def)

        # Offset all timestamps by current_time
        offset_msgs: MidiMessages = [(midi_bytes, t + current_time) for midi_bytes, t in msgs]
        all_messages.extend(offset_msgs)

        start_sample = int(current_time * sample_rate)
        end_sample = int((current_time + duration) * sample_rate)

        segments.append(
            ProbeSegment(
                probe_type=probe_def.probe_type,
                start_sample=start_sample,
                end_sample=end_sample,
                weight=probe_def.weight,
            )
        )

        current_time += duration

        # Add gap between probes (not after the last one)
        if i < len(config.probes) - 1:
            current_time += probe_def.gap_seconds

    return MultiProbeResult(
        midi_messages=all_messages,
        total_duration=current_time,
        segments=segments,
    )
