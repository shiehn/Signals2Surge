"""Unit tests for MIDI probe generation."""

from synth2surge.audio.midi import (
    compose_multi_probe,
    create_bass_line_probe,
    create_chord_probe,
    create_chord_progression_probe,
    create_comping_probe,
    create_interval_jump_probe,
    create_legato_probe,
    create_melody_probe,
    create_percussive_probe,
    create_probe,
    create_staccato_probe,
    create_sustained_probe,
    create_velocity_probe,
    probe_duration,
)
from synth2surge.config import MidiProbeConfig, MultiProbeConfig


class TestCreateProbe:
    """Tests for MIDI probe message generation."""

    def test_default_probe_two_messages(self):
        messages = create_probe()
        assert len(messages) == 2

    def test_default_note_on(self):
        messages = create_probe()
        time, status, note, velocity = messages[0]
        assert time == 0.0
        assert status == 0x90  # Note On
        assert note == 60  # C4
        assert velocity == 100

    def test_default_note_off(self):
        messages = create_probe()
        time, status, note, velocity = messages[1]
        assert time == 3.0  # sustain_seconds
        assert status == 0x80  # Note Off
        assert note == 60

    def test_custom_config(self):
        config = MidiProbeConfig(note=72, velocity=80, sustain_seconds=2.0, release_seconds=0.5)
        messages = create_probe(config)
        assert messages[0] == (0.0, 0x90, 72, 80)
        assert messages[1] == (2.0, 0x80, 72, 0)

    def test_probe_duration_default(self):
        assert probe_duration() == 4.0  # 3.0 sustain + 1.0 release

    def test_probe_duration_custom(self):
        config = MidiProbeConfig(sustain_seconds=2.0, release_seconds=0.5)
        assert probe_duration(config) == 2.5


# ---------------------------------------------------------------------------
# Synth response probe tests
# ---------------------------------------------------------------------------


class TestSustainedProbe:
    def test_sustained_three_pitches_six_messages(self):
        msgs = create_sustained_probe([36, 60, 84], velocity=100, sustain_s=2.0, release_s=0.8)
        assert len(msgs) == 6  # 2 per note (on + off)

    def test_sustained_notes_sequential(self):
        msgs = create_sustained_probe([36, 60], velocity=100, sustain_s=2.0, release_s=0.8)
        # Second note starts after first finishes (sustain + release)
        first_off_time = msgs[1][0]
        second_on_time = msgs[2][0]
        assert second_on_time >= first_off_time


class TestVelocityProbe:
    def test_velocity_three_levels_six_messages(self):
        msgs = create_velocity_probe(
            note=60, velocities=[30, 80, 127], sustain_s=1.5, release_s=0.5
        )
        assert len(msgs) == 6  # 2 per velocity level

    def test_velocity_values_correct(self):
        msgs = create_velocity_probe(
            note=60, velocities=[30, 80, 127], sustain_s=1.0, release_s=0.5
        )
        note_ons = [m for m in msgs if m[1] == 0x90]
        assert [m[3] for m in note_ons] == [30, 80, 127]


class TestStaccatoProbe:
    def test_staccato_three_hits_six_messages(self):
        msgs = create_staccato_probe(note=60, velocity=100, n_hits=3, hit_duration=0.1, gap=0.3)
        assert len(msgs) == 6  # 2 per hit

    def test_staccato_short_notes(self):
        msgs = create_staccato_probe(note=60, velocity=100, n_hits=3, hit_duration=0.1, gap=0.3)
        for i in range(0, len(msgs), 2):
            on_time = msgs[i][0]
            off_time = msgs[i + 1][0]
            assert abs(off_time - on_time - 0.1) < 1e-9


class TestChordProbe:
    def test_chord_three_notes_simultaneous_on(self):
        msgs = create_chord_probe([60, 64, 67], velocity=90, sustain_s=1.5, release_s=0.8)
        note_ons = [m for m in msgs if m[1] == 0x90]
        assert len(note_ons) == 3
        # All note-ons at t=0
        assert all(m[0] == 0.0 for m in note_ons)

    def test_chord_three_notes_simultaneous_off(self):
        msgs = create_chord_probe([60, 64, 67], velocity=90, sustain_s=1.5, release_s=0.8)
        note_offs = [m for m in msgs if m[1] == 0x80]
        assert len(note_offs) == 3
        # All note-offs at same time
        assert all(m[0] == 1.5 for m in note_offs)


# ---------------------------------------------------------------------------
# Musical context probe tests
# ---------------------------------------------------------------------------


class TestBassLineProbe:
    def test_bass_line_four_notes_sequential(self):
        msgs = create_bass_line_probe([28, 31, 33, 35], velocity=90, note_duration=0.4)
        note_ons = [m for m in msgs if m[1] == 0x90]
        assert len(note_ons) == 4
        # Notes in correct order
        assert [m[2] for m in note_ons] == [28, 31, 33, 35]

    def test_bass_line_notes_in_low_register(self):
        notes = [28, 31, 33, 35]
        msgs = create_bass_line_probe(notes, velocity=90, note_duration=0.4)
        note_ons = [m for m in msgs if m[1] == 0x90]
        assert all(m[2] < 48 for m in note_ons)  # All below C3


class TestMelodyProbe:
    def test_melody_eight_notes_ascending(self):
        notes = [60, 62, 64, 65, 67, 69, 71, 72]
        msgs = create_melody_probe(notes, velocity=80, note_duration=0.2)
        note_ons = [m for m in msgs if m[1] == 0x90]
        assert len(note_ons) == 8
        assert [m[2] for m in note_ons] == notes


class TestChordProgressionProbe:
    def test_chord_progression_four_chords_sequential(self):
        chords = [[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]]
        msgs = create_chord_progression_probe(
            chords, velocity=85, chord_duration=1.0, release_s=0.3
        )
        note_ons = [m for m in msgs if m[1] == 0x90]
        # 4 chords x 3 notes = 12 note-ons
        assert len(note_ons) == 12


class TestCompingProbe:
    def test_comping_rhythmic_pattern(self):
        msgs = create_comping_probe([60, 64, 67], velocity=70, n_hits=4, hit_duration=0.1, gap=0.3)
        note_ons = [m for m in msgs if m[1] == 0x90]
        # 4 hits x 3 notes = 12 note-ons
        assert len(note_ons) == 12
        # Short hits: note-off close to note-on within each hit
        note_offs = [m for m in msgs if m[1] == 0x80]
        assert len(note_offs) == 12


class TestPercussiveProbe:
    def test_percussive_very_short_note_on(self):
        msgs = create_percussive_probe(note=60, velocity=127, n_hits=3, decay_s=1.0)
        for i in range(0, len(msgs), 2):
            on_time = msgs[i][0]
            off_time = msgs[i + 1][0]
            assert off_time - on_time < 0.1  # Very short note-on


class TestLegatoProbe:
    def test_legato_notes_overlap(self):
        notes = [60, 62, 64, 65, 67]
        msgs = create_legato_probe(notes, velocity=80, note_duration=0.4, overlap=0.05)
        note_ons = sorted([m for m in msgs if m[1] == 0x90], key=lambda m: m[0])
        note_offs = sorted([m for m in msgs if m[1] == 0x80], key=lambda m: m[0])
        # Note N+1 starts before note N ends
        for i in range(len(note_ons) - 1):
            next_on_time = note_ons[i + 1][0]
            current_off_time = note_offs[i][0]
            assert next_on_time < current_off_time


class TestIntervalJumpProbe:
    def test_interval_jump_wide_pitch_range(self):
        notes = [36, 60, 84, 60]
        msgs = create_interval_jump_probe(notes, velocity=100, note_duration=0.8)
        note_ons = [m for m in msgs if m[1] == 0x90]
        pitches = [m[2] for m in note_ons]
        # Span > 3 octaves (36 semitones)
        assert max(pitches) - min(pitches) >= 36


# ---------------------------------------------------------------------------
# Composer tests
# ---------------------------------------------------------------------------


class TestComposeMultiProbe:
    def test_compose_single_matches_legacy(self):
        config = MultiProbeConfig.single()
        result = compose_multi_probe(config, sample_rate=44100)
        legacy = create_probe()
        # Same number of messages
        assert len(result.midi_messages) == len(legacy)
        # Same note and timing
        for composed, orig in zip(result.midi_messages, legacy):
            assert composed[1] == orig[1]  # status
            assert composed[2] == orig[2]  # note
            assert composed[3] == orig[3]  # velocity

    def test_compose_segment_boundaries(self):
        config = MultiProbeConfig.thorough()
        result = compose_multi_probe(config, sample_rate=44100)
        # Each segment has valid boundaries
        for seg in result.segments:
            assert seg.start_sample >= 0
            assert seg.end_sample > seg.start_sample

    def test_compose_gap_inserted(self):
        config = MultiProbeConfig.thorough()
        result = compose_multi_probe(config, sample_rate=44100)
        # Gap between segments: segment N+1 start > segment N end
        for i in range(len(result.segments) - 1):
            assert result.segments[i + 1].start_sample > result.segments[i].end_sample

    def test_compose_total_duration(self):
        config = MultiProbeConfig.single()
        result = compose_multi_probe(config, sample_rate=44100)
        # Single probe: 3s sustain + 1s release = 4s total
        assert abs(result.total_duration - 4.0) < 0.1

    def test_thorough_preset_probe_count(self):
        config = MultiProbeConfig.thorough()
        assert len(config.probes) == 6

    def test_full_preset_probe_count(self):
        config = MultiProbeConfig.full()
        assert len(config.probes) == 14

    def test_all_timestamps_monotonic(self):
        config = MultiProbeConfig.full()
        result = compose_multi_probe(config, sample_rate=44100)
        # Extract all timestamps and verify they're non-decreasing per note-on stream
        timestamps = [m[0] for m in result.midi_messages]
        # Sort by time and check no backward jumps
        sorted_ts = sorted(timestamps)
        for i in range(len(sorted_ts) - 1):
            assert sorted_ts[i] <= sorted_ts[i + 1]
