"""Unit tests for MIDI probe generation."""

from synth2surge.audio.midi import create_probe, probe_duration
from synth2surge.config import MidiProbeConfig


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
