"""Tests for data generator — basic logic tests without Surge XT."""

import pytest

from synth2surge.ml.experience_store import ExperienceStore


@pytest.mark.unit
class TestDataGeneratorConfig:
    """Test data generator configuration and helpers."""

    def test_random_midi_config(self):
        import random

        from synth2surge.ml.data_generator import _random_midi_config

        rng = random.Random(42)
        config = _random_midi_config(rng)

        assert 36 <= config.note <= 84
        assert 60 <= config.velocity <= 127
        assert 1.5 <= config.sustain_seconds <= 4.0
        assert 0.5 <= config.release_seconds <= 2.0

    def test_random_midi_config_reproducible(self):
        import random

        from synth2surge.ml.data_generator import _random_midi_config

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        c1 = _random_midi_config(rng1)
        c2 = _random_midi_config(rng2)

        assert c1.note == c2.note
        assert c1.velocity == c2.velocity

    def test_store_created_for_generator(self, tmp_path):
        store = ExperienceStore(tmp_path / "gen_test.db")
        assert store.count() == 0
        store.close()
