"""Unit tests for the feature extraction module."""

import numpy as np
import pytest

from synth2surge.loss.features import extract_features
from tests.factories import make_noise, make_silence, make_sine_wave

_has_clap = True
try:
    import laion_clap  # noqa: F401
except ImportError:
    try:
        import transformers  # noqa: F401
    except ImportError:
        _has_clap = False

requires_clap = pytest.mark.skipif(not _has_clap, reason="CLAP dependencies not installed")


class TestExtractFeatures:
    """Tests for audio feature extraction (both backends)."""

    @requires_clap
    def test_output_shape_clap(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio, backend="clap")
        assert features.shape == (512,)

    def test_output_shape_mel_stats(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio, backend="mel-stats")
        assert features.shape == (512,)  # 128 mels * 4 stats

    def test_output_dtype(self):
        audio = make_sine_wave(440.0, duration=1.0)
        for backend in ("clap", "mel-stats"):
            features = extract_features(audio, backend=backend)
            assert features.dtype == np.float32

    def test_l2_normalized(self):
        audio = make_sine_wave(440.0, duration=1.0)
        for backend in ("clap", "mel-stats"):
            features = extract_features(audio, backend=backend)
            norm = np.linalg.norm(features)
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_deterministic_mel_stats(self):
        audio = make_sine_wave(440.0, duration=1.0)
        f1 = extract_features(audio, backend="mel-stats")
        f2 = extract_features(audio, backend="mel-stats")
        np.testing.assert_array_equal(f1, f2)

    @requires_clap
    def test_silence_returns_zeros_clap(self):
        silence = make_silence(duration=1.0)
        features = extract_features(silence, backend="clap")
        np.testing.assert_array_equal(features, np.zeros(512, dtype=np.float32))

    def test_silence_returns_zeros_mel_stats(self):
        silence = make_silence(duration=1.0)
        features = extract_features(silence, backend="mel-stats")
        np.testing.assert_array_equal(features, np.zeros(512, dtype=np.float32))

    def test_different_signals_different_features(self):
        sine = make_sine_wave(440.0, duration=1.0)
        noise = make_noise(duration=1.0)
        for backend in ("clap", "mel-stats"):
            f_sine = extract_features(sine, backend=backend)
            f_noise = extract_features(noise, backend=backend)
            similarity = np.dot(f_sine, f_noise)
            assert similarity < 0.99

    def test_similar_signals_similar_features_mel_stats(self):
        a = make_sine_wave(440.0, duration=1.0)
        b = make_sine_wave(441.0, duration=1.0)
        f_a = extract_features(a, backend="mel-stats")
        f_b = extract_features(b, backend="mel-stats")
        similarity = np.dot(f_a, f_b)
        assert similarity > 0.9

    def test_custom_n_mels(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio, backend="mel-stats", n_mels=64)
        assert features.shape == (256,)  # 64 mels * 4 stats

    def test_empty_audio_returns_zeros(self):
        empty = np.array([], dtype=np.float32)
        for backend in ("clap", "mel-stats"):
            features = extract_features(empty, backend=backend)
            np.testing.assert_array_equal(features, np.zeros(512, dtype=np.float32))
