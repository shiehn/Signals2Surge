"""Unit tests for the feature extraction module."""

import numpy as np
import pytest

from synth2surge.loss.features import extract_features
from tests.factories import make_noise, make_silence, make_sine_wave


class TestExtractFeatures:
    """Tests for the FAISS feature vector extraction."""

    def test_output_shape(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio)
        assert features.shape == (512,)  # 128 mels * 4 stats

    def test_output_dtype(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio)
        assert features.dtype == np.float32

    def test_l2_normalized(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio)
        norm = np.linalg.norm(features)
        assert norm == pytest.approx(1.0, abs=1e-5)

    def test_deterministic(self):
        audio = make_sine_wave(440.0, duration=1.0)
        f1 = extract_features(audio)
        f2 = extract_features(audio)
        np.testing.assert_array_equal(f1, f2)

    def test_silence_returns_zeros(self):
        silence = make_silence(duration=1.0)
        features = extract_features(silence)
        np.testing.assert_array_equal(features, np.zeros(512, dtype=np.float32))

    def test_different_signals_different_features(self):
        sine = make_sine_wave(440.0, duration=1.0)
        noise = make_noise(duration=1.0)
        f_sine = extract_features(sine)
        f_noise = extract_features(noise)
        # Features should differ (cosine similarity < 1.0)
        similarity = np.dot(f_sine, f_noise)
        assert similarity < 0.99

    def test_similar_signals_similar_features(self):
        a = make_sine_wave(440.0, duration=1.0)
        b = make_sine_wave(441.0, duration=1.0)
        f_a = extract_features(a)
        f_b = extract_features(b)
        similarity = np.dot(f_a, f_b)
        assert similarity > 0.9  # Very similar tones should have similar features

    def test_custom_n_mels(self):
        audio = make_sine_wave(440.0, duration=1.0)
        features = extract_features(audio, n_mels=64)
        assert features.shape == (256,)  # 64 mels * 4 stats

    def test_empty_audio_returns_zeros(self):
        empty = np.array([], dtype=np.float32)
        features = extract_features(empty)
        np.testing.assert_array_equal(features, np.zeros(512, dtype=np.float32))
