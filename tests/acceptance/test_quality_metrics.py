"""Acceptance tests for audio quality metrics."""

import numpy as np

from synth2surge.loss.features import extract_features
from synth2surge.loss.mr_stft import mr_stft_loss
from tests.factories import make_noise, make_sine_wave


class TestQualityMetrics:
    """Verify the loss function produces meaningful quality rankings."""

    def test_loss_ranking_order(self):
        """Loss should correctly rank: identical < similar < different < noise."""
        target = make_sine_wave(440.0, duration=1.0)
        similar = make_sine_wave(445.0, duration=1.0)  # 5Hz off
        different = make_sine_wave(880.0, duration=1.0)  # Octave off
        noise = make_noise(duration=1.0)

        loss_identical = mr_stft_loss(target, target)
        loss_similar = mr_stft_loss(target, similar)
        loss_different = mr_stft_loss(target, different)
        loss_noise = mr_stft_loss(target, noise)

        assert loss_identical < loss_similar < loss_different < loss_noise

    def test_feature_similarity_ranking(self):
        """Feature cosine similarity should rank similarly to loss."""
        target = make_sine_wave(440.0, duration=1.0)
        similar = make_sine_wave(445.0, duration=1.0)
        different = make_sine_wave(880.0, duration=1.0)

        f_target = extract_features(target)
        f_similar = extract_features(similar)
        f_different = extract_features(different)

        sim_close = float(np.dot(f_target, f_similar))
        sim_far = float(np.dot(f_target, f_different))

        assert sim_close > sim_far

    def test_spectral_centroid_preservation(self):
        """Two signals with the same fundamental should have similar spectral centroids."""
        import librosa

        a = make_sine_wave(440.0, duration=1.0)
        b = make_sine_wave(441.0, duration=1.0)

        cent_a = float(np.mean(librosa.feature.spectral_centroid(y=a, sr=44100)))
        cent_b = float(np.mean(librosa.feature.spectral_centroid(y=b, sr=44100)))

        assert abs(cent_a - cent_b) < 200  # Hz
