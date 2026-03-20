"""Unit tests for the MR-STFT loss function."""

import numpy as np
import pytest

from synth2surge.loss.mr_stft import (
    log_magnitude_distance,
    mr_stft_loss,
    multi_probe_loss,
    spectral_convergence,
)
from tests.factories import make_noise, make_sine_wave


class TestSpectralConvergence:
    """Tests for the spectral convergence sub-loss."""

    def test_identical_magnitudes_return_zero(self):
        mag = np.random.rand(100, 50).astype(np.float32)
        assert spectral_convergence(mag, mag) == pytest.approx(0.0, abs=1e-6)

    def test_zero_target_returns_inf(self):
        target = np.zeros((100, 50), dtype=np.float32)
        candidate = np.ones((100, 50), dtype=np.float32)
        assert spectral_convergence(target, candidate) == float("inf")

    def test_different_magnitudes_positive(self):
        target = np.ones((100, 50), dtype=np.float32)
        candidate = np.ones((100, 50), dtype=np.float32) * 2.0
        loss = spectral_convergence(target, candidate)
        assert loss > 0.0

    def test_more_different_means_higher_loss(self):
        target = np.ones((100, 50), dtype=np.float32)
        close = np.ones((100, 50), dtype=np.float32) * 1.1
        far = np.ones((100, 50), dtype=np.float32) * 3.0
        assert spectral_convergence(target, close) < spectral_convergence(target, far)


class TestLogMagnitudeDistance:
    """Tests for the log-magnitude distance sub-loss."""

    def test_identical_magnitudes_return_zero(self):
        mag = np.random.rand(100, 50).astype(np.float32) + 0.01
        assert log_magnitude_distance(mag, mag) == pytest.approx(0.0, abs=1e-6)

    def test_different_magnitudes_positive(self):
        target = np.ones((100, 50), dtype=np.float32)
        candidate = np.ones((100, 50), dtype=np.float32) * 2.0
        assert log_magnitude_distance(target, candidate) > 0.0

    def test_empty_magnitudes_return_zero(self):
        empty = np.array([], dtype=np.float32).reshape(0, 0)
        assert log_magnitude_distance(empty, empty) == 0.0

    def test_near_zero_values_handled(self):
        """Epsilon floor prevents log(0) errors."""
        target = np.zeros((10, 10), dtype=np.float32)
        candidate = np.ones((10, 10), dtype=np.float32)
        loss = log_magnitude_distance(target, candidate)
        assert np.isfinite(loss)
        assert loss > 0.0


class TestMrStftLoss:
    """Tests for the combined MR-STFT loss function."""

    def test_identical_signals_zero_loss(self):
        audio = make_sine_wave(440.0, duration=1.0)
        loss = mr_stft_loss(audio, audio)
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_silence_target_returns_inf(self):
        target = np.zeros(44100, dtype=np.float32)
        candidate = make_sine_wave(440.0, duration=1.0)
        assert mr_stft_loss(target, candidate) == float("inf")

    def test_loss_symmetry(self):
        a = make_sine_wave(440.0, duration=1.0)
        b = make_sine_wave(880.0, duration=1.0)
        assert mr_stft_loss(a, b) == pytest.approx(mr_stft_loss(b, a), rel=1e-5)

    def test_similar_frequencies_small_loss(self):
        target = make_sine_wave(440.0, duration=1.0)
        close = make_sine_wave(441.0, duration=1.0)
        far = make_sine_wave(880.0, duration=1.0)
        loss_close = mr_stft_loss(target, close)
        loss_far = mr_stft_loss(target, far)
        assert loss_close < loss_far

    def test_noise_vs_sine_high_loss(self):
        sine = make_sine_wave(440.0, duration=1.0)
        noise = make_noise(duration=1.0)
        loss = mr_stft_loss(sine, noise)
        assert loss > 1.0  # Noise should be very different from a sine

    def test_loss_decreases_with_interpolation(self):
        """Loss should decrease as candidate moves toward target."""
        target = make_sine_wave(440.0, duration=1.0)
        other = make_sine_wave(880.0, duration=1.0)

        losses = []
        for alpha in [1.0, 0.7, 0.3, 0.0]:
            candidate = alpha * other + (1 - alpha) * target
            losses.append(mr_stft_loss(target, candidate))

        # Losses should be monotonically non-increasing as we approach target
        for i in range(len(losses) - 1):
            assert losses[i] >= losses[i + 1] - 1e-5

    def test_different_fft_sizes(self):
        target = make_sine_wave(440.0, duration=1.0)
        candidate = make_sine_wave(660.0, duration=1.0)
        # Single resolution should still work
        loss_single = mr_stft_loss(target, candidate, fft_sizes=[1024])
        loss_multi = mr_stft_loss(target, candidate, fft_sizes=[2048, 1024, 512])
        assert loss_single > 0
        assert loss_multi > 0

    def test_different_length_signals_handled(self):
        short = make_sine_wave(440.0, duration=0.5)
        long = make_sine_wave(440.0, duration=2.0)
        loss = mr_stft_loss(short, long)
        assert np.isfinite(loss)
        assert loss > 0.0  # Different due to zero-padding

    def test_empty_signals(self):
        empty = np.array([], dtype=np.float32)
        assert mr_stft_loss(empty, empty) == 0.0

    def test_alpha_weighting(self):
        target = make_sine_wave(440.0, duration=1.0)
        candidate = make_sine_wave(880.0, duration=1.0)
        loss_alpha0 = mr_stft_loss(target, candidate, alpha=0.0)
        loss_alpha1 = mr_stft_loss(target, candidate, alpha=1.0)
        loss_alpha5 = mr_stft_loss(target, candidate, alpha=5.0)
        # Higher alpha = more weight on log-magnitude = higher total loss
        assert loss_alpha0 < loss_alpha1 < loss_alpha5


class TestMultiProbeLoss:
    """Tests for the weighted multi-probe loss function."""

    def test_multi_probe_identical_segments_zero(self):
        seg1 = make_sine_wave(440.0, duration=1.0)
        seg2 = make_sine_wave(880.0, duration=1.0)
        loss = multi_probe_loss(
            [seg1, seg2], [seg1, seg2], weights=[0.5, 0.5]
        )
        assert loss == pytest.approx(0.0, abs=1e-5)

    def test_multi_probe_weights_affect_result(self):
        target1 = make_sine_wave(440.0, duration=1.0)
        target2 = make_sine_wave(880.0, duration=1.0)
        cand1 = target1.copy()  # identical
        cand2 = make_noise(duration=1.0)  # very different

        # Equal weights
        loss_equal = multi_probe_loss(
            [target1, target2], [cand1, cand2], weights=[0.5, 0.5]
        )
        # Heavy weight on the mismatched segment
        loss_heavy_mismatch = multi_probe_loss(
            [target1, target2], [cand1, cand2], weights=[0.1, 0.9]
        )
        # Doubling weight on mismatch should increase loss
        assert loss_heavy_mismatch > loss_equal

    def test_multi_probe_single_segment_matches_mr_stft(self):
        target = make_sine_wave(440.0, duration=1.0)
        candidate = make_sine_wave(660.0, duration=1.0)
        single_loss = mr_stft_loss(target, candidate)
        multi_loss = multi_probe_loss([target], [candidate], weights=[1.0])
        assert multi_loss == pytest.approx(single_loss, rel=1e-5)

    def test_multi_probe_one_silent_segment_finite(self):
        """Other segments prevent inf when one segment has silent target."""
        silent = np.zeros(44100, dtype=np.float32)
        sine = make_sine_wave(440.0, duration=1.0)
        loss = multi_probe_loss(
            [silent, sine], [sine, sine], weights=[0.5, 0.5]
        )
        assert np.isfinite(loss)
