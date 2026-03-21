"""Tests for enriched loss function — no external dependencies needed."""

import numpy as np
import pytest


def _make_sine(freq: float = 440.0, sr: int = 44100, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.unit
class TestEnrichedLossComponents:
    def test_mfcc_distance_identical(self):
        from synth2surge.loss.enriched import mfcc_distance

        audio = _make_sine(440.0)
        dist = mfcc_distance(audio, audio)
        assert dist < 0.01

    def test_mfcc_distance_different(self):
        from synth2surge.loss.enriched import mfcc_distance

        a = _make_sine(440.0)
        b = _make_sine(880.0)
        dist = mfcc_distance(a, b)
        assert dist > 0.1

    def test_envelope_distance_identical(self):
        from synth2surge.loss.enriched import envelope_distance

        audio = _make_sine(440.0)
        dist = envelope_distance(audio, audio)
        assert dist < 0.01

    def test_envelope_distance_different(self):
        from synth2surge.loss.enriched import envelope_distance

        sr = 44100
        # Sustained tone vs decaying tone
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        sustained = np.sin(2 * np.pi * 440 * t)
        decaying = sustained * np.exp(-3 * t)
        dist = envelope_distance(sustained, decaying, sr=sr)
        assert dist > 0.05

    def test_centroid_distance_identical(self):
        from synth2surge.loss.enriched import centroid_distance

        audio = _make_sine(440.0)
        dist = centroid_distance(audio, audio)
        assert dist < 0.01

    def test_centroid_distance_different_pitch(self):
        from synth2surge.loss.enriched import centroid_distance

        a = _make_sine(200.0)
        b = _make_sine(4000.0)
        dist = centroid_distance(a, b)
        assert dist > 0.05

    def test_spectral_flux_identical(self):
        from synth2surge.loss.enriched import spectral_flux_distance

        audio = _make_sine(440.0)
        dist = spectral_flux_distance(audio, audio)
        assert dist < 0.01

    def test_enriched_loss_identical(self):
        from synth2surge.loss.enriched import enriched_loss

        audio = _make_sine(440.0)
        loss = enriched_loss(audio, audio)
        assert loss < 0.5

    def test_enriched_loss_different(self):
        from synth2surge.loss.enriched import enriched_loss

        a = _make_sine(440.0)
        noise = np.random.randn(len(a)).astype(np.float32) * 0.5
        loss = enriched_loss(a, noise)
        assert loss > 1.0

    def test_enriched_loss_very_different(self):
        from synth2surge.loss.enriched import enriched_loss

        a = _make_sine(440.0)
        noise = np.random.randn(len(a)).astype(np.float32) * 0.3
        loss = enriched_loss(a, noise)
        assert loss > 0.5  # Should be meaningfully positive for dissimilar audio

    def test_enriched_loss_custom_weights(self):
        from synth2surge.loss.enriched import enriched_loss

        a = _make_sine(440.0)
        b = _make_sine(880.0)

        # Pure STFT
        loss_stft = enriched_loss(
            a, b, w_stft=1.0, w_mfcc=0.0, w_envelope=0.0, w_centroid=0.0, w_flux=0.0
        )
        # Pure MFCC
        loss_mfcc = enriched_loss(
            a, b, w_stft=0.0, w_mfcc=1.0, w_envelope=0.0, w_centroid=0.0, w_flux=0.0
        )

        # Both should be positive
        assert loss_stft > 0
        assert loss_mfcc > 0
