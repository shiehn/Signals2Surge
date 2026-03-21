"""Tests for standardized multi-probe feature extraction."""

import numpy as np
import pytest

from synth2surge.audio.standard_probes import (
    N_STANDARD_PROBES,
    STANDARD_FEATURE_DIM,
    extract_multi_probe_features,
    get_standard_probe_config,
)


@pytest.mark.unit
class TestStandardProbes:
    def test_config_has_6_probes(self):
        config = get_standard_probe_config()
        assert len(config.probes) == N_STANDARD_PROBES

    def test_feature_dim_is_3072(self):
        assert STANDARD_FEATURE_DIM == 3072

    def test_extract_multi_probe_features_shape(self):
        sr = 44100
        # Create 6 synthetic segments (1s each)
        segments = []
        for freq in [110, 440, 880, 440, 440, 440]:
            t = np.linspace(0, 1, sr, dtype=np.float32)
            segments.append(0.5 * np.sin(2 * np.pi * freq * t))

        features = extract_multi_probe_features(segments, sr=sr)
        assert features.shape == (STANDARD_FEATURE_DIM,)

    def test_features_l2_normalized(self):
        sr = 44100
        segments = [
            0.5 * np.sin(2 * np.pi * f * np.linspace(0, 1, sr, dtype=np.float32))
            for f in [110, 440, 880, 440, 440, 440]
        ]
        features = extract_multi_probe_features(segments, sr=sr)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_different_segments_different_features(self):
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)

        seg_low = [0.5 * np.sin(2 * np.pi * 110 * t)] * 6
        seg_high = [0.5 * np.sin(2 * np.pi * 4000 * t)] * 6

        feat_low = extract_multi_probe_features(seg_low, sr=sr)
        feat_high = extract_multi_probe_features(seg_high, sr=sr)

        # Should be different (L2-normalized so use cosine distance)
        cosine_sim = float(np.dot(feat_low, feat_high))
        assert cosine_sim < 0.99, f"Features should differ, cosine_sim={cosine_sim:.4f}"

    def test_padding_short_segment_list(self):
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        # Only 2 segments instead of 6
        segments = [0.5 * np.sin(2 * np.pi * 440 * t)] * 2
        features = extract_multi_probe_features(segments, sr=sr)
        assert features.shape == (STANDARD_FEATURE_DIM,)

    def test_empty_segments_returns_zeros(self):
        features = extract_multi_probe_features([], sr=44100)
        assert features.shape == (STANDARD_FEATURE_DIM,)
        assert np.allclose(features, 0)
