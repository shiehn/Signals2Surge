"""Tests for standardized multi-probe feature extraction."""

import numpy as np
import pytest

from synth2surge.audio.standard_probes import (
    STANDARD_FEATURE_DIM,
    extract_multi_probe_features,
    get_feature_dim_for_mode,
    get_probe_count,
    get_standard_probe_config,
)


@pytest.mark.unit
class TestStandardProbes:
    def test_config_has_6_probes_thorough(self):
        config = get_standard_probe_config(mode="thorough")
        assert len(config.probes) == 6

    def test_config_has_14_probes_full(self):
        config = get_standard_probe_config(mode="full")
        assert len(config.probes) == 14

    def test_feature_dim_thorough(self):
        assert STANDARD_FEATURE_DIM == 3072
        assert get_feature_dim_for_mode("thorough") == 3072

    def test_feature_dim_full(self):
        assert get_feature_dim_for_mode("full") == 7168

    def test_probe_counts(self):
        assert get_probe_count("thorough") == 6
        assert get_probe_count("full") == 14

    def test_extract_multi_probe_features_shape_mel_stats(self):
        sr = 44100
        segments = []
        for freq in [110, 440, 880, 440, 440, 440]:
            t = np.linspace(0, 1, sr, dtype=np.float32)
            segments.append(0.5 * np.sin(2 * np.pi * freq * t))

        features = extract_multi_probe_features(
            segments, sr=sr, feature_backend="mel-stats", n_probes=6
        )
        assert features.shape == (3072,)

    def test_features_l2_normalized(self):
        sr = 44100
        segments = [
            0.5 * np.sin(2 * np.pi * f * np.linspace(0, 1, sr, dtype=np.float32))
            for f in [110, 440, 880, 440, 440, 440]
        ]
        features = extract_multi_probe_features(
            segments, sr=sr, feature_backend="mel-stats", n_probes=6
        )
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 0.01

    def test_different_segments_different_features(self):
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)

        seg_low = [0.5 * np.sin(2 * np.pi * 110 * t)] * 6
        seg_high = [0.5 * np.sin(2 * np.pi * 4000 * t)] * 6

        feat_low = extract_multi_probe_features(
            seg_low, sr=sr, feature_backend="mel-stats", n_probes=6
        )
        feat_high = extract_multi_probe_features(
            seg_high, sr=sr, feature_backend="mel-stats", n_probes=6
        )

        cosine_sim = float(np.dot(feat_low, feat_high))
        assert cosine_sim < 0.99, f"Features should differ, cosine_sim={cosine_sim:.4f}"

    def test_padding_short_segment_list(self):
        sr = 44100
        t = np.linspace(0, 1, sr, dtype=np.float32)
        # Only 2 segments instead of 6 — should pad to n_probes
        segments = [0.5 * np.sin(2 * np.pi * 440 * t)] * 2
        features = extract_multi_probe_features(
            segments, sr=sr, feature_backend="mel-stats", n_probes=6
        )
        assert features.shape == (3072,)

    def test_empty_segments_returns_zeros(self):
        features = extract_multi_probe_features(
            [], sr=44100, feature_backend="mel-stats", n_probes=6
        )
        assert features.shape == (3072,)
        assert np.allclose(features, 0)
