"""Standardized MIDI probe set for training and evaluation.

Every synth patch — source or target — is rendered with the same probe set.
This produces a feature vector (N_probes x 512 features) that captures how
a patch behaves across octaves, velocities, and articulations.

Supports two modes:
- "thorough" (6 probes, 3072-dim) — fast, used during CMA-ES optimization
- "full" (14 probes, 7168-dim) — richer, used for training data generation
"""

from __future__ import annotations

import numpy as np

from synth2surge.audio.midi import compose_multi_probe
from synth2surge.config import MultiProbeConfig
from synth2surge.loss.features import FEATURES_PER_PROBE, extract_features

# Probe counts by mode
PROBES_BY_MODE = {"thorough": 6, "full": 14, "single": 1}

# Legacy constants for backward compatibility
N_STANDARD_PROBES = 6
STANDARD_FEATURE_DIM = N_STANDARD_PROBES * FEATURES_PER_PROBE  # 3072


def get_probe_count(mode: str = "full") -> int:
    """Return number of probes for a given mode."""
    return PROBES_BY_MODE.get(mode, 6)


def get_feature_dim_for_mode(mode: str = "full") -> int:
    """Return total feature dimension for a probe mode."""
    return get_probe_count(mode) * FEATURES_PER_PROBE


def get_standard_probe_config(mode: str = "thorough") -> MultiProbeConfig:
    """Return the probe configuration for the given mode.

    Args:
        mode: "thorough" (6 probes) or "full" (14 probes).
    """
    if mode == "full":
        return MultiProbeConfig.full()
    return MultiProbeConfig.thorough()


def extract_multi_probe_features(
    segments: list[np.ndarray],
    sr: int = 44100,
    feature_backend: str = "clap",
    n_probes: int | None = None,
) -> np.ndarray:
    """Extract per-segment features and concatenate.

    Args:
        segments: List of mono audio arrays, one per probe.
        sr: Sample rate.
        feature_backend: Feature extraction backend ("clap" or "mel-stats").
        n_probes: Expected number of probes. If None, uses len(segments).

    Returns:
        L2-normalized feature vector of shape (n_probes * 512,).
        If a segment is silent, its 512 dims are zeros.
    """
    if n_probes is None:
        n_probes = len(segments) if segments else N_STANDARD_PROBES

    all_features = []
    for seg in segments:
        feat = extract_features(seg, sr=sr, backend=feature_backend)
        all_features.append(feat)

    # Pad if fewer segments than expected
    while len(all_features) < n_probes:
        all_features.append(np.zeros(FEATURES_PER_PROBE, dtype=np.float32))

    # Truncate if more segments than expected
    all_features = all_features[:n_probes]

    combined = np.concatenate(all_features).astype(np.float32)

    # L2 normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 1e-10:
        combined /= norm

    return combined


def render_standard_features(
    host: object,
    sr: int = 44100,
    feature_backend: str = "clap",
    probe_mode: str = "thorough",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Render the standard probe set and extract the feature vector.

    Args:
        host: PluginHost instance (already loaded with desired params).
        sr: Sample rate.
        feature_backend: Feature extraction backend ("clap" or "mel-stats").
        probe_mode: "thorough" (6 probes) or "full" (14 probes).

    Returns:
        (features, segments) tuple.
        features: L2-normalized feature vector.
        segments: List of mono audio arrays (one per probe).
    """
    config = get_standard_probe_config(mode=probe_mode)
    n_probes = get_probe_count(probe_mode)
    multi_probe = compose_multi_probe(config, sample_rate=sr)

    # render_multi_probe returns (full_audio, [segment1, segment2, ...])
    _, segments = host.render_multi_probe(multi_probe)

    features = extract_multi_probe_features(
        segments, sr=sr, feature_backend=feature_backend, n_probes=n_probes
    )
    return features, segments
