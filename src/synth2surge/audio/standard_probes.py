"""Standardized MIDI probe set for training and evaluation.

Every synth patch — source or target — is rendered with the same 6 probes.
This produces a 3072-dim feature vector (6 probes x 512 features) that
captures how a patch behaves across octaves, velocities, and articulations.

Using the same probes for both sides makes source→target comparison
apples-to-apples.
"""

from __future__ import annotations

import numpy as np

from synth2surge.audio.midi import compose_multi_probe
from synth2surge.config import MultiProbeConfig
from synth2surge.loss.features import extract_features

# Number of probes in the standard set (thorough mode)
N_STANDARD_PROBES = 6
FEATURES_PER_PROBE = 512
STANDARD_FEATURE_DIM = N_STANDARD_PROBES * FEATURES_PER_PROBE  # 3072


def get_standard_probe_config() -> MultiProbeConfig:
    """Return the canonical probe configuration used for all training/evaluation.

    Uses the 'thorough' preset which tests:
    1. Low sustained (C2) — bass response
    2. Mid sustained (C4) — core timbre
    3. High sustained (C5) — brightness/air
    4. Velocity sweep (C4 at pp/mf/ff) — dynamic response
    5. Staccato (C4 short hits) — attack transients
    6. Chord (C4 major triad) — polyphony/phasing
    """
    return MultiProbeConfig.thorough()


def extract_multi_probe_features(
    segments: list[np.ndarray],
    sr: int = 44100,
) -> np.ndarray:
    """Extract per-segment features and concatenate.

    Args:
        segments: List of mono audio arrays, one per probe.
        sr: Sample rate.

    Returns:
        L2-normalized feature vector of shape (N_STANDARD_PROBES * 512,).
        If a segment is silent, its 512 dims are zeros.
    """
    all_features = []
    for seg in segments:
        feat = extract_features(seg, sr=sr)
        all_features.append(feat)

    # Pad if fewer segments than expected
    while len(all_features) < N_STANDARD_PROBES:
        all_features.append(np.zeros(FEATURES_PER_PROBE, dtype=np.float32))

    # Truncate if more segments than expected
    all_features = all_features[:N_STANDARD_PROBES]

    combined = np.concatenate(all_features).astype(np.float32)

    # L2 normalize the combined vector
    norm = np.linalg.norm(combined)
    if norm > 1e-10:
        combined /= norm

    return combined


def render_standard_features(
    host: object,
    sr: int = 44100,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Render the standard probe set and extract the 3072-dim feature vector.

    Args:
        host: PluginHost instance (already loaded with desired params).
        sr: Sample rate.

    Returns:
        (features_3072, segments) tuple.
        features_3072: L2-normalized (3072,) feature vector.
        segments: List of 6 mono audio arrays (one per probe).
    """
    config = get_standard_probe_config()
    multi_probe = compose_multi_probe(config, sample_rate=sr)

    # render_multi_probe returns (full_audio, [segment1, segment2, ...])
    _, segments = host.render_multi_probe(multi_probe)

    features = extract_multi_probe_features(segments, sr=sr)
    return features, segments
