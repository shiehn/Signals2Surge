"""Feature extraction for audio similarity — produces fixed-length audio fingerprints.

Supports multiple backends:
- "mel-stats": 512-dim mel-spectrogram statistics (legacy)
- "clap": 512-dim CLAP embeddings (recommended — perceptual similarity)
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Feature dimension is always 512 per probe regardless of backend
FEATURES_PER_PROBE = 512


def extract_features(
    audio: np.ndarray,
    sr: int = 44100,
    backend: str = "clap",
    **kwargs,
) -> np.ndarray:
    """Extract a fixed-length feature vector from audio.

    Args:
        audio: Mono audio signal (1-D float array).
        sr: Sample rate.
        backend: Feature extraction backend ("clap" or "mel-stats").

    Returns:
        L2-normalized feature vector of shape (512,).
    """
    if backend == "clap":
        from synth2surge.loss.clap_features import extract_clap_features
        return extract_clap_features(audio, sr=sr)
    else:
        return _extract_mel_stat_features(audio, sr=sr, **kwargs)


def _extract_mel_stat_features(
    audio: np.ndarray,
    sr: int = 44100,
    n_mels: int = 128,
    n_stats: int = 4,
) -> np.ndarray:
    """Extract mel-spectrogram statistics features (legacy 512-dim).

    Computes mel-spectrogram statistics (mean, std, skewness, kurtosis) per band,
    producing a (n_mels * n_stats)-dimensional vector. L2-normalized for cosine
    similarity via FAISS inner product.
    """
    if len(audio) == 0 or np.sqrt(np.mean(audio**2)) < 1e-8:
        return np.zeros(n_mels * n_stats, dtype=np.float32)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = np.log(np.maximum(mel_spec, 1e-7))

    # Compute per-band statistics across time frames
    mean = np.mean(log_mel, axis=1)
    std = np.std(log_mel, axis=1)

    # Skewness
    centered = log_mel - mean[:, np.newaxis]
    std_safe = np.maximum(std, 1e-10)
    skew = np.mean((centered / std_safe[:, np.newaxis]) ** 3, axis=1)

    # Kurtosis (excess)
    kurt = np.mean((centered / std_safe[:, np.newaxis]) ** 4, axis=1) - 3.0

    features = np.concatenate([mean, std, skew, kurt]).astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(features)
    if norm > 1e-10:
        features /= norm

    return features


def get_feature_dim(multi_probe: bool = False, n_probes: int | None = None) -> int:
    """Return the expected feature dimension.

    Args:
        multi_probe: If True, return the multi-probe dimension.
        n_probes: Number of probes (default 6 for thorough, 14 for full).
    """
    if multi_probe:
        if n_probes is None:
            n_probes = 6  # default thorough mode
        return n_probes * FEATURES_PER_PROBE
    return FEATURES_PER_PROBE
