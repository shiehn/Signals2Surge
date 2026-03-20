"""Feature extraction for FAISS indexing — produces fixed-length audio fingerprints."""

from __future__ import annotations

import librosa
import numpy as np


def extract_features(
    audio: np.ndarray,
    sr: int = 44100,
    n_mels: int = 128,
    n_stats: int = 4,
) -> np.ndarray:
    """Extract a fixed-length feature vector from audio for similarity search.

    Computes mel-spectrogram statistics (mean, std, skewness, kurtosis) per band,
    producing a (n_mels * n_stats)-dimensional vector. L2-normalized for cosine
    similarity via FAISS inner product.

    Args:
        audio: Mono audio signal (1-D float array).
        sr: Sample rate.
        n_mels: Number of mel bands.
        n_stats: Number of statistics per band (4 = mean, std, skew, kurtosis).

    Returns:
        L2-normalized feature vector of shape (n_mels * n_stats,).
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
