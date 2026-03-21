"""Feature extraction for FAISS indexing — produces fixed-length audio fingerprints.

Supports two modes:
- Hand-crafted: 512-dim mel-spectrogram statistics (default)
- Learned: 128-dim encoder embeddings (when a trained encoder is available)
"""

from __future__ import annotations

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


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


def get_feature_dim(multi_probe: bool = False) -> int:
    """Return the expected feature dimension for the current mode."""
    if multi_probe:
        from synth2surge.audio.standard_probes import STANDARD_FEATURE_DIM

        return STANDARD_FEATURE_DIM
    return 512


def extract_features_learned(
    audio: np.ndarray,
    encoder: object,
    sr: int = 44100,
    n_mels: int = 128,
) -> np.ndarray:
    """Extract features using a trained audio encoder.

    Returns a 128-dim L2-normalized embedding from the AudioEncoder.
    Falls back to hand-crafted features if the encoder fails.

    Args:
        audio: Mono audio signal (1-D float array).
        encoder: Trained AudioEncoder model.
        sr: Sample rate.
        n_mels: Number of mel bands.

    Returns:
        L2-normalized feature vector of shape (embed_dim,).
    """
    try:
        import torch

        from synth2surge.ml.hybrid_loss import _audio_to_mel_tensor

        mel_tensor = _audio_to_mel_tensor(audio, sr=sr, n_mels=n_mels)
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(mel_tensor).squeeze(0).numpy()
        return embedding.astype(np.float32)

    except Exception:
        logger.warning("Learned feature extraction failed, falling back to hand-crafted")
        return extract_features(audio, sr=sr, n_mels=n_mels)
