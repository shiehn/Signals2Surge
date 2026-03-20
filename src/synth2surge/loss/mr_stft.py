"""Multi-Resolution STFT loss function for perceptual audio comparison.

Computes a weighted sum of spectral convergence and log-magnitude distance
across multiple FFT resolutions. This captures both large spectral peaks
(via spectral convergence) and quieter details (via log-magnitude distance).

Reference: https://arxiv.org/abs/1910.11480
"""

from __future__ import annotations

import librosa
import numpy as np


def _compute_magnitude(
    audio: np.ndarray,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Compute the magnitude spectrogram via STFT."""
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    return np.abs(stft)


def spectral_convergence(target_mag: np.ndarray, candidate_mag: np.ndarray) -> float:
    """Spectral convergence loss — focuses on large spectral peaks.

    L_sc = ||target_mag - candidate_mag||_F / ||target_mag||_F
    """
    target_norm = np.linalg.norm(target_mag, ord="fro")
    if target_norm < 1e-10:
        return float("inf")
    diff_norm = np.linalg.norm(target_mag - candidate_mag, ord="fro")
    return float(diff_norm / target_norm)


def log_magnitude_distance(
    target_mag: np.ndarray,
    candidate_mag: np.ndarray,
    epsilon: float = 1e-7,
) -> float:
    """Log-magnitude distance — captures quieter details and noise floors.

    L_mag = (1/N) * ||log(target_mag) - log(candidate_mag)||_1
    """
    log_target = np.log(np.maximum(target_mag, epsilon))
    log_candidate = np.log(np.maximum(candidate_mag, epsilon))
    n = target_mag.size
    if n == 0:
        return 0.0
    return float(np.sum(np.abs(log_target - log_candidate)) / n)


def mr_stft_loss(
    target_audio: np.ndarray,
    candidate_audio: np.ndarray,
    fft_sizes: list[int] | None = None,
    hop_divisor: int = 4,
    alpha: float = 1.0,
    epsilon: float = 1e-7,
) -> float:
    """Multi-Resolution STFT loss.

    L = (1/M) * sum_m(L_sc_m + alpha * L_mag_m)

    Args:
        target_audio: Reference audio signal (1-D float array).
        candidate_audio: Candidate audio signal (1-D float array).
        fft_sizes: List of FFT sizes for multi-resolution analysis.
        hop_divisor: Hop length = fft_size // hop_divisor.
        alpha: Weight for the log-magnitude term.
        epsilon: Floor value for log computation.

    Returns:
        Combined MR-STFT loss (lower = more similar).
    """
    if fft_sizes is None:
        fft_sizes = [2048, 1024, 512]

    # Ensure same length by zero-padding the shorter signal
    max_len = max(len(target_audio), len(candidate_audio))
    if max_len == 0:
        return 0.0

    target = np.zeros(max_len, dtype=np.float32)
    candidate = np.zeros(max_len, dtype=np.float32)
    target[: len(target_audio)] = target_audio
    candidate[: len(candidate_audio)] = candidate_audio

    # Check for silence in target
    if np.sqrt(np.mean(target**2)) < 1e-8:
        return float("inf")

    total_loss = 0.0
    for n_fft in fft_sizes:
        hop_length = n_fft // hop_divisor
        target_mag = _compute_magnitude(target, n_fft, hop_length)
        candidate_mag = _compute_magnitude(candidate, n_fft, hop_length)

        sc = spectral_convergence(target_mag, candidate_mag)
        lm = log_magnitude_distance(target_mag, candidate_mag, epsilon=epsilon)

        total_loss += sc + alpha * lm

    return total_loss / len(fft_sizes)


def multi_probe_loss(
    target_segments: list[np.ndarray],
    candidate_segments: list[np.ndarray],
    weights: list[float],
    fft_sizes: list[int] | None = None,
    hop_divisor: int = 4,
    alpha: float = 1.0,
    epsilon: float = 1e-7,
) -> float:
    """Weighted multi-probe loss across multiple audio segments.

    Computes mr_stft_loss() for each segment pair independently and returns
    the weighted average. Returns inf only if ALL segments return inf.

    Args:
        target_segments: List of target audio segments (1-D arrays).
        candidate_segments: List of candidate audio segments (1-D arrays).
        weights: Per-segment weights.
        fft_sizes: FFT sizes for MR-STFT (passed through to mr_stft_loss).
        hop_divisor: Hop length divisor (passed through).
        alpha: Log-magnitude weight (passed through).
        epsilon: Log floor (passed through).

    Returns:
        Weighted average loss (lower = more similar).
    """
    total_weight = 0.0
    weighted_loss = 0.0
    all_inf = True

    for target_seg, cand_seg, w in zip(target_segments, candidate_segments, weights):
        loss = mr_stft_loss(
            target_seg, cand_seg,
            fft_sizes=fft_sizes,
            hop_divisor=hop_divisor,
            alpha=alpha,
            epsilon=epsilon,
        )
        if np.isfinite(loss):
            weighted_loss += w * loss
            total_weight += w
            all_inf = False

    if all_inf:
        return float("inf")

    return weighted_loss / total_weight
