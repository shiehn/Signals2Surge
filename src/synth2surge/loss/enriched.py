"""Enriched loss function combining MR-STFT with complementary perceptual metrics.

Fills gaps in MR-STFT by adding:
- MFCC distance: timbral shape (vocal-tract-like filtering)
- Envelope correlation: amplitude contour (attack, sustain, release shape)
- Spectral centroid trajectory: brightness evolution over time
- Spectral flux: rate of spectral change (transients, modulation)
"""

from __future__ import annotations

import librosa
import numpy as np

from synth2surge.loss.mr_stft import mr_stft_loss


def mfcc_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
    n_mfcc: int = 13,
) -> float:
    """Mean L2 distance between MFCC coefficient trajectories.

    Captures timbral shape invariant to pitch and volume.
    """
    if len(target) == 0 or len(candidate) == 0:
        return float("inf")

    target_mfcc = librosa.feature.mfcc(y=target, sr=sr, n_mfcc=n_mfcc)
    cand_mfcc = librosa.feature.mfcc(y=candidate, sr=sr, n_mfcc=n_mfcc)

    # Align lengths
    min_frames = min(target_mfcc.shape[1], cand_mfcc.shape[1])
    if min_frames == 0:
        return float("inf")

    target_mfcc = target_mfcc[:, :min_frames]
    cand_mfcc = cand_mfcc[:, :min_frames]

    return float(np.mean(np.sqrt(np.sum((target_mfcc - cand_mfcc) ** 2, axis=0))))


def envelope_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> float:
    """1 - correlation between amplitude envelopes.

    Captures attack/sustain/release shape differences.
    Returns 0 for identical envelopes, ~2 for anti-correlated.
    """
    if len(target) == 0 or len(candidate) == 0:
        return float("inf")

    target_rms = librosa.feature.rms(y=target, frame_length=frame_length, hop_length=hop_length)[0]
    cand_rms = librosa.feature.rms(
        y=candidate, frame_length=frame_length, hop_length=hop_length
    )[0]

    min_frames = min(len(target_rms), len(cand_rms))
    if min_frames < 2:
        return float("inf")

    target_rms = target_rms[:min_frames]
    cand_rms = cand_rms[:min_frames]

    # Correlation coefficient
    t_mean = np.mean(target_rms)
    c_mean = np.mean(cand_rms)
    t_std = np.std(target_rms)
    c_std = np.std(cand_rms)

    if t_std < 1e-10 or c_std < 1e-10:
        # One or both are constant — compare means
        return float(abs(t_mean - c_mean) / max(t_mean, c_mean, 1e-10))

    correlation = float(np.mean((target_rms - t_mean) * (cand_rms - c_mean)) / (t_std * c_std))
    return float(1.0 - correlation)


def centroid_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
) -> float:
    """Mean absolute difference of spectral centroid trajectories.

    Captures brightness evolution differences over time.
    Normalized by Nyquist frequency.
    """
    if len(target) == 0 or len(candidate) == 0:
        return float("inf")

    target_cent = librosa.feature.spectral_centroid(y=target, sr=sr)[0]
    cand_cent = librosa.feature.spectral_centroid(y=candidate, sr=sr)[0]

    min_frames = min(len(target_cent), len(cand_cent))
    if min_frames == 0:
        return float("inf")

    target_cent = target_cent[:min_frames]
    cand_cent = cand_cent[:min_frames]

    nyquist = sr / 2.0
    return float(np.mean(np.abs(target_cent - cand_cent)) / nyquist)


def spectral_flux_distance(
    target: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
) -> float:
    """Mean absolute difference of spectral flux (onset strength).

    Captures differences in temporal texture (transients, modulation rate).
    """
    if len(target) == 0 or len(candidate) == 0:
        return float("inf")

    target_flux = librosa.onset.onset_strength(y=target, sr=sr)
    cand_flux = librosa.onset.onset_strength(y=candidate, sr=sr)

    min_frames = min(len(target_flux), len(cand_flux))
    if min_frames == 0:
        return float("inf")

    target_flux = target_flux[:min_frames]
    cand_flux = cand_flux[:min_frames]

    # Normalize by max to get relative difference
    max_flux = max(np.max(target_flux), np.max(cand_flux), 1e-10)
    return float(np.mean(np.abs(target_flux - cand_flux)) / max_flux)


def enriched_loss(
    target: np.ndarray,
    candidate: np.ndarray,
    sr: int = 44100,
    *,
    w_stft: float = 0.6,
    w_mfcc: float = 0.15,
    w_envelope: float = 0.1,
    w_centroid: float = 0.1,
    w_flux: float = 0.05,
) -> float:
    """Compute enriched loss combining MR-STFT with perceptual metrics.

    Returns a weighted sum of:
    - MR-STFT loss (spectral convergence + log-magnitude)
    - MFCC distance (timbral shape)
    - Envelope distance (amplitude contour)
    - Spectral centroid distance (brightness trajectory)
    - Spectral flux distance (temporal texture)

    All component losses are designed to be in similar ranges (0-2 typical).
    """
    stft = mr_stft_loss(target, candidate)

    mfcc = mfcc_distance(target, candidate, sr=sr)
    env = envelope_distance(target, candidate, sr=sr)
    cent = centroid_distance(target, candidate, sr=sr)
    flux = spectral_flux_distance(target, candidate, sr=sr)

    # Clamp individual components to avoid inf contaminating the sum
    components = [
        (w_stft, stft),
        (w_mfcc, min(mfcc, 100.0)),
        (w_envelope, min(env, 2.0)),
        (w_centroid, min(cent, 2.0)),
        (w_flux, min(flux, 2.0)),
    ]

    total = sum(w * v for w, v in components)

    if not np.isfinite(total):
        return 1e6

    return float(total)
