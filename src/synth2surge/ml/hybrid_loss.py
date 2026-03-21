"""Hybrid loss combining enriched MR-STFT with learned audio similarity.

The learned component is always secondary (alpha <= 0.3) to prevent
adversarial gaming. The enriched MR-STFT provides the grounded signal.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _audio_to_mel_tensor(
    audio: np.ndarray, sr: int = 44100, n_mels: int = 128
) -> "torch.Tensor":
    """Convert audio array to log-mel spectrogram tensor for the encoder."""
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    log_mel = np.log(np.maximum(mel, 1e-7))
    tensor = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor


class HybridLossEvaluator:
    """Evaluates hybrid loss: enriched MR-STFT + learned similarity.

    Pre-computes the target embedding once, then evaluates candidates
    against it with minimal overhead (~3ms per candidate).
    """

    def __init__(
        self,
        encoder: "torch.nn.Module",
        target_audio: np.ndarray,
        sr: int = 44100,
        alpha: float = 0.1,
        enriched_weights: dict[str, float] | None = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for HybridLossEvaluator")

        self._encoder = encoder
        self._sr = sr
        self._alpha = alpha
        self._enriched_weights = enriched_weights or {}
        self._target_audio = target_audio

        # Pre-compute and cache target embedding
        self._encoder.eval()
        target_mel = _audio_to_mel_tensor(target_audio, sr=sr)
        with torch.no_grad():
            self._target_embedding = self._encoder(target_mel).squeeze(0)

    def __call__(self, candidate_audio: np.ndarray) -> float:
        """Compute hybrid loss for a candidate audio signal.

        Args:
            candidate_audio: 1-D float32 mono audio.

        Returns:
            Hybrid loss value (lower = more similar).
        """
        from synth2surge.loss.enriched import enriched_loss

        # Enriched MR-STFT component
        e_loss = enriched_loss(
            self._target_audio, candidate_audio, self._sr, **self._enriched_weights
        )

        # Learned similarity component
        cand_mel = _audio_to_mel_tensor(candidate_audio, sr=self._sr)
        with torch.no_grad():
            cand_embedding = self._encoder(cand_mel).squeeze(0)
            # Cosine distance: 1 - cosine_similarity (range [0, 2])
            learned_dist = float(
                1.0 - torch.dot(self._target_embedding, cand_embedding).item()
            )

        # Scale learned distance to roughly match enriched loss range
        # Enriched loss is typically 0.1 - 50; learned_dist is 0 - 2
        scaled_learned = learned_dist * e_loss if e_loss > 0 else learned_dist

        hybrid = (1.0 - self._alpha) * e_loss + self._alpha * scaled_learned

        if not np.isfinite(hybrid):
            return 1e6

        return float(hybrid)
