"""Learned audio encoder for perceptual similarity.

Maps audio (as log-mel spectrogram) to a 128-dim L2-normalized embedding.
Trained with triplet margin ranking loss on optimization trial data.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AudioEncoder(nn.Module):
    """CNN that maps log-mel spectrograms to L2-normalized embeddings.

    Perceptually similar audio should produce nearby embeddings.
    Trained via triplet ranking loss against enriched MR-STFT scores.

    Input: [batch, 1, 128, T] log-mel spectrogram (T can vary)
    Output: [batch, 128] L2-normalized embedding
    """

    def __init__(self, embed_dim: int = 128) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for AudioEncoder")
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.proj = nn.Linear(256, embed_dim)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Encode a log-mel spectrogram to a normalized embedding.

        Args:
            mel_spec: (batch, 1, 128, T) log-mel spectrogram

        Returns:
            (batch, embed_dim) L2-normalized embedding
        """
        x = self.conv(mel_spec)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=-1)
