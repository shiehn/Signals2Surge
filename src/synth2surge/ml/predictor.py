"""Neural network models for parameter prediction.

Two architectures:
- FeatureMLP: Simple MLP on 512-dim audio features (+ optional MIDI conditioning)
- SpectrogramCNN: CNN on mel-spectrograms with per-tier prediction heads

Both map audio → [0,1] parameter predictions for Surge XT.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _check_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for ML features. Install with: uv pip install 'synth2surge[ml]'"
        )


class FeatureMLP(nn.Module):
    """MLP that maps audio features → synth parameters.

    At inference, the user provides only an audio file — no MIDI info.
    The model learns timbral characteristics from standardized multi-probe
    audio features (6 probes x 512 = 3072 dims by default).

    Input: feature_dim audio feature vector.
    Output: N-dim parameter vector in [0,1] via Sigmoid.
    """

    def __init__(self, n_params: int, feature_dim: int = 3072) -> None:
        _check_torch()
        super().__init__()
        self.feature_dim = feature_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_params),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (batch, feature_dim) audio features

        Returns:
            (batch, n_params) predicted parameters in [0,1]
        """
        return self.net(features)


class SpectrogramCNN(nn.Module):
    """CNN encoder on log-mel spectrograms with per-tier prediction heads.

    Takes only audio as input — no MIDI conditioning. At inference the user
    provides an audio file and expects a Surge XT preset back, so the model
    must learn timbre from audio alone.

    Input: log-mel spectrogram [batch, 1, 128, T]
    Output: concatenated per-tier parameter predictions in [0,1]
    """

    def __init__(self, tier_sizes: dict[int, int]) -> None:
        """
        Args:
            tier_sizes: {1: n_tier1_params, 2: n_tier2_params, 3: n_tier3_params}
        """
        _check_torch()
        super().__init__()
        self.tier_sizes = tier_sizes

        self.encoder = nn.Sequential(
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

        self.shared_proj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.heads = nn.ModuleDict()
        for tier, n_params in sorted(tier_sizes.items()):
            hidden = 128 if tier <= 2 else 64
            self.heads[str(tier)] = nn.Sequential(
                nn.Linear(256, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_params),
                nn.Sigmoid(),
            )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            mel_spec: (batch, 1, 128, T) log-mel spectrogram

        Returns:
            (batch, total_params) concatenated tier predictions in [0,1]
        """
        embedding = self.encoder(mel_spec)
        shared = self.shared_proj(embedding)

        tier_outputs = []
        for tier_key in sorted(self.heads.keys()):
            tier_outputs.append(self.heads[tier_key](shared))

        return torch.cat(tier_outputs, dim=-1)

    def predict_by_tier(self, mel_spec: torch.Tensor) -> dict[int, torch.Tensor]:
        """Forward pass returning per-tier predictions separately."""
        embedding = self.encoder(mel_spec)
        shared = self.shared_proj(embedding)

        return {
            int(tier_key): self.heads[tier_key](shared)
            for tier_key in sorted(self.heads.keys())
        }
