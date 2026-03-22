"""Neural network model for parameter prediction.

FeatureMLP maps 3072-dim multi-probe audio features to [0,1] parameter
predictions for Surge XT.
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
