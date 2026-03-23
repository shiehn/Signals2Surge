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
    audio features.

    Input: feature_dim audio feature vector.
    Output: N-dim parameter vector in [0,1] via Sigmoid.
    """

    # Default hidden layer sizes for different feature dimensions
    _DEFAULT_HIDDEN = {
        512: [512, 256],
        3072: [1024, 512],
        7168: [2048, 1024, 512],
    }

    def __init__(
        self,
        n_params: int,
        feature_dim: int = 7168,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        _check_torch()
        super().__init__()
        self.feature_dim = feature_dim

        if hidden_dims is None:
            hidden_dims = self._DEFAULT_HIDDEN.get(
                feature_dim, [2048, 1024, 512]
            )
        self.hidden_dims = list(hidden_dims)

        layers: list[nn.Module] = []
        prev_dim = feature_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.extend([
            nn.Linear(prev_dim, n_params),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (batch, feature_dim) audio features

        Returns:
            (batch, n_params) predicted parameters in [0,1]
        """
        return self.net(features)
