"""Tests for ML predictor models — skips if PyTorch not installed."""

import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.unit
class TestFeatureMLP:
    def test_forward_shape_3072(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=280, feature_dim=3072)
        x = torch.randn(4, 3072)
        out = model(x)
        assert out.shape == (4, 280)

    def test_forward_shape_512(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=280, feature_dim=512)
        x = torch.randn(4, 512)
        out = model(x)
        assert out.shape == (4, 280)

    def test_output_range(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=100)
        x = torch.randn(8, 3072)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_sample(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=50)
        x = torch.randn(1, 3072)
        out = model(x)
        assert out.shape == (1, 50)

    def test_gradient_flows(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=50)
        x = torch.randn(4, 3072)
        target = torch.rand(4, 50)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


@pytest.mark.unit
class TestSpectrogramCNN:
    def test_forward_shape(self):
        from synth2surge.ml.predictor import SpectrogramCNN

        tier_sizes = {1: 45, 2: 60, 3: 175}
        model = SpectrogramCNN(tier_sizes=tier_sizes)

        mel = torch.randn(2, 1, 128, 344)
        out = model(mel)
        assert out.shape == (2, 45 + 60 + 175)

    def test_output_range(self):
        from synth2surge.ml.predictor import SpectrogramCNN

        model = SpectrogramCNN(tier_sizes={1: 10, 2: 20, 3: 30})
        mel = torch.randn(4, 1, 128, 344)
        out = model(mel)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_predict_by_tier(self):
        from synth2surge.ml.predictor import SpectrogramCNN

        tier_sizes = {1: 10, 2: 20, 3: 30}
        model = SpectrogramCNN(tier_sizes=tier_sizes)
        mel = torch.randn(2, 1, 128, 344)

        by_tier = model.predict_by_tier(mel)
        assert set(by_tier.keys()) == {1, 2, 3}
        assert by_tier[1].shape == (2, 10)
        assert by_tier[2].shape == (2, 20)
        assert by_tier[3].shape == (2, 30)

    def test_variable_length_input(self):
        """AdaptiveAvgPool handles different spectrogram lengths."""
        from synth2surge.ml.predictor import SpectrogramCNN

        model = SpectrogramCNN(tier_sizes={1: 5})
        for T in [100, 200, 344, 500]:
            mel = torch.randn(1, 1, 128, T)
            out = model(mel)
            assert out.shape == (1, 5)
