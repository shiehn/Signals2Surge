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

        model = FeatureMLP(n_params=100, feature_dim=3072)
        x = torch.randn(8, 3072)
        out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_single_sample(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=50, feature_dim=3072)
        x = torch.randn(1, 3072)
        out = model(x)
        assert out.shape == (1, 50)

    def test_gradient_flows(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=50, feature_dim=3072)
        x = torch.randn(4, 3072)
        target = torch.rand(4, 50)
        loss = ((model(x) - target) ** 2).mean()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_default_7168_dim(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=100)
        x = torch.randn(4, 7168)
        out = model(x)
        assert out.shape == (4, 100)
        assert model.hidden_dims == [2048, 1024, 512]

    def test_custom_hidden_dims(self):
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=50, feature_dim=1024, hidden_dims=[512, 256])
        x = torch.randn(2, 1024)
        out = model(x)
        assert out.shape == (2, 50)
        assert model.hidden_dims == [512, 256]
