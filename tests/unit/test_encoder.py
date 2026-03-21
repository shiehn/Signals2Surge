"""Tests for the audio encoder — skips if PyTorch not installed."""

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.unit
class TestAudioEncoder:
    def test_forward_shape(self):
        from synth2surge.ml.encoder import AudioEncoder

        model = AudioEncoder(embed_dim=128)
        mel = torch.randn(2, 1, 128, 344)
        out = model(mel)
        assert out.shape == (2, 128)

    def test_output_normalized(self):
        from synth2surge.ml.encoder import AudioEncoder

        model = AudioEncoder(embed_dim=128)
        mel = torch.randn(4, 1, 128, 344)
        out = model(mel)

        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_variable_length(self):
        from synth2surge.ml.encoder import AudioEncoder

        model = AudioEncoder(embed_dim=64)
        for T in [100, 344, 500]:
            mel = torch.randn(1, 1, 128, T)
            out = model(mel)
            assert out.shape == (1, 64)

    def test_gradient_flows(self):
        from synth2surge.ml.encoder import AudioEncoder

        model = AudioEncoder(embed_dim=128)
        mel = torch.randn(2, 1, 128, 344)
        out = model(mel)
        loss = out.sum()
        loss.backward()

        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad


@pytest.mark.unit
class TestHybridLoss:
    def test_hybrid_loss_evaluator(self):
        from synth2surge.ml.encoder import AudioEncoder
        from synth2surge.ml.hybrid_loss import HybridLossEvaluator

        encoder = AudioEncoder(embed_dim=128)
        target = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)

        evaluator = HybridLossEvaluator(encoder, target, alpha=0.1)

        # Same audio should give low loss
        loss_same = evaluator(target)
        assert loss_same < 1.0

        # Different audio should give higher loss
        noise = np.random.randn(44100).astype(np.float32) * 0.5
        loss_diff = evaluator(noise)
        assert loss_diff > loss_same


@pytest.mark.unit
class TestLearnedFeatures:
    def test_extract_features_learned(self):
        from synth2surge.loss.features import extract_features_learned
        from synth2surge.ml.encoder import AudioEncoder

        encoder = AudioEncoder(embed_dim=128)
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)

        features = extract_features_learned(audio, encoder)
        assert features.shape == (128,)
        # Should be L2-normalized
        assert abs(np.linalg.norm(features) - 1.0) < 0.01
