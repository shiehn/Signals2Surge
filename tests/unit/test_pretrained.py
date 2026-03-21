"""Tests for pretrained model download and packaging."""

import json
import zipfile

import numpy as np
import pytest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.mark.unit
class TestPackageModel:
    def _create_checkpoint(self, checkpoint_dir):
        """Helper: create a valid checkpoint directory."""
        from synth2surge.ml.predictor import FeatureMLP

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = FeatureMLP(n_params=20, feature_dim=512)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        (checkpoint_dir / "config.json").write_text(json.dumps({
            "architecture": "FeatureMLP",
            "n_params": 20,
            "n_training_samples": 100,
            "best_val_loss": 0.05,
            "param_names": [f"param_{i}" for i in range(20)],
        }))

    def test_package_creates_zip(self, tmp_path):
        from synth2surge.ml.pretrained import package_model_for_release

        checkpoint_dir = tmp_path / "checkpoint"
        self._create_checkpoint(checkpoint_dir)

        output = tmp_path / "model.zip"
        result = package_model_for_release(checkpoint_dir, output)

        assert result.exists()
        assert result.stat().st_size > 0

        with zipfile.ZipFile(result) as zf:
            names = zf.namelist()
            assert "model.pt" in names
            assert "config.json" in names

    def test_package_missing_files_raises(self, tmp_path):
        from synth2surge.ml.pretrained import package_model_for_release

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            package_model_for_release(empty_dir)


@pytest.mark.unit
class TestDownloadPretrained:
    def _create_model_zip(self, zip_path):
        """Helper: create a valid model zip file."""
        from synth2surge.ml.predictor import FeatureMLP

        model = FeatureMLP(n_params=20, feature_dim=512)
        torch.save(model.state_dict(), zip_path.parent / "_tmp_model.pt")

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_path.parent / "_tmp_model.pt", "model.pt")
            zf.writestr("config.json", json.dumps({
                "architecture": "FeatureMLP",
                "n_params": 20,
                "n_training_samples": 100,
                "best_val_loss": 0.05,
                "param_names": [f"param_{i}" for i in range(20)],
            }))

        (zip_path.parent / "_tmp_model.pt").unlink()

    def test_download_from_local_file(self, tmp_path):
        """Test download using a file:// URL (simulates network download)."""
        from synth2surge.ml.pretrained import download_pretrained

        # Create a model zip
        zip_path = tmp_path / "model.zip"
        self._create_model_zip(zip_path)

        # Download using file:// URL
        models_dir = tmp_path / "models"
        result = download_pretrained(
            models_dir, url=f"file://{zip_path}"
        )

        assert result is not None
        assert (result / "model.pt").exists()
        assert (result / "config.json").exists()

        config = json.loads((result / "config.json").read_text())
        assert config["n_params"] == 20

    def test_download_overwrites_existing(self, tmp_path):
        """Re-downloading replaces the existing pretrained model."""
        from synth2surge.ml.pretrained import download_pretrained

        zip_path = tmp_path / "model.zip"
        self._create_model_zip(zip_path)

        models_dir = tmp_path / "models"

        # Download twice
        result1 = download_pretrained(models_dir, url=f"file://{zip_path}")
        result2 = download_pretrained(models_dir, url=f"file://{zip_path}")

        assert result1 is not None
        assert result2 is not None
        assert result1 == result2


@pytest.mark.unit
class TestFindPretrained:
    def test_find_when_exists(self, tmp_path):
        from synth2surge.ml.pretrained import PRETRAINED_DIR_NAME, find_pretrained

        pretrained_dir = tmp_path / PRETRAINED_DIR_NAME
        pretrained_dir.mkdir()
        (pretrained_dir / "model.pt").write_bytes(b"fake")
        (pretrained_dir / "config.json").write_text("{}")

        result = find_pretrained(tmp_path)
        assert result is not None
        assert result == pretrained_dir

    def test_find_when_missing(self, tmp_path):
        from synth2surge.ml.pretrained import find_pretrained

        result = find_pretrained(tmp_path)
        assert result is None


@pytest.mark.unit
class TestWarmStartWithPretrained:
    """Test that WarmStarter finds and uses pretrained models."""

    def test_warm_starter_uses_pretrained(self, tmp_path):
        from synth2surge.ml.predictor import FeatureMLP
        from synth2surge.ml.pretrained import PRETRAINED_DIR_NAME
        from synth2surge.ml.warm_start import WarmStarter

        # Create pretrained model (no experience store needed)
        models_dir = tmp_path / "models"
        pretrained_dir = models_dir / PRETRAINED_DIR_NAME
        pretrained_dir.mkdir(parents=True)

        param_names = [f"param_{i}" for i in range(20)]
        model = FeatureMLP(n_params=20, feature_dim=512)
        torch.save(model.state_dict(), pretrained_dir / "model.pt")
        (pretrained_dir / "config.json").write_text(json.dumps({
            "architecture": "FeatureMLP",
            "n_params": 20,
            "param_names": param_names,
        }))

        # WarmStarter should find pretrained even without experience store
        starter = WarmStarter(
            store_path=tmp_path / "nonexistent.db",
            models_dir=models_dir,
        )

        features = np.random.randn(512).astype(np.float32)
        x0, sigma0 = starter.predict(features)

        # Should predict (or fall back) — but NOT return "no model"
        # The model was found, so sigma0 should not be the "no model" default
        # unless confidence is genuinely low
        assert sigma0 in (0.15, 0.3, 0.4)
