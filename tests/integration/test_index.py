"""Unit and integration tests for the FAISS prior index."""

import numpy as np
import pytest

from synth2surge.prior.index import PriorIndex


class TestPriorIndex:
    def test_empty_index(self):
        idx = PriorIndex(feature_dim=128)
        assert idx.size == 0

    def test_add_and_query(self):
        idx = PriorIndex(feature_dim=128)
        # Add 10 normalized random vectors
        rng = np.random.default_rng(42)
        features = rng.standard_normal((10, 128)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        paths = [f"patch_{i}.xml" for i in range(10)]
        idx.add(features, paths)
        assert idx.size == 10

        # Query with the first vector — it should be nearest to itself
        results = idx.query(features[0], k=3)
        assert len(results) == 3
        assert results[0]["path"] == "patch_0.xml"
        assert results[0]["distance"] == pytest.approx(1.0, abs=0.01)  # cosine sim

    def test_query_returns_correct_k(self):
        idx = PriorIndex(feature_dim=64)
        features = np.random.randn(20, 64).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        paths = [f"p{i}" for i in range(20)]
        idx.add(features, paths)

        results = idx.query(features[0], k=5)
        assert len(results) == 5

    def test_query_k_larger_than_index(self):
        idx = PriorIndex(feature_dim=64)
        features = np.random.randn(3, 64).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        paths = ["a", "b", "c"]
        idx.add(features, paths)

        results = idx.query(features[0], k=10)
        assert len(results) == 3

    def test_query_empty_index(self):
        idx = PriorIndex(feature_dim=64)
        q = np.random.randn(64).astype(np.float32)
        results = idx.query(q, k=5)
        assert len(results) == 0

    def test_save_load_roundtrip(self, tmp_path):
        idx = PriorIndex(feature_dim=128)
        rng = np.random.default_rng(42)
        features = rng.standard_normal((10, 128)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        paths = [f"patch_{i}.xml" for i in range(10)]
        idx.add(features, paths)

        save_path = tmp_path / "test_index"
        idx.save(save_path)

        loaded = PriorIndex.load(save_path)
        assert loaded.size == 10
        assert loaded.feature_dim == 128

        # Query should return same results
        results_orig = idx.query(features[0], k=3)
        results_loaded = loaded.query(features[0], k=3)
        assert results_orig[0]["path"] == results_loaded[0]["path"]
        assert results_orig[0]["distance"] == pytest.approx(
            results_loaded[0]["distance"], abs=1e-5
        )

    def test_feature_dim_mismatch_raises(self):
        idx = PriorIndex(feature_dim=64)
        wrong_features = np.random.randn(5, 128).astype(np.float32)
        with pytest.raises(ValueError, match="Feature dim mismatch"):
            idx.add(wrong_features, ["a"] * 5)

    def test_count_mismatch_raises(self):
        idx = PriorIndex(feature_dim=64)
        features = np.random.randn(5, 64).astype(np.float32)
        with pytest.raises(ValueError, match="Count mismatch"):
            idx.add(features, ["a", "b"])  # Only 2 paths for 5 features

    def test_single_vector_add(self):
        idx = PriorIndex(feature_dim=64)
        feature = np.random.randn(64).astype(np.float32)
        feature /= np.linalg.norm(feature)
        idx.add(feature, ["single_patch.xml"])
        assert idx.size == 1
