"""FAISS-based nearest-neighbor index for Surge XT patch retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


class PriorIndex:
    """FAISS index mapping audio feature vectors to Surge XT patch paths.

    Uses IndexFlatIP (brute-force inner product) on L2-normalized vectors,
    which is equivalent to cosine similarity. At ~10k vectors this is
    sub-millisecond query time — no approximate search needed.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        self._dim = feature_dim
        self._index = faiss.IndexFlatIP(feature_dim)
        self._patch_paths: list[str] = []

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def feature_dim(self) -> int:
        return self._dim

    def add(self, features: np.ndarray, patch_paths: list[str]) -> None:
        """Add feature vectors with associated patch paths.

        Args:
            features: (n, feature_dim) float32 array, L2-normalized.
            patch_paths: List of n patch file paths.
        """
        if features.ndim == 1:
            features = features[np.newaxis, :]
        if features.shape[1] != self._dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self._dim}, got {features.shape[1]}"
            )
        if len(patch_paths) != features.shape[0]:
            raise ValueError(
                f"Count mismatch: {features.shape[0]} features vs {len(patch_paths)} paths"
            )
        features = np.ascontiguousarray(features, dtype=np.float32)
        self._index.add(features)
        self._patch_paths.extend(patch_paths)

    def query(self, target_features: np.ndarray, k: int = 5) -> list[dict]:
        """Find the k nearest patches to the target features.

        Args:
            target_features: (feature_dim,) float32 array, L2-normalized.
            k: Number of neighbors to return.

        Returns:
            List of dicts with 'path', 'distance', and 'rank' keys.
        """
        if target_features.ndim == 1:
            target_features = target_features[np.newaxis, :]
        target_features = np.ascontiguousarray(target_features, dtype=np.float32)

        k = min(k, self._index.ntotal)
        if k == 0:
            return []

        distances, indices = self._index.search(target_features, k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:
                results.append(
                    {
                        "path": self._patch_paths[idx],
                        "distance": float(dist),
                        "rank": rank,
                    }
                )
        return results

    def save(self, path: str | Path) -> None:
        """Save the index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        meta = {"dim": self._dim, "paths": self._patch_paths}
        (path / "metadata.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> PriorIndex:
        """Load a previously saved index."""
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        instance = cls(feature_dim=meta["dim"])
        instance._index = faiss.read_index(str(path / "index.faiss"))
        instance._patch_paths = meta["paths"]
        return instance
