"""SQLite-based experience store for logging optimization runs and trials.

Stores (audio_features, parameters, loss) tuples from optimization runs,
providing the training data for the ML predictor and audio encoder.
"""

from __future__ import annotations

import io
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    timestamp           TEXT NOT NULL,
    target_features     BLOB NOT NULL,
    best_params         BLOB NOT NULL,
    ground_truth_params BLOB,
    param_names_json    TEXT NOT NULL,
    best_loss           REAL NOT NULL,
    total_trials        INTEGER NOT NULL,
    probe_mode          TEXT DEFAULT 'single',
    generation_mode     TEXT NOT NULL DEFAULT 'user',
    model_version       TEXT,
    midi_config_json    TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(run_id),
    stage     INTEGER NOT NULL,
    trial_idx INTEGER NOT NULL,
    params    BLOB NOT NULL,
    loss      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_versions (
    version_id      TEXT PRIMARY KEY,
    n_training_runs INTEGER,
    train_loss      REAL,
    val_loss        REAL,
    checkpoint_path TEXT,
    created_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_loss ON runs(best_loss);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id);
"""


def _array_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes for SQLite storage."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _blob_to_array(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy array."""
    buf = io.BytesIO(blob)
    return np.load(buf).astype(np.float32)


@dataclass
class RunRecord:
    """A single logged optimization run."""

    run_id: str
    timestamp: str
    target_features: np.ndarray
    best_params: np.ndarray
    ground_truth_params: np.ndarray | None
    param_names: list[str]
    best_loss: float
    total_trials: int
    probe_mode: str
    generation_mode: str
    model_version: str | None


class ExperienceStore:
    """Persistent store for optimization experience data."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ExperienceStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]

    def log_run(
        self,
        run_id: str,
        target_features: np.ndarray,
        best_params: np.ndarray,
        param_names: list[str],
        best_loss: float,
        total_trials: int,
        *,
        ground_truth_params: np.ndarray | None = None,
        probe_mode: str = "single",
        generation_mode: str = "user",
        model_version: str | None = None,
        midi_config_json: str | None = None,
    ) -> None:
        """Log a completed optimization run."""
        gt_blob = _array_to_blob(ground_truth_params) if ground_truth_params is not None else None
        self._conn.execute(
            """INSERT INTO runs (
                run_id, timestamp, target_features, best_params,
                ground_truth_params, param_names_json, best_loss,
                total_trials, probe_mode, generation_mode,
                model_version, midi_config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                datetime.now(timezone.utc).isoformat(),
                _array_to_blob(target_features),
                _array_to_blob(best_params),
                gt_blob,
                json.dumps(param_names),
                float(best_loss),
                int(total_trials),
                probe_mode,
                generation_mode,
                model_version,
                midi_config_json,
            ),
        )
        self._conn.commit()

    def log_trial(
        self,
        run_id: str,
        stage: int,
        trial_idx: int,
        params: np.ndarray,
        loss: float,
    ) -> None:
        """Log a single optimization trial."""
        self._conn.execute(
            "INSERT INTO trials (run_id, stage, trial_idx, params, loss) VALUES (?, ?, ?, ?, ?)",
            (run_id, stage, trial_idx, _array_to_blob(params), float(loss)),
        )
        # Commit in batches — caller can call flush() periodically
        if trial_idx % 50 == 0:
            self._conn.commit()

    def flush(self) -> None:
        """Commit any pending trial writes."""
        self._conn.commit()

    def count(self) -> int:
        """Number of completed runs."""
        row = self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0

    def trial_count(self, run_id: str | None = None) -> int:
        """Number of logged trials, optionally filtered by run."""
        if run_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM trials WHERE run_id = ?", (run_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM trials").fetchone()
        return row[0] if row else 0

    def get_training_data(
        self,
        max_loss: float | None = None,
        generation_mode: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load training data as (features_matrix, params_matrix, param_names).

        Returns:
            features: shape (N, 512) — audio features for each run
            params: shape (N, P) — best parameter vectors
            param_names: canonical parameter ordering from first run
        """
        query = "SELECT target_features, best_params, param_names_json, best_loss FROM runs"
        conditions = []
        values: list[object] = []

        if max_loss is not None:
            conditions.append("best_loss <= ?")
            values.append(max_loss)
        if generation_mode is not None:
            conditions.append("generation_mode = ?")
            values.append(generation_mode)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        rows = self._conn.execute(query, values).fetchall()
        if not rows:
            return np.empty((0, 512), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []

        features_list = []
        params_list = []
        param_names: list[str] = json.loads(rows[0][2])

        for feat_blob, params_blob, names_json, _ in rows:
            features_list.append(_blob_to_array(feat_blob))
            params_list.append(_blob_to_array(params_blob))

        return (
            np.stack(features_list),
            np.stack(params_list),
            param_names,
        )

    def get_ground_truth_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load only runs with ground truth params (self-play data).

        Returns:
            features: shape (N, 512)
            ground_truth_params: shape (N, P)
            param_names: canonical ordering
        """
        rows = self._conn.execute(
            "SELECT target_features, ground_truth_params, param_names_json "
            "FROM runs WHERE ground_truth_params IS NOT NULL"
        ).fetchall()

        if not rows:
            return np.empty((0, 512), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []

        features_list = []
        params_list = []
        param_names: list[str] = json.loads(rows[0][2])

        for feat_blob, gt_blob, _ in rows:
            features_list.append(_blob_to_array(feat_blob))
            params_list.append(_blob_to_array(gt_blob))

        return np.stack(features_list), np.stack(params_list), param_names

    def get_run(self, run_id: str) -> RunRecord | None:
        """Retrieve a single run by ID."""
        row = self._conn.execute(
            "SELECT run_id, timestamp, target_features, best_params, "
            "ground_truth_params, param_names_json, best_loss, total_trials, "
            "probe_mode, generation_mode, model_version "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return RunRecord(
            run_id=row[0],
            timestamp=row[1],
            target_features=_blob_to_array(row[2]),
            best_params=_blob_to_array(row[3]),
            ground_truth_params=_blob_to_array(row[4]) if row[4] else None,
            param_names=json.loads(row[5]),
            best_loss=row[6],
            total_trials=row[7],
            probe_mode=row[8],
            generation_mode=row[9],
            model_version=row[10],
        )

    def summary(self) -> dict:
        """Return a summary of the store contents."""
        run_count = self.count()
        trial_count = self.trial_count()

        stats: dict = {
            "total_runs": run_count,
            "total_trials": trial_count,
            "db_path": str(self._db_path),
        }

        if run_count > 0:
            row = self._conn.execute(
                "SELECT MIN(best_loss), AVG(best_loss), MAX(best_loss) FROM runs"
            ).fetchone()
            stats["best_loss_min"] = row[0]
            stats["best_loss_avg"] = row[1]
            stats["best_loss_max"] = row[2]

            row = self._conn.execute(
                "SELECT COUNT(*) FROM runs WHERE ground_truth_params IS NOT NULL"
            ).fetchone()
            stats["runs_with_ground_truth"] = row[0]

            row = self._conn.execute(
                "SELECT generation_mode, COUNT(*) FROM runs GROUP BY generation_mode"
            ).fetchone()
            modes = self._conn.execute(
                "SELECT generation_mode, COUNT(*) FROM runs GROUP BY generation_mode"
            ).fetchall()
            stats["by_mode"] = {mode: cnt for mode, cnt in modes}

        return stats

    def log_model_version(
        self,
        version_id: str,
        n_training_runs: int,
        train_loss: float,
        val_loss: float,
        checkpoint_path: str,
    ) -> None:
        """Record a trained model version."""
        self._conn.execute(
            """INSERT OR REPLACE INTO model_versions
               (version_id, n_training_runs, train_loss, val_loss, checkpoint_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                version_id,
                n_training_runs,
                train_loss,
                val_loss,
                checkpoint_path,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def latest_model_version(self) -> str | None:
        """Get the latest model version ID."""
        row = self._conn.execute(
            "SELECT version_id FROM model_versions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None
