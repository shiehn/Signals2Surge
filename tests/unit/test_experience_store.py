"""Tests for the ExperienceStore — runs without Surge XT."""

import numpy as np
import pytest

from synth2surge.ml.experience_store import ExperienceStore


@pytest.fixture
def store(tmp_path):
    s = ExperienceStore(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def param_names():
    return [f"param_{i}" for i in range(10)]


def _fake_features(dim=3072):
    return np.random.randn(dim).astype(np.float32)


def _fake_params(n=10):
    return np.random.uniform(0, 1, n).astype(np.float32)


@pytest.mark.unit
class TestExperienceStore:
    def test_empty_store(self, store):
        assert store.count() == 0
        assert store.trial_count() == 0

    def test_log_and_retrieve_run(self, store, param_names):
        run_id = store.new_run_id()
        features = _fake_features()
        params = _fake_params(len(param_names))

        store.log_run(
            run_id=run_id,
            target_features=features,
            best_params=params,
            param_names=param_names,
            best_loss=1.5,
            total_trials=100,
            ground_truth_params=params,
            generation_mode="random",
        )

        assert store.count() == 1

        record = store.get_run(run_id)
        assert record is not None
        assert record.run_id == run_id
        assert record.best_loss == 1.5
        assert record.total_trials == 100
        assert record.generation_mode == "random"
        np.testing.assert_array_almost_equal(record.target_features, features, decimal=5)
        np.testing.assert_array_almost_equal(record.best_params, params, decimal=5)
        np.testing.assert_array_almost_equal(record.ground_truth_params, params, decimal=5)
        assert record.param_names == param_names

    def test_log_trial(self, store, param_names):
        run_id = store.new_run_id()
        features = _fake_features()
        params = _fake_params(len(param_names))

        store.log_run(
            run_id=run_id,
            target_features=features,
            best_params=params,
            param_names=param_names,
            best_loss=1.0,
            total_trials=5,
            generation_mode="random",
        )

        for i in range(5):
            store.log_trial(
                run_id, stage=1, trial_idx=i,
                params=_fake_params(len(param_names)), loss=2.0 - i * 0.1,
            )
        store.flush()

        assert store.trial_count(run_id) == 5
        assert store.trial_count() == 5

    def test_get_training_data(self, store, param_names):
        for i in range(5):
            store.log_run(
                run_id=store.new_run_id(),
                target_features=_fake_features(),
                best_params=_fake_params(len(param_names)),
                param_names=param_names,
                best_loss=float(i),
                total_trials=10,
                generation_mode="random",
            )

        features, params, names = store.get_training_data()
        assert features.shape == (5, 3072)
        assert params.shape == (5, len(param_names))
        assert names == param_names

    def test_get_training_data_with_max_loss(self, store, param_names):
        for i in range(5):
            store.log_run(
                run_id=store.new_run_id(),
                target_features=_fake_features(),
                best_params=_fake_params(len(param_names)),
                param_names=param_names,
                best_loss=float(i),
                total_trials=10,
                generation_mode="random",
            )

        features, params, _ = store.get_training_data(max_loss=2.5)
        assert features.shape[0] == 3  # losses 0, 1, 2

    def test_get_ground_truth_data(self, store, param_names):
        # One with ground truth
        store.log_run(
            run_id=store.new_run_id(),
            target_features=_fake_features(),
            best_params=_fake_params(len(param_names)),
            param_names=param_names,
            best_loss=1.0,
            total_trials=0,
            ground_truth_params=_fake_params(len(param_names)),
            generation_mode="random",
        )
        # One without
        store.log_run(
            run_id=store.new_run_id(),
            target_features=_fake_features(),
            best_params=_fake_params(len(param_names)),
            param_names=param_names,
            best_loss=2.0,
            total_trials=100,
            generation_mode="user",
        )

        features, gt_params, _ = store.get_ground_truth_data()
        assert features.shape[0] == 1
        assert gt_params.shape[0] == 1

    def test_summary(self, store, param_names):
        for i in range(3):
            store.log_run(
                run_id=store.new_run_id(),
                target_features=_fake_features(),
                best_params=_fake_params(len(param_names)),
                param_names=param_names,
                best_loss=float(i + 1),
                total_trials=10,
                ground_truth_params=_fake_params(len(param_names)) if i < 2 else None,
                generation_mode="random",
            )

        stats = store.summary()
        assert stats["total_runs"] == 3
        assert stats["runs_with_ground_truth"] == 2
        assert stats["best_loss_min"] == 1.0
        assert stats["best_loss_max"] == 3.0

    def test_model_versioning(self, store):
        store.log_model_version("v1", 50, 0.1, 0.15, "/path/to/model.pt")
        store.log_model_version("v2", 100, 0.05, 0.08, "/path/to/model2.pt")

        assert store.latest_model_version() == "v2"

    def test_empty_training_data(self, store):
        features, params, names = store.get_training_data()
        assert features.shape[0] == 0
        assert len(names) == 0

    def test_get_nonexistent_run(self, store):
        assert store.get_run("nonexistent") is None

    def test_context_manager(self, tmp_path):
        with ExperienceStore(tmp_path / "ctx.db") as store:
            assert store.count() == 0
