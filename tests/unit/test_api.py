"""Unit tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from synth2surge.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestJobEndpoints:
    def test_get_nonexistent_job(self, client):
        response = client.get("/jobs/nonexistent")
        assert response.status_code == 404

    def test_cancel_nonexistent_job(self, client):
        response = client.post("/jobs/nonexistent/cancel")
        assert response.status_code == 404


class TestCaptureEndpoint:
    def test_capture_returns_job_id(self, client):
        response = client.post(
            "/capture",
            json={
                "plugin_path": "/nonexistent/plugin.vst3",
                "no_gui": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"


class TestOptimizeEndpoint:
    def test_optimize_returns_job_id(self, client):
        response = client.post(
            "/optimize",
            json={
                "target_audio_path": "/nonexistent/audio.wav",
                "trials_t1": 5,
                "trials_t2": 0,
                "trials_t3": 0,
                "stages": [1],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"


class TestPriorEndpoint:
    def test_prior_status(self, client):
        response = client.get("/prior/status")
        assert response.status_code == 200
        data = response.json()
        assert "built" in data
        assert "n_entries" in data
