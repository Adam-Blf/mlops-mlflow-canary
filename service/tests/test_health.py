"""Tests for GET /health: happy path and MLflow-unreachable-at-startup path."""
from tests.conftest import VALID_FEATURES


def test_health_ok_when_model_loaded_at_startup(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["current_model_loaded"] is True
    assert body["next_model_loaded"] is True


def test_health_degraded_when_mlflow_unreachable(make_client):
    with make_client(startup_should_fail=True) as client:
        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["current_model_loaded"] is False
        assert body["next_model_loaded"] is False


def test_predict_returns_503_when_no_model_loaded(make_client):
    with make_client(startup_should_fail=True) as client:
        response = client.post("/predict", json={"features": VALID_FEATURES})

        assert response.status_code == 503
