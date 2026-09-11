"""Deterministic tests for the canary split, using p=0 and p=1."""
from tests.conftest import VALID_FEATURES


def test_probability_one_always_serves_current(make_client):
    with make_client(probability=1.0) as client:
        # diverge the two slots first, current stays "1", next becomes "2"
        client.post("/update-model", json={"version": "3"})

        for _ in range(5):
            response = client.post("/predict", json={"features": VALID_FEATURES})
            assert response.status_code == 200
            body = response.json()
            assert body["served_by"] == "current"
            assert body["model_version"] == "1"

        status = client.get("/model-status").json()
        assert status["current"]["request_count"] == 5
        assert status["next"]["request_count"] == 0


def test_probability_zero_always_serves_next(make_client):
    with make_client(probability=0.0) as client:
        # diverge the two slots first, current stays "1", next becomes "2"
        client.post("/update-model", json={"version": "3"})

        for _ in range(5):
            response = client.post("/predict", json={"features": VALID_FEATURES})
            assert response.status_code == 200
            body = response.json()
            assert body["served_by"] == "next"
            assert body["model_version"] == "2"

        status = client.get("/model-status").json()
        assert status["current"]["request_count"] == 0
        assert status["next"]["request_count"] == 5


def test_model_status_exposes_configured_probability(make_client):
    with make_client(probability=0.9) as client:
        status = client.get("/model-status").json()
        assert status["canary_probability_current"] == 0.9
