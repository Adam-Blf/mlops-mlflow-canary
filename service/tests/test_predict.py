"""Automated test of the /predict endpoint (offline, stub model)."""
from tests.conftest import VALID_FEATURES


def test_predict_returns_a_prediction_for_valid_input(client):
    response = client.post("/predict", json={"features": VALID_FEATURES})

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["prediction"], float)
    assert body["served_by"] in ("current", "next")
    assert body["model_version"] in ("1", "2")


def test_predict_rejects_invalid_input(client):
    invalid_features = dict(VALID_FEATURES)
    invalid_features["square_footage"] = -10.0

    response = client.post("/predict", json={"features": invalid_features})

    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete_features = dict(VALID_FEATURES)
    del incomplete_features["bedrooms"]

    response = client.post("/predict", json={"features": incomplete_features})

    assert response.status_code == 422
