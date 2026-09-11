"""Tests for /update-model and /accept-next-model."""
from tests.conftest import VALID_FEATURES


def test_update_model_only_changes_the_next_slot(client):
    response = client.post("/update-model", json={"version": "3"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "updated"
    assert body["slot"] == "next"
    assert body["model_version"] == "2"

    status = client.get("/model-status").json()
    assert status["current"]["version"] == "1"
    assert status["next"]["version"] == "2"
    assert status["next"]["loaded"] is True


def test_accept_next_model_promotes_next_into_current(client):
    client.post("/update-model", json={"version": "3"})

    response = client.post("/accept-next-model")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "promoted"
    assert body["model_version"] == "2"

    status = client.get("/model-status").json()
    assert status["current"]["version"] == "2"
    assert status["next"]["version"] == "2"


def test_accept_next_model_without_prior_update_still_works(client):
    # current and next are already identical at startup
    response = client.post("/accept-next-model")

    assert response.status_code == 200
    status = client.get("/model-status").json()
    assert status["current"]["version"] == status["next"]["version"]


def test_predict_after_promotion_uses_promoted_model(client):
    client.post("/update-model", json={"version": "3"})
    client.post("/accept-next-model")

    response = client.post("/predict", json={"features": VALID_FEATURES})

    assert response.status_code == 200
    assert response.json()["model_version"] == "2"
