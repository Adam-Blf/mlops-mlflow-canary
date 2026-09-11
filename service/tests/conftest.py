"""Shared fixtures for the offline test suite.

No test here talks to a real MLflow server or reads a model from disk.
load_model_with_retry is monkeypatched wherever the app calls it, so a
stub model stands in for the real thing.
"""
from contextlib import contextmanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app import routes as routes_module
from app.mlflow_loader import LoadedModel, ModelLoadError

VALID_FEATURES = {
    "square_footage": 1500.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "age_years": 10.0,
    "distance_to_city_km": 5.0,
}


class StubModel:
    """A pyfunc-like stub: predict() returns a fixed value per row."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, frame):
        return np.array([self.value] * len(frame))


def make_stub_loader(value: float, version: str):
    def _loader(**kwargs):
        return LoadedModel(model=StubModel(value), resolved_version=version)

    return _loader


def make_failing_loader(message: str = "mlflow tracking server not reachable"):
    def _loader(**kwargs):
        raise ModelLoadError(message)

    return _loader


@pytest.fixture
def make_client(monkeypatch):
    """Returns a factory: make_client(probability, startup_should_fail).

    Use it as a context manager so the FastAPI lifespan (startup and
    shutdown) actually runs:

        with make_client(probability=1.0) as client:
            ...
    """

    @contextmanager
    def _make(probability: float = 0.9, startup_should_fail: bool = False):
        monkeypatch.setenv("MODEL_LOAD_RETRY_ATTEMPTS", "1")
        monkeypatch.setenv("MODEL_LOAD_RETRY_DELAY_SECONDS", "0")
        monkeypatch.setenv("CANARY_PROBABILITY_CURRENT", str(probability))

        startup_loader = (
            make_failing_loader() if startup_should_fail else make_stub_loader(100.0, "1")
        )
        monkeypatch.setattr(main_module, "load_model_with_retry", startup_loader)
        monkeypatch.setattr(
            routes_module, "load_model_with_retry", make_stub_loader(200.0, "2")
        )

        app = main_module.create_app()
        with TestClient(app) as test_client:
            yield test_client

    return _make


@pytest.fixture
def client(make_client):
    with make_client() as test_client:
        yield test_client
