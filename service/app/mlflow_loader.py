"""Loading pyfunc models from an MLflow Model Registry, with retries.

This module is the only place that talks to MLflow. It never reads a
model from local disk or from the docker image: every model comes from
the tracking server at runtime, which is what lets the Dockerfile stay
free of any COPY of trained artifacts.
"""
import logging
import time
from dataclasses import dataclass

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logger = logging.getLogger("mlops_service.mlflow_loader")


class ModelLoadError(RuntimeError):
    """Raised when a model could not be loaded after all retries."""


@dataclass
class LoadedModel:
    model: object
    resolved_version: str


def resolve_version(client: MlflowClient, model_name: str, version: str) -> str:
    """Turn 'latest' into a concrete registry version number.

    Any other value (a numeric version or an alias) is passed through
    untouched, the actual load call is what validates it against MLflow.
    """
    if version != "latest":
        return version

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ModelLoadError(f"no registered version found for model '{model_name}'")

    latest = max(versions, key=lambda mv: int(mv.version))
    return latest.version


def build_model_uri(model_name: str, version: str) -> str:
    if version.isdigit():
        return f"models:/{model_name}/{version}"
    return f"models:/{model_name}@{version}"


def load_model_with_retry(
    model_name: str,
    version: str,
    tracking_uri: str,
    attempts: int,
    delay_seconds: float,
) -> LoadedModel:
    """Load a pyfunc model, retrying a short number of times.

    A missing or not-yet-ready MLflow server must not crash the service
    at startup, callers use the result (or the raised ModelLoadError) to
    decide whether to keep serving in a degraded state.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resolved = resolve_version(client, model_name, version)
            uri = build_model_uri(model_name, resolved)
            model = mlflow.pyfunc.load_model(uri)
            logger.info("loaded model %s (attempt %d/%d)", uri, attempt, attempts)
            return LoadedModel(model=model, resolved_version=resolved)
        except (MlflowException, OSError, ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "model load attempt %d/%d failed: %s", attempt, attempts, exc
            )
            if attempt < attempts:
                time.sleep(delay_seconds)

    raise ModelLoadError(
        f"could not load model '{model_name}' version '{version}' "
        f"after {attempts} attempts: {last_error}"
    )
