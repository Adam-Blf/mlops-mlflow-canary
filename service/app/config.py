"""Runtime configuration read from environment variables.

Every value has a safe default so the service can boot in a plain dev
setup, but every value can be overridden without touching the code.
"""
import os
from dataclasses import dataclass


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str
    model_name: str
    initial_model_version: str
    canary_probability_current: float
    load_retry_attempts: int
    load_retry_delay_seconds: float


def get_settings() -> Settings:
    return Settings(
        mlflow_tracking_uri=os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        ),
        model_name=os.environ.get("MLFLOW_MODEL_NAME", "house-price-regressor"),
        initial_model_version=os.environ.get("MLFLOW_MODEL_VERSION", "latest"),
        canary_probability_current=_get_float("CANARY_PROBABILITY_CURRENT", 0.9),
        load_retry_attempts=_get_int("MODEL_LOAD_RETRY_ATTEMPTS", 5),
        load_retry_delay_seconds=_get_float("MODEL_LOAD_RETRY_DELAY_SECONDS", 2.0),
    )
