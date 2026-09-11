"""FastAPI application factory and startup wiring.

The model is loaded from MLflow when the ASGI server starts, never
from disk and never baked into the docker image. If MLflow is not
reachable yet, startup logs a warning and the service comes up in a
degraded state instead of crashing, /health reports that state.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.canary_state import CanaryState
from app.config import get_settings
from app.mlflow_loader import ModelLoadError, load_model_with_retry
from app.routes import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.canary_state = CanaryState(
        probability_current=settings.canary_probability_current
    )

    try:
        loaded = load_model_with_retry(
            model_name=settings.model_name,
            version=settings.initial_model_version,
            tracking_uri=settings.mlflow_tracking_uri,
            attempts=settings.load_retry_attempts,
            delay_seconds=settings.load_retry_delay_seconds,
        )
        app.state.canary_state.set_both(loaded.model, loaded.resolved_version)
        logger.info("startup: loaded model version %s", loaded.resolved_version)
    except ModelLoadError as exc:
        logger.warning(
            "startup: no model loaded (%s), service starts in degraded "
            "state, use /update-model then /accept-next-model once "
            "MLflow is reachable",
            exc,
        )

    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="House price canary service",
        description=(
            "Serves a house price regressor loaded from an MLflow Model "
            "Registry, with a current/next canary split."
        ),
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
