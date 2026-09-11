"""HTTP routes: predict, canary control, health and status.

Routes stay thin. They validate input via Pydantic, delegate routing
decisions to CanaryState, and delegate model loading to mlflow_loader.
Tests monkeypatch load_model_with_retry at module level to run without
a real MLflow server.
"""
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from app.canary_state import CanaryState
from app.config import Settings
from app.mlflow_loader import ModelLoadError, load_model_with_retry
from app.schemas import (
    AcceptNextResponse,
    HealthResponse,
    ModelSlotStatus,
    ModelStatusResponse,
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)

logger = logging.getLogger("mlops_service.routes")
router = APIRouter()


def get_state(request: Request) -> CanaryState:
    return request.app.state.canary_state


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = get_state(request)
    if state.current.loaded and state.next.loaded:
        return HealthResponse(
            status="ok",
            current_model_loaded=True,
            next_model_loaded=True,
            detail="both model slots are loaded",
        )
    return HealthResponse(
        status="degraded",
        current_model_loaded=state.current.loaded,
        next_model_loaded=state.next.loaded,
        detail=(
            "no model loaded yet, MLflow tracking server may not be "
            "reachable, call /update-model once it is"
        ),
    )


@router.get("/model-status", response_model=ModelStatusResponse)
def model_status(request: Request) -> ModelStatusResponse:
    state = get_state(request)
    return ModelStatusResponse(
        canary_probability_current=state.probability_current,
        current=ModelSlotStatus(
            version=state.current.version,
            loaded=state.current.loaded,
            request_count=state.current.request_count,
        ),
        next=ModelSlotStatus(
            version=state.next.version,
            loaded=state.next.loaded,
            request_count=state.next.request_count,
        ),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    state = get_state(request)
    slot_name = state.choose_slot()
    slot = state.slot(slot_name)

    if not slot.loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                f"the '{slot_name}' model slot is not loaded yet, "
                "check /health and /model-status"
            ),
        )

    frame = pd.DataFrame([body.features.model_dump()])
    raw_prediction = slot.model.predict(frame)
    prediction = float(raw_prediction[0])

    state.record_request(slot_name)

    return PredictResponse(
        prediction=prediction,
        served_by=slot_name,
        model_version=slot.version or "unknown",
    )


@router.post("/update-model", response_model=UpdateModelResponse)
def update_model(
    request: Request, body: UpdateModelRequest
) -> UpdateModelResponse:
    state = get_state(request)
    settings = get_settings(request)

    try:
        loaded = load_model_with_retry(
            model_name=settings.model_name,
            version=body.version,
            tracking_uri=settings.mlflow_tracking_uri,
            attempts=settings.load_retry_attempts,
            delay_seconds=settings.load_retry_delay_seconds,
        )
    except ModelLoadError as exc:
        logger.error("update-model failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    state.set_next(loaded.model, loaded.resolved_version)

    return UpdateModelResponse(
        status="updated", slot="next", model_version=loaded.resolved_version
    )


@router.post("/accept-next-model", response_model=AcceptNextResponse)
def accept_next_model(request: Request) -> AcceptNextResponse:
    state = get_state(request)

    if not state.next.loaded:
        raise HTTPException(
            status_code=400,
            detail="the 'next' slot has no model loaded, nothing to promote",
        )

    state.promote_next()

    return AcceptNextResponse(
        status="promoted", model_version=state.current.version or "unknown"
    )
