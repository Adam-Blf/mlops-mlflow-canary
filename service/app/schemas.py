"""Pydantic v2 schemas for requests and responses.

The feature set below matches what scripts/train.py is expected to train
on: a small, explicit set of numeric housing features. Keeping the schema
here, in one place, is what lets /predict validate input before it ever
reaches the model.
"""
from typing import Literal

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    square_footage: float = Field(gt=0, description="Living area in square feet")
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: float = Field(ge=0, le=20)
    age_years: float = Field(ge=0, le=200)
    distance_to_city_km: float = Field(ge=0)


class PredictRequest(BaseModel):
    features: HouseFeatures


class PredictResponse(BaseModel):
    prediction: float
    served_by: Literal["current", "next"]
    model_version: str


class UpdateModelRequest(BaseModel):
    version: str = Field(
        description=(
            "Model Registry version or alias to load into the next slot, "
            "e.g. '3' or 'champion'."
        )
    )


class UpdateModelResponse(BaseModel):
    status: Literal["updated"]
    slot: Literal["next"]
    model_version: str


class AcceptNextResponse(BaseModel):
    status: Literal["promoted"]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    current_model_loaded: bool
    next_model_loaded: bool
    detail: str


class ModelSlotStatus(BaseModel):
    version: str | None
    loaded: bool
    request_count: int


class ModelStatusResponse(BaseModel):
    canary_probability_current: float
    current: ModelSlotStatus
    next: ModelSlotStatus
