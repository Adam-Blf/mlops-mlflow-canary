"""Pydantic v2 schemas for requests and responses.

The feature set below matches scripts/train.py exactly, names AND types.
The types are not cosmetic: MLflow enforces the model signature at predict
time and refuses to convert float64 to int64, so declaring bathrooms or
age_years as float here makes every real prediction fail with a 500 while
unit tests against a stub model still pass.
"""
from typing import Literal

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    square_footage: float = Field(gt=0, description="Living area in square feet")
    bedrooms: int = Field(ge=0, le=20)
    bathrooms: int = Field(ge=0, le=20)
    age_years: int = Field(ge=0, le=200)
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
