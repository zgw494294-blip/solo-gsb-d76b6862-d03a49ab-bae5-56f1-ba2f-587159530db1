from typing import List

from pydantic import BaseModel, Field, field_validator, model_validator
import math


class Observation(BaseModel):
    """Single tide observation at integer minute t with finite level h."""

    t: int
    h: float

    @field_validator("h")
    @classmethod
    def h_must_be_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("h must be finite")
        return v


class FitRequest(BaseModel):
    """Request body for tidal harmonic fitting."""

    observations: List[Observation] = Field(..., min_length=20, max_length=500)
    periods: List[float] = Field(..., min_length=1, max_length=4)
    delta: float
    predictions: List[int] = Field(default_factory=list)

    @field_validator("periods")
    @classmethod
    def periods_positive_and_distinct(cls, v: List[float]) -> List[float]:
        if any(p <= 0 or not math.isfinite(p) for p in v):
            raise ValueError("periods must be positive and finite")
        if len(set(v)) != len(v):
            raise ValueError("periods must be distinct")
        return v

    @field_validator("delta")
    @classmethod
    def delta_positive(cls, v: float) -> float:
        if v <= 0 or not math.isfinite(v):
            raise ValueError("delta must be a positive finite number")
        return v

    @model_validator(mode="after")
    def observations_strictly_increasing(self) -> "FitRequest":
        ts = [o.t for o in self.observations]
        if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
            raise ValueError("observation t values must be strictly increasing")
        return self


class Coefficient(BaseModel):
    period: float
    a: float
    b: float


class Prediction(BaseModel):
    t: int
    h: float


class FitResponse(BaseModel):
    baseline: float
    coefficients: List[Coefficient]
    residuals: List[float]
    predictions: List[Prediction]
