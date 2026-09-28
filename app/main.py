from fastapi import FastAPI, HTTPException

from .fitting import RankDeficientError, fit_tide
from .models import FitRequest, FitResponse

app = FastAPI(title="Tide Harmonic Fitting API", version="1.0.0")


@app.post("/v1/tide/fit", response_model=FitResponse)
def tide_fit(req: FitRequest):
    try:
        result = fit_tide(
            observations=req.observations,
            periods=req.periods,
            delta=req.delta,
            predictions=req.predictions,
        )
    except RankDeficientError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result
