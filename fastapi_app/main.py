import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .predictor import SpamPredictor
from .schemas import PredictItem, PredictRequest, PredictResponse


def resolve_model_path() -> str:
    configured = os.getenv("MODEL_PATH")
    if configured:
        return configured

    candidates = [
        Path("artifacts/models/sms_spam_best_pipeline"),
        Path("models/sms_spam_best_pipeline"),
        Path("/content/artifacts/models/sms_spam_best_pipeline"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Default to the local project artifact path when not explicitly set.
    return str(candidates[0])


MODEL_PATH = resolve_model_path()

predictor = SpamPredictor(model_path=MODEL_PATH)
startup_error: str | None = None

app = FastAPI(
    title="SMS Spam Detection API",
    description="FastAPI service for SMS spam prediction using a Spark PipelineModel.",
    version="0.1.0",
)


@app.on_event("startup")
def startup_event() -> None:
    global startup_error
    try:
        predictor.load()
        startup_error = None
    except Exception as exc:
        # Keep API alive so /health can report misconfiguration.
        startup_error = str(exc)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok" if predictor.model is not None else "degraded",
        "model_path": predictor.model_path,
        "model_loaded": predictor.model is not None,
        "startup_error": startup_error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages list must not be empty")

    try:
        if predictor.model is None:
            predictor.load()

        raw_predictions = predictor.predict(payload.messages)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    prediction_items = [PredictItem(**item) for item in raw_predictions]

    return PredictResponse(
        model_path=predictor.model_path,
        count=len(prediction_items),
        predictions=prediction_items,
    )
