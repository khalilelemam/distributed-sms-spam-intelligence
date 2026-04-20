import os
from contextlib import asynccontextmanager
from pathlib import Path
import traceback
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from .predictor import SpamPredictor
from .schemas import PredictItem, PredictRequest, PredictResponse

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(APP_DIR / "models" / "sms_spam_best_pipeline")

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

predictor = SpamPredictor(model_path=MODEL_PATH)
startup_error: str | None = None
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global startup_error
    try:
        predictor.load()
        startup_error = None
    except Exception as exc:
        # Keep API alive so /health can report misconfiguration.
        startup_error = str(exc)
    yield


app = FastAPI(
    title="SMS Spam Detection API",
    description="FastAPI service for SMS spam prediction using a Spark PipelineModel.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # This prints the REAL error to your terminal logs
    print(f"ERROR: {exc}", file=sys.stderr)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "details": str(exc)},
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"startup_error": startup_error},
    )


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
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        if predictor.model is None:
            predictor.load()

        raw_predictions = predictor.predict([message])
        if not raw_predictions:
            raise RuntimeError("no prediction returned")
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    prediction_item = PredictItem(**raw_predictions[0])

    return PredictResponse(prediction=prediction_item)
