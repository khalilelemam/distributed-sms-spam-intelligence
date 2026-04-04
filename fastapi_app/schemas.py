from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    messages: list[str] = Field(..., min_length=1)


class PredictItem(BaseModel):
    message: str
    prediction: int = Field(..., description="1 means spam, 0 means ham")
    label: str
    spam_score: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    model_path: str
    count: int
    predictions: list[PredictItem]
