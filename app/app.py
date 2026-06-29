
import sys
import os
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Allow importing from src/ regardless of where uvicorn is launched from
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import FraudPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0.0")

BASE_DIR = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Load the model ONCE at startup, not per-request
predictor = FraudPredictor(params_path=os.path.join(BASE_DIR, "..", "params.yaml"))

# ---------------------------------------------------------------------------
# Prometheus metrics (step 11 in the architecture diagram)
# ---------------------------------------------------------------------------
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total", "Total number of predictions made", ["result"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds", "Time taken to make a prediction"
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """
    Matches the Kaggle dataset columns: Time, V1-V28, Amount.
    Using a flexible dict-like model so the frontend form can send
    exactly these fields without needing 30 separate hardcoded args.
    """
    Time: float = Field(..., example=0.0)
    Amount: float = Field(..., example=149.62)
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serves the simple HTML page for manual testing."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Accepts a transaction and returns a fraud prediction."""
    start = time.time()

    result = predictor.predict_one(transaction.dict())

    PREDICTION_LATENCY.observe(time.time() - start)
    PREDICTION_COUNTER.labels(result="fraud" if result["is_fraud"] else "not_fraud").inc()

    return result


@app.get("/health")
def health():
    """Used by Docker healthcheck and CI/CD to confirm the API is alive."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    """Prometheus scrapes this endpoint (see monitoring/prometheus.yml)."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)