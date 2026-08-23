import json
import os
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from tasks import celery_app, gerar_laudo_task

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

MODEL_PATH    = Path(__file__).resolve().parent.parent / "ChoosenModel" / "cancer_model.joblib"
METADATA_PATH = Path(__file__).resolve().parent.parent / "ChoosenModel" / "model_metadata.json"

OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

_model    = None
_metadata: dict = {}


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _metadata
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. "
            "Execute o projeto 1.GenAIPrediction primeiro."
        )
    _model = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            _metadata = json.load(f)
    yield


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="""
API de predição de **câncer de mama** usando Regressão Logística otimizada
por Algoritmo Genético. O laudo médico é gerado de forma **assíncrona** via
fila Celery + Redis, eliminando timeouts no browser.

### Fluxo do laudo
1. `POST /predict/breastCancer/laudo` → retorna **predição imediata** + `task_id`
2. `GET  /predict/breastCancer/laudo/status/{task_id}` → polling até `status: completed`
    """,
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:80", "http://localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class BreastCancerInput(BaseModel):
    area_pior: float = Field(..., gt=0, description="Pior valor de área do tumor (worst area)", examples=[880.5])
    textura_pior: float = Field(..., gt=0, description="Pior valor de textura (worst texture)", examples=[25.38])
    pontos_concavos_pior: float = Field(..., ge=0, description="Pior número de pontos côncavos (worst concave points)", examples=[0.2654])
    concavidade_pior: float = Field(..., ge=0, description="Pior severidade das partes côncavas (worst concavity)", examples=[0.3001])

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "area_pior": 880.5,
                "textura_pior": 25.38,
                "pontos_concavos_pior": 0.2654,
                "concavidade_pior": 0.3001,
            }]
        }
    }


class PredictionResponse(BaseModel):
    diagnostico: str           = Field(description="'Maligno' ou 'Benigno'")
    confianca: float           = Field(description="Confiança em % (0–100)")
    classe: int                = Field(description="1 = Maligno, 0 = Benigno")
    probabilidade_maligno: float = Field(description="Probabilidade bruta (0.0–1.0)")
    probabilidade_benigno: float = Field(description="Probabilidade bruta (0.0–1.0)")


class LaudoAsyncResponse(PredictionResponse):
    task_id: str = Field(description="ID da tarefa Celery para polling do laudo")


class LaudoStatusResponse(BaseModel):
    task_id: str  = Field(description="ID da tarefa")
    status: str   = Field(description="pending | processing | completed | failed")
    laudo: str | None = Field(default=None, description="Laudo gerado (apenas quando completed)")
    error: str | None = Field(default=None, description="Mensagem de erro (apenas quando failed)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    ollama_available: bool
    ollama_url: str
    ollama_model: str
    model_experiment: str | None
    model_metrics: dict | None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _run_prediction(data: BreastCancerInput) -> PredictionResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    features = [[
        data.area_pior,
        data.textura_pior,
        data.pontos_concavos_pior,
        data.concavidade_pior,
    ]]

    classe        = int(_model.predict(features)[0])
    probabilidades = _model.predict_proba(features)[0]
    prob_benigno  = round(float(probabilidades[0]), 4)
    prob_maligno  = round(float(probabilidades[1]), 4)
    confianca     = round(float(probabilidades[classe]) * 100, 2)

    return PredictionResponse(
        diagnostico="Maligno" if classe == 1 else "Benigno",
        confianca=confianca,
        classe=classe,
        probabilidade_maligno=prob_maligno,
        probabilidade_benigno=prob_benigno,
    )


def _check_redis() -> bool:
    try:
        celery_app.backend.client.ping()
        return True
    except Exception:
        return False


def _check_ollama() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post(
    "/predict/breastCancer",
    response_model=PredictionResponse,
    summary="Predição rápida (sem laudo)",
    tags=["Predição"],
)
def predict_breast_cancer(data: BreastCancerInput) -> PredictionResponse:
    return _run_prediction(data)


@app.post(
    "/predict/breastCancer/laudo",
    response_model=LaudoAsyncResponse,
    summary="Predição + dispara geração de laudo (async)",
    tags=["Predição"],
    responses={
        200: {"description": "Predição imediata + task_id para polling do laudo"},
        503: {"description": "Modelo não carregado"},
    },
)
def predict_breast_cancer_laudo(data: BreastCancerInput) -> LaudoAsyncResponse:
    pred = _run_prediction(data)

    task = gerar_laudo_task.delay(
        area_pior=data.area_pior,
        textura_pior=data.textura_pior,
        pontos_concavos_pior=data.pontos_concavos_pior,
        concavidade_pior=data.concavidade_pior,
        diagnostico=pred.diagnostico,
        confianca=pred.confianca,
        probabilidade_maligno=pred.probabilidade_maligno,
        probabilidade_benigno=pred.probabilidade_benigno,
    )

    return LaudoAsyncResponse(**pred.model_dump(), task_id=task.id)


@app.get(
    "/predict/breastCancer/laudo/status/{task_id}",
    response_model=LaudoStatusResponse,
    summary="Consulta status do laudo (polling)",
    tags=["Predição"],
)
def laudo_status(task_id: str) -> LaudoStatusResponse:
    result = AsyncResult(task_id, app=celery_app)

    state_map = {
        "PENDING":  "pending",
        "STARTED":  "processing",
        "RETRY":    "processing",
        "SUCCESS":  "completed",
        "FAILURE":  "failed",
    }
    status = state_map.get(result.state, "processing")

    laudo = result.result if status == "completed" and isinstance(result.result, str) else None
    error = str(result.result) if status == "failed" else None

    return LaudoStatusResponse(task_id=task_id, status=status, laudo=laudo, error=error)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Status da API e componentes",
    tags=["Sistema"],
)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        redis_connected=_check_redis(),
        ollama_available=_check_ollama(),
        ollama_url=OLLAMA_URL,
        ollama_model=OLLAMA_MODEL,
        model_experiment=_metadata.get("experiment"),
        model_metrics=_metadata.get("metrics"),
    )
