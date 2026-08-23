"""FastAPI que expoe o assistente medico, seguindo o mesmo padrao de
2.PredictionApi/main.py (lifespan carrega o grafo uma vez, endpoints Pydantic).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agent.graph import build_graph

_graph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph_app
    _graph_app = build_graph()
    yield


app = FastAPI(
    title="Assistente Medico Virtual — OncoTech Fase 3",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    paciente_id: int = Field(..., description="ID do paciente no prontuario mock")
    pergunta: str = Field(..., min_length=3, description="Pergunta do medico")


class AskResponse(BaseModel):
    resposta: str
    fontes: list[str]
    alerta: str | None = None
    requer_validacao_humana: bool
    status: str


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph_app is not None}


@app.post("/assistant/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """TODO: tratar excecoes (RAG/LLM nao carregados -> 503; erro inesperado -> 500),
    seguindo o padrao de codigos de erro documentado em 2.PredictionApi/PredictionApi.md.
    """
    result = _graph_app.invoke({"paciente_id": payload.paciente_id, "pergunta": payload.pergunta})
    return AskResponse(
        resposta=result["resposta_final"],
        fontes=result.get("fontes", []),
        alerta=result.get("alerta"),
        requer_validacao_humana=result["requer_validacao_humana"],
        status=result["status"],
    )
