"""FastAPI que expoe o assistente medico, seguindo o mesmo padrao de
2.PredictionApi/main.py (lifespan carrega o grafo uma vez, endpoints Pydantic).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import tools
from agent.graph import build_graph

_graph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph_app
    # Garante que o prontuario mock (SQLite) existe com schema + dados sinteticos
    # ANTES do grafo rodar — sem isso, o primeiro no do grafo que consulta o banco
    # (verificar_exames_pendentes) falha com "no such table: exames".
    tools.init_mock_db()
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
    """Segue o padrao de codigos de erro de 2.PredictionApi/PredictionApi.md:
    503 quando uma dependencia (RAG/LLM) ainda nao esta pronta, 500 para o resto.
    """
    try:
        result = _graph_app.invoke({"paciente_id": payload.paciente_id, "pergunta": payload.pergunta})
    except FileNotFoundError as exc:
        # rag/chain.py::load_vectorstore ou load_llm_and_tokenizer — indice do RAG
        # ou adapter fine-tuned ainda nao existem em disco (ver mensagem de exc).
        raise HTTPException(
            status_code=503,
            detail=(
                "O assistente ainda nao esta pronto: falta gerar um artefato "
                f"necessario (indice do RAG ou adapter fine-tuned). Detalhe: {exc}"
            ),
        ) from exc
    except NotImplementedError as exc:
        # Sinaliza que algum no do grafo ainda esta em stub — nao necessariamente
        # rag/chain.py, pode ser qualquer peca ainda nao implementada (ver
        # agent/nodes.py). str(exc) pode vir vazio se o stub nao tiver mensagem.
        raise HTTPException(
            status_code=503,
            detail=(
                "Uma parte do pipeline do assistente ainda nao foi implementada "
                f"(ver agent/nodes.py). Detalhe tecnico: {exc!r}"
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro inesperado no assistente: {exc}") from exc

    return AskResponse(
        resposta=result["resposta_final"],
        fontes=result.get("fontes", []),
        alerta=result.get("alerta"),
        requer_validacao_humana=result["requer_validacao_humana"],
        status=result["status"],
    )
