"""FastAPI que expoe o assistente medico: lifespan carrega o grafo uma vez na
inicializacao, endpoints com validacao Pydantic.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

# Libera o front-end Angular (ng serve, porta padrao 4200) para consumir a API
# em desenvolvimento local.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    # Termos de faixa etaria na resposta que nao correspondem a idade do paciente
    # (agent/demographic_guard.py). Lista vazia = resposta coerente com a idade.
    termos_etarios_incoerentes: list[str] = []


class PacienteResponse(BaseModel):
    id: int
    nome_ficticio: str
    historico: str | None = None
    sexo: str | None = None
    data_nascimento: str | None = None
    idade: int | None = None  # calculada a partir de data_nascimento (ver agent/tools.py)


class ExameResumoResponse(BaseModel):
    id: int
    tipo: str
    status: str  # "pendente" ou "concluido"
    data_solicitacao: str | None = None
    data_realizacao: str | None = None  # None enquanto o exame estiver pendente


class ExameDetailResponse(BaseModel):
    id: int
    paciente_id: int
    tipo: str
    status: str
    data_solicitacao: str | None = None
    data_realizacao: str | None = None  # None enquanto o exame estiver pendente
    resultado: str | None = None  # None enquanto o exame estiver pendente
    medico_solicitante: str | None = None
    observacoes: str | None = None


class AnamneseResponse(BaseModel):
    id: int
    paciente_id: int
    data_registro: str | None = None
    profissional_responsavel: str | None = None
    queixa_principal: str | None = None
    historia_doenca_atual: str | None = None
    historia_patologica_pregressa: str | None = None
    historia_familiar: str | None = None
    historia_ginecologica_obstetrica: str | None = None
    medicamentos_em_uso: str | None = None
    alergias: str | None = None
    cirurgias_previas: str | None = None
    tabagismo: str | None = None
    etilismo: str | None = None
    atividade_fisica: str | None = None
    pressao_arterial: str | None = None
    frequencia_cardiaca: str | None = None
    frequencia_respiratoria: str | None = None
    temperatura_c: str | None = None
    saturacao_o2: str | None = None
    peso_kg: float | None = None
    altura_cm: float | None = None
    exame_fisico: str | None = None
    hipotese_diagnostica: str | None = None
    conduta: str | None = None


class VisaoGeralResponse(BaseModel):
    pontos_relevantes: list[str]
    pontos_atencao: list[str]
    fontes: list[str]
    alerta: str | None = None
    requer_validacao_humana: bool
    status: str
    # Transparencia sobre o filtro de ancoragem (agent/overview_filter.py): quantos
    # itens gerados foram descartados por nao serem rastreaveis ao prontuario, e se
    # a geracao foi classificada como degenerada — caso em que as listas voltam
    # vazias de proposito, e nao por ausencia de achados. O front pode usar isso
    # para diferenciar "nada a sinalizar" de "resposta descartada".
    pontos_descartados: int = 0
    geracao_degenerada: bool = False


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph_app is not None}


@app.get("/patients", response_model=list[PacienteResponse])
def list_patients(
    nome: str | None = Query(
        None,
        description="Filtra por parte do nome (case-insensitive). Omitido/vazio lista todos.",
    ),
) -> list[PacienteResponse]:
    """Lista pacientes do prontuario mock — usado pelo medico para localizar o
    `paciente_id` antes de chamar POST /assistant/ask.
    """
    try:
        pacientes = tools.list_pacientes(nome)
    except Exception as exc:  # noqa: BLE001 — resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro ao consultar pacientes: {exc}") from exc
    return [PacienteResponse(**p) for p in pacientes]


@app.get("/patients/{paciente_id}/exams", response_model=list[ExameResumoResponse])
def list_patient_exams(paciente_id: int) -> list[ExameResumoResponse]:
    """Lista todos os exames do paciente (feitos e pendentes) com o status de cada
    um. 404 se o paciente nao existir — diferente de uma lista vazia, que significa
    paciente existe mas ainda nao tem exame nenhum registrado.
    """
    if not tools.paciente_existe(paciente_id):
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} nao encontrado.")
    try:
        exames = tools.list_exames_paciente(paciente_id)
    except Exception as exc:  # noqa: BLE001 — resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro ao consultar exames: {exc}") from exc
    return [ExameResumoResponse(**e) for e in exames]


@app.get("/patients/{paciente_id}/anamnesis", response_model=AnamneseResponse)
def get_patient_anamnesis(paciente_id: int) -> AnamneseResponse:
    """Ficha de anamnese completa do paciente (queixa principal, historias medica/
    familiar/ginecologica, habitos, sinais vitais, exame fisico, hipotese
    diagnostica e conduta). 404 se o paciente nao existir; 404 tambem se o
    paciente existir mas ainda nao tiver anamnese registrada.
    """
    if not tools.paciente_existe(paciente_id):
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} nao encontrado.")
    try:
        anamnese = tools.get_anamnese(paciente_id)
    except Exception as exc:  # noqa: BLE001 — resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro ao consultar anamnese: {exc}") from exc
    if anamnese is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paciente {paciente_id} nao possui anamnese registrada.",
        )
    return AnamneseResponse(**anamnese)


@app.post("/patients/{paciente_id}/overview", response_model=VisaoGeralResponse)
def get_patient_overview(paciente_id: int) -> VisaoGeralResponse:
    """Visao geral automatica gerada por IA para a tela do paciente: sintetiza a
    anamnese e os exames em pontos relevantes e pontos de atencao, sem repetir a
    ficha inteira (diferente de POST /assistant/ask, que responde a uma pergunta
    livre do medico). Usa o mesmo grafo do assistente (agent/graph.py) com
    tipo_interacao="visao_geral", que seleciona o no gerar_visao_geral em vez de
    buscar_contexto_rag_e_gerar_resposta - ver agent/nodes.py.

    Mesma convencao de erros de POST /assistant/ask: 503 quando uma dependencia
    (RAG/adapter fine-tuned) ainda nao esta pronta, 500 para o resto. 404 se o
    paciente nao existir.
    """
    if not tools.paciente_existe(paciente_id):
        raise HTTPException(status_code=404, detail=f"Paciente {paciente_id} nao encontrado.")

    try:
        result = _graph_app.invoke({"paciente_id": paciente_id, "tipo_interacao": "visao_geral"})
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "O assistente ainda nao esta pronto: falta gerar um artefato "
                f"necessario (indice do RAG ou adapter fine-tuned). Detalhe: {exc}"
            ),
        ) from exc
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Uma parte do pipeline do assistente ainda nao foi implementada "
                f"(ver agent/nodes.py). Detalhe tecnico: {exc!r}"
            ),
        ) from exc
    except Exception as exc:  # noqa: BLE001 - resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro inesperado ao gerar visao geral: {exc}") from exc

    return VisaoGeralResponse(
        pontos_relevantes=result.get("pontos_relevantes", []),
        pontos_atencao=result.get("pontos_atencao", []),
        fontes=result.get("fontes", []),
        alerta=result.get("alerta"),
        requer_validacao_humana=result["requer_validacao_humana"],
        status=result["status"],
        pontos_descartados=result.get("pontos_descartados") or 0,
        geracao_degenerada=bool(result.get("geracao_degenerada")),
    )


@app.get("/exams/{exame_id}", response_model=ExameDetailResponse)
def get_exam(exame_id: int) -> ExameDetailResponse:
    """Detalhe completo de um exame (todos os campos, incluindo o paciente_id a
    quem ele pertence).
    """
    try:
        exame = tools.get_exame(exame_id)
    except Exception as exc:  # noqa: BLE001 — resposta de erro generica de proposito
        raise HTTPException(status_code=500, detail=f"Erro ao consultar exame: {exc}") from exc
    if exame is None:
        raise HTTPException(status_code=404, detail=f"Exame {exame_id} nao encontrado.")
    return ExameDetailResponse(**exame)


@app.post("/assistant/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    """Convencao de codigos de erro: 503 quando uma dependencia (RAG/LLM) ainda
    nao esta pronta, 500 para o resto.
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
        termos_etarios_incoerentes=result.get("termos_etarios_incoerentes", []),
    )
