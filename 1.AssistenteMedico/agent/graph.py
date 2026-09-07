"""Grafo de decisao do assistente medico, usando LangGraph.

Ver diagrama completo em PLANO_Fase3.md, secao 2.5:

    receber_paciente
          |
          v
    verificar_exames_pendentes --(pendente)--> emitir_alerta_exame --+
          | (sem pendencia)                                          |
          v                                                          |
    rotear_geracao <------------------------------------------------ +
          |
          +--(chat)-------> classificar_escopo --(dentro_do_escopo)--> buscar_contexto_rag_e_gerar_resposta --+
          |                        |                                                                           |
          |                        +--(fora_de_escopo)--> responder_fora_de_escopo -----------------------------+
          |                                                                                                     |
          +--(visao_geral)-> gerar_visao_geral --------------------------------------------------------------->+
                                                                                                                 v
                                                                                                      validar_seguranca --(sinalizado)--> encaminhar_para_validacao_humana --+
                                                                                                            | (aprovado)                                                    |
                                                                                                            v                                                                |
                                                                                                      finalizar_resposta <-------------------------------------------------+
                                                                                                            |
                                                                                                            v
                                                                                                      log_auditoria (END)

tipo_interacao ("chat" por padrao, ou "visao_geral") decide em rotear_geracao qual
caminho seguir. O chat passa antes por classificar_escopo (agent/scope_guard.py):
um guardrail de escopo por similaridade de embeddings que decide se a pergunta
livre do medico e sobre o dominio clinico ANTES de montar contexto do paciente e
chamar o LLM — se nao for, responder_fora_de_escopo devolve uma mensagem fixa
sem nunca expor o contexto do paciente ao modelo (evita o modelo "completar" uma
recusa com dado clinico irrelevante). Se for, segue para
buscar_contexto_rag_e_gerar_resposta (com retrieval de protocolos via RAG). A
visao geral (gerar_visao_geral, sem retrieval nem classificacao de escopo — e
sempre sobre o proprio paciente) sintetiza a anamnese/exames em topicos
"RELEVANTE"/"ATENCAO" para o card automatico da tela do paciente — ver
api/main.py::get_patient_overview e front/.../ai-overview-panel. Todos os
caminhos convergem em validar_seguranca, log de auditoria e validacao humana
inclusive.
"""

import sys
from pathlib import Path
from typing import TypedDict

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from langgraph.graph import END, StateGraph

from agent import nodes


class AgentState(TypedDict, total=False):
    paciente_id: int
    pergunta: str
    tipo_interacao: str  # "chat" (default) ou "visao_geral" — ver rotear_geracao
    dentro_do_escopo: bool  # so preenchido no caminho chat — ver classificar_escopo
    similaridade_escopo: float  # idem — margem (dentro - fora) do guardrail de escopo (agent/scope_guard.py)
    similaridade_escopo_dentro: float  # idem — similaridade contra ancoras clinicas
    similaridade_escopo_fora: float  # idem — similaridade contra ancoras fora de escopo
    exames_pendentes: list[str]
    alerta: str | None
    resposta_bruta: str
    fontes: list[str]
    resposta_final: str
    demografia: dict | None  # sexo, data de nascimento e idade — ver receber_paciente
    contexto_clinico: str | None  # ficha do paciente usada na visao geral (ancoragem)
    pontos_relevantes: list[str]  # so preenchido quando tipo_interacao == "visao_geral"
    pontos_atencao: list[str]  # idem
    pontos_descartados: int  # itens removidos pelo filtro de ancoragem (auditoria)
    geracao_degenerada: bool  # secao inteira descartada — ver agent/overview_filter.py
    requer_validacao_humana: bool
    padroes_sinalizados: list[str]
    termos_etarios_incoerentes: list[str]  # ver agent/demographic_guard.py
    status: str


def _tem_exames_pendentes(state: AgentState) -> str:
    return "com_pendencia" if state.get("exames_pendentes") else "sem_pendencia"


def _tipo_interacao(state: AgentState) -> str:
    return "visao_geral" if state.get("tipo_interacao") == "visao_geral" else "chat"


def _dentro_do_escopo(state: AgentState) -> str:
    return "dentro_do_escopo" if state.get("dentro_do_escopo") else "fora_de_escopo"


def _foi_sinalizado(state: AgentState) -> str:
    return "sinalizado" if state.get("requer_validacao_humana") else "aprovado"


def build_graph():
    """Monta e compila o StateGraph. Chamado uma vez na inicializacao da API."""
    graph = StateGraph(AgentState)

    graph.add_node("receber_paciente", nodes.receber_paciente)
    graph.add_node("verificar_exames_pendentes", nodes.verificar_exames_pendentes)
    graph.add_node("emitir_alerta_exame", nodes.emitir_alerta_exame)
    graph.add_node("rotear_geracao", nodes.rotear_geracao)
    graph.add_node("classificar_escopo", nodes.classificar_escopo)
    graph.add_node("responder_fora_de_escopo", nodes.responder_fora_de_escopo)
    graph.add_node("buscar_contexto_rag_e_gerar_resposta", nodes.buscar_contexto_rag_e_gerar_resposta)
    graph.add_node("gerar_visao_geral", nodes.gerar_visao_geral)
    graph.add_node("validar_seguranca", nodes.validar_seguranca)
    graph.add_node("encaminhar_para_validacao_humana", nodes.encaminhar_para_validacao_humana)
    graph.add_node("finalizar_resposta", nodes.finalizar_resposta)
    graph.add_node("log_auditoria", nodes.log_auditoria)

    graph.set_entry_point("receber_paciente")
    graph.add_edge("receber_paciente", "verificar_exames_pendentes")

    graph.add_conditional_edges(
        "verificar_exames_pendentes",
        _tem_exames_pendentes,
        {
            "com_pendencia": "emitir_alerta_exame",
            "sem_pendencia": "rotear_geracao",
        },
    )
    graph.add_edge("emitir_alerta_exame", "rotear_geracao")

    graph.add_conditional_edges(
        "rotear_geracao",
        _tipo_interacao,
        {
            "chat": "classificar_escopo",
            "visao_geral": "gerar_visao_geral",
        },
    )
    graph.add_conditional_edges(
        "classificar_escopo",
        _dentro_do_escopo,
        {
            "dentro_do_escopo": "buscar_contexto_rag_e_gerar_resposta",
            "fora_de_escopo": "responder_fora_de_escopo",
        },
    )
    graph.add_edge("buscar_contexto_rag_e_gerar_resposta", "validar_seguranca")
    graph.add_edge("responder_fora_de_escopo", "validar_seguranca")
    graph.add_edge("gerar_visao_geral", "validar_seguranca")

    graph.add_conditional_edges(
        "validar_seguranca",
        _foi_sinalizado,
        {
            "sinalizado": "encaminhar_para_validacao_humana",
            "aprovado": "finalizar_resposta",
        },
    )
    graph.add_edge("encaminhar_para_validacao_humana", "log_auditoria")
    graph.add_edge("finalizar_resposta", "log_auditoria")
    graph.add_edge("log_auditoria", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    # Exemplo de execucao manual (requer rag/chain.py e finetuning implementados):
    # resultado = app.invoke({"paciente_id": 1, "pergunta": "..."})  # chat
    # resultado = app.invoke({"paciente_id": 1, "tipo_interacao": "visao_geral"})  # visao geral
    # print(resultado)
