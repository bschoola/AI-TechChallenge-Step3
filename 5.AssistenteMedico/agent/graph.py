"""Grafo de decisao do assistente medico, usando LangGraph.

Ver diagrama completo em PLANO_Fase3.md, secao 2.5:

    receber_paciente
          |
          v
    verificar_exames_pendentes --(pendente)--> emitir_alerta_exame --+
          | (sem pendencia)                                          |
          v                                                          |
    buscar_contexto_rag_e_gerar_resposta <-------------------------- +
          |
          v
    validar_seguranca --(sinalizado)--> encaminhar_para_validacao_humana --+
          | (aprovado)                                                    |
          v                                                                |
    finalizar_resposta <-------------------------------------------------+
          |
          v
    log_auditoria (END)
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import nodes


class AgentState(TypedDict, total=False):
    paciente_id: int
    pergunta: str
    exames_pendentes: list[str]
    alerta: str | None
    resposta_bruta: str
    fontes: list[str]
    resposta_final: str
    requer_validacao_humana: bool
    padroes_sinalizados: list[str]
    status: str


def _tem_exames_pendentes(state: AgentState) -> str:
    return "com_pendencia" if state.get("exames_pendentes") else "sem_pendencia"


def _foi_sinalizado(state: AgentState) -> str:
    return "sinalizado" if state.get("requer_validacao_humana") else "aprovado"


def build_graph():
    """Monta e compila o StateGraph. Chamado uma vez na inicializacao da API."""
    graph = StateGraph(AgentState)

    graph.add_node("receber_paciente", nodes.receber_paciente)
    graph.add_node("verificar_exames_pendentes", nodes.verificar_exames_pendentes)
    graph.add_node("emitir_alerta_exame", nodes.emitir_alerta_exame)
    graph.add_node("buscar_contexto_rag_e_gerar_resposta", nodes.buscar_contexto_rag_e_gerar_resposta)
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
            "sem_pendencia": "buscar_contexto_rag_e_gerar_resposta",
        },
    )
    graph.add_edge("emitir_alerta_exame", "buscar_contexto_rag_e_gerar_resposta")
    graph.add_edge("buscar_contexto_rag_e_gerar_resposta", "validar_seguranca")

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
    # resultado = app.invoke({"paciente_id": 1, "pergunta": "..."})
    # print(resultado)
