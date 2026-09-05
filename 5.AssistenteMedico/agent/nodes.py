"""Implementacao de cada no do grafo (agent/graph.py). Cada funcao recebe e
retorna o dicionario de estado do LangGraph (ver AgentState em graph.py).
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from agent import tools
from agent.guardrails import check_response
from rag.chain import ask as rag_ask


def receber_paciente(state: dict) -> dict:
    """No de entrada: apenas valida que paciente_id e pergunta foram fornecidos."""
    assert state.get("paciente_id") is not None, "paciente_id e obrigatorio"
    assert state.get("pergunta"), "pergunta e obrigatoria"
    return state


def verificar_exames_pendentes(state: dict) -> dict:
    pendentes = tools.get_exames_pendentes(state["paciente_id"])
    state["exames_pendentes"] = pendentes
    return state


def emitir_alerta_exame(state: dict) -> dict:
    """Executado apenas quando ha exames pendentes (aresta condicional em graph.py)."""
    state["alerta"] = (
        f"Atencao: paciente possui exame(s) pendente(s): {', '.join(state['exames_pendentes'])}."
    )
    return state


def buscar_contexto_rag_e_gerar_resposta(state: dict) -> dict:
    """Une busca de contexto (RAG) + geracao da resposta em uma unica chamada,
    reaproveitando rag.chain.ask (que ja faz retrieval + prompt + LLM).
    """
    historico = tools.get_historico_paciente(state["paciente_id"])
    rag_response = rag_ask(state["pergunta"], patient_context=historico)
    state["resposta_bruta"] = rag_response.answer
    state["fontes"] = rag_response.sources
    return state


def validar_seguranca(state: dict) -> dict:
    resultado = check_response(state["resposta_bruta"])
    state["resposta_final"] = resultado.safe_response
    state["requer_validacao_humana"] = resultado.is_flagged
    state["padroes_sinalizados"] = resultado.matched_patterns
    return state


def encaminhar_para_validacao_humana(state: dict) -> dict:
    """No executado quando a resposta foi sinalizada — nao bloqueia, apenas marca
    explicitamente para revisao (ver PLANO_Fase3.md secao 2.6).
    """
    state["status"] = "aguardando_validacao_humana"
    return state


def finalizar_resposta(state: dict) -> dict:
    state.setdefault("status", "concluido")
    return state


def log_auditoria(state: dict) -> dict:
    """No terminal, sempre executado. Grava a interacao completa em logs/audit.jsonl
    (timestamp, pergunta, paciente, contexto recuperado, resposta, flags de
    seguranca, decisao do grafo) — requisito de logging/auditoria do desafio.

    TODO: usar um timestamp real (nao disponivel em ambiente de teste/replay) e
    persistir via json.dumps em modo append em BASE_DIR / 'logs' / 'audit.jsonl'.
    """
    raise NotImplementedError
