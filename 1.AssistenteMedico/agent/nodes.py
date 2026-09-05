"""Implementacao de cada no do grafo (agent/graph.py). Cada funcao recebe e
retorna o dicionario de estado do LangGraph (ver AgentState em graph.py).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from agent import tools
from agent.guardrails import check_response
from rag.chain import ask as rag_ask

# Log de auditoria — requisito de rastreabilidade do desafio (PLANO_Fase3.md secao 2.6):
# uma linha JSON por interacao, incluindo pergunta, paciente, fontes citadas, resposta
# e a decisao de seguranca tomada pelo grafo.
AUDIT_LOG_DIR = _BASE_DIR / "logs"
AUDIT_LOG_PATH = AUDIT_LOG_DIR / "audit.jsonl"


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
    (timestamp, pergunta, paciente, fontes citadas, resposta, flags de seguranca,
    decisao do grafo) — requisito de logging/auditoria do desafio.

    Uma linha JSON por interacao (append), para poder auditar/replayar depois sem
    carregar tudo em memoria. Nunca levanta excecao por falha de escrita do log (um
    problema de disco/permissao nao deveria derrubar a resposta ja gerada para o
    medico) — so registra o problema no stdout.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paciente_id": state.get("paciente_id"),
        "pergunta": state.get("pergunta"),
        "exames_pendentes": state.get("exames_pendentes", []),
        "alerta": state.get("alerta"),
        "fontes": state.get("fontes", []),
        "resposta_bruta": state.get("resposta_bruta"),
        "resposta_final": state.get("resposta_final"),
        "padroes_sinalizados": state.get("padroes_sinalizados", []),
        "requer_validacao_humana": state.get("requer_validacao_humana"),
        "status": state.get("status"),
    }
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Aviso: falha ao gravar log de auditoria em {AUDIT_LOG_PATH}: {exc}")
    return state
