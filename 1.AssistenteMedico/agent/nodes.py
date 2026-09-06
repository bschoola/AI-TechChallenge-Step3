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
from agent.overview_parser import parse_overview_sections
from agent.scope_guard import FORA_DE_ESCOPO_RESPOSTA, classify_scope
from rag.chain import ask as rag_ask
from rag.chain import ask_overview as rag_ask_overview

# Log de auditoria — requisito de rastreabilidade do desafio (PLANO_Fase3.md secao 2.6):
# uma linha JSON por interacao, incluindo pergunta, paciente, fontes citadas, resposta
# e a decisao de seguranca tomada pelo grafo.
AUDIT_LOG_DIR = _BASE_DIR / "logs"
AUDIT_LOG_PATH = AUDIT_LOG_DIR / "audit.jsonl"

def receber_paciente(state: dict) -> dict:
    """No de entrada: valida paciente_id sempre; 'pergunta' e obrigatoria so no
    fluxo de chat (tipo_interacao == 'chat', o default) — a visao geral
    (tipo_interacao == 'visao_geral') nao tem uma pergunta livre do medico, ver
    agent/graph.py e api/main.py::get_patient_overview.
    """
    assert state.get("paciente_id") is not None, "paciente_id e obrigatorio"
    if state.get("tipo_interacao", "chat") == "chat":
        assert state.get("pergunta"), "pergunta e obrigatoria para o chat"
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


def rotear_geracao(state: dict) -> dict:
    """No de passagem, sem efeito no estado: existe so para dar um ponto unico de
    saida tanto do caminho com exame pendente (apos emitir_alerta_exame) quanto do
    caminho sem pendencia, de onde uma unica aresta condicional (agent/graph.py::
    _tipo_interacao) decide qual no de geracao chamar a seguir — chat ou visao
    geral — sem duplicar a logica de checagem de exames pendentes para os dois
    tipos de interacao.
    """
    return state


def classificar_escopo(state: dict) -> dict:
    """Guardrail de escopo do chat (agent/scope_guard.py): roda so no caminho de
    chat (tipo_interacao == 'chat'), antes de montar contexto do paciente e
    chamar o LLM. Decide, por similaridade RELATIVA de embeddings (ancoras
    clinicas vs. ancoras fora de escopo -- ver historico no docstring de
    agent/scope_guard.py), se a pergunta livre do medico e sobre o dominio
    clinico do assistente.

    Existe para resolver um problema observado em uso real: perguntas fora do
    dominio (ex. "onde ir na praia perto de Sao Paulo?", "devo trocar a vela de
    ignicao do carro?") faziam o modelo tentar recusar e, no mesmo texto,
    misturar pedacos do contexto do paciente que estava no prompt -- um modelo
    pequeno fine-tuned em QA factual tende a "completar" em vez de recusar de
    forma seca quando ja tem dado clinico disponivel no contexto. A aresta
    condicional em agent/graph.py usa 'dentro_do_escopo' para decidir entre
    buscar_contexto_rag_e_gerar_resposta (que injeta o contexto do paciente) e
    responder_fora_de_escopo (que nunca chega a ver esse contexto).

    As duas similaridades (dentro/fora) e a margem entre elas sao guardadas no
    estado (e vao pro log de auditoria em log_auditoria abaixo) para dar
    transparencia sobre a decisao e permitir recalibrar SCOPE_MARGIN com casos
    reais depois -- foi assim que a primeira versao (limiar absoluto) foi
    identificada como insuficiente.
    """
    resultado = classify_scope(state["pergunta"])
    state["dentro_do_escopo"] = resultado.dentro_do_escopo
    state["similaridade_escopo_dentro"] = resultado.similaridade_dentro
    state["similaridade_escopo_fora"] = resultado.similaridade_fora
    state["similaridade_escopo"] = resultado.margem
    return state


def responder_fora_de_escopo(state: dict) -> dict:
    """Executado quando classificar_escopo marca a pergunta como fora do escopo
    clinico (aresta condicional em agent/graph.py). Resposta fixa em codigo, NAO
    gerada pelo LLM -- e exatamente o ponto: sem chamar o modelo aqui, nao ha
    contexto do paciente disponivel para ele "completar" a recusa com dado
    clinico irrelevante. Converge em validar_seguranca como qualquer outra
    resposta, entao ainda passa pelos guardrails e pelo log de auditoria
    normalmente.
    """
    state["resposta_bruta"] = FORA_DE_ESCOPO_RESPOSTA
    state["fontes"] = []
    return state


def buscar_contexto_rag_e_gerar_resposta(state: dict) -> dict:
    """Une busca de contexto (RAG) + geracao da resposta em uma unica chamada,
    reaproveitando rag.chain.ask (que ja faz retrieval + prompt + LLM).

    Caminho do CHAT (tipo_interacao == 'chat'): responde a pergunta livre do
    medico. O contexto do paciente inclui o historico resumido E a ficha de
    anamnese completa (queixa, historias, habitos, sinais vitais, exame fisico,
    hipotese diagnostica e conduta — ver agent/tools.py::montar_contexto_clinico).

    A "visao geral" automatica que o front-end dispara ao abrir a tela do paciente
    usa um caminho diferente — ver gerar_visao_geral abaixo — porque pedir pra essa
    mesma chain "revisar a anamnese" fazia o modelo so parafrasear a ficha que ja
    esta visivel na tela, em vez de destacar o que e clinicamente relevante.
    """
    contexto_clinico = tools.montar_contexto_clinico(state["paciente_id"])
    rag_response = rag_ask(state["pergunta"], patient_context=contexto_clinico)
    state["resposta_bruta"] = rag_response.answer
    state["fontes"] = rag_response.sources
    return state


def gerar_visao_geral(state: dict) -> dict:
    """Gera a 'visao geral' automatica exibida no card de IA da tela do paciente —
    caminho alternativo a buscar_contexto_rag_e_gerar_resposta, escolhido pela
    aresta condicional de agent/graph.py quando tipo_interacao == 'visao_geral'.

    Diferente do chat, aqui nao ha uma pergunta livre do medico: o objetivo e
    sintetizar a anamnese e os exames em pontos relevantes e pontos de atencao,
    sem apenas repetir a ficha inteira (ver rag/chain.py::ask_overview e
    OVERVIEW_INSTRUCTION para o formato de saida pedido ao modelo, e
    _parse_overview_sections abaixo para como isso vira duas listas). Nao faz
    retrieval de protocolos via RAG: o insumo aqui e o caso do paciente, nao os
    protocolos internos do hospital.
    """
    contexto_clinico = tools.montar_contexto_clinico(state["paciente_id"])
    rag_response = rag_ask_overview(contexto_clinico)
    state["resposta_bruta"] = rag_response.answer
    state["fontes"] = rag_response.sources
    return state


def validar_seguranca(state: dict) -> dict:
    resultado = check_response(state["resposta_bruta"])
    state["resposta_final"] = resultado.safe_response
    state["requer_validacao_humana"] = resultado.is_flagged
    state["padroes_sinalizados"] = resultado.matched_patterns

    if state.get("tipo_interacao") == "visao_geral":
        pontos_relevantes, pontos_atencao = parse_overview_sections(state["resposta_final"])
        state["pontos_relevantes"] = pontos_relevantes
        state["pontos_atencao"] = pontos_atencao

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
        "tipo_interacao": state.get("tipo_interacao", "chat"),
        "paciente_id": state.get("paciente_id"),
        "pergunta": state.get("pergunta"),
        "dentro_do_escopo": state.get("dentro_do_escopo"),
        "similaridade_escopo": state.get("similaridade_escopo"),
        "similaridade_escopo_dentro": state.get("similaridade_escopo_dentro"),
        "similaridade_escopo_fora": state.get("similaridade_escopo_fora"),
        "exames_pendentes": state.get("exames_pendentes", []),
        "alerta": state.get("alerta"),
        "fontes": state.get("fontes", []),
        "resposta_bruta": state.get("resposta_bruta"),
        "resposta_final": state.get("resposta_final"),
        "pontos_relevantes": state.get("pontos_relevantes", []),
        "pontos_atencao": state.get("pontos_atencao", []),
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
