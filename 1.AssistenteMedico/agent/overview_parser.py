"""Parsing tolerante da resposta da 'visao geral' automatica (rag/chain.py::
ask_overview) nas duas secoes RELEVANTE(S):/ATENCAO(ES): pedidas pelo prompt
(rag/chain.py::OVERVIEW_INSTRUCTION).

Extraido para um modulo proprio (mesmo padrao de agent/scope_guard.py e
agent/overview_summarizer.py) para poder ser testado sem depender de
agent/nodes.py -- que importa rag.chain (torch/langchain_huggingface) so por
estar no mesmo arquivo.

## Por que o parsing precisa ser tolerante

O modelo fine-tuned pequeno nao segue o formato pedido com confiabilidade. Em
uso real (ver logs/audit.jsonl e discussao com o usuario) ja foram
observados:

- "RELEVANTES:" no PLURAL em vez de "RELEVANTE:" -- quebrava o regex de secao
  por completo (a v1 exigia o singular exato), fazendo o parsing cair no
  fallback e devolver o texto INTEIRO -- com os dois rotulos ainda visiveis
  dentro do texto -- como um unico item, sem nenhuma divisao relevante/
  atencao. Esse era o "texto sem formatar" relatado pelo usuario.
- Itens sem marcador de lista ("- "/"* "), separados so por quebra de linha,
  ou so por virgula numa unica linha (uma enumeracao "CSV").

Este modulo tenta, em ordem, do mais estruturado ao menos estruturado: (1) o
formato pedido no prompt (marcadores '-'/'*'), (2) uma linha por item, (3)
itens separados por virgula numa linha so, (4) o texto inteiro como item
unico -- so cai no proximo nivel quando o anterior nao encontra nada.
"""

import re

from agent.overview_summarizer import merge_similar_bullets

# Aceita RELEVANTE(S) e ATEN(C/Ç)(A/Ã)O(ES/S) -- o modelo ja foi visto
# escrevendo no plural apesar do prompt pedir singular (ver historico acima).
OVERVIEW_SECTION_PATTERN = re.compile(
    r"RELEVANTES?\s*:(?P<relevante>.*?)(?:ATEN[CÇ][AÃ]O(?:ES|S)?\s*:(?P<atencao>.*))?$",
    re.IGNORECASE | re.DOTALL,
)

# Bullet precisa ser "- item" ou "* item" (com espaco depois do marcador) - so
# "startswith('*')" colidiria com **negrito** em markdown, como o aviso que
# agent/guardrails.py acrescenta a resposta sinalizada ("**Atencao:** ...").
_BULLET_PATTERN = re.compile(r"^[-*]\s+(.*)$")


def _extract_marked_bullets(texto: str) -> list[str]:
    bullets = []
    for linha in (texto or "").splitlines():
        linha = linha.strip()
        match = _BULLET_PATTERN.match(linha)
        if match:
            item = match.group(1).strip()
            if item:
                bullets.append(item)
    return bullets


def _split_enumeracao_flat(linha: str) -> list[str]:
    """Tenta separar uma enumeracao "flat" (tudo numa linha so, sem marcador
    de lista) em itens. O modelo ja foi visto usando tanto virgula quanto
    ponto-e-virgula como separador dependendo da geracao (ver
    logs/audit.jsonl) -- tenta virgula primeiro, ponto-e-virgula depois.
    So considera separador valido quando produz mais de 2 itens -- uma frase
    comum com uma unica virgula ou ponto-e-virgula nao deveria virar dois
    bullets por causa disso.
    """
    for separador in (",", ";"):
        itens = [item.strip().rstrip(".") for item in linha.split(separador) if item.strip()]
        if len(itens) > 2:
            return itens
    return []


def extract_bullets(texto: str) -> list[str]:
    """Extrai os itens de um bloco de texto (uma secao RELEVANTE/ATENCAO ja
    isolada por OVERVIEW_SECTION_PATTERN, ou o texto inteiro no caso de
    fallback total). Tenta, em ordem, os tres formatos descritos no docstring
    do modulo, so avancando pro proximo quando o anterior nao acha nada.
    """
    bullets = _extract_marked_bullets(texto)
    if bullets:
        return bullets

    linhas = [linha.strip() for linha in (texto or "").splitlines() if linha.strip()]
    if len(linhas) > 1:
        return [linha.rstrip(",") for linha in linhas]

    linha_unica = linhas[0] if linhas else ""
    itens_flat = _split_enumeracao_flat(linha_unica)
    if itens_flat:
        return itens_flat

    return [linha_unica] if linha_unica else []


def parse_overview_sections(texto: str) -> tuple[list[str], list[str]]:
    """Quebra a resposta da 'visao geral' nas duas listas RELEVANTE(S):/
    ATENCAO(ES): pedidas pelo prompt, aplicando merge_similar_bullets em cada
    uma (agent/overview_summarizer.py -- consolida itens repetidos do mesmo
    template).

    Fallback quando NENHUMA das duas secoes e encontrada (nem RELEVANTE nem
    RELEVANTES aparecem no texto): em vez de devolver o texto inteiro -- com
    rotulos e tudo -- como um unico item sem estrutura nenhuma, ainda tenta
    separar em itens via extract_bullets, pelos mesmos criterios de
    linha/virgula. O front ainda mostra uma lista legivel em vez de um
    paragrafo cru.
    """
    match = OVERVIEW_SECTION_PATTERN.search(texto or "")
    if not match:
        texto_limpo = (texto or "").strip()
        pontos_relevantes = merge_similar_bullets(extract_bullets(texto_limpo)) if texto_limpo else []
        return pontos_relevantes, []

    pontos_relevantes = merge_similar_bullets(extract_bullets(match.group("relevante") or ""))
    pontos_atencao = merge_similar_bullets(extract_bullets(match.group("atencao") or ""))
    return pontos_relevantes, pontos_atencao
