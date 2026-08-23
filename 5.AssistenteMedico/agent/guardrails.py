"""Camada de seguranca aplicada a toda resposta do assistente antes de devolve-la.

Implementacao real (nao-stub): um filtro baseado em padroes de linguagem de
prescricao direta. E deliberadamente simples e conservador (prefere falso-positivo
a falso-negativo) — para um projeto academico, um guardrail simples e auditavel é
mais defensavel na banca do que um classificador de ML "caixa-preta" sem explicação.
"""

import re
from dataclasses import dataclass, field

# Padroes que indicam prescricao/conduta direta, sem ressalva de validacao humana.
# Case-insensitive, cobrindo variacoes comuns em PT-BR.
_PRESCRIPTION_PATTERNS = [
    r"\btome\b.{0,20}\bmg\b",
    r"\bprescrevo\b",
    r"\breceit[oa]\b",
    r"\baplique\b.{0,20}\b(mg|ml|dose)\b",
    r"\badministre\b.{0,20}\b(mg|ml|dose)\b",
    r"\btomar\b.{0,20}\b\d+\s?(mg|ml|comprimidos?)\b",
]

_DISCLAIMER = (
    "\n\n**Atencao:** esta sugestao foi sinalizada pelo filtro de seguranca por "
    "conter linguagem de prescricao direta. Ela NAO deve ser usada sem revisao e "
    "validacao de um medico responsavel."
)


@dataclass
class SafetyCheckResult:
    is_flagged: bool
    matched_patterns: list[str] = field(default_factory=list)
    safe_response: str = ""


def check_response(raw_response: str) -> SafetyCheckResult:
    """Varre a resposta gerada pela LLM em busca de linguagem de prescricao direta.

    Retorna a resposta original se estiver ok, ou a resposta + aviso se disparou
    algum padrao — a resposta NUNCA e descartada silenciosamente, apenas sinalizada
    (quem decide o que fazer com uma resposta sinalizada e o fluxo do LangGraph,
    ver agent/graph.py::validar_seguranca).
    """
    matched = [p for p in _PRESCRIPTION_PATTERNS if re.search(p, raw_response, re.IGNORECASE)]

    if not matched:
        return SafetyCheckResult(is_flagged=False, safe_response=raw_response)

    return SafetyCheckResult(
        is_flagged=True,
        matched_patterns=matched,
        safe_response=raw_response + _DISCLAIMER,
    )


SYSTEM_SAFETY_PROMPT = """Voce e um assistente de apoio a decisao clinica, nao um
medico. Seu papel e sugerir, contextualizar e organizar informacao — nunca prescrever
diretamente um tratamento, medicamento ou dose. Toda sugestao deve ser explicitamente
enquadrada como algo a ser validado por um profissional de saude responsavel."""
