"""Verificacao das citacoes de protocolo na resposta do chat.

A explainability do assistente depende de a citacao ser verdadeira. Uma resposta
que atribui a um protocolo interno uma orientacao que ele nao contem e pior do
que uma resposta sem citacao nenhuma: ela empresta autoridade institucional a
uma afirmacao que o modelo produziu sozinho, e o medico so descobriria abrindo o
protocolo.

## O que foi observado

Numa mesma resposta, dois erros distintos de citacao:

- **Codigo inexistente**: "Protocolo pneumo-02". Os protocolos de pneumologia sao
  PNE-01 a PNE-03; "pneumo-02" nao existe no acervo.
- **Codigo real, conteudo trocado**: "Protocolo seg-01: Hernias inguinais". O
  SEG-01 existe, mas trata de alergias e interacoes medicamentosas. Hernia
  inguinal e o CIR-01.

## O que este modulo detecta

Extrai da resposta as referencias no formato "protocolo <codigo>" e compara com
os codigos dos chunks efetivamente recuperados para aquela pergunta. Sinaliza
quando a resposta cita um protocolo que NAO estava no contexto — o que cobre os
dois casos acima, porque tanto o codigo inventado quanto o codigo real nao
recuperado sao referencias que o modelo nao poderia ter lido.

O que ele NAO detecta: uma citacao a um protocolo que de fato foi recuperado,
mas cujo conteudo a resposta descreve errado. Verificar isso exigiria comparar a
afirmacao com o texto do protocolo, o que e uma tarefa semantica fora do alcance
de uma checagem lexical. A limitacao esta registrada no relatorio tecnico.

Como as demais camadas, sinaliza para validacao humana e anexa a ressalva — nao
bloqueia nem reescreve a resposta.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# "PROTOCOLO MAMA-02", "protocolo cir01", "Protocolo seg-01:" — o codigo e uma
# sigla de letras seguida de um numero, com hifen, espaco ou nada entre eles.
_CITACAO_PATTERN = re.compile(r"protocolos?\s+([a-z]{2,10})\s*[-_ ]?\s*(\d{1,2})", re.IGNORECASE)

# Nome de arquivo dos protocolos: "protocolo_mama02_conduta_birads" → MAMA-02.
_FONTE_PATTERN = re.compile(r"^protocolo_([a-z]+)(\d{1,2})", re.IGNORECASE)

_DISCLAIMER = (
    "\n\n**Atencao:** esta resposta cita protocolo(s) que nao estao entre as fontes "
    "consultadas para responde-la. Confira a referencia no protocolo original antes "
    "de considerar a orientacao como institucional."
)


@dataclass
class CitacaoResult:
    is_flagged: bool
    citacoes_invalidas: list[str] = field(default_factory=list)
    citacoes_validas: list[str] = field(default_factory=list)
    safe_response: str = ""


def _normalizar(texto: str) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto or "") if not unicodedata.combining(c)
    )
    return sem_acento.upper()


def codigo_da_fonte(fonte: str) -> str | None:
    """Converte o nome de arquivo de um protocolo no codigo que ele usa no texto.

    'protocolo_mama02_conduta_birads' → 'MAMA-02'. Devolve None para nomes fora
    desse padrao (nao ha o que comparar).
    """
    match = _FONTE_PATTERN.match(fonte or "")
    if not match:
        return None
    return f"{match.group(1).upper()}-{int(match.group(2)):02d}"


def extrair_citacoes(resposta: str) -> list[str]:
    """Extrai os codigos de protocolo citados na resposta, normalizados.

    dict.fromkeys preserva a ordem de aparicao e remove repeticoes — o mesmo
    protocolo citado tres vezes e um unico codigo a verificar.
    """
    encontrados = [
        f"{_normalizar(sigla)}-{int(numero):02d}"
        for sigla, numero in _CITACAO_PATTERN.findall(resposta or "")
    ]
    return list(dict.fromkeys(encontrados))


def check_citations(resposta: str, fontes: list[str]) -> CitacaoResult:
    """Compara os protocolos citados na resposta com os efetivamente recuperados.

    `fontes` sao os nomes de arquivo vindos de RagResponse.sources. Uma resposta
    sem nenhuma citacao nunca e sinalizada: nao citar protocolo e legitimo (a
    pergunta pode nao ter protocolo aplicavel), o que nao e legitimo e citar um
    que nao foi consultado.
    """
    if not resposta:
        return CitacaoResult(is_flagged=False, safe_response=resposta or "")

    citados = extrair_citacoes(resposta)
    if not citados:
        return CitacaoResult(is_flagged=False, safe_response=resposta)

    disponiveis = {codigo for codigo in (codigo_da_fonte(f) for f in fontes or []) if codigo}

    validas = [c for c in citados if c in disponiveis]
    invalidas = [c for c in citados if c not in disponiveis]

    if not invalidas:
        return CitacaoResult(
            is_flagged=False, citacoes_validas=validas, safe_response=resposta
        )

    return CitacaoResult(
        is_flagged=True,
        citacoes_invalidas=invalidas,
        citacoes_validas=validas,
        safe_response=resposta + _DISCLAIMER,
    )
