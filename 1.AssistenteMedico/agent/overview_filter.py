"""Filtro de ancoragem dos pontos da 'visao geral' automatica.

Ultima camada antes de a visao geral chegar ao medico. As camadas anteriores
resolvem problemas de FORMA -- agent/overview_parser.py quebra a resposta nas
duas secoes e em itens; agent/overview_summarizer.py funde itens que repetem o
mesmo template. Nenhuma delas olha para o CONTEUDO, e e no conteudo que esta o
risco real de um assistente clinico: um item plausivel na forma e inventado no
fundo.

## O problema que este modulo resolve

Um modelo pequeno, ao qual se pede sintese analitica sobre um bloco de texto
longo, pode degenerar em associacao livre: parte de um achado real do paciente
e deriva por vizinhanca semantica para termos cada vez mais distantes, ate
produzir morfologia inventada. O padrao observado tem tres marcas:

1. **Volume**: dezenas ou centenas de itens, apesar de o prompt pedir ate 5.
2. **Deriva tematica**: itens sem nenhuma relacao com o prontuario do paciente.
3. **Familias morfologicas**: sequencias de variacoes do mesmo radical
   ("carcinoma X", "carcinoma Y", ...; "fibrose", "fibrotic", "fibrilari", ...).

`no_repeat_ngram_size` nao contem esse padrao, porque cada item e uma sequencia
de tokens diferente. `merge_similar_bullets` tambem nao, porque exige prefixo e
sufixo comuns em nivel de PALAVRA -- itens de uma ou duas palavras nao formam
template.

## A regra

Um ponto de atencao so pode ser exibido se for rastreavel ao prontuario deste
paciente. Cada item precisa compartilhar ao menos um termo de conteudo com o
contexto clinico montado por agent/tools.py::montar_contexto_clinico. Itens que
falham sao descartados: nao ha como um medico distinguir, numa lista, o achado
real do termo alucinado, entao a lista precisa carregar essa garantia.

A comparacao e feita por radical (prefixo de RADICAL_LEN caracteres, sem acento
e sem caixa) para tolerar flexao -- "hernia"/"hernias", "renal"/"renais" -- sem
depender de um lematizador.

## O custo assumido

A regra descarta tambem inferencia clinica legitima que use vocabulario ausente
da ficha (ex.: "risco de encarceramento" para uma hernia inguinal, quando a
ficha nao contem a palavra "encarceramento"). E uma perda real e consciente: num
apoio a decisao clinica, exibir um achado fabricado custa mais do que omitir um
achado correto que o medico ja obtem lendo a ficha e o protocolo. O numero de
itens descartados vai para o log de auditoria, o que torna esse custo
mensuravel em vez de invisivel.

Quando a geracao e classificada como degenerada (ver `_degenerado`), o
resultado inteiro e descartado, nao apenas os itens sem ancora. Uma lista
parcial extraida de uma geracao degenerada nao e confiavel: os poucos itens
ancorados podem ter casado por coincidencia lexical, e o medico nao tem como
saber de qual regime de geracao cada item veio.
"""

import re
import unicodedata
from dataclasses import dataclass, field

# Numero maximo de itens exibidos por secao. O prompt (rag/chain.py::
# OVERVIEW_INSTRUCTION) ja pede "ate 5 itens"; aqui isso passa a ser garantido
# em codigo, e nao apenas solicitado ao modelo.
MAX_ITENS = 5

# Tamanho do radical usado na comparacao com o contexto do paciente. 5 tolera a
# flexao comum do portugues sem colapsar termos distintos ("cardi" ainda separa
# "cardiaco" de "renal", mas une "cardiaco"/"cardiacos").
RADICAL_LEN = 5

# Comprimento minimo de uma palavra para ser considerada termo de conteudo.
MIN_PALAVRA_CONTEUDO = 4

# A partir de quantos itens brutos numa secao a geracao e considerada
# degenerada. O prompt pede 5; 12 da folga confortavel para o modelo exceder um
# pouco sem acionar o descarte total.
LIMITE_ITENS_DEGENERACAO = 12

# Proporcao minima de itens ancorados para a secao ser considerada confiavel,
# aplicada apenas quando ha itens suficientes para a proporcao significar algo.
MIN_PROPORCAO_ANCORADA = 0.34
MIN_ITENS_PARA_PROPORCAO = 5

# A partir de quantos itens com o mesmo radical inicial a sequencia e tratada
# como familia morfologica (deriva do modelo), preservando apenas o primeiro.
LIMITE_FAMILIA_RADICAL = 3

# Frase que o proprio prompt manda o modelo escrever quando nao ha nada a
# sinalizar. Nao tem -- e nao deve ter -- ancora no prontuario.
_SENTINELAS = ("nenhum ponto de atencao identificado", "nenhum ponto de atencao")

_PALAVRAS_VAZIAS = {
    "para", "pela", "pelo", "pelas", "pelos", "sobre", "como", "quando", "onde",
    "esse", "essa", "esses", "essas", "este", "esta", "estes", "estas", "isso",
    "aquele", "aquela", "seus", "suas", "dele", "dela", "deles", "delas",
    "mais", "menos", "muito", "muita", "pouco", "pouca", "todo", "toda",
    "todos", "todas", "outro", "outra", "outros", "outras", "cada", "entre",
    "porem", "portanto", "tambem", "apenas", "ainda", "sendo", "pode", "podem",
    "deve", "devem", "havia", "houve", "sido", "sera", "foram", "estao",
    "paciente", "pacientes", "relata", "relatou", "apresenta", "apresentou",
    "nao", "sem", "com", "dos", "das", "nos", "nas", "que", "por", "uma", "seu",
}

_TOKEN_PATTERN = re.compile(r"[0-9a-zA-ZÀ-ÿ]+")


@dataclass
class ResultadoFiltro:
    """Itens aprovados e a contabilidade do que foi descartado, para auditoria."""

    itens: list[str] = field(default_factory=list)
    total_bruto: int = 0
    descartados_sem_ancora: int = 0
    descartados_por_familia: int = 0
    descartados_por_limite: int = 0
    degenerado: bool = False


def _normalizar(texto: str) -> str:
    sem_acento = "".join(
        c for c in unicodedata.normalize("NFKD", texto or "") if not unicodedata.combining(c)
    )
    return sem_acento.lower()


def _termos_conteudo(texto: str) -> list[str]:
    """Palavras que carregam conteudo: descarta as muito curtas e as de uso
    generico clinico ('paciente', 'apresenta'), que casariam com qualquer ficha
    e tornariam a ancoragem inutil.
    """
    return [
        token
        for token in _TOKEN_PATTERN.findall(_normalizar(texto))
        if len(token) >= MIN_PALAVRA_CONTEUDO and token not in _PALAVRAS_VAZIAS
    ]


def _radicais(texto: str) -> set[str]:
    return {termo[:RADICAL_LEN] for termo in _termos_conteudo(texto)}


def _e_sentinela(item: str) -> bool:
    normalizado = _normalizar(item).strip().rstrip(".")
    return any(normalizado.startswith(s) for s in _SENTINELAS)


def _ancorado(item: str, radicais_contexto: set[str]) -> bool:
    """True se o item compartilha ao menos um radical de conteudo com o
    contexto clinico do paciente.
    """
    radicais_item = _radicais(item)
    if not radicais_item:
        return False
    return bool(radicais_item & radicais_contexto)


def _colapsar_familias(itens: list[str]) -> tuple[list[str], int]:
    """Reduz sequencias de itens que comecam pelo mesmo radical a um unico
    representante -- a assinatura da deriva morfologica descrita no docstring
    do modulo. Preserva a ordem original e o primeiro membro de cada familia.
    """
    contagem: dict[str, int] = {}
    for item in itens:
        termos = _termos_conteudo(item)
        if termos:
            radical = termos[0][:RADICAL_LEN]
            contagem[radical] = contagem.get(radical, 0) + 1

    familias = {r for r, n in contagem.items() if n >= LIMITE_FAMILIA_RADICAL}
    if not familias:
        return list(itens), 0

    vistos: set[str] = set()
    mantidos: list[str] = []
    descartados = 0
    for item in itens:
        termos = _termos_conteudo(item)
        radical = termos[0][:RADICAL_LEN] if termos else ""
        if radical in familias:
            if radical in vistos:
                descartados += 1
                continue
            vistos.add(radical)
        mantidos.append(item)
    return mantidos, descartados


def _degenerado(total_bruto: int, ancorados: int) -> bool:
    """Classifica a geracao da secao como degenerada.

    Dois sinais independentes, qualquer um basta:

    - **Volume**: mais itens do que o prompt pede por uma margem larga. Uma
      secao com dezenas de itens nao e uma sintese, e um despejo.
    - **Deriva**: proporcao muito baixa de itens ancorados no prontuario, o que
      indica que a geracao perdeu o vinculo com o caso. So se aplica a partir de
      MIN_ITENS_PARA_PROPORCAO itens, porque com poucos itens a proporcao e
      ruidosa demais para decidir.
    """
    if total_bruto > LIMITE_ITENS_DEGENERACAO:
        return True
    if total_bruto >= MIN_ITENS_PARA_PROPORCAO:
        return (ancorados / total_bruto) < MIN_PROPORCAO_ANCORADA
    return False


def filtrar_pontos(
    bullets: list[str],
    contexto_clinico: str | None,
    max_itens: int = MAX_ITENS,
) -> ResultadoFiltro:
    """Aplica a ancoragem no prontuario, o colapso de familias morfologicas e o
    teto de itens a uma secao ja parseada da visao geral.

    contexto_clinico None ou vazio (paciente sem historico nem anamnese
    registrados) desativa a ancoragem: sem ficha nao ha com o que comparar, e
    descartar tudo seria arbitrario. As demais protecoes -- familias e teto --
    continuam valendo.
    """
    resultado = ResultadoFiltro(total_bruto=len(bullets))
    if not bullets:
        return resultado

    itens = [item.strip() for item in bullets if item and item.strip()]

    # A frase-sentinela de "nada a sinalizar" atravessa o filtro intacta: e uma
    # resposta valida do prompt, nao um achado clinico a ser ancorado.
    if len(itens) == 1 and _e_sentinela(itens[0]):
        resultado.itens = itens
        return resultado

    radicais_contexto = _radicais(contexto_clinico or "")
    if radicais_contexto:
        ancorados = [item for item in itens if _ancorado(item, radicais_contexto)]
        resultado.descartados_sem_ancora = len(itens) - len(ancorados)
    else:
        ancorados = itens

    resultado.degenerado = _degenerado(resultado.total_bruto, len(ancorados))
    if resultado.degenerado:
        # Descarte total: ver "Quando a geracao e classificada como degenerada"
        # no docstring do modulo.
        resultado.itens = []
        return resultado

    mantidos, descartados_familia = _colapsar_familias(ancorados)
    resultado.descartados_por_familia = descartados_familia

    if len(mantidos) > max_itens:
        resultado.descartados_por_limite = len(mantidos) - max_itens
        mantidos = mantidos[:max_itens]

    resultado.itens = mantidos
    return resultado
