"""Consolida bullets repetitivos da 'visao geral' automatica (agent/nodes.py::
gerar_visao_geral) que seguem o mesmo template, mudando so uma palavra ou
trecho curto -- padrao observado em uso real com o modelo fine-tuned pequeno:

    - Paciente nao apresenta sinais de comprometimento cardiovascular
    - Paciente nao apresenta sinais de comprometimento neurologico
    - Paciente nao apresenta sinais de comprometimento metabolico
    - ... (um item por sistema, ate 8 itens)

vira um unico item:

    - Paciente nao apresenta sinais de comprometimento cardiovascular,
      neurologico, metabolico, ...

O prompt (rag/chain.py::OVERVIEW_INSTRUCTION) ja pede no maximo 5 itens e para
consolidar padroes repetidos, mas o modelo (pequeno, treinado majoritariamente
em QA factual extrativo) nao segue essa instrucao com confiabilidade -- por
isso esta funcao existe como rede de seguranca deterministica em vez de
depender so do prompt.

Heuristica: compara bullets par a par por PREFIXO e SUFIXO comuns (em nivel de
palavra, sem diferenciar maiusculas/minusculas). Se duas linhas compartilham
pelo menos MIN_SHARED_WORDS palavras (prefixo + sufixo somados) e o trecho que
varia entre elas tem no maximo MAX_VARYING_WORDS palavras, elas sao tratadas
como o mesmo template e agrupadas -- mesmo que o trecho variavel fique no meio
da frase (ex.: "Paciente com <X> controlada" com sufixo apos <X>), nao so no
final. Nao e um parafraseador (nao junta frases que dizem a mesma coisa com
palavras diferentes) -- e deliberadamente conservador, para nunca descartar
informacao clinica distinta por engano.
"""

# Minimo de palavras (prefixo + sufixo somados) que duas linhas precisam
# compartilhar para serem consideradas o mesmo template. Um valor baixo demais
# arriscaria fundir frases sobre achados clinicos diferentes que so comecam
# parecido (ex. "Paciente nao apresenta" sozinho, 3 palavras, e comum demais
# para servir de template).
MIN_SHARED_WORDS = 4

# Tamanho maximo (em palavras) do trecho que pode variar entre duas linhas do
# mesmo template. Trechos maiores que isso normalmente indicam frases
# distintas que so comecam/terminam parecido, nao variacoes do mesmo ponto.
MAX_VARYING_WORDS = 6


def _tokenize(bullet: str) -> list[str]:
    return bullet.strip().rstrip(".").split()


def _common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for wa, wb in zip(a, b):
        if wa.lower() != wb.lower():
            break
        n += 1
    return n


def _common_suffix_len(a: list[str], b: list[str], max_len: int) -> int:
    n = 0
    for wa, wb in zip(reversed(a), reversed(b)):
        if n >= max_len or wa.lower() != wb.lower():
            break
        n += 1
    return n


def _template_comum(tokens_a: list[str], tokens_b: list[str]):
    """Descobre um template (prefixo, sufixo) comum entre duas linhas
    tokenizadas. Retorna (prefixo_tokens, sufixo_tokens) ou None se as linhas
    nao compartilharem o suficiente para serem consideradas o mesmo template.
    """
    prefix_len = _common_prefix_len(tokens_a, tokens_b)
    max_suffix = min(len(tokens_a), len(tokens_b)) - prefix_len
    suffix_len = _common_suffix_len(tokens_a, tokens_b, max_suffix) if max_suffix > 0 else 0

    if prefix_len + suffix_len < MIN_SHARED_WORDS:
        return None

    meio_a_len = len(tokens_a) - prefix_len - suffix_len
    meio_b_len = len(tokens_b) - prefix_len - suffix_len
    if not (1 <= meio_a_len <= MAX_VARYING_WORDS):
        return None
    if not (1 <= meio_b_len <= MAX_VARYING_WORDS):
        return None

    prefixo = tokens_a[:prefix_len]
    sufixo = tokens_a[len(tokens_a) - suffix_len:] if suffix_len else []
    return prefixo, sufixo


def _extrai_variacao(tokens: list[str], prefixo: list[str], sufixo: list[str]):
    """Se `tokens` bater exatamente com o template (prefixo/sufixo dados),
    devolve o trecho que varia; senao devolve None.
    """
    pf, sf = len(prefixo), len(sufixo)
    if len(tokens) <= pf + sf:
        return None
    if [t.lower() for t in tokens[:pf]] != [t.lower() for t in prefixo]:
        return None
    if sf and [t.lower() for t in tokens[-sf:]] != [t.lower() for t in sufixo]:
        return None
    meio = tokens[pf: len(tokens) - sf] if sf else tokens[pf:]
    if not (1 <= len(meio) <= MAX_VARYING_WORDS):
        return None
    return meio


def merge_similar_bullets(bullets: list[str]) -> list[str]:
    """Agrupa bullets que repetem o mesmo template e funde cada grupo em um
    unico item, juntando os trechos que variam separados por virgula (ver
    docstring do modulo para o formato exato). Bullets sem par (nenhum outro
    item repete o template) sao devolvidos como estavam, na ordem original.
    """
    if len(bullets) <= 1:
        return list(bullets)

    tokens_list = [_tokenize(b) for b in bullets]
    usados = [False] * len(bullets)
    resultado: list[str] = []

    for i in range(len(bullets)):
        if usados[i]:
            continue

        template = None
        for j in range(i + 1, len(bullets)):
            if usados[j]:
                continue
            template = _template_comum(tokens_list[i], tokens_list[j])
            if template is not None:
                break

        if template is None:
            resultado.append(bullets[i])
            usados[i] = True
            continue

        prefixo, sufixo = template
        variacoes: list[str] = []
        membros: list[int] = []
        for k in range(i, len(bullets)):
            if usados[k]:
                continue
            meio = _extrai_variacao(tokens_list[k], prefixo, sufixo)
            if meio is None:
                continue
            variacoes.append(" ".join(meio))
            membros.append(k)

        for k in membros:
            usados[k] = True

        # dict.fromkeys em vez de set() para preservar a ordem original e
        # descartar variacoes duplicadas (ex.: o modelo repetiu o mesmo item
        # duas vezes).
        variacoes_unicas = list(dict.fromkeys(v for v in variacoes if v))
        prefixo_txt = " ".join(prefixo)
        sufixo_txt = " ".join(sufixo)
        corpo = ", ".join(variacoes_unicas)
        item = " ".join(parte for parte in (prefixo_txt, corpo, sufixo_txt) if parte)
        resultado.append(item.rstrip(".") + ".")

    return resultado
