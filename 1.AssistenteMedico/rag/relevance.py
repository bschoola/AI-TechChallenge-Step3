"""Corte de relevancia dos chunks recuperados do RAG.

Extraido para um modulo proprio pelo mesmo motivo de agent/scope_guard.py e
agent/overview_parser.py: e logica pura, e mante-la dentro de rag/chain.py a
tornaria intestavel sem instalar torch, transformers e langchain_chroma so para
verificar uma comparacao entre numeros.

## Por que o corte existe

Sem ele, o retriever devolve sempre a mesma quantidade fixa de chunks, existam ou
nao chunks pertinentes a pergunta. Quando nenhum protocolo trata do assunto, ele
devolve os menos irrelevantes — e o modelo, recebendo protocolos que nao se
aplicam, passa a comentar um por um em vez de responder a pergunta.

## Por que o corte e relativo, e nao um limiar fixo

Mesma razao documentada em agent/scope_guard.py: o multilingual-e5-small tem um
piso de similaridade alto entre quaisquer dois textos curtos em portugues. Um
limiar absoluto, sozinho, ou deixa passar tudo ou corta tudo, dependendo do
comprimento da pergunta. Comparar os candidatos entre si cancela esse piso comum
e usa so a diferenca de relevancia entre eles.

O corte relativo, porem, nao resolve o caso em que TODOS os candidatos sao
ruins — ai eles ficam igualmente ruins e proximos entre si, e a margem os aprova
em bloco. Por isso ha tambem um piso absoluto, deliberadamente baixo: ele nao
escolhe entre chunks bons, so descarta a situacao em que nenhum tem relacao com a
pergunta.
"""

# Quantos chunks buscar antes do corte, e quantos podem sobrar depois dele.
RETRIEVER_FETCH_K = 6
RETRIEVER_TOP_K = 4

# Distancia maxima de similaridade em relacao ao melhor chunk para um candidato
# ser aceito.
RELATIVE_SCORE_MARGIN = 0.05

# Piso absoluto: se nem o melhor candidato o supera, nenhum protocolo do acervo
# trata da pergunta.
MIN_SCORE_ABSOLUTO = 0.5


def selecionar_chunks_relevantes(
    candidatos: list[tuple[object, float]],
    margem: float = RELATIVE_SCORE_MARGIN,
    piso: float = MIN_SCORE_ABSOLUTO,
    limite: int = RETRIEVER_TOP_K,
) -> list[tuple[object, float]]:
    """Aplica os dois cortes sobre os candidatos (documento, score), ja ordenados
    do mais para o menos similar.

    Devolve lista vazia quando nem o melhor candidato supera o piso — o caso
    "nenhum protocolo interno trata deste assunto", que rag/chain.py::ask trata
    explicitamente no prompt em vez de despejar chunks irrelevantes.
    """
    if not candidatos:
        return []

    melhor = candidatos[0][1]
    if melhor < piso:
        return []

    return [(doc, score) for doc, score in candidatos if melhor - score <= margem][:limite]
