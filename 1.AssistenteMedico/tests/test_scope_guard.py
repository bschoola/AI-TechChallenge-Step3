"""Testes de agent/scope_guard.py.

So cobre a matematica pura (cosine_similarity) e a decisao de classify_scope
com embeddings falsos (sem carregar o modelo E5 de verdade, que exige download
do Hugging Face) -- suficiente para garantir que a logica de decisao esta
correta. A qualidade real da classificacao (SCOPE_MARGIN contra perguntas
reais) so pode ser validada com o modelo carregado de verdade -- ver sugestao
no tests/README.md. Foi assim, alias, que a primeira versao deste modulo (um
limiar absoluto contra so ancoras "dentro do escopo") foi descoberta como
insuficiente: perguntas obviamente fora de escopo tinham similaridade
0.82-0.85 contra as ancoras clinicas, acima do limiar de 0.75 -- ver
"Historico" no docstring de agent/scope_guard.py.
"""

import math

import agent.scope_guard as scope_guard
from agent.scope_guard import classify_scope, cosine_similarity


def test_cosine_similarity_vetores_identicos():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)


def test_cosine_similarity_vetores_ortogonais():
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_similarity_vetores_opostos():
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, rel_tol=1e-9)


def test_cosine_similarity_vetor_nulo_nao_gera_erro():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def _fake_embeddings(query_vec, ancoras_dentro, ancoras_fora):
    """Substitui agent.scope_guard._get_embeddings por um objeto falso, e
    intercepta os dois conjuntos de ancoras (dentro/fora) para testar
    classify_scope sem carregar o modelo real. classify_scope chama
    embed_documents primeiro para as ancoras dentro do escopo
    (_get_ancoras_dentro_vecs), depois para as de fora
    (_get_ancoras_fora_vecs) -- diferenciamos pela ORDEM da chamada (as duas
    listas de ancoras tem o mesmo tamanho, entao nao da pra diferenciar so
    pelo tamanho de texts).
    """

    class _Fake:
        def __init__(self):
            self._chamadas = 0

        def embed_query(self, text):
            return query_vec

        def embed_documents(self, texts):
            self._chamadas += 1
            return ancoras_dentro if self._chamadas == 1 else ancoras_fora

    return _Fake()


def _reset_cache(monkeypatch):
    monkeypatch.setattr(scope_guard, "_embeddings", None)
    monkeypatch.setattr(scope_guard, "_ancoras_dentro_vecs", None)
    monkeypatch.setattr(scope_guard, "_ancoras_fora_vecs", None)


def test_classify_scope_dentro_quando_mais_perto_da_ancora_clinica(monkeypatch):
    _reset_cache(monkeypatch)
    # Pergunta identica a uma ancora clinica (sim=1.0) e ortogonal as ancoras
    # fora de escopo (sim=0.0) -> margem = 1.0, bem acima de SCOPE_MARGIN (0.0).
    fake = _fake_embeddings(
            query_vec=[1.0, 0.0],
            ancoras_dentro=[[1.0, 0.0]] * len(scope_guard.ANCORAS_DENTRO_DO_ESCOPO),
            ancoras_fora=[[0.0, 1.0]] * len(scope_guard.ANCORAS_FORA_DO_ESCOPO),
    )
    monkeypatch.setattr(scope_guard, "_get_embeddings", lambda: fake)

    resultado = classify_scope("Qual o historico deste paciente?")

    assert resultado.dentro_do_escopo is True
    assert math.isclose(resultado.similaridade_dentro, 1.0, rel_tol=1e-9)
    assert math.isclose(resultado.similaridade_fora, 0.0, abs_tol=1e-9)
    assert math.isclose(resultado.margem, 1.0, rel_tol=1e-9)


def test_classify_scope_fora_quando_mais_perto_da_ancora_fora_de_escopo(monkeypatch):
    _reset_cache(monkeypatch)
    fake = _fake_embeddings(
            query_vec=[0.0, 1.0],
            ancoras_dentro=[[1.0, 0.0]] * len(scope_guard.ANCORAS_DENTRO_DO_ESCOPO),
            ancoras_fora=[[0.0, 1.0]] * len(scope_guard.ANCORAS_FORA_DO_ESCOPO),
    )
    monkeypatch.setattr(scope_guard, "_get_embeddings", lambda: fake)

    resultado = classify_scope("Devo trocar a vela de ignicao do carro?")

    assert resultado.dentro_do_escopo is False
    assert resultado.margem < 0


def test_classify_scope_reproduz_falha_do_limiar_absoluto_mas_corrige(monkeypatch):
    """Regressao do bug real: uma pergunta com similaridade ALTA (0.85) contra
    as ancoras clinicas (o que enganava o limiar absoluto da v1) mas MAIS alta
    ainda (0.9) contra as ancoras fora de escopo deve ser classificada como
    fora de escopo pela comparacao relativa.
    """
    _reset_cache(monkeypatch)
    fake = _fake_embeddings(
            query_vec=[1.0, 0.0],
            ancoras_dentro=[[0.85, math.sqrt(1 - 0.85**2)]] * len(scope_guard.ANCORAS_DENTRO_DO_ESCOPO),
            ancoras_fora=[[0.9, math.sqrt(1 - 0.9**2)]] * len(scope_guard.ANCORAS_FORA_DO_ESCOPO),
    )
    monkeypatch.setattr(scope_guard, "_get_embeddings", lambda: fake)

    resultado = classify_scope("Devo trocar a vela de ignicao do meu carro a cada 12 meses?")

    assert resultado.dentro_do_escopo is False
    assert resultado.similaridade_dentro > 0.75  # teria passado no limiar absoluto da v1
    assert resultado.margem < 0  # mas perde na comparacao relativa
