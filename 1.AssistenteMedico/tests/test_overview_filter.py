"""Testes do filtro de ancoragem da visao geral (agent/overview_filter.py).

Nenhum teste carrega LLM ou modelo de embeddings: o filtro e logica pura sobre
texto, deliberadamente, para poder ser exercitado nesta velocidade.

O caso central e o `_ATENCAO_DEGENERADA` abaixo -- uma saida real do modelo,
com 150 itens que partem de um achado plausivel e derivam para morfologia
inventada. E o cenario que motivou o modulo.
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from agent.overview_filter import (
    LIMITE_ITENS_DEGENERACAO,
    MAX_ITENS,
    ResultadoFiltro,
    filtrar_pontos,
)

# Contexto no formato de agent/tools.py::montar_contexto_clinico.
_CONTEXTO = (
    "Historico resumido: Paciente com abaulamento inguinal direito ha 8 meses, "
    "redutivel, sem episodios de encarceramento.\n"
    "Queixa principal: aumento de volume na regiao inguinal direita ao esforco.\n"
    "Historia patologica pregressa: hipertensao arterial em tratamento regular.\n"
    "Medicamentos em uso: anti-hipertensivo de uso continuo.\n"
    "Alergias: nega alergias medicamentosas.\n"
    "Exame fisico: abaulamento inguinal direito redutivel a manobra, sem sinais "
    "flogisticos, sem dor a palpacao.\n"
    "Hipotese diagnostica: Hernia inguinal direita, nao complicada no momento.\n"
    "Sinais vitais: PA 138/86 mmHg, FC 76 bpm, SpO2 97%"
)

# Saida real observada, abreviada: mantem a estrutura (achado plausivel inicial,
# deriva tematica, familias morfologicas de radical repetido).
_ATENCAO_DEGENERADA = [
    "Encarceramento",
    "alteracao hemodynamic",
    "insuficiencia renal cronica",
    "diabetes mellitus tipo 2",
    "edema pulmonar",
    "pneumonia bronquilar aguda",
    "pneumonia biliar",
    "pneumonia tuberculosa",
    "pneumonia viral",
    "carcinoma epiteliano",
    "carcinoma colon",
    "carcinoma laringe",
    "meningitis",
    "meningovascular",
    "meningococcemia",
    "fibrose cutanea",
    "fibrilari",
    "filtracao",
    "filtranca",
    "filtrapta",
    "filtreiras",
]


# --- caso central: geracao degenerada -------------------------------------


def test_geracao_degenerada_devolve_lista_vazia():
    r = filtrar_pontos(_ATENCAO_DEGENERADA, _CONTEXTO)
    assert r.degenerado is True
    assert r.itens == []


def test_geracao_degenerada_registra_o_volume_bruto():
    r = filtrar_pontos(_ATENCAO_DEGENERADA, _CONTEXTO)
    assert r.total_bruto == len(_ATENCAO_DEGENERADA)
    assert r.descartados_sem_ancora > 15


def test_volume_excessivo_sozinho_ja_marca_degeneracao():
    # Todos ancorados, mas em quantidade incompativel com uma sintese.
    itens = [f"Hernia inguinal — observacao {i}" for i in range(LIMITE_ITENS_DEGENERACAO + 1)]
    assert filtrar_pontos(itens, _CONTEXTO).degenerado is True


def test_deriva_tematica_marca_degeneracao_mesmo_com_poucos_itens():
    itens = [
        "Hernia inguinal direita redutivel",
        "Carcinoma de laringe",
        "Meningite bacteriana",
        "Leucemia mieloide",
        "Fibrose pulmonar",
    ]
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.degenerado is True
    assert r.itens == []


# --- caso normal: lista curta e ancorada ----------------------------------


def test_lista_curta_e_ancorada_passa_intacta():
    itens = [
        "Hernia inguinal direita redutivel, sem sinais de complicacao.",
        "Hipertensao arterial em tratamento regular.",
        "Nega alergias medicamentosas.",
    ]
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.degenerado is False
    assert r.itens == itens
    assert r.descartados_sem_ancora == 0


def test_item_sem_ancora_no_prontuario_e_descartado():
    itens = [
        "Hernia inguinal direita redutivel.",
        "Hipertensao arterial em tratamento.",
        "Risco de carcinoma de mama a investigar.",
    ]
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.degenerado is False
    assert r.descartados_sem_ancora == 1
    assert not any("carcinoma" in i.lower() for i in r.itens)


def test_ancoragem_tolera_flexao_de_palavra():
    # "medicamentosas" no contexto; "medicamentosa" no item.
    r = filtrar_pontos(["Sem alergia medicamentosa conhecida."], _CONTEXTO)
    assert len(r.itens) == 1


def test_teto_de_itens_e_aplicado():
    # Radicais iniciais distintos de proposito: aqui o que precisa disparar e o
    # teto, nao o colapso de familia morfologica.
    itens = [
        "Hernia inguinal direita redutivel.",
        "Hipertensao arterial em tratamento regular.",
        "Abaulamento aumenta ao esforco.",
        "Alergias medicamentosas negadas.",
        "Medicamentos de uso continuo em curso.",
        "Exame fisico sem sinais flogisticos.",
        "Sinais vitais dentro da faixa esperada.",
    ]
    assert len(itens) == MAX_ITENS + 2
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.descartados_por_familia == 0
    assert len(r.itens) == MAX_ITENS
    assert r.descartados_por_limite == 2


# --- familias morfologicas -------------------------------------------------


def test_familia_de_radical_repetido_e_colapsada():
    itens = [
        "Hernia inguinal redutivel.",
        "Hipertensao arterial controlada.",
        "Hipertensao arterial sistemica.",
        "Hipertensao arterial estagio 1.",
    ]
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.descartados_por_familia == 2
    assert sum(1 for i in r.itens if i.lower().startswith("hipertensao")) == 1


def test_duas_ocorrencias_do_mesmo_radical_nao_formam_familia():
    itens = [
        "Hipertensao arterial controlada.",
        "Hipertensao arterial em tratamento.",
    ]
    r = filtrar_pontos(itens, _CONTEXTO)
    assert r.descartados_por_familia == 0
    assert len(r.itens) == 2


# --- bordas ---------------------------------------------------------------


def test_sentinela_de_nada_a_sinalizar_atravessa_o_filtro():
    r = filtrar_pontos(["Nenhum ponto de atencao identificado."], _CONTEXTO)
    assert r.itens == ["Nenhum ponto de atencao identificado."]
    assert r.descartados_sem_ancora == 0


def test_lista_vazia_devolve_resultado_vazio():
    r = filtrar_pontos([], _CONTEXTO)
    assert r == ResultadoFiltro(itens=[], total_bruto=0)


def test_sem_contexto_clinico_a_ancoragem_e_desativada():
    # Paciente sem historico nem anamnese: nao ha com o que comparar, entao
    # descartar tudo seria arbitrario. Teto e familias continuam valendo.
    itens = ["Achado A relevante", "Achado B relevante"]
    r = filtrar_pontos(itens, None)
    assert r.itens == itens
    assert r.descartados_sem_ancora == 0


def test_itens_em_branco_sao_ignorados():
    r = filtrar_pontos(["Hernia inguinal redutivel.", "   ", ""], _CONTEXTO)
    assert len(r.itens) == 1


def test_palavras_genericas_nao_servem_de_ancora():
    # "paciente"/"apresenta" aparecem em qualquer ficha; ancorar por elas
    # deixaria qualquer alucinacao passar.
    r = filtrar_pontos(["Paciente apresenta neoplasia pancreatica."], _CONTEXTO)
    assert r.itens == []
    assert r.descartados_sem_ancora == 1
