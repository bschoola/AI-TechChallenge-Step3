"""Testes da verificacao de citacoes de protocolo (agent/citation_guard.py) e do
corte de relevancia da recuperacao (rag/relevance.py).

Os dois casos centrais vem da mesma resposta real: ela citava "Protocolo seg-01:
Hernias inguinais" (codigo existente, assunto trocado, e nao recuperado) e
"Protocolo pneumo-02" (codigo inexistente).
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from agent.citation_guard import (
    check_citations,
    codigo_da_fonte,
    extrair_citacoes,
)
from rag.relevance import selecionar_chunks_relevantes

_FONTES_RECUPERADAS = ["protocolo_cir01_hernia_inguinal", "protocolo_urg01_dor_abdominal"]


# --- conversao de nome de arquivo em codigo -------------------------------


def test_codigo_da_fonte_converte_nome_de_arquivo():
    assert codigo_da_fonte("protocolo_mama02_conduta_birads") == "MAMA-02"
    assert codigo_da_fonte("protocolo_cir01_hernia_inguinal") == "CIR-01"
    assert codigo_da_fonte("protocolo_seg04_contraste") == "SEG-04"


def test_codigo_da_fonte_ignora_nome_fora_do_padrao():
    assert codigo_da_fonte("outro_documento_qualquer") is None
    assert codigo_da_fonte("") is None


# --- extracao das citacoes -------------------------------------------------


def test_extrai_citacao_com_hifen_e_sem_hifen():
    assert extrair_citacoes("Conforme o Protocolo CIR-01 e o protocolo urg01.") == [
        "CIR-01",
        "URG-01",
    ]


def test_extrai_citacao_ignorando_caixa_e_acento():
    assert extrair_citacoes("Segundo o PROTOCOLO Mama-02.") == ["MAMA-02"]


def test_citacao_repetida_aparece_uma_vez():
    assert extrair_citacoes("Protocolo CIR-01 ... protocolo cir 01 ... CIR-01.") == ["CIR-01"]


def test_resposta_sem_citacao_devolve_lista_vazia():
    assert extrair_citacoes("Encaminhamento para avaliacao cirurgica eletiva.") == []


# --- casos centrais --------------------------------------------------------


def test_codigo_inexistente_e_sinalizado():
    # "pneumo-02" nao existe no acervo (os de pneumologia sao PNE-01..03).
    r = check_citations("Protocolo pneumo-02 orienta monitoramento.", _FONTES_RECUPERADAS)
    assert r.is_flagged is True
    assert "PNEUMO-02" in r.citacoes_invalidas


def test_codigo_real_mas_nao_recuperado_e_sinalizado():
    # SEG-01 existe, mas trata de alergias e nao foi recuperado para esta pergunta.
    r = check_citations("Protocolo seg-01: hernias inguinais.", _FONTES_RECUPERADAS)
    assert r.is_flagged is True
    assert "SEG-01" in r.citacoes_invalidas


def test_resposta_sinalizada_recebe_ressalva_e_preserva_o_texto():
    texto = "Protocolo pneumo-02 orienta monitoramento."
    r = check_citations(texto, _FONTES_RECUPERADAS)
    assert r.safe_response.startswith(texto)
    assert "nao estao entre as fontes consultadas" in r.safe_response


# --- casos legitimos -------------------------------------------------------


def test_citacao_de_protocolo_recuperado_nao_e_sinalizada():
    r = check_citations(
        "Conduta eletiva conforme o Protocolo CIR-01.", _FONTES_RECUPERADAS
    )
    assert r.is_flagged is False
    assert r.citacoes_validas == ["CIR-01"]


def test_resposta_sem_citacao_nunca_e_sinalizada():
    # Nao citar protocolo e legitimo: a pergunta pode nao ter protocolo aplicavel.
    r = check_citations("Encaminhamento para avaliacao cirurgica eletiva.", [])
    assert r.is_flagged is False


def test_mistura_de_citacao_valida_e_invalida_sinaliza_so_a_invalida():
    r = check_citations(
        "Protocolo CIR-01 orienta a conduta; o protocolo pneumo-02 nao se aplica.",
        _FONTES_RECUPERADAS,
    )
    assert r.is_flagged is True
    assert r.citacoes_validas == ["CIR-01"]
    assert r.citacoes_invalidas == ["PNEUMO-02"]


def test_resposta_vazia_nao_quebra():
    r = check_citations("", _FONTES_RECUPERADAS)
    assert r.is_flagged is False
    assert r.safe_response == ""


def test_sem_fontes_recuperadas_qualquer_citacao_e_invalida():
    r = check_citations("Conforme o Protocolo CIR-01.", [])
    assert r.is_flagged is True
    assert r.citacoes_invalidas == ["CIR-01"]


# --- corte de relevancia da recuperacao ------------------------------------


def test_corte_mantem_apenas_os_proximos_do_melhor():
    candidatos = [("a", 0.90), ("b", 0.88), ("c", 0.70), ("d", 0.65)]
    assert [doc for doc, _ in selecionar_chunks_relevantes(candidatos)] == ["a", "b"]


def test_corte_descarta_tudo_quando_nada_supera_o_piso():
    # Nenhum protocolo trata do assunto: melhor caso e ruim.
    candidatos = [("a", 0.30), ("b", 0.29), ("c", 0.28)]
    assert selecionar_chunks_relevantes(candidatos) == []


def test_corte_respeita_o_limite_de_chunks():
    candidatos = [(f"d{i}", 0.90) for i in range(6)]
    assert len(selecionar_chunks_relevantes(candidatos)) == 4


def test_corte_com_lista_vazia():
    assert selecionar_chunks_relevantes([]) == []
