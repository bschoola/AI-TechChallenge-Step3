"""Testes da checagem de coerencia etaria (agent/demographic_guard.py).

O caso que motivou o modulo: paciente de 66 anos com hernia inguinal, resposta
descrevendo a conduta da hernia inguinal pediatrica.
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from agent.demographic_guard import check_age_coherence

_RESPOSTA_PEDIATRICA = (
    "Para a hernia inguinal em criancas, a conduta indicada e a herniotomia, com "
    "ligadura alta do saco herniario. Em lactentes, a correcao deve ser precoce "
    "pelo risco de encarceramento."
)


# --- caso central ----------------------------------------------------------


def test_resposta_pediatrica_para_paciente_idoso_e_sinalizada():
    r = check_age_coherence(_RESPOSTA_PEDIATRICA, idade=66)
    assert r.is_flagged is True
    assert r.idade_paciente == 66
    assert any("crianca" in t for t in r.termos_incoerentes)


def test_resposta_sinalizada_recebe_a_ressalva_e_preserva_o_texto():
    r = check_age_coherence(_RESPOSTA_PEDIATRICA, idade=66)
    assert r.safe_response.startswith(_RESPOSTA_PEDIATRICA)
    assert "nao corresponde a idade registrada" in r.safe_response


def test_a_mesma_resposta_nao_e_sinalizada_para_paciente_na_faixa():
    # "criancas" (0-12) contem a idade 4, entao a resposta esta orientada pela
    # faixa certa mesmo citando tambem "lactentes" (0-2).
    r = check_age_coherence(_RESPOSTA_PEDIATRICA, idade=4)
    assert r.is_flagged is False
    assert r.safe_response == _RESPOSTA_PEDIATRICA


def test_mencao_contrastiva_com_a_faixa_correta_nao_e_sinalizada():
    texto = (
        "Diferente do que ocorre em criancas, em pacientes idosos a hernia "
        "inguinal costuma ser indireta e a correcao eletiva e a conduta usual."
    )
    r = check_age_coherence(texto, idade=66)
    assert r.is_flagged is False


def test_resposta_ancorada_so_na_faixa_errada_e_sinalizada():
    texto = "Em criancas, a herniotomia com ligadura alta do saco e a conduta."
    r = check_age_coherence(texto, idade=66)
    assert r.is_flagged is True


# --- faixas ----------------------------------------------------------------


def test_termo_de_idoso_em_paciente_jovem_e_sinalizado():
    r = check_age_coherence("Em pacientes idosos, considerar avaliacao geriatrica.", idade=30)
    assert r.is_flagged is True


def test_termo_de_idoso_em_paciente_idoso_nao_e_sinalizado():
    r = check_age_coherence("Em pacientes idosos, considerar avaliacao geriatrica.", idade=72)
    assert r.is_flagged is False


def test_adolescente_em_paciente_adulto_e_sinalizado():
    r = check_age_coherence("Conduta usual em adolescentes com esse quadro.", idade=45)
    assert r.is_flagged is True


def test_recem_nascido_em_paciente_adulto_e_sinalizado():
    r = check_age_coherence("Em recem-nascidos, a abordagem e diferente.", idade=66)
    assert r.is_flagged is True


def test_deteccao_funciona_com_e_sem_acento():
    com = check_age_coherence("Conduta pediátrica indicada.", idade=66)
    sem = check_age_coherence("Conduta pediatrica indicada.", idade=66)
    assert com.is_flagged and sem.is_flagged


# --- ausencia de incoerencia ----------------------------------------------


def test_resposta_sem_termo_etario_nao_e_sinalizada():
    texto = "Hernia inguinal redutivel: encaminhamento para avaliacao cirurgica eletiva."
    r = check_age_coherence(texto, idade=66)
    assert r.is_flagged is False
    assert r.safe_response == texto


def test_idade_desconhecida_desativa_a_checagem():
    # Paciente sem data de nascimento registrada: sem idade real nao ha com o que
    # comparar, e sinalizar seria arbitrario.
    r = check_age_coherence(_RESPOSTA_PEDIATRICA, idade=None)
    assert r.is_flagged is False
    assert r.safe_response == _RESPOSTA_PEDIATRICA


def test_resposta_vazia_nao_quebra():
    r = check_age_coherence("", idade=66)
    assert r.is_flagged is False
    assert r.safe_response == ""


# --- relato -----------------------------------------------------------------


def test_termos_sao_reportados_sem_duplicata():
    texto = "Em criancas e em outras criancas com o mesmo quadro, a conduta pediatrica difere."
    r = check_age_coherence(texto, idade=66)
    assert len(r.termos_incoerentes) == len(set(r.termos_incoerentes))


def test_varios_termos_distintos_sao_reportados():
    texto = "Em lactentes e adolescentes a conduta difere."
    r = check_age_coherence(texto, idade=66)
    assert len(r.termos_incoerentes) >= 2
