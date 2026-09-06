"""Testes de agent/overview_summarizer.py::merge_similar_bullets."""

from agent.overview_summarizer import merge_similar_bullets


def test_lista_vazia_ou_unico_item_nao_muda():
    assert merge_similar_bullets([]) == []
    assert merge_similar_bullets(["Paciente estavel"]) == ["Paciente estavel"]


def test_funde_o_exemplo_real_do_usuario():
    bullets = [
        "Paciente não apresenta sinais de comprometimento cardiovascular",
        "Paciente não apresenta sinais de comprometimento neurológico",
        "Paciente não apresenta sinais de comprometimento metabólico",
        "Paciente não apresenta sinais de comprometimento imunológico",
        "Paciente não apresenta sinais de comprometimento sistêmico",
        "Paciente não apresenta sinais de comprometimento endocrinológico",
        "Paciente não apresenta sinais de comprometimento dermatológico",
        "Paciente não apresenta sinais de comprometimento urogenital",
    ]

    resultado = merge_similar_bullets(bullets)

    assert resultado == [
        "Paciente não apresenta sinais de comprometimento cardiovascular, "
        "neurológico, metabólico, imunológico, sistêmico, endocrinológico, "
        "dermatológico, urogenital."
    ]


def test_funde_template_com_sufixo_apos_o_trecho_variavel():
    bullets = [
        "Paciente com pressão arterial controlada sem intercorrencias",
        "Paciente com glicemia controlada sem intercorrencias",
    ]

    resultado = merge_similar_bullets(bullets)

    assert resultado == [
        "Paciente com pressão arterial, glicemia controlada sem intercorrencias."
    ]


def test_nao_funde_pontos_clinicos_sem_template_em_comum():
    bullets = [
        "Exame de sangue pendente ha 3 semanas",
        "Alergia a penicilina registrada na anamnese",
        "Paciente relata dor lombar ha 2 dias",
    ]

    assert merge_similar_bullets(bullets) == bullets


def test_nao_funde_quando_prefixo_compartilhado_e_curto_demais():
    # "Paciente nao apresenta" sozinho (3 palavras) e generico demais para
    # servir de template -- nao deveria fundir achados clinicos diferentes.
    bullets = [
        "Paciente não apresenta risco de complicações associado à hipertensão",
        "Paciente não apresenta histórico de cirurgias recentes",
    ]

    assert merge_similar_bullets(bullets) == bullets


def test_funde_apenas_o_subgrupo_que_compartilha_template():
    bullets = [
        "Paciente não apresenta sinais de comprometimento cardiovascular",
        "Paciente não apresenta sinais de comprometimento neurológico",
        "Exame de imagem pendente ha 10 dias",
    ]

    resultado = merge_similar_bullets(bullets)

    assert resultado == [
        "Paciente não apresenta sinais de comprometimento cardiovascular, neurológico.",
        "Exame de imagem pendente ha 10 dias",
    ]


def test_remove_variacao_duplicada_mantendo_ordem():
    bullets = [
        "Paciente não apresenta sinais de comprometimento cardiovascular",
        "Paciente não apresenta sinais de comprometimento neurológico",
        "Paciente não apresenta sinais de comprometimento cardiovascular",
    ]

    resultado = merge_similar_bullets(bullets)

    assert resultado == [
        "Paciente não apresenta sinais de comprometimento cardiovascular, neurológico."
    ]
