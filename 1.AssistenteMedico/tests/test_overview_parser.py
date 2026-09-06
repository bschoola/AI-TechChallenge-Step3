"""Testes de agent/overview_parser.py -- inclui regressao dos casos reais
observados em logs/audit.jsonl (ver discussao com o usuario)."""

from agent.overview_parser import extract_bullets, parse_overview_sections


def test_formato_bem_formado_com_marcadores():
    texto = (
        "RELEVANTE:\n"
        "- Hipertensao controlada\n"
        "- Historico familiar de diabetes\n\n"
        "ATENCAO:\n"
        "- Exame pendente ha 3 semanas\n"
    )

    relevantes, atencao = parse_overview_sections(texto)

    assert relevantes == ["Hipertensao controlada", "Historico familiar de diabetes"]
    assert atencao == ["Exame pendente ha 3 semanas"]


def test_regressao_relevantes_no_plural_nao_quebra_mais_o_parsing():
    """Caso real do audit.jsonl: o modelo escreveu 'RELEVANTES:' (plural) e a
    v1 do regex (so aceitava 'RELEVANTE:' singular) nao encontrava a secao,
    caindo no fallback que devolvia o texto INTEIRO -- com os dois rotulos
    ainda visiveis -- como um unico item, sem nenhuma divisao. Esse era o
    "texto sem formatar" relatado pelo usuario.
    """
    texto = (
        "RELEVANTES: \n"
        "Febre alta, dor gargantona, mIALGIA, coriza \n\n"
        "ATENÇÃO: \n"
        "Dor de cabeça, dificuldade respiratoria, hipertensão arterial, "
        "insuficiência renal, consulta urgente."
    )

    relevantes, atencao = parse_overview_sections(texto)

    # Antes da correcao isso vinha como uma lista de 1 item com o texto
    # inteiro (rotulos inclusos). Agora deve separar as duas secoes E quebrar
    # a enumeracao "CSV" de cada uma em itens individuais.
    assert relevantes == ["Febre alta", "dor gargantona", "mIALGIA", "coriza"]
    assert "RELEVANTES" not in " ".join(relevantes)
    assert len(atencao) > 1
    assert "ATENÇÃO" not in " ".join(atencao)


def test_regressao_itens_sem_marcador_um_por_linha():
    """Caso real: o modelo escreveu 'RELEVANTES:' com itens separados por
    virgula MAS quebrados em duas linhas -- extract_bullets deve tratar cada
    linha nao vazia como um item antes de tentar separar por virgula.
    """
    texto = (
        "RELEVANTES: Febrilidade alta, Dolorosa garganta,\n"
        "Dores músculo-esqueléticas, Congestionado nasal,\n\n"
        "ATENÇÃO: Excesso de paracetamól, Falta de histórico médico, \n"
        "Fevereira persistente após 5dias, Pontuação respiratória normalizada,\n"
        "Ausência de outros signos clínicos relevantes, Paciente não fumador."
    )

    relevantes, atencao = parse_overview_sections(texto)

    assert len(relevantes) >= 2
    assert len(atencao) >= 2


def test_extract_bullets_marcadores_tem_prioridade_sobre_csv():
    texto = "- Item um, com virgula interna\n- Item dois"
    assert extract_bullets(texto) == ["Item um, com virgula interna", "Item dois"]


def test_extract_bullets_csv_precisa_de_pelo_menos_tres_itens():
    # Uma frase comum com UMA virgula nao deveria virar dois bullets.
    texto = "Pressao controlada, sem intercorrencias"
    assert extract_bullets(texto) == ["Pressao controlada, sem intercorrencias"]


def test_extract_bullets_csv_com_tres_ou_mais_itens_separa():
    texto = "Febre, tosse, dor de garganta, coriza"
    assert extract_bullets(texto) == ["Febre", "tosse", "dor de garganta", "coriza"]


def test_parse_overview_sections_sem_nenhum_rotulo_ainda_tenta_estruturar():
    texto = "Febre, tosse, dor de garganta, coriza"

    relevantes, atencao = parse_overview_sections(texto)

    assert relevantes == ["Febre", "tosse", "dor de garganta", "coriza"]
    assert atencao == []


def test_extract_bullets_ponto_e_virgula_como_separador():
    """Caso real do audit.jsonl: o modelo separou a enumeracao com ';' em vez
    de ',' -- sem esse fallback o item inteiro (dezenas de "ausencia de X")
    virava um unico bullet gigante.
    """
    texto = (
        "Fim de ciclo de paracetamol; nao ha evidencias de pneumonia; "
        "ausencia de sinal de desconforto respiratorio; hipotensao arterial leve"
    )

    assert extract_bullets(texto) == [
        "Fim de ciclo de paracetamol",
        "nao ha evidencias de pneumonia",
        "ausencia de sinal de desconforto respiratorio",
        "hipotensao arterial leve",
    ]


def test_extract_bullets_ponto_e_virgula_unico_nao_separa():
    texto = "Pressao controlada; sem intercorrencias"
    assert extract_bullets(texto) == ["Pressao controlada; sem intercorrencias"]


def test_parse_overview_sections_texto_vazio():
    assert parse_overview_sections("") == ([], [])
    assert parse_overview_sections(None) == ([], [])


def test_parse_overview_sections_aplica_merge_de_bullets_repetidos():
    texto = (
        "RELEVANTE:\n"
        "- Paciente não apresenta sinais de comprometimento cardiovascular\n"
        "- Paciente não apresenta sinais de comprometimento neurológico\n\n"
        "ATENCAO:\n"
        "- Nenhum ponto de atencao identificado.\n"
    )

    relevantes, _ = parse_overview_sections(texto)

    assert relevantes == [
        "Paciente não apresenta sinais de comprometimento cardiovascular, neurológico."
    ]
