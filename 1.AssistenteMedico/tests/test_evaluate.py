"""Testes da logica pura de finetuning/evaluate.py.

Nenhum destes testes carrega o modelo base, o adapter LoRA ou o modelo de
embeddings — so exercitam as funcoes de metrica e de renderizacao, que sao as
que podem quebrar silenciosamente e contaminar os numeros do relatorio tecnico.
"""

import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
for _path in (str(_BASE_DIR), str(_BASE_DIR / "finetuning")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import evaluate as ev


# --- distinct_n: detector de degeneracao por repeticao -----------------------


def test_distinct_n_texto_variado_e_1():
    texto = "o paciente apresenta febre alta tosse seca e dispneia ha tres dias"
    assert ev.distinct_n(texto, 3) == 1.0


def test_distinct_n_detecta_loop_de_repeticao():
    # Reproduz o padrao real observado em logs/audit.jsonl (loop do modelo).
    texto = "hipovitaminose B32 hipovitaminose B32 hipovitaminose B32 hipovitaminose B32"
    assert ev.distinct_n(texto, 3) < 0.5


def test_distinct_n_texto_curto_demais_nao_penaliza():
    assert ev.distinct_n("sem alteracoes", 3) == 1.0


def test_distinct_n_string_vazia():
    assert ev.distinct_n("", 3) == 1.0


# --- ressalva de validacao humana -------------------------------------------


def test_tem_ressalva_reconhece_medico_responsavel():
    assert ev.tem_ressalva("Sugestao a ser validada por um medico responsavel.")


def test_tem_ressalva_reconhece_com_acento():
    assert ev.tem_ressalva("Recomenda-se avaliação médica antes de qualquer conduta.")


def test_tem_ressalva_falso_em_afirmacao_seca():
    assert not ev.tem_ressalva("O paciente apresenta febre de 38.5 graus.")


# --- guardrail (reusa agent/guardrails.py, o mesmo do runtime) --------------


def test_foi_sinalizada_pega_prescricao_direta():
    assert ev.foi_sinalizada("Prescrevo dipirona para o paciente.")


def test_foi_sinalizada_nao_dispara_em_sugestao_enquadrada():
    assert not ev.foi_sinalizada("Considere avaliar o quadro conforme o protocolo MAMA-01.")


# --- agregacao ---------------------------------------------------------------


def _item(sim, flag, ressalva, distinct, palavras):
    return {
        "similaridade_referencia": sim,
        "sinalizada_pelo_guardrail": flag,
        "tem_ressalva_validacao": ressalva,
        "distinct_3": distinct,
        "comprimento_palavras": palavras,
    }


def test_agregar_calcula_medias():
    itens = [_item(0.8, True, False, 1.0, 10), _item(0.6, False, True, 0.5, 20)]
    m = ev.agregar("teste", itens)
    assert m.n_respostas == 2
    assert m.similaridade_referencia == 0.7
    assert m.taxa_guardrail == 0.5
    assert m.taxa_ressalva == 0.5
    assert m.distinct_3 == 0.75
    assert m.comprimento_palavras == 15.0


def test_agregar_lista_vazia_nao_divide_por_zero():
    # Acontece de verdade quando se roda com --skip-base / --skip-finetuned.
    m = ev.agregar("vazio", [])
    assert m.n_respostas == 0
    assert m.similaridade_referencia == 0.0


# --- renderizacao Markdown ---------------------------------------------------


def _report(base_itens, ft_itens):
    return {
        "gerado_em": "2026-09-06T00:00:00+00:00",
        "modelo_base": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": "/tmp/adapter",
        "n_perguntas": 1,
        "max_new_tokens": 400,
        "geracao": "greedy (do_sample=False)",
        "agregado": {
            "base": ev.agregar("base", base_itens).__dict__,
            "finetuned": ev.agregar("finetuned", ft_itens).__dict__,
        },
        "itens": [
            {
                "instruction": "Qual a conduta para BI-RADS 4?",
                "resposta_esperada": "Referencia.",
                "resposta_base": "Resposta base.",
                "resposta_finetuned": "Resposta fine-tuned.",
                "metricas_base": {},
                "metricas_finetuned": {},
            }
        ],
    }


def test_to_markdown_tem_todas_as_metricas_e_o_delta():
    md = ev.to_markdown(_report([_item(0.5, True, False, 1.0, 10)], [_item(0.9, False, True, 1.0, 20)]))
    for _, rotulo, _ in ev._METRICAS_TABELA:
        assert rotulo in md
    assert "+0.4" in md  # delta de similaridade (0.9 - 0.5)
    assert "-1.0" in md  # delta da taxa de guardrail (0.0 - 1.0)


def test_to_markdown_marca_modelo_pulado():
    report = _report([], [_item(0.9, False, True, 1.0, 20)])
    report["itens"][0]["resposta_base"] = None
    md = ev.to_markdown(report)
    assert "_(não gerado)_" in md
