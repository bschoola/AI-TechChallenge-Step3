"""Avaliacao qualitativa/quantitativa: modelo base vs. modelo fine-tuned.

Objetivo do desafio: "Avaliacao do modelo e analise dos resultados" no relatorio
tecnico. Este script gera as respostas lado a lado para o mesmo conjunto de
perguntas de teste, para comparacao manual (e opcionalmente uma metrica automatica).

Uso:
    python finetuning/evaluate.py
"""

import json

from config import ADAPTER_OUTPUT_DIR, DATASET_VAL_PATH


def load_eval_questions() -> list[dict]:
    """Reaproveita DATASET_VAL_PATH (nunca visto no treino) como conjunto de teste."""
    with open(DATASET_VAL_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def generate_with_base_model(question: str) -> str:
    """Gera resposta com o modelo base (sem adapter LoRA), para comparacao."""
    raise NotImplementedError


def generate_with_finetuned_model(question: str) -> str:
    """Gera resposta com o modelo base + adapter LoRA carregado de ADAPTER_OUTPUT_DIR."""
    raise NotImplementedError


def compare(questions: list[dict]) -> list[dict]:
    """Monta a tabela comparativa base vs. fine-tuned para cada pergunta.

    TODO: opcionalmente calcular uma metrica automatica simples (ex.: similaridade
    de embeddings entre resposta gerada e o 'output' esperado do dataset de
    validacao) alem da comparacao qualitativa manual que vai para o relatorio.
    """
    results = []
    for q in questions:
        results.append(
            {
                "instruction": q["instruction"],
                "resposta_esperada": q["output"],
                "resposta_base": generate_with_base_model(q["instruction"]),
                "resposta_finetuned": generate_with_finetuned_model(q["instruction"]),
            }
        )
    return results


if __name__ == "__main__":
    qs = load_eval_questions()
    out = compare(qs)
    print(json.dumps(out, ensure_ascii=False, indent=2))
