"""Preprocessing, curadoria e split do dataset de fine-tuning.

Le os documentos sinteticos em data/raw/faqs e data/raw/laudos_modelo, normaliza
para o formato instrucao->resposta (estilo Alpaca) e grava train/val em JSONL.

Uso:
    python finetuning/prepare_dataset.py
"""

import json
import random
from pathlib import Path

from config import DATA_RAW_DIR, DATASET_TRAIN_PATH, DATASET_VAL_PATH, VAL_SPLIT_RATIO, RANDOM_STATE


def load_raw_examples() -> list[dict]:
    """Carrega os pares instrucao/resposta a partir de data/raw/faqs e laudos_modelo.

    TODO: definir o formato de origem exato (ex.: um .md por FAQ, ou um .csv/.jsonl
    unico exportado do processo de geracao sintetica) e implementar o parser aqui.
    Cada exemplo deve virar um dict {"instruction": str, "input": str, "output": str}.
    """
    raise NotImplementedError(
        "Implementar leitura de data/raw/faqs/*.md e data/raw/laudos_modelo/*.md "
        "e conversao para {'instruction', 'input', 'output'}."
    )


def clean_and_dedupe(examples: list[dict]) -> list[dict]:
    """Remove duplicatas exatas e exemplos vazios/curtos demais.

    TODO: normalizar espacos em branco, remover PII residual (checagem extra de
    seguranca, mesmo em dados sinteticos) e descartar exemplos com output < N chars.
    """
    seen = set()
    deduped = []
    for ex in examples:
        key = (ex.get("instruction", "").strip(), ex.get("output", "").strip())
        if key in seen or not key[0] or not key[1]:
            continue
        seen.add(key)
        deduped.append(ex)
    return deduped


def split_train_val(examples: list[dict], val_ratio: float = VAL_SPLIT_RATIO) -> tuple[list[dict], list[dict]]:
    rng = random.Random(RANDOM_STATE)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_ratio))
    return shuffled[cut:], shuffled[:cut]


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    examples = load_raw_examples()
    examples = clean_and_dedupe(examples)
    train, val = split_train_val(examples)
    write_jsonl(train, DATASET_TRAIN_PATH)
    write_jsonl(val, DATASET_VAL_PATH)
    print(f"Dataset preparado: {len(train)} treino / {len(val)} validacao")
    print(f"  -> {DATASET_TRAIN_PATH}")
    print(f"  -> {DATASET_VAL_PATH}")


if __name__ == "__main__":
    main()
