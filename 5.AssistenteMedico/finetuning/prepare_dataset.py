"""Preprocessing, curadoria e split do dataset de fine-tuning.

Le os documentos sinteticos em data/raw/faqs e data/raw/laudos_modelo, normaliza
para o formato instrucao->resposta (estilo Alpaca) e grava train/val em JSONL.

Uso:
    python finetuning/prepare_dataset.py
"""

import json
import random
import unicodedata
from pathlib import Path

from config import DATA_RAW_DIR, DATASET_TRAIN_PATH, DATASET_VAL_PATH, VAL_SPLIT_RATIO, RANDOM_STATE

# Subpastas de data/raw/ que contem documentos no formato instrucao->resposta.
# protocolos/ fica de fora aqui de proposito: aqueles documentos alimentam o RAG
# (rag/ingest.py), nao o fine-tuning — sao "conhecimento a consultar", nao
# "exemplo de como responder".
_SOURCE_SUBDIRS = ["faqs", "laudos_modelo"]

# Cabecalhos aceitos em cada arquivo .md (sem acento, minusculo) e para qual
# campo do exemplo eles mapeiam. Cobre a variacao natural entre FAQs
# ("Pergunta"/"Resposta") e modelos de documento ("Instrucao"/"Resposta").
_INSTRUCTION_HEADERS = {"instrucao", "pergunta", "contexto"}
_OUTPUT_HEADERS = {"resposta", "laudo", "receita", "parecer", "encaminhamento"}

_MIN_OUTPUT_LENGTH = 20


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _parse_sections(text: str) -> dict[str, str]:
    """Divide um arquivo .md em secoes por cabecalho '## Titulo', retornando
    {titulo_normalizado: conteudo}. Normaliza acento/caixa do titulo para casar
    com _INSTRUCTION_HEADERS / _OUTPUT_HEADERS sem exigir grafia exata.
    """
    sections: dict[str, list[str]] = {}
    current_title = None
    for line in text.splitlines():
        if line.startswith("## "):
            current_title = _strip_accents(line[3:].strip().lower())
            sections[current_title] = []
        elif current_title is not None:
            sections[current_title].append(line)
    return {title: "\n".join(lines).strip() for title, lines in sections.items()}


def _parse_md_file(path: Path) -> dict | None:
    sections = _parse_sections(path.read_text(encoding="utf-8"))

    instruction = next((sections[h] for h in _INSTRUCTION_HEADERS if sections.get(h)), None)
    output = next((sections[h] for h in _OUTPUT_HEADERS if sections.get(h)), None)

    if not instruction or not output:
        print(f"Aviso: {path} sem secoes de instrucao/resposta reconhecidas — pulando.")
        return None

    return {"instruction": instruction, "input": "", "output": output}


def load_raw_examples() -> list[dict]:
    """Carrega os pares instrucao/resposta a partir de data/raw/faqs e laudos_modelo.

    Cada arquivo .md deve ter uma secao '## Instrucao' (ou 'Pergunta'/'Contexto')
    e uma secao '## Resposta' (ou 'Laudo'/'Receita'/'Parecer'/'Encaminhamento') —
    ver os exemplos ja gerados em data/raw/faqs/ e data/raw/laudos_modelo/.
    """
    examples = []
    for subdir in _SOURCE_SUBDIRS:
        folder = DATA_RAW_DIR / subdir
        if not folder.exists():
            print(f"Aviso: pasta {folder} nao existe, pulando.")
            continue
        for path in sorted(folder.glob("*.md")):
            parsed = _parse_md_file(path)
            if parsed:
                examples.append(parsed)

    if not examples:
        raise RuntimeError(
            f"Nenhum exemplo encontrado em {[str(DATA_RAW_DIR / s) for s in _SOURCE_SUBDIRS]}. "
            "Confira se os arquivos .md existem e seguem o formato de secoes esperado."
        )
    return examples


def clean_and_dedupe(examples: list[dict]) -> list[dict]:
    """Remove duplicatas exatas e exemplos vazios/curtos demais."""
    seen = set()
    deduped = []
    for ex in examples:
        instruction = " ".join(ex.get("instruction", "").split())
        output = " ".join(ex.get("output", "").split())
        key = (instruction, output)
        if key in seen or not instruction or len(output) < _MIN_OUTPUT_LENGTH:
            continue
        seen.add(key)
        deduped.append({"instruction": instruction, "input": ex.get("input", ""), "output": output})
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
