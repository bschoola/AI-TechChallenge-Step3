# Dados processados

Gerados automaticamente — não editar manualmente.

- `dataset_train.jsonl` / `dataset_val.jsonl` — saída de `finetuning/prepare_dataset.py`, formato `{"instruction", "input", "output"}`.
- `prontuarios_mock.db` — banco SQLite sintético, criado por `agent/tools.py` (função `init_mock_db`), com tabelas `pacientes` e `exames`.
- `chroma/` — índice vetorial persistido pelo Chroma, criado por `rag/ingest.py`.

Recomenda-se adicionar `processed/chroma/` e `*.db` ao `.gitignore` do módulo se os arquivos ficarem grandes — manter apenas os scripts que os regeneram.
