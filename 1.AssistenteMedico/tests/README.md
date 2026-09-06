# Testes sugeridos

Nao ha suite automatizada ainda — este arquivo lista o que vale cobrir, na ordem
de prioridade para a entrega (o que a banca provavelmente vai perguntar/testar no
video).

1. **`agent/guardrails.py::check_response`** — o mais facil de testar isoladamente
   e o mais citado no PDF (seguranca). Casos: resposta com "prescrevo X mg" deve
   sinalizar; resposta sem linguagem de prescricao nao deve sinalizar.
1b. **`agent/scope_guard.py`** (novo guardrail de escopo do chat) — ja tem uma
   suite em `tests/test_scope_guard.py` cobrindo a matematica de similaridade de
   cosseno e a decisao `is_in_scope` com embeddings falsos (sem depender do
   modelo E5 de verdade). O que falta e validacao "de olho", com o modelo real
   carregado: rodar um punhado de perguntas fora de escopo (viagem, clima,
   programacao, etc.) e perguntas clinicas informais contra
   `agent.scope_guard.scope_score`, conferir os valores no
   `logs/audit.jsonl` (campo `similaridade_escopo`) e ajustar
   `SIMILARITY_THRESHOLD` se necessario.
2. **`agent/tools.py`** — `init_mock_db` + `get_exames_pendentes` /
   `get_historico_paciente` contra o banco sintetico.
3. **`agent/graph.py`** — teste de integracao do grafo completo com um paciente
   com exame pendente (deve passar por `emitir_alerta_exame`) e outro sem
   pendencia (deve pular direto para a busca RAG).
4. **`rag/chain.py`** — dado um protocolo de teste indexado, verificar que a
   resposta cita a fonte correta (explainability).
5. **`finetuning/evaluate.py`** — nao e teste automatizado, mas a saida dele
   (comparacao base vs. fine-tuned) e evidencia a incluir no relatorio tecnico.

Framework: `pytest` (ja adicionado ao `requirements.txt` — `tests/test_scope_guard.py`
e a primeira suite escrita).
