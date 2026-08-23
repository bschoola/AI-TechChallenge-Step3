# Testes sugeridos

Nao ha suite automatizada ainda — este arquivo lista o que vale cobrir, na ordem
de prioridade para a entrega (o que a banca provavelmente vai perguntar/testar no
video).

1. **`agent/guardrails.py::check_response`** — o mais facil de testar isoladamente
   e o mais citado no PDF (seguranca). Casos: resposta com "prescrevo X mg" deve
   sinalizar; resposta sem linguagem de prescricao nao deve sinalizar.
2. **`agent/tools.py`** — `init_mock_db` + `get_exames_pendentes` /
   `get_historico_paciente` contra o banco sintetico.
3. **`agent/graph.py`** — teste de integracao do grafo completo com um paciente
   com exame pendente (deve passar por `emitir_alerta_exame`) e outro sem
   pendencia (deve pular direto para a busca RAG).
4. **`rag/chain.py`** — dado um protocolo de teste indexado, verificar que a
   resposta cita a fonte correta (explainability).
5. **`finetuning/evaluate.py`** — nao e teste automatizado, mas a saida dele
   (comparacao base vs. fine-tuned) e evidencia a incluir no relatorio tecnico.

Framework sugerido: `pytest`. Adicionar `pytest` ao `requirements.txt` quando a
primeira suite for escrita.
