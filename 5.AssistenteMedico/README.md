# 5. Assistente Médico Virtual (Fase 3)

Módulo novo do projeto **OncoTech**, entregue na Fase 3 da pós-graduação. Enquanto as fases anteriores tratavam de **predição** (classificador + laudo gerado por LLM), este módulo entrega um **assistente conversacional** treinado com dados internos fictícios de um hospital, capaz de responder dúvidas de médicos, consultar prontuários e sugerir condutas — sempre com validação humana obrigatória.

> Plano completo de arquitetura, decisões técnicas e cronograma em [`PLANO_Fase3.md`](../PLANO_Fase3.md), na raiz do repositório.

## Visão geral

```
Pergunta do médico + ID do paciente
        │
        ▼
   LangGraph (agent/graph.py)
        │
        ├─ verifica exames pendentes (SQLite mock)
        ├─ busca contexto nos protocolos internos (RAG / Chroma)
        ├─ gera resposta com a LLM fine-tuned
        └─ valida segurança (nunca prescreve sem validação humana)
        │
        ▼
Resposta + fontes citadas + log de auditoria
```

## Estrutura

```
5.AssistenteMedico/
├── data/
│   ├── raw/            # protocolos, FAQs, laudos-modelo (dados sintéticos)
│   └── processed/       # dataset.jsonl (fine-tuning) + prontuarios_mock.db
├── finetuning/          # pipeline de fine-tuning QLoRA
├── rag/                 # ingestão e chain de RAG (LangChain)
├── agent/               # grafo de decisão (LangGraph) + guardrails
├── api/                 # FastAPI expõe o assistente
└── tests/
```

## Status

Scaffold inicial — assinaturas de função e TODOs criados a partir do plano técnico. Implementação em andamento (ver checklist no `PLANO_Fase3.md`, seção 6).

## Como rodar (quando implementado)

```bash
cd 5.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Preparar dataset sintético
python finetuning/prepare_dataset.py

# 2. Indexar protocolos no RAG
python rag/ingest.py

# 3. Fine-tuning (recomendado em Colab com GPU — ver finetuning/train_qlora.py)
python finetuning/train_qlora.py

# 4. Subir a API
uvicorn api.main:app --reload --port 8001
```

## Aviso

Assistente de apoio à decisão clínica. Nenhuma resposta deve ser usada como prescrição direta sem validação de um profissional de saúde habilitado — ver `agent/guardrails.py`.
