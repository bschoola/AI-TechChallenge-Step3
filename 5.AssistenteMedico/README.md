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

# 3. Fine-tuning — recomendado em GPU (Colab), funciona em CPU como fallback (mais lento).
# Opção A (recomendada): abra finetuning/train_qlora_colab.ipynb no Google Colab
#   (GPU T4 gratuita) e rode as células em ordem.
# Opção B (CPU local, precisa de bastante RAM — ver seção abaixo):
python finetuning/train_qlora.py

# 4. Subir a API
uvicorn api.main:app --reload --port 8001
```

## Fine-tuning: GPU (recomendado) ou CPU (fallback mais lento)

QLoRA "de verdade" (4-bit, via `bitsandbytes`) exige GPU com CUDA. Sem GPU dedicada, o caminho recomendado é o **Google Colab gratuito (GPU T4)**: abra `finetuning/train_qlora_colab.ipynb`, monte o Google Drive, e siga as células — o notebook já cuida de instalar dependências, gerar o dataset e salvar o adapter treinado direto no Drive (assim uma queda de sessão do Colab não perde o progresso).

`finetuning/train_qlora.py` também roda **localmente em CPU** como fallback: se não detectar GPU, carrega o modelo sem quantização (LoRA "cru", não QLoRA) no dtype definido em `config.py::CPU_DTYPE` (bfloat16 por padrão). Com 32GB de RAM isso é tecnicamente viável para o `Qwen2.5-3B-Instruct` (~6GB só para os pesos em bf16), mas é ordens de magnitude mais lento que GPU — um treino de minutos na T4 pode levar horas na CPU. Se a velocidade incomodar, troque `BASE_MODEL_NAME` em `config.py` para `Qwen2.5-1.5B-Instruct` ou `Qwen2.5-0.5B-Instruct` antes de treinar.

## Aviso

Assistente de apoio à decisão clínica. Nenhuma resposta deve ser usada como prescrição direta sem validação de um profissional de saúde habilitado — ver `agent/guardrails.py`.
