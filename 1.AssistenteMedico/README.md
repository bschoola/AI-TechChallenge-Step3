# 1. Assistente Médico Virtual (Tech Challenge Fase 3)

Projeto **OncoTech** — Tech Challenge Fase 3 da pós-graduação em IA. Um **assistente conversacional** treinado (fine-tuning) com dados internos fictícios de um hospital, capaz de responder dúvidas de médicos, consultar prontuários e sugerir condutas — sempre com validação humana obrigatória. Combina fine-tuning de LLM, RAG (LangChain) e um fluxo de decisão (LangGraph).

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
1.AssistenteMedico/
├── data/
│   ├── raw/            # protocolos e laudos-modelo (sintéticos) + FAQs (MedQuAD/PubMedQA traduzidos)
│   └── processed/       # dataset.jsonl (fine-tuning) + chroma/ + prontuarios_mock.db
├── finetuning/          # pipeline de fine-tuning QLoRA (+ notebook Colab)
├── rag/                 # ingestão e chain de RAG (LangChain)
├── agent/               # grafo de decisão (LangGraph) + guardrails + prontuário mock
├── api/                 # FastAPI expõe o assistente
├── logs/                # audit.jsonl, gerado em runtime (git-ignorado)
└── tests/
```

## Status

Pipeline completo implementado e funcional de ponta a ponta: preparação de dataset, ingestão RAG, fine-tuning (GPU ou fallback CPU), grafo LangGraph (guardrails + log de auditoria) e API FastAPI. Falta apenas `finetuning/evaluate.py` (comparação formal base vs. fine-tuned, usada só no relatório técnico — não faz parte do caminho de execução da API). Checklist completo de entregáveis no `PLANO_Fase3.md`, seção 6.

## Como rodar

```bash
cd 1.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Preparar dataset (protocolos/laudos sintéticos + FAQs MedQuAD/PubMedQA traduzidos)
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

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e se o grafo foi carregado |
| `GET` | `/patients` | Lista pacientes do prontuário mock (com sexo, data de nascimento e idade); aceita `?nome=` para filtrar por parte do nome (case-insensitive) |
| `GET` | `/patients/{paciente_id}/exams` | Lista os exames do paciente com status, data de solicitação e data de realização; 404 se o paciente não existir |
| `GET` | `/exams/{exame_id}` | Detalhe completo de um exame (status, datas, resultado, médico solicitante, observações); 404 se não existir |
| `POST` | `/assistant/ask` | Pergunta do médico (`paciente_id` + `pergunta`) → resposta do assistente |

`GET /patients` existe para o médico localizar o `paciente_id` (usado em `/assistant/ask`) sem precisar consultar o banco diretamente. Exemplos:

```bash
curl http://localhost:8001/patients                 # lista todos
curl "http://localhost:8001/patients?nome=ficticio"  # filtra por parte do nome
curl http://localhost:8001/patients/1/exams          # exames do paciente 1, com status
curl http://localhost:8001/exams/1                   # detalhe completo do exame 1
```

> Se você já tinha rodado o projeto antes dos campos de exame (`data_solicitacao`, `data_realizacao`, `resultado`, `medico_solicitante`, `observacoes`) ou de paciente (`sexo`, `data_nascimento`) existirem, não precisa apagar o `prontuarios_mock.db` — `init_mock_db()` migra as tabelas automaticamente na próxima vez que a API subir.
>
> Nota de design: `idade` não é uma coluna do banco — ela é calculada na hora a partir de `data_nascimento` (`agent/tools.py::_calcular_idade`) toda vez que um paciente é retornado, em vez de ficar guardada e desatualizar com o tempo.

## Fine-tuning: GPU (recomendado) ou CPU (fallback mais lento)

QLoRA "de verdade" (4-bit, via `bitsandbytes`) exige GPU com CUDA. Sem GPU dedicada, o caminho recomendado é o **Google Colab gratuito (GPU T4)**: abra `finetuning/train_qlora_colab.ipynb`, monte o Google Drive, e siga as células — o notebook já cuida de instalar dependências, gerar o dataset e salvar o adapter treinado direto no Drive (assim uma queda de sessão do Colab não perde o progresso).

`finetuning/train_qlora.py` também roda **localmente em CPU** como fallback: se não detectar GPU, carrega o modelo sem quantização (LoRA "cru", não QLoRA) no dtype definido em `config.py::CPU_DTYPE` (bfloat16 por padrão). Com 32GB de RAM isso é tecnicamente viável para o `Qwen2.5-3B-Instruct` (~6GB só para os pesos em bf16), mas é ordens de magnitude mais lento que GPU — um treino de minutos na T4 pode levar horas na CPU. Se a velocidade incomodar, troque `BASE_MODEL_NAME` em `config.py` para `Qwen2.5-1.5B-Instruct` ou `Qwen2.5-0.5B-Instruct` antes de treinar.

## Aviso

Assistente de apoio à decisão clínica. Nenhuma resposta deve ser usada como prescrição direta sem validação de um profissional de saúde habilitado — ver `agent/guardrails.py`.
