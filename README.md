# Tech Challenge — OncoTech: Assistente Médico Virtual

Projeto acadêmico da pós-graduação em IA (FIAP). Este repositório contém a entrega da **Fase 3** do Tech Challenge: um assistente médico virtual de apoio à decisão clínica que atende **todas as especialidades** de um hospital fictício, combinando **fine-tuning de LLM (QLoRA)**, **RAG com LangChain**, um **fluxo de decisão em LangGraph**, guardrails de segurança e log de auditoria — servido por uma API FastAPI e consumido por um painel médico em Angular ("Hospital VEinstein").

> As fases anteriores do projeto OncoTech (algoritmo genético para o classificador, API de predição com laudo gerado por LLM) já foram entregues e avaliadas; seus artefatos foram removidos deste repositório para manter o foco na entrega atual.

---

## Entregáveis da Fase 3

| Entregável | Onde |
|---|---|
| Pipeline de fine-tuning (QLoRA) | [`1.AssistenteMedico/finetuning/`](1.AssistenteMedico/finetuning/) |
| Assistente com LangChain (RAG) + fluxo LangGraph | [`1.AssistenteMedico/rag/`](1.AssistenteMedico/rag/), [`1.AssistenteMedico/agent/`](1.AssistenteMedico/agent/) |
| Dataset anonimizado/sintético | [`1.AssistenteMedico/data/raw/`](1.AssistenteMedico/data/raw/) |
| **Relatório técnico** | [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md) |
| Plano de arquitetura e decisões | [`PLANO_Fase3.md`](PLANO_Fase3.md) |

## Estrutura do repositório

```
AI-TechChallenge-Step3/
├── RELATORIO_TECNICO.md      # relatório técnico da entrega (fine-tuning, avaliação, diagramas)
├── PLANO_Fase3.md            # plano técnico: arquitetura, decisões, cronograma, checklist
├── README.md                 # este arquivo
├── 1.AssistenteMedico/       # backend: fine-tuning, RAG, LangGraph e API FastAPI
└── front/                    # frontend Angular ("Hospital VEinstein") — painel médico
```

- Backend — implementação, endpoints e instruções de execução: [`1.AssistenteMedico/README.md`](1.AssistenteMedico/README.md)
- Frontend — painel do médico (pacientes, anamnese, exames, visão geral por IA, chat): [`front/README.md`](front/README.md)

## O que o sistema faz

- **Lista e consulta pacientes** de um prontuário mock (SQLite) com 32 pacientes fictícios cobrindo **30 quadros clínicos distintos** — clínica médica, urgência, ortopedia, pneumologia, cardiologia, endocrinologia, gastroenterologia, urologia, neurologia, otorrino, oftalmologia, dermatologia, infectologia, cirurgia geral, ginecologia e oncologia — cada um com anamnese completa e exames coerentes com o quadro.
- **Visão geral automática por IA** ao abrir a ficha do paciente: sintetiza anamnese e exames em *pontos relevantes* e *pontos de atenção*, em vez de repetir a ficha que já está na tela.
- **Chat com o assistente** sobre o paciente em consulta, com resposta ancorada nos protocolos internos recuperados via RAG e com as fontes citadas.
- **Alerta de exames pendentes** disparado por um nó do grafo antes da geração da resposta.
- **Guardrails em duas camadas**: um guardrail de escopo que recusa perguntas fora do domínio clínico *antes* de expor o prontuário ao modelo, e um filtro de linguagem de prescrição direta aplicado a toda resposta.
- **Auditoria**: uma linha JSON por interação em `logs/audit.jsonl`, com pergunta, fontes, resposta, decisões do grafo e as similaridades que motivaram a classificação de escopo.

## Arquitetura em uma linha

```
Angular  →  FastAPI  →  LangGraph  →  { SQLite (tools) · Chroma (RAG) · Qwen2.5-1.5B + adapter LoRA }
                             ↓
                     guardrails + logs/audit.jsonl
```

Detalhamento — incluindo o diagrama completo do grafo e a divisão de papéis entre fine-tuning, RAG e tools — em [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md).

## Início rápido

### Backend

```bash
cd 1.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python finetuning/prepare_dataset.py         # 1. prepara o dataset de fine-tuning (98 exemplos)
python rag/ingest.py                         # 2. indexa os 32 protocolos no RAG
python finetuning/train_qlora.py             # 3. fine-tuning (ou o notebook Colab — ver README do módulo)
uvicorn api.main:app --reload --port 8001    # 4. sobe a API
```

### Frontend

Com o backend rodando em `http://localhost:8001`:

```bash
cd front
npm install
npm start
```

Acesse `http://localhost:4200`.

### Avaliação e testes

```bash
cd 1.AssistenteMedico
python finetuning/evaluate.py    # compara modelo base vs. fine-tuned e salva as métricas
python -m pytest tests/ -q       # 38 testes de lógica pura (sem carregar LLM)
```

## Cobertura

O escopo é hospitalar geral e as três camadas de dados acompanham esse escopo:

| Camada | Cobertura |
|---|---|
| Prontuário simulado | 32 pacientes · 30 quadros clínicos distintos |
| Protocolos indexados no RAG | 32 protocolos · 122 chunks · clínica médica, urgência, pneumologia, infectologia, gastro, neuro, otorrino, oftalmo, dermato, cirurgia, oncologia e segurança do paciente |
| Dataset de fine-tuning | 98 exemplos — 43 FAQs internas multiespecialidade + 51 de base pública (oncologia mamária) + 5 modelos de documento |

## Aviso

Este sistema é um projeto acadêmico de pós-graduação. O assistente é uma ferramenta de **apoio** à decisão clínica — nenhuma resposta deve ser usada como prescrição direta sem validação de um profissional de saúde habilitado. Os protocolos são sintéticos, escritos para o projeto, e parte das FAQs vem de bases públicas (MedQuAD, PubMedQA) traduzidas automaticamente, sem revisão clínica humana.
