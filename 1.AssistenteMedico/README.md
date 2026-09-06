# 1. Assistente Médico Virtual (Tech Challenge Fase 3)

Projeto **OncoTech** — Tech Challenge Fase 3 da pós-graduação em IA (FIAP). Um **assistente conversacional de apoio à decisão clínica** que combina três técnicas com papéis distintos:

- **Fine-tuning (QLoRA)** — ensina *estilo, tom clínico e formato* de resposta.
- **RAG (LangChain + Chroma)** — injeta *conhecimento factual citável* (protocolos internos), que vira a "fonte" exibida na resposta.
- **Tools sobre banco estruturado** — injetam *dado do paciente em tempo real* (histórico, anamnese, exames), que muda a cada consulta e por isso não pode vir do fine-tuning.

Tudo orquestrado por um **grafo de decisão em LangGraph**, com guardrails de segurança e log de auditoria por interação.

> - Relatório técnico completo (fine-tuning, avaliação, diagramas): [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md)
> - Plano de arquitetura e decisões: [`PLANO_Fase3.md`](../PLANO_Fase3.md)

---

## Visão geral do fluxo

```
Pergunta do médico + ID do paciente          |  Abertura da tela do paciente
              │                              │              │
              ▼                              ▼              ▼
                    LangGraph (agent/graph.py)
                              │
    ├─ verifica exames pendentes (SQLite mock) e emite alerta
    ├─ [chat] classifica escopo por embeddings — recusa fora do domínio clínico
    ├─ [chat] recupera protocolos no Chroma (RAG) e gera resposta com a LLM fine-tuned
    ├─ [visão geral] sintetiza anamnese/exames em RELEVANTE / ATENÇÃO (sem RAG)
    ├─ valida segurança (filtro de linguagem de prescrição direta)
    └─ registra a interação completa em logs/audit.jsonl
                              │
                              ▼
            Resposta + fontes citadas + flag de validação humana
```

Diagrama detalhado do grafo, com todos os nós e arestas condicionais, em [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md#4-diagrama-do-fluxo-langchain--langgraph).

## Estrutura

```
1.AssistenteMedico/
├── data/
│   ├── raw/
│   │   ├── faqs/            # 51 pares instrução→resposta (MedQuAD + PubMedQA traduzidos) → fine-tuning
│   │   ├── laudos_modelo/   # 5 modelos de laudo/parecer sintéticos → fine-tuning (formato/tom)
│   │   ├── protocolos/      # 5 protocolos clínicos sintéticos → RAG (não entram no fine-tuning)
│   │   └── _arquivado_sintetico_hospital_vida_nova/   # FAQs sintéticas originais, preservadas
│   └── processed/           # dataset_train/val.jsonl + chroma/ + prontuarios_mock.db + evaluation/
├── finetuning/              # config, preparação do dataset, treino QLoRA (+ notebook Colab), avaliação
├── rag/                     # embeddings E5, ingestão no Chroma, chain de geração
├── agent/                   # grafo LangGraph, nós, guardrails, guardrail de escopo, parsers, tools
├── api/                     # FastAPI
├── tests/                   # 38 testes de lógica pura (sem carregar LLM)
└── logs/                    # audit.jsonl, gerado em runtime (git-ignorado)
```

## Como rodar

```bash
cd 1.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Preparar o dataset de fine-tuning (FAQs + laudos-modelo → JSONL)
python finetuning/prepare_dataset.py

# 2. Indexar os protocolos no RAG (Chroma)
python rag/ingest.py

# 3. Fine-tuning — GPU recomendado, CPU como fallback
#   Opção A (recomendada): abrir finetuning/train_qlora_colab.ipynb no Google Colab (T4 gratuita)
#   Opção B (CPU local, bem mais lento):
python finetuning/train_qlora.py

# 4. Subir a API
uvicorn api.main:app --reload --port 8001
```

O front-end Angular consome a API em `http://localhost:8001` — ver [`../front/README.md`](../front/README.md).

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e se o grafo foi carregado |
| `GET` | `/patients` | Lista os pacientes do prontuário mock (com sexo, data de nascimento e idade calculada); `?nome=` filtra por parte do nome |
| `GET` | `/patients/{id}/exams` | Exames do paciente com status, data de solicitação e de realização; 404 se o paciente não existir |
| `GET` | `/exams/{id}` | Detalhe completo de um exame (resultado, médico solicitante, observações) |
| `GET` | `/patients/{id}/anamnesis` | Ficha de anamnese completa; 404 se o paciente não existir ou não tiver anamnese |
| `POST` | `/assistant/ask` | Pergunta livre do médico (`paciente_id` + `pergunta`) → resposta + fontes citadas |
| `POST` | `/patients/{id}/overview` | Visão geral automática por IA (sem corpo) → `pontos_relevantes` / `pontos_atencao` estruturados |

```bash
curl http://localhost:8001/patients                        # lista os 32 pacientes
curl "http://localhost:8001/patients?nome=ficticio"        # filtra por nome
curl http://localhost:8001/patients/1/exams                # exames do paciente 1
curl http://localhost:8001/patients/1/anamnesis            # anamnese completa
curl -X POST http://localhost:8001/patients/1/overview     # visão geral por IA
curl -X POST http://localhost:8001/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"paciente_id": 1, "pergunta": "Qual a conduta para este achado BI-RADS 4?"}'
```

`POST /assistant/ask` e `POST /patients/{id}/overview` percorrem o **mesmo grafo**, diferenciados pelo campo `tipo_interacao` do estado — o que muda é o nó de geração; os nós de exame pendente, guardrail, validação humana e auditoria são compartilhados.

**Convenção de erros:** `503` quando falta um artefato (índice do RAG ou adapter fine-tuned ainda não gerados), `404` para paciente/exame inexistente, `500` para o resto.

## Dados

| Pasta | Conteúdo | Uso | Origem |
|---|---|---|---|
| `data/raw/faqs/` | 51 pares instrução→resposta sobre oncologia mamária | **Fine-tuning** | MedQuAD (NIH, CC BY 4.0) e PubMedQA (MIT), filtrados para câncer de mama e traduzidos com apoio de LLM |
| `data/raw/laudos_modelo/` | 5 modelos de laudo/parecer/encaminhamento | **Fine-tuning** (formato e tom) | Sintético |
| `data/raw/protocolos/` | 5 protocolos clínicos do hospital fictício | **RAG** (conhecimento citável) | Sintético |
| `data/processed/prontuarios_mock.db` | 32 pacientes, 38 exames, 32 anamneses completas | **Tools** (dado em tempo real) | Sintético, SQLite |

O dataset final de fine-tuning tem **55 exemplos** (47 treino / 8 validação). Atribuição de licença e a ressalva sobre tradução automática sem revisão manual estão em [`data/raw/README.md`](data/raw/README.md) e [`data/raw/faqs/_FONTE.md`](data/raw/faqs/_FONTE.md).

> O seed do prontuário mock (`agent/tools.py::init_mock_db`) é 100% idempotente (`INSERT OR IGNORE` por id): reiniciar a API preenche o que faltar sem duplicar nem sobrescrever, e migra bancos antigos automaticamente. Não é preciso apagar o `.db` ao atualizar o projeto.

## Segurança, guardrails e explainability

| Camada | Onde | O que faz |
|---|---|---|
| **System prompt** | `finetuning/config.py::SYSTEM_PROMPT` | Um único prompt, usado tanto no treino quanto na inferência — o modelo aprende a responder dentro dos mesmos limites que o guardrail depois verifica |
| **Guardrail de escopo** | `agent/scope_guard.py` | Antes de montar o contexto do paciente, classifica a pergunta por similaridade **relativa** de embeddings (âncoras dentro vs. fora do domínio clínico). Fora de escopo → resposta fixa em código, que nunca vê o prontuário |
| **Guardrail de prescrição** | `agent/guardrails.py` | Filtro de padrões de prescrição direta ("prescrevo", "tome X mg"). Nunca descarta a resposta: marca `requer_validacao_humana=True` e acrescenta a ressalva |
| **Validação humana** | `agent/nodes.py::encaminhar_para_validacao_humana` | Nó dedicado no grafo — respostas sinalizadas saem com `status="aguardando_validacao_humana"` |
| **Explainability** | `rag/chain.py` + `agent/nodes.py::log_auditoria` | Toda resposta de chat cita os protocolos recuperados; o log guarda pergunta, fontes, resposta bruta, resposta final, padrões sinalizados e as similaridades do guardrail de escopo |

`logs/audit.jsonl` recebe **uma linha JSON por interação** e nunca derruba a resposta em caso de falha de escrita.

## Avaliação do modelo

```bash
python finetuning/evaluate.py                # base vs. fine-tuned nas 8 perguntas de validação
python finetuning/evaluate.py --limit 3      # amostra menor, para teste rápido
python finetuning/evaluate.py --skip-base    # só o modelo fine-tuned
```

Gera `data/processed/evaluation/evaluation.json` (respostas completas + métricas por item) e `evaluation.md` (tabela comparativa pronta para o relatório). Métricas: similaridade de embeddings com a resposta de referência, taxa de acionamento do guardrail, taxa de ressalva de validação humana, trigramas distintos (detector de loop de repetição) e comprimento médio.

Geração **greedy** (`do_sample=False`) para ser reproduzível — diferente do runtime da API, que usa `temperature=0.3`. Análise dos resultados em [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md#5-avaliação-do-modelo-e-análise-dos-resultados).

## Testes

```bash
python -m pytest tests/ -q     # 38 testes
```

Cobrem lógica pura, sem carregar LLM nem modelo de embeddings: guardrail de escopo, consolidação de bullets repetidos, parsing da visão geral (com regressão de cada caso real observado em produção) e métricas de avaliação.

## Fine-tuning: GPU (recomendado) ou CPU (fallback)

Modelo base atual: **`Qwen/Qwen2.5-1.5B-Instruct`** (Apache 2.0, sem gate de licença), definido em `finetuning/config.py::BASE_MODEL_NAME`.

QLoRA "de verdade" (4-bit via `bitsandbytes`) exige GPU com CUDA. O caminho recomendado é o **Google Colab gratuito (T4)**: abra `finetuning/train_qlora_colab.ipynb`, monte o Drive e rode as células — o adapter treinado é salvo direto no Drive, então uma queda de sessão não perde o progresso.

`finetuning/train_qlora.py` detecta a ausência de GPU e cai para **LoRA sem quantização** em CPU, no dtype de `config.py::CPU_DTYPE` (bfloat16). Funciona, mas é ordens de magnitude mais lento. Para acelerar em hardware próprio, troque `BASE_MODEL_NAME` por `Qwen2.5-0.5B-Instruct` antes de treinar.

## Aviso

Assistente de **apoio** à decisão clínica, em contexto acadêmico. Nenhuma resposta deve ser usada como prescrição sem validação de um profissional de saúde habilitado — ver `agent/guardrails.py`. As FAQs de treino foram traduzidas automaticamente, sem revisão clínica humana: o conteúdo é material de estudo, não fonte validada para uso real.
