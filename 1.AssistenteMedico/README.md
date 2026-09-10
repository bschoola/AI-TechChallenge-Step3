# 1. Assistente Médico Virtual (Tech Challenge Fase 3)

Entrega da Fase 3 do Tech Challenge da pós-graduação em IA (FIAP). É um **assistente conversacional de apoio à decisão clínica** que atende **todas as especialidades** do hospital fictício, e combina três técnicas com papéis distintos:

- **Fine-tuning (QLoRA)** ensina *estilo, tom clínico e formato* de resposta.
- **RAG (LangChain e Chroma)** injeta *conhecimento factual citável*, vindo dos protocolos internos, que vira a "fonte" exibida na resposta.
- **Tools sobre banco estruturado** injetam *dado do paciente em tempo real*, como histórico, anamnese e exames. Esse dado muda a cada consulta e por isso não pode vir do fine-tuning.

Tudo é orquestrado por um **grafo de decisão em LangGraph**, com guardrails de segurança e log de auditoria por interação.

> - Relatório técnico completo, com fine-tuning, avaliação e diagramas: [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md)
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
    ├─ [chat] classifica escopo por embeddings e recusa fora do domínio clínico
    ├─ [chat] recupera protocolos no Chroma, descarta os irrelevantes e gera a resposta
    ├─ [chat] confere se os protocolos citados na resposta foram mesmo consultados
    ├─ confere a coerência etária da resposta com a idade real do paciente
    ├─ [visão geral] sintetiza anamnese e exames em RELEVANTE / ATENÇÃO, sem RAG
    ├─ [visão geral] filtra os pontos gerados e só passa o que está ancorado no prontuário
    ├─ valida segurança com o filtro de linguagem de prescrição direta
    └─ registra a interação completa em logs/audit.jsonl
                              │
                              ▼
            Resposta + fontes citadas + flag de validação humana
```

O diagrama detalhado do grafo, com todos os nós e arestas condicionais, está em [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md#4-diagrama-do-fluxo-langchain--langgraph).

## Estrutura

```
1.AssistenteMedico/
├── data/
│   ├── raw/
│   │   ├── faqs/            # 93 pares instrução/resposta (43 internos e 50 MedQuAD/PubMedQA)
│   │   ├── laudos_modelo/   # 5 modelos de laudo e parecer sintéticos, para formato e tom
│   │   ├── protocolos/      # 32 protocolos clínicos sintéticos, usados só no RAG
│   │   └── _arquivado_sintetico_hospital_vida_nova/   # FAQs de uma versão anterior, preservadas
│   └── processed/           # dataset_train/val.jsonl, chroma/, prontuarios_mock.db, evaluation/
├── finetuning/              # config, preparação do dataset, treino QLoRA, notebook Colab, avaliação
├── rag/                     # embeddings E5, ingestão no Chroma, corte de relevância, chain
├── agent/                   # grafo LangGraph, nós, guardrails, parsers e tools
├── api/                     # FastAPI
├── tests/                   # 86 testes de lógica pura, sem carregar LLM
└── logs/                    # audit.jsonl, gerado em runtime e ignorado pelo git
```

## Como rodar

```bash
cd 1.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Preparar o dataset de fine-tuning (FAQs e laudos-modelo viram JSONL)
python finetuning/prepare_dataset.py

# 2. Indexar os protocolos no RAG (Chroma)
python rag/ingest.py

# 3. Fine-tuning. GPU é o caminho recomendado, CPU funciona como fallback
#   Opção A (recomendada): abrir finetuning/train_qlora_colab.ipynb no Google Colab (T4 gratuita)
#   Opção B (CPU local, bem mais lento):
python finetuning/train_qlora.py

# 4. Subir a API
uvicorn api.main:app --reload --port 8001
```

O front-end Angular consome a API em `http://localhost:8001`. Veja [`../front/README.md`](../front/README.md).

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Status da API e se o grafo foi carregado |
| `GET` | `/patients` | Lista os pacientes do prontuário mock, com sexo, data de nascimento e idade calculada. O parâmetro `?nome=` filtra por parte do nome |
| `GET` | `/patients/{id}/exams` | Exames do paciente com status, data de solicitação e de realização. Devolve 404 se o paciente não existir |
| `GET` | `/exams/{id}` | Detalhe completo de um exame, com resultado, médico solicitante e observações |
| `GET` | `/patients/{id}/anamnesis` | Ficha de anamnese completa. Devolve 404 se o paciente não existir ou não tiver anamnese |
| `POST` | `/assistant/ask` | Pergunta livre do médico, com `paciente_id` e `pergunta`. Devolve a resposta, as fontes citadas, `termos_etarios_incoerentes` e `citacoes_invalidas` |
| `POST` | `/patients/{id}/overview` | Visão geral automática por IA, sem corpo na requisição. Devolve `pontos_relevantes` e `pontos_atencao` estruturados, mais `pontos_descartados` e `geracao_degenerada` |

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

O `POST /assistant/ask` e o `POST /patients/{id}/overview` percorrem o **mesmo grafo**, diferenciados pelo campo `tipo_interacao` do estado. O que muda é o nó de geração, enquanto os nós de exame pendente, guardrail, validação humana e auditoria são compartilhados.

**Convenção de erros:** `503` quando falta um artefato, como o índice do RAG ou o adapter ainda não gerados. `404` para paciente ou exame inexistente. `500` para o resto.

## Dados

| Pasta | Conteúdo | Uso | Origem |
|---|---|---|---|
| `data/raw/faqs/` | 93 pares instrução/resposta | **Fine-tuning** | 43 sintéticos internos e multiespecialidade (prefixo `hvn_`), mais 50 de MedQuAD (NIH, CC BY 4.0) e PubMedQA (MIT) sobre oncologia mamária, traduzidos com apoio de LLM |
| `data/raw/laudos_modelo/` | 5 modelos de laudo, parecer e encaminhamento | **Fine-tuning**, para formato e tom | Sintético |
| `data/raw/protocolos/` | 32 protocolos clínicos do hospital fictício, cobrindo todas as especialidades atendidas | **RAG**, como conhecimento citável | Sintético |
| `data/processed/prontuarios_mock.db` | 32 pacientes cobrindo 30 quadros clínicos distintos, entre clínica médica, urgência, pneumologia, infectologia, neurologia, cirurgia, ginecologia e oncologia, com 38 exames e 32 anamneses completas | **Tools**, como dado em tempo real | Sintético, em SQLite |

O dataset final de fine-tuning tem **98 exemplos** (84 de treino e 14 de validação), e a base do RAG tem **122 chunks** a partir dos 32 protocolos. A atribuição de licença e a ressalva sobre a tradução automática sem revisão manual estão em [`data/raw/README.md`](data/raw/README.md) e [`data/raw/faqs/_FONTE.md`](data/raw/faqs/_FONTE.md).

> **Reindexação do RAG.** O `rag/ingest.py` reconstrói o índice do zero a cada execução, pela função `reset_vectorstore`. Isso é necessário porque o `Chroma.from_documents` sobre um `persist_directory` existente *acrescenta* documentos em vez de substituí-los. Executar a ingestão repetidamente duplicaria todos os chunks, o que degrada a recuperação em silêncio: o retriever passa a devolver N cópias do mesmo trecho em vez de N trechos distintos.

> **Alterou os dados?** Mudança em `faqs/` ou `laudos_modelo/` exige rodar o `prepare_dataset.py` **e re-treinar o modelo**. Mudança em `protocolos/` exige apenas o `rag/ingest.py`.

> O seed do prontuário mock (`agent/tools.py::init_mock_db`) é totalmente idempotente, com `INSERT OR IGNORE` por id. Reiniciar a API preenche o que faltar sem duplicar nem sobrescrever, e migra bancos antigos automaticamente. Não é preciso apagar o `.db` ao atualizar o projeto.

## Segurança, guardrails e explainability

| Camada | Onde | O que faz |
|---|---|---|
| **System prompt** | `finetuning/config.py::SYSTEM_PROMPT` | Um único prompt, usado tanto no treino quanto na inferência. O modelo aprende a responder dentro dos mesmos limites que o guardrail depois verifica |
| **Guardrail de escopo** | `agent/scope_guard.py` | Antes de montar o contexto do paciente, classifica a pergunta por similaridade **relativa** de embeddings, comparando âncoras dentro e fora do domínio clínico. Fora de escopo, devolve uma resposta fixa em código que nunca vê o prontuário |
| **Guardrail de prescrição** | `agent/guardrails.py` | Filtro de padrões de prescrição direta, como "prescrevo" e "tome X mg". Nunca descarta a resposta: marca `requer_validacao_humana=True` e acrescenta a ressalva |
| **Coerência etária** | `agent/demographic_guard.py` | Sinaliza resposta ancorada numa faixa etária que não é a do paciente, como conduta pediátrica para paciente idoso. Só dispara quando *nenhuma* faixa citada contém a idade, então menção contrastiva não gera alarme |
| **Verificação de citações** | `agent/citation_guard.py` | Sinaliza resposta que cita protocolo inexistente ou que não estava entre as fontes recuperadas. Não citar protocolo nunca é sinalizado |
| **Filtro de ancoragem** | `agent/overview_filter.py` | Só exibe um ponto da visão geral se ele for rastreável ao prontuário do paciente. Impõe o teto de 5 itens, colapsa famílias morfológicas e descarta o bloco inteiro quando a geração degenera, sinalizando `geracao_degenerada` |
| **Validação humana** | `agent/nodes.py::encaminhar_para_validacao_humana` | Nó dedicado no grafo. Respostas sinalizadas saem com `status="aguardando_validacao_humana"` |
| **Explainability** | `rag/chain.py` e `agent/nodes.py::log_auditoria` | Toda resposta de chat cita os protocolos recuperados, e o log guarda pergunta, fontes, resposta bruta, resposta final, padrões sinalizados e as similaridades de cada classificação |

O `logs/audit.jsonl` recebe **uma linha JSON por interação** e nunca derruba a resposta em caso de falha de escrita.

## Avaliação do modelo

```bash
python finetuning/evaluate.py                # base e treinado nas 14 perguntas de validação
python finetuning/evaluate.py --limit 3      # amostra menor, para teste rápido
python finetuning/evaluate.py --skip-base    # só o modelo treinado
```

O script gera `data/processed/evaluation/evaluation.json`, com as respostas completas e as métricas por item, e `evaluation.md`, com a tabela comparativa pronta para o relatório. As métricas são: similaridade de embeddings com a resposta de referência, taxa de acionamento do guardrail, taxa de ressalva de validação humana, trigramas distintos como detector de loop de repetição, e comprimento médio.

A geração é **gulosa** (`do_sample=False`) para ser reproduzível entre execuções, igual à do runtime da API. A análise dos resultados está em [`RELATORIO_TECNICO.md`](../RELATORIO_TECNICO.md#5-avaliação-do-modelo-e-análise-dos-resultados).

## Testes

```bash
python -m pytest tests/ -q     # 86 testes
```

Eles cobrem lógica pura, sem carregar LLM nem modelo de embeddings: guardrail de escopo, consolidação de bullets repetidos, parsing da visão geral, filtro de ancoragem, coerência etária, verificação de citações, corte de relevância e métricas de avaliação. Cada caso real observado em produção entra como regressão: a saída degenerada de 167 itens, a resposta pediátrica para paciente de 66 anos e as citações `SEG-01` e `PNEUMO-02`.

## Fine-tuning: GPU (recomendado) ou CPU (fallback)

O modelo base atual é o **`Qwen/Qwen2.5-1.5B-Instruct`** (Apache 2.0, sem gate de licença), definido em `finetuning/config.py::BASE_MODEL_NAME`.

QLoRA de verdade, em 4 bits via `bitsandbytes`, exige GPU com CUDA. O caminho recomendado é o **Google Colab gratuito com T4**: abra `finetuning/train_qlora_colab.ipynb`, monte o Drive e rode as células. O adapter treinado é salvo direto no Drive, então uma queda de sessão não perde o progresso.

O `finetuning/train_qlora.py` detecta a ausência de GPU e cai para **LoRA sem quantização** em CPU, no dtype de `config.py::CPU_DTYPE`, que é bfloat16. Funciona, mas é ordens de magnitude mais lento. Para acelerar em hardware próprio, troque o `BASE_MODEL_NAME` por `Qwen2.5-0.5B-Instruct` antes de treinar.

## Aviso

Este é um assistente de **apoio** à decisão clínica, em contexto acadêmico. Nenhuma resposta deve ser usada como prescrição sem validação de um profissional de saúde habilitado, conforme o `agent/guardrails.py`. Os protocolos são sintéticos e parte das FAQs foi traduzida automaticamente, sem revisão clínica humana. O conteúdo é material de estudo, não fonte validada para uso real.
