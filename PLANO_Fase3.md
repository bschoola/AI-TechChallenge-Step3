# Plano Técnico — Tech Challenge Fase 3: Assistente Médico Virtual

> Continuação do projeto **OncoTech** (fases anteriores — Algoritmo Genético, API de Predição + LLM, Frontend Angular — já entregues e avaliadas; os artefatos dessas fases foram removidos deste repositório após a entrega, que agora contém só o módulo da Fase 3). Este documento planeja o assistente médico virtual com fine-tuning de LLM, LangChain e LangGraph.

Data do plano: 23/08/2026 · Prazo assumido: 2 a 4 semanas · Sem GPU dedicada disponível (uso de Colab gratuito / CPU).

---

## 1. Recapitulando o que a banca pede

Do PDF do desafio (`8IADT — Fase 3 — Tech Challenge`), os requisitos obrigatórios são:

1. **Fine-tuning de LLM** com dados médicos internos (protocolos, FAQs de médicos, modelos de laudo/receita), com preprocessing, anonimização e curadoria.
2. **Assistente com LangChain**: pipeline que integra a LLM customizada, consulta bases estruturadas (prontuários) e contextualiza respostas com dados atualizados do paciente.
3. **Segurança e validação**: limites de atuação (nunca prescrever sem validação humana), logging detalhado e explainability (indicar fonte da informação).
4. **Organização**: projeto modularizado em Python, README completo.
5. **Entregáveis**: repositório Git (fine-tuning + LangChain + fluxos LangGraph), dataset anonimizado/sintético, relatório técnico, vídeo de até 15 min.

Nada disso é fornecido pela FIAP em termos de dados reais de hospital — vamos ter que construir um dataset sintético plausível e deixar isso explícito no relatório (é uma decisão de design válida e esperada em contexto acadêmico).

---

## 2. Decisões de arquitetura e por quê

### 2.1 Modelo base para fine-tuning

Sem GPU própria, a rota mais viável é **QLoRA em 4-bit rodando no Colab gratuito (GPU T4, ~15 GB VRAM)**. Isso exclui modelos grandes (7B+) por segurança de tempo/memória, mas é mais do que suficiente para o objetivo do desafio (demonstrar a técnica, não bater estado da arte).

| Opção | Licença | Observação |
|---|---|---|
| **Qwen2.5-3B-Instruct** (recomendado) | Apache 2.0, sem gate | Bom suporte multilíngue (inclui PT-BR), roda confortavelmente em QLoRA 4-bit no T4. |
| Qwen2.5-1.5B-Instruct | Apache 2.0 | Plano B se o Colab cortar a sessão antes do treino terminar — treina mais rápido. |
| Phi-3.5-mini-instruct | MIT | Alternativa sólida, inglês mais forte que PT-BR. |
| Llama-3.2-3B-Instruct | Licença Meta (precisa aceitar termos no HF) | Evitar se possível — o gate de acesso costuma travar quem está sem tempo sobrando. |

**Stack de treino:** `transformers` + `peft` (LoRA) + `bitsandbytes` (4-bit) + `trl` (`SFTTrainer`). Alternativa que acelera bastante em Colab free: **Unsloth** (mesma técnica, 2x mais rápido e usa menos VRAM) — vale usar se não houver conflito de versão no ambiente.

### 2.2 Fine-tuning + RAG (não é "ou", é "e")

Fine-tuning sozinho não resolve "consultar prontuário atualizado" (dado muda a cada paciente) nem "citar a fonte" (explainability exigida). Por isso o desenho combina as duas técnicas, com papéis diferentes — e isso vale a pena deixar bem explícito no relatório técnico, porque mostra que o grupo entende a diferença:

- **Fine-tuning** ensina *estilo, tom clínico e formato de resposta* (como um médico do hospital fictício escreveria uma resposta, formato de laudo/receita, vocabulário interno).
- **RAG (LangChain + vector store)** injeta *conhecimento factual e citável* — os protocolos internos e o que foi recuperado é o que aparece como "fonte" na resposta (explainability).
- **Tool de banco estruturado** injeta *dado do paciente em tempo real* (exames pendentes, histórico) — isso não pode vir do fine-tuning porque muda a cada consulta.

### 2.3 RAG

- **Vector store:** Chroma (local, persistente em disco, zero custo, integração nativa com LangChain).
- **Embeddings:** `intfloat/multilingual-e5-small` (leve, boa qualidade multilíngue/PT-BR, roda em CPU sem problema).
- **Chain:** LCEL (LangChain Expression Language) — retrieval → prompt com contexto + citação da fonte → LLM fine-tuned → resposta estruturada.

### 2.4 Servindo o modelo fine-tuned

Duas opções, da mais simples à mais "produção":

1. **`HuggingFacePipeline` direto** (recomendado para o prazo do desafio): carrega o modelo + adapter LoRA via `transformers`, embrulha como LLM do LangChain. Sem etapas extras de conversão.
2. **Merge + GGUF + Ollama**: mais trabalho (conversão via `llama.cpp`), útil se quiser servir o modelo fora do processo Python (ex.: via Ollama). Deixar como "melhoria" se sobrar tempo, não como caminho crítico.

### 2.5 LangGraph — fluxo de decisão

Grafo de estados proposto (nomes de nó em `agent/graph.py`):

```
receber_paciente
      │
      ▼
verificar_exames_pendentes ──(pendente)──► emitir_alerta_exame ─┐
      │ (sem pendência)                                          │
      ▼                                                          │
buscar_contexto_rag  ◄────────────────────────────────────────────┘
      │
      ▼
gerar_resposta_llm  (fine-tuned model + contexto RAG)
      │
      ▼
validar_seguranca (guardrails) ──(bloqueado)──► encaminhar_para_validacao_humana
      │ (aprovado)                                        │
      ▼                                                    │
finalizar_resposta  ◄───────────────────────────────────────┘
      │
      ▼
log_auditoria (sempre executa, registra tudo)
```

Cada nó é uma função Python simples (`state -> state`) — é exatamente o desenho que o PDF pede quando fala em "ao receber informações sobre um paciente, o sistema possa acionar diferentes etapas".

### 2.6 Segurança, guardrails e explainability

- **Guardrail de prescrição:** filtro (regex + lista de padrões, ex.: "tome X mg", "prescrevo", "receite") aplicado à resposta do LLM *antes* de devolver ao usuário. Se disparar, a resposta é marcada como `requer_validacao_humana=True` e reescrita com uma ressalva — nunca é bloqueada silenciosamente (o médico vê o rascunho, mas sinalizado).
- **System prompt fixo** reforçando o papel de apoio, não substituição, do assistente.
- **Logging estruturado** (`logs/audit.jsonl`): cada interação grava `timestamp`, `pergunta`, `paciente_id`, `contexto_recuperado` (chunks + fonte), `resposta`, `flag_seguranca`, `decisão_langgraph`. Isso serve tanto para auditoria quanto de evidência no relatório técnico.
- **Explainability:** toda resposta que usa RAG inclui uma seção "Fontes consultadas" com o nome do documento/protocolo e o trecho recuperado — sem isso não tem como cumprir literalmente o requisito do PDF.

### 2.7 "Prontuário" estruturado

Sem acesso a sistema hospitalar real, simulamos com **SQLite** (`data/prontuarios_mock.db`), com uma tabela `pacientes` (id, nome fictício, histórico) e `exames` (paciente_id, tipo, status: pendente/concluído). É consultado como uma **tool do LangChain** (function calling), não hardcoded no prompt.

---

## 3. Dataset

A FIAP sugere PubMedQA e MedQuAD (ambos em inglês, focados em conhecimento médico geral) — bons como *complemento* de conhecimento geral, mas não substituem o "dado interno do hospital" que o desafio pede. Como não há hospital real, a saída é gerar **dados sintéticos** e documentar isso como decisão consciente de curadoria/anonimização "by design" (não há PII porque nada é real).

Sugestão de composição:

| Conjunto | Tamanho alvo | Uso |
|---|---|---|
| Protocolos clínicos fictícios (curtos, por especialidade/condição) | 30–50 documentos | Indexados no RAG (Chroma) |
| Perguntas frequentes de médicos + resposta padrão | 100–150 pares instrução→resposta | Fine-tuning (formato tipo Alpaca: `instruction`, `input`, `output`) |
| Modelos de laudo/receita fictícios | 15–20 exemplos | Fine-tuning (ensina formato/tom) |
| Amostra traduzida/adaptada do MedQuAD ou PubMedQA (opcional) | 50–100 pares | Complemento de conhecimento geral no fine-tuning ou RAG |

Formato de treino: JSONL, um exemplo por linha:
```json
{"instruction": "Paciente com histórico de hipertensão pergunta se pode tomar ibuprofeno.", "input": "", "output": "Segundo o protocolo interno de anti-inflamatórios (Protocolo AINE-03)... [resposta] Esta sugestão não substitui avaliação médica direta."}
```

Gerar esse dataset com apoio de um LLM (Claude/GPT) é uma abordagem legítima e comum em projetos acadêmicos — só precisa constar no relatório como "dataset sintético gerado com apoio de IA generativa, curado manualmente pelo grupo".

---

## 4. Estrutura de pastas proposta

Módulo único do repositório, `1.AssistenteMedico/`:

```
1.AssistenteMedico/
├── README.md                    # instruções completas (requisito do PDF)
├── requirements.txt
├── data/
│   ├── raw/                     # protocolos, faqs, laudos-modelo (sintéticos)
│   └── processed/                # dataset.jsonl (train/val) + chroma/ + prontuarios_mock.db
├── finetuning/
│   ├── config.py                 # modelo base, hiperparâmetros LoRA, paths
│   ├── prepare_dataset.py        # limpeza, dedup, split train/val
│   ├── train_qlora.py            # script de treino (GPU/QLoRA ou fallback CPU/LoRA)
│   ├── train_qlora_colab.ipynb   # mesmo treino, adaptado para Google Colab (GPU T4)
│   └── evaluate.py               # comparação antes/depois do fine-tuning
├── rag/
│   ├── embeddings.py              # wrapper de embeddings E5 (prefixos query/passage)
│   ├── ingest.py                  # indexa data/raw no Chroma
│   └── chain.py                   # RAG chain com citação de fonte
├── agent/
│   ├── graph.py                   # StateGraph do LangGraph
│   ├── nodes.py                   # implementação de cada nó (incl. log_auditoria)
│   ├── guardrails.py              # filtro de prescrição + validação de segurança
│   └── tools.py                   # tool de consulta ao prontuário (SQLite)
├── api/
│   └── main.py                    # FastAPI expõe POST /assistant/ask
├── logs/
│   └── audit.jsonl                # gerado em runtime, git-ignorado
└── tests/
    └── README.md                  # o que testar (guardrails, RAG retrieval, grafo)
```

Ver seção 7 para o status atual de implementação de cada peça.

---

## 5. Cronograma sugerido (2–4 semanas)

| Semana | Foco | Entregas parciais |
|---|---|---|
| 1 | Dataset sintético (protocolos, FAQs, laudos) + preprocessing + ingestão no Chroma | `data/raw`, `data/processed/dataset.jsonl`, RAG respondendo sem fine-tuning ainda |
| 2 | Fine-tuning QLoRA no Colab + avaliação antes/depois | Adapter LoRA salvo, `evaluate.py` com comparação |
| 3 | LangChain (RAG chain + tool prontuário) + LangGraph (grafo completo) + guardrails + logging | Fluxo ponta a ponta funcionando localmente |
| 4 | Integração final, API FastAPI, relatório técnico, README, gravação do vídeo (≤15 min) | Repositório final + vídeo |

Se o prazo real for mais apertado, a semana 2 (fine-tuning) é a que mais aceita corte de escopo: reduzir para o modelo 1.5B e/ou menos épocas ainda cumpre o requisito ("realizar o fine-tuning"), só com resultado mais modesto — o que é aceitável academicamente desde que documentado.

---

## 6. Checklist de entregáveis (mapeado ao PDF)

- [x] Pipeline de fine-tuning (código + dataset) — `finetuning/`, GPU (Colab) ou fallback CPU
- [x] Integração com LangChain (RAG + tool de prontuário) — `rag/`, `agent/tools.py`
- [x] Fluxos do LangGraph (grafo com decisão automatizada) — `agent/graph.py`, `agent/nodes.py`
- [x] Dataset anonimizado/sintético incluído no repo — `data/raw/` (sintético por design, sem PII)
- [x] Guardrail: assistente nunca prescreve diretamente sem validação humana — `agent/guardrails.py`
- [x] Logging detalhado para auditoria (`logs/audit.jsonl`) — `agent/nodes.py::log_auditoria`
- [x] Explainability (fonte da informação citada na resposta) — `rag/chain.py` (campo `fontes`)
- [x] Projeto modularizado em Python + README completo
- [ ] Relatório técnico: explicação do fine-tuning, descrição do assistente, diagrama do fluxo LangChain/LangGraph, avaliação do modelo
- [ ] Vídeo (≤15 min): treinamento/funcionamento da LLM, execução de fluxo automatizado, resposta a pergunta clínica contextualizada, logs e validação

> Itens marcados como concluídos estão implementados e testados (localmente e/ou com componentes reduzidos, quando o ambiente de desenvolvimento não tinha GPU/rede); falta confirmar uma rodada 100% end-to-end no seu ambiente (fine-tuning real + índice RAG real + endpoint respondendo) antes de considerar "pronto para gravar o vídeo".

---

## 7. Status de implementação

Todos os componentes descritos na seção 4 estão implementados em `1.AssistenteMedico/`, não apenas esqueletados:

- `finetuning/prepare_dataset.py`, `rag/ingest.py`, `rag/embeddings.py`, `rag/chain.py`, `agent/tools.py`, `agent/guardrails.py`, `agent/nodes.py`, `agent/graph.py`, `api/main.py` — implementados e testados.
- `finetuning/train_qlora.py` — implementado com dois caminhos: QLoRA 4-bit real (GPU/CUDA) ou LoRA sem quantização como fallback de CPU (mais lento, mas funciona com RAM suficiente). `finetuning/train_qlora_colab.ipynb` cobre o caminho recomendado (Colab, GPU T4 gratuita).
- `finetuning/evaluate.py` — único item ainda pendente (comparação formal base vs. fine-tuned para o relatório técnico); não faz parte do caminho de execução da API.

Próximos passos: rodar o fine-tuning e o `rag/ingest.py` de ponta a ponta no seu ambiente (gera o adapter LoRA e o índice Chroma reais), confirmar o endpoint `/assistant/ask` respondendo com um caso real, implementar `evaluate.py`, e então partir para o relatório técnico e o vídeo demo.

## 8. Riscos e pontos de atenção

- **Sessão do Colab pode cair antes do treino terminar** → salvar checkpoints periodicamente no Google Drive; preferir o modelo 1.5B se isso acontecer mais de uma vez.
- **Licença de modelo gated (Llama)** pode travar quem está sem tempo de esperar aprovação → por isso a recomendação é Qwen2.5 (sem gate).
- **Dataset sintético** — deixar isso explícito e justificado no relatório evita que pareça que o grupo "não entendeu" o requisito de anonimização; é o contrário: não há dado real para anonimizar, então a curadoria acontece na geração.
- **RAG com poucos documentos** (30–50 protocolos) pode recuperar contexto pobre em alguns casos — vale ter 2-3 perguntas de teste "garantidas" para o vídeo, onde a recuperação funciona bem.
