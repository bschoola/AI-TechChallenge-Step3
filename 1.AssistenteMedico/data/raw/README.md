# Dados brutos

```
raw/
├── protocolos/                          # 32 protocolos clínicos sintéticos → RAG (Chroma)
├── faqs/                                # 94 pares instrução→resposta → fine-tuning
├── laudos_modelo/                       # 5 modelos de laudo/parecer sintéticos → fine-tuning
└── _arquivado_sintetico_hospital_vida_nova/
    └── faqs/                            # FAQs sintéticas de uma versão anterior, preservadas
```

## Escopo

O assistente atende **todas as especialidades** do hospital fictício. As três
pastas acima cobrem, juntas, os quadros clínicos presentes no prontuário
simulado (`data/processed/prontuarios_mock.db`, 32 pacientes / 30 quadros
distintos).

## `protocolos/` — base de conhecimento do RAG (32 arquivos)

Protocolos internos sintéticos do "Hospital Vida Nova", indexados por
`rag/ingest.py` no Chroma. **Não entram no fine-tuning**: são conhecimento a
consultar e citar, e treinar o modelo neles destruiria a rastreabilidade que o
requisito de explainability exige.

| Código | Área |
|---|---|
| `MAMA-01`, `MAMA-02`, `ONCO-01`, `ONCO-03` | Oncologia mamária |
| `CLI-01` … `CLI-05` | Clínica médica: hipertensão, diabetes, anemia, tireoide, dor crônica |
| `URG-01` … `URG-04` | Urgência: dor abdominal, cólica renal, trauma de extremidade, TVP |
| `PNE-01` … `PNE-03` | Pneumologia: pneumonia, asma, tosse aguda |
| `INF-01` … `INF-03` | Infectologia: síndromes febris, ITU, uso racional de antimicrobianos |
| `GAS-01`, `GAS-02` | Gastroenterologia: gastroenterite, DRGE |
| `NEU-01`, `NEU-02` | Neurologia: cefaleia, lombalgia |
| `OTO-01`, `OFT-01`, `DER-01` | Otorrino, oftalmologia, dermatologia |
| `CIR-01` | Cirurgia geral: hérnia inguinal |
| `SEG-01` … `SEG-04` | Segurança: alergias e interações, sinais vitais, exames pendentes, contraste |
| `ADM-01` | Consulta de rotina e rastreamento |

Nenhum protocolo contém posologia específica. A conduta é sempre descrita como
orientação a ser validada pelo médico responsável, coerente com o guardrail de
prescrição (`agent/guardrails.py`).

> **Reindexação.** `rag/ingest.py` reconstrói o índice do zero a cada execução
> (`reset_vectorstore`). Isso é intencional: `Chroma.from_documents` sobre um
> `persist_directory` existente **acrescenta** documentos em vez de substituí-los,
> e executar a ingestão repetidamente duplicava todos os chunks — o que corrompia
> silenciosamente a recuperação, fazendo o retriever devolver N cópias do mesmo
> trecho em vez de N trechos distintos.

## `faqs/` — dataset de fine-tuning (94 arquivos)

Duas origens, identificáveis pelo prefixo do nome. Detalhe e atribuição de
licença em [`faqs/_FONTE.md`](faqs/_FONTE.md).

| Prefixo | Origem | Arquivos | Cobertura |
|---|---|---:|---|
| `hvn_` | Sintético, interno ao hospital fictício | 43 | Todas as especialidades |
| `cancergov_`, `pubmedqa_`, `seniorhealth_`, `gard_`, `ghr_`, `mplustopics_` | MedQuAD (NIH, CC BY 4.0) e PubMedQA (MIT) | 51 | Oncologia mamária |

**Limitação conhecida das 51 FAQs de base pública**: a tradução e o resumo para
PT-BR foram feitos com apoio de LLM, sem revisão por profissional de saúde. Há
risco residual de imprecisão terminológica. O conteúdo é material de estudo, não
fonte validada para uso clínico real. Isso consta do relatório técnico.

## `laudos_modelo/` — formato de documentos (5 arquivos)

Modelos de laudo, parecer, encaminhamento e orientação pós-procedimento. Entram
no fine-tuning para ensinar **estrutura e tom** de documento institucional, não
conteúdo clínico.

## Regenerando o dataset e o índice

```bash
python finetuning/prepare_dataset.py   # faqs/ + laudos_modelo/ → dataset_train/val.jsonl
python rag/ingest.py                   # protocolos/ → data/processed/chroma/
```

Qualquer alteração em `faqs/` ou `laudos_modelo/` exige **re-treinar o modelo**
para ter efeito; alteração em `protocolos/` exige apenas reindexar.
