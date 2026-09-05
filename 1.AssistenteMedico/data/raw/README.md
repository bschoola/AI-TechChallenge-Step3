# Dados brutos

```
raw/
├── protocolos/                          # protocolos clínicos sintéticos (RAG / Chroma) -- inalterado
├── faqs/                                # pares instrução->resposta para o fine-tuning (ver origem abaixo)
├── laudos_modelo/                       # modelos de laudo/receita sintéticos (formato/tom) -- inalterado
└── _arquivado_sintetico_hospital_vida_nova/
    └── faqs/                            # FAQs sintéticas originais do Hospital Vida Nova, arquivadas
```

## Origem de `faqs/` (atualizado em 05/09/2026)

Por decisão do usuário, o conjunto de FAQs sintéticas do "Hospital Vida Nova" foi
**substituído** por um conjunto de 50 pares instrução->resposta traduzidos e
resumidos a partir de duas bases públicas sugeridas pela FIAP:

- **MedQuAD** (NIH, licença CC BY 4.0) -- https://github.com/abachaa/MedQuAD
  Ben Abacha, A. & Demner-Fushman, D. "A Question-Entailment Approach to Question
  Answering." BMC Bioinformatics, 2019.
- **PubMedQA** (licença MIT) -- https://github.com/pubmedqa/pubmedqa
  Jin, Q. et al. "PubMedQA: A Dataset for Biomedical Research Question Answering." 2019.

Filtragem: apenas documentos/perguntas com foco em **câncer de mama** (oncologia
mamária), para manter coerência com o domínio do resto do projeto (RAG,
system prompt, prontuário mock). Fonte por arquivo: prefixo do nome
(`cancergov_`, `seniorhealth_`, `gard_`, `ghr_`, `mplustopics_`, `pubmedqa_`).
Detalhe completo em `faqs/_FONTE.md`.

**Tradução e resumo**: feitos com apoio de LLM (Claude), **sem etapa de revisão
manual humana adicional** -- decisão explícita do usuário, por questão de tempo.
Isso deve constar no relatório técnico como limitação conhecida: conteúdo
clínico traduzido automaticamente tem risco residual de imprecisão
terminológica e deve ser tratado como material de estudo, não como fonte
validada para uso clínico real.

**Desvio consciente do requisito do desafio**: o PDF da FIAP pede fine-tuning
com "dados médicos internos" do hospital fictício. As FAQs agora vêm de bases
públicas reais, não de dado sintético interno -- isso deve ser documentado
explicitamente no relatório como uma decisão deliberada (estudo dirigido às
bases sugeridas pela própria FIAP), e não como uma omissão. Os `protocolos/`
(RAG) e os `laudos_modelo/` (formato de laudo/receita) permanecem sintéticos e
"internos ao hospital fictício", pois não há equivalente de laudo/receita nas
bases públicas usadas.

## Licenciamento

MedQuAD é CC BY 4.0 (exige atribuição, já incluída acima e em `_FONTE.md`).
PubMedQA é MIT. Ambas permitem uso e redistribuição do conteúdo aqui reproduzido,
incluindo em contexto acadêmico.

Ver `PLANO_Fase3.md` (raiz do repo), seção 3, para o histórico da decisão.
