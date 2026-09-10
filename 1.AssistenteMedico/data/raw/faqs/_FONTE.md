# Origem destes arquivos

As FAQs desta pasta alimentam o fine-tuning (`finetuning/prepare_dataset.py`) e
vem de duas origens, identificaveis pelo prefixo do nome do arquivo.

## 1. Conteudo interno sintetico, prefixo `hvn_` (43 arquivos)

Material clinico do hospital ficticio "Hospital vEinstein", escrito para o
projeto, cobrindo as especialidades atendidas pelo hospital: clinica medica,
urgencia, ortopedia, pneumologia, infectologia, gastroenterologia, urologia,
neurologia, otorrinolaringologia, oftalmologia, dermatologia, cirurgia geral e
temas transversais de seguranca do paciente.

Cada FAQ referencia o protocolo interno correspondente em `../protocolos/`, para
que o modelo aprenda a citar a fonte institucional na resposta. O conteudo e de
nivel de orientacao clinica geral e **nao contem posologia especifica**. O
assistente e de apoio a decisao e nunca prescreve (ver `agent/guardrails.py`).

## 2. Conteudo derivado de bases publicas, oncologia mamaria (50 arquivos)

Prefixos `cancergov_`, `pubmedqa_`, `seniorhealth_`, `gard_`, `ghr_`,
`mplustopics_`. Derivado e traduzido de:

- **MedQuAD** (NIH, licenca CC BY 4.0): https://github.com/abachaa/MedQuAD
  Ben Abacha, A. & Demner-Fushman, D. "A Question-Entailment Approach to
  Question Answering." BMC Bioinformatics, 2019.
- **PubMedQA** (licenca MIT): https://github.com/pubmedqa/pubmedqa
  Jin, Q. et al. "PubMedQA: A Dataset for Biomedical Research Question
  Answering." 2019.

Traducao e resumo feitos com apoio de LLM, **sem etapa de revisao manual por
profissional de saude**. Conteudo clinico traduzido automaticamente tem risco
residual de imprecisao terminologica e deve ser tratado como material de estudo,
nao como fonte validada para uso clinico real.

## Licenciamento

MedQuAD e CC BY 4.0 (exige atribuicao, incluida acima). PubMedQA e MIT. Ambas
permitem uso e redistribuicao do conteudo aqui reproduzido, inclusive em
contexto academico. O conteudo com prefixo `hvn_` e sintetico e proprio do
projeto.
