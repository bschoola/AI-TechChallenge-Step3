# Tech Challenge — OncoTech: Assistente Médico Virtual

Projeto acadêmico da pós-graduação em IA (FIAP). Este repositório contém a entrega da **Fase 3** do Tech Challenge: um assistente médico virtual conversacional, combinando fine-tuning de LLM, RAG (LangChain) e um fluxo de decisão automatizado (LangGraph), com guardrails de segurança e logging de auditoria.

> As fases anteriores do projeto OncoTech (algoritmo genético para o classificador, API de predição com laudo gerado por LLM, frontend Angular) já foram entregues e avaliadas; seus artefatos foram removidos deste repositório para manter o foco na entrega atual.

---

## Estrutura do repositório

```
AI-TechChallenge-Step3/
├── PLANO_Fase3.md            # plano técnico: arquitetura, decisões, cronograma, checklist
├── README.md                 # este arquivo
└── 1.AssistenteMedico/       # módulo único: código, dados sintéticos e documentação da entrega
```

Toda a implementação, instruções de instalação e execução estão documentadas em [`1.AssistenteMedico/README.md`](1.AssistenteMedico/README.md). O plano de arquitetura completo (decisões técnicas, cronograma, checklist de entregáveis) está em [`PLANO_Fase3.md`](PLANO_Fase3.md).

## Início rápido

```bash
cd 1.AssistenteMedico
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

python finetuning/prepare_dataset.py   # 1. prepara o dataset sintético
python rag/ingest.py                   # 2. indexa os protocolos no RAG
python finetuning/train_qlora.py       # 3. fine-tuning (ou use o notebook Colab — ver README do módulo)
uvicorn api.main:app --reload --port 8001   # 4. sobe a API
```

Detalhes de cada etapa (incluindo o caminho recomendado de fine-tuning via Google Colab e o fallback local em CPU) em [`1.AssistenteMedico/README.md`](1.AssistenteMedico/README.md).

## Aviso

Este sistema é um projeto acadêmico de pós-graduação. O assistente é uma ferramenta de **apoio** à decisão clínica — nenhuma resposta deve ser usada como prescrição direta sem validação de um profissional de saúde habilitado.
