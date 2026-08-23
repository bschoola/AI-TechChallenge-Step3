# Fase 2 — Prediction API

API REST em Python (FastAPI) que carrega o modelo treinado na Fase 1, expõe endpoints de predição de câncer de mama e gera laudos médicos via LLM local (Ollama).

---

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Instalação e execução](#instalação-e-execução)
- [Documentação interativa](#documentação-interativa)
- [Endpoints](#endpoints)
  - [GET /](#get-)
  - [POST /predict/breastCancer](#post-predictbreastcancer)
  - [POST /predict/breastCancer/laudo](#post-predictbreastcancerlaudo)
  - [GET /health](#get-health)
- [Variáveis de ambiente](#variáveis-de-ambiente)
- [Estrutura](#estrutura)
- [Aviso](#aviso)

---

## Pré-requisitos

- Execute o projeto [1.GenAIPrediction](../1.GenAIPrediction/) ao menos uma vez para gerar o modelo em `ChoosenModel/cancer_model.joblib`.
- [Ollama](https://ollama.com/download) instalado e rodando com o modelo `llama3.2:3b`:

```bash
ollama serve
ollama pull llama3.2:3b
```

---

## Instalação e execução

```bash
cd 2.PredictionApi

# Criar e ativar o ambiente virtual
python -m venv venv

# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
uvicorn main:app --reload
```

A API sobe em `http://localhost:8000`.

---

## Documentação interativa

| Interface | URL |
|---|---|
| Swagger UI | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## Endpoints

### `GET /`

Redireciona automaticamente para `/docs` (Swagger UI).

---

### `POST /predict/breastCancer`

Executa apenas o classificador (Regressão Logística). Resposta rápida, sem LLM.

**Request body:**

```json
{
  "area_pior": 880.5,
  "textura_pior": 25.38,
  "pontos_concavos_pior": 0.2654,
  "concavidade_pior": 0.3001
}
```

| Campo | Tipo | Restrição | Descrição |
|---|---|---|---|
| `area_pior` | float | > 0 | Pior valor de área do tumor (worst area) |
| `textura_pior` | float | > 0 | Pior valor de textura — desvio padrão da escala de cinza |
| `pontos_concavos_pior` | float | ≥ 0 | Pior número de pontos côncavos no contorno |
| `concavidade_pior` | float | ≥ 0 | Pior severidade das partes côncavas do contorno |

**Response `200`:**

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "classe": 1,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `diagnostico` | string | `"Maligno"` ou `"Benigno"` |
| `confianca` | float | Grau de confiança da predição em % (0–100) |
| `classe` | int | `1` = Maligno, `0` = Benigno |
| `probabilidade_maligno` | float | Probabilidade bruta de ser maligno (0.0–1.0) |
| `probabilidade_benigno` | float | Probabilidade bruta de ser benigno (0.0–1.0) |

**Erros:**

| Código | Descrição |
|---|---|
| `503` | Modelo classificador não carregado — execute o projeto 1 primeiro |

---

### `POST /predict/breastCancer/laudo`

Executa o classificador e, em seguida, gera um laudo médico em português via LLM local (Ollama). Aceita o mesmo body do endpoint acima.

**Response `200`:**

```json
{
  "diagnostico": "Maligno",
  "confianca": 94.73,
  "classe": 1,
  "probabilidade_maligno": 0.9473,
  "probabilidade_benigno": 0.0527,
  "laudo": "**Achados morfológicos**\nOs parâmetros morfológicos indicam área tumoral elevada (880.5 mm²)...\n\n**Interpretação**\nO classificador atribuiu diagnóstico de malignidade com 94.73% de confiança...\n\n**Conduta sugerida**\nRecomenda-se encaminhamento para equipe de oncologia...\n\n**Observação importante**\nEste laudo é auxiliar e deve ser validado por médico responsável."
}
```

O campo `laudo` contém os mesmos campos de `PredictionResponse` mais:

| Campo | Tipo | Descrição |
|---|---|---|
| `laudo` | string | Parecer clínico em Markdown gerado pela LLM |

O laudo é estruturado em quatro seções: **Achados morfológicos**, **Interpretação**, **Conduta sugerida** e **Observação importante**.

> O campo `laudo` usa Markdown (`**negrito**`) renderizado pelo frontend.

**Erros:**

| Código | Descrição |
|---|---|
| `502` | Falha na comunicação com o Ollama — verifique se está rodando na porta 11434 |
| `503` | Modelo classificador ou LLM não carregados |

---

### `GET /health`

Verifica o status de todos os componentes da API.

**Response `200`:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "llm_available": true,
  "ollama_url": "http://localhost:11434",
  "ollama_model": "llama3.2:3b",
  "model_experiment": "experiment_3",
  "model_metrics": {
    "f1": 0.978,
    "recall": 0.991,
    "accuracy": 0.974,
    "roc_auc": 0.995
  }
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `status` | string | `"ok"` quando todos os componentes estão operacionais |
| `model_loaded` | bool | Se o classificador foi carregado com sucesso |
| `llm_available` | bool | Se o cliente Ollama foi inicializado |
| `ollama_url` | string | URL configurada para o Ollama |
| `ollama_model` | string | Modelo LLM em uso |
| `model_experiment` | string | Identificador do experimento do AG que gerou o modelo |
| `model_metrics` | object | Métricas do modelo: F1, Recall, Accuracy, ROC-AUC |

---

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | URL base do servidor Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Modelo LLM a ser usado para geração do laudo |

Ao rodar via Docker, essas variáveis são injetadas automaticamente pelo `docker-compose.yml`.

---

## Estrutura

```
2.PredictionApi/
├── main.py           # FastAPI: endpoints, schemas Pydantic, CORS, integração Ollama
├── requirements.txt
├── Dockerfile
└── PredictionApi.md  # Esta documentação
```

---

## Aviso

Esta API é parte de um projeto acadêmico (FIAP — Pós-Graduação em IA). Os resultados **não substituem** o diagnóstico médico profissional e não devem ser utilizados como única base para decisões clínicas.
