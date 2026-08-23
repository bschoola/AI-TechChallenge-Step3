# 4. Stress Tests — Breast Cancer Prediction API

Testes de carga para a `2.PredictionApi`, com foco no endpoint que aciona a LLM (Ollama) para geração de laudos médicos.

---

## Arquitetura de fila (modo laudo)

O endpoint `/predict/breastCancer/laudo` utiliza um modelo assíncrono baseado em fila (Celery + Redis). O stress test respeita esse fluxo em dois passos:

```
Cliente
  │
  ├─ POST /predict/breastCancer/laudo   ──► API (enqueue)
  │         ◄── { task_id, diagnostico, confianca }
  │
  └─ GET /predict/breastCancer/laudo/status/{task_id}   (polling a cada Xs)
             ◄── { status: "pending" | "processing" | "completed" | "failed" }
```

**Por que fila?**
O Ollama processa requisições de forma sequencial. Sem fila, N requisições simultâneas causariam timeouts em cascata. Com Celery + Redis, o POST retorna imediatamente com um `task_id` e o worker Ollama processa os laudos um a um, na ordem de chegada.

**Métricas coletadas pelo stress test:**
- `enqueue_s` — tempo do POST até receber o `task_id` (deve ser < 1s)
- `total_s` — tempo do POST até o laudo ficar pronto (`status: completed`)
- `poll_cycles` — quantas vezes foi necessário fazer polling

---

## Pré-requisitos

1. API rodando com Celery + Redis (`docker-compose up` na raiz do projeto)
2. Python 3.11+
3. Dependência instalada:

```bash
cd 4.StressTests
pip install -r requirements.txt
```

---

## Como rodar

### Modo laudo (padrão) — testa a fila + LLM

```bash
# 10 requisições simultâneas (padrão)
python stress_test.py

# 20 requisições simultâneas
python stress_test.py --n 20

# 3 ondas de 10 requisições cada (total 30)
python stress_test.py --n 10 --waves 3

# Polling mais frequente (a cada 1s em vez de 2s)
python stress_test.py --poll-interval 1.0

# API em host/porta diferente
python stress_test.py --url http://localhost:9000
```

### Modo predict — só o classificador, sem LLM

Útil para isolar a latência do modelo ML e verificar que a API responde rápido sem a LLM.

```bash
python stress_test.py --endpoint predict
python stress_test.py --endpoint predict --n 50
```

---

## Saída esperada

```
  Onda 1  |  10 requisições simultâneas
  Endpoint : POST /predict/breastCancer/laudo  →  polling /status/{task_id}
  Polling  : a cada 2.0s
  ─────────────────────────────────────────────────────────────────
  #    Status    Diagnóstico  Confiança  Enqueue   Total   Polls
  ─────────────────────────────────────────────────────────────────
  #  1  ✓ ok     Maligno       91.3%    0.12s    14.55s      7x
  #  2  ✓ ok     Benigno       99.1%    0.11s    28.02s     14x
  ...

  RELATÓRIO FINAL
  Total de requisições  : 10
  Sucesso               : 10
  Taxa de sucesso       : 100.0%

  Latência de ENQUEUE (POST → task_id)
    Mín    : 0.10s
    Máx    : 0.18s
    Média  : 0.13s

  Latência TOTAL (POST → laudo pronto)
    Mín    : 14.55s
    Máx    : 142.30s   ← 10 laudos em fila sequencial no Ollama
    Média  : 78.40s

  Tempo médio em fila   : 78.27s
  Média de polls/req    : 39.1x
```

> **Nota:** latências altas no `total_s` são esperadas — o Ollama gera laudos sequencialmente. O importante é que o `enqueue_s` seja baixo (< 1s) e que não haja erros.

---

## Argumentos disponíveis

| Argumento         | Padrão                    | Descrição                                      |
|-------------------|---------------------------|------------------------------------------------|
| `--url`           | `http://localhost:8000`   | URL base da API                                |
| `--n`             | `10`                      | Requisições simultâneas por onda               |
| `--waves`         | `1`                       | Número de ondas (com pausa de 2s entre elas)   |
| `--endpoint`      | `laudo`                   | `laudo` (fila + LLM) ou `predict` (só ML)     |
| `--poll-interval` | `2.0`                     | Intervalo em segundos entre polls do status    |
