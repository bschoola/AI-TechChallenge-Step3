# Build context: raiz do projeto (definido no docker-compose.yml)
FROM python:3.11-slim

WORKDIR /app

# Instala curl para o health check do docker-compose
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copia o modelo treinado (path relativo à raiz do projeto)
COPY ChoosenModel /ChoosenModel

# Instala dependências
COPY 2.PredictionApi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API
COPY 2.PredictionApi/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
