import os
from celery import Celery
from openai import OpenAI

REDIS_URL   = os.getenv("REDIS_URL",    "redis://localhost:6379/0")
OLLAMA_URL  = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

celery_app = Celery(
    "oncotech",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,          # laudos ficam disponíveis por 1 hora
    worker_prefetch_multiplier=1, # processa uma tarefa por vez (respeita GPU única)
)


@celery_app.task(name="tasks.gerar_laudo", bind=True, max_retries=2)
def gerar_laudo_task(
    self,
    area_pior: float,
    textura_pior: float,
    pontos_concavos_pior: float,
    concavidade_pior: float,
    diagnostico: str,
    confianca: float,
    probabilidade_maligno: float,
    probabilidade_benigno: float,
) -> str:
    """Gera laudo médico via Ollama. Executado pelo Celery worker."""
    client = OpenAI(
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        timeout=180.0,
    )

    prompt = f"""Você é um assistente médico em oncologia mamária.
Redija um parecer clínico em português para leitura por um médico especialista.

REGRAS OBRIGATÓRIAS:
- NÃO inclua campos em branco nem placeholders como "Nome:", "Médico:", "CRM:", "Paciente:", "Assinatura:", "Data:" ou similares.
- Escreva apenas o texto corrido do parecer, sem formulários.
- Use **negrito** para destacar termos clínicos importantes.
- Máximo de 200 palavras.

DADOS DO EXAME (Wisconsin Breast Cancer Dataset):
- Área do tumor (pior): {area_pior}
- Textura (pior): {textura_pior}
- Pontos côncavos (pior): {pontos_concavos_pior}
- Concavidade (pior): {concavidade_pior}

RESULTADO DO CLASSIFICADOR:
- Diagnóstico: **{diagnostico}** (confiança: {confianca}%)
- P(maligno): {probabilidade_maligno * 100:.1f}% | P(benigno): {probabilidade_benigno * 100:.1f}%

Estruture em quatro parágrafos com os títulos em negrito:
**Achados morfológicos**, **Interpretação**, **Conduta sugerida**, **Observação importante**.
"""

    try:
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise self.retry(exc=exc, countdown=5)
