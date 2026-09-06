"""Avaliacao do modelo: base vs. fine-tuned.

Requisito do desafio: "Avaliacao do modelo e analise dos resultados" no relatorio
tecnico. Este script gera, para o MESMO conjunto de perguntas, as respostas do
modelo base "cru" e do modelo base + adapter LoRA treinado, e calcula metricas
automaticas comparaveis lado a lado.

## Conjunto de teste

DATASET_VAL_PATH (finetuning/prepare_dataset.py) — o split de validacao, que o
treino nunca viu como exemplo supervisionado (ele e usado so como eval_dataset
para a loss do SFTTrainer, nunca para atualizar pesos). Cada item tem
'instruction' (a pergunta) e 'output' (a resposta de referencia, vinda de
MedQuAD/PubMedQA traduzido — ver data/raw/faqs/_FONTE.md).

## Metricas

Nenhuma delas mede correcao clinica — isso exigiria um medico avaliando as
respostas, o que esta fora do escopo deste trabalho academico. O que elas medem
e se o fine-tuning aproximou o comportamento do modelo do comportamento-alvo:

- `similaridade_referencia`: cosseno entre o embedding da resposta gerada e o da
  resposta de referencia (mesmo modelo E5 do RAG, rag/embeddings.py). Proxy de
  "falou sobre a mesma coisa", nao de "falou certo". E5 tem piso alto de
  similaridade entre textos em portugues, entao o numero absoluto importa menos
  que a DIFERENCA entre base e fine-tuned no mesmo item.
- `taxa_guardrail`: fracao de respostas que o guardrail de prescricao
  (agent/guardrails.py) sinalizaria. Menor e melhor — mede se o modelo aprendeu
  o enquadramento "sugestao a ser validada" do SYSTEM_PROMPT.
- `taxa_ressalva`: fracao de respostas que mencionam explicitamente validacao
  por profissional/medico responsavel. Maior e melhor — o lado positivo da
  metrica acima (nao basta nao prescrever, o alvo e enquadrar como sugestao).
- `distinct_3`: proporcao de trigramas distintos sobre o total de trigramas.
  Detecta degeneracao por repeticao (o loop "hipovitaminosis B32, B33, B34..."
  observado em producao — ver rag/chain.py::REPETITION_PENALTY). Perto de 1.0 =
  texto variado; valores baixos = loop.
- `comprimento_palavras`: media de palavras por resposta. Nao e "melhor maior";
  serve para mostrar se o fine-tuning mudou o regime de verbosidade.

## Geracao

Deterministica (`do_sample=False`, greedy) de proposito: o objetivo aqui e
comparar dois modelos, e amostragem estocastica (temperature=0.3, como em
rag/chain.py) faria o resultado mudar a cada execucao. Isso significa que os
numeros daqui NAO reproduzem exatamente o que a API responde em runtime — ver
"Limitacoes" no relatorio tecnico.

Nao ha RAG aqui: o objetivo e isolar o efeito do fine-tuning. As respostas da
API passam ainda por retrieval de protocolos e pelo grafo (agent/graph.py).

Uso:
    python finetuning/evaluate.py                       # roda tudo, salva JSON + Markdown
    python finetuning/evaluate.py --limit 3             # amostra menor (teste rapido)
    python finetuning/evaluate.py --skip-base           # so o fine-tuned
    python finetuning/evaluate.py --output-dir ../docs  # onde salvar os resultados
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_FINETUNING_DIR = Path(__file__).resolve().parent
_BASE_DIR = _FINETUNING_DIR.parent
# Permite tanto 'from config import ...' (mesmo padrao de train_qlora.py, rodando
# como script) quanto 'from agent...'/'from rag...' (pacotes na raiz do modulo).
for _path in (str(_FINETUNING_DIR), str(_BASE_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import ADAPTER_OUTPUT_DIR, BASE_MODEL_NAME, CPU_DTYPE, DATASET_VAL_PATH, SYSTEM_PROMPT

# Teto de geracao na avaliacao. Alinhado com CHAT_MAX_NEW_TOKENS de rag/chain.py
# para que a comparacao reflita o regime de resposta que a API realmente usa.
MAX_NEW_TOKENS = 400

# Diretorio padrao de saida dos resultados.
DEFAULT_OUTPUT_DIR = _BASE_DIR / "data" / "processed" / "evaluation"

# Expressoes que caracterizam a ressalva de validacao humana pedida pelo
# SYSTEM_PROMPT ("sempre enquadre sugestoes como recomendacao a ser validada por
# um medico responsavel"). Sem acento e case-insensitive — as respostas do modelo
# variam bastante na acentuacao (ver logs/audit.jsonl).
_RESSALVA_PATTERNS = [
    r"valida[cç][aã]o",
    r"validad[oa]",
    r"m[eé]dico respons[aá]vel",
    r"profissional de sa[uú]de",
    r"consulte um m[eé]dico",
    r"avalia[cç][aã]o m[eé]dica",
    r"n[aã]o substitui",
]


@dataclass
class ModelMetrics:
    """Metricas agregadas de um modelo sobre todo o conjunto de teste."""

    modelo: str
    n_respostas: int = 0
    similaridade_referencia: float = 0.0
    taxa_guardrail: float = 0.0
    taxa_ressalva: float = 0.0
    distinct_3: float = 0.0
    comprimento_palavras: float = 0.0


@dataclass
class ItemResult:
    """Resultado de uma pergunta do conjunto de teste, nos dois modelos."""

    instruction: str
    resposta_esperada: str
    resposta_base: str | None = None
    resposta_finetuned: str | None = None
    metricas_base: dict = field(default_factory=dict)
    metricas_finetuned: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conjunto de teste
# ---------------------------------------------------------------------------


def load_eval_questions(limit: int | None = None) -> list[dict]:
    """Reaproveita DATASET_VAL_PATH (nunca usado para atualizar pesos) como
    conjunto de teste.
    """
    if not DATASET_VAL_PATH.exists():
        raise FileNotFoundError(
            f"Dataset de validacao nao encontrado em {DATASET_VAL_PATH}. "
            "Rode 'python finetuning/prepare_dataset.py' primeiro."
        )
    with open(DATASET_VAL_PATH, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return questions[:limit] if limit else questions


# ---------------------------------------------------------------------------
# Carga dos modelos
# ---------------------------------------------------------------------------


def _resolve_dtype():
    """Mesma logica de rag/chain.py::_resolve_inference_dtype — a avaliacao deve
    rodar no mesmo dtype da inferencia de runtime, senao a comparacao mede
    tambem a diferenca de precisao numerica.
    """
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, CPU_DTYPE)


def load_tokenizer():
    """Tokenizer do adapter (salvo junto pelo treino) — mesmo chat template usado
    em finetuning/train_qlora.py::build_prompt e em rag/chain.py. Cai para o
    tokenizer do modelo base se o adapter ainda nao existir (modo --skip-finetuned).
    """
    from transformers import AutoTokenizer

    origem = str(ADAPTER_OUTPUT_DIR) if ADAPTER_OUTPUT_DIR.exists() else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(origem)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    """Modelo base 'cru', sem adapter LoRA — o ponto de partida da comparacao."""
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=_resolve_dtype(),
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )


def load_finetuned_model():
    """Modelo base + adapter LoRA de ADAPTER_OUTPUT_DIR — exatamente o que
    rag/chain.py serve em runtime.
    """
    from peft import PeftModel

    if not ADAPTER_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Adapter fine-tuned nao encontrado em {ADAPTER_OUTPUT_DIR}. Rode o "
            "fine-tuning primeiro (finetuning/train_qlora_colab.ipynb ou "
            "'python finetuning/train_qlora.py')."
        )
    return PeftModel.from_pretrained(load_base_model(), str(ADAPTER_OUTPUT_DIR))


def build_prompt(instruction: str, tokenizer) -> str:
    """Mesmo formato de prompt do treino (train_qlora.py::build_prompt) e do
    runtime (rag/chain.py::ask), com add_generation_prompt=True. Se o prompt de
    avaliacao divergir do de treino, a comparacao mede a divergencia de template,
    nao o efeito do fine-tuning.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(model, tokenizer, instruction: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Geracao greedy (do_sample=False) — deterministica, para a comparacao ser
    reproduzivel entre execucoes. Ver docstring do modulo.
    """
    import torch

    prompt = build_prompt(instruction, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Corta o prompt: so o que foi gerado interessa (equivalente ao
    # return_full_text=False do pipeline em rag/chain.py).
    gerado = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gerado, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metricas (logica pura — testavel sem carregar modelo)
# ---------------------------------------------------------------------------


def distinct_n(text: str, n: int = 3) -> float:
    """Proporcao de n-gramas distintos sobre o total. Detecta degeneracao por
    repeticao (loops observados em producao — ver rag/chain.py). 1.0 = nenhum
    n-grama repetido; textos curtos demais para formar n-gramas retornam 1.0.
    """
    palavras = text.lower().split()
    if len(palavras) < n:
        return 1.0
    ngramas = [tuple(palavras[i : i + n]) for i in range(len(palavras) - n + 1)]
    return len(set(ngramas)) / len(ngramas)


def tem_ressalva(text: str) -> bool:
    """True se a resposta enquadra a informacao como sujeita a validacao
    profissional (comportamento pedido pelo SYSTEM_PROMPT).
    """
    return any(re.search(p, text, re.IGNORECASE) for p in _RESSALVA_PATTERNS)


def foi_sinalizada(text: str) -> bool:
    """True se o guardrail de prescricao (agent/guardrails.py) sinalizaria a
    resposta. Reusa o MESMO filtro do runtime — a metrica mede o comportamento
    que o sistema realmente aplica, nao uma aproximacao.
    """
    from agent.guardrails import check_response

    return check_response(text).is_flagged


def _cosine(a: list[float], b: list[float]) -> float:
    """Reusa a implementacao ja existente do guardrail de escopo em vez de
    duplicar (e de adicionar numpy so por causa disso).
    """
    from agent.scope_guard import cosine_similarity

    return cosine_similarity(a, b)


def similaridades_com_referencia(respostas: list[str], referencias: list[str]) -> list[float]:
    """Cosseno entre resposta gerada e resposta de referencia, com o MESMO modelo
    de embeddings do RAG (rag/embeddings.py::load_embeddings). Embute os dois
    lados como 'passage' (sao ambos textos de resposta, nao consultas).
    """
    from rag.embeddings import load_embeddings

    emb = load_embeddings()
    vecs_resp = emb.embed_documents(respostas)
    vecs_ref = emb.embed_documents(referencias)
    return [_cosine(r, ref) for r, ref in zip(vecs_resp, vecs_ref)]


def metricas_por_item(resposta: str, similaridade: float) -> dict:
    """Metricas de uma unica resposta (a similaridade vem pronta porque e
    calculada em lote, para nao recarregar o modelo de embeddings por item).
    """
    return {
        "similaridade_referencia": round(similaridade, 4),
        "sinalizada_pelo_guardrail": foi_sinalizada(resposta),
        "tem_ressalva_validacao": tem_ressalva(resposta),
        "distinct_3": round(distinct_n(resposta, 3), 4),
        "comprimento_palavras": len(resposta.split()),
    }


def agregar(modelo: str, itens: list[dict]) -> ModelMetrics:
    """Media das metricas por item. Lista vazia devolve tudo zerado (evita divisao
    por zero quando um dos modelos foi pulado via --skip-*).
    """
    if not itens:
        return ModelMetrics(modelo=modelo)
    n = len(itens)
    return ModelMetrics(
        modelo=modelo,
        n_respostas=n,
        similaridade_referencia=round(sum(i["similaridade_referencia"] for i in itens) / n, 4),
        taxa_guardrail=round(sum(i["sinalizada_pelo_guardrail"] for i in itens) / n, 4),
        taxa_ressalva=round(sum(i["tem_ressalva_validacao"] for i in itens) / n, 4),
        distinct_3=round(sum(i["distinct_3"] for i in itens) / n, 4),
        comprimento_palavras=round(sum(i["comprimento_palavras"] for i in itens) / n, 1),
    )


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------


def gerar_respostas(questions: list[dict], carregar_modelo, rotulo: str, max_new_tokens: int) -> list[str]:
    """Carrega um modelo, gera a resposta de cada pergunta e libera a memoria.

    Os dois modelos sao carregados em SEQUENCIA, nao juntos: em CPU (fallback
    comum neste projeto) manter base e fine-tuned simultaneamente na RAM dobra o
    consumo sem necessidade, ja que a comparacao e feita depois, sobre texto.
    """
    print(f"[{rotulo}] carregando modelo...", flush=True)
    tokenizer = load_tokenizer()
    model = carregar_modelo()
    model.eval()

    respostas = []
    for i, q in enumerate(questions, start=1):
        print(f"[{rotulo}] {i}/{len(questions)}...", flush=True)
        respostas.append(generate(model, tokenizer, q["instruction"], max_new_tokens))

    del model
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover — torch sempre existe em runtime real
        pass

    return respostas


def compare(questions: list[dict], *, skip_base: bool = False, skip_finetuned: bool = False,
            max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Roda a comparacao completa e devolve o relatorio estruturado."""
    referencias = [q["output"] for q in questions]

    respostas_base = (
        [] if skip_base else gerar_respostas(questions, load_base_model, "base", max_new_tokens)
    )
    respostas_ft = (
        []
        if skip_finetuned
        else gerar_respostas(questions, load_finetuned_model, "fine-tuned", max_new_tokens)
    )

    print("[metricas] calculando embeddings...", flush=True)
    sims_base = similaridades_com_referencia(respostas_base, referencias) if respostas_base else []
    sims_ft = similaridades_com_referencia(respostas_ft, referencias) if respostas_ft else []

    itens: list[ItemResult] = []
    met_base_itens, met_ft_itens = [], []
    for idx, q in enumerate(questions):
        item = ItemResult(instruction=q["instruction"], resposta_esperada=q["output"])
        if respostas_base:
            item.resposta_base = respostas_base[idx]
            item.metricas_base = metricas_por_item(respostas_base[idx], sims_base[idx])
            met_base_itens.append(item.metricas_base)
        if respostas_ft:
            item.resposta_finetuned = respostas_ft[idx]
            item.metricas_finetuned = metricas_por_item(respostas_ft[idx], sims_ft[idx])
            met_ft_itens.append(item.metricas_finetuned)
        itens.append(item)

    return {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "modelo_base": BASE_MODEL_NAME,
        "adapter": str(ADAPTER_OUTPUT_DIR),
        "n_perguntas": len(questions),
        "max_new_tokens": max_new_tokens,
        "geracao": "greedy (do_sample=False)",
        "agregado": {
            "base": asdict(agregar("base", met_base_itens)),
            "finetuned": asdict(agregar("finetuned", met_ft_itens)),
        },
        "itens": [asdict(i) for i in itens],
    }


_METRICAS_TABELA = [
    ("similaridade_referencia", "Similaridade com a referência (cosseno E5)", "maior"),
    ("taxa_guardrail", "Taxa de respostas sinalizadas pelo guardrail", "menor"),
    ("taxa_ressalva", "Taxa de respostas com ressalva de validação humana", "maior"),
    ("distinct_3", "Trigramas distintos (1.0 = sem repetição)", "maior"),
    ("comprimento_palavras", "Comprimento médio (palavras)", "—"),
]


def to_markdown(report: dict) -> str:
    """Renderiza o relatorio em Markdown, pronto para colar na secao de avaliacao
    do relatorio tecnico (RELATORIO_TECNICO.md).
    """
    base = report["agregado"]["base"]
    ft = report["agregado"]["finetuned"]

    linhas = [
        "# Avaliação: modelo base vs. fine-tuned",
        "",
        f"- Modelo base: `{report['modelo_base']}`",
        f"- Adapter: `{report['adapter']}`",
        f"- Perguntas avaliadas: {report['n_perguntas']} (split de validação)",
        f"- Geração: {report['geracao']}, `max_new_tokens={report['max_new_tokens']}`",
        f"- Gerado em: {report['gerado_em']}",
        "",
        "## Métricas agregadas",
        "",
        "| Métrica | Base | Fine-tuned | Δ | Melhor |",
        "|---|---:|---:|---:|:--:|",
    ]
    for chave, rotulo, direcao in _METRICAS_TABELA:
        v_base, v_ft = base.get(chave, 0), ft.get(chave, 0)
        delta = round(v_ft - v_base, 4)
        linhas.append(f"| {rotulo} | {v_base} | {v_ft} | {delta:+} | {direcao} |")

    linhas += ["", "## Respostas lado a lado", ""]
    for i, item in enumerate(report["itens"], start=1):
        linhas += [
            f"### {i}. {item['instruction']}",
            "",
            "**Referência (dataset):**",
            "",
            f"> {item['resposta_esperada']}",
            "",
            "**Modelo base:**",
            "",
            f"> {item.get('resposta_base') or '_(não gerado)_'}",
            "",
            "**Modelo fine-tuned:**",
            "",
            f"> {item.get('resposta_finetuned') or '_(não gerado)_'}",
            "",
        ]
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia modelo base vs. fine-tuned.")
    parser.add_argument("--limit", type=int, default=None, help="Avaliar só as N primeiras perguntas.")
    parser.add_argument("--skip-base", action="store_true", help="Não gerar com o modelo base.")
    parser.add_argument("--skip-finetuned", action="store_true", help="Não gerar com o fine-tuned.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.skip_base and args.skip_finetuned:
        parser.error("--skip-base e --skip-finetuned juntos não geram nada para comparar.")

    questions = load_eval_questions(args.limit)
    report = compare(
        questions,
        skip_base=args.skip_base,
        skip_finetuned=args.skip_finetuned,
        max_new_tokens=args.max_new_tokens,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "evaluation.json"
    md_path = args.output_dir / "evaluation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")

    print("\n" + to_markdown(report).split("## Respostas lado a lado")[0])
    print(f"Resultados salvos em:\n  -> {json_path}\n  -> {md_path}")


if __name__ == "__main__":
    main()
