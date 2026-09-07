"""RAG chain: recupera contexto no Chroma, monta prompt e chama a LLM
fine-tuned, retornando resposta + fontes citadas (explainability).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import torch
from langchain_chroma import Chroma

from finetuning.config import ADAPTER_OUTPUT_DIR, BASE_MODEL_NAME, CPU_DTYPE, SYSTEM_PROMPT
from rag.embeddings import load_embeddings
from rag.ingest import CHROMA_PERSIST_DIR

# Quantos chunks recuperar por pergunta — ver PLANO_Fase3.md secao 2.4 (RAG).
RETRIEVER_TOP_K = 4

# Tamanho maximo de geracao. O chat responde UMA pergunta (400 tokens sobra); a
# visao geral pede DOIS blocos (RELEVANTE + ATENCAO, ate 5 itens cada) e na
# pratica estourava 400 tokens e cortava a resposta no meio de uma palavra (ver
# logs/audit.jsonl e discussao com o usuario) -- por isso tem um teto proprio,
# maior. Ainda e um teto, nao uma garantia: se o modelo entrar num loop
# degenerado (ver REPETITION_PENALTY/NO_REPEAT_NGRAM_SIZE abaixo) a resposta
# pode cortar de novo com um valor mais alto so adiando o problema.
CHAT_MAX_NEW_TOKENS = 400
# A visao geral pede dois blocos de ate 5 itens curtos — na pratica, menos de
# 200 palavras. Um teto generoso demais nao "da espaco para uma resposta melhor":
# da espaco para o modelo continuar gerando depois de ja ter dito o que tinha a
# dizer, que e exatamente quando ele deriva para associacao livre. O teto e a
# primeira contencao; o filtro de ancoragem (agent/overview_filter.py) e a
# ultima.
OVERVIEW_MAX_NEW_TOKENS = 450

# Controles de repeticao do pipeline transformers (aplicados aos dois casos).
# Adicionados depois de observar, em uso real, o modelo entrar em loop
# repetindo a mesma frase/padrao ate estourar o teto de tokens (ex.: uma visao
# geral que gerou "hipovitaminosis B32, hipovitaminosis B33, ... B37" ate
# cortar no meio da palavra) -- sintoma classico de amostragem sem penalidade
# de repeticao em modelos pequenos. repetition_penalty desincentiva reusar
# tokens ja gerados; no_repeat_ngram_size proibe repetir a mesma sequencia
# exata de 3 tokens (pega frases inteiras repetidas, tipo "terapia de
# reabilitacao ... terapia de reabilitacao"). Valores conservadores -- nao
# aparecem em nenhum benchmark deste projeto, so evitam o pior caso; se a
# qualidade das respostas piorar, o primeiro suspeito e REPETITION_PENALTY.
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 3

# Cache em memoria dos componentes pesados (vectorstore + modelo), para nao recarregar
# a cada chamada de ask() — mesma ideia de lifespan em api/main.py, mas no nivel da
# chain em si (assim tambem funciona fora do FastAPI, ex. em testes/notebooks).
_chain_cache: dict | None = None


@dataclass
class RagResponse:
    answer: str
    sources: list[str]


def load_vectorstore() -> Chroma:
    """Carrega o Chroma persistido em CHROMA_PERSIST_DIR com o mesmo modelo de
    embeddings usado em rag/ingest.py — precisa ser o mesmo, senao a similaridade
    dos vetores nao faz sentido.
    """
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Indice do RAG nao encontrado em {CHROMA_PERSIST_DIR}. "
            "Rode 'python rag/ingest.py' primeiro."
        )
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=load_embeddings(),
    )


def _resolve_inference_dtype() -> torch.dtype:
    """Mesma logica de train_qlora.py::_resolve_compute_dtype, mas para inferencia:
    GPU usa bfloat16 quando suportado nativamente (senao float16); sem GPU usa o
    dtype definido em config.CPU_DTYPE (mesmo usado no fallback de treino em CPU).
    """
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return getattr(torch, CPU_DTYPE)


def load_llm_and_tokenizer():
    """Carrega o modelo base + adapter LoRA treinado (finetuning/train_qlora.py ou
    o notebook Colab) e devolve (llm_chat, llm_overview, tokenizer).

    Dois wrappers HuggingFacePipeline em vez de um: chat e visao geral tem
    necessidades de tamanho de saida bem diferentes (ver CHAT_MAX_NEW_TOKENS/
    OVERVIEW_MAX_NEW_TOKENS acima) -- a visao geral pede dois blocos com ate 5
    itens cada e estourava o teto do chat, cortando a resposta no meio de uma
    palavra. Os dois pipelines reaproveitam o MESMO `model`/`tokenizer` ja
    carregados (nao ha peso duplicado em memoria, so dois objetos de
    configuracao de geracao diferentes por cima do mesmo modelo).

    O tokenizer e devolvido separadamente (alem de ir embutido nos pipelines)
    porque ask()/ask_overview() precisam dele para montar o prompt via
    apply_chat_template, exatamente como em finetuning/train_qlora.py::
    build_prompt — mesma formatacao em treino e inferencia, senao o adapter
    treinado nao se comporta como esperado.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from langchain_huggingface import HuggingFacePipeline

    if not ADAPTER_OUTPUT_DIR.exists():
        raise FileNotFoundError(
            f"Adapter fine-tuned nao encontrado em {ADAPTER_OUTPUT_DIR}. Rode o "
            "fine-tuning primeiro: 'python finetuning/train_qlora.py' (CPU, mais "
            "lento) ou finetuning/train_qlora_colab.ipynb (GPU, recomendado)."
        )

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_OUTPUT_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _resolve_inference_dtype()
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, dtype=dtype, device_map=device_map
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_OUTPUT_DIR))

    def _build_llm(max_new_tokens: int, do_sample: bool = True) -> "HuggingFacePipeline":
        # do_sample=False (greedy) para a visao geral: extrair pontos de uma ficha
        # num formato fixo e tarefa estruturada, nao criativa. A amostragem so
        # adiciona variancia — e variancia, num modelo pequeno, e por onde comeca
        # a deriva para termos cada vez mais distantes do caso.
        parametros_amostragem = {"temperature": 0.3} if do_sample else {}
        text_gen_pipeline = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **parametros_amostragem,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            # Sem isso, o pipeline devolve o prompt + a resposta concatenados — so
            # queremos o texto gerado (ver HuggingFacePipeline._generate, que usa esse
            # campo do retorno do pipeline diretamente).
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=text_gen_pipeline)

    llm_chat = _build_llm(CHAT_MAX_NEW_TOKENS)
    llm_overview = _build_llm(OVERVIEW_MAX_NEW_TOKENS, do_sample=False)
    return llm_chat, llm_overview, tokenizer


def _format_docs_with_sources(docs) -> str:
    """Formata os chunks recuperados com prefixo '[Fonte: ...]', para que a resposta
    final deixe claro de onde cada trecho de contexto veio — parte do requisito de
    explainability (ver PLANO_Fase3.md secao 2.5).
    """
    partes = [f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}]\n{doc.page_content}" for doc in docs]
    return "\n\n".join(partes)


def build_rag_chain() -> dict:
    """Monta os componentes da chain: retriever (Chroma) + llm_chat/llm_overview +
    tokenizer.

    Devolvido como dicionario (nao como uma unica Runnable encadeada) porque ask()
    precisa tanto da resposta gerada quanto dos metadados de fonte dos chunks
    recuperados, para popular RagResponse.sources — informacao que se perde se tudo
    for encadeado numa unica chain LCEL que so devolve a string final.
    """
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    llm_chat, llm_overview, tokenizer = load_llm_and_tokenizer()
    return {
        "retriever": retriever,
        "llm_chat": llm_chat,
        "llm_overview": llm_overview,
        "tokenizer": tokenizer,
    }


def _get_chain() -> dict:
    """Cache em memoria de build_rag_chain() — evita recarregar o modelo (o passo
    mais caro, na ordem de segundos a minutos) a cada pergunta.
    """
    global _chain_cache
    if _chain_cache is None:
        _chain_cache = build_rag_chain()
    return _chain_cache


def ask(question: str, patient_context: str | None = None) -> RagResponse:
    """Ponto de entrada usado pelo no `buscar_contexto_rag_e_gerar_resposta` do
    LangGraph (agent/nodes.py).

    patient_context: dados vindos de agent/tools.py (historico do paciente),
    injetados no prompt junto do contexto recuperado do RAG.
    """
    chain = _get_chain()
    retriever = chain["retriever"]
    llm = chain["llm_chat"]
    tokenizer = chain["tokenizer"]

    docs = retriever.invoke(question)
    contexto_rag = _format_docs_with_sources(docs)

    user_content = f"Contexto (protocolos internos):\n{contexto_rag}\n\n"
    if patient_context:
        user_content += f"Historico do paciente:\n{patient_context}\n\n"
    user_content += f"Pergunta: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # Mesmo template de chat usado em finetuning/train_qlora.py::build_prompt,
    # com add_generation_prompt=True para sinalizar ao modelo que agora e a vez
    # dele responder (o exemplo de treino nao inclui esse marcador ao final, ja
    # que ja contem a resposta completa).
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    resposta = llm.invoke(prompt)

    fontes = sorted({doc.metadata.get("fonte", "desconhecida") for doc in docs})
    return RagResponse(answer=resposta.strip(), sources=fontes)


# Formato de saida pedido para a "visao geral" automatica (agent/nodes.py::
# gerar_visao_geral) — marcadores fixos em vez de prosa livre, para
# agent/nodes.py::_parse_overview_sections conseguir quebrar a resposta em duas
# listas (mais robusto do que pedir JSON a um modelo pequeno servido por um
# pipeline `text-generation` simples, sem JSON mode).
OVERVIEW_INSTRUCTION = (
    "Gere uma visao geral clinica deste paciente para a equipe medica, SEM repetir "
    "textualmente os dados que ja aparecem na ficha de anamnese. Com base apenas no "
    "contexto do paciente abaixo, produza dois blocos curtos:\n"
    "RELEVANTE: pontos que resumem o caso (ate 5 itens)\n"
    "ATENCAO: riscos e pontos que a equipe deve observar - anormalidades, exames "
    "pendentes, alergias, interacoes medicamentosas, divergencias entre historico e "
    "conduta (ate 5 itens; se nao houver nenhum, escreva apenas 'Nenhum ponto de "
    "atencao identificado.')\n\n"
    "Se varios pontos seguirem exatamente o mesmo padrao (por exemplo, ausencia de "
    "sinais em diferentes sistemas do corpo), junte-os em UM SO item citando os "
    "sistemas separados por virgula, em vez de repetir um item para cada um.\n\n"
    "Cada item deve citar algo que esta escrito no contexto do paciente acima. Nao "
    "liste doencas, complicacoes ou termos que nao aparecem nesse contexto.\n\n"
    "Responda EXATAMENTE nesse formato: as duas palavras 'RELEVANTE:' e 'ATENCAO:' "
    "em linhas separadas, cada uma seguida so de itens em lista iniciados por '-', "
    "no maximo 5 itens por bloco. Nao escreva nada fora desses dois blocos."
)


def ask_overview(patient_context: str) -> RagResponse:
    """Ponto de entrada usado pelo no `gerar_visao_geral` do LangGraph (agent/
    nodes.py), para a "visao geral" automatica da tela do paciente.

    Diferente de ask(): nao ha uma pergunta livre do medico (so o contexto clinico
    do paciente), e deliberadamente NAO faz retrieval de protocolos via RAG - o
    insumo aqui e o caso do paciente, nao os protocolos internos do hospital.
    """
    chain = _get_chain()
    llm = chain["llm_overview"]
    tokenizer = chain["tokenizer"]

    user_content = f"Dados clinicos do paciente:\n{patient_context}\n\n{OVERVIEW_INSTRUCTION}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    resposta = llm.invoke(prompt)

    return RagResponse(answer=resposta.strip(), sources=[])
