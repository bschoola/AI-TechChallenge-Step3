"""RAG chain (LCEL): recupera contexto no Chroma, monta prompt e chama a LLM
fine-tuned, retornando resposta + fontes citadas (explainability).
"""

import sys
from dataclasses import dataclass
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from rag.embeddings import EMBEDDING_MODEL_NAME, load_embeddings
from rag.ingest import CHROMA_PERSIST_DIR

SYSTEM_PROMPT = """Voce e um assistente medico de apoio a decisao clinica do hospital.
Use APENAS as informacoes do contexto abaixo para responder. Se o contexto nao for
suficiente, diga isso explicitamente. NUNCA prescreva um tratamento diretamente: sempre
enquadre sugestoes como recomendacao a ser validada por um medico responsavel. Ao final
da resposta, cite as fontes usadas (nome do protocolo).
"""


@dataclass
class RagResponse:
    answer: str
    sources: list[str]


def load_vectorstore():
    """Carrega o Chroma persistido em CHROMA_PERSIST_DIR com o mesmo modelo de embeddings
    usado em rag/ingest.py.
    """
    raise NotImplementedError


def load_llm():
    """Carrega o modelo fine-tuned (base + adapter LoRA) embrulhado como LLM do LangChain.

    TODO: HuggingFacePipeline sobre o modelo + adapter salvo em
    finetuning/config.py::ADAPTER_OUTPUT_DIR.
    """
    raise NotImplementedError


def build_rag_chain():
    """Monta a chain LCEL: retriever -> prompt (SYSTEM_PROMPT + contexto) -> llm -> parser.

    O retorno deve preservar os metadados de fonte de cada chunk recuperado para
    popular RagResponse.sources.
    """
    raise NotImplementedError


def ask(question: str, patient_context: str | None = None) -> RagResponse:
    """Ponto de entrada usado pelo no `gerar_resposta_llm` do LangGraph (agent/nodes.py).

    patient_context: dados vindos de agent/tools.py (exames pendentes, historico),
    injetados no prompt junto do contexto recuperado do RAG.
    """
    raise NotImplementedError
