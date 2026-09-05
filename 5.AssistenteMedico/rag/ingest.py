"""Indexa os protocolos internos (data/raw/protocolos) em um vector store Chroma.

Uso:
    python rag/ingest.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Permite rodar este arquivo diretamente ("python rag/ingest.py") sem depender de
# import relativo, que so funciona quando o modulo e importado como pacote.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.embeddings import load_embeddings


PROTOCOLOS_DIR = BASE_DIR / "data" / "raw" / "protocolos"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "processed" / "chroma"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents() -> list[Document]:
    """Le cada arquivo .md em PROTOCOLOS_DIR como um documento fonte.

    O metadado 'fonte' (nome do arquivo, sem extensao) e o que a resposta final
    vai citar como explainability (ver rag/chain.py e agent/nodes.py) — por isso
    e guardado desde a leitura, antes do chunking.
    """
    if not PROTOCOLOS_DIR.exists():
        raise RuntimeError(f"Pasta {PROTOCOLOS_DIR} nao existe.")

    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"fonte": path.stem})
        for path in sorted(PROTOCOLOS_DIR.glob("*.md"))
    ]

    if not documents:
        raise RuntimeError(
            f"Nenhum arquivo .md encontrado em {PROTOCOLOS_DIR}. "
            "Confira PLANO_Fase3.md secao 3 para o formato esperado dos protocolos."
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunking com overlap, preservando o metadado 'fonte' de cada documento em
    todos os seus chunks (comportamento padrao do RecursiveCharacterTextSplitter).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Gera embeddings (E5Embeddings, ver rag/embeddings.py) e persiste em Chroma."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=load_embeddings(),
        persist_directory=str(CHROMA_PERSIST_DIR),
    )


def main() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print(f"Indexados {len(chunks)} chunks (a partir de {len(docs)} documentos) em {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
