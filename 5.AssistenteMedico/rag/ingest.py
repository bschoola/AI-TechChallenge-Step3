"""Indexa os protocolos internos (data/raw/protocolos) em um vector store Chroma.

Uso:
    python rag/ingest.py
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROTOCOLOS_DIR = BASE_DIR / "data" / "raw" / "protocolos"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "processed" / "chroma"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


def load_documents() -> list[dict]:
    """Le cada arquivo .md/.txt em PROTOCOLOS_DIR como um documento fonte.

    TODO: usar langchain_community.document_loaders (DirectoryLoader/TextLoader) e
    manter o nome do arquivo como metadado 'fonte' — necessario para explainability
    (a resposta final precisa citar de qual protocolo veio o trecho usado).
    """
    raise NotImplementedError


def split_documents(documents: list[dict]) -> list[dict]:
    """Chunking dos documentos (ex.: RecursiveCharacterTextSplitter, ~500 tokens,
    overlap ~50) preservando o metadado de fonte em cada chunk.
    """
    raise NotImplementedError


def build_vectorstore(chunks: list[dict]) -> None:
    """Gera embeddings (EMBEDDING_MODEL_NAME) e persiste em Chroma (CHROMA_PERSIST_DIR).

    TODO: langchain_huggingface.HuggingFaceEmbeddings + langchain_community.vectorstores.Chroma
    .from_documents(..., persist_directory=str(CHROMA_PERSIST_DIR)).
    """
    raise NotImplementedError


def main() -> None:
    docs = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print(f"Indexados {len(chunks)} chunks em {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()
