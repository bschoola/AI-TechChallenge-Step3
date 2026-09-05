"""Wrapper de embeddings com os prefixos exigidos pela familia de modelos E5
(intfloat/multilingual-e5-*): documentos devem ser embutidos como "passage: ..."
e buscas como "query: ...". Sem isso a qualidade de recuperacao do E5 cai bastante
— e uma particularidade documentada do modelo (ver model card no Hugging Face),
nao um detalhe estilistico opcional.

Usado tanto por rag/ingest.py (indexacao) quanto por rag/chain.py (busca), para
garantir que os dois lados usem exatamente o mesmo modelo e a mesma convencao de
prefixo — se um lado usar prefixo e o outro nao, a recuperacao piora silenciosamente.
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


class E5Embeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings com os prefixos 'passage:'/'query:' do E5 aplicados
    automaticamente, para quem for usar a classe nao esquecer de fazer isso na mao.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"query: {text}")


def load_embeddings() -> E5Embeddings:
    return E5Embeddings(model_name=EMBEDDING_MODEL_NAME)
