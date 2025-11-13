from typing import Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from chromadb.config import Settings

from ..constants import VECTORSTORE_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_NORMALIZE


def get_vector_store(
    persist_directory: str = VECTORSTORE_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Chroma:
    """
    Load an existing Chroma vector store from disk.
    """
    encode_kwargs = {"normalize_embeddings": EMBEDDING_NORMALIZE}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)

    client_settings = Settings(anonymized_telemetry=False)

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        client_settings=client_settings,   # <- désactive la télémétrie
    )
    
    return vector_store


def get_retriever(
    persist_directory: str = VECTORSTORE_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
    k: int = 20,
    use_reranker: bool = False,
    rerank_top_k: int = 5,          # top-K after reranking
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Tuple[BaseRetriever, Chroma]:
    vector_store = get_vector_store(persist_directory=persist_directory, model_name=model_name)

    base_retriever: VectorStoreRetriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    if not use_reranker:
        return base_retriever, vector_store

    rerank_retriever = RerankRetriever(
        vector_retriever=base_retriever,
        model_name=reranker_model_name,
        top_k=rerank_top_k,
    )
    return rerank_retriever, vector_store