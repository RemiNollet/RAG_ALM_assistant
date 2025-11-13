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
    k: int = 5,
) -> Tuple[VectorStoreRetriever, Chroma]:
    """
    Convenience function: load Chroma and return a retriever.
    """
    vector_store = get_vector_store(persist_directory=persist_directory, model_name=model_name)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever, vector_store