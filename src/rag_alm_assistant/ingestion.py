import glob
import os
from pathlib import Path
from typing import List, Tuple
from chromadb.config import Settings

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .constants import (
    DIC_DIR,
    VECTORSTORE_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_NORMALIZE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from .utils.logger import get_logger
logger = get_logger(__name__)


def load_dic_documents(dic_dir: str = DIC_DIR) -> List[Document]:
    """
    Load all DIC PDFs from the given directory, one Document per page,
    and enrich metadata with dic_name + page number.

    Returns a flat list of Documents.
    """
    dic_path = os.path.join(dic_dir, "*.pdf")
    documents: List[Document] = []

    for file_path in glob.glob(dic_path):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()  # one Document per page

            dic_name = Path(file_path).name  # e.g. 'Allianz.pdf'
            for i, doc in enumerate(pages):
                doc.metadata["dic_name"] = dic_name
                doc.metadata["page"] = i + 1 

            documents.extend(pages)
        except Exception as e:
            logger.info(f"[WARNING] Error while loading '{file_path}': {e}")

    logger.info("{} documents processed.".format(len(documents)))
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    logger.info(f"{len(chunks)} chunks created from {len(documents)} documents.")
    return chunks


def build_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """
    Create a HuggingFaceEmbeddings instance with sensible defaults.
    """
    encode_kwargs = {"normalize_embeddings": EMBEDDING_NORMALIZE}
    return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)


def build_vectorstore(
    docs: List[Document],
    persist_directory: str = VECTORSTORE_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Chroma:
    """
    Build and persist a Chroma vector store from the given documents.
    """
    embeddings = build_embeddings(model_name=model_name)

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    logger.info(f"Vector store built and persisted at: {persist_directory}")
    return vector_store


def full_ingestion_pipeline(
    dic_dir: str = DIC_DIR,
    persist_directory: str = VECTORSTORE_DIR,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Tuple[Chroma, List[Document], List[Document]]:
    """
    High-level helper:
    1. Load DIC PDFs
    2. Chunk them
    3. Build & persist Chroma vector store

    Returns: (vector_store, raw_documents, chunked_documents)
    """
    logger.info(f"[INGESTION] Loading DIC documents from: {dic_dir}")
    raw_docs = load_dic_documents(dic_dir)

    logger.info("[INGESTION] Chunking documents...")
    chunks = chunk_documents(raw_docs)

    logger.info("[INGESTION] Building vector store...")
    vector_store = build_vectorstore(chunks, persist_directory=persist_directory, model_name=model_name)

    return vector_store, raw_docs, chunks