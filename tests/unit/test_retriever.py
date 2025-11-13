# tests/unit/test_retriever.py
import types

import pytest
from langchain_core.documents import Document

from rag_alm_assistant.retrieval.retriever import get_vector_store, get_retriever


class DummyEmbeddings:
    """Embeddings fake qui renvoie des vecteurs de taille fixe."""
    def embed_documents(self, texts):
        return [[0.1] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 4


def test_get_vector_store_with_dummy_embeddings(monkeypatch, tmp_path):
    # monkeypatch pour remplacer HuggingFaceEmbeddings par DummyEmbeddings
    import rag_alm_assistant.retrieval.retriever as rmod

    def fake_hf_embeddings(*args, **kwargs):
        return DummyEmbeddings()

    monkeypatch.setattr(rmod, "HuggingFaceEmbeddings", fake_hf_embeddings)

    persist_dir = tmp_path / "chroma"
    vs = get_vector_store(persist_directory=str(persist_dir))

    # on peut au moins vérifier qu'on sait ajouter un doc
    vs.add_documents([Document(page_content="test OPCVM", metadata={})])

    # Chroma stocke ça dans une collection interne
    assert vs._collection.count() == 1


def test_get_retriever_returns_retriever_and_store(monkeypatch, tmp_path):
    import rag_alm_assistant.retrieval.retriever as rmod

    def fake_hf_embeddings(*args, **kwargs):
        return DummyEmbeddings()

    monkeypatch.setattr(rmod, "HuggingFaceEmbeddings", fake_hf_embeddings)

    persist_dir = tmp_path / "chroma"
    retriever, store = get_retriever(
        persist_directory=str(persist_dir),
        k=2,
    )

    assert store is not None
    assert hasattr(retriever, "get_relevant_documents")