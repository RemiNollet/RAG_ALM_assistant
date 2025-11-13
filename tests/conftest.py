# tests/conftest.py
import pytest
from langchain_core.documents import Document


class FakeRetriever:
    """Retriever minimal qui renvoie toujours le même document."""
    def __init__(self, docs=None):
        self.docs = docs or [
            Document(page_content="OPCVM est un organisme de placement collectif.", metadata={"dic_name": "test.pdf", "page": 1})
        ]

    def invoke(self, query: str):
        return self.docs

    def get_relevant_documents(self, query: str):
        return self.invoke(query)


class FakeLLM:
    """LLM minimal"""
    def invoke(self, prompt: str):
        # dans la vraie vie tu peux parser un peu, mais ici on s'en fiche
        return "Réponse factice basée sur le contexte."


@pytest.fixture
def fake_retriever():
    return FakeRetriever()


@pytest.fixture
def fake_llm():
    return FakeLLM()