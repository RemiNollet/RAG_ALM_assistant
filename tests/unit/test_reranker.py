# tests/unit/test_reranker.py
from langchain_core.documents import Document

from rag_alm_assistant.retrieval.reranker import RerankRetriever


class DummyVectorRetriever:
    def __init__(self):
        self.docs = [
            Document(page_content="Document A", metadata={"id": "A"}),
            Document(page_content="Document B", metadata={"id": "B"}),
            Document(page_content="Document C", metadata={"id": "C"}),
        ]

    def get_relevant_documents(self, query: str):
        return self.docs


def test_rerank_retriever(monkeypatch):
    base_retriever = DummyVectorRetriever()

    # instancier avec un modèle quelconque (il sera patché)
    rr = RerankRetriever(
        vector_retriever=base_retriever,
        model_name="fake-model",
        top_k=2,
    )

    # Monkeypatch CrossEncoder.predict pour renvoyer des scores fixes
    def fake_predict(self, pairs):
        # scorer B > C > A
        # len(pairs) == 3
        return [0.1, 0.9, 0.5]

    monkeypatch.setattr(
        type(rr._cross_encoder),
        "predict",
        fake_predict,
        raising=True,
    )

    docs = rr._get_relevant_documents("test query")

    # on s'attend à ce que les deux premiers soient B puis C
    assert len(docs) == 2
    ids = [d.metadata["id"] for d in docs]
    assert ids == ["B", "C"]