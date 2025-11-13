# src/rag_alm_assistant/retrieval/reranker.py

from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import PrivateAttr
from sentence_transformers import CrossEncoder


class RerankRetriever(BaseRetriever):
    """
    Retriever qui wrappe un retriever existant (ex: Chroma)
    et applique un reranking avec un modèle cross-encoder.

    Usage:
        base_retriever = vector_store.as_retriever(k=20)
        rerank_retriever = RerankRetriever(
            vector_retriever=base_retriever,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=5,
        )
    """

    vector_retriever: BaseRetriever
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5

    _cross_encoder: CrossEncoder = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # chargé une seule fois
        self._cross_encoder = CrossEncoder(self.model_name)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) récupération brute depuis Chroma (top-K large)
        docs = self.vector_retriever.get_relevant_documents(query)
        if not docs:
            return []

        # 2) scoring cross-encoder
        pairs = [(query, d.page_content) for d in docs]
        scores = self._cross_encoder.predict(pairs)

        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 3) on garde les top_k meilleurs
        reranked = [d for d, _ in scored_docs[: self.top_k]]
        return reranked

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # version async minimale
        return self._get_relevant_documents(query)