"""
Main RAG conversation orchestrator.

Steps:
1. Receive user query
2. Retrieve relevant passages from the vector store
3. Generate a grounded answer with the local open-weight model
4. Return the final answer along with cited sources
"""

from typing import Any, Dict, List, Optional

from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

from .llm import get_llm
from .retrieval.retriever import get_retriever


class RAGOrchestrator:
    def __init__(
        self,
        retriever=None,
        llm=None,
        memory = None,
        k: int = 5,
    ):
        """
        If retriever/llm/memory are not provided, they are built using helpers.
        One orchestrator instance = one conversation memory.
        """
        if retriever is None:
            retriever, _ = get_retriever(k=k)
        if llm is None:
            llm = get_llm()
        if memory is None:
            memory = self._build_memory()

        self.retriever = retriever
        self.llm = llm
        self.memory = memory

        self.prompt = self._build_prompt()
        self.qa_chain = self._create_qa_chain()

    def _build_prompt(self) -> PromptTemplate:
        """
        Build the RAG prompt template used by the ConversationalRetrievalChain.
        """
        return PromptTemplate.from_template(
            "You are an assistant for question-answer tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Chat history:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

    def _build_memory(self) -> ConversationBufferMemory:
        """
        Conversation memory used to maintain the chat history for a session.
        """
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question",
        )

    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """
        Build the full RAG conversational chain (LLM + retriever + memory).
        """
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )

    @staticmethod
    def format_sources(source_docs: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for downstream API / UI consumption.
        """
        sources = []
        for d in source_docs:
            m = d.metadata
            sources.append(
                {
                    "dic_name": m.get("dic_name"),
                    "page": m.get("page")
                }
            )
        return sources

    # ---------- main API----------

    def ask(self, question: str):
        """
        Main entrypoint:
        - takes a question
        - runs the qa_chain
        - returns (answer, sources)
        """
        result = self.qa_chain({"question": question})
        answer = result["answer"]
        source_docs = result["source_documents"]
        sources = self.format_sources(source_docs)
        return answer, sources