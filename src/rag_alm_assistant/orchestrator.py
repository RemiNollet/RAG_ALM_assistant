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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

from .llm import get_llm
from .retrieval.retriever import get_retriever
from .constants import K, K_RERANK

from .utils.logger import get_logger
logger = get_logger(__name__)

class RAGOrchestrator:
    def __init__(
        self,
        retriever=None,
        llm=None,
        memory = None,
        k: int = K,
        k_rerank: int = K_RERANK,
        use_reranker: bool = True,
        use_memory: bool = True
    ):
        """
        If retriever/llm/memory are not provided, they are built using helpers.
        One orchestrator instance = one conversation memory.
        """
        self.use_memory = use_memory
        
        if retriever is None:
            retriever, _ = get_retriever(k=k, use_reranker=use_reranker, rerank_top_k=k_rerank)
        if llm is None:
            llm = get_llm()

        self.retriever = retriever
        self.llm = llm

        # --- Memory handling ---
        if use_memory:
            self.memory = memory or self._build_memory()
        else:
            self.memory = None   # critical: disables chat_history requirement

        self.prompt = self._build_prompt()
        self.qa_chain = self._create_qa_chain()


    def _build_prompt(self) -> PromptTemplate:
        """
        Build the RAG prompt template used by the ConversationalRetrievalChain.
        """
        return PromptTemplate.from_template(
            "Tu es un assistant financier qui répond à des questions sur des documents DIC/KID "
            "rédigés en français.\n"
            "Utilise UNIQUEMENT le contexte fourni pour répondre.\n"
            "Si tu ne sais pas, dis explicitement « Je ne sais pas ».\n\n"
            "NE PAS inventer, NE répondre qu’à partir des documents fournit en context, NE PAS extrapoler.\n"
            "IMPORTANT : Réponds en français, de façon claire et professionnelle, "
            "en 1 phrase, maximum 2 si nécessaire.\n\n"
            "Historique de conversation :\n{chat_history}\n\n"
            "Contexte :\n{context}\n\n"
            "Question : {question}\n\n"
            "Réponse :"
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
        common_kwargs = dict(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.prompt},
        )

        if self.memory is not None:
            common_kwargs["memory"] = self.memory

        return ConversationalRetrievalChain.from_llm(**common_kwargs)

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
        # CASE 1: memory enabled → standard call
        if self.use_memory:
            logger.info("Generate answer with memory for question: {}".format(question))
            result = self.qa_chain.invoke({"question": question})

        # CASE 2: memory disabled → must pass BOTH keys
        else:
            logger.info("Generate answer without memory for question: {}".format(question))
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": []    # prevents Missing input keys
            })
            
        answer = result["answer"]
        source_docs = result["source_documents"]
        sources = self.format_sources(source_docs)
        return answer, sources

    def reset(self):
        if self.memory:
            self.memory.clear()