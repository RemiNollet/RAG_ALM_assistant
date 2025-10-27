"""
Main RAG conversation orchestrator.

Steps:
1. Receive user query
2. Retrieve relevant passages from the vector store
3. Generate a grounded answer with the local open-weight model
4. Return the final answer along with cited sources
"""
class RAGOrchestrator:
    def __init__(self, retriever, llm, memory):
        self.retriever = retriever
        self.llm = llm
        self.memory = memory

    def answer(self, query: str, session_id: str):
        context = self.retriever.retrieve(query)
        prompt = f"Context:\\n{context}\\n\\nQuestion:\\n{query}\\nAnswer clearly and concisely in English."
        response = self.llm.generate(prompt)
        return {"answer": response, "sources": self.retriever.sources}