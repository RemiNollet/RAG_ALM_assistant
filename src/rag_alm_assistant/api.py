"""
To run: uvicorn src.api:app --reload
"""

from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .orchestrator import RAGOrchestrator

app = FastAPI(
    title="RAG ALM Assistant",
    description="Conversational RAG assistant over internal DIC (Documents d’Informations Clés).",
    version="0.1.0",
)

orc = RAGOrchestrator(use_reranker = True, use_memory=True, k_rerank = 5, k = 10)

class ChatRequest(BaseModel):
    question: str

class SourceRef(BaseModel):
    dic_name: Optional[str] = None
    page: Optional[int] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceRef]


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.

    - Input: user question
    - Output: answer + list of sources (DIC name, page)
    """
    answer, sources = orc.ask(req.question)

    source_models = [SourceRef(**s) for s in sources]
    return ChatResponse(answer=answer, sources=source_models)

