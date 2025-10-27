from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_alm_assistant.conversation.orchestrator import RAGOrchestrator

app = FastAPI(title="RAG ALM Assistant")

class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    # Placeholder orchestrator call
    answer = f"(Mock reply) You asked: {req.question}"
    return {"session_id": req.session_id, "answer": answer, "sources": []}

@app.get("/health")
def health_check():
    return {"status": "ok"}