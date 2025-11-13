# tests/unit/test_orchestrator.py
from rag_alm_assistant.orchestrator import RAGOrchestrator


def test_orchestrator_ask_returns_answer_and_sources(fake_retriever, fake_llm):
    orchestrator = RAGOrchestrator(
        retriever=fake_retriever,
        llm=fake_llm,
        memory=None,   # laisser l'orchestrator construire sa mémoire interne
        k=5,
        k_rerank=3,    # n'a pas d'impact ici car on fournit déjà un retriever fake
    )

    answer, sources = orchestrator.ask("Qu'est-ce que l'OPCVM ?")

    assert isinstance(answer, str)
    assert "Réponse factice" in answer

    assert isinstance(sources, list)
    assert len(sources) >= 1
    assert sources[0]["dic_name"] == "test.pdf"
    assert sources[0]["page"] == 1