
import argparse
import json
import random
import sys
from pathlib import Path

from bert_score import score

from .orchestrator import RAGOrchestrator
from .constants import EVAL_DIR

from .utils.logger import get_logger
logger = get_logger(__name__)

def run_evaluation(n_samples: int = 20):
    
    with open(EVAL_DIR / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    with open(EVAL_DIR / "answers.json", "r", encoding="utf-8") as f:
        gold_answers = json.load(f)

    uuids = list(queries.keys())

    orchestrator = RAGOrchestrator(
        use_reranker=True,
        use_memory=False,
        k_rerank=5,
        k=10
    )

    sample_uuids = random.sample(uuids, min(n_samples, len(uuids)))

    preds = []
    refs = []

    for uid in sample_uuids:
        question = queries[uid]
        gold_answer = gold_answers[uid]

        pred_answer, pred_sources = orchestrator.ask(question)

        preds.append(pred_answer)
        refs.append(gold_answer)

    P, R, F1 = score(preds, refs, lang="fr")
    f1_mean = F1.mean().item()

    logger.info(f"Mean BERTScore F1: {f1_mean:.4f}  ({f1_mean * 100:.2f}%)")
    return f1_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick RAG evaluation.")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to evaluate")
    args = parser.parse_args()

    run_evaluation(args.n)