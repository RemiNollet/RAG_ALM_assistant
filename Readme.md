# RAG ALM Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot assistant designed for the **Asset-Liability Management (ALM)** department of a life insurance company.

## Objective
Help ALM teams instantly access and reason over information contained in **Key Information Documents (KIDs)**, while ensuring full data confidentiality and on-prem execution.

## Project structure
- **src/** – Core RAG logic (ingestion, retrieval, generation, evaluation, API)
- **data/** – Local datasets (not versioned except data/eval)
- **notebooks/** – Research & exploration notebooks
- **docker/** – Container setup for on-prem deployment

## Quick start
\`\`\`bash
pip install -r requirements.txt
python src/scripts/index_corpus.py
uvicorn src.rag_alm_assistant.api.server:app --reload
\`\`\`