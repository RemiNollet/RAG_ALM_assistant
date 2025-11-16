# RAG Assistant for ALM – Insurance Key Information Document (DIC) Retrieval

## Business Context

A major life insurance company performs a large number of financial investments every year.  
For each investment, the responsible fund must provide a **Key Information Document (KID/DIC)**, a European-standardized PRIIPs document containing:

- the nature of the financial product  
- risks  
- performance scenarios  
- costs  
- exit conditions  
- management rules

The **ALM (Asset & Liability Management)** department uses these documents to make allocation decisions, but the scale and diversity of products make information retrieval long and tedious.

The goal is to provide a **conversational assistant** that allows:

- querying all DICs instantly  
- retrieving precise information hidden inside hundreds of pages  
- citing the original source documents  
- maintaining a **conversational mode** to follow up with related questions

All of this **without external services** (confidentiality), relying exclusively on:  
- **open-weight models (Mistral, Llama)**  
- **local vectorization (ChromaDB)**

---

## Project Architecture

PNG diagram located in `docs/architecture_diagram.png`

---

## Main Constraints

The following requirements must be respected:

1. **Absolutely no external API calls or cloud storage**
   - Models run locally  
   - DIC data stored and processed fully on-premise  

2. **Use of an open-weight LLM** 

3. **Local embedding storage**
   - Persistent ChromaDB  

4. **Conversational mode**
   - Memory management togglable (`use_memory=True/False`)  

5. **Mandatory source citation**

6. **Mandatory evaluation**
   - Provided dataset: `queries.json`, `answers.json`, `corpus.json`, `relevant_docs.json`  
   - Required metric: **BERTScore F1 > 60%**

---

# Installation & Quick Start

Below is the complete setup, run, and evaluation workflow.

## Installation

To install dependencies and download all required datasets (DIC PDFs + evaluation set):

```bash
git clone https://github.com/RemiNollet/RAG_ALM_assistant.git
cd RAG_ALM_assistant

make setup
env HF_TOKEN= YOUR_HF_TOKEN
```

To launch the FastAPI server with Uvicorn:
```bash
make run
```
The API will be available at: http://localhost:8000
Useful endpoints:
	- GET /health : health check
	- POST /chat : send a question
    - POST reset : reset conversation memory

Exemple question:
```bash
curl -X POST http://127.0.0.1:8000/chat \     
    -H "Content-Type: application/json" \
    -d '{"question": "Quel est le montant de l’investissement initial ?"}'
```

To run the evaluation script (BERTScore on random samples):
```bash
make eval
```