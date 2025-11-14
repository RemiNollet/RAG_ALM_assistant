"""
Global constants for the RAG ALM Assistant project.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Data paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DIC_DIR = DATA_DIR / "DIC"
VECTORSTORE_DIR = str(DATA_DIR / "vector_store")

# Embeddings
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_NORMALIZE = True

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# LLM
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 64
LLM_DO_TOKENS = False
RETURN_FULL_TEXT = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# reranker
#RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
