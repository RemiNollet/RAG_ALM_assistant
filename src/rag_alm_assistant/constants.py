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
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_NORMALIZE = True

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

# LLM
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 128
LLM_DO_TOKENS = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")