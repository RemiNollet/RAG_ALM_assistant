"""
Global constants for the RAG ALM Assistant project.
"""

import os
from dotenv import load_dotenv

# Data paths
DIC_DIR = os.path.join(os.path.expanduser("~/RAG_ALM_assistant"), "data/DIC")
VECTORSTORE_DIR = os.path.join(os.path.expanduser("~/RAG_ALM_assistant"), "data/vector_store")

# Embeddings
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_NORMALIZE = True

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 60

# LLM
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 512
LLM_DO_TOKENS = False

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")