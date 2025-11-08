"""
Application configuration settings
"""
import os

# Model settings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_QA_MODEL = "deepset/roberta-base-squad2"
SUPPORTED_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L3-v2"
]
SUPPORTED_QA_MODELS = [
    "deepset/roberta-base-squad2",
    "distilbert-base-cased-distilled-squad"
]

# Chunking settings
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 50
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 500
MIN_OVERLAP = 0
MAX_OVERLAP = 200

# Retrieval settings
TOP_K = 2
BATCH_SIZE = 32

# File settings
SUPPORTED_FILE_TYPES = ["txt", "pdf"]
MAX_DISPLAY_PASSAGES = 100
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB in bytes

# Session state keys
SESSION_STATE_KEYS = {
    "analytics": "analytics",
    "history": "history",
    "corpus": "corpus",
    "corpus_embeddings": "corpus_embeddings",
    "index": "index",
    "embedder": "embedder",
    "qa_pipeline": "qa_pipeline"
}