"""
Model utility functions for loading and managing NLP models.
"""
import logging
from typing import Optional
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L3-v2": "paraphrase-MiniLM-L3-v2",
    "all-mpnet-base-v2": "all-mpnet-base-v2"
}

DEFAULT_QA_MODELS = {
    "deepset/roberta-base-squad2": "deepset/roberta-base-squad2",
    "distilbert-base-cased-distilled-squad": "distilbert-base-cased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "bert-large-uncased-whole-word-masking-finetuned-squad"
}

def is_gpu_available() -> bool:
    """
    Check if CUDA is available for GPU acceleration.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Get or create a sentence transformer embedder model.
    
    Args:
        model_name: Name of the embedding model to load
        
    Returns:
        SentenceTransformer model instance
    """
    if model_name not in DEFAULT_EMBEDDING_MODELS:
        logger.warning(f"Unknown embedding model: {model_name}. Using default.")
        model_name = "all-MiniLM-L6-v2"
    
    device = "cuda" if is_gpu_available() else "cpu"
    logger.info(f"Loading embedding model '{model_name}' on {device}")
    
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"Failed to load embedding model {model_name}: {e}")
        raise

@st.cache_resource(show_spinner=False)
def get_qa(model_name: str = "deepset/roberta-base-squad2"):
    """
    Get or create a question-answering pipeline.
    
    Args:
        model_name: Name of the QA model to load
        
    Returns:
        Transformers pipeline for question answering
    """
    if model_name not in DEFAULT_QA_MODELS:
        logger.warning(f"Unknown QA model: {model_name}. Using default.")
        model_name = "deepset/roberta-base-squad2"
    
    device = 0 if is_gpu_available() else -1
    logger.info(f"Loading QA model '{model_name}' on {'GPU' if device == 0 else 'CPU'}")
    
    try:
        return pipeline(
            "question-answering",
            model=model_name,
            tokenizer=model_name,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load QA model {model_name}: {e}")
        raise

def get_available_models() -> dict:
    """
    Get available model options.
    
    Returns:
        Dictionary containing available embedding and QA models
    """
    return {
        "embedding_models": list(DEFAULT_EMBEDDING_MODELS.keys()),
        "qa_models": list(DEFAULT_QA_MODELS.keys())
    }