"""
Model selection component
"""
import streamlit as st
from modules.model_utils import get_embedder, get_qa


def render_model_selector():
    """Render model selection UI and return selected models."""
    # Model selection with descriptions
    embed_model = st.selectbox(
        "Choose embedding model:",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"],
        help=(
            "**all-MiniLM-L6-v2**: Fast, general-purpose sentence embedding model (recommended for most cases).\n"
            "**paraphrase-MiniLM-L3-v2**: Smaller, faster model, good for paraphrase detection and resource-constrained environments."
        )
    )
    
    qa_model = st.selectbox(
        "Choose QA model:",
        ["deepset/roberta-base-squad2", "distilbert-base-cased-distilled-squad"],
        help=(
            "**deepset/roberta-base-squad2**: Larger, more accurate QA model (recommended for best answers).\n"
            "**distilbert-base-cased-distilled-squad**: Smaller, faster QA model, suitable for quick responses."
        )
    )
    
    # Get or create models
    embedder = get_embedder(embed_model)
    qa_pipeline = get_qa(qa_model)
    
    return embedder, qa_pipeline, embed_model, qa_model