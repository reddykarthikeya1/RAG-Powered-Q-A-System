"""
Core processing logic
"""
import numpy as np
import streamlit as st
from modules.rag_utils import build_faiss_index
from config.settings import BATCH_SIZE


def process_corpus_and_build_index(corpus, embedder):
    """
    Process corpus and build FAISS index.
    This function handles the embedding and indexing process with caching.
    """
    if not corpus:
        return np.array([]), None

    # Check if embeddings are already computed and cached
    if ("corpus_embeddings" in st.session_state and
        "index" in st.session_state and
        "corpus" in st.session_state):
        cached_embeddings = st.session_state["corpus_embeddings"]
        cached_corpus = st.session_state["corpus"]

        # Check if corpus has changed
        if cached_corpus == corpus:
            st.success(f"Using cached embeddings and index for {len(corpus)} passages.")
            return cached_embeddings, st.session_state["index"]

    # Compute embeddings
    with st.spinner(f"Computing embeddings for {len(corpus)} passages... This may take a moment."):
        try:
            corpus_embeddings = embedder.encode(
                corpus,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        except Exception as e:
            st.error(f"Error computing embeddings: {str(e)}")
            raise

    # Build FAISS index
    with st.spinner("Building search index..."):
        index = build_faiss_index(corpus_embeddings)

    # Cache the results
    st.session_state["corpus_embeddings"] = corpus_embeddings
    st.session_state["index"] = index
    st.session_state["corpus"] = corpus

    st.success(f"Successfully processed {len(corpus)} passages and built search index.")
    return corpus_embeddings, index