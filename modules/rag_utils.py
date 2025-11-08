"""
RAG (Retrieval-Augmented Generation) utility functions for document retrieval and QA.
"""
import logging
from typing import List, Optional, Tuple
import faiss
import numpy as np

logger = logging.getLogger(__name__)

def build_faiss_index(corpus_embeddings: np.ndarray):
    """
    Build a FAISS index for efficient similarity search.

    Args:
        corpus_embeddings: Numpy array of embeddings for the corpus

    Returns:
        FAISS index for similarity search
    """
    if corpus_embeddings.size == 0:
        logger.warning("Empty embeddings provided for index building")
        return None

    if len(corpus_embeddings.shape) != 2:
        raise ValueError("corpus_embeddings must be a 2D array")

    dim = corpus_embeddings.shape[1]

    # Validate embedding dimensions
    if dim <= 0:
        raise ValueError(f"Invalid embedding dimension: {dim}")

    # Check for NaN or infinite values in embeddings
    if np.any(np.isnan(corpus_embeddings)) or np.any(np.isinf(corpus_embeddings)):
        raise ValueError("Embeddings contain NaN or infinite values")

    index = faiss.IndexFlatL2(dim)

    # Convert to float32 as required by FAISS
    embeddings = corpus_embeddings.astype('float32')
    index.add(embeddings)

    logger.info(f"Built FAISS index with {index.ntotal} vectors of dimension {dim}")
    return index

def retrieve(
    query: str,
    embedder,
    corpus: List[str],
    corpus_embeddings: np.ndarray,
    index,
    top_k: int = 2
) -> List[str]:
    """
    Retrieve relevant passages from the corpus based on a query.

    Args:
        query: The query string
        embedder: Sentence transformer model for encoding
        corpus: List of text passages
        corpus_embeddings: Precomputed embeddings for the corpus
        index: FAISS index for similarity search
        top_k: Number of top results to return

    Returns:
        List of relevant text passages
    """
    if not query or not corpus or index is None or corpus_embeddings.size == 0:
        logger.warning("Empty query, corpus, index, or embeddings in retrieval")
        return []

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    
    if top_k > len(corpus):
        top_k = len(corpus)  # Adjust top_k if it's larger than corpus size

    try:
        query_emb = embedder.encode([query], convert_to_numpy=True)
        query_emb = query_emb.astype('float32')

        # Perform similarity search
        scores, indices = index.search(query_emb, top_k)

        # Filter out invalid indices
        valid_indices = [i for i in indices[0] if i < len(corpus) and i >= 0]
        return [corpus[i] for i in valid_indices]
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

def answer_question(
    question: str,
    embedder,
    qa_pipeline,
    corpus: List[str],
    corpus_embeddings: np.ndarray,
    index,
    top_k: int = 2
) -> Tuple[str, List[str]]:
    """
    Answer a question using the RAG approach.

    Args:
        question: The question to answer
        embedder: Sentence transformer model for encoding
        qa_pipeline: Question-answering pipeline
        corpus: List of text passages
        corpus_embeddings: Precomputed embeddings for the corpus
        index: FAISS index for similarity search
        top_k: Number of top results to retrieve

    Returns:
        Tuple of (answer, list of used contexts)
    """
    try:
        contexts = retrieve(
            question, embedder, corpus, corpus_embeddings, index, top_k=top_k
        )

        if not contexts:
            return "No context available to answer the question.", []

        combined = " ".join(contexts)

        result = qa_pipeline({
            "question": question,
            "context": combined
        })

        return result["answer"], contexts
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return f"Error processing question: {str(e)}", []

def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks

    Returns:
        List of text chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    words = text.split()
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    step_size = max(1, chunk_size - overlap)

    for i in range(0, len(words), step_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    logger.info(f"Text chunked into {len(chunks)} chunks of size {chunk_size} with overlap {overlap}")
    return chunks