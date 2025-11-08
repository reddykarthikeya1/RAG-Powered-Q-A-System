"""
File upload and processing component
"""
import streamlit as st
from modules.file_utils import load_corpus, parse_uploaded_file
from modules.rag_utils import chunk_text


def render_file_uploader():
    """Render file upload UI and return corpus."""
    # Chunking option with descriptions
    chunking = st.checkbox(
        "Enable chunking for large documents",
        value=True,
        help="Splits large documents into overlapping chunks for better retrieval. Recommended for PDFs or long texts."
    )
    
    chunk_size = None
    overlap = None
    
    if chunking:
        chunk_size = st.slider(
            "Chunk size (words)",
            100, 500, 200, 50,
            help="Number of words per chunk. Larger chunks may improve context, but too large can reduce retrieval accuracy."
        )
        overlap = st.slider(
            "Chunk overlap (words)",
            0, 200, 50, 10,
            help="Number of overlapping words between chunks. Helps preserve context across chunk boundaries."
        )

    uploaded_file = st.file_uploader("Upload context file (.txt or .pdf)", type=["txt", "pdf"])

    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            chunk_fn = (lambda text: chunk_text(text, chunk_size, overlap)) if chunking else None
            corpus = parse_uploaded_file(uploaded_file, chunk_fn=chunk_fn)
            if corpus:
                st.success("File uploaded and loaded!")
                return corpus
            else:
                st.warning("No text found in the uploaded file.")
                return []
    else:
        # Only load default sample corpus if it exists and no file was uploaded
        try:
            from pathlib import Path
            default_corpus_path = Path("data/sample_corpus.txt")
            if default_corpus_path.exists():
                corpus = load_corpus(default_corpus_path)
                if not corpus:
                    st.warning("Default corpus is empty. Please upload a document.")
                    return []
                st.info("Using default sample corpus.")
                return corpus
            else:
                st.info("No file uploaded. You can upload a document to build your knowledge base.")
                return []
        except Exception as e:
            st.warning(f"Could not load default corpus: {str(e)}. Please upload a document.")
            return []