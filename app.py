"""
RAG-Powered Q&A System
Main application file
"""
import streamlit as st
import numpy as np
from modules.model_utils import is_gpu_available
from config.settings import MAX_DISPLAY_PASSAGES

# Import components
from utils.helpers import initialize_session_state, display_analytics, display_history
from components.model_selector import render_model_selector
from components.file_uploader import render_file_uploader
from components.qa_interface import render_qa_interface
from core.processors import process_corpus_and_build_index


def main():
    """Main application function."""
    # Set page config
    st.set_page_config(
        page_title="RAG Q&A System", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("RAG-Powered Q&A System (FAISS + GPU)")
    st.write("Upload a `.txt` or `.pdf` file (one passage per line or paragraph) to use as your knowledge base.")

    # Initialize session state
    initialize_session_state()

    # Display analytics
    display_analytics()

    # Check GPU availability
    if is_gpu_available():
        st.success("GPU detected and in use! 🚀")
    else:
        st.info("Using CPU for processing. GPU acceleration is recommended for better performance.")

    # Render model selector
    embedder, qa_pipeline, embed_model, qa_model = render_model_selector()

    # Render file uploader
    corpus = render_file_uploader()

    # Process corpus and build index only if corpus exists
    corpus_embeddings, index = None, None
    if corpus:
        try:
            corpus_embeddings, index = process_corpus_and_build_index(corpus, embedder)
        except Exception as e:
            st.error(f"Error processing corpus: {str(e)}")
            st.stop()

    # Display knowledge base
    with st.expander("Knowledge Base"):
        if corpus:
            display_count = min(len(corpus), MAX_DISPLAY_PASSAGES)
            for i, passage in enumerate(corpus[:display_count]):
                st.markdown(f"**{i+1}.** {passage}")
            if len(corpus) > MAX_DISPLAY_PASSAGES:
                st.markdown(f"...and {len(corpus)-MAX_DISPLAY_PASSAGES} more passages.")
            
            # Add statistics about the corpus
            st.markdown(f"**Total passages:** {len(corpus)}")
            avg_length = sum(len(p) for p in corpus) / len(corpus) if corpus else 0
            st.markdown(f"**Average passage length:** {avg_length:.0f} characters")
        else:
            st.write("No knowledge base loaded.")

    # Render Q&A interface only if we have the required components
    if corpus and corpus_embeddings is not None and index is not None:
        render_qa_interface(embedder, qa_pipeline, corpus, corpus_embeddings, index)
    elif corpus:  # Only show warning if user uploaded a file but processing failed
        st.warning("Could not process the document. Please check the file format or try another file.")
    else:  # Show info if no file was uploaded
        st.info("Please upload a document to enable the Q&A functionality.")

    # Display history
    display_history()

    # Add footer with app information
    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io/), [FAISS](https://github.com/facebookresearch/faiss), "
                "[Sentence Transformers](https://www.sbert.net/), and [Hugging Face Transformers](https://huggingface.co/).")


if __name__ == "__main__":
    main()