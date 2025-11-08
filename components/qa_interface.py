"""
Q&A interface component
"""
import streamlit as st
from modules.rag_utils import answer_question
from utils.helpers import update_analytics


def render_qa_interface(embedder, qa_pipeline, corpus, corpus_embeddings, index):
    """Render Q&A interface."""
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question:")
    
    if question:
        with st.spinner("Finding answer..."):
            try:
                answer, used_context = answer_question(
                    question, embedder, qa_pipeline, corpus, corpus_embeddings, index
                )
                
                if answer and "No context available to answer the question." not in answer and "Error processing question" not in answer:
                    st.success(f"**Answer:** {answer}")
                    
                    with st.expander("Context used"):
                        for i, ctx in enumerate(used_context, 1):
                            st.markdown(f"**Context {i}:** {ctx}")
                    
                    # Add to history
                    st.session_state["history"].append({
                        "Question": question,
                        "Answer": answer,
                        "Context": " | ".join(used_context)
                    })
                    
                    # Update analytics
                    update_analytics(question)
                else:
                    st.warning("Could not find an answer. Try rephrasing your question.")
            except Exception as e:
                st.error(f"An error occurred while processing your question: {str(e)}")