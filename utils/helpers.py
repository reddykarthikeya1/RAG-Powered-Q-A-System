"""
Utility functions for the RAG application
"""
import collections
import pandas as pd
import streamlit as st
import time


def initialize_session_state():
    """Initialize all required session state variables."""
    if "analytics" not in st.session_state:
        st.session_state["analytics"] = {
            "total_queries": 0,
            "question_types": collections.Counter(),
            "start_time": time.time()
        }

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Initialize model cache if not present
    if "embedder" not in st.session_state:
        st.session_state["embedder"] = None
    if "qa_pipeline" not in st.session_state:
        st.session_state["qa_pipeline"] = None


def update_analytics(question: str):
    """Update analytics with new question."""
    import re
    st.session_state["analytics"]["total_queries"] += 1
    question_words = question.strip().split()
    if question_words:
        # Remove punctuation from the first word to get a cleaner question type
        first_word = re.sub(r'[^\w\s]', '', question_words[0])
        if first_word:
            qtype = first_word.capitalize()
            st.session_state["analytics"]["question_types"][qtype] += 1
        else:
            st.session_state["analytics"]["question_types"]["Other"] += 1
    else:
        st.session_state["analytics"]["question_types"]["Other"] += 1


def display_analytics():
    """Display usage analytics."""
    st.markdown("### 📊 Usage Analytics")
    st.write(f"**Total queries served:** {st.session_state['analytics']['total_queries']}")
    
    # Calculate uptime
    uptime = time.time() - st.session_state["analytics"]["start_time"]
    st.write(f"**Application uptime:** {uptime // 3600:.0f}h {(uptime % 3600) // 60:.0f}m")

    if st.session_state["analytics"]["question_types"]:
        st.write("**Top question types:**")
        top_qtypes = st.session_state["analytics"]["question_types"].most_common(5)
        df = pd.DataFrame({
            "Type": [k for k, v in top_qtypes],
            "Count": [v for k, v in top_qtypes]
        })
        st.table(df)
    else:
        st.write("No queries yet.")


def display_history():
    """Display Q&A history with download option."""
    if st.session_state["history"]:
        st.subheader("Q&A History")
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Q&A History",
            csv,
            "qa_history.csv",
            "text/csv"
        )