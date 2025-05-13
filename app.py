import streamlit as st
import numpy as np
import pandas as pd
import collections
from modules.file_utils import load_corpus, parse_uploaded_file
from modules.model_utils import is_gpu_available, get_embedder, get_qa
from modules.rag_utils import build_faiss_index, answer_question, chunk_text

st.set_page_config(page_title="RAG Q&A System", layout="wide")
st.title("RAG-Powered Q&A System (FAISS + GPU)")
st.write("Upload a `.txt` or `.pdf` file (one passage per line or paragraph) to use as your knowledge base.")

# --- Analytics state ---
if "analytics" not in st.session_state:
    st.session_state["analytics"] = {
        "total_queries": 0,
        "question_types": collections.Counter()
    }

# --- Display analytics ---
st.markdown("### 📊 Usage Analytics")
st.write(f"**Total queries served:** {st.session_state['analytics']['total_queries']}")
if st.session_state["analytics"]["question_types"]:
    st.write("**Top question types:**")
    top_qtypes = st.session_state["analytics"]["question_types"].most_common(5)
    st.table({"Type": [k for k, v in top_qtypes], "Count": [v for k, v in top_qtypes]})
else:
    st.write("No queries yet.")

if is_gpu_available():
    st.success("GPU detected and in use! 🚀")

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

# Chunking option with descriptions
chunking = st.checkbox(
    "Enable chunking for large documents",
    value=True,
    help="Splits large documents into overlapping chunks for better retrieval. Recommended for PDFs or long texts."
)
chunk_size = st.slider(
    "Chunk size (words)",
    100, 500, 200, 50,
    help="Number of words per chunk. Larger chunks may improve context, but too large can reduce retrieval accuracy."
) if chunking else None
overlap = st.slider(
    "Chunk overlap (words)",
    0, 200, 50, 10,
    help="Number of overlapping words between chunks. Helps preserve context across chunk boundaries."
) if chunking else None

uploaded_file = st.file_uploader("Upload context file (.txt or .pdf)", type=["txt", "pdf"])

if uploaded_file:
    chunk_fn = (lambda text: chunk_text(text, chunk_size, overlap)) if chunking else None
    corpus = parse_uploaded_file(uploaded_file, chunk_fn=chunk_fn)
    if corpus:
        st.success("File uploaded and loaded!")
    else:
        st.warning("No text found in the uploaded file.")
else:
    st.info("Using default sample corpus.")
    corpus = load_corpus("data/sample_corpus.txt")

embedder = get_embedder(embed_model)
qa_pipeline = get_qa(qa_model)

if corpus:
    corpus_embeddings = embedder.encode(corpus, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    index = build_faiss_index(corpus_embeddings)
else:
    corpus_embeddings = np.array([])
    index = None

with st.expander("Knowledge Base"):
    if corpus:
        for i, passage in enumerate(corpus[:100]):
            st.markdown(f"**{i+1}.** {passage}")
        if len(corpus) > 100:
            st.markdown(f"...and {len(corpus)-100} more passages.")
    else:
        st.write("No knowledge base loaded.")

# Q&A and history
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")
if "history" not in st.session_state:
    st.session_state["history"] = []

if question:
    answer, used_context = answer_question(
        question, embedder, qa_pipeline, corpus, corpus_embeddings, index
    )
    st.success(f"**Answer:** {answer}")
    with st.expander("Context used"):
        for ctx in used_context:
            st.write(ctx)
    st.session_state["history"].append({
        "Question": question,
        "Answer": answer,
        "Context": " | ".join(used_context)
    })
    # --- Update analytics ---
    st.session_state["analytics"]["total_queries"] += 1
    qtype = question.strip().split()[0].capitalize() if question.strip() else "Other"
    st.session_state["analytics"]["question_types"][qtype] += 1

if st.session_state["history"]:
    st.subheader("Q&A History")
    df = pd.DataFrame(st.session_state["history"])
    st.dataframe(df)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Q&A History", csv, "qa_history.csv", "text/csv")