import faiss
import numpy as np

def build_faiss_index(corpus_embeddings):
    dim = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(corpus_embeddings)
    return index

def retrieve(query, embedder, corpus, corpus_embeddings, index, top_k=2):
    if not corpus or index is None or corpus_embeddings.shape[0] == 0:
        return []
    query_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    return [corpus[i] for i in I[0]]

def answer_question(question, embedder, qa_pipeline, corpus, corpus_embeddings, index, top_k=2):
    contexts = retrieve(question, embedder, corpus, corpus_embeddings, index, top_k=top_k)
    if not contexts:
        return "No context available.", []
    combined = " ".join(contexts)
    try:
        result = qa_pipeline({
            "question": question,
            "context": combined
        })
        return result["answer"], contexts
    except Exception as e:
        return f"Error: {e}", contexts

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks