from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch
import streamlit as st

def is_gpu_available():
    return torch.cuda.is_available()

@st.cache_resource
def get_embedder(model_name="all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)

@st.cache_resource
def get_qa(model_name="deepset/roberta-base-squad2"):
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "question-answering",
        model=model_name,
        tokenizer=model_name,
        device=device
    )