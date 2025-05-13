![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-brightgreen?logo=streamlit)

# RAG-Powered Q\&A System

A modular, production-ready Retrieval-Augmented Generation (RAG) Question Answering system built with **Streamlit**, **FAISS**, and **Hugging Face Transformers**.

![image](https://github.com/user-attachments/assets/5a5886b9-34bd-4faf-b5ba-4f4e089de5b5)


[Live Demo](https://rag-chatbot-reddykarthikeya1.streamlit.app)

---

## 🚀 Features

* **Semantic Search**: Uses Sentence Transformers for state-of-the-art semantic retrieval.
* **Fast Vector Search**: FAISS enables lightning-fast similarity search, even for large corpora.
* **Flexible QA Models**: Choose from multiple Hugging Face QA models for best speed/accuracy tradeoff.
* **Document Upload**: Supports `.txt` and `.pdf` uploads; chunking for large documents.
* **User-Friendly UI**: All settings explained with tooltips; easy for non-experts to use.
* **Downloadable Q&A History**: Export your session as a CSV.
* **GPU Support**: Automatically uses GPU if available for blazing-fast inference.
* **Modular Codebase**: Clean, extensible Python modules for easy maintenance and upgrades.
* **Usage Analytics**: Displays total queries served and top question types for monitoring and observability.

## 🖥️ How It Works

1. **Upload a Document**: Upload a `.txt` or `.pdf` file, or use the default sample corpus.
2. **Chunking (Optional)**: Large documents are split into overlapping chunks for better retrieval.
3. **Semantic Embedding**: Each passage is embedded using your selected Sentence Transformer model.
4. **Vector Indexing**: FAISS builds a fast similarity index over all embeddings.
5. **Ask Questions**: Enter your question; the app retrieves the most relevant passages and uses a QA model to extract the answer.
6. **Review & Download**: See the context used for each answer and download your Q&A history.
7. **Monitor Usage**: View analytics for total queries and top question types.

## 🛠️ Project Structure

```text
rag_qa_streamlit/
│
├── app.py                  # Streamlit app entry point
├── requirements.txt        # All dependencies
├── data/
│   └── sample_corpus.txt   # Default knowledge base
├── modules/
│   ├── __init__.py
│   ├── file_utils.py       # File loading, PDF parsing, chunking
│   ├── model_utils.py      # Model loading, GPU detection
│   └── rag_utils.py        # FAISS, retrieval, QA logic
└── README.md
```

## ⚡ Quickstart

1. **Clone the repo**:

   ```bash
   git clone https://github.com/<your-username>/rag_qa_streamlit.git
   cd rag_qa_streamlit
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally**:

   ```bash
   streamlit run app.py
   ```

4. **Or try it live**: [Live Demo](https://rag-chatbot-reddykarthikeya1.streamlit.app)

## 🧩 Customization

* **Add your own corpus**: Place a `.txt` or `.pdf` in the app, or edit `data/sample_corpus.txt`.
* **Add more models**: Edit the dropdowns in `app.py` and update `requirements.txt` as needed.
* **Tune chunking**: Adjust chunk size and overlap for best retrieval results.

## 📚 Example Use Cases

* Enterprise Knowledge Base Q&A
* Research Paper Summarization
* Customer Support Automation
* Educational Chatbots

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgements

* [Streamlit](https://streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [PyPDF2](https://pypi.org/project/PyPDF2/)
* [Pandas](https://pandas.pydata.org/)

Enjoy your RAG-powered Q&A system! If you use this project, please ⭐️ the repo and share your feedback.