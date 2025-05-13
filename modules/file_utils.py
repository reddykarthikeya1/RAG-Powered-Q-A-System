import PyPDF2
import io

def load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    passages = [p.strip() for p in text.split('\n') if p.strip()]
    return passages

def parse_uploaded_file(uploaded_file, chunk_fn=None):
    if uploaded_file.type == "text/plain":
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
    elif uploaded_file.type == "application/pdf":
        text = "\n".join(extract_text_from_pdf(uploaded_file))
    else:
        return []
    if chunk_fn:
        return chunk_fn(text)
    else:
        return [line.strip() for line in text.split('\n') if line.strip()]