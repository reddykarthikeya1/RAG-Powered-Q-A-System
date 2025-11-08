"""
File utility functions for loading and parsing text files and PDFs.
"""
import io
import logging
from typing import List, Optional, Callable, Union
from pathlib import Path

import pypdf

logger = logging.getLogger(__name__)

def load_corpus(path: Union[str, Path]) -> List[str]:
    """
    Load a text corpus from a file.
    
    Args:
        path: Path to the text file containing the corpus
        
    Returns:
        List of non-empty lines from the file
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        UnicodeDecodeError: If the file cannot be decoded as UTF-8
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        logger.error(f"Failed to decode file {path} as UTF-8")
        raise

def extract_text_from_pdf(file) -> List[str]:
    """
    Extract text from a PDF file.
    
    Args:
        file: A file-like object containing PDF data
        
    Returns:
        List of text passages extracted from the PDF
    """
    try:
        pdf_reader = pypdf.PdfReader(file)
        
        if len(pdf_reader.pages) == 0:
            logger.warning("PDF file has no pages")
            return []
        
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Could not extract text from a PDF page: {e}")
                continue  # Skip problematic pages
                
        passages = [p.strip() for p in text.split('\n') if p.strip()]
        return passages
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise

def parse_uploaded_file(
    uploaded_file, 
    chunk_fn: Optional[Callable[[str], List[str]]] = None
) -> List[str]:
    """
    Parse an uploaded file based on its type.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        chunk_fn: Optional function to chunk the text
        
    Returns:
        List of text passages from the file
    """
    try:
        # Validate file size (limit to 50MB)
        if hasattr(uploaded_file, 'size') and uploaded_file.size > 50 * 1024 * 1024:  # 50MB
            logger.error("File size exceeds 50MB limit")
            return []

        if uploaded_file.type == "text/plain":
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            text = stringio.read()
        elif uploaded_file.type == "application/pdf":
            text = "\n".join(extract_text_from_pdf(uploaded_file))
        else:
            logger.warning(f"Unsupported file type: {uploaded_file.type}")
            return []
        
        # Sanitize text to remove potential security issues
        text = sanitize_text(text)

        if chunk_fn:
            return chunk_fn(text)
        else:
            return [line.strip() for line in text.split('\n') if line.strip()]
    except UnicodeDecodeError:
        logger.error("Failed to decode uploaded file as UTF-8")
        return []
    except AttributeError:
        # Handle cases where uploaded_file doesn't have required attributes
        logger.error("Uploaded file object is missing required attributes")
        return []
    except Exception as e:
        logger.error(f"Error parsing uploaded file: {e}")
        return []


def sanitize_text(text: str) -> str:
    """
    Sanitize text to remove potential security issues.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text
    """
    # Remove null bytes which can cause issues
    text = text.replace('\x00', '')
    
    # Remove control characters that might cause issues
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
    
    return text