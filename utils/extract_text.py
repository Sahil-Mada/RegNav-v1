"""
Text extraction utilities for PDF, DOCX, and TXT files.
Supports PyMuPDF for PDFs, python-docx for DOCX, and plain text for TXT files.
"""

import os
from typing import Optional
import fitz  # PyMuPDF
from docx import Document


def extract_pdf_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_docx_text(file_path: str) -> str:
    """
    Extract text from a DOCX file using python-docx.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")


def extract_txt_text(file_path: str) -> str:
    """
    Extract text from a TXT file (plain text).
    
    Args:
        file_path: Path to the TXT file
        
    Returns:
        Extracted text as a string
    """
    try:
        # Try UTF-8 first, then fallback to other encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        # If all encodings fail, try with error handling
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.read()
    except Exception as e:
        raise Exception(f"Error extracting text from TXT: {str(e)}")


def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Extract text from a file based on its extension.
    Supports PDF, DOCX, and TXT files.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text as a string, or None if file type is not supported
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        return extract_pdf_text(file_path)
    elif file_ext in ['.docx', '.doc']:
        return extract_docx_text(file_path)
    elif file_ext == '.txt':
        return extract_txt_text(file_path)
    else:
        return None

