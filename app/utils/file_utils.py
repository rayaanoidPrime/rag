import os
from enum import Enum

class FileType(Enum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    HTML = "html"
    HTM = "htm"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    OTHER = "other"

KNOWN_EXTENSIONS = {
    ".pdf": FileType.PDF,
    ".txt": FileType.TXT,
    ".md": FileType.MD,
    ".docx": FileType.DOCX,
    ".pptx": FileType.PPTX,
    ".xlsx": FileType.XLSX,
    ".csv": FileType.CSV,
    ".html": FileType.HTML,
    ".htm": FileType.HTM,
    ".png": FileType.PNG,
    ".jpg": FileType.JPG,
    ".jpeg": FileType.JPEG,
}

def get_file_type(filename: str) -> FileType:
    """
    Determines the FileType based on the filename's extension.
    """
    _, ext = os.path.splitext(filename)
    return KNOWN_EXTENSIONS.get(ext.lower(), FileType.OTHER)

def generate_safe_filename(kb_id: str, original_filename: str) -> str:
    # ... (existing function)
    import time
    import re
    
    timestamp = int(time.time())
    sanitized_name = re.sub(r'[^\w\-\.]', '_', original_filename)
    sanitized_name = sanitized_name[:50] 
    return f"{timestamp}_{sanitized_name}"

def get_pdf_page_count(file_path: str) -> int:
    """
    Gets the total number of pages in a PDF file.
    Returns 0 if the file is not a PDF or an error occurs.
    """
    if not file_path.lower().endswith(".pdf"):
        return 0
    try:
        with pdfplumber.open(file_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        # Consider logging this error
        # import logging
        # logging.error(f"Could not get page count for PDF {file_path}: {e}")
        print(f"Warning: Could not get page count for PDF {file_path}: {e}") # Use print if logger not set up here
        return 0 # Or raise an error if page count is critical