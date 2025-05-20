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
    # Add more image types if DeepDoc or other parsers handle them as primary docs
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    # Audio/Video if needed later
    # MP3 = "mp3"
    # WAV = "wav"
    # MP4 = "mp4"
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
    """
    Generates a unique and safe filename for storage.
    Example: <kb_id>_<timestamp>_<original_filename_sanitized>
    This is a simple version; RAGflow's `duplicate_name` checked DB.
    For local storage, ensuring uniqueness within the KB's folder is key.
    """
    import time
    import re
    
    timestamp = int(time.time())
    # Sanitize original filename: remove special chars, limit length
    sanitized_name = re.sub(r'[^\w\-\.]', '_', original_filename)
    sanitized_name = sanitized_name[:50] # Limit length of original name part
    
    # This doesn't guarantee global uniqueness if called rapidly,
    # but for single user, combined with kb_id folder, it's often sufficient.
    # A more robust way would be UUIDs for filenames.
    # RAGflow checked against existing names in DB for that KB.
    # For now, we'll just prefix. The actual stored path will be in DB.
    return f"{timestamp}_{sanitized_name}"