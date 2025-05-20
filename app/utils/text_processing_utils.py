import re
from app.config import RETURN_SEPARATOR # Define this in config.py if not already

# From app/config.py, add:
# RETURN_SEPARATOR = "\n" 
# METADATA_KEYS_TO_IGNORE_FOR_SEMANTIC = ["doc_id", "kb_id", "source_file_path", "embedding"] # Example
# METADATA_KEYS_TO_IGNORE_FOR_KEYWORD = ["embedding"] # Example

def get_metadata_suffix(metadata: dict, semantic: bool = True) -> str:
    if not metadata:
        return ""

    suffix_parts = []
    keys_to_ignore = METADATA_KEYS_TO_IGNORE_FOR_SEMANTIC if semantic else METADATA_KEYS_TO_IGNORE_FOR_KEYWORD
    
    if semantic:
        # For semantic, include key-value pairs for better context
        for key, value in metadata.items():
            if key in keys_to_ignore:
                continue
            
            value_str = ""
            if isinstance(value, list):
                value_str = ", ".join(map(str, value))
            elif value is not None:
                value_str = str(value)
            
            if value_str:
                suffix_parts.append(f"{key}: {value_str}")
        
        if suffix_parts:
            return RETURN_SEPARATOR + "Metadata:" + RETURN_SEPARATOR + (RETURN_SEPARATOR.join(suffix_parts))
        return ""
    else:
        # For keyword, just include values
        for key, value in metadata.items():
            if key in keys_to_ignore:
                continue
            if isinstance(value, list):
                suffix_parts.extend(map(str, value))
            elif value is not None:
                suffix_parts.append(str(value))
        
        if suffix_parts:
            return RETURN_SEPARATOR + " ".join(suffix_parts) # Space separated values
        return ""

def extract_blurb(text: str, max_length: int = 100) -> str: # Simplified blurb
    if not text: return ""
    # Simple first N chars, can be improved with sentence splitting
    return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()
