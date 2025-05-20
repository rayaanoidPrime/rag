import re
from app.config import RETURN_SEPARATOR, METADATA_KEYS_TO_IGNORE_FOR_SEMANTIC_LLM_CONTEXT, METADATA_KEYS_TO_IGNORE_FOR_KEYWORD

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
                suffix_parts.append(f"{key.replace('_', ' ').title()}: {value_str}")
        
        if suffix_parts:
            return RETURN_SEPARATOR + "--- Metadata ---" + RETURN_SEPARATOR + (RETURN_SEPARATOR.join(suffix_parts)) + RETURN_SEPARATOR + "--- End Metadata ---"
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
            return RETURN_SEPARATOR + "--- Metadata ---" + RETURN_SEPARATOR + (RETURN_SEPARATOR.join(suffix_parts)) + RETURN_SEPARATOR + "--- End Metadata ---"
        return ""

def extract_blurb(text: str, max_length: int = 100) -> str: # Simplified blurb
    if not text: return ""
    # Simple first N chars, can be improved with sentence splitting
    return text[:max_length].strip() + "..." if len(text) > max_length else text.strip()

def format_metadata_for_llm_context(metadata: dict) -> str:
    """
    Formats selected metadata into a human-readable string for LLM context.
    Example output: "[Source: MyDoc.pdf, Page: 5, Style: heading_1]"
    """
    if not metadata:
        return ""

    parts = []
    doc_name = metadata.get("doc_name")
    page_num = metadata.get("original_page_number") # 0-indexed from parser
    style = metadata.get("style")
    
    # Add more important metadata fields as needed
    # is_table = metadata.get("is_table", False) # Example

    if doc_name:
        parts.append(f"Source: {doc_name}")
    if page_num is not None: # Check for None, as 0 is a valid page
        parts.append(f"Page: {page_num + 1}") # Display as 1-indexed to user/LLM
    if style and style not in ["parsed_text_section", "text"]: # Avoid generic styles
        parts.append(f"Section Type: {style.replace('_', ' ').title()}")
    # if is_table:
    #     parts.append("Content Type: Table")

    if not parts:
        return ""
        
    return f" [{', '.join(parts)}]" # Enclose in brackets, comma-separated