import chardet
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# --- For find_codec ---
def find_codec(file_content_bytes: bytes) -> str:
    """Detects encoding of byte content."""
    result = chardet.detect(file_content_bytes)
    return result['encoding'] if result['encoding'] else 'utf-8' # Fallback to utf-8

# --- For num_tokens_from_string ---
def num_tokens_from_string(text: str) -> int:
    """Basic token count. TODO: Replace with proper tokenizer for accuracy."""
    return len(text.split())

# --- For rag_tokenizer (Huqie) ---
# Attempt to import the full tokenizer implementation.
# If it fails, a very basic fallback is used.
try:
    from app.deepdoc_components.parser.rag_tokenizer_impl import tokenizer as rag_tokenizer_instance
    _rag_tokenizer_available = True
except ImportError as e:
    print(f"WARNING: Full rag_tokenizer_impl not available (ImportError: {e}). Using basic fallback tokenizer for CJK text.")
    _rag_tokenizer_available = False

    class BasicFallbackTokenizer:
        def __init__(self):
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()

        def tokenize(self, line: str) -> str:
            # Extremely naive tokenization
            line = re.sub(r"\s+", " ", line).strip()
            # Simple split for CJK, word_tokenize for others (very rough)
            if any('\u4e00' <= char <= '\u9fff' for char in line): # Basic CJK check
                return " ".join(list(line.replace(" ", ""))) 
            
            # Fallback for non-CJK or mixed, basic English tokenization
            try:
                tokens = word_tokenize(line.lower())
                tokens = [self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in tokens if t.isalnum()]
                return " ".join(tokens)
            except Exception: # NLTK might not be fully set up
                 return " ".join(line.split())


        def tag(self, token: str) -> str:
            return "UNK" # Fallback tag

        def is_chinese(self, char_str: str) -> bool:
            if char_str >= '\u4e00' and char_str <= '\u9fa5':
                return True
            return False
            
    rag_tokenizer_instance = BasicFallbackTokenizer()

def tokenize(text: str) -> str:
    return rag_tokenizer_instance.tokenize(text)

def tag(text: str) -> str:
    return rag_tokenizer_instance.tag(text) # type: ignore

def is_chinese(text: str) -> bool:
    if hasattr(rag_tokenizer_instance, 'is_chinese'):
        return rag_tokenizer_instance.is_chinese(text) # type: ignore
    # Fallback if the method is missing on the stub
    return any('\u4e00' <= char <= '\u9fff' for char in text)


# --- For clean_markdown_block (used by picture_stub potentially) ---
def clean_markdown_block(text:str) -> str:
    # Basic placeholder, RAGFlow's might be more complex
    text = text.replace("```markdown", "").replace("```", "")
    return text.strip()