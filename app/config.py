import os
import platform
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# General App Settings
APP_TITLE = os.environ.get("APP_TITLE", "🚀 RAG with Graph Capabilities")
TOKENIZERS_PARALLELISM = os.environ.get("TOKENIZERS_PARALLELISM", "false")

# LLM Configuration
DEFAULT_LLM_MODEL = (
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    if platform.machine() in ("x86_64", "AMD")
    else "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)
LLM_MODEL = os.environ.get("LLM", DEFAULT_LLM_MODEL)
LLM_MAX_LENGTH = int(os.environ.get("LLM_MAX_LENGTH", 4096)) # Max output tokens for LLM
LLM_INFER_TOPICS_MAX_LENGTH = int(os.environ.get("LLM_INFER_TOPICS_MAX_LENGTH", 64)) # Shorter for topics
TOPICS_BATCH_SIZE = os.environ.get("TOPICSBATCH")
if TOPICS_BATCH_SIZE:
    TOPICS_BATCH_SIZE = int(TOPICS_BATCH_SIZE)

# Embedding Model Configuration
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "intfloat/e5-large")
EMBEDDING_INSTRUCTIONS = {"query": "query: ", "data": "passage: "}

# Vector Store / Embeddings Database Configuration
VECTOR_DB_PATH = os.environ.get("EMBEDDINGS", "data/embeddings_db") # Local path for txtai
# VECTOR_DB_CLOUD_PROVIDER = "huggingface-hub" (Example for cloud loading)
# VECTOR_DB_CLOUD_CONTAINER = "neuml/txtai-wikipedia-slim" (Example)
PERSIST_DB_PATH = os.environ.get("PERSIST", "data/embeddings_db_persisted") # Path to save/persist
INITIAL_DATA_PATH = os.environ.get("DATA") # Optional path to data directory to index on startup

# RAG Configuration
RAG_CONTEXT_SIZE = int(os.environ.get("CONTEXT", 10))
RAG_SYSTEM_PROMPT = "You are a friendly assistant. You answer questions from users."
RAG_TEMPLATE = """
Answer the following question based *only* on the provided context.
When you use information from a specific part of the context, you **MUST** cite it using the corresponding identifier like [CHUNK_ID] at the end of the sentence or paragraph that uses it.
For example: "The sky is blue [CHUNK_1]. Photosynthesis requires sunlight [CHUNK_2]."
If multiple context parts support a statement, cite all relevant ones, e.g., "Water is essential [CHUNK_1][CHUNK_3]."
Do not make up information. If the context doesn't provide an answer, say so.

Context:
{context}

Question: {question}

Answer:
"""

# Document Processing Configuration
TEXTRACTOR_BACKEND = os.environ.get("TEXTBACKEND", "available") # For txtai's Textractor

# GraphRAG Configuration
GRAPH_APPROXIMATE_SEARCH = False # As per original
GRAPH_MIN_SCORE = 0.7 # As per original
GRAPH_DEDUPLICATION_THRESHOLD = 0.9
GRAPH_CONTEXT_LIMIT = RAG_CONTEXT_SIZE # Max nodes to return for graph context

# UI Examples
UI_EXAMPLES = [
    x.strip()
    for x in os.environ.get(
        "EXAMPLES",
        "Who created Linux?;gq: Tell me about Linux;linux -> macos -> microsoft windows;linux -> macos -> microsoft windows gq: Tell me about Linux",
    ).split(";")
]

# Ensure data directories exist if specified for local storage
if VECTOR_DB_PATH and not VECTOR_DB_PATH.startswith("neuml/"): # Basic check if it's a local path
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
if PERSIST_DB_PATH:
    os.makedirs(os.path.dirname(PERSIST_DB_PATH), exist_ok=True)

# PostgreSQL Database Configuration
# Loaded from .env file or defaults
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", 5432))
DB_USER = os.environ.get("DB_USER", "your_db_user") # CHANGE THIS
DB_PASSWORD = os.environ.get("DB_PASSWORD", "your_db_password") # CHANGE THIS
DB_NAME = os.environ.get("DB_NAME", "rag_app_db") # CHANGE THIS
DB_MAX_CONNECTIONS = int(os.environ.get("DB_MAX_CONNECTIONS", 20))
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Local Storage Path for Uploaded Documents (replaces MinIO for now)
LOCAL_FILE_STORAGE_PATH = os.environ.get("LOCAL_FILE_STORAGE_PATH", "data/uploaded_files")
os.makedirs(LOCAL_FILE_STORAGE_PATH, exist_ok=True) # Ensure directory exists

# Chunking and Parser Defaults (can be overridden per KB)
DEFAULT_PARSER_ID = "naive" # Corresponds to a parser type, e.g., TxtaiDocumentParser for now
DEFAULT_PARSER_CONFIG = {"pages": [[1, 1000000]], "layout_recognize": "Plain Text"} # Example
DEFAULT_CHUNK_TOKEN_NUM = int(os.environ.get("DEFAULT_CHUNK_TOKEN_NUM", 512))
DEFAULT_CHUNK_DELIMITER = "\n!?;。；！？"

# Default settings for KnowledgeBase if not specified
DEFAULT_KB_DESCRIPTION = "A knowledge base."
DEFAULT_KB_LANGUAGE = "English"
DEFAULT_KB_EMBED_MODEL = EMBEDDING_MODEL_PATH # Use the same embedding model
DEFAULT_KB_SIMILARITY_THRESHOLD = 0.2 # From RAGflow
DEFAULT_KB_VECTOR_WEIGHT = 0.3 # From RAGflow

# For Task Processing (simplified from RAGflow's Redis queue)
# We might use ThreadPoolExecutor initially
MAX_TASK_WORKERS = int(os.environ.get("MAX_TASK_WORKERS", 4))

# --- Parser Configuration ---
DEFAULT_PARSER_ID = "naive_txtai" # Default if not specified by KB

# PARSER_MAPPING maps a KnowledgeBase.parser_id to a parser class and its type
# 'type' can be 'naive' (like TxtaiDocumentParser) or 'deepdoc'
PARSER_MAPPING = {
    "naive_txtai": {"class_path": "app.document_processing.txtai_parser.TxtaiDocumentParser", "type": "naive", "is_page_aware": False},
    
    "deepdoc_pdf": {"class_path": "app.deepdoc_components.parser.pdf_parser.RAGFlowPdfParser", "type": "deepdoc", "is_page_aware": True},
    "deepdoc_docx": {"class_path": "app.deepdoc_components.parser.docx_parser.RAGFlowDocxParser", "type": "deepdoc", "is_page_aware": True},
    "deepdoc_excel": {"class_path": "app.deepdoc_components.parser.excel_parser.RAGFlowExcelParser", "type": "deepdoc", "is_page_aware": False}, # Excel usually sheet-based, not page
    "deepdoc_ppt": {"class_path": "app.deepdoc_components.parser.ppt_parser.RAGFlowPptParser", "type": "deepdoc", "is_page_aware": True},
    "deepdoc_txt": {"class_path": "app.deepdoc_components.parser.txt_parser.RAGFlowTxtParser", "type": "deepdoc", "is_page_aware": False},
    "deepdoc_md": {"class_path": "app.deepdoc_components.parser.markdown_parser.RAGFlowMarkdownParser", "type": "deepdoc", "is_page_aware": False},
    "deepdoc_html": {"class_path": "app.deepdoc_components.parser.html_parser.RAGFlowHtmlParser", "type": "deepdoc", "is_page_aware": False},
    "deepdoc_json": {"class_path": "app.deepdoc_components.parser.json_parser.RAGFlowJsonParser", "type": "deepdoc", "is_page_aware": False},
    
    "deepdoc_auto": {"type": "deepdoc_auto", "is_page_aware": None}, # Page-awareness determined by resolved parser
}

# Mapping file extensions to their default deepdoc parser_id (if using 'deepdoc_auto')
DEEPDOC_PARSER_BY_FILE_TYPE = {
    "pdf": "deepdoc_pdf",
    "docx": "deepdoc_docx",
    "xlsx": "deepdoc_excel",
    "pptx": "deepdoc_ppt",
    "txt": "deepdoc_txt",
    "md": "deepdoc_md",
    "html": "deepdoc_html",
    "htm": "deepdoc_html",
    "json": "deepdoc_json",
    # Note: CSV is handled by excel_parser in RAGFlow, map it if needed.
    # For 'other' file types, 'deepdoc_auto' might fall back to naive or raise error.
}


DEFAULT_PARSER_CONFIG = {
    "pages": [[1, 1000000]], # Default for PDF if using deepdoc
    "layout_recognize": True, # Default for deepdoc PDF
    "pdf_parser_options": { # Specific options for RAGFlowPdfParser
        "need_image": False, # Set to False to avoid figure/table image extraction initially
        "zoomin": 3,
        "return_html_tables": True # Get HTML for tables for easier text conversion
    },
    "txt_parser_options": { # Specific options for RAGFlowTxtParser
        "chunk_token_num": 256, # Example
        "delimiter": "\n!?;。；！？"
    },
    # Add other parser default configs as needed
}

# DeepDoc specific settings (examples, expand as needed)
DEEPDOC_LIGHTEN_MODE = False # Corresponds to LIGHTEN in RAGflow's api.settings
DEEPDOC_PARALLEL_DEVICES = 1 # Set based on available resources, or use MAX_TASK_WORKERS

# Max chunks to process from a single parser call in one task segment
# to prevent memory issues or overly long tasks.
MAX_CHUNKS_PER_TASK_SEGMENT = 500

RETURN_SEPARATOR = "\n" 
# Customize these based on what you want to show/hide in LLM context vs keyword search
METADATA_KEYS_TO_IGNORE_FOR_SEMANTIC_LLM_CONTEXT = ["embedding", "parser_source", "is_table_or_figure", "artifact_image_path"] # For LLM
METADATA_KEYS_TO_IGNORE_FOR_KEYWORD = ["embedding"]

TXT_AI_HYBRID_SEARCH_ENABLED = True
TXT_AI_KEYWORD_INDEX_ENABLED = True  # Set to True to enable hybrid search
# Example for SQLite FTS5 backend (default if keyword=True and no other config)
TXT_AI_KEYWORD_CONFIG = {"backend": "sqlite", "content": True} 
# Example for Whoosh backend (requires `pip install txtai[pipeline-text]`)
# TXT_AI_KEYWORD_CONFIG = {"backend": "whoosh", "path": "data/keyword_index_whoosh"}
# If using Whoosh, ensure its path directory exists:
# if TXT_AI_KEYWORD_INDEX_ENABLED and TXT_AI_KEYWORD_CONFIG.get("backend") == "whoosh":
#     os.makedirs(TXT_AI_KEYWORD_CONFIG["path"], exist_ok=True)