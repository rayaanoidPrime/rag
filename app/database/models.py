from peewee import CharField, BigIntegerField, TextField, FloatField, IntegerField, CompositeKey, ForeignKeyField
from app.database import BaseModel, JSONField, ListField, current_timestamp_ms, timestamp_ms_to_datetime
from app.config import DEFAULT_PARSER_ID, DEFAULT_PARSER_CONFIG, DEFAULT_KB_DESCRIPTION, DEFAULT_KB_LANGUAGE, DEFAULT_KB_EMBED_MODEL, DEFAULT_KB_SIMILARITY_THRESHOLD, DEFAULT_KB_VECTOR_WEIGHT
from . import get_db
import streamlit as st

# We won't implement User, LLMFactory, LLM, Dialog, Conversation models for now
# to keep focus on KB, Document, Task. These can be added later if needed.
logger = st.logger.get_logger(__name__)

class KnowledgeBase(BaseModel):
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the knowledge base")
    name = CharField(max_length=128, index=True)
    description = TextField(null=True, default=DEFAULT_KB_DESCRIPTION)
    language = CharField(max_length=32, default=DEFAULT_KB_LANGUAGE)
    # embd_model refers to the embedding model used, e.g., path or name
    embd_model = CharField(max_length=255, default=DEFAULT_KB_EMBED_MODEL)
    # parser_id identifies the parsing strategy (e.g., 'deepdoc', 'naive')
    parser_id = CharField(max_length=32, default=DEFAULT_PARSER_ID)
    # parser_config stores specific settings for the chosen parser, as JSON
    parser_config = JSONField(default=lambda: DEFAULT_PARSER_CONFIG) # Use lambda for mutable default

    # Fields from RAGflow that might be relevant for retrieval strategy
    similarity_threshold = FloatField(default=DEFAULT_KB_SIMILARITY_THRESHOLD)
    vector_weight = FloatField(default=DEFAULT_KB_VECTOR_WEIGHT) # If using hybrid search later

    # created_by removed (single user context for now)
    # document_count can be a derived property or a field updated by triggers/app logic
    document_count = IntegerField(default=0)

    class Meta:
        db_table = 'knowledge_base'

class Document(BaseModel):
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the document")
    # kb_id links this document to a KnowledgeBase
    kb_id = ForeignKeyField(KnowledgeBase, backref='documents', field='id', on_delete='CASCADE', index=True)
    
    source_type = CharField(max_length=128, default='local', help_text="e.g., 'local', 'url'")
    name = CharField(max_length=255, help_text="Original filename or document title")
    # location stores the path/URL or internal storage key
    # For local storage, this will be the path relative to LOCAL_FILE_STORAGE_PATH or absolute.
    file_path = CharField(max_length=1024) # Changed from 'location' for clarity
    file_size = BigIntegerField(default=0, help_text="Size in bytes")
    file_type = CharField(max_length=32, null=True, help_text="Detected file type, e.g., 'pdf', 'txt'")

    # Processing status and results
    status = CharField(max_length=32, default='pending', index=True, help_text="e.g., pending, processing, completed, failed")
    chunk_count = IntegerField(default=0, help_text="Number of chunks generated")
    token_count = IntegerField(default=0, help_text="Total tokens from content (approximate)")
    progress = FloatField(default=0.0, help_text="Processing progress (0.0 to 1.0)")
    error_message = TextField(null=True, help_text="Error message if processing failed")

    # thumbnail_path = CharField(max_length=1024, null=True) # Path to thumbnail if generated

    class Meta:
        db_table = 'document'
        indexes = (
            (('kb_id', 'name'), True), # Unique document name per KB
        )

class Task(BaseModel):
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the task")
    doc_id = ForeignKeyField(Document, backref='tasks', field='id', on_delete='CASCADE', index=True)
    
    # For chunking/processing parts of a document, e.g., page ranges for PDF
    from_page = IntegerField(default=0) # 0-indexed
    to_page = IntegerField(default=-1) # -1 can mean to end of document or not applicable

    status = CharField(max_length=32, default='pending', index=True, help_text="e.g., pending, processing, completed, failed")
    progress = FloatField(default=0.0, help_text="Task progress (0.0 to 1.0)")
    # digest = CharField(max_length=128, null=True, index=True, help_text="Hash digest of task parameters for potential reuse")
    # chunk_ids = ListField(default=lambda: []) # Store IDs of chunks produced by this task (if vector store uses own IDs)
    error_message = TextField(null=True)

    class Meta:
        db_table = 'task'

# --- Utility to create tables ---
def create_tables():
    """Creates all database tables if they don't exist."""
    db = get_db()
    with db.connection_context(): # Ensures connection is managed
        # Add models in dependency order if necessary, though Peewee handles it
        tables_to_create = [KnowledgeBase, Document, Task]
        # Check existence before creating (optional, Peewee create_tables has if_not_exists)
        # for model_class in tables_to_create:
        #     if not db.table_exists(model_class._meta.table_name):
        #         logger.info(f"Table {model_class._meta.table_name} does not exist. Creating.")
        #     else:
        #         logger.info(f"Table {model_class._meta.table_name} already exists.")
        db.create_tables(tables_to_create, safe=True) # safe=True means IF NOT EXISTS
    logger.info("Database tables checked/created.")

