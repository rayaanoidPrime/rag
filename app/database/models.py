# In app/database/models.py

from peewee import CharField, BigIntegerField, TextField, FloatField, IntegerField, CompositeKey, ForeignKeyField
from app.database import BaseModel, JSONField, ListField, current_timestamp_ms, timestamp_ms_to_datetime
from app.config import DEFAULT_PARSER_ID, DEFAULT_PARSER_CONFIG, DEFAULT_KB_DESCRIPTION, DEFAULT_KB_LANGUAGE, DEFAULT_KB_EMBED_MODEL, DEFAULT_KB_SIMILARITY_THRESHOLD, DEFAULT_KB_VECTOR_WEIGHT
from . import get_db
import streamlit as st

logger = st.logger.get_logger(__name__)

class KnowledgeBase(BaseModel):
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the knowledge base")
    # Add unique=True to name. If migrating an existing DB, handle potential duplicates first.
    name = CharField(max_length=128, unique=True, index=True) 
    description = TextField(null=True, default=DEFAULT_KB_DESCRIPTION)
    language = CharField(max_length=32, default=DEFAULT_KB_LANGUAGE)
    embd_model = CharField(max_length=255, default=DEFAULT_KB_EMBED_MODEL)
    parser_id = CharField(max_length=32, default=DEFAULT_PARSER_ID)
    parser_config = JSONField(default=lambda: DEFAULT_PARSER_CONFIG)
    similarity_threshold = FloatField(default=DEFAULT_KB_SIMILARITY_THRESHOLD)
    vector_weight = FloatField(default=DEFAULT_KB_VECTOR_WEIGHT)
    document_count = IntegerField(default=0)

    class Meta:
        db_table = 'knowledge_base'

class Document(BaseModel):
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the document")
    kb_id = ForeignKeyField(KnowledgeBase, backref='documents', field='id', on_delete='CASCADE', index=True)
    
    source_type = CharField(max_length=128, default='local', help_text="e.g., 'local', 'url'")
    name = CharField(max_length=255, help_text="Original filename or document title")
    file_path = CharField(max_length=1024)
    file_size = BigIntegerField(default=0, help_text="Size in bytes")
    file_type = CharField(max_length=32, null=True, help_text="Detected file type, e.g., 'pdf', 'txt'")

    status = CharField(max_length=32, default='pending', index=True, help_text="e.g., pending, processing, completed, failed")
    chunk_count = IntegerField(default=0, help_text="Number of text chunks generated")
    token_count = IntegerField(default=0, help_text="Total tokens from content (approximate)")
    progress = FloatField(default=0.0, help_text="Processing progress (0.0 to 1.0)")
    error_message = TextField(null=True, help_text="Error message if processing failed")

    # --- New Fields for C4 ---
    total_pages = IntegerField(default=0, help_text="Total pages in the document, if applicable (e.g., for PDFs)")
    doc_summary = TextField(null=True, help_text="LLM-generated or extracted summary of the document")
    # layout_analysis_results can store list of dicts:
    # e.g., [{"type": "figure", "page": 5, "bbox": [x,y,w,h], "stored_image_path": "/path/to/fig.png", "caption": "..."}]
    layout_analysis_results = JSONField(null=True, help_text="Structured data from layout analysis, e.g., paths to extracted images/tables")
    # --- End New Fields ---

    class Meta:
        db_table = 'document'
        indexes = (
            (('kb_id', 'name'), True), 
        )

class Task(BaseModel):
    # ... (Task model remains unchanged for C4 unless specific artifact processing tasks are added)
    id = CharField(max_length=32, primary_key=True, help_text="Unique ID for the task")
    doc_id = ForeignKeyField(Document, backref='tasks', field='id', on_delete='CASCADE', index=True)
    from_page = IntegerField(default=0) 
    to_page = IntegerField(default=-1) 
    status = CharField(max_length=32, default='pending', index=True, help_text="e.g., pending, processing, completed, failed")
    progress = FloatField(default=0.0, help_text="Task progress (0.0 to 1.0)")
    error_message = TextField(null=True)

    class Meta:
        db_table = 'task'


def create_tables():
    db = get_db()
    with db.connection_context():
        tables_to_create = [KnowledgeBase, Document, Task]
        db.create_tables(tables_to_create, safe=True) 
    logger.info("Database tables (KnowledgeBase, Document, Task) checked/created.")