import logging
import uuid
import xxhash # For task digest, if we implement task reuse later
from typing import List, Dict, Any, Optional
from peewee import DoesNotExist

from app.database import DB
from app.database.models import Task, Document, KnowledgeBase
from app.utils.file_utils import FileType # Assuming FileType enum is in file_utils
from app.config import MAX_TASK_WORKERS # For thread pool later

# Import specific parser logic if needed for task breakdown (e.g., PDF page count)
# For now, we'll keep it generic or assume info is in doc.parser_config
# from app.document_processing.txtai_parser import TxtaiDocumentParser # Example
# Potentially, we'd need a way to get PDF page count without full parsing here.
# For simplicity, we'll assume page ranges are provided in parser_config or we process whole doc.

logger = logging.getLogger(__name__)

# Simulate a task queue (in-memory for now, or could use a DB table as a queue)
# For actual async, we'll use ThreadPoolExecutor in the Orchestrator.
# This service primarily focuses on DB operations for tasks.

class TaskService:

    @staticmethod
    @DB.connection_context()
    def create_task_in_db(doc_id: str,
                          from_page: int = 0,
                          to_page: int = -1, # -1 means not applicable or to end
                          status: str = "pending",
                          # digest: Optional[str] = None # For task reuse, if implemented
                          ) -> Task:
        """Creates a single task record in the database."""
        task_id = str(uuid.uuid4().hex)
        task = Task.create(
            id=task_id,
            doc_id=doc_id,
            from_page=from_page,
            to_page=to_page,
            status=status,
            progress=0.0,
            # digest=digest
        )
        logger.info(f"Task record created: ID={task_id} for Doc ID={doc_id}, Pages: {from_page}-{to_page}")
        return task

    @staticmethod
    @DB.connection_context()
    def get_task_by_id(task_id: str) -> Optional[Task]:
        try:
            return Task.get(Task.id == task_id)
        except DoesNotExist:
            return None

     @staticmethod
    @DB.connection_context()
    def get_tasks_for_document_by_status_in(doc_id: str, statuses: List[str]) -> List[Task]:
        return list(Task.select().where((Task.doc_id == doc_id) & (Task.status.in_(statuses))))

    @staticmethod
    @DB.connection_context()
    def update_task_status(task_id: str, status: str, progress: Optional[float] = None, error_message: Optional[str] = None):
        try:
            task = Task.get(Task.id == task_id)
            task.status = status
            if progress is not None:
                task.progress = progress
            if status == "failed" and error_message:
                task.error_message = error_message
            elif status == "completed":
                task.progress = 1.0
                task.error_message = None
            task.save()
            logger.info(f"Task '{task_id}' status updated to '{status}'. Progress: {task.progress}")
        except DoesNotExist:
            logger.error(f"Failed to update status for non-existent task ID '{task_id}'.")

    @staticmethod
    @DB.connection_context()
    def delete_tasks_for_document(doc_id: str):
        deleted_count = Task.delete().where(Task.doc_id == doc_id).execute()
        logger.info(f"Deleted {deleted_count} tasks for document ID '{doc_id}'.")


     @staticmethod
    @DB.connection_context()
    def create_processing_tasks_for_document(doc: Document) -> List[Task]:
        # ... (initial checks for doc and kb) ...
        kb = doc.kb_id 
        parser_config_from_kb = kb.parser_config if kb else {}
        doc_file_type = doc.file_type
        # Get the actual parser ID configured for this KB
        kb_parser_id_setting = kb.parser_id # e.g., 'naive', 'deepdoc' (future)

        created_tasks_in_db = []

        # --- IF CURRENT PARSER (e.g., TxtaiDocumentParser) IS NOT PAGE-AWARE ---
        # And we are using a parser like our current TxtaiDocumentParser ('naive' might map to this)
        # that processes the whole file regardless of page ranges.
        # For such parsers, we should only create ONE task for the whole document to avoid duplication.
        
        # This is a temporary check. Ideally, parsers would declare their capabilities.
        # Assuming 'naive' is our TxtaiDocumentParser or similar non-page-aware.
        IS_PARSER_PAGE_AWARE = False # Default to false
        if kb_parser_id_setting == "deepdoc": # Placeholder for when DeepDoc is integrated
            IS_PARSER_PAGE_AWARE = True
        
        # If the document is a PDF AND the configured parser is page-aware AND there are page settings:
        if doc_file_type == FileType.PDF.value and IS_PARSER_PAGE_AWARE:
            task_page_size = parser_config_from_kb.get("task_page_size", 0)
            page_ranges_config = parser_config_from_kb.get("pages")

            if page_ranges_config and isinstance(page_ranges_config, list):
                logger.info(f"PDF {doc.id} with page-aware parser: Using page ranges from config: {page_ranges_config}")
                for start_page, end_page in page_ranges_config:
                    from_p = max(0, int(start_page) - 1)
                    to_p = int(end_page) - 1 
                    
                    if task_page_size > 0:
                        current_sub_page = from_p
                        while current_sub_page <= to_p:
                            sub_to_page = min(current_sub_page + task_page_size - 1, to_p)
                            db_task = TaskService.create_task_in_db(doc.id, current_sub_page, sub_to_page)
                            created_tasks_in_db.append(db_task)
                            current_sub_page = sub_to_page + 1
                    else:
                        db_task = TaskService.create_task_in_db(doc.id, from_p, to_p)
                        created_tasks_in_db.append(db_task)
            elif task_page_size > 0:
                # TODO: Need total page count for the PDF to use task_page_size effectively without explicit ranges.
                # This would require a lightweight PDF page count utility callable here or info stored on Document.
                logger.warning(f"PDF {doc.id}: task_page_size defined but no explicit page_ranges and no total page count available. Creating single task.")
                db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=-1)
                created_tasks_in_db.append(db_task)
            else:
                logger.info(f"PDF {doc.id} with page-aware parser: No specific page ranges/size. Creating single task.")
                db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=-1)
                created_tasks_in_db.append(db_task)
        else:
            # For non-PDFs, or PDFs with non-page-aware parsers, or if no page splitting is configured:
            # Create one task for the whole document.
            if doc_file_type == FileType.PDF.value and not IS_PARSER_PAGE_AWARE:
                 logger.info(f"PDF {doc.id} with NON-page-aware parser ('{kb_parser_id_setting}'). Creating single task for whole document to avoid duplication.")
            else:
                logger.info(f"Document {doc.id} (type: {doc_file_type}, parser: '{kb_parser_id_setting}'): Creating single processing task.")
            db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=-1)
            created_tasks_in_db.append(db_task)

        if created_tasks_in_db:
            from app.services.document_service import DocumentService 
            DocumentService.update_document_status(doc.id, status="queued", progress=0.05)
        
        return created_tasks_in_db

    # RAGflow's `reuse_prev_task_chunks` and digest logic is complex and depends on how
    # chunks are identified and stored. We'll skip this optimization for now.
    # If implemented, it would involve:
    # 1. Calculating a `digest` for task parameters (doc_id, pages, parser_config).
    # 2. When creating new tasks, check if a completed task with the same digest exists.
    # 3. If so, link the new document to the existing chunks instead of re-processing.
    # This requires chunks to be stored independently of documents in the vector store,
    # or a robust way to copy/reference them.