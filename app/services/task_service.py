import logging
import uuid
import xxhash # For task digest, if we implement task reuse later
from typing import List, Dict, Any, Optional
from peewee import DoesNotExist

from app.database import DB
from app.database.models import Task, Document, KnowledgeBase
from app.utils.file_utils import FileType , get_pdf_page_count # Assuming FileType enum is in file_utils
from app.config import MAX_TASK_WORKERS, PARSER_MAPPING, DEEPDOC_PARSER_BY_FILE_TYPE, DEFAULT_PARSER_ID
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
    def get_tasks_for_document(doc_id: str, status: Optional[str] = None) -> List[Task]:
        """Helper to get tasks for a document, optionally filtered by status."""
        query = Task.select().where(Task.doc_id == doc_id)
        if status:
            query = query.where(Task.status == status)
        return list(query)

    @staticmethod
    @DB.connection_context()
    def create_processing_tasks_for_document(doc: Document) -> List[Task]:
        if not doc or not doc.kb_id:
            logger.error(f"Cannot create tasks: Document {getattr(doc, 'id', 'N/A')} or its KB is invalid.")
            return []

        kb = doc.kb_id
        parser_config_from_kb = kb.parser_config or {}
        doc_file_type_str = doc.file_type # e.g., "pdf", "docx"

        # Determine the actual parser_id and its page-awareness
        effective_parser_id = kb.parser_id or DEFAULT_PARSER_ID
        is_parser_page_aware = False

        if effective_parser_id == "deepdoc_auto":
            resolved_parser_key = DEEPDOC_PARSER_BY_FILE_TYPE.get(doc_file_type_str.lower())
            if resolved_parser_key:
                parser_info = PARSER_MAPPING.get(resolved_parser_key, {})
                is_parser_page_aware = parser_info.get("is_page_aware", False)
                effective_parser_id = resolved_parser_key # Update for logging
            else: # Fallback for deepdoc_auto if file type not in DEEPDOC_PARSER_BY_FILE_TYPE
                parser_info = PARSER_MAPPING.get("naive_txtai", {}) # Default fallback
                is_parser_page_aware = parser_info.get("is_page_aware", False)
                effective_parser_id = "naive_txtai"
        else:
            parser_info = PARSER_MAPPING.get(effective_parser_id, {})
            is_parser_page_aware = parser_info.get("is_page_aware", False)
        
        logger.info(f"For Doc {doc.id} (type: {doc_file_type_str}), effective parser_id: '{effective_parser_id}', page_aware: {is_parser_page_aware}")

        created_tasks_in_db = []

        # If the document is a PDF AND the configured parser is page-aware:
        if doc_file_type_str == FileType.PDF.value and is_parser_page_aware:
            task_page_size = int(parser_config_from_kb.get("task_page_size", 0)) # Ensure int
            page_ranges_config = parser_config_from_kb.get("pages") # List of [start, end] (1-indexed)

            total_pages = get_pdf_page_count(doc.file_path)
            if total_pages == 0:
                logger.warning(f"PDF {doc.id}: Could not determine total page count. Creating a single task for the whole document.")
                db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=-1) # 0-indexed, -1 for all
                created_tasks_in_db.append(db_task)
            elif page_ranges_config and isinstance(page_ranges_config, list):
                logger.info(f"PDF {doc.id} with page-aware parser: Using page ranges from config: {page_ranges_config}")
                for start_page_1_idx, end_page_1_idx in page_ranges_config:
                    # Convert 1-indexed from config to 0-indexed for tasks
                    from_p_0_idx = max(0, int(start_page_1_idx) - 1)
                    to_p_0_idx = min(int(end_page_1_idx) - 1, total_pages - 1) # Cap at actual last page

                    if from_p_0_idx > to_p_0_idx:
                        logger.warning(f"PDF {doc.id}: Invalid page range [{start_page_1_idx}-{end_page_1_idx}] skipped.")
                        continue
                    
                    if task_page_size > 0: # Further split this range by task_page_size
                        current_sub_page_0_idx = from_p_0_idx
                        while current_sub_page_0_idx <= to_p_0_idx:
                            sub_to_page_0_idx = min(current_sub_page_0_idx + task_page_size - 1, to_p_0_idx)
                            db_task = TaskService.create_task_in_db(doc.id, current_sub_page_0_idx, sub_to_page_0_idx)
                            created_tasks_in_db.append(db_task)
                            current_sub_page_0_idx = sub_to_page_0_idx + 1
                    else: # Process the whole configured range as one task
                        db_task = TaskService.create_task_in_db(doc.id, from_p_0_idx, to_p_0_idx)
                        created_tasks_in_db.append(db_task)
            elif task_page_size > 0:
                logger.info(f"PDF {doc.id} with page-aware parser: Splitting by task_page_size: {task_page_size}")
                current_page_0_idx = 0
                while current_page_0_idx < total_pages:
                    to_p_0_idx = min(current_page_0_idx + task_page_size - 1, total_pages - 1)
                    db_task = TaskService.create_task_in_db(doc.id, current_page_0_idx, to_p_0_idx)
                    created_tasks_in_db.append(db_task)
                    current_page_0_idx = to_p_0_idx + 1
            else:
                logger.info(f"PDF {doc.id} with page-aware parser: No specific page ranges/size. Creating single task for all {total_pages} pages.")
                db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=total_pages -1 if total_pages > 0 else -1)
                created_tasks_in_db.append(db_task)
        else:
            # For non-PDFs, or PDFs with non-page-aware parsers, or if no page splitting is configured:
            # Create one task for the whole document.
            log_reason = "non-PDF" if doc_file_type_str != FileType.PDF.value else f"parser '{effective_parser_id}' not page-aware or no page splitting defined"
            logger.info(f"Document {doc.id} (type: {doc_file_type_str}, reason: {log_reason}): Creating single processing task for whole document.")
            db_task = TaskService.create_task_in_db(doc.id, from_page=0, to_page=-1)
            created_tasks_in_db.append(db_task)

        if created_tasks_in_db:
            from app.services.document_service import DocumentService # Local import to avoid circularity
            DocumentService.update_document_status(doc.id, status="queued", progress=0.05)
        else:
             logger.warning(f"No tasks were created for document {doc.id}. This might indicate an issue with configuration or PDF page count.")
             from app.services.document_service import DocumentService
             DocumentService.update_document_status(doc.id, status="failed", error_message="Task creation failed (e.g. PDF page count error or invalid page ranges).")


        return created_tasks_in_db
