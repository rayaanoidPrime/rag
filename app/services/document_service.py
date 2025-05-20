import os
import uuid
import shutil
import logging
from typing import Optional, List, Any, Tuple, IO
from peewee import DoesNotExist

from app.database import DB
from app.database.models import Document, KnowledgeBase, Task
from app.config import LOCAL_FILE_STORAGE_PATH
from app.utils.file_utils import get_file_type, generate_safe_filename, FileType, get_pdf_page_count 
from .knowledge_base_service import KnowledgeBaseService # To update KB doc count

logger = logging.getLogger(__name__)

class DocumentService:

    @staticmethod
    def _get_kb_storage_path(kb_id: str) -> str:
        """Returns the dedicated storage path for a given KB."""
        return os.path.join(LOCAL_FILE_STORAGE_PATH, kb_id)

    @staticmethod
    def _ensure_kb_storage_path_exists(kb_id: str):
        """Creates the KB's storage directory if it doesn't exist."""
        path = DocumentService._get_kb_storage_path(kb_id)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    @DB.connection_context()
    def add_document_to_kb(kb: KnowledgeBase,
                           original_filename: str,
                           file_content_stream: IO[bytes], # Stream of bytes (e.g., from request.files)
                           source_type: str = "local",
                           # created_by: Optional[str] = None, # Removed for single user
                           initial_status: str = "pending"
                           ) -> Tuple[Optional[Document], Optional[str]]:
        """
        Adds a document to a Knowledge Base.
        Saves the file locally and creates a Document record in the DB.

        Args:
            kb: The KnowledgeBase Peewee model instance.
            original_filename: The original name of the uploaded file.
            file_content_stream: A file-like object (BytesIO or temp file) to read content from.
            source_type: Origin of the document (e.g., 'local', 'url').
            initial_status: Initial status of the document.

        Returns:
            A tuple (Document instance or None, error message or None).
        """
        if not kb:
            return None, "Knowledge Base not provided or not found."

        # 1. Determine file type
        file_type_enum = get_file_type(original_filename)
        if file_type_enum == FileType.OTHER:
            # Decide if you want to strictly disallow 'OTHER' types or allow and attempt parsing
            # RAGflow had: raise RuntimeError("This type of file has not been supported yet!")
            # For now, we allow it, parsing might fail later.
            logger.warning(f"File '{original_filename}' has an unsupported or unknown type: {file_type_enum.value}. Proceeding, but parsing may fail.")
            
        # 2. Store the file locally
        kb_storage_path = DocumentService._ensure_kb_storage_path_exists(kb.id)
        # Generate a unique name for storage to avoid conflicts within the KB folder
        # RAGflow's `duplicate_name` checked DB to ensure `name` field was unique for that KB.
        # Here, `safe_stored_filename` is for the actual file on disk.
        # `doc_db_name` will be used for the 'name' field in DB and should be unique per KB.
        
        # Check for duplicate original_filename within the same KB
        existing_doc_with_same_name = Document.select().where(
            (Document.kb_id == kb.id) & (Document.name == original_filename)
        ).first()
        if existing_doc_with_same_name:
            # RAGflow appended suffixes. We'll return an error for simplicity or you can implement suffix logic.
            err_msg = f"A document named '{original_filename}' already exists in this Knowledge Base."
            logger.error(err_msg)
            return None, err_msg
        
        doc_db_name = original_filename # Use original filename for DB 'name' field
        
        # For file system storage, ensure a unique name.
        # We can use a UUID for the actual stored file name to guarantee uniqueness.
        stored_file_basename = str(uuid.uuid4().hex) + os.path.splitext(original_filename)[1]
        stored_file_path = os.path.join(kb_storage_path, stored_file_basename)

        file_size = 0
        total_doc_pages = 0 # New variable
         try:
            with open(stored_file_path, "wb") as f_disk:
                shutil.copyfileobj(file_content_stream, f_disk)
            file_size = os.path.getsize(stored_file_path)
            logger.info(f"File '{original_filename}' saved to '{stored_file_path}' for KB '{kb.id}'. Size: {file_size} bytes.")
            
            # --- Get total_pages if it's a PDF ---
            if file_type_enum == FileType.PDF:
                total_doc_pages = get_pdf_page_count(stored_file_path)
                logger.info(f"PDF '{original_filename}' has {total_doc_pages} pages.")
            # For other paginated types (DOCX, PPTX), this would require their respective parsers,
            # which is too heavy for this stage. Can be updated later if needed.

        except IOError as e:
            # ... (error handling for file saving as before) ...
            return None, err_msg

        doc_id = str(uuid.uuid4().hex)
        try:
            document = Document.create(
                id=doc_id,
                kb_id=kb,
                name=doc_db_name, # Using original_filename as DB name (checked for uniqueness per KB)
                source_type=source_type,
                file_path=stored_file_path,
                file_size=file_size,
                file_type=file_type_enum.value,
                status=initial_status,
                chunk_count=0,
                token_count=0,
                progress=0.0,
                total_pages=total_doc_pages, # --- Populate new field ---
                doc_summary=None,            # --- Init new field ---
                layout_analysis_results=None # --- Init new field ---
            )
            KnowledgeBaseService.increment_kb_document_count(kb.id)
            logger.info(f"Document record created for '{doc_db_name}' (ID: {doc_id}) in KB '{kb.id}'.")
            return document, None
        except Exception as e:
            # ... (error handling for DB creation as before) ...
            return None, err_msg

    @staticmethod
    @DB.connection_context()
    def get_document_by_id(doc_id: str) -> Optional[Document]:
        try:
            return Document.get(Document.id == doc_id)
        except DoesNotExist:
            logger.warning(f"Document with ID '{doc_id}' not found.")
            return None

     @staticmethod
    @DB.connection_context()
    def get_document_by_kb_id_and_name(kb_id: str, name: str) -> Optional[Document]:
        try:
            return Document.get((Document.kb_id == kb_id) & (Document.name == name))
        except DoesNotExist:
            return None

    @staticmethod
    @DB.connection_context()
    def increment_document_chunk_and_token_count(doc_id: str, chunk_increment: int, token_increment: int):
        try:
            query = Document.update(
                chunk_count=Document.chunk_count + chunk_increment,
                token_count=Document.token_count + token_increment
            ).where(Document.id == doc_id)
            updated_rows = query.execute()
            if updated_rows > 0:
                logger.debug(f"Incremented chunk/token count for doc {doc_id}.")
            else:
                logger.warning(f"Failed to increment chunk/token count for doc {doc_id} (not found or no change).")
        except Exception as e:
            logger.error(f"Error incrementing chunk/token count for doc {doc_id}: {e}", exc_info=True)

    @staticmethod
    @DB.connection_context()
    def update_document_status(doc_id: str, status: str, progress: Optional[float] = None, error_message: Optional[str] = None):
        try:
            doc = Document.get(Document.id == doc_id)
            doc.status = status
            if progress is not None:
                doc.progress = progress
            if status == "failed" and error_message:
                doc.error_message = error_message
            elif status == "completed":
                doc.progress = 1.0 # Ensure progress is 100% on completion
                doc.error_message = None # Clear any previous error
            doc.save()
            logger.info(f"Document '{doc_id}' status updated to '{status}'. Progress: {doc.progress}")
        except DoesNotExist:
            logger.error(f"Failed to update status for non-existent document ID '{doc_id}'.")
        except Exception as e:
            logger.error(f"Error updating document {doc_id} status: {e}", exc_info=True)

    @staticmethod
    @DB.connection_context()
    def update_document_chunk_info(doc_id: str, chunk_count: int, token_count: int):
        try:
            doc = Document.get(Document.id == doc_id)
            doc.chunk_count = chunk_count
            doc.token_count = token_count
            doc.save()
            logger.info(f"Document '{doc_id}' chunk info updated: Chunks={chunk_count}, Tokens={token_count}")
        except DoesNotExist:
            logger.error(f"Failed to update chunk info for non-existent document ID '{doc_id}'.")

    @staticmethod
    @DB.connection_context()
    def delete_document(doc_id: str, vector_store_deleter: Optional[callable] = None) -> bool:
        """
        Deletes a document record from DB and its associated file from local storage.
        Also attempts to delete associated tasks.
        Optionally calls a function to delete associated data from the vector store.
        """
        doc = DocumentService.get_document_by_id(doc_id)
        if not doc:
            return False

        kb_id_for_count_update = doc.kb_id_id # Store before deleting doc

        # 1. Delete from vector store (if deleter provided)
        if vector_store_deleter:
            try:
                # The vector store needs document ID, and potentially KB ID for specific index.
                # Assuming our vector store can delete by a list containing single ID.
                vector_store_deleter([doc.id]) 
                logger.info(f"Requested deletion of document {doc_id} from vector store.")
            except Exception as e:
                logger.error(f"Error requesting deletion of doc {doc_id} from vector store: {e}", exc_info=True)
                # Decide if failure to delete from vector store should halt DB deletion.
                # For now, we proceed with DB/file deletion.

        # 2. Delete associated tasks from DB
        Task.delete().where(Task.doc_id == doc.id).execute()
        logger.info(f"Associated tasks for document {doc_id} deleted from DB.")

        # 3. Delete file from local storage
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
                logger.info(f"File '{doc.file_path}' for document {doc_id} deleted from local storage.")
                # Attempt to remove the KB directory if it's empty, be careful here
                kb_dir = os.path.dirname(doc.file_path)
                if os.path.exists(kb_dir) and not os.listdir(kb_dir): # Check if empty
                    try:
                        os.rmdir(kb_dir)
                        logger.info(f"Removed empty KB directory: {kb_dir}")
                    except OSError as e_rmdir:
                        logger.warning(f"Could not remove empty KB directory {kb_dir}: {e_rmdir}")
            except OSError as e:
                logger.error(f"Error deleting file '{doc.file_path}' for document {doc_id}: {e}", exc_info=True)
                # Continue to delete DB record even if file deletion fails.

        # 4. Delete Document record from DB
        doc.delete_instance()
        logger.info(f"Document record {doc_id} deleted from DB.")
        
        # 5. Decrement KB document count
        if kb_id_for_count_update:
             KnowledgeBaseService.decrement_kb_document_count(kb_id_for_count_update)

        return True

    @staticmethod
    @DB.connection_context()
    def get_parser_config_for_doc(doc_id: str) -> Optional[dict]:
        """
        Retrieves the effective parser configuration for a document.
        This comes from its parent KnowledgeBase.
        """
        doc = DocumentService.get_document_by_id(doc_id)
        if doc and doc.kb_id: # kb_id is the ForeignKeyField instance
            kb = doc.kb_id # Access the related KnowledgeBase object
            return kb.parser_config
        return None

    @staticmethod
    @DB.connection_context()
    def get_embedding_model_for_doc(doc_id: str) -> Optional[str]:
        """Retrieves the embedding model for a document from its KB."""
        doc = DocumentService.get_document_by_id(doc_id)
        if doc and doc.kb_id:
            return doc.kb_id.embd_model
        return None