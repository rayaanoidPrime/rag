import os
import uuid # For generating default KB ID if needed
from typing import Optional, List, Dict, Any, Union, Generator, IO
from PIL.Image import Image as PILImage
from app.services.task_service import TaskService
from concurrent.futures import ThreadPoolExecutor # For background tasks
from app.config import MAX_TASK_WORKERS
from app.config import (
    PERSIST_DB_PATH, # For txtai vector store
    INITIAL_DATA_PATH,
    RAG_CONTEXT_SIZE,
    APP_TITLE,
    UI_EXAMPLES,
    LOCAL_FILE_STORAGE_PATH # New config
)
from app.document_processing import BaseDocumentParser, TxtaiDocumentParser
from app.embedding_services import E5LargeEmbedder
from app.vector_store import BaseVectorStore, TxtaiVectorStore
from app.llm_services import TxtaiLLM
from app.graph_services import GraphRAGBuilder
from app.rag_pipelines import TxtaiRAGSystem
from app.config import DB_NAME; 
from app.config import PERSIST_DB_PATH # Ensure this is imported
from app.utils.file_utils import FileType # For checking PDF if special 
from app.services.knowledge_base_service import KnowledgeBaseService
from app.services.document_service import DocumentService
from app.utils.text_processing_utils import get_metadata_suffix, extract_blurb
from app.config import RETURN_SEPARATOR
import streamlit as st

logger = st.logger.get_logger(__name__)

class AppOrchestrator:
    def __init__(self):
        # import atexit # At the top of orchestrator.py
        # atexit.register(self.shutdown_task_executor) # In __init__
        logger.info("Initializing AppOrchestrator...")
        self.doc_parser: BaseDocumentParser = TxtaiDocumentParser()
        self.embedder = E5LargeEmbedder()
        self.vector_store: BaseVectorStore = TxtaiVectorStore() # This is the txtai instance
        self.llm_service = TxtaiLLM()

        # Initialize DB-dependent services
        self.kb_service = KnowledgeBaseService()
        self.doc_service = DocumentService()
        self.task_service = TaskService() 

        self._load_or_create_vector_store() # For txtai embeddings

        # Ensure a default KB exists for simplicity in single-user mode
        self.default_kb_id = self._ensure_default_kb()
        logger.info(f"Using default Knowledge Base ID: {self.default_kb_id}")

         # --- ThreadPoolExecutor for background task processing ---
        self.task_executor = ThreadPoolExecutor(max_workers=MAX_TASK_WORKERS, thread_name_prefix="DocProcWorker")
        logger.info(f"Task executor initialized with max_workers={MAX_TASK_WORKERS}")


       # ... (GraphRAGBuilder and RAGSystem initialization) ...
        underlying_txtai_embeddings = self.vector_store.get_underlying_embeddings_instance()
        if underlying_txtai_embeddings:
            self.graph_builder = GraphRAGBuilder(underlying_txtai_embeddings, self.llm_service)
        else:
            self.graph_builder = None
            logger.warning("GraphRAGBuilder could not be initialized.")
        try:
            self.rag_system = TxtaiRAGSystem(self.vector_store, self.llm_service)
        except ValueError as e:
            logger.error(f"Failed to initialize RAG system: {e}.")
            self.rag_system = None

        self._initial_data_ingestion_v2() # Call the revised initial data ingestion

        logger.info("AppOrchestrator initialized.")


     def _ensure_default_kb(self) -> str:
        default_kb_name = "Default Knowledge Base"
        kb = self.kb_service.get_kb_by_name(default_kb_name)
        if not kb:
            from app.config import EMBEDDING_MODEL_PATH
            kb = self.kb_service.create_kb(name=default_kb_name, embd_model=EMBEDDING_MODEL_PATH)
        return kb.id
        
    def _load_or_create_vector_store(self):
        logger.info("Attempting to load TxtaiVectorStore (for embeddings and graph)...")
        is_hf_hub_id = "/" in self.vector_store.embeddings.config.get("path", "") and \
                       not os.path.exists(self.vector_store.embeddings.config.get("path", ""))
        
        primary_load_path = PERSIST_DB_PATH 
        fallback_load_path = self.vector_store.embeddings.config.get("path", "data/embeddings_db")

        loaded = False
        if primary_load_path and os.path.exists(primary_load_path) and self.vector_store.embeddings.exists(primary_load_path):
            logger.info(f"Loading TxtaiVectorStore from persisted path: {primary_load_path}")
            self.vector_store.load(path=primary_load_path)
            loaded = self.vector_store.is_loaded()
        
        if not loaded and not is_hf_hub_id and fallback_load_path and os.path.exists(fallback_load_path) and \
             self.vector_store.embeddings.exists(fallback_load_path):
            logger.info(f"Loading TxtaiVectorStore from configured EMBEDDINGS path: {fallback_load_path}")
            self.vector_store.load(path=fallback_load_path)
            loaded = self.vector_store.is_loaded()
        
        if not loaded and is_hf_hub_id:
            hf_path = self.vector_store.embeddings.config.get("path")
            logger.info(f"Attempting to load TxtaiVectorStore from Hugging Face Hub: {hf_path}")
            try:
                self.vector_store.load(path=hf_path) 
                if not self.vector_store.is_loaded():
                     self.vector_store.load(cloud_config={"provider": "huggingface-hub", "container": hf_path})
                loaded = self.vector_store.is_loaded()
            except Exception as e:
                logger.error(f"Failed to load from Hugging Face Hub {hf_path}: {e}")
        
        if not loaded:
            logger.info("No existing TxtaiVectorStore found or load failed. Initializing new one.")
            self.vector_store._initialize_embeddings()

    def _initial_data_ingestion_v2(self):
        """Revised initial data ingestion using the new DB-backed flow."""
        if INITIAL_DATA_PATH and os.path.isdir(INITIAL_DATA_PATH):
            logger.info(f"Scanning initial data path: {INITIAL_DATA_PATH} for default KB: {self.default_kb_id}")
            default_kb = self.kb_service.get_kb_by_id(self.default_kb_id)
            if not default_kb:
                logger.error("Default KB not found for initial data ingestion. Skipping.")
                return

            for item_name in os.listdir(INITIAL_DATA_PATH):
                item_path = os.path.join(INITIAL_DATA_PATH, item_name)
                if os.path.isfile(item_path):
                    logger.info(f"Found initial file: {item_path}")
                    # Check if document already exists by name in this KB to avoid duplicates
                    existing_doc = self.doc_service.get_document_by_kb_id_and_name(default_kb.id, item_name)
                    if existing_doc:
                        logger.info(f"Document '{item_name}' already exists in KB '{default_kb.name}'. Status: {existing_doc.status}. Skipping initial ingestion if already processed/queued.")
                        # Optionally, re-queue if status is 'failed' or 'pending' and old.
                        if existing_doc.status not in ["completed", "processing", "queued"]:
                             logger.info(f"Re-queueing existing document '{item_name}' as its status is '{existing_doc.status}'.")
                             # This would involve re-creating tasks for it. For now, just log.
                             # Potentially: self._trigger_processing_for_document(existing_doc)
                        continue 

                    try:
                        with open(item_path, "rb") as f_content:
                            self.add_document_and_trigger_processing(
                                kb_id=self.default_kb_id,
                                original_filename=item_name,
                                file_content_stream=f_content
                            )
                    except Exception as e:
                        logger.error(f"Failed to process initial file {item_name}: {e}", exc_info=True)
        else:
            logger.info("No initial data path configured or path is invalid for initial ingestion.")



    def add_document_and_trigger_processing(self,
                                       kb_id: str,
                                       original_filename: str,
                                       file_content_stream: IO[bytes],
                                       source_type: str = "local"
                                       ) -> Dict[str, Any]:
        response = {"success": False, "message": "", "doc_id": None}
        kb = self.kb_service.get_kb_by_id(kb_id)
        if not kb:
            response["message"] = f"Knowledge Base ID '{kb_id}' not found."
            return response

        doc_instance, error = self.doc_service.add_document_to_kb(
            kb=kb, original_filename=original_filename,
            file_content_stream=file_content_stream, source_type=source_type
        )

        if error or not doc_instance:
            response["message"] = error or "Failed to add document to database."
            return response

        response["doc_id"] = doc_instance.id
        
        # --- Trigger asynchronous processing ---
        try:
            self._trigger_processing_for_document(doc_instance)
            response["message"] = (f"Document '{original_filename}' (ID: {doc_instance.id}) "
                                   f"added to KB '{kb.name}'. Processing started in background.")
            response["success"] = True
        except Exception as e:
            logger.error(f"Failed to trigger processing for doc {doc_instance.id}: {e}", exc_info=True)
            self.doc_service.update_document_status(doc_instance.id, status="failed", error_message=f"Failed to start processing: {e}")
            response["message"] = f"Document added (ID: {doc_instance.id}), but failed to start background processing: {e}"
            # success might still be true if doc was added, but processing couldn't start
            response["success"] = False # Or keep true and rely on detailed message

        return response

    
    def _trigger_processing_for_document(self, doc: Document):
        """
        Creates tasks for the document and submits them to the executor.
        """
        logger.info(f"Creating and queueing tasks for document: {doc.id} ('{doc.name}')")
        # 1. Create task definitions in DB
        # This also updates doc status to "queued"
        tasks_to_process = self.task_service.create_processing_tasks_for_document(doc)

        if not tasks_to_process:
            logger.warning(f"No tasks created for document {doc.id}. Nothing to process.")
            self.doc_service.update_document_status(doc.id, status="failed", error_message="No processing tasks could be generated.")
            return

        logger.info(f"Submitting {len(tasks_to_process)} tasks for document {doc.id} to executor.")
        # Update document status to "processing" before submitting first task
        self.doc_service.update_document_status(doc.id, status="processing", progress=0.0)

        for task_db_obj in tasks_to_process:
            # Submit each DB task object to the actual processing function
            # The `_execute_single_task_async` will be defined in Step 4
            self.task_executor.submit(self._execute_single_task_async, task_db_obj.id)
        
        logger.info(f"All tasks for document {doc.id} submitted to executor.")
        # Note: Overall document completion will be tracked as tasks complete.

    def _execute_single_task_async(self, task_id: str):
        """
        The actual function that gets executed by the ThreadPoolExecutor.
        It fetches the task, performs parsing, embedding, indexing for that task's scope.
        """
        logger.info(f"[TASK_RUNNER:{task_id}] Starting execution.")
        self.task_service.update_task_status(task_id, status="processing", progress=0.01)

        task_db_obj = self.task_service.get_task_by_id(task_id)
        if not task_db_obj:
            logger.error(f"[TASK_RUNNER:{task_id}] Task not found in DB. Aborting.")
            # No status update needed as task doesn't exist to update
            return

        doc = self.doc_service.get_document_by_id(task_db_obj.doc_id_id) # doc_id_id is the actual ID value
        if not doc:
            logger.error(f"[TASK_RUNNER:{task_id}] Document {task_db_obj.doc_id_id} not found for task. Aborting.")
            self.task_service.update_task_status(task_id, status="failed", error_message="Associated document not found.")
            # Check and finalize to potentially mark the document as failed if this was its only task
            self._check_and_finalize_document_processing(task_db_obj.doc_id_id)
            return

        kb = doc.kb_id # Access the related KnowledgeBase peewee object
        if not kb:
            logger.error(f"[TASK_RUNNER:{task_id}] KnowledgeBase not found for document {doc.id}. Aborting.")
            self.task_service.update_task_status(task_id, status="failed", error_message="Associated KnowledgeBase not found.")
            self.doc_service.update_document_status(doc.id, status="failed", error_message="KnowledgeBase not found.")
            self._check_and_finalize_document_processing(doc.id)
            return

        doc_file_path = doc.file_path
        doc_title = doc.name
        parser_config_from_kb = kb.parser_config # This includes 'pages', 'layout_recognize' etc.
        # embed_model_name_from_kb = kb.embd_model # We use one embedder for now.
        # --- Prepare document-level metadata for chunks ---
        doc_metadata_for_chunks = {
            "doc_id": doc.id,
            "doc_name": doc.name,
            "kb_id": kb.id,
            "source_file_path": doc.file_path,
            "doc_title": doc_title # Adding title explicitly
            # Add other relevant fields from 'doc' model if needed
        }


        chunks_for_vector_store = []
        total_tokens_for_task = 0
        processed_successfully = False

        try:
            logger.info(f"[TASK_RUNNER:{task_id}] Processing document: '{doc.name}' (ID: {doc.id}), file: {doc_file_path}")
            logger.info(f"[TASK_RUNNER:{task_id}] Task scope: Pages {task_db_obj.from_page} to {task_db_obj.to_page}")
            logger.info(f"[TASK_RUNNER:{task_id}] Parser config from KB: {parser_config_from_kb}")

            # --- 1. Parse Document Content for Task Scope ---
            # self.doc_parser is currently TxtaiDocumentParser.
            # TxtaiDocumentParser's parse method takes a file path or URL.
            # It doesn't directly support page ranges from its interface.
            # DeepDoc will be better here.
            # For now, if it's a PDF and task has page ranges, we need a workaround or accept it processes the whole file.
            
            # TODO: Enhance self.doc_parser interface or implementations to accept task_scope (pages, etc.)
            # For TxtaiDocumentParser, it currently processes the whole file.
            # If DeepDoc is used, it would handle page ranges internally or via its API.
            
            # Simple simulation for now:
            # If it's a PDF and from_page != 0 or to_page != -1, it's a sub-task.
            # Our current TxtaiDocumentParser will re-parse the whole doc for each such task.
            # This is inefficient but a limitation of not having a page-aware parser interface yet.
            
            parsed_content_generator = self.doc_parser.parse(doc_file_path)
            # TxtaiDocumentParser yields dicts like {"text": "...", "source": "filepath"}

            # We need to simulate page filtering if task_db_obj specifies pages
            # This is a very rough simulation for PDF. A real solution needs parser support.
            current_chunk_index_in_doc = 0 # To generate unique chunk IDs for this document
            # Max chunks per task to avoid overwhelming memory or a single task taking too long.
            MAX_CHUNKS_PER_TASK_SEGMENT = 500 # Configurable

            for i, content_dict in enumerate(parsed_content_generator):
                # Rough page filtering (INEFFICIENT - re-parses all for each task)
                # This needs to be handled by the parser itself ideally.
                # If task is for specific pages of a PDF, and current parser doesn't support it,
                # this loop processes all chunks from the doc. We'll add all and let txtai index them.
                # The task definition (from_page, to_page) is more for future page-aware parsers.
                
                # For now, we process all chunks yielded by the parser for the given file.
                # If this task is one of many for a PDF (e.g. task 1 for p0-9, task 2 for p10-19),
                # this code will add *all* chunks from the PDF to the vector store *for each task*.
                # This is a major duplication and needs to be fixed when DeepDoc or a page-aware
                # parser is integrated.
                #
                # Workaround idea for TxtaiDocumentParser if tasks represent full doc:
                # Only process if task.from_page == 0 and task.to_page == -1
                # if not (doc.file_type == FileType.PDF.value and (task_db_obj.from_page != 0 or task_db_obj.to_page != -1)):
                #    # This is not a sub-page task for a PDF, or not a PDF, so process all content from parser
                #    pass # continue processing this content_dict
                # else:
                #    # This IS a sub-page task for a PDF. Our current parser CANNOT handle this.
                #    # So we skip adding chunks to avoid duplication if multiple tasks exist for this PDF.
                #    # The *first* task (e.g. pages 0 to -1, or 0 to N) would process it.
                #    # This is still a hack.
                #    logger.warning(f"[TASK_RUNNER:{task_id}] PDF sub-page task. Current parser processes whole file. "
                #                   f"Skipping chunk addition in this task to avoid duplication if other tasks cover it.")
                #    # To make this workaround function, we'd need to ensure only ONE task for a PDF
                #    # actually processes content if the parser isn't page-aware.
                #    # For now, let's assume the task creation logic for PDF with TxtaiParser
                #    # only creates ONE task (from_page=0, to_page=-1).
                #    # If TaskService.create_processing_tasks_for_document for PDF and TxtaiParser
                #    # creates multiple tasks, then we have a duplication problem here.
                #
                # Let's assume `TaskService.create_processing_tasks_for_document` is smart enough for now
                # for the current `TxtaiDocumentParser` to only create one task if the parser isn't page-aware.
                # (Review TaskService: it does create one task if not PDF with page ranges/size in config)

                text_content = content_dict.get("text")
                if not text_content or not text_content.strip():
                    continue

                # Estimate tokens (simple split, not accurate as model tokenizer)
                # TODO: Use self.embedder.tokenizer if available for better token count
                total_tokens_for_task += len(text_content.split())
                
                # --- Chunk ID generation ---
                # We need a unique ID for each chunk within the txtai vector store.
                # Format: <doc_id>_chunk_<index_within_doc>
                # This requires knowing the global chunk index for this document across all its tasks.
                # This is hard with parallel tasks.
                # Alternative: <task_id>_chunk_<index_within_task> - simpler, but less globally unique for doc.
                # Best: <doc_id>_chunk_<sequential_number_for_doc>
                # For now, use UUID for chunk ID for simplicity with txtai's autoid="uuid5"
                # TxtaiVectorStore's add_documents will handle ID generation if `doc.get("id")` is None.
                # To link back, we need to store the `doc_id` as metadata in the chunk.
                
                # Onyx: `DocAwareChunk` with `chunk_id` (sequential within doc).
                # We can simulate this by querying current chunk_count for the doc before adding.
                # But this is prone to race conditions.
                # Safest for now: let txtai generate chunk ID, store doc_id and kb_id as metadata.

                chunk_data_for_txtai = {
                    "id": None, # Let TxtaiVectorStore generate with autoid="uuid5"
                    "text": text_content,
                    # --- METADATA ---
                    # Crucial for filtering, context, and linking back
                    "doc_id": doc.id,
                    "doc_name": doc.name,
                    "kb_id": kb.id,
                    "source_file_path": doc_file_path, # Original source of the document
                    # "chunk_source_info": f"Task_{task_id}_Index_{i}" # For debugging linkage
                    # Add more metadata from Onyx ideas later: title, summaries, etc.
                }
                chunks_for_vector_store.append(chunk_data_for_txtai)
                
                if len(chunks_for_vector_store) % 50 == 0: # Log progress
                    logger.debug(f"[TASK_RUNNER:{task_id}] Parsed {len(chunks_for_vector_store)} chunks so far...")
                
                if len(chunks_for_vector_store) >= MAX_CHUNKS_PER_TASK_SEGMENT:
                    logger.warning(f"[TASK_RUNNER:{task_id}] Reached max chunks ({MAX_CHUNKS_PER_TASK_SEGMENT}) for this task segment. Processing what we have.")
                    break # Stop parsing more for this specific task to prevent memory issues

            self.task_service.update_task_status(task_id, status="processing", progress=0.3)
            logger.info(f"[TASK_RUNNER:{task_id}] Parsed {len(chunks_for_vector_store)} chunks from '{doc.name}'.")

            # --- 2. Embed Chunks (if any) ---
            # TxtaiVectorStore handles embedding internally if embeddings are not provided to add_documents.
            # So, we don't explicitly call self.embedder here if TxtaiVectorStore is used.
            # If we were to use a vector store that requires pre-computed embeddings:
            #   if chunks_for_vector_store:
            #       texts_to_embed = [c["text"] for c in chunks_for_vector_store]
            #       embeddings = self.embedder.get_embeddings(texts_to_embed)
            #       for i, chunk in enumerate(chunks_for_vector_store):
            #           chunk["embedding"] = embeddings[i] # Add precomputed embedding
            #       self.task_service.update_task_status(task_id, status="processing", progress=0.6)

            # --- 3. Add Chunks to Vector Store (txtai) ---
            if chunks_for_vector_store:
                logger.info(f"[TASK_RUNNER:{task_id}] Adding {len(chunks_for_vector_store)} chunks to TxtaiVectorStore.")
                # TxtaiVectorStore.add_documents will use its configured embedding model.
                # The "id" in chunk_data_for_txtai is None, so txtai will autogenerate one.
                self.vector_store.add_documents(chunks_for_vector_store)
                logger.info(f"[TASK_RUNNER:{task_id}] Successfully added chunks to vector store.")
            else:
                logger.info(f"[TASK_RUNNER:{task_id}] No chunks to add to vector store for this task.")

            self.task_service.update_task_status(task_id, status="processing", progress=0.9)
            processed_successfully = True

        except Exception as e:
            error_msg = f"Error during task execution for task {task_id} (Doc: {doc.id}): {e}"
            logger.error(f"[TASK_RUNNER:{task_id}] {error_msg}", exc_info=True)
            self.task_service.update_task_status(task_id, status="failed", error_message=str(e))
            # No need to update document status here, _check_and_finalize will handle it

        finally:
            # --- 4. Update Task and Document Status ---
            if processed_successfully:
                self.task_service.update_task_status(task_id, status="completed", progress=1.0)
                # Increment document's overall chunk/token count
                self.doc_service.increment_document_chunk_and_token_count(
                    doc.id,
                    chunk_increment=len(chunks_for_vector_store),
                    token_increment=total_tokens_for_task # This is an estimate
                )
                logger.info(f"[TASK_RUNNER:{task_id}] Task completed successfully.")
            
            # This will check if all tasks for the doc are done and update doc status,
            # potentially triggering GraphRAG topic inference and vector store persistence.
            self._check_and_finalize_document_processing(doc.id)
            logger.info(f"[TASK_RUNNER:{task_id}] Finished execution.")


    def _check_and_finalize_document_processing(self, doc_id: str):
        """
        Checks if all tasks for a document are completed.
        If so, updates the overall document status to 'completed'.
        """
        pending_or_processing_tasks = self.task_service.get_tasks_for_document_by_status_in(
            doc_id, ["pending", "processing", "queued"]
        )
        
        doc = self.doc_service.get_document_by_id(doc_id)
        if not doc: return

        if not pending_or_processing_tasks:
            # Check for failed tasks
            failed_tasks = self.task_service.get_tasks_for_document(doc_id, status="failed")
            if failed_tasks:
                logger.warning(f"Document {doc_id} has {len(failed_tasks)} failed tasks. Marking document as failed.")
                # Consolidate error messages or pick first one
                error_msg = "; ".join([ft.error_message for ft in failed_tasks if ft.error_message])
                self.doc_service.update_document_status(doc_id, status="failed", progress=1.0, error_message=error_msg or "One or more processing tasks failed.")
            else:
                l logger.info(f"All tasks for document {doc.id} completed. Finalizing document.")
                self.doc_service.update_document_status(doc.id, status="completed", progress=1.0)
                kb = doc.kb_id
                # --- Trigger GraphRAG topic inference for the completed document ---
                # This should happen after all chunks are in the vector store (and thus in txtai graph)
                
                  # --- Onyx-inspired Post-Processing ---
                # 1. Generate Document Summary (if enabled)
                if kb and kb.parser_config.get("generate_document_summary", False):
                    logger.info(f"Generating document summary for {doc.id}...")
                    try:
                        # To generate a good summary, we need representative text from the document.
                        # Option A: Retrieve top N chunks from vector store for this doc_id.
                        # Option B: If raw text was stored, fetch it (might be too large).
                        # Option C: If DocumentService stored full text during upload (new field on Document model).
                        # For now, let's try retrieving some chunks.
                        
                        # This assumes TxtaiVectorStore search can filter by doc_id metadata.
                        # We need to ensure `doc_id` is indexed and searchable in txtai.
                        # Txtai search on dict fields: `embeddings.search("select id, text, score, data where data->>'doc_id' = :doc_id", parameters={"doc_id": doc_id})`
                        # This syntax is for SQLite backend. For others, it might vary.
                        # Our TxtaiVectorStore.search needs to support metadata filtering.
                        # For now, this is a placeholder for that capability.
                        
                        # Placeholder: Get first few stored chunks. This is not ideal for summary.
                        # A better way: concatenate text from N random or first N chunks.
                        # For now, let's assume we'll add a `full_text_for_summary` field to Document or fetch it.
                        # This step is complex to do efficiently without full text.
                        
                        # For DEMO, let's assume a simplified summary based on doc name.
                        # doc_summary_text = self.llm_service.generate(f"Provide a brief topic for a document named '{doc.name}'.")
                        # Store this summary on the Document model (e.g., `doc.summary_text = doc_summary_text; doc.save()`)
                        # And then, potentially update all chunks of this doc in txtai to include this doc_summary in their metadata. (Expensive!)
                        # Onyx approach: doc_summary is part of DocAwareChunk at creation time.
                        # This implies doc_summary must be generated *before or during* chunking.
                        # This means `add_document_summaries` in Onyx is called *before* embedding.
                        # This is a significant reordering if we adopt it fully.
                        # Let's table full doc summary injection into chunks for now due to complexity.
                        # The `GraphRAGBuilder.infer_topics_for_all_nodes` is our current proxy for doc-level understanding.
                        logger.info(f"Placeholder for document summary generation for {doc.id}.")

                    except Exception as e:
                        logger.error(f"Failed to generate document summary for {doc.id}: {e}")

                # 2. Generate Large Chunks (if enabled)
                if kb and kb.parser_config.get("enable_large_chunks", False):
                    logger.info(f"Generating large chunks for {doc.id}...")
                    try:
                        # This also requires retrieving all "normal" chunks for this document from Txtai.
                        # Then, use a function like Onyx's `generate_large_chunks`.
                        # And then index these new large chunks into Txtai.
                        # This is also a significant operation.
                        logger.info(f"Placeholder for large chunk generation for {doc.id}.")
                    except Exception as e:
                        logger.error(f"Failed to generate large chunks for {doc.id}: {e}")

                # --- Original finalization steps ---
                
                if self.graph_builder:
                    logger.info(f"Running GraphRAG topic inference for completed document {doc_id}...")
                    # infer_topics_for_all_nodes is too broad.
                    # We need a way to tell it to process nodes related to this specific doc_id.
                    # For now, we call the general one, but this needs refinement for efficiency.
                    # A better way: GraphRAGBuilder.infer_topics_for_document_nodes(doc_id, self.vector_store)
                    self.graph_builder.infer_topics_for_all_nodes() # Inefficient, but for now
                    logger.info(f"GraphRAG topic inference triggered for document {doc_id}.")

                # --- Persist TxtaiVectorStore if changes were made ---
                if PERSIST_DB_PATH and self.vector_store:
                    logger.info(f"Persisting TxtaiVectorStore after document {doc_id} processing.")
                    self.vector_store.save(PERSIST_DB_PATH)
        else:
            # Update overall document progress based on completed tasks
            all_tasks_for_doc = self.task_service.get_tasks_for_document(doc_id)
            if all_tasks_for_doc:
                completed_progress_sum = sum(t.progress for t in all_tasks_for_doc if t.status == "completed")
                current_processing_progress = sum(t.progress for t in all_tasks_for_doc if t.status == "processing") * 0.5 # Assume processing tasks are 50% of their reported progress
                total_tasks = len(all_tasks_for_doc)
                overall_doc_progress = (completed_progress_sum + current_processing_progress) / total_tasks
                self.doc_service.update_document_status(doc.id, status="processing", progress=min(overall_doc_progress, 0.99))


    def shutdown_task_executor(self):
        """Shuts down the task executor. Call on application exit."""
        if self.task_executor:
            logger.info("Shutting down task executor... (waiting for tasks to complete)")
            self.task_executor.shutdown(wait=True) # Wait for all current tasks to finish
            logger.info("Task executor shut down.")
              
    def process_query(self, user_query: str, kb_id_filter: Optional[str] = None, stream_response: bool = False) -> Dict[str, Any]:
        """
        Processes a user query.
        kb_id_filter: Optional KB ID to restrict search/context. Not yet fully implemented in txtai search.
        """
        if not self.rag_system:
            return {"answer": "RAG system is not available.", "type": "error", "graph_image": None}

        # TODO: If kb_id_filter is provided, how do we pass this to txtai.RAG or txtai.Embeddings.search?
        # txtai search can take a `where` clause if using SQL backend, or custom filtering logic might be needed.
        # For GraphRAG, concept searches might also need to be scoped to a KB.
        # This is a complex part when moving from a single global index to KB-specific contexts
        # while still using a single txtai.Embeddings instance.
        # One way is to add a `kb_id` tag/metadata to each item in txtai and filter during search result post-processing
        # or if txtai allows metadata filtering directly in .search().
        
        # For now, the query is global across all data in the txtai vector store.
        # The GraphRAG is also global.
        
        graph_image = None
        rag_context_override = None
        final_query_for_llm = user_query
        response_type = "rag" 

        if self.graph_builder:
            modified_query, graph_rag_data, plot_img = self.graph_builder.get_graph_rag_context(user_query)
            if graph_rag_data:
                rag_context_override = [item["text"] for item in graph_rag_data]
                final_query_for_llm = modified_query
                graph_image = plot_img
                response_type = "graph_rag"
                logger.debug(f"GraphRAG context ({len(rag_context_override)} items) for query: '{final_query_for_llm}'")
            else:
                # Standard RAG context logging (example)
                # search_results = self.vector_store.search(final_query_for_llm, k=RAG_CONTEXT_SIZE)
                # logger.debug(f"Standard RAG context for query: '{final_query_for_llm}' will be fetched by RAG pipeline.")
                pass # RAG pipeline will fetch its own context
        
        answer_content = self.rag_system.answer(
            question=final_query_for_llm,
            context_override=rag_context_override,
            stream=stream_response
        )
        return {"answer": answer_content, "type": response_type, "graph_image": graph_image}


     def get_app_info(self) -> Dict[str, Any]:
        kbs_from_db = self.kb_service.list_kbs()
        kb_list_for_ui = [{"id": kb.id, "name": kb.name, "doc_count": kb.document_count} for kb in kbs_from_db]

        # Optionally, count active tasks
        # active_tasks = Task.select().where(Task.status << ["pending", "processing", "queued"]).count()

        return {
            "title": APP_TITLE,
            "vector_store_total_count": self.vector_store.count(),
            "knowledge_bases": kb_list_for_ui,
            "default_kb_id": self.default_kb_id,
            # "active_processing_tasks": active_tasks, # Example
            "examples": UI_EXAMPLES,
            "supports_graph_rag": self.graph_builder is not None,
            "config_summary": {
                "LLM Model": self.llm_service.model_path if self.llm_service else "N/A",
                "Embedding Model": self.embedder.embed_model.config.get("path") if self.embedder and self.embedder.embed_model else "N/A",
                "Vector DB (txtai) Path": self.vector_store.embeddings.config.get("path", "N/A") if self.vector_store and self.vector_store.embeddings else "N/A",
                "Vector DB (txtai) Persist Path": PERSIST_DB_PATH if PERSIST_DB_PATH else "Not configured",
                "PostgreSQL DB Name": from app.config import DB_NAME; DB_NAME,
                "Local File Storage": LOCAL_FILE_STORAGE_PATH,
                "Max Task Workers": MAX_TASK_WORKERS,
            }
        }

    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        kbs = self.kb_service.list_kbs()
        return [{"id": kb.id, "name": kb.name, "description": kb.description, "doc_count": kb.document_count, "parser_id": kb.parser_id} for kb in kbs]

    def create_knowledge_base(self, name: str, description: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        # kwargs can pass other KB params like embd_model, parser_id, etc.
        try:
            kb = self.kb_service.create_kb(name=name, description=description, **kwargs)
            return {"success": True, "id": kb.id, "name": kb.name, "message": "Knowledge Base created."}
        except Exception as e: # Catch potential peewee.IntegrityError for duplicate name if not handled by service
            logger.error(f"Error creating KB '{name}': {e}", exc_info=True)
            return {"success": False, "message": str(e)}

    def get_documents_for_kb(self, kb_id: str) -> List[Dict[str, Any]]:
        docs = self.doc_service.get_documents_by_kb_id(kb_id)
        return [
            {
                "id": doc.id, "name": doc.name, "status": doc.status,
                "size": doc.file_size, "type": doc.file_type,
                "progress": doc.progress, "chunk_count": doc.chunk_count,
                "create_time": doc.create_time # Send as ms timestamp
            } for doc in docs
        ]

    def delete_kb_orchestrated(self, kb_id: str) -> bool:
        """ Orchestrates KB deletion: DB records and vector store entries """
        kb = self.kb_service.get_kb_by_id(kb_id)
        if not kb:
            logger.warning(f"Attempted to delete non-existent KB ID: {kb_id}")
            return False

        doc_ids_to_delete_from_vector_store = [doc.id for doc in kb.documents]

        # 1. Delete from vector store (txtai)
        if doc_ids_to_delete_from_vector_store and self.vector_store:
            try:
                success = self.vector_store.delete_documents(doc_ids_to_delete_from_vector_store)
                if success:
                    logger.info(f"Successfully deleted {len(doc_ids_to_delete_from_vector_store)} documents from vector store for KB {kb_id}.")
                else:
                    logger.warning(f"Vector store reported no/partial success deleting documents for KB {kb_id}.")
            except Exception as e:
                logger.error(f"Error deleting documents from vector store for KB {kb_id}: {e}", exc_info=True)
                # Decide if this error should prevent DB deletion. For now, it does not.

        # 2. Delete KB and associated document/task records from PostgreSQL
        # The DocumentService.delete_document also tries to delete files.
        # KnowledgeBaseService.delete_kb will cascade delete documents,
        # and Document.delete_instance (called by cascade) should ideally trigger file cleanup too.
        # However, to be explicit and ensure files associated with *this* KB are handled:
        for doc_id in doc_ids_to_delete_from_vector_store: # Iterate again, but doc objects are gone after kb.delete_instance
            doc_to_delete = self.doc_service.get_document_by_id(doc_id)
            if doc_to_delete and doc_to_delete.file_path and os.path.exists(doc_to_delete.file_path):
                try:
                    os.remove(doc_to_delete.file_path)
                except OSError as e:
                    logger.error(f"Error explicitly deleting file {doc_to_delete.file_path} for doc {doc_id} during KB deletion: {e}")
        
        # Delete KB from DB (cascades to Documents and Tasks in DB)
        db_delete_success = self.kb_service.delete_kb(kb_id)
        
        # Attempt to remove the KB's main storage directory if it's empty
        kb_storage_dir = self.doc_service._get_kb_storage_path(kb_id)
        if db_delete_success and os.path.exists(kb_storage_dir) and not os.listdir(kb_storage_dir):
            try:
                os.rmdir(kb_storage_dir)
                logger.info(f"Removed empty main KB storage directory: {kb_storage_dir}")
            except OSError as e:
                logger.warning(f"Could not remove main KB storage directory {kb_storage_dir} (it might not be empty or permission issue): {e}")
        
        # Persist changes to the txtai vector store if it was modified
        if doc_ids_to_delete_from_vector_store and PERSIST_DB_PATH and self.vector_store:
            self.vector_store.save(PERSIST_DB_PATH)

        return db_delete_success

    def delete_document_orchestrated(self, doc_id: str) -> bool:
        """ Orchestrates Document deletion: DB record, file, and vector store entries """
        # vector_store_deleter callback for DocumentService.delete_document
        def _vs_deleter(doc_ids_list):
            if self.vector_store:
                return self.vector_store.delete_documents(doc_ids_list)
            return False
        
        deleted = self.doc_service.delete_document(doc_id, vector_store_deleter=_vs_deleter)
        
        if deleted and PERSIST_DB_PATH and self.vector_store:
            self.vector_store.save(PERSIST_DB_PATH) # Persist txtai changes
        return deleted