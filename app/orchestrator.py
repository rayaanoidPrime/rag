import os
import json
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
import importlib # For dynamic imports
from app.config import PARSER_MAPPING, DEEPDOC_PARSER_BY_FILE_TYPE # Import new configs
from app.utils.file_utils import FileType # Make sure this is your app's FileType
# Import deepdoc parser classes for type hinting and direct use if needed
from app.deepdoc_components.parser import (
    RAGFlowPdfParser, RAGFlowDocxParser, RAGFlowExcelParser, RAGFlowPptParser,
    RAGFlowTxtParser, RAGFlowMarkdownParser, RAGFlowHtmlParser, RAGFlowJsonParser
    # Add other specific deepdoc parsers if you have them
)
from app.document_processing.base_parser import BaseDocumentParser # For 

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

    def _get_parser_instance(self, kb: KnowledgeBase, doc_file_type_str: str) -> Optional[BaseDocumentParser]:
        """
        Gets an instance of the appropriate parser based on KB settings and doc type.
        """
        parser_id_to_use = kb.parser_id or DEFAULT_PARSER_ID
        
        if parser_id_to_use == "deepdoc_auto":
            # Dispatch to specific deepdoc parser based on file type
            parser_key = DEEPDOC_PARSER_BY_FILE_TYPE.get(doc_file_type_str.lower())
            if not parser_key:
                logger.warning(f"DeepDoc Auto: No specific deepdoc parser for file type '{doc_file_type_str}'. Falling back to naive.")
                parser_id_to_use = "naive_txtai" # Fallback
            else:
                parser_id_to_use = parser_key
        
        parser_info = PARSER_MAPPING.get(parser_id_to_use)

        if not parser_info:
            logger.error(f"No parser mapping found for parser_id: {parser_id_to_use}")
            return None

        class_path_str = parser_info.get("class_path")
        if not class_path_str:
            logger.error(f"Class path not defined for parser_id: {parser_id_to_use}")
            return None

        try:
            module_path, class_name = class_path_str.rsplit('.', 1)
            module = importlib.import_module(module_path)
            parser_class = getattr(module, class_name)
            
            # Instantiate parser
            # Deepdoc parsers might take specific init args from kb.parser_config
            # For now, most deepdoc parsers are simple __init__ or take chunk_token_num
            parser_init_args = {}
            if parser_info.get("type") == "deepdoc":
                if parser_class == RAGFlowTxtParser: # Example: RAGFlowTxtParser takes chunk_token_num
                    txt_options = kb.parser_config.get("txt_parser_options", {})
                    chunk_token_num = txt_options.get("chunk_token_num", DEFAULT_CHUNK_TOKEN_NUM)
                    parser_init_args["chunk_token_num"] = chunk_token_num
                elif parser_class == RAGFlowJsonParser:
                    json_options = kb.parser_config.get("json_parser_options", {})
                    max_chunk_size = json_options.get("max_chunk_size", 2000) # Default from JsonParser
                    min_chunk_size = json_options.get("min_chunk_size")
                    parser_init_args["max_chunk_size"] = max_chunk_size
                    if min_chunk_size is not None:
                         parser_init_args["min_chunk_size"] = min_chunk_size
                # Add more specific initializations if other deepdoc parsers need them

            instance = parser_class(**parser_init_args)
            logger.info(f"Successfully instantiated parser: {class_name} for parser_id: {parser_id_to_use}")
            return instance
        except (ImportError, AttributeError, Exception) as e:
            logger.error(f"Failed to instantiate parser {class_path_str}: {e}", exc_info=True)
            return None
    
    def _store_artifact_image(self, kb_id: str, doc_id: str, artifact_pil_image: PILImage, artifact_type: str, original_page_num: int) -> Optional[str]:
        """Saves an artifact image (PIL.Image) to a dedicated path and returns the path."""
        try:
            # Create a subdirectory for artifacts if it doesn't exist
            # e.g., LOCAL_FILE_STORAGE_PATH / kb_id / doc_id_artifacts /
            doc_artifact_dir = os.path.join(LOCAL_FILE_STORAGE_PATH, kb_id, f"{doc_id}_artifacts")
            os.makedirs(doc_artifact_dir, exist_ok=True)

            artifact_filename = f"{artifact_type}_{original_page_num}_{uuid.uuid4().hex}.png"
            artifact_path = os.path.join(doc_artifact_dir, artifact_filename)
            
            artifact_pil_image.save(artifact_path, "PNG")
            logger.info(f"Stored artifact image to: {artifact_path}")
            return artifact_path # Return relative or absolute path as needed for DB
        except Exception as e:
            logger.error(f"Failed to store artifact image for doc {doc_id}: {e}", exc_info=True)
            return None


    def _create_chunk_dict(self, text_content: str, doc: Document, kb: KnowledgeBase, metadata_extras: Optional[Dict] = None) -> Dict[str, Any]:
        """ Helper to create the dictionary structure for a chunk. """
        chunk_data = {
            "id": None, # Let TxtaiVectorStore generate with autoid
            "text": text_content,
            "doc_id": doc.id,
            "doc_name": doc.name,
            "kb_id": kb.id,
            "source_file_path": doc.file_path, # Stored path, not original filename
            "doc_title": doc.name # Simple title for now
        }
        if metadata_extras:
            chunk_data.update(metadata_extras)
        return chunk_data


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
        logger.info(f"[TASK_RUNNER:{task_id}] Starting execution.")
        self.task_service.update_task_status(task_id, status="processing", progress=0.01)

        task_db_obj = self.task_service.get_task_by_id(task_id)
        if not task_db_obj or not doc or not kb:
            # Logging and status updates are handled in the initial checks of this method
            return

        doc = self.doc_service.get_document_by_id(task_db_obj.doc_id_id)
        if not doc: # ... (error handling as before) ...
            logger.error(f"[TASK_RUNNER:{task_id}] Document {task_db_obj.doc_id_id} not found. Aborting task.")
            self.task_service.update_task_status(task_id, status="failed", error_message="Associated document not found.")
            self._check_and_finalize_document_processing(task_db_obj.doc_id_id) # Check doc status
            return

        kb = doc.kb_id
        if not kb: # ... (error handling as before) ...
            logger.error(f"[TASK_RUNNER:{task_id}] KnowledgeBase not found for document {doc.id}. Aborting task.")
            self.task_service.update_task_status(task_id, status="failed", error_message="Associated KnowledgeBase not found.")
            self.doc_service.update_document_status(doc.id, status="failed", error_message="KnowledgeBase not found.")
            self._check_and_finalize_document_processing(doc.id)
            return
        
        parser_instance = self._get_parser_instance(kb, doc.file_type)
        if not parser_instance:
            err_msg = f"No suitable parser found for KB config '{kb.parser_id}' and file type '{doc.file_type}'."
            logger.error(f"[TASK_RUNNER:{task_id}] {err_msg}")
            self.task_service.update_task_status(task_id, status="failed", error_message=err_msg)
            self.doc_service.update_document_status(doc.id, status="failed", error_message=err_msg) # Also mark doc as failed
            self._check_and_finalize_document_processing(doc.id)
            return

        doc_file_path = doc.file_path
        parser_config_from_kb = kb.parser_config or {} # Ensure it's a dict
        chunks_for_vector_store = []
        total_tokens_for_task = 0 # Placeholder for M2
        task_layout_artifacts = []
        processed_successfully = False

        try:
            logger.info(f"[TASK_RUNNER:{task_id}] Using parser: {parser_instance.__class__.__name__} for doc: '{doc.name}' (ID: {doc.id})")
            # Use 0-indexed page numbers from the task object for processing
            task_from_page_0_idx = task_db_obj.from_page 
            task_to_page_0_idx = task_db_obj.to_page # This is 0-indexed inclusive, or -1 for all

            logger.info(f"[TASK_RUNNER:{task_id}] Task processing scope: Pages {task_from_page_0_idx} to {task_to_page_0_idx} (0-indexed, inclusive)")

            # ... (determine parser_type as before) ...
            current_parser_id = kb.parser_id or DEFAULT_PARSER_ID
            if current_parser_id == "deepdoc_auto":
                resolved_parser_key = DEEPDOC_PARSER_BY_FILE_TYPE.get(doc.file_type.lower(), "naive_txtai")
                parser_type = PARSER_MAPPING.get(resolved_parser_key, {}).get("type", "naive")
            else:
                parser_type = PARSER_MAPPING.get(current_parser_id, {}).get("type", "naive")


            if parser_type == "deepdoc":
                # --- DeepDoc Parser Handling ---
               
                file_content_bytes = None # For parsers that need binary content
                # For PDF Parser
                if isinstance(parser_instance, RAGFlowPdfParser):
                    pdf_options = kb.parser_config.get("pdf_parser_options", {})
                    need_image = pdf_options.get("need_image", True) # Set to True if you want to extract images
                    zoomin = pdf_options.get("zoomin", 3)
                    return_html_tables = pdf_options.get("return_html_tables", True)
                    
                    # Pass page ranges to RAGFlowPdfParser's __call__
                    text_sections, tables_figures = parser_instance(
                        doc_file_path,
                        from_page=task_from_page_0_idx,
                        to_page=task_to_page_0_idx, # RAGFlowPdfParser's __images__ handles -1 or large numbers
                        need_image=need_image,
                        zoomin=zoomin,
                        return_html=return_html_tables
                    )
                    for text, style in text_sections: # RAGFlowPdfParser now returns (text, style) list
                        if text and text.strip():
                            # The `style_info` from RAGFlowPdfParser (via __filterout_scraps) contains "@@page-x0-x1-top-bottom##"
                            # We need to parse this tag to get coordinates and actual page number.
                            page_meta = {}
                            tag_match = re.search(r"@@([0-9-]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)##", style_info)
                            if tag_match:
                                page_str, x0, x1, top, bottom = tag_match.groups()
                                # page_str can be "1-2-3" if content spans pages. Take the first one.
                                actual_page_num = int(page_str.split('-')[0]) -1 # 1-indexed from tag, convert to 0-indexed
                                page_meta = {
                                    "original_page_number": actual_page_num, # 0-indexed absolute
                                    "bbox_on_page": [float(x0), float(top), float(x1), float(bottom)] # Coords relative to that page
                                }
                                style_info = "parsed_text_section" # Replace tag with generic style
                            
                            chunk_metadata = {"style": style_info, "parser": "deepdoc_pdf"}
                            chunk_metadata.update(page_meta)
                            chunks_for_vector_store.append(self._create_chunk_dict(text, doc, kb, chunk_metadata))
                    
                    for image_pil, content_data in tables_figures: # Process 
                        text_representation = ""
                        is_table = False
                        artifact_type_str = "figure" # Default
                        item_page_meta = {}
                        # Try to extract positional tag from content_data if it's a list of strings
                        # or from the text_representation if it's a string
                        raw_text_for_tag = ""
                        if isinstance(content_data, list) and content_data:
                            raw_text_for_tag = content_data[0] # Assume tag is in the first caption line
                        elif isinstance(content_data, str):
                            raw_text_for_tag = content_data 
                        
                        tag_match_item = re.search(r"@@([0-9-]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)\t([\d.]+)##", raw_text_for_tag)
                        if tag_match_item:
                            page_str_item, x0_item, x1_item, top_item, bottom_item = tag_match_item.groups()
                            actual_page_num_item = int(page_str_item.split('-')[0]) -1 # 0-indexed absolute
                            item_page_meta = {
                                "original_page_number": actual_page_num_item,
                                "bbox_on_page": [float(x0_item), float(top_item), float(x1_item), float(bottom_item)]
                            }
                            # Clean the tag from content_data
                            if isinstance(content_data, list):
                                content_data = [parser_instance.remove_tag(t) for t in content_data]
                            elif isinstance(content_data, str):
                                content_data = parser_instance.remove_tag(content_data)
                        
                        # Prepare text and determine type
                        if isinstance(content_data, list): # Usually figure captions
                            text_representation = " ".join(content_data)
                            artifact_type_str = "figure"
                        elif isinstance(content_data, str): # Usually HTML table
                            import html2text 
                            h_converter = html2text.HTML2Text()
                            h_converter.ignore_links = True; h_converter.ignore_images = True
                            text_representation = h_converter.handle(content_data)
                            is_table = True
                            artifact_type_str = "table"
                        
                        stored_image_path = None
                        if need_image and image_pil and item_page_meta.get("original_page_number") is not None:
                            stored_image_path = self._store_artifact_image(
                                kb.id, doc.id, image_pil, artifact_type_str, item_page_meta["original_page_number"]
                            )
                        
                        # Add to layout_artifacts for Document DB field
                        artifact_record = {
                            "type": artifact_type_str,
                            "page": item_page_meta.get("original_page_number"),
                            "bbox": item_page_meta.get("bbox_on_page"),
                            "stored_image_path": stored_image_path, # Path to the PNG/JPG on disk
                            "text_content_preview": (text_representation[:200] + "...") if len(text_representation) > 200 else text_representation,
                        }
                        if is_table and return_html_tables and isinstance(content_data, str): # Store original HTML for tables
                            artifact_record["html_content"] = content_data
                        task_layout_artifacts.append(artifact_record)

                        # Add text representation to vector store
                        if text_representation.strip():
                            chunk_meta = {"parser": f"deepdoc_pdf_{artifact_type_str}"}
                            chunk_meta.update(item_page_meta)
                            chunk_meta["is_table_or_figure"] = True
                            chunk_meta["artifact_image_path"] = stored_image_path
                            chunks_for_vector_store.append(self._create_chunk_dict(text_representation, doc, kb, chunk_meta))

                elif isinstance(parser_instance, RAGFlowDocxParser):
                    # RAGFlowDocxParser takes from_page, to_page in __call__
                    secs, tbls = parser_instance(
                        doc_file_path, 
                        from_page=task_from_page_0_idx, # DocxParser uses 0-indexed
                        to_page=task_to_page_0_idx if task_to_page_0_idx != -1 else 100000000 # Max pages
                    )
                    for text, style in secs:
                         chunk_metadata = {"style": style, "parser": "deepdoc_docx"}
                        # Add page scope information if meaningful
                        if task_from_page_0_idx != 0 or task_to_page_0_idx != -1:
                             chunk_metadata["page_scope"] = f"{task_from_page_0_idx}-{task_to_page_0_idx}"
                        if text and text.strip():
                            chunks_for_vector_store.append(self._create_chunk_dict(text, doc, kb, chunk_metadata))
                    for i, table_lines_list in enumerate(tbls):
                        table_text = "\n".join(table_lines_list)
                        if table_text.strip():
                            chunk_metadata = {"is_table": True, "parser": "deepdoc_docx_table"}
                            if task_from_page_0_idx != 0 or task_to_page_0_idx != -1:
                                 chunk_metadata["page_scope"] = f"{task_from_page_0_idx}-{task_to_page_0_idx}"
                            chunks_for_vector_store.append(self._create_chunk_dict(f"Table Content {i+1}:\n{table_text}", doc, kb, chunk_metadata))
                            task_layout_artifacts.append({
                                "type": "table_text",
                                # "page": ? needs more info from docx parser if available
                                "text_content_preview": (table_text[:200] + "...") if len(table_text) > 200 else table_text,
                            })

                
                elif isinstance(parser_instance, RAGFlowPptParser):
                    slide_texts = parser_instance(
                        doc_file_path, 
                        from_page=task_from_page_0_idx, # PptParser uses 0-indexed
                        to_page=task_to_page_0_idx if task_to_page_0_idx != -1 else 100000000 # Max slides
                    )
                    # ... (process slide_texts as before) ...
                    for slide_text in slide_texts:
                        if slide_text and slide_text.strip():
                            chunks_for_vector_store.append(self._create_chunk_dict(slide_text, doc, kb, {"parser": "deepdoc_ppt"}))
                    
                elif isinstance(parser_instance, (RAGFlowTxtParser, RAGFlowMarkdownParser, RAGFlowHtmlParser, RAGFlowJsonParser, RAGFlowExcelParser)):
                    # These parsers have varied __call__ signatures.
                    # Some take `binary` content, some take `fnm`.
                    # Some return list of strings, some list of (text, style), some structured.
                    # This requires careful adaptation for each.
                    
                    # General strategy: Read file as bytes, pass to parser if it expects `binary`.
                    file_content_bytes = None
                    try:
                        with open(doc_file_path, "rb") as f_bin:
                            file_content_bytes = f_bin.read()
                    except Exception as e_read:
                        raise RuntimeError(f"Could not read file {doc_file_path} for deepdoc parser: {e_read}")

                    if isinstance(parser_instance, RAGFlowTxtParser):
                        if file_content_bytes is None: # Load if not already for other parsers
                        with open(doc_file_path, "rb") as f_bin: 
                            file_content_bytes = f_bin.read()
                        txt_options = parser_config_from_kb.get("txt_parser_options", {})
                        delimiter = txt_options.get("delimiter", DEFAULT_CHUNK_DELIMITER)
                        parsed_output = parser_instance(doc_file_path, binary=file_content_bytes, delimiter=delimiter)
                        for text, _ in parsed_output:
                            if text and text.strip(): chunks_for_vector_store.append(self._create_chunk_dict(text, doc, kb, {"parser": "deepdoc_txt"}))
                    
                    elif isinstance(parser_instance, RAGFlowMarkdownParser):
                        if file_content_bytes is None:
                            with open(doc_file_path, "rb") as f_bin: file_content_bytes = f_bin.read()
                        md_text_content = file_content_bytes.decode(find_codec(file_content_bytes) or 'utf-8')
                        remainder_text, tables_str_list = parser_instance.extract_tables_and_remainder(md_text_content)
                        if remainder_text.strip(): chunks_for_vector_store.append(self._create_chunk_dict(remainder_text, doc, kb, {"parser": "deepdoc_md"}))
                        for table_str in tables_str_list:
                            if table_str.strip(): chunks_for_vector_store.append(self._create_chunk_dict(f"Table Content:\n{table_str}", doc, kb, {"is_table": True, "parser": "deepdoc_md_table"}))

                    elif isinstance(parser_instance, RAGFlowHtmlParser):
                    if file_content_bytes is None:
                        with open(doc_file_path, "rb") as f_bin: file_content_bytes = f_bin.read()
                    sections = parser_instance(doc_file_path, binary=file_content_bytes)
                    for section_text in sections:
                        if section_text and section_text.strip(): chunks_for_vector_store.append(self._create_chunk_dict(section_text, doc, kb, {"parser": "deepdoc_html"}))

                elif isinstance(parser_instance, RAGFlowJsonParser):
                    if file_content_bytes is None:
                        with open(doc_file_path, "rb") as f_bin: file_content_bytes = f_bin.read()
                    sections = parser_instance(file_content_bytes)
                    for json_chunk_str in sections:
                        if json_chunk_str and json_chunk_str.strip(): chunks_for_vector_store.append(self._create_chunk_dict(json_chunk_str, doc, kb, {"parser": "deepdoc_json", "content_type": "json_chunk"}))

                elif isinstance(parser_instance, RAGFlowExcelParser):
                    # Excel parser does not take page ranges
                    excel_options = parser_config_from_kb.get("excel_parser_options", {})
                    use_html_output = excel_options.get("use_html_output", False)
                    if use_html_output:
                        html_chunks = parser_instance.html(doc_file_path) 
                        for html_table_chunk in html_chunks:
                            import html2text
                            h_converter = html2text.HTML2Text(); h_converter.ignore_links = True; h_converter.ignore_images = True
                            text_representation = h_converter.handle(html_table_chunk)
                            if text_representation.strip(): chunks_for_vector_store.append(self._create_chunk_dict(text_representation, doc, kb, {"parser": "deepdoc_excel_html_table"}))
                    else:
                        row_texts = parser_instance(doc_file_path)
                        for row_text in row_texts:
                            if row_text and row_text.strip(): chunks_for_vector_store.append(self._create_chunk_dict(row_text, doc, kb, {"parser": "deepdoc_excel_row"}))
                else:
                    logger.error(f"[TASK_RUNNER:{task_id}] Unhandled deepdoc parser type: {parser_instance.__class__.__name__}")
                    raise NotImplementedError(f"Deepdoc parser {parser_instance.__class__.__name__} output handling not implemented.")

            else: # Naive TxtaiDocumentParser (not page-aware)
                # This branch should only be hit if this is the *only* task for this document,
                # or if page-aware logic in TaskService decided to create a single task.
                if task_from_page_0_idx != 0 or task_to_page_0_idx != -1:
                     logger.warning(f"[TASK_RUNNER:{task_id}] Non-page-aware parser '{parser_instance.__class__.__name__}' received a page-scoped task. "
                                   f"It will process the entire document. Task page range: {task_from_page_0_idx}-{task_to_page_0_idx}. "
                                   f"Ensure TaskService creates only one task for such (parser, file_type) combinations to avoid duplication.")
                
                parsed_content_generator = parser_instance.parse(doc_file_path)
                for content_dict in parsed_content_generator:
                    text_content = content_dict.get("text")
                    source_info = content_dict.get("source") 
                    if text_content and text_content.strip():
                        chunks_for_vector_store.append(self._create_chunk_dict(text_content, doc, kb, {"original_source_tag": source_info, "parser": "naive_txtai"}))
            
            self.task_service.update_task_status(task_id, status="processing", progress=0.3)
            logger.info(f"[TASK_RUNNER:{task_id}] Parsed {len(chunks_for_vector_store)} chunks from '{doc.name}'.")

            if chunks_for_vector_store:
                if len(chunks_for_vector_store) > MAX_CHUNKS_PER_TASK_SEGMENT:
                    logger.warning(f"[TASK_RUNNER:{task_id}] Task produced {len(chunks_for_vector_store)} chunks, "
                                   f"truncating to {MAX_CHUNKS_PER_TASK_SEGMENT} for this segment.")
                    chunks_for_vector_store = chunks_for_vector_store[:MAX_CHUNKS_PER_TASK_SEGMENT]
                
                logger.info(f"[TASK_RUNNER:{task_id}] Adding {len(chunks_for_vector_store)} chunks to TxtaiVectorStore.")
                self.vector_store.add_documents(chunks_for_vector_store)
                logger.info(f"[TASK_RUNNER:{task_id}] Successfully added chunks to vector store.")
            else:
                logger.info(f"[TASK_RUNNER:{task_id}] No chunks to add to vector store for this task.")

            self.task_service.update_task_status(task_id, status="processing", progress=0.9)
            processed_successfully = True

        except Exception as e:
            error_msg = f"Error during task execution for task {task_id} (Doc: {doc.id}, Parser: {parser_instance.__class__.__name__ if parser_instance else 'N/A'}): {e}"
            logger.error(f"[TASK_RUNNER:{task_id}] {error_msg}", exc_info=True)
            self.task_service.update_task_status(task_id, status="failed", error_message=str(e))
        
        finally:
            if processed_successfully:
                self.task_service.update_task_status(task_id, status="completed", progress=1.0)
                estimated_tokens = sum(len(c.get("text","").split()) for c in chunks_for_vector_store) # Placeholder for M2
                self.doc_service.increment_document_chunk_and_token_count(
                    doc.id,
                    chunk_increment=len(chunks_for_vector_store),
                    token_increment=estimated_tokens 
                )
                logger.info(f"[TASK_RUNNER:{task_id}] Task completed successfully.")
                if task_layout_artifacts:
                        try:
                            # If doc.layout_analysis_results already has data (e.g. from another task for same doc),
                            # you might want to merge instead of overwrite.
                            # For simplicity in C4, this example overwrites if it's the first task,
                            # or appends if data already exists (assuming list format).
                            
                            # Fetch current doc layout results
                            current_doc_obj = self.doc_service.get_document_by_id(doc.id) # Get fresh instance
                            if current_doc_obj:
                                existing_artifacts = current_doc_obj.layout_analysis_results or []
                                if not isinstance(existing_artifacts, list): existing_artifacts = []
                                
                                updated_artifacts = existing_artifacts + task_layout_artifacts
                                
                                current_doc_obj.layout_analysis_results = updated_artifacts
                                current_doc_obj.save()
                                logger.info(f"[TASK_RUNNER:{task_id}] Updated Document.layout_analysis_results for doc {doc.id} with {len(task_layout_artifacts)} new artifacts.")
                        except Exception as e_save_layout:
                            logger.error(f"[TASK_RUNNER:{task_id}] Failed to save layout_analysis_results for doc {doc.id}: {e_save_layout}")
            
            self._check_and_finalize_document_processing(doc.id) # This checks all tasks for the doc
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
                    self.graph_builder.infer_topics_for_all_nodes() 
                    logger.info(f"GraphRAG topic inference triggered for document {doc_id}.")
                if PERSIST_DB_PATH and self.vector_store:
                    logger.info(f"Persisting TxtaiVectorStore after document {doc.id} processing.")
                    self.vector_store.save(PERSIST_DB_PATH)
        else:
            # Update overall document progress based on completed tasks
            all_tasks_for_doc = self.task_service.get_tasks_for_document(doc_id)
            if all_tasks_for_doc:
                completed_progress_sum = sum(t.progress for t in all_tasks_for_doc if t.status == "completed")
                current_processing_progress = sum(t.progress for t in all_tasks_for_doc if t.status == "processing") * 0.5 
                total_tasks_count = len(all_tasks_for_doc)
                overall_doc_progress = (completed_progress_sum + current_processing_progress) / total_tasks_count if total_tasks_count > 0 else 0.0
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
        # context_override for RAG system should be list of dicts: [{"text": ..., "metadata": ...}]
        rag_context_override_list_of_dicts: Optional[List[Dict[str, Any]]] = None
        final_query_for_llm = user_query
        response_type = "rag" 

        if self.graph_builder:
            modified_query, graph_rag_data_list_of_dicts, plot_img = self.graph_builder.get_graph_rag_context(user_query)
            # graph_rag_data_list_of_dicts is now list of {"text": ..., "metadata": ...}
            if graph_rag_data_list_of_dicts:
                rag_context_override_list_of_dicts = graph_rag_data_list_of_dicts
                final_query_for_llm = modified_query
                graph_image = plot_img
                response_type = "graph_rag"
                logger.debug(f"GraphRAG context ({len(rag_context_override_list_of_dicts)} items) for query: '{final_query_for_llm}'")
            else:
                pass # RAG pipeline will fetch its own context
        
        # `rag_context_override_list_of_dicts` will be passed to rag_system.answer
        # TxtaiRAGSystem._prepare_context_for_llm will then format it correctly.
        answer_content = self.rag_system.answer(
            question=final_query_for_llm,
            context_override=rag_context_override_list_of_dicts, 
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