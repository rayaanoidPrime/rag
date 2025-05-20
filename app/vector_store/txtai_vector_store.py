from typing import List, Dict, Any, Optional
from txtai import Embeddings
from app.config import (
    VECTOR_DB_PATH,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_INSTRUCTIONS,
    GRAPH_APPROXIMATE_SEARCH,
    GRAPH_MIN_SCORE,
    # Add new config for hybrid search if desired
    TXT_AI_KEYWORD_INDEX_ENABLED, # e.g., True
    TXT_AI_KEYWORD_CONFIG,      # e.g., {"backend": "sqlite"} or {"backend": "whoosh", "path": "data/keywords"}
)
from .base_vector_store import BaseVectorStore
import streamlit as st
import os

logger = st.logger.get_logger(__name__)

class TxtaiVectorStore(BaseVectorStore):
    """
    Vector store implementation using txtai.Embeddings.
    This class will also hold the graph data for now, as GraphRAG
    is tightly coupled with txtai.Embeddings.
    """
    def __init__(self, embedder=None): # embedder is for future when we separate embedding generation
        self.embeddings: Optional[Embeddings] = None
        self._embedder = embedder
        self._is_loaded = False
        self._keyword_index_enabled = TXT_AI_KEYWORD_INDEX_ENABLED # From app.config
        self._keyword_config = TXT_AI_KEYWORD_CONFIG # From app.config

    def _initialize_embeddings(self):
        logger.info("Initializing new txtai Embeddings instance for vector store.")
        
        embeddings_config = {
            "autoid": "uuid5",
            "path": EMBEDDING_MODEL_PATH,
            "instructions": EMBEDDING_INSTRUCTIONS,
            "content": True, # Store content (the full metadata dict)
            "graph": {
                "approximate": GRAPH_APPROXIMATE_SEARCH,
                "minscore": GRAPH_MIN_SCORE,
            },
        }
        
        if self._keyword_index_enabled:
            embeddings_config["keyword"] = True # Enable keyword indexing
            if self._keyword_config:
                 embeddings_config["keyword"] = self._keyword_config # Pass specific backend config
            logger.info(f"Keyword indexing enabled for txtai with config: {embeddings_config['keyword']}")
        else:
            logger.info("Keyword indexing disabled for txtai.")

        self.embeddings = Embeddings(**embeddings_config)
        self._is_loaded = True

    def load(self, path: Optional[str] = VECTOR_DB_PATH, cloud_config: Optional[Dict] = None):
        logger.info(f"Attempting to load TxtaiVectorStore from path: {path}, cloud: {cloud_config}")
        temp_embeddings = Embeddings() # Temp instance to check existence

        loaded_from_cloud = False
        if cloud_config and cloud_config.get("provider") and cloud_config.get("container"):
            if temp_embeddings.exists(cloud=cloud_config):
                logger.info(f"Found existing index in cloud: {cloud_config['container']}. Loading...")
                self.embeddings = Embeddings() # Fresh instance for loading
                self.embeddings.load(provider=cloud_config["provider"], container=cloud_config["container"])
                self._is_loaded = True
                loaded_from_cloud = True
                logger.info("Loaded TxtaiVectorStore from cloud.")
            else:
                logger.info(f"No index found in cloud: {cloud_config['container']}")

        if not loaded_from_cloud and path:
            if os.path.exists(path) and temp_embeddings.exists(path):
                logger.info(f"Found existing index at local path: {path}. Loading...")
                self.embeddings = Embeddings() # Fresh instance for loading
                self.embeddings.load(path)
                self._is_loaded = True
                logger.info("Loaded TxtaiVectorStore from local path.")
            else:
                logger.info(f"No existing index found at local path: {path}. Will create new if data added.")
                self._initialize_embeddings() # Create a new one if not found locally
        elif not loaded_from_cloud and not path:
            logger.info("No path or cloud config provided for loading. Initializing new store.")
            self._initialize_embeddings()

        if not self._is_loaded: # Fallback if all attempts fail
             self._initialize_embeddings()


    def save(self, path: str):
        if not self.embeddings:
            logger.error("Cannot save, Embeddings instance not initialized.")
            return
        if not path:
            logger.warning("No path provided for saving TxtaiVectorStore.")
            return
        logger.info(f"Saving TxtaiVectorStore to: {path}")
        self.embeddings.save(path)
        logger.info("TxtaiVectorStore saved.")

    def add_documents(self, documents: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        if not self.embeddings:
            self._initialize_embeddings()
            if not self.embeddings:
                 logger.error("Failed to initialize Embeddings for adding documents.")
                 return

        data_to_upsert = []
        for i, doc_dict in enumerate(documents): # doc_dict is our chunk dict with metadata
            doc_id = doc_dict.get("id") # This will be None, txtai will generate
            
            # The entire doc_dict (which includes "text" and all other metadata)
            # is passed as the 'data' field to txtai.Embeddings.upsert.
            # txtai will embed doc_dict["text"].
            # If keyword=True, txtai will also index doc_dict["text"] for keyword search.
            
            if not doc_dict.get("text"):
                logger.warning(f"Document at index {i} (intended ID: {doc_id}) missing 'text', skipping.")
                continue
            
            # txtai's autoid will handle the first element of the tuple.
            # The second element is the data to store and from which text is extracted for embedding/keywords.
            # The third is tags (optional).
            data_to_upsert.append((doc_id, doc_dict, None))


        if data_to_upsert:
            logger.info(f"Adding/updating {len(data_to_upsert)} documents to TxtaiVectorStore.")
            self.embeddings.upsert(data_to_upsert)
            if self._keyword_index_enabled and hasattr(self.embeddings, 'index'):
                # After upserting, if keywords are enabled, txtai typically builds/updates
                # its keyword index. For some backends (like manual Whoosh), you might need
                # to explicitly call index() or ensure it's done on save/load.
                # With keyword=True and default SQLite FTS, it should handle it.
                # Calling self.embeddings.index() explicitly can rebuild various indexes.
                # logger.info("Rebuilding keyword index after upsert (if applicable).")
                # self.embeddings.index() # This can be slow, use with caution or rely on auto-indexing.
                pass
        else:
            logger.info("No valid documents to add to TxtaiVectorStore.")

        if data_to_upsert:
            logger.info(f"Adding/updating {len(data_to_upsert)} documents to TxtaiVectorStore.")
            self.embeddings.upsert(data_to_upsert)
        else:
            logger.info("No valid documents to add to TxtaiVectorStore.")


    def search(self, query: str, k: int = 5, query_embedding: Optional[List[float]] = None, hybrid: bool = False) -> List[Dict[str, Any]]:
        if not self.embeddings:
            logger.warning("Search called but Embeddings instance not initialized.")
            return []
        
        if query_embedding:
            logger.warning("Precomputed query_embedding not directly used by TxtaiVectorStore.search, using query string for (hybrid) search.")

        # txtai's search method handles hybrid search automatically if:
        # 1. `keyword=True` (or keyword config) was set during Embeddings init.
        # 2. The query string is suitable for both keyword and vector search.
        # By default, if keyword indexing is on, .search() performs a hybrid query.
        # The ratio/weighting of keyword vs vector score might be configurable in txtai
        # or it uses a default fusion method.
        
        # The `hybrid` parameter here is more of a signal to log what type of search we *intend*.
        # `txtai.Embeddings.search` itself determines if it *can* do hybrid.
        
        search_type_log = "hybrid" if hybrid and self._keyword_index_enabled and self.embeddings.config.get("keyword") else "vector"
        logger.info(f"Performing {search_type_log} search for query: '{query}' with k={k}")

        results = self.embeddings.search(query, limit=k)
        
        formatted_results = []
        for res in results:
            # res["text"] is the full dictionary we stored
            doc_content_dict = res.get("text") 
            actual_text = ""
            metadata_for_formatting = {}

            if isinstance(doc_content_dict, dict):
                actual_text = doc_content_dict.get("text", "")
                metadata_for_formatting = doc_content_dict
            elif isinstance(doc_content_dict, str): # Fallback if somehow only text was stored
                actual_text = doc_content_dict
            
            formatted_results.append({
                "id": res.get("id"),
                "text": actual_text, # The raw text of the chunk
                "score": res.get("score"),
                "metadata": metadata_for_formatting # Pass the whole metadata dict
            })
        return formatted_results

    def delete_documents(self, ids: List[str]) -> bool:
        if not self.embeddings: return False
        logger.info(f"Deleting {len(ids)} documents from TxtaiVectorStore.")
        self.embeddings.delete([str(doc_id) for doc_id in ids])
        return True # txtai delete doesn't return status, assume success if no error

    def count(self) -> int:
        return self.embeddings.count() if self.embeddings else 0

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        if not self.embeddings or not self.embeddings.issparse(): # Search by ID works best with sparse index for content
            logger.warning("Cannot get document by ID efficiently without a sparse index or if not loaded.")
            # Fallback: could do a search for the ID if it's text, but that's not ideal
            # For now, if we can't use graph attributes (which is what original code does)
            # this method is limited.
            # The graph attribute access is the primary way original code got text for graph nodes.
            # This method is more for a generic vector store.
            # We can try searching for id if it was indexed as text, or rely on graph.
            # Let's assume the graph_services will handle fetching text for its nodes.
            return None

        # If content is stored, txtai doesn't have a direct "get by id" that returns the full doc easily.
        # It's usually through search or graph attributes.
        # This method will be more useful for non-txtai stores.
        # For txtai, the graph service's access to attributes via ID is more relevant.
        # Let's return what we can, assuming ID is indexed.
        # results = self.embeddings.search(f"id:{doc_id}", limit=1) # This requires id to be part of indexed text
        # if results and results[0]['id'] == doc_id:
        #     return results[0]
        return None # Placeholder - see above comments

    def get_underlying_embeddings_instance(self) -> Optional[Embeddings]:
        """Returns the underlying txtai.Embeddings instance."""
        return self.embeddings

    def is_loaded(self) -> bool:
        return self._is_loaded and self.embeddings is not None