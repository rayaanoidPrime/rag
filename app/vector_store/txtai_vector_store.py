from typing import List, Dict, Any, Optional
from txtai import Embeddings
from app.config import (
    VECTOR_DB_PATH,
    # VECTOR_DB_CLOUD_PROVIDER,
    # VECTOR_DB_CLOUD_CONTAINER,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_INSTRUCTIONS,
    GRAPH_APPROXIMATE_SEARCH,
    GRAPH_MIN_SCORE,
)
from .base_vector_store import BaseVectorStore
import streamlit as st # For logger
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
        self._embedder = embedder # Not used yet, txtai.Embeddings handles its own
        self._is_loaded = False

    def _initialize_embeddings(self):
        logger.info("Initializing new txtai Embeddings instance for vector store.")
        self.embeddings = Embeddings(
            autoid="uuid5", # As in original
            path=EMBEDDING_MODEL_PATH, # Path for the embedding model itself
            instructions=EMBEDDING_INSTRUCTIONS,
            content=True, # Store content within txtai
            graph={ # Configure graph capabilities
                "approximate": GRAPH_APPROXIMATE_SEARCH,
                "minscore": GRAPH_MIN_SCORE,
            },
        )
        self._is_loaded = True # Mark as "loaded" because it's newly created

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
            self._initialize_embeddings() # Ensure it's initialized
            if not self.embeddings: # Still not initialized after attempt
                 logger.error("Failed to initialize Embeddings for adding documents.")
                 return

        # txtai's upsert expects list of (id, data, tags) tuples.
        # 'data' can be text (for auto-embedding) or a precomputed vector.
        # 'id' from document dict, 'text' from document dict.
        
        data_to_upsert = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id")
            doc_text = doc.get("text")
            doc_source = doc.get("source") # Or other metadata

            if not doc_id:
                logger.warning(f"Document at index {i} missing 'id', generating one.")
                doc_id = self.embeddings.config.get("autoid", "uuid5")() # Use configured autoid

            if not doc_text:
                logger.warning(f"Document '{doc_id}' missing 'text', skipping.")
                continue

            # txtai can take precomputed embeddings if data is set to the vector
            # and then text is passed as a tag or separate metadata field.
            # For simplicity now, we let txtai handle embedding if not provided.
            # If embeddings are provided, they should correspond to doc_text.
            # txtai's upsert handles this if data is the vector.
            # However, the original code implies upserting text and letting txtai embed.
            
            # For now, we pass text directly to txtai for embedding & indexing.
            # The `content=True` in Embeddings init ensures text is stored.
            # The 'source' can be stored as a tag or custom attribute.
            # txtai upsert expects (id, data_dict_or_text_or_vector, tags_tuple_or_None)
            # If data is a dict, it's stored as metadata. If text, it's embedded.
            
            # To store source and other metadata alongside the text for content=True:
            # We can pass a dictionary as the data field.
            # txtai will look for a 'text' key in this dict to embed if content=True.
            # Other keys are stored as attributes.
            data_item = {"text": doc_text, "source": doc_source}
            # Add any other metadata from doc to data_item
            for key, value in doc.items():
                if key not in ["id", "text", "source"]: # already handled or standard
                    data_item[key] = value
            
            data_to_upsert.append((str(doc_id), data_item, None))

        if data_to_upsert:
            logger.info(f"Adding/updating {len(data_to_upsert)} documents to TxtaiVectorStore.")
            self.embeddings.upsert(data_to_upsert)
        else:
            logger.info("No valid documents to add to TxtaiVectorStore.")


    def search(self, query: str, k: int = 5, query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        if not self.embeddings:
            logger.warning("Search called but Embeddings instance not initialized.")
            return []
        
        # txtai search returns list of dicts with 'id', 'text', 'score'
        # If query_embedding is provided, txtai doesn't have a direct way to use it
        # in the .search() method if it also needs to fetch text.
        # .search() embeds the query string. .similarity() can take raw vectors.
        # For now, stick to query string for simplicity.
        if query_embedding:
            logger.warning("Precomputed query_embedding not directly used by TxtaiVectorStore.search, using query string.")

        results = self.embeddings.search(query, limit=k)
        # Ensure results match expected format (they usually do from txtai)
        # txtai search results are like: [{"id": uid, "text": text, "score": score}, ...]
        # If content=False, 'text' might be missing or be the ID itself.
        # Since we use content=True and store text in a dict, 'text' should be the actual text content.
        # The "text" field in the result dict from txtai when content=True and data is a dict
        # will actually be the dict itself. We need to extract the actual text.
        formatted_results = []
        for res in results:
            doc_content = res.get("text") # This is the dict we stored
            text_to_display = doc_content.get("text") if isinstance(doc_content, dict) else doc_content
            formatted_results.append({
                "id": res.get("id"),
                "text": text_to_display,
                "score": res.get("score"),
                "source": doc_content.get("source") if isinstance(doc_content, dict) else None
                # Add other metadata from doc_content if needed
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