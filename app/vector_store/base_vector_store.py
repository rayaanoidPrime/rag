from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseVectorStore(ABC):
    """
    Abstract base class for vector store operations.
    """

    @abstractmethod
    def load(self, path: Optional[str] = None, cloud_config: Optional[Dict] = None):
        """Loads an existing vector store from a path or cloud storage."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the current vector store to a path."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        """
        Adds documents to the vector store.
        'documents' is a list of dicts, each must have 'text' and 'id'.
        'embeddings' is an optional list of pre-computed embeddings.
        If embeddings are not provided, the store should generate them.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5, query_embedding: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector store for similar documents.
        'query_embedding' is an optional pre-computed embedding for the query.
        Returns a list of search results (dictionaries with 'id', 'text', 'score', etc.).
        """
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """Deletes documents by their IDs."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Returns the total number of documents in the store."""
        pass

    @abstractmethod
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a document by its ID."""
        pass

    @abstractmethod
    def get_underlying_embeddings_instance(self) -> Any:
        """
        Returns the underlying txtai.Embeddings instance, if applicable.
        This is specifically for compatibility with GraphRAG which needs direct
        access to txtai's graph features.
        Returns None if not applicable or not a txtai-based store.
        """
        pass