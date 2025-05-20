import logging
import uuid
from typing import Optional, List, Dict, Any
from peewee import DoesNotExist

from app.database import DB # Direct access to DB for transactions
from app.database.models import KnowledgeBase, Document
from app.config import (
    DEFAULT_KB_DESCRIPTION, DEFAULT_KB_LANGUAGE, DEFAULT_KB_EMBED_MODEL,
    DEFAULT_PARSER_ID, DEFAULT_PARSER_CONFIG, DEFAULT_KB_SIMILARITY_THRESHOLD,
    DEFAULT_KB_VECTOR_WEIGHT
)

logger = logging.getLogger(__name__)

class KnowledgeBaseService:

    @staticmethod
    @DB.connection_context()
    def create_kb(name: str,
                  description: Optional[str] = None,
                  language: Optional[str] = None,
                  embd_model: Optional[str] = None,
                  parser_id: Optional[str] = None,
                  parser_config: Optional[Dict[str, Any]] = None,
                  similarity_threshold: Optional[float] = None,
                  vector_weight: Optional[float] = None
                  ) -> KnowledgeBase:
        """
        Creates a new Knowledge Base.
        """
        kb_id = str(uuid.uuid4().hex)
        kb = KnowledgeBase.create(
            id=kb_id,
            name=name,
            description=description or DEFAULT_KB_DESCRIPTION,
            language=language or DEFAULT_KB_LANGUAGE,
            embd_model=embd_model or DEFAULT_KB_EMBED_MODEL,
            parser_id=parser_id or DEFAULT_PARSER_ID,
            parser_config=parser_config or DEFAULT_PARSER_CONFIG, # Ensure lambda default used if None
            similarity_threshold=similarity_threshold if similarity_threshold is not None else DEFAULT_KB_SIMILARITY_THRESHOLD,
            vector_weight=vector_weight if vector_weight is not None else DEFAULT_KB_VECTOR_WEIGHT,
            document_count=0
        )
        logger.info(f"Knowledge Base created: ID={kb_id}, Name='{name}'")
        return kb

    @staticmethod
    @DB.connection_context()
    def get_kb_by_id(kb_id: str) -> Optional[KnowledgeBase]:
        try:
            return KnowledgeBase.get(KnowledgeBase.id == kb_id)
        except DoesNotExist:
            logger.warning(f"Knowledge Base with ID '{kb_id}' not found.")
            return None

    @staticmethod
    @DB.connection_context()
    def get_kb_by_name(name: str) -> Optional[KnowledgeBase]:
        try:
            return KnowledgeBase.get(KnowledgeBase.name == name)
        except DoesNotExist:
            logger.warning(f"Knowledge Base with name '{name}' not found.")
            return None
    
    @staticmethod
    @DB.connection_context()
    def list_kbs() -> List[KnowledgeBase]:
        return list(KnowledgeBase.select())

    @staticmethod
    @DB.connection_context()
    def update_kb(kb_id: str, updates: Dict[str, Any]) -> Optional[KnowledgeBase]:
        kb = KnowledgeBaseService.get_kb_by_id(kb_id)
        if not kb:
            return None
        
        allowed_fields = ["name", "description", "language", "embd_model",
                          "parser_id", "parser_config", "similarity_threshold", "vector_weight"]
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(kb, field, value)
        
        kb.save()
        logger.info(f"Knowledge Base updated: ID='{kb_id}'")
        return kb

    @staticmethod
    @DB.connection_context()
    def delete_kb(kb_id: str) -> bool:
        # This will also delete associated documents due to on_delete='CASCADE'
        # and associated tasks for those documents.
        # Vector store entries need separate cleanup.
        kb = KnowledgeBaseService.get_kb_by_id(kb_id)
        if not kb:
            return False
        
        # TODO: Add logic here to delete associated data from the vector store (txtai index)
        # This is crucial to prevent orphaned embeddings.
        # For txtai, this might involve deleting documents by IDs that were part of this KB.
        # We need a way to get all document IDs for this KB first.
        
        doc_ids_to_delete_from_vector_store = [doc.id for doc in kb.documents] # kb.documents is Peewee backref
        if doc_ids_to_delete_from_vector_store:
            logger.info(f"KB Delete: Will need to remove {len(doc_ids_to_delete_from_vector_store)} docs from vector store for KB {kb_id}.")
            # Placeholder for actual vector store deletion logic in Orchestrator or a dedicated service
            # For now, we just log. The Orchestrator will need to handle this.

        kb.delete_instance(recursive=True) # recursive=True handles related objects if not cascaded by DB
        logger.info(f"Knowledge Base and associated DB records deleted: ID='{kb_id}'")
        return True

    @staticmethod
    @DB.connection_context()
    def increment_kb_document_count(kb_id: str, count: int = 1):
        try:
            query = KnowledgeBase.update(document_count=KnowledgeBase.document_count + count).where(KnowledgeBase.id == kb_id)
            query.execute()
        except Exception as e:
            logger.error(f"Error incrementing document count for KB {kb_id}: {e}")

    @staticmethod
    @DB.connection_context()
    def decrement_kb_document_count(kb_id: str, count: int = 1):
        try:
            # Ensure count doesn't go below zero
            query = KnowledgeBase.update(document_count=KnowledgeBase.document_count - count).where(KnowledgeBase.id == kb_id, KnowledgeBase.document_count >= count)
            # If you want to clamp at 0 instead of conditional update:
            # kb = KnowledgeBase.get(KnowledgeBase.id == kb_id)
            # kb.document_count = max(0, kb.document_count - count)
            # kb.save()
            query.execute()
        except Exception as e:
            logger.error(f"Error decrementing document count for KB {kb_id}: {e}")