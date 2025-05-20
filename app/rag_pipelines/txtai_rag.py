from txtai import RAG
from app.llm_services import TxtaiLLM # Assuming TxtaiLLM
from app.vector_store import BaseVectorStore # For type hinting
from app.config import RAG_SYSTEM_PROMPT, RAG_TEMPLATE, RAG_CONTEXT_SIZE, LLM_MAX_LENGTH
import streamlit as st
from typing import Optional, Union, Generator


logger = st.logger.get_logger(__name__)

class TxtaiRAGSystem:
    """
    Encapsulates the RAG (Retrieval Augmented Generation) pipeline using txtai.
    """
    def __init__(self, vector_store: BaseVectorStore, llm_service: TxtaiLLM):
        """
        Args:
            vector_store: A vector store instance (must be TxtaiVectorStore for txtai.RAG).
            llm_service: An LLM service instance (must provide underlying txtai.LLM).
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.rag_pipeline = None
        self._initialize_rag()

    def _initialize_rag(self):
        # txtai.RAG expects a txtai.Embeddings instance and a txtai.LLM instance
        underlying_embeddings = self.vector_store.get_underlying_embeddings_instance()
        if not underlying_embeddings:
            # This is a critical error for TxtaiRAGSystem
            logger.error("TxtaiRAGSystem requires a TxtaiVectorStore with an underlying txtai.Embeddings instance.")
            raise ValueError("Cannot initialize TxtaiRAGSystem without txtai.Embeddings from vector store.")

        # Assuming llm_service.llm is the actual txtai.LLM object after lazy loading
        self.llm_service._load_llm() # Ensure LLM is loaded
        underlying_llm = self.llm_service.llm
        if not underlying_llm:
            logger.error("TxtaiRAGSystem requires an underlying txtai.LLM instance from the LLM service.")
            raise ValueError("Cannot initialize TxtaiRAGSystem without txtai.LLM from LLM service.")

        logger.info("Initializing txtai.RAG pipeline...")
        self.rag_pipeline = RAG(
            embeddings=underlying_embeddings,
            llm=underlying_llm,
            system=RAG_SYSTEM_PROMPT,
            template=RAG_TEMPLATE,
            context=RAG_CONTEXT_SIZE, # Default context size for vector search part of RAG
        )
        logger.info("txtai.RAG pipeline initialized.")

    def answer(self, question: str, context_override: Optional[list[str]] = None, stream: bool = False) -> Union[str, Generator[str, None, None]]:
        """
        Generates an answer using the RAG pipeline.

        Args:
            question: The user's question.
            context_override: Optional list of strings to use as context instead of
                              performing a vector search. Used by GraphRAG.
            stream: Whether to stream the response.

        Returns:
            The generated answer (string) or a generator for streaming.
        """
        if not self.rag_pipeline:
            logger.error("RAG pipeline not initialized. Cannot answer.")
            return "Error: RAG system not ready." if not stream else (lambda: (yield "Error: RAG system not ready."))()

        logger.info(f"RAG answering question: '{question}'. Stream: {stream}. Context override provided: {context_override is not None}")
        
        # txtai.RAG __call__ method:
        # (query, request=None, system=None, template=None, context=None, passages=None, **kwargs)
        # 'context' here is number of results from vector search.
        # 'passages' is for context_override.
        
        rag_args = {
            "maxlength": LLM_MAX_LENGTH, # Max output tokens
            "stream": stream
        }
        if context_override is not None:
            rag_args["passages"] = context_override
            # When passages are provided, RAG_CONTEXT_SIZE (for vector search) is ignored.
        
        return self.rag_pipeline(question, **rag_args)