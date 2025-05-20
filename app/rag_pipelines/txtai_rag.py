from txtai import RAG
from app.llm_services import TxtaiLLM
from app.vector_store import BaseVectorStore
from app.config import (
    RAG_SYSTEM_PROMPT, RAG_TEMPLATE, RAG_CONTEXT_SIZE, LLM_MAX_LENGTH,
    RETURN_SEPARATOR, # For formatting context
    TXT_AI_HYBRID_SEARCH_ENABLED # New config flag
)
from app.utils.text_processing_utils import format_metadata_for_llm_context # Import new formatter
import streamlit as st
from typing import Optional, Union, Generator, List, Dict, Any

logger = st.logger.get_logger(__name__)

class TxtaiRAGSystem:
    def __init__(self, vector_store: BaseVectorStore, llm_service: TxtaiLLM):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.rag_pipeline = None
        self._initialize_rag() # This needs to be robust to vector_store not being fully ready if called early
        self.hybrid_search_enabled = TXT_AI_HYBRID_SEARCH_ENABLED # From app.config

    def _initialize_rag(self):
        # txtai.RAG expects a txtai.Embeddings instance and a txtai.LLM instance
        underlying_embeddings = self.vector_store.get_underlying_embeddings_instance()
        if not underlying_embeddings:
            # This is a critical error for TxtaiRAGSystem
            logger.error("TxtaiRAGSystem requires a TxtaiVectorStore with an underlying txtai.Embeddings instance.")
            return

        # Assuming llm_service.llm is the actual txtai.LLM object after lazy loading
        self.llm_service._load_llm() # Ensure LLM is loaded
        underlying_llm = self.llm_service.llm
        if not underlying_llm:
            logger.error("TxtaiRAGSystem requires an underlying txtai.LLM instance from the LLM service.")
            return

        logger.info("Initializing txtai.RAG pipeline...")
        self.rag_pipeline = RAG(
            embeddings=underlying_embeddings,
            llm=underlying_llm,
            system=RAG_SYSTEM_PROMPT,
            template=RAG_TEMPLATE, # The template expects {question} and {context}
            context=RAG_CONTEXT_SIZE, 
            # output_format="dictionaries" # if we want RAG to return source passages
        )
        logger.info("txtai.RAG pipeline initialized.")

    def _prepare_context_for_llm(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks (text + metadata) into a single string for the LLM.
        """
        context_parts = []
        for chunk_dict in retrieved_chunks:
            text = chunk_dict.get("text", "")
            metadata = chunk_dict.get("metadata", {}) # This is the full original chunk dict
            
            metadata_prefix = format_metadata_for_llm_context(metadata)
            
            # Combine prefix and text
            # Example: "[Source: DocA.pdf, Page: 3] This is the chunk text."
            # Or: "This is the chunk text. [Source: DocA.pdf, Page: 3]"
            # Let's try prefixing.
            
            # If metadata_prefix is not empty, add a space after it.
            formatted_chunk = f"{metadata_prefix} {text}".strip() if metadata_prefix else text
            
            context_parts.append(formatted_chunk)
        
        return RETURN_SEPARATOR.join(context_parts) # Join with configured separator

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
            err_msg = "Error: RAG system not ready."
            return err_msg if not stream else (lambda: (yield err_msg))()

        logger.info(f"RAG answering question: '{question}'. Stream: {stream}. Context override: {context_override is not None}. Hybrid search intent: {self.hybrid_search_enabled}")
        
        rag_args = {
            "maxlength": LLM_MAX_LENGTH,
            "stream": stream
        }

        final_context_str_for_llm = ""

        if context_override is not None:
            # If context_override is provided (e.g., from GraphRAG), it's expected to be a list of text strings.
            # We need their corresponding metadata if we want to format them with `format_metadata_for_llm_context`.
            # The current `GraphRAGBuilder.get_graph_rag_context` returns `graph_rag_data` as list of dicts
            # `[{"id": original_doc_id, "text": text_content}]`.
            # We need to adapt this. For now, assume `context_override` contains dicts with "text" and "metadata".
            # This means `GraphRAGBuilder` needs to return richer objects.

            # Let's assume context_override is now a list of dicts like:
            # [{"text": "...", "metadata": {...original_chunk_metadata...}}]
            if context_override and isinstance(context_override[0], dict):
                 final_context_str_for_llm = self._prepare_context_for_llm(context_override)
            elif context_override: # Fallback if it's just list of strings (old behavior)
                 final_context_str_for_llm = RETURN_SEPARATOR.join(context_override)
            
            rag_args["passages"] = final_context_str_for_llm # txtai.RAG takes this as the context string
        else:
            # Perform retrieval using the vector store if no override
            retrieved_chunks = self.vector_store.search(
                query=question, 
                k=RAG_CONTEXT_SIZE, 
                hybrid=self.hybrid_search_enabled # Signal intent for hybrid
            )
            if retrieved_chunks:
                final_context_str_for_llm = self._prepare_context_for_llm(retrieved_chunks)
            else:
                logger.info("No relevant chunks found by vector store search.")
                # Let RAG proceed with empty context, it might still answer from its base knowledge or indicate it can't.
            
            # Because we've prepared the context string manually, we pass it as `passages`
            # and txtai.RAG will use it directly instead of doing its own search.
            rag_args["passages"] = final_context_str_for_llm
            # We could also let RAG do its own search and then re-format its output,
            # but passing passages gives more control. If we pass passages, RAG's own
            # `context` parameter (number of search results) is ignored.

        # Now, call RAG with the question. It will use the `passages` if provided.
        # The template {context} will be filled by `final_context_str_for_llm`.
        return self.rag_pipeline(question, **rag_args)