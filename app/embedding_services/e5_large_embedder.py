from txtai import Embeddings # We use txtai.Embeddings here for its embedding model loading
from app.config import EMBEDDING_MODEL_PATH, EMBEDDING_INSTRUCTIONS
import streamlit as st

logger = st.logger.get_logger(__name__)

class E5LargeEmbedder:
    """
    Handles embedding generation using a pre-configured e5-large model
    via txtai's Embeddings class (solely for its .transform method).
    """
    def __init__(self):
        # We initialize an Embeddings object just to use its .transform method for generating embeddings
        # This is a lightweight way to access the sentence-transformer model via txtai.
        # We don't intend to use this instance for indexing, just for transforming text to vectors.
        self.embed_model = Embeddings(
            path=EMBEDDING_MODEL_PATH,
            instructions=EMBEDDING_INSTRUCTIONS,
            autoid="uuid5" # Required, but not used for indexing here
        )
        logger.info(f"E5LargeEmbedder initialized with model: {EMBEDDING_MODEL_PATH}")

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generates embeddings for a list of texts.

        Args:
            texts: A list of strings.

        Returns:
            A list of embedding vectors (list of floats).
        """
        if not texts:
            return []
        # The transform method of txtai.Embeddings can take a list of (id, text, tags)
        # or just a list of texts. We'll provide it in a way that ensures it processes text.
        # Format: (uid, data, tags) - id and tags can be None
        documents_to_transform = [(None, text, None) for text in texts]
        return self.embed_model.transform(documents_to_transform)

    def get_embedding_dimension(self) -> int:
        """
        Returns the dimension of the embeddings produced by this model.
        """
        # Generate a dummy embedding to find its dimension
        dummy_embedding = self.get_embeddings(["test"])
        return len(dummy_embedding[0]) if dummy_embedding else 0