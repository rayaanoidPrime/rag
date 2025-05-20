import os
from typing import List, Dict, Any, Generator, Union
from txtai.pipeline import Textractor
from app.config import TEXTRACTOR_BACKEND
from .base_parser import BaseDocumentParser
import streamlit as st # For logger

logger = st.logger.get_logger(__name__)

class TxtaiDocumentParser(BaseDocumentParser):
    """
    A document parser using txtai's Textractor.
    """
    def __init__(self):
        self.textractor = Textractor(
            paragraphs=True, # Extract paragraph by paragraph
            backend=TEXTRACTOR_BACKEND,
        )
        logger.info(f"TxtaiDocumentParser initialized with backend: {TEXTRACTOR_BACKEND}")

    def parse(self, file_paths_or_urls: Union[str, List[str]]) -> Generator[Dict[str, Any], None, None]:
        """
        Parses documents using Textractor.
        Textractor returns a list of (id, text, tags) tuples or a stream of sections.
        We will yield dictionaries with "text" and "source" (id).
        """
        if isinstance(file_paths_or_urls, str):
            inputs = [file_paths_or_urls]
        else:
            inputs = file_paths_or_urls

        logger.info(f"Parsing documents: {inputs}")
        # Textractor can return a generator of (id, data, tags) or a list of these for multiple files
        # Or if processing single file, it might return a generator of sections.
        # We need to adapt this to consistently yield dicts with "text".

        # Textractor's output for multiple files is typically a list of generators,
        # or a generator of (id, text, tag) tuples.
        # For a single file, it can be a generator of sections.

        results = self.textractor(inputs)

        if isinstance(results, list) and all(isinstance(res, tuple) and len(res) == 3 for res in results):
            # Case: results = [(id1, text1, tags1), (id2, text2, tags2), ...]
            for uid, text, _ in results:
                if text: # Ensure text is not empty
                    yield {"text": text, "source": str(uid)}
        elif hasattr(results, '__iter__') and not isinstance(results, (str, bytes)):
            # Case: results is a generator
            # This could be a generator of (id, text, tags) for multiple files,
            # or a generator of sections for a single file.
            processed_sources = set()
            for item in results:
                if isinstance(item, tuple) and len(item) == 3:
                    # (id, text, tags) from processing multiple files
                    uid, text, _ = item
                    if text:
                        yield {"text": text, "source": str(uid)}
                        processed_sources.add(str(uid))
                elif isinstance(item, dict) and "text" in item: # Already structured
                    yield item
                elif isinstance(item, str): # A section from a single file
                    # Try to associate with an input if only one input was given
                    source_name = inputs[0] if len(inputs) == 1 else "unknown_section"
                    if item: # Ensure text is not empty
                        yield {"text": item, "source": source_name}
                        processed_sources.add(source_name)
                else:
                    logger.warning(f"Unknown item type from Textractor: {type(item)}. Item: {item}")

        else:
            logger.error(f"Unexpected output format from Textractor: {type(results)}")


    def stream_text_from_directory(self, directory_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Uses Textractor to stream sections from content in a data directory.
        """
        import glob
        logger.info(f"Streaming text from directory: {directory_path}")
        # Textractor can directly handle a list of file paths
        file_paths = glob.glob(f"{directory_path}/**/*", recursive=True)
        file_paths = [f for f in file_paths if os.path.isfile(f)] # Ensure only files

        if not file_paths:
            logger.info(f"No files found in directory: {directory_path}")
            return

        # Textractor with multiple file paths yields (id, text, tag) tuples
        for uid, text, _ in self.textractor(file_paths):
            if text: # Ensure text is not empty
                yield {"text": text, "source": str(uid)}