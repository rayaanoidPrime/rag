from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator, Union

class BaseDocumentParser(ABC):
    """
    Abstract base class for document parsers.
    Parsers are responsible for extracting text and metadata from various file types.
    """

    @abstractmethod
    def parse(self, file_path_or_url: Union[str, List[str]]) -> Generator[Dict[str, Any], None, None]:
        """
        Parses a document or a list of documents from a file path or URL.

        Args:
            file_path_or_url: A single file path/URL or a list of file paths/URLs.

        Yields:
            Dictionaries, where each dictionary represents a chunk or segment
            of the document. Expected keys include 'text' and optionally 'id',
            'source', 'metadata', etc.
            Example: {"id": "doc1_chunk1", "text": "This is a sentence.", "source": "doc1.pdf"}
        """
        pass

    def stream_text_from_directory(self, directory_path: str) -> Generator[Dict[str, Any], None, None]:
        """
        Scans a directory recursively and parses all supported files.
        This method can be overridden if a parser has a more efficient way
        to handle directories.

        Args:
            directory_path: The path to the directory.

        Yields:
            Parsed document chunks from files in the directory.
        """
        import glob
        # Basic implementation, can be refined for specific parsers
        files_to_parse = glob.glob(f"{directory_path}/**/*", recursive=True)
        # Filter out directories, though some parsers might handle them
        files_to_parse = [f for f in files_to_parse if os.path.isfile(f)]
        if files_to_parse:
            yield from self.parse(files_to_parse)