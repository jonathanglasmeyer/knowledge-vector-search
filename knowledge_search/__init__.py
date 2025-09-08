"""Knowledge Vector Search - Lightweight semantic search for knowledge bases."""

from .search_engine import KnowledgeSearch
from .embeddings import ONNXEmbeddings, HuggingFaceEmbeddings
from .indexer import DocumentIndexer
from .types import SearchResult

__version__ = "0.1.0"

__all__ = [
    "KnowledgeSearch",
    "ONNXEmbeddings",
    "HuggingFaceEmbeddings",
    "DocumentIndexer",
    "SearchResult",
]
