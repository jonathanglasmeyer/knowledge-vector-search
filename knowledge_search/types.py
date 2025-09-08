"""Type definitions for knowledge vector search."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class SearchResult:
    """A single search result with metadata."""

    file_path: Path
    filename: str
    content: str
    preview: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None
    folder: Optional[str] = None
    file_size: Optional[int] = None

    def __str__(self) -> str:
        return f"{self.filename} (similarity: {self.similarity:.3f})"


@dataclass
class DocumentInfo:
    """Information about a document being indexed."""

    file_path: Path
    content: str
    metadata: Dict[str, Any]
    file_hash: str
    modified_time: float
    file_size: int
