"""Main search engine interface."""

from pathlib import Path
from typing import List, Optional, Dict, Any
from .database import VectorDatabase
from .embeddings import HuggingFaceEmbeddings
from .indexer import DocumentIndexer
from .types import SearchResult


class KnowledgeSearch:
    """Main interface for knowledge vector search."""

    def __init__(
        self,
        documents_path: Path,
        db_path: Optional[Path] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents_path = Path(documents_path)
        self.db_path = db_path or self.documents_path / "knowledge_vectors.db"
        self.model_name = model_name

        # Lazy initialization
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._indexer: Optional[DocumentIndexer] = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialize embeddings."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._embeddings

    @property
    def indexer(self) -> DocumentIndexer:
        """Lazily initialize indexer."""
        if self._indexer is None:
            self._indexer = DocumentIndexer(
                documents_path=self.documents_path, db_path=self.db_path, model_name=self.model_name
            )
        return self._indexer

    def build_index(self, incremental: bool = True, batch_size: int = 50) -> Dict[str, Any]:
        """Build or update the document index."""
        return self.indexer.build_index(incremental=incremental, batch_size=batch_size)

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        """Perform semantic search on the knowledge base."""
        if not query.strip():
            return []

        # Check if database exists
        if not self.db_path.exists():
            print("❌ Database not found. Run build_index() first.")
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search database
            with VectorDatabase(self.db_path) as db:
                results = db.search_similar(query_embedding, limit=limit)

            return results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with VectorDatabase(self.db_path) as db:
                return {
                    "document_count": db.get_document_count(),
                    "database_path": str(self.db_path),
                    "documents_path": str(self.documents_path),
                    "model_name": self.model_name,
                }
        except Exception:
            return {
                "document_count": 0,
                "database_path": str(self.db_path),
                "documents_path": str(self.documents_path),
                "model_name": self.model_name,
            }

    def is_database_stale(self) -> bool:
        """Check if the database needs updating based on file modification times."""
        if not self.db_path.exists():
            return True

        db_mtime = self.db_path.stat().st_mtime

        # Find newest document
        newest_doc_time = 0.0
        for ext in [".md", ".txt", ".markdown"]:
            for doc_path in self.documents_path.rglob(f"**/*{ext}"):
                if not self.indexer.processor.should_skip_file(doc_path):
                    try:
                        doc_mtime = doc_path.stat().st_mtime
                        if doc_mtime > newest_doc_time:
                            newest_doc_time = doc_mtime
                    except OSError:
                        continue

        return newest_doc_time > db_mtime

    def smart_search(
        self, query: str, limit: int = 5, auto_update: bool = True
    ) -> List[SearchResult]:
        """Smart search that automatically updates index if needed."""
        if auto_update and self.is_database_stale():
            print("📊 Database is stale - updating index...")
            self.build_index(incremental=True)

        return self.search(query, limit=limit)
