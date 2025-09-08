"""Database operations for vector storage and retrieval."""

import sqlite3
import sqlite_vec  # type: ignore[import-untyped]
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import time
from .types import SearchResult, DocumentInfo


class VectorDatabase:
    """Manages SQLite vector database operations."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_directory()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _ensure_directory(self) -> None:
        """Ensure database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_database(self) -> None:
        """Initialize database connection and schema."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)  # type: ignore[attr-defined]
        self._conn.enable_load_extension(False)

        # Create optimized table schema
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                folder TEXT,
                content TEXT NOT NULL,
                content_preview TEXT,
                metadata TEXT,
                file_size INTEGER,
                modified_time REAL,
                file_hash TEXT,
                embedding BLOB NOT NULL,
                created_at REAL DEFAULT (julianday('now'))
            )
        """)

        # Create indexes for performance
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON documents(file_path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_folder ON documents(folder)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_modified ON documents(modified_time)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON documents(file_hash)")

        self._conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VectorDatabase":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def insert_document(self, doc_info: DocumentInfo, embedding: List[float]) -> bool:
        """Insert or update a document in the database."""
        try:
            embedding_blob = sqlite_vec.serialize_float32(embedding)  # type: ignore[attr-defined]
            metadata_json = json.dumps(doc_info.metadata) if doc_info.metadata else None

            assert self._conn is not None
            self._conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (file_path, file_name, folder, content, content_preview, metadata, 
                 file_size, modified_time, file_hash, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(doc_info.file_path),
                    doc_info.file_path.name,
                    str(doc_info.file_path.parent),
                    doc_info.content[:50000],  # Limit content size
                    doc_info.content[:500],  # Preview
                    metadata_json,
                    doc_info.file_size,
                    doc_info.modified_time,
                    doc_info.file_hash,
                    embedding_blob,
                ),
            )

            assert self._conn is not None
            self._conn.commit()
            return True

        except Exception as e:
            print(f"❌ Failed to insert document {doc_info.file_path}: {e}")
            return False

    def search_similar(self, query_embedding: List[float], limit: int = 5) -> List[SearchResult]:
        """Search for similar documents using vector similarity."""
        try:
            query_blob = sqlite_vec.serialize_float32(query_embedding)  # type: ignore[attr-defined]

            assert self._conn is not None
            results = self._conn.execute(
                """
                SELECT file_path, file_name, content_preview, metadata, folder, file_size,
                       vec_distance_cosine(embedding, ?) as distance
                FROM documents 
                ORDER BY distance
                LIMIT ?
            """,
                (query_blob, limit),
            ).fetchall()

            search_results = []
            for row in results:
                file_path, file_name, preview, metadata_json, folder, file_size, distance = row
                similarity = 1 - distance  # Convert distance to similarity

                metadata = json.loads(metadata_json) if metadata_json else {}

                search_results.append(
                    SearchResult(
                        file_path=Path(file_path),
                        filename=file_name,
                        content="",  # Don't return full content in search results
                        preview=preview or "",
                        similarity=similarity,
                        metadata=metadata,
                        folder=folder,
                        file_size=file_size,
                    )
                )

            return search_results

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def get_document_hash(self, file_path: Path) -> Optional[str]:
        """Get the stored hash for a document."""
        try:
            assert self._conn is not None
            result = self._conn.execute(
                "SELECT file_hash FROM documents WHERE file_path = ?", (str(file_path),)
            ).fetchone()

            return result[0] if result else None

        except Exception:
            return None

    def document_exists(self, file_path: Path) -> bool:
        """Check if a document exists in the database."""
        try:
            assert self._conn is not None
            result = self._conn.execute(
                "SELECT COUNT(*) FROM documents WHERE file_path = ?", (str(file_path),)
            ).fetchone()

            return result[0] > 0 if result else False

        except Exception:
            return False

    def get_document_count(self) -> int:
        """Get total number of documents in the database."""
        try:
            assert self._conn is not None
            result = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def remove_document(self, file_path: Path) -> bool:
        """Remove a document from the database."""
        try:
            assert self._conn is not None
            self._conn.execute("DELETE FROM documents WHERE file_path = ?", (str(file_path),))
            self._conn.commit()
            return True
        except Exception as e:
            print(f"❌ Failed to remove document {file_path}: {e}")
            return False
