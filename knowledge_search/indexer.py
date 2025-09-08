"""Document indexing functionality."""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from .database import VectorDatabase
from .embeddings import HuggingFaceEmbeddings
from .document_processor import DocumentProcessor
from .types import DocumentInfo


class DocumentIndexer:
    """Indexes documents for vector search."""

    def __init__(
        self,
        documents_path: Path,
        db_path: Optional[Path] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.documents_path = Path(documents_path)
        self.db_path = db_path or self.documents_path / "knowledge_vectors.db"

        # Initialize components
        print(f"🏠 Documents path: {self.documents_path}")
        print(f"💾 Database: {self.db_path}")

        print("\n🚀 Initializing embedding model...")
        start_time = time.time()
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        init_time = time.time() - start_time
        print(f"✅ Model initialized in {init_time:.2f}s")

        self.processor = DocumentProcessor()

        # Statistics
        self.stats = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "total_size": 0,
            "start_time": 0.0,
        }

    def _find_documents(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all documents to index."""
        if extensions is None:
            extensions = [".md", ".txt", ".markdown"]

        documents = []
        for ext in extensions:
            pattern = f"**/*{ext}"
            for file_path in self.documents_path.rglob(pattern):
                if not self.processor.should_skip_file(file_path):
                    documents.append(file_path)

        return sorted(documents)

    def _process_document(self, file_path: Path) -> Optional[DocumentInfo]:
        """Process a single document and extract information."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return None

            # Extract frontmatter and clean content
            clean_content, metadata = self.processor.extract_frontmatter(content)
            clean_content = self.processor.clean_content(clean_content)

            if not clean_content.strip():
                return None

            # Calculate file info
            stat = file_path.stat()
            file_hash = self.processor.calculate_file_hash(file_path)

            return DocumentInfo(
                file_path=file_path,
                content=clean_content,
                metadata=metadata,
                file_hash=file_hash,
                modified_time=stat.st_mtime,
                file_size=stat.st_size,
            )

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            self.stats["errors"] += 1
            return None

    def _should_update_document(self, doc_info: DocumentInfo, db: VectorDatabase) -> bool:
        """Check if a document needs to be updated in the database."""
        if not db.document_exists(doc_info.file_path):
            return True

        stored_hash = db.get_document_hash(doc_info.file_path)
        return stored_hash != doc_info.file_hash

    def build_index(self, incremental: bool = True, batch_size: int = 50) -> Dict[str, Any]:
        """Build or update the document index."""
        print(f"🔍 {'Incremental' if incremental else 'Full'} indexing: {self.documents_path}")

        start_time = time.time()
        self.stats["start_time"] = start_time
        documents = self._find_documents()

        print(f"📄 Found {len(documents)} potential documents")

        with VectorDatabase(self.db_path) as db:
            documents_to_process = []

            for doc_path in documents:
                doc_info = self._process_document(doc_path)
                if not doc_info:
                    self.stats["skipped"] += 1
                    continue

                if incremental and not self._should_update_document(doc_info, db):
                    self.stats["skipped"] += 1
                    continue

                documents_to_process.append(doc_info)

            print(f"📋 Processing {len(documents_to_process)} documents...")

            # Process in batches for better memory management
            for i in range(0, len(documents_to_process), batch_size):
                batch = documents_to_process[i : i + batch_size]
                self._process_batch(batch, db)

                print(
                    f"✅ Processed batch {i // batch_size + 1}/{(len(documents_to_process) - 1) // batch_size + 1}"
                )

        # Print final statistics
        elapsed_time = time.time() - start_time

        print(f"\n📊 Indexing complete!")
        print(f"   ✅ Processed: {self.stats['processed']} documents")
        print(f"   ⏭️  Skipped: {self.stats['skipped']} documents")
        print(f"   ❌ Errors: {self.stats['errors']} documents")
        print(f"   💾 Total size: {self.stats['total_size'] / (1024 * 1024):.1f} MB")
        print(f"   ⏱️  Time: {elapsed_time:.1f}s")

        return {
            "processed": self.stats["processed"],
            "skipped": self.stats["skipped"],
            "errors": self.stats["errors"],
            "total_size": self.stats["total_size"],
            "elapsed_time": elapsed_time,
        }

    def _process_batch(self, batch: List[DocumentInfo], db: VectorDatabase) -> None:
        """Process a batch of documents."""
        if not batch:
            return

        try:
            # Generate embeddings for the batch
            texts = [doc.content for doc in batch]
            embeddings = self.embeddings.embed_documents(texts)

            # Insert documents into database
            for doc_info, embedding in zip(batch, embeddings):
                if db.insert_document(doc_info, embedding):
                    self.stats["processed"] += 1
                    self.stats["total_size"] += doc_info.file_size or 0
                else:
                    self.stats["errors"] += 1

        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            self.stats["errors"] += len(batch)
