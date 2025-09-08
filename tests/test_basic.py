"""Basic tests for knowledge vector search functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from knowledge_search import KnowledgeSearch, ONNXEmbeddings
from knowledge_search.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processing functionality."""

    def test_extract_frontmatter(self):
        """Test frontmatter extraction."""
        content = """---
title: Test Document
tags: [test, sample]
---

This is the main content."""

        processor = DocumentProcessor()
        clean_content, metadata = processor.extract_frontmatter(content)

        assert metadata["title"] == "Test Document"
        assert metadata["tags"] == ["test", "sample"]
        assert "This is the main content." in clean_content

    def test_clean_content(self):
        """Test content cleaning."""
        content = """# Header

This is **bold** and *italic* text with `code`.

[Link text](https://example.com)

![Image](image.png)"""

        processor = DocumentProcessor()
        cleaned = processor.clean_content(content)

        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "`" not in cleaned
        assert "[" not in cleaned
        assert "Link text" in cleaned
        assert "Header" in cleaned

    def test_create_preview(self):
        """Test preview creation."""
        content = "This is a long content. " * 20  # Make it longer than preview limit

        processor = DocumentProcessor()
        preview = processor.create_preview(content, max_length=50)

        # Preview should not exceed max_length significantly (accounting for "..." or sentence endings)
        assert len(preview) <= 53  # 50 + "..."

        # Test short content (should return as-is)
        short_content = "Short text."
        short_preview = processor.create_preview(short_content, max_length=50)
        assert short_preview == short_content
        assert not short_preview.endswith("...")

        # Test that long content gets truncated
        assert len(preview) < len(content)  # Should be shorter than original


class TestEmbeddings:
    """Test embedding functionality."""

    @pytest.mark.slow
    def test_onnx_embeddings(self):
        """Test ONNX embeddings generation."""
        embeddings = ONNXEmbeddings()

        # Test single query
        query_emb = embeddings.embed_query("test query")
        assert isinstance(query_emb, list)
        assert len(query_emb) > 0
        assert all(isinstance(x, float) for x in query_emb)

        # Test multiple documents
        docs = ["document one", "document two"]
        doc_embs = embeddings.embed_documents(docs)
        assert len(doc_embs) == 2
        assert all(isinstance(emb, list) for emb in doc_embs)


class TestKnowledgeSearch:
    """Test main search functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test documents
        (self.temp_dir / "doc1.md").write_text("""---
title: Machine Learning
tags: [ai, ml]
---

This document discusses machine learning algorithms and neural networks.""")

        (self.temp_dir / "doc2.md").write_text("""---
title: Python Programming
tags: [programming, python]
---

Python is a versatile programming language used for data science.""")

        self.search = KnowledgeSearch(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    def test_build_and_search(self):
        """Test building index and performing search."""
        # Build index
        stats = self.search.build_index()
        assert stats["processed"] == 2
        assert stats["errors"] == 0

        # Test search
        results = self.search.search("machine learning", limit=2)
        assert len(results) > 0
        assert any("Machine Learning" in result.filename for result in results)

    def test_stats(self):
        """Test getting statistics."""
        stats = self.search.get_stats()
        assert "document_count" in stats
        assert "database_path" in stats
        assert "documents_path" in stats
