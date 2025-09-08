#!/usr/bin/env python3
"""Test the exact Python API examples from README."""

from pathlib import Path
from knowledge_search import KnowledgeSearch

# Initialize search system
search = KnowledgeSearch(Path("mixed_demo"))

# Index documents (one-time or when documents change)
stats = search.build_index()
print(f"Build stats: {stats}")

# Semantic search
results = search.search("machine learning", limit=5)
for result in results:
    print(f"{result.filename} (similarity: {result.similarity:.3f})")
    print(f"Preview: {result.preview}")
