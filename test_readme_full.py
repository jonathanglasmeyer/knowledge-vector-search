#!/usr/bin/env python3
"""Test exact Python API from README"""

from pathlib import Path
from knowledge_search import KnowledgeSearch

# Initialize search system
search = KnowledgeSearch(Path("mixed_demo"))

# Index documents (one-time or when documents change)
search.build_index()

# Semantic search
results = search.search("machine learning", limit=5)
for result in results:
    print(f"{result.filename} (similarity: {result.similarity:.3f})")
    print(f"Preview: {result.preview}")
