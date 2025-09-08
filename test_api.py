#!/usr/bin/env python3
"""Test the Python API of knowledge vector search."""

from pathlib import Path
from knowledge_search import KnowledgeSearch


def test_api():
    """Test the main API functionality."""
    print("🧪 Testing Knowledge Search API...")

    # Initialize search system
    search = KnowledgeSearch(Path("mixed_demo"))

    # Get statistics
    stats = search.get_stats()
    print(f"📊 Database stats: {stats['document_count']} documents indexed")

    # Perform search
    print("\n🔍 Searching for 'machine learning'...")
    results = search.search("machine learning", limit=2)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.filename} (similarity: {result.similarity:.3f})")
        print(f"   Path: {result.file_path}")
        print(f"   Preview: {result.preview[:100]}...")
        if result.metadata:
            print(f"   Tags: {result.metadata.get('tags', 'None')}")
        print()

    print("✅ API test completed successfully!")


if __name__ == "__main__":
    test_api()
