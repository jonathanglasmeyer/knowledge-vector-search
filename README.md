# Knowledge Vector Search

Find relevant information in your documents using natural language queries - no more keyword matching or manual browsing through hundreds of files. Built for speed and efficiency, this lightweight system searches through 500+ documents in under 300ms while using only 100MB of memory. Perfect for knowledge bases, research collections, and personal document libraries.

## Features

- **ONNX-optimized embeddings** - FastEmbed with 384-dimensional vectors (23MB vs 6.8GB PyTorch models, CPU-optimized)
- **Fast semantic search** - ~300ms queries across 500+ documents using sqlite-vec cosine similarity
- **Incremental updates** - SHA256-based change detection, only reprocess modified files
- **Smart search wrapper** - automatic index updates when documents change
- **SQLite-based storage** - zero external dependencies, portable, ~6KB per document
- **Metadata extraction** - YAML frontmatter and document properties (ideal for Obsidian and other Markdown knowledge bases)
- **Simple Python API** and command-line tools

**Processing Flow:** Markdown → text extraction + frontmatter parsing → FastEmbed embedding → sqlite-vec storage → cosine similarity search

## Try It Yourself

Want to see semantic search in action? We've included both tech docs and recipes to show how it finds the right content type based on context:

```bash
# Clone the repo and try with our mixed demo dataset
git clone https://github.com/your-username/knowledge-vector-search.git
cd knowledge-vector-search
uv sync

# Index both tech docs AND recipes together
uv run python -m knowledge_search mixed_demo build

# Recipe query → finds only recipes, not tech docs
uv run python -m knowledge_search mixed_demo search "simple Italian pasta" --limit 3
# → Results: Spaghetti Aglio e Olio (0.616 similarity, perfect match)

# Tech query → finds only tech content, not recipes
uv run python -m knowledge_search mixed_demo search "programming best practices" --limit 3
# → Results: Programming Best Practices (0.627), AI Tools (no recipes mixed in)

# Breakfast query → finds morning foods, not programming concepts
uv run python -m knowledge_search mixed_demo search "healthy morning drink" --limit 3
# → Results: Overnight Oats, Green Smoothie (perfect breakfast matches)
```

**Notice:** Even with mixed content types, it finds semantically relevant results without cross-contamination!

## Quick Start

### Installation

**Requirements**: Python 3.12+

**Recommended: Using [uv](https://docs.astral.sh/uv/) (fast, reliable dependency management)**

```bash
# Clone and setup
git clone https://github.com/your-username/knowledge-vector-search.git
cd knowledge-vector-search
uv sync
```

**Alternative: Using pip**

```bash
git clone https://github.com/your-username/knowledge-vector-search.git
cd knowledge-vector-search
pip install -e .
```

### Basic Usage

```bash
# Index your documents
uv run python -m knowledge_search /path/to/documents build

# Search
uv run pycthon -m knowledge_search /path/to/documents search "machine learning concepts" --limit 5

# Smart search (auto-updates index if needed)
uv run python -m knowledge_search /path/to/documents smart-search "AI tools" --limit 10

# Show statistics
uv run python -m knowledge_search /path/to/documents stats
```

## Search Behavior

**Regular Search**: Searches existing index, returns document previews (~500 chars) with similarity scores.

**Smart Search**: Automatically updates index if documents changed, then performs regular search.

**Results Format**: Each result includes filename, file path, content preview, similarity score, and metadata - but not the full document content.

## Python API

```python
from knowledge_search import KnowledgeSearch

# Initialize search system
search = KnowledgeSearch("/path/to/documents")

# Index documents (one-time or when documents change)
search.build_index()

# Semantic search
results = search.search("machine learning", limit=5)
for result in results:
    print(f"{result.filename} (similarity: {result.similarity:.3f})")
    print(f"Preview: {result.preview}")
```

## Claude Code Integration

The real power comes from integrating with Claude Code to create a natural language interface to your knowledge base. Instead of manually searching through files, ask Claude natural language questions and it will use vector search to find the right documents.

**Example workflow:**
1. You ask: *"How do we handle user authentication?"*
2. Claude Code runs: `uv run python -m knowledge_search docs/ search "authentication middleware setup"`
3. Claude gets relevant file previews and provides accurate, project-specific answers

### CLAUDE.md Setup

Add this to your project's `CLAUDE.md`:

```markdown
# Knowledge Search Integration

Use semantic search to find relevant information from our documentation:

```bash
# Search for implementation patterns
uv run python -m knowledge_search docs/ search "authentication middleware setup" --limit 3

# Find API documentation
uv run python -m knowledge_search docs/ search "database connection pooling" --limit 5

# Look for troubleshooting guides
uv run python -m knowledge_search docs/ smart-search "performance optimization tips" --limit 3
```

Now Claude Code can automatically find relevant files when you ask:
- "How do we handle user authentication?"
- "Show me examples of database optimization"
- "What's our error handling strategy?"
```

### Custom Claude Command

Create `~/.claude/commands/search-docs.md`:

```markdown
# Search Project Documentation

```bash
uv run python -m knowledge_search . smart-search "{{query}}" --limit 5
```

Usage: Ask Claude to search your docs with natural language queries.


### Development

```bash
# Setup development environment
uv sync

# Run tests
uv run python -m pytest

# Format code
uv run ruff format .

# Type check
uv run mypy knowledge_search/
```

## License

MIT License - see LICENSE file for details.
