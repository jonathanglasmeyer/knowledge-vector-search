# Knowledge Vector Search - Claude Code Instructions

## Project Overview
ONNX-optimized vector search system for knowledge bases. Uses FastEmbed (23MB) + sqlite-vec for semantic search across documents with incremental updates.

## Tech Stack
- **Python 3.12+** 
- **FastEmbed** - ONNX embeddings (CPU-optimized)
- **sqlite-vec** - Vector similarity in SQLite
- **PyYAML** - Frontmatter parsing

## Development Commands
```bash
# Environment
uv sync

# Testing & Quality
python -m pytest
ruff format .
mypy knowledge_search/

# Usage
python -m knowledge_search.embed /path/to/docs
python -m knowledge_search.search "query" --limit 5
python -m knowledge_search.smart_search "query" --limit 10
```

## Key Components
- `KnowledgeSearch` class - main API
- Incremental processing via SHA256 change detection
- Metadata extraction from YAML frontmatter
- ~300ms query performance, ~6KB per document storage

## Architecture Notes
- No external DB required (SQLite-based)
- Optimized for Obsidian/Markdown knowledge bases
- Smart search wrapper with auto-index updates