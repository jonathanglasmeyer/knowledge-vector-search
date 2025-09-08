"""Command-line interface for knowledge vector search."""

import argparse
import sys
from pathlib import Path
from typing import Any
from .search_engine import KnowledgeSearch


def cmd_build(args: Any) -> None:
    """Build or update the search index."""
    search = KnowledgeSearch(args.documents_path)
    stats = search.build_index(incremental=not args.full, batch_size=args.batch_size)

    print(f"\n✅ Index {'built' if not args.full else 'updated'} successfully!")


def cmd_search(args: Any) -> None:
    """Perform a search query."""
    search = KnowledgeSearch(args.documents_path)
    results = search.search(args.query, limit=args.limit)

    if not results:
        print(f"No results found for: '{args.query}'")
        return

    print(f"🔍 Search: '{args.query}' ({len(results)} results)")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.filename} (similarity: {result.similarity:.3f})")
        print(f"   📁 {result.file_path}")
        print(f"   📄 {result.preview}")
        print()


def cmd_smart_search(args: Any) -> None:
    """Smart search with automatic index updates."""
    search = KnowledgeSearch(args.documents_path)
    results = search.smart_search(args.query, limit=args.limit, auto_update=not args.no_update)

    if not results:
        print(f"No results found for: '{args.query}'")
        return

    print(f"🔍 Smart search: '{args.query}' ({len(results)} results)")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.filename} (similarity: {result.similarity:.3f})")
        print(f"   📁 {result.file_path}")
        print(f"   📄 {result.preview}")
        print()


def cmd_stats(args: Any) -> None:
    """Show database statistics."""
    search = KnowledgeSearch(args.documents_path)
    stats = search.get_stats()

    print("📊 Knowledge Search Statistics")
    print("=" * 40)
    print(f"Documents indexed: {stats['document_count']}")
    print(f"Documents path: {stats['documents_path']}")
    print(f"Database path: {stats['database_path']}")
    print(f"Model: {stats['model_name']}")

    if search.is_database_stale():
        print("⚠️  Database may be stale - consider running build/update")
    else:
        print("✅ Database is up to date")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Knowledge Vector Search - Semantic search for document collections"
    )

    parser.add_argument("documents_path", type=Path, help="Path to documents directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build or update search index")
    build_parser.add_argument(
        "--full", action="store_true", help="Full rebuild (ignore incremental updates)"
    )
    build_parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for processing documents"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--limit", "-l", type=int, default=5, help="Number of results to return"
    )

    # Smart search command
    smart_parser = subparsers.add_parser("smart-search", help="Smart search with auto-update")
    smart_parser.add_argument("query", help="Search query")
    smart_parser.add_argument(
        "--limit", "-l", type=int, default=5, help="Number of results to return"
    )
    smart_parser.add_argument(
        "--no-update", action="store_true", help="Disable automatic index updates"
    )

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")

    args = parser.parse_args()

    if not args.documents_path.exists():
        print(f"❌ Documents path does not exist: {args.documents_path}")
        sys.exit(1)

    if not args.documents_path.is_dir():
        print(f"❌ Documents path is not a directory: {args.documents_path}")
        sys.exit(1)

    # Route to appropriate command
    commands = {
        "build": cmd_build,
        "search": cmd_search,
        "smart-search": cmd_smart_search,
        "stats": cmd_stats,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except KeyboardInterrupt:
            print("\n⏹️  Operation cancelled")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
