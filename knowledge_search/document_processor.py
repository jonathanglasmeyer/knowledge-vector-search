"""Document processing utilities for knowledge vector search."""

import hashlib
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


class DocumentProcessor:
    """Processes documents and extracts content and metadata."""

    @staticmethod
    def extract_frontmatter(content: str) -> Tuple[str, Dict[str, Any]]:
        """Extract YAML frontmatter from document content."""
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            try:
                frontmatter = yaml.safe_load(match.group(1)) or {}
                content_without_frontmatter = content[match.end() :]
                return content_without_frontmatter, frontmatter
            except yaml.YAMLError:
                # If YAML parsing fails, treat as regular content
                return content, {}

        return content, {}

    @staticmethod
    def clean_content(content: str) -> str:
        """Clean markdown content for better search quality."""
        # Remove markdown links but keep text
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        # Remove image references
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", content)

        # Remove markdown formatting
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*([^*]+)\*", r"\1", content)  # Italic
        content = re.sub(r"`([^`]+)`", r"\1", content)  # Code

        # Remove headers but keep text
        content = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)

        # Clean up extra whitespace
        content = re.sub(r"\n\s*\n\s*\n", r"\n\n", content)
        content = content.strip()

        return content

    @staticmethod
    def create_preview(content: str, max_length: int = 200) -> str:
        """Create a preview of the document content."""
        clean_content = DocumentProcessor.clean_content(content)

        if len(clean_content) <= max_length:
            return clean_content

        # Find a good breaking point (end of sentence if possible)
        truncated = clean_content[:max_length]

        # Look for sentence endings in the last 50 characters
        sentence_ends = [".", "!", "?"]
        for end_char in sentence_ends:
            last_sentence_end = truncated.rfind(end_char)
            if last_sentence_end > max_length - 50:
                return truncated[: last_sentence_end + 1]

        # If no good sentence end, find last word boundary
        last_space = truncated.rfind(" ")
        if last_space > max_length - 20:
            return truncated[:last_space] + "..."

        return truncated + "..."

    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA256 hash of file content."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
        except (OSError, IOError):
            # If we can't read the file, return a hash of the path
            sha256_hash.update(str(file_path).encode())

        return sha256_hash.hexdigest()

    @staticmethod
    def should_skip_file(file_path: Path, skip_patterns: Optional[List[str]] = None) -> bool:
        """Check if a file should be skipped during indexing."""
        if skip_patterns is None:
            skip_patterns = [
                ".obsidian",
                ".git",
                ".trash",
                "_templates",
                "node_modules",
                "__pycache__",
                ".pytest_cache",
            ]

        # Check if any parent directory matches skip patterns
        for part in file_path.parts:
            if any(pattern in part.lower() for pattern in skip_patterns):
                return True

        # Skip hidden files (starting with .)
        if file_path.name.startswith("."):
            return True

        # Skip non-text files
        if file_path.suffix.lower() not in [".md", ".txt", ".markdown"]:
            return True

        return False
