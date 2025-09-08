"""FastEmbed ONNX Neural Embeddings for Vector Search."""

import numpy as np
from typing import List, Optional, Any
from pathlib import Path


class ONNXEmbeddings:
    """Ultra-fast ONNX-based embeddings using FastEmbed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of the embedding model."""
        if self._initialized:
            return

        try:
            from fastembed import TextEmbedding

            print(f"🔥 Loading ONNX model: {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
            self._initialized = True
            print("✅ ONNX model loaded successfully")
        except ImportError:
            raise ImportError("FastEmbed not installed. Run: pip install fastembed")
        except Exception as e:
            print(f"❌ Failed to load ONNX model: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        self._ensure_initialized()

        if not texts:
            return []

        try:
            # FastEmbed returns generator, convert to list
            assert self._model is not None
            embeddings = list(self._model.embed(texts))
            # Convert numpy arrays to lists for consistency
            return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

        except Exception as e:
            print(f"❌ Document embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        self._ensure_initialized()

        try:
            # FastEmbed expects list, returns generator
            assert self._model is not None
            embeddings = list(self._model.embed([text]))
            if embeddings:
                embedding = embeddings[0]
                return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            else:
                raise RuntimeError("No embedding generated for query")

        except Exception as e:
            print(f"❌ Query embedding failed: {e}")
            raise


# Compatibility wrapper for langchain interface
class HuggingFaceEmbeddings:
    """ONNX-optimized drop-in replacement using FastEmbed."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", **kwargs: Any):
        self.model_name = model_name
        self.embeddings = ONNXEmbeddings(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple documents."""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query."""
        return self.embeddings.embed_query(text)
