"""
Magnus RAG System - Embeddings Manager
======================================

Manages embedding generation using sentence-transformers.
Optimized for performance with batching and caching.

Author: AI Engineer Agent
Created: 2025-11-20
"""

import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingsManager:
    """
    Manages embedding generation for Magnus RAG system

    Features:
    - Efficient batch processing
    - Automatic caching to disk
    - GPU acceleration when available
    - Multiple model support
    - Normalization and optimization
    """

    # Default model configurations
    MODELS = {
        "default": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim, fast
        "high_quality": "sentence-transformers/all-mpnet-base-v2",  # 768-dim, better quality
        "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768-dim, multilingual
    }

    def __init__(
        self,
        model_name: str = "default",
        cache_dir: str = "c:/code/Magnus/data/embeddings_cache",
        device: Optional[str] = None,
        batch_size: int = 32,
        enable_cache: bool = True
    ):
        """
        Initialize embeddings manager

        Args:
            model_name: Model identifier or path
            cache_dir: Directory for embedding cache
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            enable_cache: Enable disk caching
        """
        # Resolve model name
        self.model_name = self.MODELS.get(model_name, model_name)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.enable_cache = enable_cache

        # Create cache directory
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Model loaded successfully - "
                f"Dimension: {self.embedding_dimension}, Device: {self.device}"
            )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # Cache statistics
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }

    def encode_single(
        self,
        text: str,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[float]]:
        """
        Encode a single text into an embedding vector

        Args:
            text: Text to encode
            normalize: Normalize the embedding vector
            convert_to_numpy: Convert to numpy array

        Returns:
            Embedding vector
        """
        # Check cache first
        if self.enable_cache:
            cache_key = self._get_cache_key(text)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                self._cache_stats["hits"] += 1
                self._cache_stats["total_requests"] += 1
                return cached

            self._cache_stats["misses"] += 1

        self._cache_stats["total_requests"] += 1

        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_numpy=convert_to_numpy,
                show_progress_bar=False
            )

            # Cache result
            if self.enable_cache:
                self._save_to_cache(cache_key, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            raise

    def encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode multiple texts in batch

        Args:
            texts: List of texts to encode
            normalize: Normalize the embedding vectors
            show_progress: Show progress bar

        Returns:
            Array of embedding vectors
        """
        if not texts:
            return np.array([])

        # Check cache for each text
        embeddings = []
        texts_to_encode = []
        cache_keys = []

        for text in texts:
            if self.enable_cache:
                cache_key = self._get_cache_key(text)
                cached = self._load_from_cache(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    self._cache_stats["hits"] += 1
                    self._cache_stats["total_requests"] += 1
                    continue

            texts_to_encode.append(text)
            cache_keys.append(cache_key if self.enable_cache else None)
            embeddings.append(None)
            self._cache_stats["misses"] += 1
            self._cache_stats["total_requests"] += 1

        # Encode remaining texts
        if texts_to_encode:
            try:
                logger.info(f"Encoding {len(texts_to_encode)} texts in batch")

                # Encode in batches
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=self.batch_size,
                    normalize_embeddings=normalize,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress
                )

                # Cache results and fill in embeddings list
                new_embedding_idx = 0
                for i, embedding in enumerate(embeddings):
                    if embedding is None:
                        embeddings[i] = new_embeddings[new_embedding_idx]

                        # Cache
                        if self.enable_cache and cache_keys[i]:
                            self._save_to_cache(cache_keys[i], new_embeddings[new_embedding_idx])

                        new_embedding_idx += 1

                logger.info(f"Successfully encoded {len(texts_to_encode)} texts")

            except Exception as e:
                logger.error(f"Error encoding batch: {e}")
                raise

        return np.array(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query (optimized for search)

        Args:
            query: Query text

        Returns:
            Query embedding vector
        """
        # For symmetric models, query encoding is same as document encoding
        return self.encode_single(query, normalize=True, convert_to_numpy=True)

    def compute_similarity(
        self,
        embedding1: Union[np.ndarray, List[float]],
        embedding2: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        # Convert to numpy arrays
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)

        # Compute cosine similarity
        similarity = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

        return float(similarity)

    def compute_similarity_batch(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between query and multiple documents

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Array of document embedding vectors

        Returns:
            Array of similarity scores
        """
        query_emb = np.array(query_embedding)

        # Normalize query
        query_norm = query_emb / np.linalg.norm(query_emb)

        # Normalize documents
        doc_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
        doc_normalized = document_embeddings / doc_norms

        # Compute cosine similarities
        similarities = np.dot(doc_normalized, query_norm)

        return similarities

    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text

        Args:
            text: Input text

        Returns:
            Cache key (hash)
        """
        # Include model name in key
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Load embedding from cache

        Args:
            cache_key: Cache key

        Returns:
            Cached embedding or None
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Error loading from cache: {e}")

        return None

    def _save_to_cache(self, cache_key: str, embedding: np.ndarray):
        """
        Save embedding to cache

        Args:
            cache_key: Cache key
            embedding: Embedding vector
        """
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        if not self.enable_cache:
            logger.warning("Cache is not enabled")
            return

        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()

            logger.info(f"Cleared {len(cache_files)} cached embeddings")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dictionary with cache stats
        """
        cache_size = 0
        if self.enable_cache:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            cache_size = sum(f.stat().st_size for f in cache_files)

        hit_rate = 0
        if self._cache_stats["total_requests"] > 0:
            hit_rate = (
                self._cache_stats["hits"] / self._cache_stats["total_requests"] * 100
            )

        return {
            "enabled": self.enable_cache,
            "total_requests": self._cache_stats["total_requests"],
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "cache_directory": str(self.cache_dir)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information

        Returns:
            Dictionary with model info
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_sequence_length": self.model.max_seq_length
        }


if __name__ == "__main__":
    # Test the embeddings manager
    print("Magnus Embeddings Manager - Test")
    print("=" * 60)

    # Initialize
    manager = EmbeddingsManager()

    # Print model info
    info = manager.get_model_info()
    print("\nModel Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test single encoding
    print("\nTesting single text encoding...")
    text = "Cash-secured puts are an options strategy for generating income."
    embedding = manager.encode_single(text)
    print(f"  Text: {text[:50]}...")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  First 5 values: {embedding[:5]}")

    # Test batch encoding
    print("\nTesting batch encoding...")
    texts = [
        "The wheel strategy involves selling cash-secured puts.",
        "Covered calls generate income on stock positions.",
        "Technical analysis uses chart patterns to predict prices."
    ]
    embeddings = manager.encode_batch(texts, show_progress=True)
    print(f"  Encoded {len(texts)} texts")
    print(f"  Embeddings shape: {embeddings.shape}")

    # Test similarity
    print("\nTesting similarity computation...")
    sim = manager.compute_similarity(embeddings[0], embeddings[1])
    print(f"  Similarity between text 1 and 2: {sim:.4f}")

    # Print cache stats
    print("\nCache Stats:")
    stats = manager.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
