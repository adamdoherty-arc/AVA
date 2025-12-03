"""
Magnus RAG System - Knowledge Base
==================================

Central knowledge base for Magnus RAG system.
Provides high-level interface for querying and managing documents.

Author: AI Engineer Agent
Created: 2025-11-20
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

from src.rag.vector_store import VectorStore
from src.rag.embeddings_manager import EmbeddingsManager
from src.rag.document_processor import DocumentProcessor, DocumentType, ProcessedDocument

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    document_title: str


@dataclass
class RetrievalContext:
    """Context built from retrieved chunks"""
    context_text: str
    sources: List[SearchResult]
    total_tokens: int
    retrieval_time_ms: float


class MagnusKnowledgeBase:
    """
    Central knowledge base for Magnus RAG system

    Features:
    - Document ingestion and management
    - Semantic search with metadata filtering
    - Context building for LLM prompts
    - Caching for performance
    - Statistics and monitoring
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        document_processor: Optional[DocumentProcessor] = None,
        enable_caching: bool = True
    ):
        """
        Initialize Magnus Knowledge Base

        Args:
            vector_store: Vector store instance
            embeddings_manager: Embeddings manager instance
            document_processor: Document processor instance
            enable_caching: Enable caching for queries
        """
        # Initialize components
        self.vector_store = vector_store or VectorStore()
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self.document_processor = document_processor or DocumentProcessor()
        self.enable_caching = enable_caching

        # Simple in-memory cache
        self._query_cache: Dict[str, RetrievalContext] = {}
        self._cache_max_size = 100

        # Statistics
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "total_documents_indexed": 0,
            "total_chunks_indexed": 0
        }

        logger.info("Magnus Knowledge Base initialized")

    def add_document(
        self,
        file_path: str,
        document_type: DocumentType = DocumentType.GENERAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a document to the knowledge base

        Args:
            file_path: Path to document file
            document_type: Type of document
            metadata: Additional metadata

        Returns:
            Document ID
        """
        logger.info(f"Adding document to knowledge base: {file_path}")

        try:
            # Process document
            processed_doc = self.document_processor.process_document(
                file_path=file_path,
                document_type=document_type,
                metadata=metadata
            )

            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in processed_doc.chunks]
            embeddings = self.embeddings_manager.encode_batch(
                texts=chunk_texts,
                show_progress=True
            )

            # Prepare data for vector store
            chunk_ids = [chunk.chunk_id for chunk in processed_doc.chunks]
            chunk_metadatas = []

            for chunk in processed_doc.chunks:
                # Add document-level metadata to each chunk
                chunk_meta = {
                    **chunk.metadata,
                    "document_id": processed_doc.document_id,
                    "document_title": processed_doc.title,
                    "document_type": processed_doc.document_type.value,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": processed_doc.total_chunks
                }
                chunk_metadatas.append(chunk_meta)

            # Add to vector store
            self.vector_store.add_documents(
                documents=chunk_texts,
                embeddings=embeddings.tolist(),
                metadatas=chunk_metadatas,
                ids=chunk_ids
            )

            # Update statistics
            self._stats["total_documents_indexed"] += 1
            self._stats["total_chunks_indexed"] += len(processed_doc.chunks)

            logger.info(
                f"Successfully added document: {processed_doc.title} "
                f"({len(processed_doc.chunks)} chunks)"
            )

            return processed_doc.document_id

        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise

    def add_documents_from_directory(
        self,
        directory_path: str,
        document_type: DocumentType = DocumentType.GENERAL,
        recursive: bool = True
    ) -> List[str]:
        """
        Add all documents from a directory

        Args:
            directory_path: Path to directory
            document_type: Type of documents
            recursive: Search subdirectories

        Returns:
            List of document IDs
        """
        logger.info(f"Adding documents from directory: {directory_path}")

        # Process all documents in directory
        processed_docs = self.document_processor.batch_process_directory(
            directory_path=directory_path,
            document_type=document_type,
            recursive=recursive
        )

        document_ids = []

        for doc in processed_docs:
            try:
                # Generate embeddings
                chunk_texts = [chunk.text for chunk in doc.chunks]
                embeddings = self.embeddings_manager.encode_batch(chunk_texts)

                # Prepare data
                chunk_ids = [chunk.chunk_id for chunk in doc.chunks]
                chunk_metadatas = [
                    {
                        **chunk.metadata,
                        "document_id": doc.document_id,
                        "document_title": doc.title,
                        "document_type": doc.document_type.value
                    }
                    for chunk in doc.chunks
                ]

                # Add to vector store
                self.vector_store.add_documents(
                    documents=chunk_texts,
                    embeddings=embeddings.tolist(),
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )

                document_ids.append(doc.document_id)

                # Update statistics
                self._stats["total_documents_indexed"] += 1
                self._stats["total_chunks_indexed"] += len(doc.chunks)

            except Exception as e:
                logger.error(f"Error adding document {doc.title}: {e}")

        logger.info(f"Successfully added {len(document_ids)} documents")
        return document_ids

    def search(
        self,
        query: str,
        n_results: int = 5,
        document_type: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search the knowledge base

        Args:
            query: Search query
            n_results: Number of results to return
            document_type: Filter by document type
            metadata_filters: Additional metadata filters

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()

        # Generate query embedding
        query_embedding = self.embeddings_manager.encode_query(query)

        # Build filters
        where_filter = None
        if document_type or metadata_filters:
            where_filter = {}
            if document_type:
                where_filter["document_type"] = document_type
            if metadata_filters:
                where_filter.update(metadata_filters)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding.tolist(),
            n_results=n_results,
            where=where_filter
        )

        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results["ids"])):
            result = SearchResult(
                chunk_id=results["ids"][i],
                text=results["documents"][i],
                score=results["scores"][i],
                metadata=results["metadatas"][i],
                document_id=results["metadatas"][i].get("document_id", "unknown"),
                document_title=results["metadatas"][i].get("document_title", "Unknown")
            )
            search_results.append(result)

        # Update statistics
        self._stats["total_queries"] += 1

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Search completed in {elapsed_ms:.0f}ms - Found {len(search_results)} results"
        )

        return search_results

    def build_context(
        self,
        query: str,
        n_results: int = 5,
        max_tokens: int = 2000,
        document_type: Optional[str] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> RetrievalContext:
        """
        Build context from retrieved chunks for LLM

        Args:
            query: Search query
            n_results: Number of chunks to retrieve
            max_tokens: Maximum tokens in context
            document_type: Filter by document type
            metadata_filters: Additional metadata filters
            use_cache: Use cached results if available

        Returns:
            RetrievalContext object
        """
        start_time = time.time()

        # Check cache
        cache_key = f"{query}:{document_type}:{n_results}"
        if use_cache and self.enable_caching and cache_key in self._query_cache:
            logger.info("Cache hit for query")
            self._stats["cache_hits"] += 1
            return self._query_cache[cache_key]

        # Search knowledge base
        search_results = self.search(
            query=query,
            n_results=n_results,
            document_type=document_type,
            metadata_filters=metadata_filters
        )

        # Build context text
        context_parts = []
        total_tokens = 0
        included_results = []

        for result in search_results:
            # Estimate tokens (rough: 4 chars per token)
            chunk_tokens = len(result.text) // 4

            # Check if adding this chunk would exceed max tokens
            if total_tokens + chunk_tokens > max_tokens:
                logger.info(
                    f"Reached max tokens ({max_tokens}), stopping at {len(included_results)} chunks"
                )
                break

            # Add chunk to context
            context_parts.append(
                f"[Source: {result.document_title}]\n{result.text}\n"
            )
            total_tokens += chunk_tokens
            included_results.append(result)

        # Combine context
        context_text = "\n---\n".join(context_parts)

        # Create retrieval context
        elapsed_ms = (time.time() - start_time) * 1000

        retrieval_context = RetrievalContext(
            context_text=context_text,
            sources=included_results,
            total_tokens=total_tokens,
            retrieval_time_ms=elapsed_ms
        )

        # Cache result
        if use_cache and self.enable_caching:
            # Manage cache size
            if len(self._query_cache) >= self._cache_max_size:
                # Remove oldest entry
                self._query_cache.pop(next(iter(self._query_cache)))

            self._query_cache[cache_key] = retrieval_context

        logger.info(
            f"Built context with {len(included_results)} chunks "
            f"({total_tokens} tokens) in {elapsed_ms:.0f}ms"
        )

        return retrieval_context

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the knowledge base

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful
        """
        try:
            # Delete all chunks with this document_id
            self.vector_store.delete_by_filter({"document_id": document_id})
            logger.info(f"Deleted document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear the query cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics

        Returns:
            Dictionary with statistics
        """
        vector_store_stats = self.vector_store.get_stats()
        embeddings_stats = self.embeddings_manager.get_cache_stats()

        cache_hit_rate = 0
        if self._stats["total_queries"] > 0:
            cache_hit_rate = (
                self._stats["cache_hits"] / self._stats["total_queries"] * 100
            )

        return {
            **self._stats,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cache_size": len(self._query_cache),
            "vector_store": vector_store_stats,
            "embeddings": embeddings_stats
        }

    def health_check(self) -> Tuple[bool, str]:
        """
        Check knowledge base health

        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            # Check vector store
            vs_healthy, vs_message = self.vector_store.health_check()
            if not vs_healthy:
                return False, f"Vector store unhealthy: {vs_message}"

            # Check embeddings manager
            model_info = self.embeddings_manager.get_model_info()
            if not model_info:
                return False, "Embeddings manager not initialized"

            # Check document count
            doc_count = self.vector_store.count()

            return True, f"Healthy - {doc_count} chunks indexed"

        except Exception as e:
            return False, f"Health check failed: {str(e)}"


if __name__ == "__main__":
    # Test the knowledge base
    print("Magnus Knowledge Base - Test")
    print("=" * 60)

    # Initialize
    kb = MagnusKnowledgeBase()

    # Health check
    is_healthy, message = kb.health_check()
    print(f"\nHealth Check: {'✓' if is_healthy else '✗'} {message}")

    # Print stats
    print("\nKnowledge Base Stats:")
    stats = kb.get_stats()
    print(f"  Total Documents: {stats['total_documents_indexed']}")
    print(f"  Total Chunks: {stats['total_chunks_indexed']}")
    print(f"  Total Queries: {stats['total_queries']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate_percent']}%")

    print("\nVector Store:")
    for key, value in stats['vector_store'].items():
        print(f"  {key}: {value}")

    print("\nEmbeddings:")
    for key, value in stats['embeddings'].items():
        print(f"  {key}: {value}")
