"""
Simple RAG (Retrieval Augmented Generation) System for Magnus
Uses ChromaDB for vector storage and sentence-transformers for embeddings
"""

import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    chromadb = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class SimpleRAG:
    """
    Simple Retrieval Augmented Generation system.

    Features:
    - Local vector database (ChromaDB)
    - Local embeddings (sentence-transformers)
    - Document chunking and indexing
    - Semantic search
    - Context injection for LLM queries
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "magnus_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize RAG system.

        Args:
            persist_directory: Directory to store ChromaDB data
            collection_name: Name of the collection in ChromaDB
            embedding_model: Sentence transformer model to use
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "RAG dependencies not installed. Run: "
                "pip install chromadb sentence-transformers"
            )

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Create persist directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Magnus financial knowledge base"}
            )
            logger.info(f"Created new collection: {collection_name}")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        logger.info("RAG system initialized successfully")

    def add_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> List[str]:
        """
        Add a document to the knowledge base.

        Args:
            text: Document text content
            metadata: Optional metadata (source, date, etc.)
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            List of chunk IDs that were added
        """
        if not text or not text.strip():
            logger.warning("Empty document provided, skipping")
            return []

        # Chunk the document
        chunks = self._chunk_text(text)

        if not chunks:
            logger.warning("No chunks generated from document")
            return []

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()

        # Generate IDs
        if doc_id:
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        else:
            import uuid
            base_id = str(uuid.uuid4())[:8]
            chunk_ids = [f"doc_{base_id}_chunk_{i}" for i in range(len(chunks))]

        # Prepare metadata
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk)
            })
            metadatas.append(chunk_metadata)

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=chunk_ids
        )

        logger.info(f"Added document with {len(chunks)} chunks")
        return chunk_ids

    def add_documents_from_directory(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        recursive: bool = True
    ) -> Dict[str, List[str]]:
        """
        Add all documents from a directory.

        Args:
            directory: Directory path
            file_pattern: Glob pattern for files (e.g., "*.txt", "*.pdf")
            recursive: Whether to search recursively

        Returns:
            Dictionary mapping file paths to chunk IDs
        """
        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return {}

        # Find files
        if recursive:
            files = list(path.rglob(file_pattern))
        else:
            files = list(path.glob(file_pattern))

        logger.info(f"Found {len(files)} files matching {file_pattern}")

        results = {}
        for file_path in files:
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Add document
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix
                }

                chunk_ids = self.add_document(
                    text=text,
                    metadata=metadata,
                    doc_id=file_path.stem
                )

                results[str(file_path)] = chunk_ids
                logger.info(f"Added {file_path.name}: {len(chunk_ids)} chunks")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

        return results

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base.

        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List of results with documents, metadata, and scores
        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query provided")
            return []

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])[0].tolist()

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i]
                })

        logger.info(f"Query returned {len(formatted_results)} results")
        return formatted_results

    def get_context_for_query(
        self,
        query_text: str,
        n_results: int = 3,
        max_context_length: int = 2000
    ) -> str:
        """
        Get formatted context for LLM injection.

        Args:
            query_text: Query text
            n_results: Number of results to retrieve
            max_context_length: Maximum context length in characters

        Returns:
            Formatted context string for LLM
        """
        results = self.query(query_text, n_results=n_results)

        if not results:
            return ""

        # Build context string
        context_parts = []
        total_length = 0

        for i, result in enumerate(results):
            source = result['metadata'].get('source', 'Unknown')
            filename = result['metadata'].get('filename', 'Unknown')
            doc_text = result['document']

            # Format context entry
            entry = f"[Source {i+1}: {filename}]\n{doc_text}\n"
            entry_length = len(entry)

            # Check if adding this would exceed max length
            if total_length + entry_length > max_context_length:
                # Add truncated version if possible
                remaining = max_context_length - total_length
                if remaining > 100:  # Only add if we have reasonable space
                    truncated = doc_text[:remaining-50] + "..."
                    entry = f"[Source {i+1}: {filename}]\n{truncated}\n"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_length += entry_length

        context = "\n".join(context_parts)

        # Wrap in clear markers
        formatted_context = f"""
=== RELEVANT KNOWLEDGE BASE CONTEXT ===
{context}
=== END KNOWLEDGE BASE CONTEXT ===
"""

        return formatted_context

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            doc_id: Document ID

        Returns:
            True if successful
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )

            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted document {doc_id} ({len(results['ids'])} chunks)")
                return True
            else:
                logger.warning(f"Document {doc_id} not found")
                return False

        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with stats
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model.get_sentence_embedding_dimension(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Get chunk
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary if not at end
            if end < text_length:
                # Look for sentence endings in last 100 chars
                last_part = chunk[-100:]
                sentence_end = max(
                    last_part.rfind('. '),
                    last_part.rfind('.\n'),
                    last_part.rfind('? '),
                    last_part.rfind('! ')
                )

                if sentence_end > 0:
                    # Adjust chunk to end at sentence
                    chunk = chunk[:len(chunk)-100+sentence_end+2]

            # Add chunk if not empty
            chunk = chunk.strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward (with overlap)
            chunk_length = len(chunk)
            start = start + chunk_length - self.chunk_overlap

            # Prevent infinite loop
            if chunk_length == 0 or start <= 0:
                start += self.chunk_size

        return chunks

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Magnus financial knowledge base"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")


# Singleton instance
_rag_instance: Optional[SimpleRAG] = None


def get_rag() -> Optional[SimpleRAG]:
    """
    Get singleton RAG instance.

    Returns:
        SimpleRAG instance or None if dependencies not available
    """
    global _rag_instance

    if not DEPENDENCIES_AVAILABLE:
        logger.warning("RAG dependencies not available")
        return None

    if _rag_instance is None:
        try:
            _rag_instance = SimpleRAG()
            logger.info("RAG instance created")
        except Exception as e:
            logger.error(f"Error creating RAG instance: {e}")
            return None

    return _rag_instance
