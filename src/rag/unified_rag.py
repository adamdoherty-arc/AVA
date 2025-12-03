"""
Unified RAG Interface for Magnus
=================================

Single entry point for all RAG operations, resolving the SimpleRAG vs RAGService conflict.

This module provides:
- Unified query interface
- Document management (add, delete, list)
- Context extraction for LLM prompt injection
- Consistent API across all Magnus components

Usage:
    from src.rag.unified_rag import get_unified_rag, UnifiedRAG

    rag = get_unified_rag()

    # Query the knowledge base
    response = rag.query("What is the wheel strategy?")
    print(response.answer)

    # Get context for prompt injection
    context = rag.get_context_for_prompt("options trading")

    # Add documents
    rag.add_document("Document content here", metadata={"source": "guide.md"})
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Unified RAG response format.

    Provides consistent structure regardless of underlying implementation.
    """
    answer: str
    context: str  # Formatted context for prompt injection
    sources: List[Dict[str, Any]]
    confidence: float
    retrieval_method: str
    query_complexity: str
    processing_time_ms: float
    was_cached: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @property
    def has_results(self) -> bool:
        """Check if meaningful results were found"""
        return self.confidence > 0.3 and len(self.sources) > 0


@dataclass
class DocumentInfo:
    """Information about an indexed document"""
    doc_id: str
    filename: str
    source: str
    category: str
    chunk_count: int
    added_date: str
    metadata: Dict[str, Any]


class UnifiedRAG:
    """
    Unified RAG interface providing a single entry point for all RAG operations.

    This class wraps RAGService (the advanced implementation) and provides:
    - Simplified query interface
    - Document management
    - Context extraction for LLM prompts
    - Consistent error handling
    - Integration with Magnus Local LLM

    Architecture:
        UnifiedRAG
            └── RAGService (production-ready with hybrid search, reranking)
                    └── ChromaDB (vector storage)
                    └── Sentence Transformers (embeddings)
                    └── Cross-Encoder (reranking)
                    └── Magnus Local LLM (response generation)
    """

    _instance: Optional['UnifiedRAG'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern for consistent state across the application"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        collection_name: str = "magnus_knowledge",
        min_confidence: float = 0.3,
        default_n_results: int = 5,
        cache_enabled: bool = True
    ):
        """
        Initialize UnifiedRAG.

        Args:
            collection_name: ChromaDB collection name
            min_confidence: Minimum confidence threshold for valid results
            default_n_results: Default number of results to retrieve
            cache_enabled: Whether to enable query caching
        """
        if self._initialized:
            return

        logger.info("Initializing UnifiedRAG...")

        self.collection_name = collection_name
        self.min_confidence = min_confidence
        self.default_n_results = default_n_results
        self.cache_enabled = cache_enabled

        # Initialize the underlying RAGService
        try:
            from src.rag.rag_service import RAGService
            self._service = RAGService(
                collection_name=collection_name,
                min_confidence_threshold=min_confidence
            )
            logger.info("RAGService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {e}")
            self._service = None

        # Track document inventory
        self._document_registry: Dict[str, DocumentInfo] = {}

        self._initialized = True
        logger.info("UnifiedRAG initialized successfully")

    @property
    def is_available(self) -> bool:
        """Check if RAG service is available"""
        return self._service is not None

    def query(
        self,
        question: str,
        n_results: int = None,
        min_confidence: float = None,
        use_cache: bool = True,
        include_context: bool = True
    ) -> RAGResponse:
        """
        Query the knowledge base.

        Args:
            question: The question to answer
            n_results: Number of results to retrieve (default: 5)
            min_confidence: Minimum confidence threshold (default: 0.3)
            use_cache: Whether to use cached results
            include_context: Whether to build context string for prompts

        Returns:
            RAGResponse with answer, sources, and metadata
        """
        if not self.is_available:
            logger.warning("RAG service not available")
            return RAGResponse(
                answer="Knowledge base is not available.",
                context="",
                sources=[],
                confidence=0.0,
                retrieval_method="none",
                query_complexity="unknown",
                processing_time_ms=0.0,
                was_cached=False
            )

        if not question or not question.strip():
            logger.warning("Empty query provided")
            return RAGResponse(
                answer="Please provide a question.",
                context="",
                sources=[],
                confidence=0.0,
                retrieval_method="none",
                query_complexity="unknown",
                processing_time_ms=0.0,
                was_cached=False
            )

        try:
            # Use RAGService for the query
            result = self._service.query(
                question=question,
                use_cache=use_cache and self.cache_enabled
            )

            # Build context string for prompt injection
            context = ""
            if include_context and result.sources:
                context = self._build_context(result.sources, question)

            return RAGResponse(
                answer=result.answer,
                context=context,
                sources=result.sources,
                confidence=result.confidence,
                retrieval_method=result.retrieval_method,
                query_complexity=result.query_complexity,
                processing_time_ms=result.processing_time_ms,
                was_cached=result.was_cached
            )

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return RAGResponse(
                answer=f"Error querying knowledge base: {str(e)}",
                context="",
                sources=[],
                confidence=0.0,
                retrieval_method="error",
                query_complexity="unknown",
                processing_time_ms=0.0,
                was_cached=False
            )

    def get_context_for_prompt(
        self,
        question: str,
        max_tokens: int = 2000,
        n_results: int = 5
    ) -> str:
        """
        Get formatted context string for LLM prompt injection.

        This is the primary method for integrating RAG with the chatbot.
        The returned context should be inserted into the LLM prompt.

        Args:
            question: The question/query to retrieve context for
            max_tokens: Maximum context length in tokens (approx 4 chars/token)
            n_results: Number of sources to include

        Returns:
            Formatted context string ready for prompt injection
        """
        response = self.query(question, n_results=n_results, include_context=True)

        if not response.has_results:
            return ""

        # Truncate if needed
        max_chars = max_tokens * 4
        context = response.context

        if len(context) > max_chars:
            context = context[:max_chars] + "\n... [context truncated]"

        return context

    def _build_context(
        self,
        sources: List[Dict[str, Any]],
        query: str,
        max_chars: int = 8000
    ) -> str:
        """
        Build formatted context string from retrieved sources.

        Args:
            sources: List of source documents
            query: Original query (for context)
            max_chars: Maximum context length

        Returns:
            Formatted context string
        """
        if not sources:
            return ""

        context_parts = []
        total_chars = 0

        for i, source in enumerate(sources):
            content = source.get('content', '')
            metadata = source.get('metadata', {})
            score = source.get('score', 0)

            # Get source identifier
            source_name = metadata.get('filename', metadata.get('source', f'Source {i+1}'))
            category = metadata.get('category', 'general')

            # Build entry
            entry = f"[{source_name} | {category} | relevance: {score:.2f}]\n{content}\n"

            if total_chars + len(entry) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 100:
                    truncated_content = content[:remaining - 100] + "..."
                    entry = f"[{source_name} | {category}]\n{truncated_content}\n"
                    context_parts.append(entry)
                break

            context_parts.append(entry)
            total_chars += len(entry)

        context = "\n---\n".join(context_parts)

        # Wrap in clear markers for LLM
        formatted = f"""
=== RELEVANT KNOWLEDGE BASE CONTEXT ===
Query: {query}

{context}
=== END KNOWLEDGE BASE CONTEXT ===
"""
        return formatted.strip()

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> int:
        """
        Add a document to the knowledge base.

        Args:
            content: Document text content
            metadata: Document metadata (source, category, etc.)
            doc_id: Optional document ID (auto-generated if not provided)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks

        Returns:
            Number of chunks added
        """
        if not self.is_available:
            logger.error("RAG service not available")
            return 0

        if not content or not content.strip():
            logger.warning("Empty document content")
            return 0

        # Generate doc_id if not provided
        if not doc_id:
            doc_id = hashlib.md5(content[:500].encode()).hexdigest()[:12]

        # Prepare metadata
        meta = metadata.copy() if metadata else {}
        meta['doc_id'] = doc_id
        meta['added_date'] = datetime.now().isoformat()

        # Chunk the document
        chunks = self._chunk_text(content, chunk_size, chunk_overlap)

        if not chunks:
            logger.warning("No chunks generated from document")
            return 0

        # Prepare for ChromaDB
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(chunks):
            chunk_meta = meta.copy()
            chunk_meta['chunk_index'] = i
            chunk_meta['total_chunks'] = len(chunks)

            documents.append(chunk)
            metadatas.append(chunk_meta)
            ids.append(f"{doc_id}_chunk_{i}")

        # Add to RAGService
        try:
            self._service.add_documents(documents, metadatas, ids)

            # Register document
            self._document_registry[doc_id] = DocumentInfo(
                doc_id=doc_id,
                filename=meta.get('filename', 'unknown'),
                source=meta.get('source', 'direct'),
                category=meta.get('category', 'general'),
                chunk_count=len(chunks),
                added_date=meta['added_date'],
                metadata=meta
            )

            logger.info(f"Added document {doc_id}: {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return 0

    def add_documents_from_directory(
        self,
        directory: str,
        file_patterns: List[str] = None,
        recursive: bool = True,
        category: str = None
    ) -> Dict[str, int]:
        """
        Add all documents from a directory.

        Args:
            directory: Directory path
            file_patterns: File patterns to match (default: *.md, *.txt)
            recursive: Search recursively
            category: Category to assign to all documents

        Returns:
            Dictionary mapping filenames to chunk counts
        """
        if file_patterns is None:
            file_patterns = ['*.md', '*.txt', '*.rst']

        path = Path(directory)
        if not path.exists():
            logger.error(f"Directory not found: {directory}")
            return {}

        results = {}

        for pattern in file_patterns:
            files = list(path.rglob(pattern) if recursive else path.glob(pattern))

            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    metadata = {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_type': file_path.suffix,
                        'category': category or self._infer_category(file_path)
                    }

                    chunk_count = self.add_document(
                        content=content,
                        metadata=metadata,
                        doc_id=file_path.stem
                    )

                    results[file_path.name] = chunk_count
                    logger.info(f"Indexed {file_path.name}: {chunk_count} chunks")

                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    results[file_path.name] = 0

        return results

    def _infer_category(self, file_path: Path) -> str:
        """Infer document category from path"""
        path_str = str(file_path).lower()

        if 'options' in path_str:
            return 'options_education'
        elif 'risk' in path_str:
            return 'risk_management'
        elif 'technical' in path_str:
            return 'technical_analysis'
        elif 'platform' in path_str or 'magnus' in path_str:
            return 'platform_docs'
        elif 'sports' in path_str or 'betting' in path_str:
            return 'sports_betting'
        else:
            return 'general'

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> List[str]:
        """
        Split text into semantic chunks.

        Tries to break at paragraph/sentence boundaries.

        Args:
            text: Text to chunk
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # First, split by paragraphs (double newlines)
        paragraphs = text.split('\n\n')

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > chunk_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Start new chunk with overlap from previous
                    if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                        # Find a good break point for overlap
                        overlap_text = current_chunk[-chunk_overlap:]
                        # Try to start at sentence boundary
                        sentence_end = overlap_text.find('. ')
                        if sentence_end > 0:
                            overlap_text = overlap_text[sentence_end + 2:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                else:
                    # Paragraph itself is too long, need to split it
                    if len(para) > chunk_size:
                        # Split long paragraph by sentences
                        sentences = para.replace('. ', '.|').split('|')
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 > chunk_size:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence
                            else:
                                current_chunk += " " + sentence if current_chunk else sentence
                    else:
                        current_chunk = para
            else:
                # Add paragraph to current chunk
                current_chunk += "\n\n" + para if current_chunk else para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def list_documents(self) -> List[DocumentInfo]:
        """
        List all indexed documents.

        Returns:
            List of DocumentInfo objects
        """
        return list(self._document_registry.values())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with stats
        """
        if not self.is_available:
            return {'error': 'RAG service not available'}

        try:
            service_stats = self._service.get_collection_stats()

            return {
                'total_documents': len(self._document_registry),
                'total_chunks': service_stats.get('total_documents', 0),
                'collection_name': self.collection_name,
                'embedding_dimension': service_stats.get('embedding_model', 768),
                'cache_size': service_stats.get('cache_size', 0),
                'metrics': service_stats.get('metrics', {}),
                'categories': self._get_category_stats()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}

    def _get_category_stats(self) -> Dict[str, int]:
        """Get document counts by category"""
        categories = {}
        for doc in self._document_registry.values():
            cat = doc.category
            categories[cat] = categories.get(cat, 0) + 1
        return categories

    def clear_cache(self) -> None:
        """Clear query cache"""
        if self.is_available:
            self._service.clear_cache()
            logger.info("Cache cleared")

    def clear_all(self) -> None:
        """Clear all documents and cache (use with caution)"""
        if self.is_available:
            try:
                # Clear the ChromaDB collection
                self._service.chroma_client.delete_collection(self.collection_name)
                self._service.collection = self._service.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self._document_registry = {}
                self._service.clear_cache()
                logger.info("All documents and cache cleared")
            except Exception as e:
                logger.error(f"Error clearing knowledge base: {e}")

    def clear_collection(self) -> None:
        """Alias for clear_all - clears all documents from knowledge base"""
        self.clear_all()


# Singleton instance
_unified_rag_instance: Optional[UnifiedRAG] = None


def get_unified_rag() -> UnifiedRAG:
    """
    Get singleton UnifiedRAG instance.

    This is the recommended way to access the RAG system.

    Returns:
        UnifiedRAG instance
    """
    global _unified_rag_instance

    if _unified_rag_instance is None:
        _unified_rag_instance = UnifiedRAG()

    return _unified_rag_instance


# Convenience function for quick context retrieval
def get_rag_context(question: str, max_tokens: int = 2000) -> str:
    """
    Quick helper to get RAG context for a question.

    Args:
        question: The question to get context for
        max_tokens: Maximum context length

    Returns:
        Formatted context string for LLM prompt injection
    """
    rag = get_unified_rag()
    return rag.get_context_for_prompt(question, max_tokens)


if __name__ == "__main__":
    # Test the UnifiedRAG
    print("Testing UnifiedRAG Interface\n")

    rag = get_unified_rag()

    # Check stats
    print("Initial Stats:")
    print(rag.get_stats())
    print()

    # Add test document
    test_doc = """
    # The Wheel Strategy

    The wheel strategy is an options trading strategy that combines:
    1. Selling cash-secured puts (CSPs)
    2. Getting assigned shares if the put expires ITM
    3. Selling covered calls on the assigned shares
    4. Repeat the cycle

    ## Benefits
    - Generate consistent income from premium
    - Buy stocks at a discount
    - Lower cost basis through premium collection

    ## Risks
    - Stock may drop significantly after assignment
    - Opportunity cost if stock rises sharply
    - Requires capital to secure puts
    """

    chunks = rag.add_document(
        content=test_doc,
        metadata={
            'filename': 'wheel_strategy.md',
            'category': 'options_education',
            'source': 'test'
        },
        doc_id='wheel_test'
    )

    print(f"Added test document: {chunks} chunks")
    print()

    # Test query
    print("Testing query: 'What is the wheel strategy?'")
    result = rag.query("What is the wheel strategy?")
    print(f"Answer: {result.answer[:200]}...")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Sources: {len(result.sources)}")
    print()

    # Test context extraction
    print("Testing context extraction:")
    context = rag.get_context_for_prompt("wheel strategy risks", max_tokens=500)
    print(context[:500])
    print()

    # Final stats
    print("Final Stats:")
    print(rag.get_stats())
