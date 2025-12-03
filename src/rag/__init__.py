"""
Magnus RAG System
=================

Retrieval Augmented Generation for financial knowledge base.

Primary Interface (Recommended):
- UnifiedRAG: Single entry point for all RAG operations
- get_unified_rag(): Get singleton instance
- get_rag_context(): Quick helper for context retrieval

Components:
- RAGService: Production-ready RAG with Cross-Encoder & LLM
- DocumentIngestionPipeline: Document ingestion and embedding

Usage:
    from src.rag import get_unified_rag, get_rag_context

    # Full interface
    rag = get_unified_rag()
    response = rag.query("What is the wheel strategy?")
    print(response.answer)

    # Quick context for prompts
    context = get_rag_context("options trading")
"""

# Primary interface (recommended)
from src.rag.unified_rag import (
    UnifiedRAG,
    get_unified_rag,
    get_rag_context,
    RAGResponse,
    DocumentInfo
)

# Advanced components
from src.rag.rag_service import RAGService, QueryResult
from src.rag.document_ingestion_pipeline import (
    DocumentIngestionPipeline,
    DocumentCategory,
    DocumentSource,
    ingest_xtrades_daily,
    ingest_all_xtrades_history
)

__version__ = "3.0.0"

# Singleton instance for backward compatibility
_rag_instance = None


def get_rag() -> UnifiedRAG:
    """
    Get singleton instance of UnifiedRAG.

    Note: This now returns UnifiedRAG instead of RAGService.
    For direct RAGService access, use RAGService() directly.

    Returns:
        UnifiedRAG instance
    """
    return get_unified_rag()


__all__ = [
    # Primary interface (recommended)
    "UnifiedRAG",
    "get_unified_rag",
    "get_rag_context",
    "get_rag",
    "RAGResponse",
    "DocumentInfo",
    # Advanced components
    "RAGService",
    "QueryResult",
    "DocumentIngestionPipeline",
    "DocumentCategory",
    "DocumentSource",
    "ingest_xtrades_daily",
    "ingest_all_xtrades_history",
]
