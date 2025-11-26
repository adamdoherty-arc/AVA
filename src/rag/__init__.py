"""
Magnus RAG System
Retrieval Augmented Generation for financial knowledge base

Components:
- RAGService: Production-ready RAG with Cross-Encoder & LLM
- DocumentIngestionPipeline: Document ingestion and embedding
"""

from src.rag.rag_service import RAGService
from src.rag.document_ingestion_pipeline import (
    DocumentIngestionPipeline,
    DocumentCategory,
    DocumentSource,
    ingest_xtrades_daily,
    ingest_all_xtrades_history
)

__version__ = "2.0.0"

_rag_instance = None

def get_rag() -> RAGService:
    """Get singleton instance of RAG Service"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService()
    return _rag_instance

__all__ = [
    "RAGService",
    "get_rag",
    "DocumentIngestionPipeline",
    "DocumentCategory",
    "DocumentSource",
    "ingest_xtrades_daily",
    "ingest_all_xtrades_history",
]
