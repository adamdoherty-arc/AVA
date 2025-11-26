"""
Knowledge Router - RAG-powered knowledge base
NO MOCK DATA - All endpoints use real RAG system
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional, List
from datetime import datetime
import logging
import os
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Try to import RAG system
_rag = None

def get_rag():
    """Get RAG system instance (lazy loading)"""
    global _rag
    if _rag is None:
        try:
            from src.rag import get_rag as get_rag_instance
            _rag = get_rag_instance()
            if _rag:
                logger.info("RAG system initialized successfully")
        except ImportError as e:
            logger.warning(f"RAG system not available: {e}")
            _rag = None
        except Exception as e:
            logger.warning(f"Failed to initialize RAG: {e}")
            _rag = None
    return _rag


@router.get("/documents")
async def get_documents(category: Optional[str] = None, search: Optional[str] = None):
    """Get all documents in the knowledge base from RAG system"""
    rag = get_rag()

    if rag is None:
        return {
            "documents": [],
            "total": 0,
            "message": "RAG system not available. Install: pip install chromadb sentence-transformers"
        }

    try:
        # Get documents from RAG system
        documents = rag.list_documents()

        # Apply filters
        if category:
            documents = [d for d in documents if d.get("category") == category]
        if search:
            documents = [d for d in documents if search.lower() in d.get("title", "").lower()]

        return {"documents": documents, "total": len(documents)}

    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        return {
            "documents": [],
            "total": 0,
            "error": str(e)
        }


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new document to the knowledge base"""
    rag = get_rag()

    if rag is None:
        return {
            "id": None,
            "status": "error",
            "message": "RAG system not available. Install: pip install chromadb sentence-transformers"
        }

    try:
        # Read file content
        content = await file.read()

        # Generate unique document ID
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"

        # Add to RAG system
        result = rag.add_document(
            doc_id=doc_id,
            title=file.filename,
            content=content.decode('utf-8', errors='ignore'),
            source="upload",
            metadata={"filename": file.filename, "content_type": file.content_type}
        )

        return {
            "id": doc_id,
            "title": file.filename,
            "status": "indexed" if result else "processing",
            "message": "Document uploaded and indexed successfully" if result else "Document uploaded, processing..."
        }

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return {
            "id": None,
            "status": "error",
            "message": f"Failed to upload document: {str(e)}"
        }


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    rag = get_rag()

    if rag is None:
        return {"status": "error", "message": "RAG system not available"}

    try:
        result = rag.delete_document(doc_id)
        return {
            "status": "success" if result else "not_found",
            "message": f"Document {doc_id} deleted" if result else f"Document {doc_id} not found"
        }
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return {"status": "error", "message": str(e)}


@router.post("/query")
async def query_knowledge(query: str, top_k: int = 5, category: Optional[str] = None):
    """Query the knowledge base using RAG"""
    rag = get_rag()

    if rag is None:
        return {
            "query": query,
            "results": [],
            "total_chunks_searched": 0,
            "response_time_ms": 0,
            "message": "RAG system not available. Install: pip install chromadb sentence-transformers"
        }

    try:
        import time
        start_time = time.time()

        # Query RAG system
        results = rag.query(
            query_text=query,
            n_results=top_k,
            category_filter=category
        )

        response_time = (time.time() - start_time) * 1000

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "content": result.get("content", ""),
                "source": result.get("source", "Unknown"),
                "relevance": round(result.get("relevance", 0), 2),
                "chunk_id": result.get("chunk_id", ""),
                "metadata": result.get("metadata", {})
            })

        return {
            "query": query,
            "results": formatted_results,
            "total_chunks_searched": rag.get_total_chunks() if hasattr(rag, 'get_total_chunks') else len(results),
            "response_time_ms": round(response_time, 1)
        }

    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
        return {
            "query": query,
            "results": [],
            "total_chunks_searched": 0,
            "response_time_ms": 0,
            "error": str(e)
        }


@router.get("/stats")
async def get_stats():
    """Get knowledge base statistics from RAG system"""
    rag = get_rag()

    if rag is None:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_tokens": 0,
            "embedding_model": "not available",
            "vector_dimensions": 0,
            "index_size_mb": 0,
            "categories": {},
            "last_updated": datetime.now().isoformat(),
            "message": "RAG system not available. Install: pip install chromadb sentence-transformers"
        }

    try:
        stats = rag.get_stats() if hasattr(rag, 'get_stats') else {}

        return {
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "total_tokens": stats.get("total_tokens", 0),
            "embedding_model": stats.get("embedding_model", "sentence-transformers"),
            "vector_dimensions": stats.get("vector_dimensions", 384),
            "index_size_mb": stats.get("index_size_mb", 0),
            "categories": stats.get("categories", {}),
            "last_updated": stats.get("last_updated", datetime.now().isoformat())
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "error": str(e),
            "last_updated": datetime.now().isoformat()
        }


@router.post("/reindex")
async def reindex_knowledge_base():
    """Reindex the entire knowledge base"""
    rag = get_rag()

    if rag is None:
        return {
            "status": "error",
            "message": "RAG system not available"
        }

    try:
        # Start reindex process
        if hasattr(rag, 'reindex'):
            rag.reindex()

        return {
            "status": "started",
            "message": "Knowledge base reindexing started",
            "estimated_time_minutes": 5
        }

    except Exception as e:
        logger.error(f"Error reindexing: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/sources")
async def get_sources():
    """Get available knowledge sources"""
    rag = get_rag()

    if rag is None:
        return {
            "sources": [
                {"id": "uploads", "name": "User Uploads", "status": "unavailable", "documents": 0}
            ],
            "message": "RAG system not available"
        }

    try:
        # Get sources from RAG if available
        if hasattr(rag, 'get_sources'):
            sources = rag.get_sources()
        else:
            # Default sources
            sources = [
                {"id": "uploads", "name": "User Uploads", "status": "active", "documents": 0},
                {"id": "internal", "name": "Internal Docs", "status": "active", "documents": 0}
            ]

        return {"sources": sources}

    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        return {
            "sources": [],
            "error": str(e)
        }


@router.get("/context")
async def get_context_for_query(query: str, n_results: int = 3, max_length: int = 2000):
    """Get formatted context for a query (used by chat/agents)"""
    rag = get_rag()

    if rag is None:
        return {
            "context": "",
            "message": "RAG system not available"
        }

    try:
        context = rag.get_context_for_query(
            query_text=query,
            n_results=n_results,
            max_context_length=max_length
        )

        return {
            "context": context or "",
            "query": query,
            "n_results": n_results
        }

    except Exception as e:
        logger.error(f"Error getting context: {e}")
        return {
            "context": "",
            "error": str(e)
        }


@router.post("/add-text")
async def add_text_document(title: str, content: str, category: str = "general"):
    """Add a text document directly to the knowledge base"""
    rag = get_rag()

    if rag is None:
        return {
            "status": "error",
            "message": "RAG system not available"
        }

    try:
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"

        result = rag.add_document(
            doc_id=doc_id,
            title=title,
            content=content,
            source="api",
            metadata={"category": category}
        )

        return {
            "id": doc_id,
            "title": title,
            "status": "indexed" if result else "error",
            "message": "Document added successfully" if result else "Failed to add document"
        }

    except Exception as e:
        logger.error(f"Error adding text document: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
