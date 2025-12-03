"""
RAG Document Ingestion Pipeline
================================

Automated pipeline for ingesting, processing, and embedding documents
into the RAG knowledge base.

Features:
- Multi-source document ingestion (files, URLs, databases)
- Automatic chunking with smart overlap
- Embedding generation (OpenAI, Sentence Transformers, local)
- Metadata extraction and categorization
- ChromaDB integration with collections
- Daily XTrades message sync
- Deduplication and versioning
- Progress tracking and error handling

Supported Sources:
- Local files (PDF, TXT, MD, DOCX)
- URLs (web scraping)
- PostgreSQL (XTrades messages, Discord, alerts)
- APIs (financial data, research papers)

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
from typing import Dict, Any, Optional, List, Union, Iterator
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import os

logger = logging.getLogger(__name__)


# ============================================================================
# Document Types and Categories
# ============================================================================

class DocumentCategory(Enum):
    """Document categories for knowledge base"""
    XTRADES_MESSAGES = "xtrades_messages"  # Daily trader signals
    DISCORD_MESSAGES = "discord_messages"  # Community discussions
    TRADING_STRATEGIES = "trading_strategies"  # Strategy papers
    MARKET_RESEARCH = "market_research"  # Research reports
    TECHNICAL_DOCS = "technical_docs"  # Platform documentation
    SEC_FILINGS = "sec_filings"  # Regulatory filings
    EARNINGS_REPORTS = "earnings_reports"  # Company earnings
    OPTIONS_EDUCATION = "options_education"  # Educational content
    NEWS_ARTICLES = "news_articles"  # Market news
    PLATFORM_DOCS = "platform_docs"  # Magnus documentation


class DocumentSource(Enum):
    """Sources of documents"""
    DATABASE = "database"  # PostgreSQL
    LOCAL_FILE = "local_file"  # File system
    URL = "url"  # Web scraping
    API = "api"  # External APIs
    UPLOAD = "upload"  # User uploads


# ============================================================================
# Document Ingestion Pipeline
# ============================================================================

class DocumentIngestionPipeline:
    """
    Main pipeline for document ingestion into RAG system

    Handles the complete lifecycle from raw documents to embedded vectors.
    """

    def __init__(
        self,
        chroma_client=None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize ingestion pipeline

        Args:
            chroma_client: ChromaDB client (creates if None)
            embedding_model: Embedding model name
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        # Initialize ChromaDB
        self.chroma_client = chroma_client or self._init_chromadb()

        # Initialize embedder
        self.embedder = self._init_embedder(embedding_model)

        # Statistics
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_generated': 0,
            'errors': 0,
            'duplicates_skipped': 0
        }

        logger.info(f"Document ingestion pipeline initialized with {embedding_model}")

    def _init_chromadb(self) -> None:
        """Initialize ChromaDB client"""
        try:
            import chromadb
            from chromadb.config import Settings

            # Use persistent storage
            persist_directory = Path("data/chroma_db")
            persist_directory.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            logger.info(f"ChromaDB initialized at {persist_directory}")
            return client

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def _init_embedder(self, model_name: str):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
            return embedder

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            logger.info("Falling back to OpenAI embeddings")
            return None  # Will use OpenAI as fallback

    def ingest_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        category: DocumentCategory,
        source: DocumentSource = DocumentSource.LOCAL_FILE
    ) -> Dict[str, Any]:
        """
        Ingest a single document

        Args:
            content: Document text content
            metadata: Document metadata
            category: Document category
            source: Document source

        Returns:
            Dict with ingestion results
        """
        try:
            # Generate document ID
            doc_id = self._generate_doc_id(content, metadata)

            # Check for duplicates
            if self._is_duplicate(doc_id, category):
                logger.info(f"Skipping duplicate document: {doc_id}")
                self.stats['duplicates_skipped'] += 1
                return {
                    'status': 'skipped',
                    'reason': 'duplicate',
                    'doc_id': doc_id
                }

            # Chunk the document
            chunks = self._chunk_document(content)

            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)

            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    **metadata,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'category': category.value,
                    'source': source.value,
                    'ingestion_date': datetime.now().isoformat(),
                    'chunk_size': len(chunk)
                }
                chunk_metadata.append(meta)

            # Get or create collection
            collection = self._get_collection(category)

            # Add to ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadata,
                ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            )

            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['embeddings_generated'] += len(embeddings)

            logger.info(f"Ingested document {doc_id}: {len(chunks)} chunks")

            return {
                'status': 'success',
                'doc_id': doc_id,
                'chunks_created': len(chunks),
                'category': category.value
            }

        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            self.stats['errors'] += 1
            return {
                'status': 'error',
                'error': str(e)
            }

    def ingest_batch(
        self,
        documents: List[Dict[str, Any]],
        category: DocumentCategory,
        source: DocumentSource = DocumentSource.LOCAL_FILE
    ) -> Dict[str, Any]:
        """
        Ingest multiple documents in batch

        Args:
            documents: List of dicts with 'content' and 'metadata'
            category: Document category
            source: Document source

        Returns:
            Dict with batch ingestion results
        """
        results = {
            'total': len(documents),
            'success': 0,
            'skipped': 0,
            'errors': 0,
            'results': []
        }

        for doc in documents:
            result = self.ingest_document(
                content=doc['content'],
                metadata=doc.get('metadata', {}),
                category=category,
                source=source
            )

            results['results'].append(result)

            if result['status'] == 'success':
                results['success'] += 1
            elif result['status'] == 'skipped':
                results['skipped'] += 1
            else:
                results['errors'] += 1

        logger.info(f"Batch ingestion complete: {results['success']}/{results['total']} success")

        return results

    def ingest_from_database(
        self,
        table: str,
        query: str,
        category: DocumentCategory,
        text_column: str = 'content',
        metadata_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest documents from PostgreSQL database

        Args:
            table: Database table name
            query: SQL query to fetch documents
            category: Document category
            text_column: Column containing text content
            metadata_columns: Columns to include as metadata

        Returns:
            Dict with ingestion results
        """
        try:
            from src.database.connection_pool import get_connection

            documents = []

            with get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)

                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()

                for row in rows:
                    row_dict = dict(zip(columns, row))

                    # Extract content
                    content = str(row_dict.get(text_column, ''))

                    if not content:
                        continue

                    # Extract metadata
                    metadata = {
                        'table': table,
                        'source_query': query[:100]  # Truncate query
                    }

                    if metadata_columns:
                        for col in metadata_columns:
                            if col in row_dict:
                                value = row_dict[col]
                                # Convert datetime to string
                                if isinstance(value, datetime):
                                    value = value.isoformat()
                                metadata[col] = value

                    documents.append({
                        'content': content,
                        'metadata': metadata
                    })

            logger.info(f"Fetched {len(documents)} documents from {table}")

            # Ingest batch
            return self.ingest_batch(
                documents=documents,
                category=category,
                source=DocumentSource.DATABASE
            )

        except Exception as e:
            logger.error(f"Database ingestion failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def ingest_xtrades_messages(
        self,
        days_back: int = 1,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest XTrades messages from database (daily sync)

        Args:
            days_back: Number of days to fetch (default: 1 for daily sync)
            force_reload: Re-ingest even if duplicates

        Returns:
            Dict with ingestion results
        """
        logger.info(f"Starting XTrades message ingestion (last {days_back} days)")

        # Build query to fetch recent messages
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = f"""
        SELECT
            id,
            content,
            timestamp,
            channel_name,
            author_name,
            ticker,
            alert_type,
            entry_price,
            target_price,
            stop_loss,
            confidence_score
        FROM discord_messages
        WHERE timestamp >= '{cutoff_date.isoformat()}'
        AND channel_name LIKE '%xtrades%'
        ORDER BY timestamp DESC
        """

        # Temporarily disable duplicate check if force reload
        original_check = self._is_duplicate
        if force_reload:
            self._is_duplicate = lambda *args: False

        result = self.ingest_from_database(
            table='discord_messages',
            query=query,
            category=DocumentCategory.XTRADES_MESSAGES,
            text_column='content',
            metadata_columns=[
                'timestamp', 'channel_name', 'author_name', 'ticker',
                'alert_type', 'entry_price', 'target_price', 'stop_loss',
                'confidence_score'
            ]
        )

        # Restore duplicate check
        if force_reload:
            self._is_duplicate = original_check

        logger.info(f"XTrades ingestion complete: {result.get('success', 0)} messages added")

        return result

    def ingest_discord_messages(
        self,
        days_back: int = 7,
        channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest Discord messages (trader discussions, alerts)

        Args:
            days_back: Number of days to fetch
            channels: Specific channels to ingest (None = all)

        Returns:
            Dict with ingestion results
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        query = f"""
        SELECT
            content,
            timestamp,
            channel_name,
            author_name,
            message_type
        FROM discord_messages
        WHERE timestamp >= '{cutoff_date.isoformat()}'
        """

        if channels:
            channel_list = "', '".join(channels)
            query += f" AND channel_name IN ('{channel_list}')"

        query += " ORDER BY timestamp DESC"

        return self.ingest_from_database(
            table='discord_messages',
            query=query,
            category=DocumentCategory.DISCORD_MESSAGES,
            text_column='content',
            metadata_columns=['timestamp', 'channel_name', 'author_name', 'message_type']
        )

    def ingest_local_directory(
        self,
        directory: Union[str, Path],
        category: DocumentCategory,
        file_extensions: List[str] = ['.txt', '.md', '.pdf'],
        recursive: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest all files from a local directory

        Args:
            directory: Directory path
            category: Document category
            file_extensions: File extensions to include
            recursive: Recursively scan subdirectories

        Returns:
            Dict with ingestion results
        """
        directory = Path(directory)

        if not directory.exists():
            return {
                'status': 'error',
                'error': f"Directory not found: {directory}"
            }

        # Find files
        files = []
        if recursive:
            for ext in file_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in file_extensions:
                files.extend(directory.glob(f"*{ext}"))

        logger.info(f"Found {len(files)} files in {directory}")

        # Read and ingest files
        documents = []
        for file_path in files:
            try:
                content = self._read_file(file_path)

                if content:
                    documents.append({
                        'content': content,
                        'metadata': {
                            'filename': file_path.name,
                            'file_path': str(file_path),
                            'file_size': file_path.stat().st_size,
                            'modified_date': datetime.fromtimestamp(
                                file_path.stat().st_mtime
                            ).isoformat()
                        }
                    })

            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")

        return self.ingest_batch(
            documents=documents,
            category=category,
            source=DocumentSource.LOCAL_FILE
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _generate_doc_id(self, content: str, metadata: Dict) -> str:
        """Generate unique document ID from content and metadata"""
        # Create hash from content + key metadata fields
        hash_input = content[:1000]  # First 1000 chars

        # Add stable metadata fields
        for key in ['filename', 'url', 'ticker', 'timestamp']:
            if key in metadata:
                hash_input += str(metadata[key])

        return hashlib.md5(hash_input.encode()).hexdigest()

    def _is_duplicate(self, doc_id: str, category: DocumentCategory) -> bool:
        """Check if document already exists in collection"""
        try:
            collection = self._get_collection(category)

            # Query for doc_id in metadata
            results = collection.get(
                where={"doc_id": doc_id},
                limit=1
            )

            return len(results['ids']) > 0

        except Exception as e:
            logger.warning(f"Duplicate check failed: {e}")
            return False

    def _chunk_document(self, content: str) -> List[str]:
        """Split document into chunks with overlap"""
        chunks = []
        start = 0

        while start < len(content):
            # Get chunk
            end = start + self.chunk_size
            chunk = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence end in last 100 chars
                last_period = chunk.rfind('. ', max(0, len(chunk) - 100))
                if last_period > 0:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())

            # Move start with overlap
            start = end - self.chunk_overlap

        return [c for c in chunks if len(c) > 50]  # Filter tiny chunks

    def _generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        if self.embedder:
            # Use Sentence Transformers
            embeddings = self.embedder.encode(chunks, show_progress_bar=False)
            return embeddings.tolist()
        else:
            # Fallback to OpenAI
            import openai
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

            embeddings = []
            for chunk in chunks:
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                embeddings.append(response.data[0].embedding)

            return embeddings

    def _get_collection(self, category: DocumentCategory):
        """Get or create ChromaDB collection for category"""
        collection_name = f"magnus_{category.value}"

        try:
            collection = self.chroma_client.get_collection(collection_name)
        except:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"category": category.value}
            )
            logger.info(f"Created new collection: {collection_name}")

        return collection

    def _read_file(self, file_path: Path) -> Optional[str]:
        """Read file content based on extension"""
        ext = file_path.suffix.lower()

        if ext in ['.txt', '.md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        elif ext == '.pdf':
            try:
                from src.multimodal.pdf_parser import parse_pdf_document

                result = parse_pdf_document(file_path)
                return result.get('text', '')
            except:
                logger.warning(f"Failed to parse PDF: {file_path}")
                return None

        elif ext == '.docx':
            try:
                import docx

                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except:
                logger.warning(f"Failed to parse DOCX: {file_path}")
                return None

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics"""
        return {
            **self.stats,
            'total_collections': len(self.chroma_client.list_collections()),
            'embedding_model': self.embedding_model
        }

    def reset_category(self, category: DocumentCategory):
        """Delete all documents in a category"""
        collection_name = f"magnus_{category.value}"

        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except:
            logger.warning(f"Collection not found: {collection_name}")


# ============================================================================
# Convenience Functions
# ============================================================================

def ingest_xtrades_daily():
    """Daily cron job to ingest XTrades messages"""
    pipeline = DocumentIngestionPipeline()

    result = pipeline.ingest_xtrades_messages(days_back=1)

    logger.info(f"Daily XTrades sync: {result}")

    return result


def ingest_all_xtrades_history(days: int = 90):
    """One-time ingestion of historical XTrades messages"""
    pipeline = DocumentIngestionPipeline()

    result = pipeline.ingest_xtrades_messages(
        days_back=days,
        force_reload=False
    )

    logger.info(f"Historical XTrades ingestion: {result}")

    return result


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    # Test pipeline
    pipeline = DocumentIngestionPipeline()

    # Test XTrades ingestion
    result = pipeline.ingest_xtrades_messages(days_back=7)

    print(json.dumps(result, indent=2))
    print("\nStatistics:")
    print(json.dumps(pipeline.get_stats(), indent=2))
