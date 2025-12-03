"""
AVA Memory Manager
==================

Multi-level memory system for persistent user context, preferences, and conversation history.

Memory Levels:
1. Session Memory - Current conversation context (Redis)
2. User Memory - Long-term preferences and facts (PostgreSQL)
3. Entity Memory - Ticker/strategy tracking (PostgreSQL)
4. Conversation Summaries - Compressed history with embeddings (PostgreSQL + pgvector)

Author: Magnus Trading Platform
Created: 2025-11-23
"""

import json
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from loguru import logger
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    key: str
    value: Any
    memory_type: str
    category: Optional[str] = None
    confidence: float = 1.0
    importance: int = 5
    source: str = "explicit"
    context: Optional[str] = None


@dataclass
class EntityMemory:
    """Represents entity tracking memory"""
    entity_type: str
    entity_id: str
    entity_name: Optional[str] = None
    mention_count: int = 1
    sentiment: str = "neutral"
    interest_score: int = 5
    contexts: List[str] = None
    tags: List[str] = None

    def __post_init__(self) -> None:
        if self.contexts is None:
            self.contexts = []
        if self.tags is None:
            self.tags = []


@dataclass
class ConversationSummary:
    """Represents a conversation summary"""
    summary: str
    key_topics: List[str]
    entities_mentioned: List[str]
    message_count: int
    conversation_start: datetime
    conversation_end: datetime
    original_tokens: int
    summary_tokens: int
    sentiment: str = "neutral"
    model_used: str = "gpt-3.5-turbo"


class MemoryManager:
    """
    Multi-level memory management system for AVA

    Features:
    - Persistent user preferences and facts
    - Entity tracking (tickers, strategies)
    - Conversation summarization with vector embeddings
    - Semantic similarity search
    - Automatic context compression
    """

    def __init__(self, db_connection_string: str):
        """
        Initialize Memory Manager

        Args:
            db_connection_string: PostgreSQL connection string
        """
        self.db_connection_string = db_connection_string
        self._conn = None
        self._ensure_connection()
        logger.info("Memory Manager initialized")

    def _ensure_connection(self) -> None:
        """Ensure database connection is alive"""
        try:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self.db_connection_string)
                self._conn.autocommit = False
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _get_cursor(self) -> None:
        """Get database cursor with dict factory"""
        self._ensure_connection()
        return self._conn.cursor(cursor_factory=RealDictCursor)

    # =========================================================================
    # User Memory Operations
    # =========================================================================

    def store_user_memory(
        self,
        user_id: str,
        platform: str,
        memory: MemoryEntry
    ) -> bool:
        """
        Store a user memory entry

        Args:
            user_id: User identifier
            platform: Platform (telegram, discord, web)
            memory: MemoryEntry object

        Returns:
            Success status
        """
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    INSERT INTO ava_user_memory (
                        user_id, platform, memory_type, category, key, value,
                        confidence_score, importance, source, context
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform, memory_type, key)
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        confidence_score = EXCLUDED.confidence_score,
                        importance = EXCLUDED.importance,
                        updated_at = NOW()
                """, (
                    user_id, platform, memory.memory_type, memory.category,
                    memory.key, Json(memory.value), memory.confidence,
                    memory.importance, memory.source, memory.context
                ))
                self._conn.commit()
                logger.debug(f"Stored memory: {memory.key} for {user_id}")
                return True
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to store user memory: {e}")
            return False

    def get_user_memory(
        self,
        user_id: str,
        platform: str,
        memory_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve user memories

        Args:
            user_id: User identifier
            platform: Platform
            memory_type: Filter by type (optional)
            category: Filter by category (optional)

        Returns:
            List of memory dictionaries
        """
        try:
            with self._get_cursor() as cur:
                query = """
                    SELECT * FROM ava_user_memory
                    WHERE user_id = %s AND platform = %s
                    AND (expires_at IS NULL OR expires_at > NOW())
                """
                params = [user_id, platform]

                if memory_type:
                    query += " AND memory_type = %s"
                    params.append(memory_type)

                if category:
                    query += " AND category = %s"
                    params.append(category)

                query += " ORDER BY importance DESC, updated_at DESC"

                cur.execute(query, params)
                results = cur.fetchall()

                # Update access tracking
                if results:
                    ids = [r['id'] for r in results]
                    cur.execute("""
                        UPDATE ava_user_memory
                        SET access_count = access_count + 1,
                            last_accessed_at = NOW()
                        WHERE id = ANY(%s)
                    """, (ids,))
                    self._conn.commit()

                return [dict(r) for r in results]
        except Exception as e:
            logger.error(f"Failed to get user memory: {e}")
            return []

    def get_user_preference(
        self,
        user_id: str,
        platform: str,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get a specific user preference

        Args:
            user_id: User identifier
            platform: Platform
            key: Preference key
            default: Default value if not found

        Returns:
            Preference value or default
        """
        memories = self.get_user_memory(user_id, platform, memory_type="preference")
        for memory in memories:
            if memory['key'] == key:
                return memory['value']
        return default

    def update_user_preference(
        self,
        user_id: str,
        platform: str,
        key: str,
        value: Any,
        importance: int = 5
    ) -> bool:
        """
        Update a user preference (convenience method)

        Args:
            user_id: User identifier
            platform: Platform
            key: Preference key
            value: Preference value
            importance: Importance score (1-10)

        Returns:
            Success status
        """
        memory = MemoryEntry(
            key=key,
            value=value,
            memory_type="preference",
            importance=importance,
            source="explicit"
        )
        return self.store_user_memory(user_id, platform, memory)

    # =========================================================================
    # Entity Memory Operations
    # =========================================================================

    def track_entity(
        self,
        user_id: str,
        platform: str,
        entity: EntityMemory,
        context: Optional[str] = None
    ) -> bool:
        """
        Track an entity mention (ticker, strategy, etc.)

        Args:
            user_id: User identifier
            platform: Platform
            entity: EntityMemory object
            context: Context where mentioned

        Returns:
            Success status
        """
        try:
            with self._get_cursor() as cur:
                # Add context to list (keep last 5)
                contexts = entity.contexts[-5:] if entity.contexts else []
                if context:
                    contexts.append(context)
                    contexts = contexts[-5:]  # Keep last 5

                cur.execute("""
                    INSERT INTO ava_entity_memory (
                        user_id, platform, entity_type, entity_id, entity_name,
                        mention_count, overall_sentiment, interest_score,
                        contexts, tags
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (user_id, platform, entity_type, entity_id)
                    DO UPDATE SET
                        mention_count = ava_entity_memory.mention_count + 1,
                        last_mentioned_at = NOW(),
                        contexts = %s,
                        overall_sentiment = EXCLUDED.overall_sentiment,
                        interest_score = EXCLUDED.interest_score,
                        updated_at = NOW()
                """, (
                    user_id, platform, entity.entity_type, entity.entity_id,
                    entity.entity_name, entity.mention_count, entity.sentiment,
                    entity.interest_score, contexts, entity.tags,
                    contexts  # For the UPDATE clause
                ))
                self._conn.commit()
                logger.debug(f"Tracked entity: {entity.entity_id} for {user_id}")
                return True
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to track entity: {e}")
            return False

    def get_user_entities(
        self,
        user_id: str,
        platform: str,
        entity_type: Optional[str] = None,
        min_mentions: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get tracked entities for a user

        Args:
            user_id: User identifier
            platform: Platform
            entity_type: Filter by type (optional)
            min_mentions: Minimum mention count

        Returns:
            List of entity dictionaries
        """
        try:
            with self._get_cursor() as cur:
                query = """
                    SELECT * FROM ava_entity_memory
                    WHERE user_id = %s AND platform = %s
                    AND mention_count >= %s
                """
                params = [user_id, platform, min_mentions]

                if entity_type:
                    query += " AND entity_type = %s"
                    params.append(entity_type)

                query += " ORDER BY interest_score DESC, mention_count DESC"

                cur.execute(query, params)
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []

    # =========================================================================
    # Conversation Summary Operations
    # =========================================================================

    def store_conversation_summary(
        self,
        user_id: str,
        platform: str,
        summary: ConversationSummary,
        embedding: Optional[np.ndarray] = None
    ) -> bool:
        """
        Store a conversation summary with optional embedding

        Args:
            user_id: User identifier
            platform: Platform
            summary: ConversationSummary object
            embedding: Vector embedding (1536 dimensions)

        Returns:
            Success status
        """
        try:
            with self._get_cursor() as cur:
                compression_ratio = summary.original_tokens / summary.summary_tokens if summary.summary_tokens > 0 else 1.0
                tokens_saved = summary.original_tokens - summary.summary_tokens

                # Convert numpy array to list for PostgreSQL
                embedding_list = embedding.tolist() if embedding is not None else None

                cur.execute("""
                    INSERT INTO ava_conversation_summaries (
                        user_id, platform, summary, key_topics, entities_mentioned,
                        message_count, conversation_start, conversation_end,
                        original_tokens, summary_tokens, compression_ratio,
                        tokens_saved, sentiment, model_used, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, platform, summary.summary, summary.key_topics,
                    summary.entities_mentioned, summary.message_count,
                    summary.conversation_start, summary.conversation_end,
                    summary.original_tokens, summary.summary_tokens,
                    compression_ratio, tokens_saved, summary.sentiment,
                    summary.model_used, embedding_list
                ))
                self._conn.commit()
                logger.info(f"Stored conversation summary for {user_id}, saved {tokens_saved} tokens")
                return True
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to store conversation summary: {e}")
            return False

    def get_recent_summaries(
        self,
        user_id: str,
        platform: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation summaries

        Args:
            user_id: User identifier
            platform: Platform
            limit: Maximum number of summaries

        Returns:
            List of summary dictionaries
        """
        try:
            with self._get_cursor() as cur:
                cur.execute("""
                    SELECT id, summary, key_topics, entities_mentioned,
                           conversation_start, conversation_end, message_count,
                           tokens_saved, sentiment
                    FROM ava_conversation_summaries
                    WHERE user_id = %s AND platform = %s
                    ORDER BY conversation_end DESC
                    LIMIT %s
                """, (user_id, platform, limit))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get recent summaries: {e}")
            return []

    def search_similar_conversations(
        self,
        user_id: str,
        platform: str,
        query_embedding: np.ndarray,
        limit: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar past conversations using vector similarity

        Args:
            user_id: User identifier
            platform: Platform
            query_embedding: Query vector (1536 dimensions)
            limit: Maximum results
            similarity_threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of similar conversations with similarity scores
        """
        try:
            with self._get_cursor() as cur:
                embedding_list = query_embedding.tolist()

                cur.execute("""
                    SELECT
                        id,
                        summary,
                        key_topics,
                        conversation_start,
                        1 - (embedding <=> %s::vector) as similarity
                    FROM ava_conversation_summaries
                    WHERE user_id = %s
                    AND platform = %s
                    AND embedding IS NOT NULL
                    AND 1 - (embedding <=> %s::vector) >= %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (
                    embedding_list, user_id, platform,
                    embedding_list, similarity_threshold,
                    embedding_list, limit
                ))

                results = [dict(r) for r in cur.fetchall()]
                logger.info(f"Found {len(results)} similar conversations for {user_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to search similar conversations: {e}")
            return []

    # =========================================================================
    # Memory Analytics
    # =========================================================================

    def get_memory_stats(
        self,
        user_id: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Get memory statistics for a user

        Args:
            user_id: User identifier
            platform: Platform

        Returns:
            Dictionary of memory statistics
        """
        try:
            with self._get_cursor() as cur:
                # User memory stats
                cur.execute("""
                    SELECT * FROM get_user_memory_summary(%s, %s)
                """, (user_id, platform))
                memory_summary = [dict(r) for r in cur.fetchall()]

                # Entity stats
                cur.execute("""
                    SELECT
                        entity_type,
                        COUNT(*) as count,
                        SUM(mention_count) as total_mentions
                    FROM ava_entity_memory
                    WHERE user_id = %s AND platform = %s
                    GROUP BY entity_type
                """, (user_id, platform))
                entity_stats = [dict(r) for r in cur.fetchall()]

                # Conversation stats
                cur.execute("""
                    SELECT * FROM v_ava_conversation_stats
                    WHERE user_id = %s AND platform = %s
                """, (user_id, platform))
                conv_stats = cur.fetchone()
                conv_stats = dict(conv_stats) if conv_stats else {}

                return {
                    "user_id": user_id,
                    "platform": platform,
                    "memory_summary": memory_summary,
                    "entity_stats": entity_stats,
                    "conversation_stats": conv_stats,
                    "generated_at": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

    def cleanup_expired_memories(self) -> int:
        """
        Clean up expired memories

        Returns:
            Number of memories deleted
        """
        try:
            with self._get_cursor() as cur:
                cur.execute("SELECT cleanup_expired_memories()")
                deleted = cur.fetchone()[0]
                self._conn.commit()
                logger.info(f"Cleaned up {deleted} expired memories")
                return deleted
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Failed to cleanup memories: {e}")
            return 0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def close(self) -> None:
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Memory Manager connection closed")

    def __enter__(self) -> None:
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# =============================================================================
# Singleton Access
# =============================================================================

_memory_manager = None


def get_memory_manager(db_connection_string: str) -> MemoryManager:
    """
    Get singleton MemoryManager instance

    Args:
        db_connection_string: PostgreSQL connection string

    Returns:
        MemoryManager instance
    """
    global _memory_manager

    if _memory_manager is None:
        _memory_manager = MemoryManager(db_connection_string)

    return _memory_manager
