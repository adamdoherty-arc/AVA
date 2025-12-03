"""
Memory Manager for Orchestrator
Manages short/medium/long-term memory using local storage
- Short-term: ChromaDB (semantic search)
- Long-term: SQLite (structured storage)
100% Local, No Cloud
"""
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Unified memory management for orchestrator
    - Short-term: Current session context (ChromaDB)
    - Medium-term: Recent tasks (7 days, SQLite)
    - Long-term: Project knowledge (90 days, SQLite)
    """

    def __init__(self) -> None:
        self.short_term = None  # Will initialize ChromaDB
        self.long_term_db = Path(".claude/orchestrator/databases/memory.db")
        self._init_storage()
        logger.info("Memory manager initialized")

    def _init_storage(self) -> None:
        """Initialize all storage backends"""
        # Initialize ChromaDB for short-term semantic memory
        try:
            import chromadb
            self.short_term = chromadb.PersistentClient(
                path=".claude/orchestrator/databases/chromadb"
            )
            logger.info("ChromaDB initialized for short-term memory")
        except ImportError:
            logger.warning("ChromaDB not installed. Short-term semantic search disabled.")
            logger.info("Install with: pip install chromadb")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}")
            logger.info("Short-term semantic search disabled")
            self.short_term = None

        # Initialize SQLite for long-term memory
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite schema"""
        import sqlite3
        self.long_term_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        # Knowledge base table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT,
                confidence REAL DEFAULT 0.5,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        """)

        # Context cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS context_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_name TEXT NOT NULL,
                context_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                access_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kb_key ON knowledge_base(key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kb_category ON knowledge_base(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_context_feature ON context_cache(feature_name)")

        conn.commit()
        conn.close()

    def store_knowledge(self, key: str, value: Any, category: str = "general",
                       confidence: float = 0.8, source: str = "learned"):
        """Store knowledge in long-term memory"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        value_json = json.dumps(value) if not isinstance(value, str) else value

        cursor.execute("""
            INSERT OR REPLACE INTO knowledge_base
            (key, value, category, confidence, source, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (key, value_json, category, confidence, source))

        conn.commit()
        conn.close()
        logger.debug(f"Stored knowledge: {key}")

    def retrieve_knowledge(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve knowledge from long-term memory"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT value, category, confidence, source, created_at, access_count
            FROM knowledge_base
            WHERE key = ?
        """, (key,))

        result = cursor.fetchone()

        if result:
            # Update access count
            cursor.execute("""
                UPDATE knowledge_base
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE key = ?
            """, (key,))
            conn.commit()

        conn.close()

        if result:
            try:
                value = json.loads(result[0])
            except:
                value = result[0]

            return {
                "value": value,
                "category": result[1],
                "confidence": result[2],
                "source": result[3],
                "created_at": result[4],
                "access_count": result[5] + 1
            }
        return None

    def search_knowledge(self, category: str = None, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        if category:
            cursor.execute("""
                SELECT key, value, category, confidence, source
                FROM knowledge_base
                WHERE category = ? AND confidence >= ?
                ORDER BY confidence DESC, access_count DESC
            """, (category, min_confidence))
        else:
            cursor.execute("""
                SELECT key, value, category, confidence, source
                FROM knowledge_base
                WHERE confidence >= ?
                ORDER BY confidence DESC, access_count DESC
            """, (min_confidence,))

        results = []
        for row in cursor.fetchall():
            try:
                value = json.loads(row[1])
            except:
                value = row[1]

            results.append({
                "key": row[0],
                "value": value,
                "category": row[2],
                "confidence": row[3],
                "source": row[4]
            })

        conn.close()
        return results

    def store_context(self, feature_name: str, context: Dict[str, Any],
                     ttl_hours: int = 168):  # 7 days default
        """Store feature context (medium-term memory)"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        expires_at = datetime.now() + timedelta(hours=ttl_hours)

        cursor.execute("""
            INSERT INTO context_cache (feature_name, context_data, expires_at)
            VALUES (?, ?, ?)
        """, (feature_name, json.dumps(context), expires_at))

        conn.commit()
        conn.close()
        logger.debug(f"Stored context for feature: {feature_name}")

    def retrieve_context(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve feature context"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT context_data, created_at
            FROM context_cache
            WHERE feature_name = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ORDER BY created_at DESC
            LIMIT 1
        """, (feature_name,))

        result = cursor.fetchone()

        if result:
            # Update access count
            cursor.execute("""
                UPDATE context_cache
                SET access_count = access_count + 1
                WHERE feature_name = ?
                AND context_data = ?
            """, (feature_name, result[0]))
            conn.commit()

        conn.close()

        if result:
            return json.loads(result[0])
        return None

    def add_to_short_term(self, text: str, metadata: Dict[str, Any] = None):
        """Add to short-term semantic memory (ChromaDB)"""
        if not self.short_term:
            logger.warning("ChromaDB not available")
            return

        try:
            collection = self.short_term.get_or_create_collection("short_term_memory")
            collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[str(datetime.now().timestamp())]
            )
            logger.debug("Added to short-term memory")
        except Exception as e:
            logger.error(f"Failed to add to short-term memory: {e}")

    def search_short_term(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search short-term memory"""
        if not self.short_term:
            logger.warning("ChromaDB not available")
            return []

        try:
            collection = self.short_term.get_or_create_collection("short_term_memory")
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

            return [
                {
                    "document": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Short-term search failed: {e}")
            return []

    def cleanup_expired(self) -> None:
        """Clean up expired context cache"""
        import sqlite3
        conn = sqlite3.connect(str(self.long_term_db))
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM context_cache
            WHERE expires_at IS NOT NULL
            AND expires_at < CURRENT_TIMESTAMP
        """)

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        logger.info(f"Cleaned up {deleted} expired context entries")


# Singleton
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get singleton memory manager"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


if __name__ == "__main__":
    # Test memory manager
    memory = get_memory_manager()

    # Test knowledge storage
    memory.store_knowledge(
        "test_pattern",
        {"pattern": "example", "success_rate": 0.95},
        category="patterns",
        confidence=0.9
    )

    # Retrieve knowledge
    knowledge = memory.retrieve_knowledge("test_pattern")
    print(f"\nRetrieved knowledge: {json.dumps(knowledge, indent=2)}")

    # Search knowledge
    results = memory.search_knowledge(category="patterns")
    print(f"\nKnowledge search results: {len(results)}")

    # Test context storage
    memory.store_context("test-feature", {"last_used": "now", "config": {"x": 1}})
    context = memory.retrieve_context("test-feature")
    print(f"\nRetrieved context: {json.dumps(context, indent=2)}")

    print("\nMemory manager test complete!")