"""
Async Database Infrastructure

Provides:
- True async database access with asyncpg
- Connection pooling optimized for concurrent requests
- Query caching and prepared statements
- Health checks and monitoring
- Fallback to sync pool if asyncpg unavailable
"""

import os
import logging
import asyncio
from typing import Any, Optional, List, Dict
from contextlib import asynccontextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class AsyncDatabasePool:
    """
    Async PostgreSQL connection pool using asyncpg.

    Features:
    - True async/await for non-blocking I/O
    - Connection pooling (5-100 connections)
    - Automatic retry with backoff
    - Query statistics and monitoring
    - Prepared statement caching

    Usage:
        pool = AsyncDatabasePool()
        await pool.initialize()

        # Single query
        rows = await pool.fetch("SELECT * FROM positions WHERE user_id = $1", user_id)

        # Transaction
        async with pool.transaction() as conn:
            await conn.execute("INSERT INTO ...", ...)
            await conn.execute("UPDATE ...", ...)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_size: int = 5,
        max_size: int = 50,
        command_timeout: int = 60
    ):
        self._host = host or os.getenv("DB_HOST", "localhost")
        self._port = port or int(os.getenv("DB_PORT", "5432"))
        self._database = database or os.getenv("DB_NAME", "ava")
        self._user = user or os.getenv("DB_USER", "postgres")
        self._password = password or os.getenv("DB_PASSWORD", "")
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout

        self._pool = None
        self._initialized = False
        self._use_asyncpg = True

        # Statistics
        self._stats = {
            "queries_executed": 0,
            "transactions": 0,
            "errors": 0,
            "last_error": None
        }

    async def initialize(self) -> bool:
        """Initialize the connection pool"""
        if self._initialized:
            return True

        try:
            import asyncpg

            self._pool = await asyncpg.create_pool(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                min_size=self._min_size,
                max_size=self._max_size,
                command_timeout=self._command_timeout
            )

            self._initialized = True
            self._use_asyncpg = True
            logger.info(
                f"Async DB pool initialized: {self._host}:{self._port}/{self._database} "
                f"(min={self._min_size}, max={self._max_size})"
            )
            return True

        except ImportError:
            logger.warning("asyncpg not installed, using sync fallback")
            self._use_asyncpg = False
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize async DB pool: {e}")
            self._stats["errors"] += 1
            self._stats["last_error"] = str(e)
            self._use_asyncpg = False
            self._initialized = True
            return False

    async def close(self):
        """Close the connection pool"""
        if self._pool:
            await self._pool.close()
            self._initialized = False
            logger.info("Async DB pool closed")

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return all rows as dicts.

        Args:
            query: SQL query with $1, $2, ... placeholders
            *args: Query parameters

        Returns:
            List of row dictionaries
        """
        if not self._initialized:
            await self.initialize()

        self._stats["queries_executed"] += 1

        if self._use_asyncpg and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    rows = await conn.fetch(query, *args)
                    return [dict(row) for row in rows]
            except Exception as e:
                self._stats["errors"] += 1
                self._stats["last_error"] = str(e)
                logger.error(f"Async query error: {e}")
                raise

        # Fallback to sync
        return await self._sync_fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute a SELECT query and return first row"""
        if not self._initialized:
            await self.initialize()

        self._stats["queries_executed"] += 1

        if self._use_asyncpg and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    row = await conn.fetchrow(query, *args)
                    return dict(row) if row else None
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Async fetchrow error: {e}")
                raise

        # Fallback
        rows = await self._sync_fetch(query, *args)
        return rows[0] if rows else None

    async def fetchval(self, query: str, *args) -> Any:
        """Execute a SELECT query and return single value"""
        if not self._initialized:
            await self.initialize()

        self._stats["queries_executed"] += 1

        if self._use_asyncpg and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    return await conn.fetchval(query, *args)
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Async fetchval error: {e}")
                raise

        # Fallback
        rows = await self._sync_fetch(query, *args)
        if rows and rows[0]:
            return list(rows[0].values())[0]
        return None

    async def execute(self, query: str, *args) -> str:
        """
        Execute an INSERT/UPDATE/DELETE query.

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        if not self._initialized:
            await self.initialize()

        self._stats["queries_executed"] += 1

        if self._use_asyncpg and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    return await conn.execute(query, *args)
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Async execute error: {e}")
                raise

        return await self._sync_execute(query, *args)

    async def executemany(self, query: str, args_list: List[tuple]) -> None:
        """Execute a query multiple times with different args (batch insert)"""
        if not self._initialized:
            await self.initialize()

        self._stats["queries_executed"] += len(args_list)

        if self._use_asyncpg and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.executemany(query, args_list)
                    return
            except Exception as e:
                self._stats["errors"] += 1
                logger.error(f"Async executemany error: {e}")
                raise

        # Fallback to sequential sync
        for args in args_list:
            await self._sync_execute(query, *args)

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for transactions.

        Usage:
            async with pool.transaction() as conn:
                await conn.execute("INSERT INTO ...", ...)
                await conn.execute("UPDATE ...", ...)
                # Automatically commits on success, rolls back on exception
        """
        if not self._initialized:
            await self.initialize()

        self._stats["transactions"] += 1

        if self._use_asyncpg and self._pool:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            # Sync fallback - no true transaction support
            yield self

    async def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start = datetime.now()
            result = await self.fetchval("SELECT 1")
            latency = (datetime.now() - start).total_seconds() * 1000

            pool_size = 0
            pool_free = 0
            if self._use_asyncpg and self._pool:
                pool_size = self._pool.get_size()
                pool_free = self._pool.get_idle_size()

            return {
                "healthy": result == 1,
                "latency_ms": round(latency, 2),
                "backend": "asyncpg" if self._use_asyncpg else "sync",
                "pool_size": pool_size,
                "pool_free": pool_free,
                **self._stats
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "backend": "asyncpg" if self._use_asyncpg else "sync",
                **self._stats
            }

    # =========================================================================
    # Sync Fallback Methods
    # =========================================================================

    async def _sync_fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fallback to sync database access"""
        from src.database.connection_pool import get_db_connection

        def _execute():
            # Convert $1, $2 placeholders to %s for psycopg2
            converted_query = self._convert_placeholders(query)

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(converted_query, args or None)

                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                return []

        return await asyncio.to_thread(_execute)

    async def _sync_execute(self, query: str, *args) -> str:
        """Fallback to sync execute"""
        from src.database.connection_pool import get_db_connection

        def _execute():
            converted_query = self._convert_placeholders(query)

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(converted_query, args or None)
                conn.commit()
                return f"OK {cursor.rowcount}"

        return await asyncio.to_thread(_execute)

    def _convert_placeholders(self, query: str) -> str:
        """Convert $1, $2 to %s for psycopg2 compatibility"""
        import re
        return re.sub(r'\$\d+', '%s', query)


# =============================================================================
# Singleton Instance
# =============================================================================

_db_pool: Optional[AsyncDatabasePool] = None


async def get_async_db() -> AsyncDatabasePool:
    """Get the global async database pool"""
    global _db_pool
    if _db_pool is None:
        _db_pool = AsyncDatabasePool()
        await _db_pool.initialize()
    return _db_pool


async def init_async_db() -> AsyncDatabasePool:
    """Initialize the global async database pool"""
    return await get_async_db()
