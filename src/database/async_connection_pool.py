"""
Async Database Connection Pool
================================

AsyncPG-based connection pool for high-performance async database operations.

Benefits over psycopg2:
- True async/await support
- Better performance (2-3x faster for concurrent queries)
- Connection pooling with automatic lifecycle management
- Better resource utilization

Author: Magnus Trading Platform
Created: 2025-11-21
"""

import logging
import os
from typing import Optional, Any, List, Dict
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# ============================================================================
# Async Connection Pool
# ============================================================================

class AsyncConnectionPool:
    """
    Async PostgreSQL connection pool using asyncpg

    Manages connection lifecycle, pooling, and provides async context managers.
    """

    _instance: Optional['AsyncConnectionPool'] = None
    _pool = None

    def __new__(cls):
        """Singleton pattern for connection pool"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def initialize(
        self,
        min_size: int = 5,
        max_size: int = 20,
        timeout: float = 30.0
    ):
        """
        Initialize connection pool

        Args:
            min_size: Minimum pool size
            max_size: Maximum pool size
            timeout: Connection timeout in seconds
        """
        if self._pool is not None:
            logger.warning("Pool already initialized")
            return

        try:
            import asyncpg

            # Get database credentials from environment
            db_config = {
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432')),
                'database': os.getenv('POSTGRES_DB', 'magnus'),
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
            }

            self._pool = await asyncpg.create_pool(
                **db_config,
                min_size=min_size,
                max_size=max_size,
                command_timeout=timeout
            )

            logger.info(f"Async connection pool initialized (min={min_size}, max={max_size})")

        except Exception as e:
            logger.error(f"Failed to initialize async connection pool: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Async connection pool closed")

    @asynccontextmanager
    async def acquire(self) -> None:
        """
        Acquire connection from pool

        Usage:
            async with pool.acquire() as conn:
                result = await conn.fetch("SELECT * FROM table")
        """
        if self._pool is None:
            await self.initialize()

        async with self._pool.acquire() as connection:
            yield connection

    async def execute(self, query: str, *args) -> str:
        """
        Execute a query that doesn't return results

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Query status (e.g., "INSERT 0 1")
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """
        Fetch all results from query

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of rows as dicts
        """
        async with self.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """
        Fetch single row from query

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single row as dict or None
        """
        async with self.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """
        Fetch single value from query

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    @asynccontextmanager
    async def transaction(self) -> None:
        """
        Transaction context manager

        Usage:
            async with pool.transaction():
                await pool.execute("INSERT ...")
                await pool.execute("UPDATE ...")
        """
        async with self.acquire() as conn:
            async with conn.transaction():
                yield conn


# ============================================================================
# Global Pool Instance
# ============================================================================

_global_pool: Optional[AsyncConnectionPool] = None


async def get_async_pool() -> AsyncConnectionPool:
    """
    Get global async connection pool

    Returns:
        AsyncConnectionPool instance
    """
    global _global_pool

    if _global_pool is None:
        _global_pool = AsyncConnectionPool()
        await _global_pool.initialize()

    return _global_pool


async def close_async_pool():
    """Close global async connection pool"""
    global _global_pool

    if _global_pool:
        await _global_pool.close()
        _global_pool = None


# ============================================================================
# Sync/Async Compatibility Layer
# ============================================================================

def run_async(coro):
    """
    Run async function from sync context

    Args:
        coro: Coroutine to run

    Returns:
        Result of coroutine
    """
    try:
        # Try to get running loop
        loop = asyncio.get_running_loop()

        # We're already in async context, just await
        return coro

    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)


class SyncAsyncWrapper:
    """
    Wrapper to use async functions in sync contexts

    Provides sync-compatible methods that internally use async pool.
    """

    def __init__(self) -> None:
        self._pool: Optional[AsyncConnectionPool] = None

    def _ensure_pool(self) -> None:
        """Ensure pool is initialized"""
        if self._pool is None:
            self._pool = AsyncConnectionPool()
            run_async(self._pool.initialize())

    def execute(self, query: str, *args) -> str:
        """Sync wrapper for execute"""
        self._ensure_pool()
        return run_async(self._pool.execute(query, *args))

    def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Sync wrapper for fetch"""
        self._ensure_pool()
        return run_async(self._pool.fetch(query, *args))

    def fetchrow(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Sync wrapper for fetchrow"""
        self._ensure_pool()
        return run_async(self._pool.fetchrow(query, *args))

    def fetchval(self, query: str, *args) -> Any:
        """Sync wrapper for fetchval"""
        self._ensure_pool()
        return run_async(self._pool.fetchval(query, *args))


# ============================================================================
# Convenience Functions
# ============================================================================

async def execute_query(query: str, *args) -> str:
    """
    Execute query (convenience function)

    Args:
        query: SQL query
        *args: Query parameters

    Returns:
        Query status
    """
    pool = await get_async_pool()
    return await pool.execute(query, *args)


async def fetch_all(query: str, *args) -> List[Dict[str, Any]]:
    """
    Fetch all rows (convenience function)

    Args:
        query: SQL query
        *args: Query parameters

    Returns:
        List of rows
    """
    pool = await get_async_pool()
    return await pool.fetch(query, *args)


async def fetch_one(query: str, *args) -> Optional[Dict[str, Any]]:
    """
    Fetch one row (convenience function)

    Args:
        query: SQL query
        *args: Query parameters

    Returns:
        Single row or None
    """
    pool = await get_async_pool()
    return await pool.fetchrow(query, *args)


async def fetch_value(query: str, *args) -> Any:
    """
    Fetch single value (convenience function)

    Args:
        query: SQL query
        *args: Query parameters

    Returns:
        Single value
    """
    pool = await get_async_pool()
    return await pool.fetchval(query, *args)


# ============================================================================
# Testing
# ============================================================================

async def test_async_pool():
    """Test async connection pool"""
    pool = await get_async_pool()

    # Test simple query
    result = await pool.fetchval("SELECT 1 + 1")
    assert result == 2, "Basic query failed"

    # Test fetch
    rows = await pool.fetch("SELECT * FROM pg_database LIMIT 5")
    assert len(rows) > 0, "Fetch query failed"

    # Test transaction
    async with pool.transaction():
        await pool.execute("CREATE TEMP TABLE test_table (id INT, name TEXT)")
        await pool.execute("INSERT INTO test_table VALUES (1, 'test')")
        result = await pool.fetchval("SELECT COUNT(*) FROM test_table")
        assert result == 1, "Transaction test failed"

    print("✅ All async pool tests passed")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_async_pool())
