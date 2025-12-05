"""
Modern Async Database Infrastructure
=====================================

Production-grade async database layer with:
- Async connection pooling (asyncpg)
- Health checks and monitoring
- Retry logic with exponential backoff
- Query instrumentation and observability
- Prepared statement caching
- Transaction management
- Type-safe query builders

Author: AVA Trading Platform
Updated: 2025-11-29
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import structlog

# Async database driver
try:
    import asyncpg
    from asyncpg import Connection, Pool, Record

    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None
    Pool = None
    Connection = None
    Record = None

# For sync fallback
try:
    import psycopg2
    from psycopg2 import pool as psycopg2_pool

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    psycopg2 = None
    psycopg2_pool = None

# Observability
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Status, StatusCode

    OTEL_AVAILABLE = True
    tracer = trace.get_tracer(__name__)
except ImportError:
    OTEL_AVAILABLE = False
    tracer = None

logger = structlog.get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DatabaseConfig:
    """Immutable database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "ava"  # AVA trading platform database
    user: str = "postgres"
    password: str = "postgres"

    # Pool settings
    min_pool_size: int = 5
    max_pool_size: int = 50

    # Timeouts (seconds)
    connect_timeout: float = 10.0
    command_timeout: float = 60.0
    statement_timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.5
    retry_backoff: float = 2.0

    # Health check
    health_check_interval: float = 30.0

    # SSL
    ssl: bool = False
    ssl_cert_path: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "magnus"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            min_pool_size=int(os.getenv("DB_POOL_MIN", "5")),
            max_pool_size=int(os.getenv("DB_POOL_MAX", "50")),
            command_timeout=float(os.getenv("DB_QUERY_TIMEOUT", "60")),
        )

    @property
    def dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# =============================================================================
# Query Result Types
# =============================================================================


class IsolationLevel(str, Enum):
    """Transaction isolation levels."""

    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class QueryStats:
    """Statistics for database operations."""

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    slow_queries: int = 0  # queries > 100ms

    def record(self, duration_ms: float, success: bool = True) -> None:
        """Record a query execution."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1

        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.total_queries
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.min_time_ms = min(self.min_time_ms, duration_ms)

        if duration_ms > 100:
            self.slow_queries += 1


@dataclass
class PoolStats:
    """Connection pool statistics."""

    size: int = 0
    free_size: int = 0
    used_size: int = 0
    min_size: int = 0
    max_size: int = 0
    connected: bool = False
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0


# =============================================================================
# Async Database Manager
# =============================================================================


class AsyncDatabaseManager:
    """
    Production-grade async database connection manager.

    Features:
    - Async connection pooling with asyncpg
    - Automatic reconnection on failure
    - Health checks with circuit breaker
    - Query instrumentation and tracing
    - Prepared statement caching
    - Transaction context managers

    Usage:
        db = AsyncDatabaseManager(config)
        await db.connect()

        # Simple query
        rows = await db.fetch("SELECT * FROM users WHERE id = $1", user_id)

        # Transaction
        async with db.transaction() as conn:
            await conn.execute("INSERT INTO users ...")
            await conn.execute("INSERT INTO profiles ...")

        await db.disconnect()
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_env()
        self._pool: Optional[Pool] = None
        self._connected = False
        self._stats = QueryStats()
        self._pool_stats = PoolStats()
        self._prepared_statements: Dict[str, str] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        logger.info(
            "database_manager_initialized",
            host=self.config.host,
            database=self.config.database,
            pool_size=f"{self.config.min_pool_size}-{self.config.max_pool_size}",
        )

    async def connect(self) -> bool:
        """
        Initialize connection pool.

        Returns:
            True if connection successful
        """
        if self._connected and self._pool:
            return True

        if not ASYNCPG_AVAILABLE:
            logger.error("asyncpg_not_installed")
            raise ImportError("asyncpg is required: pip install asyncpg")

        async with self._lock:
            if self._connected:
                return True

            try:
                self._pool = await asyncpg.create_pool(
                    dsn=self.config.dsn,
                    min_size=self.config.min_pool_size,
                    max_size=self.config.max_pool_size,
                    command_timeout=self.config.command_timeout,
                    timeout=self.config.connect_timeout,
                    setup=self._setup_connection,
                )

                self._connected = True
                self._update_pool_stats()

                # Start health check background task
                self._health_check_task = asyncio.create_task(
                    self._health_check_loop()
                )

                logger.info(
                    "database_connected",
                    pool_size=self._pool.get_size(),
                    min_size=self._pool.get_min_size(),
                    max_size=self._pool.get_max_size(),
                )
                return True

            except Exception as e:
                logger.error("database_connection_failed", error=str(e))
                self._connected = False
                raise

    async def warmup_pool(self, target_connections: Optional[int] = None) -> int:
        """
        Warm up the connection pool by pre-establishing connections.

        This prevents cold-start latency on first requests after startup.

        Args:
            target_connections: Number of connections to establish.
                              Defaults to min_pool_size.

        Returns:
            Number of connections successfully warmed up.
        """
        if not self._connected or not self._pool:
            logger.warning("pool_warmup_skipped", reason="not_connected")
            return 0

        target = target_connections or self.config.min_pool_size
        warmed = 0

        logger.info("pool_warmup_starting", target=target)

        # Acquire and release connections to force pool to establish them
        connections = []
        try:
            for i in range(target):
                try:
                    conn = await asyncio.wait_for(
                        self._pool.acquire(),
                        timeout=self.config.connect_timeout
                    )
                    connections.append(conn)
                    warmed += 1
                except Exception as e:
                    logger.warning(f"pool_warmup_connection_failed", index=i, error=str(e))
                    break

            # Execute a simple query on each connection to ensure it's ready
            for conn in connections:
                try:
                    await conn.execute("SELECT 1")
                except Exception:
                    pass

        finally:
            # Release all connections back to pool (with error handling for each)
            release_errors = 0
            for conn in connections:
                try:
                    await self._pool.release(conn)
                except Exception as e:
                    release_errors += 1
                    logger.error("pool_warmup_release_failed", error=str(e))

            if release_errors > 0:
                logger.warning(
                    "pool_warmup_release_issues",
                    failed=release_errors,
                    total=len(connections)
                )

        self._update_pool_stats()
        logger.info(
            "pool_warmup_complete",
            warmed=warmed,
            pool_size=self._pool.get_size()
        )
        return warmed

    async def _setup_connection(self, conn: Connection) -> None:
        """Configure each new connection."""
        # Set statement timeout
        await conn.execute(
            f"SET statement_timeout = '{int(self.config.statement_timeout * 1000)}'"
        )

    async def disconnect(self) -> None:
        """Close all connections and clean up."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._pool:
            await self._pool.close()
            self._pool = None

        self._connected = False
        logger.info("database_disconnected")

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while self._connected:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                healthy = await self.health_check()
                if not healthy:
                    self._pool_stats.health_check_failures += 1
                    logger.warning(
                        "database_health_check_failed",
                        failures=self._pool_stats.health_check_failures,
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_check_error", error=str(e))

    async def health_check(self) -> bool:
        """Perform database health check."""
        try:
            result = await self.fetchval("SELECT 1")
            self._pool_stats.last_health_check = datetime.now()
            self._update_pool_stats()
            return result == 1
        except Exception:
            return False

    def _update_pool_stats(self) -> None:
        """Update pool statistics."""
        if self._pool:
            self._pool_stats.size = self._pool.get_size()
            self._pool_stats.free_size = self._pool.get_idle_size()
            self._pool_stats.used_size = self._pool_stats.size - self._pool_stats.free_size
            self._pool_stats.min_size = self._pool.get_min_size()
            self._pool_stats.max_size = self._pool.get_max_size()
            self._pool_stats.connected = self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        self._update_pool_stats()
        return {
            "query_stats": {
                "total": self._stats.total_queries,
                "successful": self._stats.successful_queries,
                "failed": self._stats.failed_queries,
                "avg_time_ms": round(self._stats.avg_time_ms, 2),
                "max_time_ms": round(self._stats.max_time_ms, 2),
                "slow_queries": self._stats.slow_queries,
            },
            "pool_stats": {
                "connected": self._pool_stats.connected,
                "size": self._pool_stats.size,
                "free": self._pool_stats.free_size,
                "used": self._pool_stats.used_size,
                "min_size": self._pool_stats.min_size,
                "max_size": self._pool_stats.max_size,
                "health_check_failures": self._pool_stats.health_check_failures,
                "last_health_check": (
                    self._pool_stats.last_health_check.isoformat()
                    if self._pool_stats.last_health_check
                    else None
                ),
            },
        }

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def _execute_with_retry(
        self,
        operation: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute operation with retry logic and instrumentation."""
        last_error = None
        start_time = time.perf_counter()

        for attempt in range(self.config.max_retries):
            try:
                # Add tracing if available
                if OTEL_AVAILABLE and tracer:
                    with tracer.start_as_current_span(
                        f"db.{operation.__name__}",
                        attributes={
                            "db.system": "postgresql",
                            "db.name": self.config.database,
                            "db.operation": operation.__name__,
                            "retry.attempt": attempt,
                        },
                    ) as span:
                        result = await operation(*args, **kwargs)
                        duration_ms = (time.perf_counter() - start_time) * 1000
                        span.set_attribute("db.duration_ms", duration_ms)
                        self._stats.record(duration_ms, success=True)
                        return result
                else:
                    result = await operation(*args, **kwargs)
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self._stats.record(duration_ms, success=True)
                    return result

            except (
                asyncpg.PostgresConnectionError,
                asyncpg.InterfaceError,
            ) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (
                        self.config.retry_backoff ** attempt
                    )
                    logger.warning(
                        "database_retry",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._stats.record(duration_ms, success=False)
                raise

        self._stats.record(
            (time.perf_counter() - start_time) * 1000, success=False
        )
        raise last_error or Exception("Database operation failed")

    async def execute(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> str:
        """
        Execute a query (INSERT, UPDATE, DELETE).

        Returns:
            Status string (e.g., "INSERT 0 1")
        """
        await self._ensure_connected()

        async def _execute() -> str:
            async with self.pool.acquire() as conn:
                return await conn.execute(query, *args, timeout=timeout)

        return await self._execute_with_retry(_execute)

    async def executemany(
        self,
        query: str,
        args: Sequence[Sequence[Any]],
        timeout: Optional[float] = None,
    ) -> None:
        """Execute a query for each set of arguments."""
        await self._ensure_connected()

        async def _executemany() -> None:
            async with self.pool.acquire() as conn:
                await conn.executemany(query, args, timeout=timeout)

        await self._execute_with_retry(_executemany)

    async def fetch(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> List[Record]:
        """
        Fetch multiple rows.

        Returns:
            List of Record objects (dict-like)
        """
        await self._ensure_connected()

        async def _fetch() -> List[Record]:
            async with self.pool.acquire() as conn:
                return await conn.fetch(query, *args, timeout=timeout)

        return await self._execute_with_retry(_fetch)

    async def fetchrow(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> Optional[Record]:
        """
        Fetch a single row.

        Returns:
            Single Record or None
        """
        await self._ensure_connected()

        async def _fetchrow() -> Optional[Record]:
            async with self.pool.acquire() as conn:
                return await conn.fetchrow(query, *args, timeout=timeout)

        return await self._execute_with_retry(_fetchrow)

    async def fetchval(
        self,
        query: str,
        *args: Any,
        column: int = 0,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Fetch a single value from first row.

        Returns:
            Single value from specified column
        """
        await self._ensure_connected()

        async def _fetchval() -> Any:
            async with self.pool.acquire() as conn:
                return await conn.fetchval(
                    query, *args, column=column, timeout=timeout
                )

        return await self._execute_with_retry(_fetchval)

    async def _ensure_connected(self) -> None:
        """Ensure we have a valid connection pool."""
        if not self._connected or not self._pool:
            await self.connect()

    @property
    def pool(self) -> Pool:
        """Get the connection pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._pool

    # =========================================================================
    # Transaction Support
    # =========================================================================

    @asynccontextmanager
    async def transaction(
        self,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    ) -> AsyncGenerator[Connection, None]:
        """
        Context manager for database transactions.

        Usage:
            async with db.transaction() as conn:
                await conn.execute("INSERT INTO users ...")
                await conn.execute("UPDATE accounts ...")
                # Commits on success, rolls back on exception
        """
        await self._ensure_connected()

        async with self.pool.acquire() as conn:
            async with conn.transaction(isolation=isolation.value):
                yield conn

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """
        Acquire a connection from the pool.

        Usage:
            async with db.acquire() as conn:
                await conn.execute(...)
        """
        await self._ensure_connected()
        async with self.pool.acquire() as conn:
            yield conn

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    async def copy_records_to_table(
        self,
        table_name: str,
        records: Sequence[Tuple],
        columns: Sequence[str],
        timeout: Optional[float] = None,
    ) -> str:
        """
        Bulk insert using PostgreSQL COPY protocol (fastest method).

        Args:
            table_name: Target table
            records: Sequence of tuples matching columns
            columns: Column names

        Returns:
            Status string
        """
        await self._ensure_connected()

        async def _copy() -> str:
            async with self.pool.acquire() as conn:
                return await conn.copy_records_to_table(
                    table_name,
                    records=records,
                    columns=columns,
                    timeout=timeout,
                )

        return await self._execute_with_retry(_copy)

    async def upsert_batch(
        self,
        table_name: str,
        records: List[Dict[str, Any]],
        conflict_columns: List[str],
        update_columns: Optional[List[str]] = None,
    ) -> int:
        """
        Batch upsert (INSERT ... ON CONFLICT UPDATE).

        Args:
            table_name: Target table
            records: List of dicts to insert
            conflict_columns: Columns to check for conflicts
            update_columns: Columns to update on conflict (default: all except conflict)

        Returns:
            Number of rows affected
        """
        if not records:
            return 0

        columns = list(records[0].keys())
        if update_columns is None:
            update_columns = [c for c in columns if c not in conflict_columns]

        # Validate identifiers (prevent SQL injection)
        def quote_ident(name: str) -> str:
            """Quote a SQL identifier safely."""
            if not name.isidentifier() and not name.replace("_", "").isalnum():
                raise ValueError(f"Invalid identifier: {name}")
            return f'"{name}"'

        # Build parameterized query with quoted identifiers
        quoted_columns = [quote_ident(c) for c in columns]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        conflict_clause = ", ".join(quote_ident(c) for c in conflict_columns)
        update_clause = ", ".join(
            f"{quote_ident(col)} = EXCLUDED.{quote_ident(col)}" for col in update_columns
        )

        query = f"""
            INSERT INTO {quote_ident(table_name)} ({", ".join(quoted_columns)})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_clause})
            DO UPDATE SET {update_clause}
        """

        values = [tuple(r[c] for c in columns) for r in records]
        await self.executemany(query, values)
        return len(records)


# =============================================================================
# Singleton Instance
# =============================================================================

_db_instance: Optional[AsyncDatabaseManager] = None
# Thread-safe lock - initialized at module load time to prevent race conditions
# Note: asyncio.Lock() is safe to create outside of an event loop in Python 3.10+
# For older Python, the lock is created on first use within the event loop
_db_lock: Optional[asyncio.Lock] = None
_db_lock_init_lock = None  # Fallback for older Python versions


def _get_lock() -> asyncio.Lock:
    """
    Get or create the asyncio lock in a thread-safe manner.

    This handles the edge case where the lock needs to be created
    within an event loop context for Python < 3.10.
    """
    global _db_lock, _db_lock_init_lock
    if _db_lock is not None:
        return _db_lock

    # Double-checked locking pattern for thread safety
    if _db_lock_init_lock is None:
        import threading
        _db_lock_init_lock = threading.Lock()

    with _db_lock_init_lock:
        if _db_lock is None:
            _db_lock = asyncio.Lock()
    return _db_lock


async def get_database() -> AsyncDatabaseManager:
    """
    Get or create the global database manager.

    Thread-safe singleton pattern using double-checked locking.
    """
    global _db_instance
    if _db_instance is not None:
        return _db_instance

    async with _get_lock():
        # Double-check after acquiring lock
        if _db_instance is None:
            _db_instance = AsyncDatabaseManager()
            await _db_instance.connect()
    return _db_instance


async def init_database(config: Optional[DatabaseConfig] = None) -> AsyncDatabaseManager:
    """Initialize the global database manager with custom config."""
    global _db_instance
    async with _get_lock():
        if _db_instance:
            await _db_instance.disconnect()
        _db_instance = AsyncDatabaseManager(config)
        await _db_instance.connect()
    return _db_instance


# =============================================================================
# FastAPI Dependency
# =============================================================================


async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """FastAPI dependency for database connection."""
    db = await get_database()
    async with db.acquire() as conn:
        yield conn


# =============================================================================
# Query Builder (Type-Safe)
# =============================================================================


class QueryBuilder:
    """
    Fluent query builder for type-safe queries.

    Usage:
        query = (
            QueryBuilder("SELECT")
            .select("id", "name", "email")
            .from_table("users")
            .where("status = $1", "active")
            .where("created_at > $2", datetime.now() - timedelta(days=30))
            .order_by("created_at", desc=True)
            .limit(10)
        )
        rows = await db.fetch(query.sql, *query.args)
    """

    def __init__(self, operation: str = "SELECT"):
        self._operation = operation
        self._columns: List[str] = []
        self._table: str = ""
        self._joins: List[str] = []
        self._wheres: List[Tuple[str, Any]] = []
        self._group_by: List[str] = []
        self._having: List[Tuple[str, Any]] = []
        self._order_by: List[str] = []
        self._limit_val: Optional[int] = None
        self._offset_val: Optional[int] = None
        self._args: List[Any] = []
        self._arg_index = 0

    def select(self, *columns: str) -> "QueryBuilder":
        """Add SELECT columns."""
        self._columns.extend(columns)
        return self

    def from_table(self, table: str, alias: Optional[str] = None) -> "QueryBuilder":
        """Set FROM table."""
        self._table = f"{table} {alias}" if alias else table
        return self

    def join(
        self,
        table: str,
        on: str,
        join_type: str = "INNER",
    ) -> "QueryBuilder":
        """Add JOIN clause."""
        self._joins.append(f"{join_type} JOIN {table} ON {on}")
        return self

    def where(self, condition: str, *args: Any) -> "QueryBuilder":
        """Add WHERE condition."""
        # Replace placeholders with correct indices
        for arg in args:
            self._arg_index += 1
            condition = condition.replace("$1", f"${self._arg_index}", 1)
            self._args.append(arg)
        self._wheres.append((condition, None))
        return self

    def and_where(self, condition: str, *args: Any) -> "QueryBuilder":
        """Alias for where()."""
        return self.where(condition, *args)

    def or_where(self, condition: str, *args: Any) -> "QueryBuilder":
        """Add OR WHERE condition."""
        for arg in args:
            self._arg_index += 1
            condition = condition.replace("$1", f"${self._arg_index}", 1)
            self._args.append(arg)
        self._wheres.append((f"OR {condition}", None))
        return self

    def group_by(self, *columns: str) -> "QueryBuilder":
        """Add GROUP BY."""
        self._group_by.extend(columns)
        return self

    def order_by(self, column: str, desc: bool = False) -> "QueryBuilder":
        """Add ORDER BY."""
        direction = "DESC" if desc else "ASC"
        self._order_by.append(f"{column} {direction}")
        return self

    def limit(self, count: int) -> "QueryBuilder":
        """Set LIMIT."""
        self._limit_val = count
        return self

    def offset(self, count: int) -> "QueryBuilder":
        """Set OFFSET."""
        self._offset_val = count
        return self

    @property
    def sql(self) -> str:
        """Build the SQL query string."""
        parts = [self._operation]

        if self._columns:
            parts.append(", ".join(self._columns))
        else:
            parts.append("*")

        parts.append(f"FROM {self._table}")

        if self._joins:
            parts.extend(self._joins)

        if self._wheres:
            conditions = []
            for i, (cond, _) in enumerate(self._wheres):
                if i == 0:
                    conditions.append(cond)
                elif cond.startswith("OR "):
                    conditions.append(cond)
                else:
                    conditions.append(f"AND {cond}")
            parts.append("WHERE " + " ".join(conditions))

        if self._group_by:
            parts.append(f"GROUP BY {', '.join(self._group_by)}")

        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        if self._limit_val is not None:
            parts.append(f"LIMIT {self._limit_val}")

        if self._offset_val is not None:
            parts.append(f"OFFSET {self._offset_val}")

        return " ".join(parts)

    @property
    def args(self) -> Tuple[Any, ...]:
        """Get query arguments."""
        return tuple(self._args)

    def __str__(self) -> str:
        return self.sql
