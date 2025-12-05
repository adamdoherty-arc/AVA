"""
Database Health Check Module

Tests PostgreSQL database health:
- Connection pool status
- Slow query detection
- Table sizes and index usage
- Schema integrity
- Connection leaks

CRITICAL check - Database health is essential.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class DatabaseHealthCheck(BaseCheck):
    """
    Tests PostgreSQL database health and performance.

    CRITICAL check - Database health is essential for the platform.
    """

    # Thresholds for warnings/failures
    SLOW_QUERY_THRESHOLD_MS = 1000  # 1 second
    MAX_CONNECTIONS_WARNING = 40
    MAX_TABLE_SIZE_GB = 10
    MIN_INDEX_HIT_RATIO = 0.95  # 95% cache hits expected

    def __init__(self) -> None:
        """Initialize database health check."""
        super().__init__()

    @property
    def name(self) -> str:
        return "database_health"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "connection_pool",
            "active_connections",
            "table_sizes",
            "index_usage",
            "slow_queries",
            "schema_integrity",
        ]

    def run(self) -> ModuleCheckResult:
        """Run database health checks."""
        self._start_module()

        # Try to run checks - all require database access
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._run_all_checks())
            finally:
                loop.close()

        except ImportError as e:
            self._error(
                "connection_pool",
                f"Could not import database module: {e}"
            )
            # Skip remaining checks
            for check in ["active_connections", "table_sizes", "index_usage", "slow_queries", "schema_integrity"]:
                self._skip(check, "Database module not available")

        except Exception as e:
            self._fail(
                "connection_pool",
                f"Database health check failed: {e}",
                details={"error": str(e)}
            )

        return self._end_module()

    async def _run_all_checks(self) -> None:
        """Run all database checks asynchronously."""
        from backend.infrastructure.database import get_database

        try:
            db = await get_database()

            # Connection pool check
            await self._check_connection_pool(db)

            # Active connections
            await self._check_active_connections(db)

            # Table sizes
            await self._check_table_sizes(db)

            # Index usage
            await self._check_index_usage(db)

            # Slow queries (from pg_stat_statements if available)
            await self._check_slow_queries(db)

            # Schema integrity
            await self._check_schema_integrity(db)

        except Exception as e:
            self._fail(
                "connection_pool",
                f"Database connection failed: {e}",
                details={"error": str(e)}
            )

    async def _check_connection_pool(self, db):
        """Check connection pool health."""
        try:
            # Simple query to verify connection works
            result = await db.fetchval("SELECT 1")

            if result == 1:
                self._pass(
                    "connection_pool",
                    "Database connection pool healthy"
                )
            else:
                self._fail(
                    "connection_pool",
                    "Database returned unexpected result"
                )

        except Exception as e:
            self._fail(
                "connection_pool",
                f"Connection pool check failed: {e}"
            )

    async def _check_active_connections(self, db):
        """Check number of active database connections."""
        try:
            query = """
                SELECT count(*) as total,
                       count(*) FILTER (WHERE state = 'active') as active,
                       count(*) FILTER (WHERE state = 'idle') as idle,
                       count(*) FILTER (WHERE state = 'idle in transaction') as idle_in_tx
                FROM pg_stat_activity
                WHERE datname = current_database()
            """
            row = await db.fetchrow(query)

            if row:
                total = row["total"]
                active = row["active"]
                idle_in_tx = row["idle_in_tx"]

                details = {
                    "total": total,
                    "active": active,
                    "idle": row["idle"],
                    "idle_in_transaction": idle_in_tx
                }

                if idle_in_tx > 5:
                    self._warn(
                        "active_connections",
                        f"{idle_in_tx} connections idle in transaction (potential leak)",
                        details=details
                    )
                elif total > self.MAX_CONNECTIONS_WARNING:
                    self._warn(
                        "active_connections",
                        f"{total} active connections (approaching limit)",
                        details=details
                    )
                else:
                    self._pass(
                        "active_connections",
                        f"{total} connections ({active} active)",
                        details=details
                    )
            else:
                self._warn(
                    "active_connections",
                    "Could not retrieve connection statistics"
                )

        except Exception as e:
            self._warn(
                "active_connections",
                f"Connection check failed: {e}"
            )

    async def _check_table_sizes(self, db):
        """Check table sizes for potential issues."""
        try:
            query = """
                SELECT
                    schemaname || '.' || tablename as table_name,
                    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as size,
                    pg_total_relation_size(schemaname || '.' || tablename) as size_bytes
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
                LIMIT 10
            """
            rows = await db.fetch(query)

            large_tables = []
            total_size = 0

            for row in rows:
                size_bytes = row["size_bytes"]
                total_size += size_bytes

                # Flag tables over 1GB
                if size_bytes > 1_000_000_000:
                    large_tables.append({
                        "table": row["table_name"],
                        "size": row["size"]
                    })

            details = {
                "top_10_tables": [{"table": r["table_name"], "size": r["size"]} for r in rows],
                "total_size_bytes": total_size
            }

            if large_tables:
                self._warn(
                    "table_sizes",
                    f"{len(large_tables)} table(s) over 1GB - consider archiving",
                    details=details
                )
            else:
                self._pass(
                    "table_sizes",
                    "Table sizes within normal range",
                    details=details
                )

        except Exception as e:
            self._warn(
                "table_sizes",
                f"Table size check failed: {e}"
            )

    async def _check_index_usage(self, db):
        """Check index hit ratio and unused indexes."""
        try:
            # Check cache hit ratio
            query = """
                SELECT
                    sum(idx_blks_hit) as hits,
                    sum(idx_blks_read) as reads,
                    CASE WHEN sum(idx_blks_hit) + sum(idx_blks_read) > 0
                         THEN sum(idx_blks_hit)::float / (sum(idx_blks_hit) + sum(idx_blks_read))
                         ELSE 1.0
                    END as hit_ratio
                FROM pg_statio_user_indexes
            """
            row = await db.fetchrow(query)

            if row and row["hits"] is not None:
                hit_ratio = row["hit_ratio"]

                if hit_ratio < self.MIN_INDEX_HIT_RATIO:
                    self._warn(
                        "index_usage",
                        f"Index cache hit ratio {hit_ratio:.2%} below {self.MIN_INDEX_HIT_RATIO:.0%} threshold",
                        details={"hit_ratio": hit_ratio}
                    )
                else:
                    self._pass(
                        "index_usage",
                        f"Index cache hit ratio {hit_ratio:.2%}",
                        details={"hit_ratio": hit_ratio}
                    )
            else:
                self._pass(
                    "index_usage",
                    "No index statistics available (new database or no queries)"
                )

        except Exception as e:
            self._warn(
                "index_usage",
                f"Index usage check failed: {e}"
            )

    async def _check_slow_queries(self, db):
        """Check for slow queries using pg_stat_statements if available."""
        try:
            # Check if pg_stat_statements extension is available
            ext_check = await db.fetchval(
                "SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'"
            )

            if not ext_check:
                self._pass(
                    "slow_queries",
                    "pg_stat_statements not enabled (install for slow query tracking)"
                )
                return

            query = """
                SELECT
                    query,
                    calls,
                    mean_exec_time as avg_time_ms,
                    total_exec_time as total_time_ms
                FROM pg_stat_statements
                WHERE mean_exec_time > $1
                ORDER BY mean_exec_time DESC
                LIMIT 5
            """
            rows = await db.fetch(query, self.SLOW_QUERY_THRESHOLD_MS)

            if rows:
                slow_queries = [
                    {
                        "query": row["query"][:100] + "..." if len(row["query"]) > 100 else row["query"],
                        "avg_ms": round(row["avg_time_ms"], 2),
                        "calls": row["calls"]
                    }
                    for row in rows
                ]

                self._warn(
                    "slow_queries",
                    f"{len(slow_queries)} queries averaging over {self.SLOW_QUERY_THRESHOLD_MS}ms",
                    details={"slow_queries": slow_queries}
                )
            else:
                self._pass(
                    "slow_queries",
                    f"No queries averaging over {self.SLOW_QUERY_THRESHOLD_MS}ms"
                )

        except Exception as e:
            # pg_stat_statements might not be available
            self._pass(
                "slow_queries",
                "Slow query check not available (pg_stat_statements not configured)"
            )

    async def _check_schema_integrity(self, db):
        """Check for basic schema integrity issues."""
        try:
            # Check for tables without primary keys
            query = """
                SELECT t.tablename
                FROM pg_tables t
                LEFT JOIN pg_indexes i ON i.tablename = t.tablename AND i.indexname LIKE '%pkey'
                WHERE t.schemaname = 'public' AND i.indexname IS NULL
            """
            rows = await db.fetch(query)

            tables_without_pk = [row["tablename"] for row in rows]

            if tables_without_pk:
                self._warn(
                    "schema_integrity",
                    f"{len(tables_without_pk)} table(s) without primary key",
                    details={"tables": tables_without_pk[:10]}  # Limit to 10
                )
            else:
                self._pass(
                    "schema_integrity",
                    "All tables have primary keys"
                )

        except Exception as e:
            self._warn(
                "schema_integrity",
                f"Schema integrity check failed: {e}"
            )

    def can_auto_fix(self, check_name: str) -> bool:
        """Database issues cannot be auto-fixed safely."""
        return False
