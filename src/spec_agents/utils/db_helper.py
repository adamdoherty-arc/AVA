"""
Database Helper for SpecAgents

Provides:
- Database connection pool access
- Common query patterns
- Data validation helpers
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SpecDBHelper:
    """
    Database helper for SpecAgent data validation.

    Features:
    - Connection pool management
    - Common query patterns
    - Data freshness checking
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database helper

        Args:
            database_url: Database connection URL (default: from env)
        """
        self.database_url = database_url or os.getenv(
            'DATABASE_URL',
            'postgresql://postgres:postgres@localhost:5432/magnus'
        )
        self._pool = None

    async def _get_pool(self) -> None:
        """Get or create database connection pool"""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=30,
                )
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                raise

        return self._pool

    async def query(self, sql: str, *args) -> List[Dict[str, Any]]:
        """
        Execute query and return results as list of dicts

        Args:
            sql: SQL query
            *args: Query parameters

        Returns:
            List of row dictionaries
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *args)
            return [dict(row) for row in rows]

    async def query_one(self, sql: str, *args) -> Optional[Dict[str, Any]]:
        """
        Execute query and return single result

        Args:
            sql: SQL query
            *args: Query parameters

        Returns:
            Row dictionary or None
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *args)
            return dict(row) if row else None

    async def query_value(self, sql: str, *args) -> Any:
        """
        Execute query and return single value

        Args:
            sql: SQL query
            *args: Query parameters

        Returns:
            Single value or None
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(sql, *args)

    async def close(self) -> None:
        """Close database pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # Common validation queries

    async def check_data_freshness(
        self,
        table: str,
        timestamp_column: str = 'updated_at',
        max_age_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Check if data in table is fresh

        Returns:
            Dict with is_fresh, last_update, age_minutes
        """
        try:
            result = await self.query_one(f"""
                SELECT MAX({timestamp_column}) as last_update
                FROM {table}
            """)

            if not result or not result.get('last_update'):
                return {
                    'is_fresh': False,
                    'last_update': None,
                    'age_minutes': None,
                    'error': 'No data found',
                }

            last_update = result['last_update']
            age = datetime.now(last_update.tzinfo) - last_update
            age_minutes = age.total_seconds() / 60

            return {
                'is_fresh': age_minutes <= max_age_minutes,
                'last_update': last_update.isoformat(),
                'age_minutes': round(age_minutes, 1),
            }

        except Exception as e:
            return {
                'is_fresh': False,
                'last_update': None,
                'age_minutes': None,
                'error': str(e),
            }

    async def count_rows(self, table: str, where: Optional[str] = None) -> int:
        """
        Count rows in table

        Args:
            table: Table name
            where: Optional WHERE clause

        Returns:
            Row count
        """
        sql = f"SELECT COUNT(*) FROM {table}"
        if where:
            sql += f" WHERE {where}"

        return await self.query_value(sql) or 0

    async def check_table_exists(self, table: str) -> bool:
        """Check if table exists"""
        result = await self.query_value("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """, table)
        return bool(result)

    async def get_table_columns(self, table: str) -> List[str]:
        """Get list of columns in table"""
        rows = await self.query("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table)
        return [row['column_name'] for row in rows]

    async def check_null_values(
        self,
        table: str,
        columns: List[str],
    ) -> Dict[str, int]:
        """
        Check for NULL values in specified columns

        Returns:
            Dict mapping column name to NULL count
        """
        results = {}
        for col in columns:
            count = await self.query_value(f"""
                SELECT COUNT(*) FROM {table}
                WHERE {col} IS NULL
            """)
            results[col] = count or 0
        return results

    async def get_positions_summary(self) -> Optional[Dict[str, Any]]:
        """Get positions summary for validation"""
        return await self.query_one("""
            SELECT
                COUNT(*) as total_positions,
                SUM(CASE WHEN type = 'stock' THEN 1 ELSE 0 END) as stock_count,
                SUM(CASE WHEN type = 'option' THEN 1 ELSE 0 END) as option_count,
                SUM(current_value) as total_value,
                SUM(pl) as total_pl
            FROM positions
            WHERE is_active = true
        """)

    async def get_options_with_missing_greeks(self) -> List[Dict[str, Any]]:
        """Find option positions missing Greeks data"""
        return await self.query("""
            SELECT symbol, expiration, strike, type
            FROM option_positions
            WHERE delta IS NULL OR gamma IS NULL OR theta IS NULL
            AND is_active = true
        """)

    async def validate_portfolio_totals(
        self,
        expected_total: float,
        tolerance: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Validate portfolio total matches sum of positions

        Returns:
            Dict with is_valid, calculated_total, difference
        """
        result = await self.query_one("""
            SELECT SUM(current_value) as calculated_total
            FROM positions
            WHERE is_active = true
        """)

        calculated = result.get('calculated_total', 0) or 0
        diff = abs(expected_total - calculated)

        return {
            'is_valid': diff <= tolerance,
            'expected_total': expected_total,
            'calculated_total': calculated,
            'difference': diff,
        }
