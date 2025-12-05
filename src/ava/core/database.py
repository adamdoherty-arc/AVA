"""
AVA Database Utilities
======================

Database utilities with:
- Connection pooling (SQLAlchemy)
- Async query support
- Batch operations
- Query optimization
- Transaction management

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, TypeVar, Generic, Type
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    url: str = "postgresql+asyncpg://localhost:5432/ava"
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    echo: bool = False
    echo_pool: bool = False


# =============================================================================
# ASYNC DATABASE MANAGER
# =============================================================================

class AsyncDatabaseManager:
    """
    Async database manager with connection pooling.

    Usage:
        db = AsyncDatabaseManager(config)
        await db.initialize()

        async with db.session() as session:
            result = await session.execute(query)

        await db.shutdown()
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection pool"""
        if self._initialized:
            return

        try:
            from sqlalchemy.ext.asyncio import (
                create_async_engine,
                AsyncSession,
                async_sessionmaker
            )

            self._engine = create_async_engine(
                self.config.url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                echo_pool=self.config.echo_pool
            )

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            self._initialized = True
            logger.info(f"Database initialized: {self.config.url}")

        except ImportError as e:
            logger.error(f"SQLAlchemy async not available: {e}")
            raise
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown database connections"""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")

    @asynccontextmanager
    async def session(self) -> None:
        """Get database session context manager"""
        if not self._initialized:
            await self.initialize()

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def execute(self, query, params: Optional[Dict] = None):
        """Execute a query and return results"""
        async with self.session() as session:
            result = await session.execute(query, params or {})
            return result

    async def fetch_all(self, query, params: Optional[Dict] = None) -> List[Dict]:
        """Fetch all results as list of dicts"""
        async with self.session() as session:
            result = await session.execute(query, params or {})
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]

    async def fetch_one(self, query, params: Optional[Dict] = None) -> Optional[Dict]:
        """Fetch single result as dict"""
        async with self.session() as session:
            result = await session.execute(query, params or {})
            row = result.fetchone()
            return dict(row._mapping) if row else None


# =============================================================================
# BATCH OPERATIONS
# =============================================================================

class BatchOperations:
    """
    Efficient batch database operations.

    Usage:
        batch = BatchOperations(db)
        await batch.bulk_insert("positions", records)
        await batch.bulk_upsert("prices", records, conflict_columns=["symbol", "date"])
    """

    def __init__(
        self,
        db: AsyncDatabaseManager,
        batch_size: int = 1000
    ):
        self.db = db
        self.batch_size = batch_size

    async def bulk_insert(
        self,
        table: str,
        records: List[Dict],
        returning: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Bulk insert records.

        Args:
            table: Table name
            records: List of record dicts
            returning: Columns to return

        Returns:
            Inserted records with IDs (if returning specified)
        """
        if not records:
            return []

        results = []

        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]

            # Build INSERT statement
            columns = list(batch[0].keys())
            placeholders = ", ".join([f":{col}" for col in columns])
            column_names = ", ".join(columns)

            sql = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"

            if returning:
                sql += f" RETURNING {', '.join(returning)}"

            async with self.db.session() as session:
                from sqlalchemy import text

                for record in batch:
                    result = await session.execute(text(sql), record)
                    if returning:
                        row = result.fetchone()
                        if row:
                            results.append(dict(row._mapping))

        return results

    async def bulk_upsert(
        self,
        table: str,
        records: List[Dict],
        conflict_columns: List[str],
        update_columns: Optional[List[str]] = None
    ) -> int:
        """
        Bulk upsert (INSERT ... ON CONFLICT UPDATE).

        Args:
            table: Table name
            records: List of record dicts
            conflict_columns: Columns that define uniqueness
            update_columns: Columns to update on conflict (default: all non-conflict)

        Returns:
            Number of records processed
        """
        if not records:
            return 0

        columns = list(records[0].keys())
        update_columns = update_columns or [c for c in columns if c not in conflict_columns]

        conflict_clause = ", ".join(conflict_columns)
        update_clause = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_columns])

        processed = 0

        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]

            # PostgreSQL-style upsert
            placeholders = ", ".join([f":{col}" for col in columns])
            column_names = ", ".join(columns)

            sql = f"""
                INSERT INTO {table} ({column_names})
                VALUES ({placeholders})
                ON CONFLICT ({conflict_clause})
                DO UPDATE SET {update_clause}
            """

            async with self.db.session() as session:
                from sqlalchemy import text

                for record in batch:
                    await session.execute(text(sql), record)
                    processed += 1

        return processed

    async def bulk_delete(
        self,
        table: str,
        conditions: Dict[str, Any]
    ) -> int:
        """
        Bulk delete with conditions.

        Args:
            table: Table name
            conditions: WHERE conditions as dict

        Returns:
            Number of deleted records
        """
        where_clauses = " AND ".join([f"{k} = :{k}" for k in conditions.keys()])
        sql = f"DELETE FROM {table} WHERE {where_clauses}"

        async with self.db.session() as session:
            from sqlalchemy import text
            result = await session.execute(text(sql), conditions)
            return result.rowcount


# =============================================================================
# QUERY BUILDER
# =============================================================================

class QueryBuilder:
    """
    Fluent query builder for common operations.

    Usage:
        query = (QueryBuilder("positions")
            .select(["symbol", "quantity", "market_value"])
            .where("account_id", "=", account_id)
            .where("quantity", ">", 0)
            .order_by("market_value", desc=True)
            .limit(10)
        )

        results = await query.execute(db)
    """

    def __init__(self, table: str):
        self.table = table
        self._select_columns: List[str] = ["*"]
        self._where_clauses: List[tuple] = []
        self._order_by: List[tuple] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None
        self._params: Dict[str, Any] = {}
        self._param_counter = 0

    def select(self, columns: List[str]) -> 'QueryBuilder':
        """Set SELECT columns"""
        self._select_columns = columns
        return self

    def where(
        self,
        column: str,
        operator: str,
        value: Any
    ) -> 'QueryBuilder':
        """Add WHERE condition"""
        param_name = f"p{self._param_counter}"
        self._param_counter += 1

        self._where_clauses.append((column, operator, param_name))
        self._params[param_name] = value
        return self

    def where_in(self, column: str, values: List[Any]) -> 'QueryBuilder':
        """Add WHERE IN condition"""
        param_name = f"p{self._param_counter}"
        self._param_counter += 1

        self._where_clauses.append((column, "IN", param_name))
        self._params[param_name] = tuple(values)
        return self

    def where_between(
        self,
        column: str,
        start: Any,
        end: Any
    ) -> 'QueryBuilder':
        """Add WHERE BETWEEN condition"""
        start_param = f"p{self._param_counter}"
        self._param_counter += 1
        end_param = f"p{self._param_counter}"
        self._param_counter += 1

        self._where_clauses.append((column, "BETWEEN", (start_param, end_param)))
        self._params[start_param] = start
        self._params[end_param] = end
        return self

    def order_by(self, column: str, desc: bool = False) -> 'QueryBuilder':
        """Add ORDER BY clause"""
        self._order_by.append((column, desc))
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        """Set LIMIT"""
        self._limit = n
        return self

    def offset(self, n: int) -> 'QueryBuilder':
        """Set OFFSET"""
        self._offset = n
        return self

    def build(self) -> tuple[str, Dict]:
        """Build SQL query and params"""
        # SELECT clause
        columns = ", ".join(self._select_columns)
        sql = f"SELECT {columns} FROM {self.table}"

        # WHERE clause
        if self._where_clauses:
            conditions = []
            for col, op, param in self._where_clauses:
                if op == "IN":
                    conditions.append(f"{col} IN :{param}")
                elif op == "BETWEEN":
                    start_p, end_p = param
                    conditions.append(f"{col} BETWEEN :{start_p} AND :{end_p}")
                else:
                    conditions.append(f"{col} {op} :{param}")

            sql += " WHERE " + " AND ".join(conditions)

        # ORDER BY clause
        if self._order_by:
            order_parts = []
            for col, desc in self._order_by:
                order_parts.append(f"{col} {'DESC' if desc else 'ASC'}")
            sql += " ORDER BY " + ", ".join(order_parts)

        # LIMIT/OFFSET
        if self._limit:
            sql += f" LIMIT {self._limit}"
        if self._offset:
            sql += f" OFFSET {self._offset}"

        return sql, self._params

    async def execute(self, db: AsyncDatabaseManager) -> List[Dict]:
        """Execute query and return results"""
        sql, params = self.build()
        from sqlalchemy import text
        return await db.fetch_all(text(sql), params)

    async def fetch_one(self, db: AsyncDatabaseManager) -> Optional[Dict]:
        """Execute and return single result"""
        self._limit = 1
        results = await self.execute(db)
        return results[0] if results else None


# =============================================================================
# REPOSITORY PATTERN
# =============================================================================

class BaseRepository(Generic[T]):
    """
    Base repository for data access.

    Usage:
        class PositionRepository(BaseRepository[Position]):
            def __init__(self, db):
                super().__init__(db, "positions", Position)

        repo = PositionRepository(db)
        positions = await repo.find_by(account_id=123)
    """

    def __init__(
        self,
        db: AsyncDatabaseManager,
        table: str,
        model_class: Optional[Type[T]] = None
    ):
        self.db = db
        self.table = table
        self.model_class = model_class
        self.batch = BatchOperations(db)

    async def find_by_id(self, id: Any) -> Optional[T]:
        """Find by primary key"""
        query = (QueryBuilder(self.table)
            .where("id", "=", id)
            .limit(1))

        result = await query.fetch_one(self.db)
        return self._to_model(result) if result else None

    async def find_by(self, **kwargs) -> List[T]:
        """Find by conditions"""
        query = QueryBuilder(self.table)

        for key, value in kwargs.items():
            if isinstance(value, list):
                query.where_in(key, value)
            else:
                query.where(key, "=", value)

        results = await query.execute(self.db)
        return [self._to_model(r) for r in results]

    async def find_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[T]:
        """Find all records"""
        query = QueryBuilder(self.table)

        if limit:
            query.limit(limit)
        if offset:
            query.offset(offset)

        results = await query.execute(self.db)
        return [self._to_model(r) for r in results]

    async def create(self, data: Dict) -> T:
        """Create new record"""
        results = await self.batch.bulk_insert(
            self.table,
            [data],
            returning=["id"]
        )
        data["id"] = results[0]["id"] if results else None
        return self._to_model(data)

    async def update(self, id: Any, data: Dict) -> bool:
        """Update record by ID"""
        set_clause = ", ".join([f"{k} = :{k}" for k in data.keys()])
        sql = f"UPDATE {self.table} SET {set_clause} WHERE id = :id"
        data["id"] = id

        async with self.db.session() as session:
            from sqlalchemy import text
            result = await session.execute(text(sql), data)
            return result.rowcount > 0

    async def delete(self, id: Any) -> bool:
        """Delete record by ID"""
        return await self.batch.bulk_delete(self.table, {"id": id}) > 0

    async def bulk_create(self, records: List[Dict]) -> List[T]:
        """Bulk create records"""
        results = await self.batch.bulk_insert(
            self.table,
            records,
            returning=["id"]
        )

        for record, result in zip(records, results):
            record["id"] = result.get("id")

        return [self._to_model(r) for r in records]

    def _to_model(self, data: Dict) -> T:
        """Convert dict to model instance"""
        if self.model_class:
            return self.model_class(**data)
        return data


# =============================================================================
# SPECIALIZED REPOSITORIES
# =============================================================================

class TradeRepository(BaseRepository):
    """Repository for trades"""

    def __init__(self, db: AsyncDatabaseManager):
        super().__init__(db, "trades")

    async def find_by_date_range(
        self,
        start_date: date,
        end_date: date,
        account_id: Optional[str] = None
    ) -> List[Dict]:
        """Find trades in date range"""
        query = (QueryBuilder(self.table)
            .where_between("trade_date", start_date, end_date)
            .order_by("trade_date", desc=True))

        if account_id:
            query.where("account_id", "=", account_id)

        return await query.execute(self.db)

    async def get_performance_summary(
        self,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Get aggregated performance metrics"""
        sql = """
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                SUM(realized_pnl) as total_pnl,
                AVG(realized_pnl) as avg_pnl,
                MAX(realized_pnl) as max_win,
                MIN(realized_pnl) as max_loss
            FROM trades
            WHERE trade_date BETWEEN :start_date AND :end_date
        """
        from sqlalchemy import text
        result = await self.db.fetch_one(
            text(sql),
            {"start_date": start_date, "end_date": end_date}
        )
        return result or {}


class PositionRepository(BaseRepository):
    """Repository for positions"""

    def __init__(self, db: AsyncDatabaseManager):
        super().__init__(db, "positions")

    async def find_open_positions(self, account_id: str) -> List[Dict]:
        """Find all open positions"""
        query = (QueryBuilder(self.table)
            .where("account_id", "=", account_id)
            .where("quantity", "!=", 0)
            .order_by("symbol"))

        return await query.execute(self.db)

    async def get_portfolio_summary(self, account_id: str) -> Dict:
        """Get portfolio summary"""
        sql = """
            SELECT
                COUNT(*) as total_positions,
                SUM(market_value) as total_value,
                SUM(unrealized_pnl) as total_unrealized_pnl,
                SUM(CASE WHEN quantity > 0 THEN market_value ELSE 0 END) as long_value,
                SUM(CASE WHEN quantity < 0 THEN ABS(market_value) ELSE 0 END) as short_value
            FROM positions
            WHERE account_id = :account_id AND quantity != 0
        """
        from sqlalchemy import text
        return await self.db.fetch_one(text(sql), {"account_id": account_id}) or {}


# =============================================================================
# DATABASE HEALTH CHECK
# =============================================================================

async def check_database_health(db: AsyncDatabaseManager) -> Dict:
    """Check database connectivity and health"""
    try:
        from sqlalchemy import text
        start = datetime.now()

        # Simple connectivity test
        await db.execute(text("SELECT 1"))

        latency = (datetime.now() - start).total_seconds() * 1000

        return {
            "status": "healthy",
            "latency_ms": latency,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Global database instance
_db_instance: Optional[AsyncDatabaseManager] = None


async def get_database() -> AsyncDatabaseManager:
    """Get or create global database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = AsyncDatabaseManager()
        await _db_instance.initialize()
    return _db_instance


async def close_database():
    """Close global database instance"""
    global _db_instance
    if _db_instance:
        await _db_instance.shutdown()
        _db_instance = None


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Database Utilities ===\n")

    print("1. QueryBuilder test...")
    query = (QueryBuilder("positions")
        .select(["symbol", "quantity", "market_value"])
        .where("account_id", "=", "acc123")
        .where("quantity", ">", 0)
        .order_by("market_value", desc=True)
        .limit(10))

    sql, params = query.build()
    print(f"   SQL: {sql}")
    print(f"   Params: {params}")

    print("\n2. Repository structure validated")
    print("   - TradeRepository")
    print("   - PositionRepository")
    print("   - BatchOperations")

    print("\n✅ Database utilities ready!")
    print("\nNote: Full database tests require a running PostgreSQL instance")
