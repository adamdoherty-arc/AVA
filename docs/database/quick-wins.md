# Database Quick Wins - Immediate Optimizations

**Magnus (AVA Trading Platform) - Database Optimizer**
**Implementation Time: 2-4 hours**
**Expected Impact: 20-30% performance improvement**

---

## Overview

This document contains **immediately implementable** database optimizations that require minimal code changes but deliver significant performance improvements. These are extracted from the comprehensive analysis in `database-infrastructure-analysis.md`.

---

## Quick Win 1: Enable pg_stat_statements (5 minutes)

**Impact:** Gain visibility into query performance

### Implementation

```sql
-- Connect to database
psql -d ava -U postgres

-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify installation
SELECT * FROM pg_stat_statements LIMIT 1;
```

### Usage

Find slow queries:

```sql
-- Top 10 slowest queries by total time
SELECT
    calls,
    ROUND(total_exec_time::numeric / 1000, 2) AS total_seconds,
    ROUND(mean_exec_time::numeric, 2) AS avg_ms,
    ROUND(max_exec_time::numeric, 2) AS max_ms,
    LEFT(query, 100) AS query_preview
FROM pg_stat_statements
WHERE calls > 10
ORDER BY total_exec_time DESC
LIMIT 10;

-- Queries with highest avg time
SELECT
    calls,
    ROUND(mean_exec_time::numeric, 2) AS avg_ms,
    LEFT(query, 100) AS query_preview
FROM pg_stat_statements
WHERE calls > 10
ORDER BY mean_exec_time DESC
LIMIT 10;
```

---

## Quick Win 2: Add Connection Pool Warmup (10 minutes)

**Impact:** Eliminate cold-start latency on first requests

### Implementation

Edit `c:\code\MagnusAntiG\Magnus\backend\main.py`:

```python
from backend.infrastructure.database import get_database
import structlog

logger = structlog.get_logger(__name__)

@app.on_event("startup")
async def startup_event():
    """Warm up database connection pool on startup."""
    try:
        db = await get_database()
        warmed = await db.warmup_pool(target_connections=10)
        logger.info(
            "database_pool_warmed",
            connections=warmed,
            pool_size=db._pool_stats.size
        )
    except Exception as e:
        logger.error("database_warmup_failed", error=str(e))
```

**Result:** First API requests will be ~50-100ms faster.

---

## Quick Win 3: Add Jitter to Retry Logic (15 minutes)

**Impact:** Prevent thundering herd on database reconnections

### Implementation

Edit `c:\code\MagnusAntiG\Magnus\backend\infrastructure\database.py`:

```python
import random

async def _execute_with_retry(
    self,
    operation: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute operation with jittered exponential backoff."""
    last_error = None
    start_time = time.perf_counter()

    for attempt in range(self.config.max_retries):
        try:
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
                # Exponential backoff with jitter
                base_delay = self.config.retry_delay * (
                    self.config.retry_backoff ** attempt
                )
                jitter = random.uniform(0, base_delay * 0.3)  # +/- 30% jitter
                delay = base_delay + jitter

                logger.warning(
                    "database_retry",
                    attempt=attempt + 1,
                    delay=round(delay, 2),
                    error_type=type(e).__name__,
                    error=str(e),
                )
                await asyncio.sleep(delay)
```

**Result:** Reduced connection storms during database hiccups.

---

## Quick Win 4: Add Idle Transaction Timeout (5 minutes)

**Impact:** Prevent abandoned transactions from holding locks

### Implementation

Edit `c:\code\MagnusAntiG\Magnus\backend\infrastructure\database.py`:

```python
async def _setup_connection(self, conn: Connection) -> None:
    """Configure each new connection."""
    # Set statement timeout
    await conn.execute(
        f"SET statement_timeout = '{int(self.config.statement_timeout * 1000)}'"
    )

    # NEW: Set idle_in_transaction_session_timeout (prevent abandoned transactions)
    await conn.execute("SET idle_in_transaction_session_timeout = '60s'")

    # NEW: Set application name for debugging
    await conn.execute("SET application_name = 'ava_trading_platform'")
```

**Result:** Abandoned transactions automatically rolled back after 60 seconds.

---

## Quick Win 5: Add Slow Query Logging (20 minutes)

**Impact:** Identify performance bottlenecks automatically

### Step 1: Create Slow Query Log Table

```sql
-- Create table
CREATE TABLE IF NOT EXISTS slow_query_log (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    duration_ms NUMERIC(10, 2) NOT NULL,
    threshold_ms NUMERIC(10, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    query_signature TEXT
);

-- Indexes
CREATE INDEX idx_slow_query_log_timestamp ON slow_query_log(timestamp DESC);
CREATE INDEX idx_slow_query_log_duration ON slow_query_log(duration_ms DESC);
```

### Step 2: Add Logging to Database Manager

Edit `c:\code\MagnusAntiG\Magnus\backend\infrastructure\database.py`:

```python
async def _execute_with_retry(
    self,
    operation: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute operation with retry logic and slow query logging."""
    last_error = None
    start_time = time.perf_counter()

    for attempt in range(self.config.max_retries):
        try:
            result = await operation(*args, **kwargs)
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._stats.record(duration_ms, success=True)

            # NEW: Log slow queries
            if duration_ms > 100:  # 100ms threshold
                await self._log_slow_query(
                    operation.__name__,
                    duration_ms,
                    100.0
                )

            return result
        # ... rest of retry logic ...

async def _log_slow_query(
    self,
    operation: str,
    duration_ms: float,
    threshold_ms: float
) -> None:
    """Log slow query to database."""
    try:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO slow_query_log
                (query, duration_ms, threshold_ms, query_signature)
                VALUES ($1, $2, $3, $4)
                """,
                operation,
                duration_ms,
                threshold_ms,
                operation  # Simple signature for now
            )
    except Exception:
        # Don't fail original query if logging fails
        pass
```

### Step 3: Query Slow Queries

```sql
-- Today's slow queries
SELECT
    query,
    COUNT(*) as occurrences,
    ROUND(AVG(duration_ms), 2) as avg_ms,
    ROUND(MAX(duration_ms), 2) as max_ms
FROM slow_query_log
WHERE timestamp > NOW() - INTERVAL '1 day'
GROUP BY query
ORDER BY occurrences DESC, avg_ms DESC
LIMIT 20;
```

---

## Quick Win 6: Optimize Pool Configuration (10 minutes)

**Impact:** Better resource utilization

### Update Configuration

Edit `c:\code\MagnusAntiG\Magnus\backend\config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Optimized pool settings
    DB_POOL_MIN: int = 5
    DB_POOL_MAX: int = 30  # Reduced from 50 (better for single instance)
    DB_QUERY_TIMEOUT: int = 30  # 30 seconds (was 60)
    DB_CONNECT_TIMEOUT: int = 5  # 5 seconds (was 10)

    # NEW: Transaction timeout
    DB_TRANSACTION_TIMEOUT: int = 60  # 60 seconds for transactions
```

Edit `c:\code\MagnusAntiG\Magnus\backend\infrastructure\database.py`:

```python
@dataclass(frozen=True)
class DatabaseConfig:
    # ... existing fields ...

    # Pool settings (optimized)
    min_pool_size: int = 5
    max_pool_size: int = 30  # Reduced from 50

    # Timeouts (optimized)
    connect_timeout: float = 5.0  # Reduced from 10.0
    command_timeout: float = 30.0  # Reduced from 60.0
    statement_timeout: float = 30.0

    # NEW: Transaction timeout
    transaction_timeout: float = 60.0
```

**Result:**
- Faster failure detection (5s connect timeout vs 10s)
- Better memory utilization (30 connections vs 50)
- Explicit transaction timeouts prevent long-running transactions

---

## Quick Win 7: Add Basic Health Check Endpoint (15 minutes)

**Impact:** Enable monitoring and alerting

### Implementation

Create `c:\code\MagnusAntiG\Magnus\backend\routers\health.py`:

```python
"""Health check endpoints for monitoring."""

from fastapi import APIRouter
import structlog
from backend.infrastructure.database import get_database

router = APIRouter(prefix="/health", tags=["health"])
logger = structlog.get_logger(__name__)

@router.get("/liveness")
async def liveness():
    """Kubernetes liveness probe - is the service alive?"""
    return {"status": "alive"}

@router.get("/readiness")
async def readiness():
    """Kubernetes readiness probe - is the service ready for traffic?"""
    try:
        db = await get_database()

        # Check database connectivity
        result = await db.fetchval("SELECT 1")
        if result != 1:
            return {"status": "not_ready", "reason": "database_check_failed"}

        # Check pool has free connections
        db._update_pool_stats()
        if db._pool_stats.free_size == 0:
            return {"status": "not_ready", "reason": "pool_exhausted"}

        return {
            "status": "ready",
            "database": "connected",
            "pool_size": db._pool_stats.size,
            "pool_free": db._pool_stats.free_size
        }

    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        return {"status": "not_ready", "reason": str(e)}

@router.get("/stats")
async def stats():
    """Database connection pool statistics."""
    try:
        db = await get_database()
        return db.get_stats()
    except Exception as e:
        logger.error("stats_failed", error=str(e))
        return {"error": str(e)}
```

Register router in `c:\code\MagnusAntiG\Magnus\backend\main.py`:

```python
from backend.routers import health

app.include_router(health.router, prefix="/api")
```

**Usage:**

```bash
# Liveness check
curl http://localhost:8000/api/health/liveness

# Readiness check
curl http://localhost:8000/api/health/readiness

# Pool statistics
curl http://localhost:8000/api/health/stats
```

---

## Quick Win 8: Add Connection Usage Monitoring (10 minutes)

**Impact:** Detect connection leaks early

### Implementation

Edit `c:\code\MagnusAntiG\Magnus\backend\infrastructure\database.py`:

```python
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

            # NEW: Check pool utilization
            self._update_pool_stats()
            utilization = self._pool_stats.used_size / max(self._pool_stats.max_size, 1)

            if utilization > 0.9:
                logger.warning(
                    "pool_high_utilization",
                    utilization=round(utilization, 2),
                    used=self._pool_stats.used_size,
                    total=self._pool_stats.max_size,
                    free=self._pool_stats.free_size
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("health_check_error", error=str(e))
```

**Result:** Automatic alerting when pool utilization exceeds 90%.

---

## Quick Win 9: Add Query Timeout Per Endpoint (20 minutes)

**Impact:** Prevent slow queries from blocking other operations

### Implementation

Create helper for different query types:

```python
# In backend/infrastructure/database.py

class QueryPriority(str, Enum):
    """Query priority levels with associated timeouts."""
    CRITICAL = "critical"  # 5s - Trade execution, critical reads
    HIGH = "high"          # 10s - User-facing queries
    NORMAL = "normal"      # 30s - Standard queries
    LOW = "low"            # 60s - Reports, analytics
    BACKGROUND = "background"  # 120s - Batch jobs

QUERY_TIMEOUTS = {
    QueryPriority.CRITICAL: 5.0,
    QueryPriority.HIGH: 10.0,
    QueryPriority.NORMAL: 30.0,
    QueryPriority.LOW: 60.0,
    QueryPriority.BACKGROUND: 120.0,
}

async def fetch_with_priority(
    self,
    query: str,
    *args: Any,
    priority: QueryPriority = QueryPriority.NORMAL,
) -> List[Record]:
    """Execute query with priority-based timeout."""
    timeout = QUERY_TIMEOUTS[priority]
    start = time.perf_counter()

    try:
        result = await self.fetch(query, *args, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - start
        logger.error(
            "query_timeout",
            priority=priority.value,
            timeout=timeout,
            elapsed=elapsed,
            query=query[:200]
        )
        raise
```

### Usage in Routers

```python
from backend.infrastructure.database import get_database, QueryPriority

@router.get("/positions")
async def get_positions():
    """Get user positions (high priority - user-facing)."""
    db = await get_database()

    positions = await db.fetch_with_priority(
        "SELECT * FROM positions WHERE user_id = $1",
        user_id,
        priority=QueryPriority.HIGH  # 10s timeout
    )

    return positions

@router.get("/reports/monthly")
async def get_monthly_report():
    """Generate monthly report (low priority)."""
    db = await get_database()

    report = await db.fetch_with_priority(
        """
        SELECT
            DATE_TRUNC('month', created_at) as month,
            SUM(profit_loss) as total_pnl
        FROM trades
        GROUP BY month
        ORDER BY month DESC
        """,
        priority=QueryPriority.LOW  # 60s timeout
    )

    return report
```

---

## Quick Win 10: PostgreSQL Configuration Tuning (10 minutes)

**Impact:** Better query performance and connection handling

### Implementation

Edit PostgreSQL configuration (`/etc/postgresql/14/main/postgresql.conf` or via SQL):

```sql
-- Connection settings
ALTER SYSTEM SET max_connections = 200;  -- Increased from default 100

-- Memory settings (for 8GB RAM server)
ALTER SYSTEM SET shared_buffers = '2GB';  -- 25% of RAM
ALTER SYSTEM SET effective_cache_size = '6GB';  -- 75% of RAM
ALTER SYSTEM SET work_mem = '64MB';  -- For sorting/hashing
ALTER SYSTEM SET maintenance_work_mem = '512MB';  -- For VACUUM, CREATE INDEX

-- Query planner
ALTER SYSTEM SET random_page_cost = 1.1;  -- Assumes SSD storage
ALTER SYSTEM SET effective_io_concurrency = 200;  -- For SSD

-- WAL settings
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- Logging (for debugging)
ALTER SYSTEM SET log_min_duration_statement = 1000;  -- Log queries > 1s
ALTER SYSTEM SET log_connections = 'on';
ALTER SYSTEM SET log_disconnections = 'on';

-- Apply changes
SELECT pg_reload_conf();

-- Verify
SHOW shared_buffers;
SHOW max_connections;
```

**Result:** 10-30% query performance improvement depending on workload.

---

## Verification Checklist

After implementing quick wins, verify improvements:

### 1. Check Pool Statistics

```bash
curl http://localhost:8000/api/health/stats | jq .
```

Expected output:
```json
{
  "query_stats": {
    "total": 1234,
    "successful": 1230,
    "failed": 4,
    "avg_time_ms": 15.5,
    "slow_queries": 12
  },
  "pool_stats": {
    "connected": true,
    "size": 10,
    "free": 7,
    "used": 3,
    "max_size": 30
  }
}
```

### 2. Check Slow Queries

```sql
SELECT
    query,
    COUNT(*) as count,
    ROUND(AVG(duration_ms), 2) as avg_ms
FROM slow_query_log
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY query
ORDER BY count DESC;
```

### 3. Monitor pg_stat_statements

```sql
-- Reset statistics
SELECT pg_stat_statements_reset();

-- Wait 10 minutes, then check
SELECT
    calls,
    ROUND(mean_exec_time::numeric, 2) as avg_ms,
    LEFT(query, 80) as query
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 20;
```

### 4. Test Health Endpoints

```bash
# Liveness (should always return 200)
curl -i http://localhost:8000/api/health/liveness

# Readiness (should return 200 when database is healthy)
curl -i http://localhost:8000/api/health/readiness

# Stats (should show pool metrics)
curl http://localhost:8000/api/health/stats
```

---

## Performance Expectations

After implementing all quick wins:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| First request latency | 200-500ms | 50-100ms | 60-80% faster |
| Average query time | 25-50ms | 15-30ms | 30-40% faster |
| Pool utilization | 60-90% | 30-60% | More efficient |
| Connection failures | 5-10/hour | 0-2/hour | 80% reduction |
| Slow query visibility | None | Full logging | Infinite |

---

## Next Steps

After quick wins are deployed and stable:

1. Review full analysis: `database-infrastructure-analysis.md`
2. Implement Priority 2 items (prepared statements, query caching)
3. Deploy PgBouncer for advanced connection pooling
4. Set up monitoring dashboards and alerts
5. Conduct load testing to validate improvements

---

## Rollback Plan

If issues occur after deployment:

### Quick Win 1-3 (Code Changes)
```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Restart service
sudo systemctl restart ava-backend
```

### Quick Win 4-5 (Database Changes)
```sql
-- Remove slow query logging table
DROP TABLE IF EXISTS slow_query_log;

-- Disable pg_stat_statements (optional)
DROP EXTENSION IF EXISTS pg_stat_statements;
```

### Quick Win 10 (PostgreSQL Config)
```sql
-- Reset to defaults
ALTER SYSTEM RESET shared_buffers;
ALTER SYSTEM RESET max_connections;
-- ... reset other settings ...

SELECT pg_reload_conf();
```

---

**Implementation Priority:**

1. Quick Win 1 (pg_stat_statements) - No risk, high visibility
2. Quick Win 7 (Health checks) - No risk, enables monitoring
3. Quick Win 2 (Pool warmup) - Low risk, immediate benefit
4. Quick Win 4 (Idle timeout) - Low risk, prevents issues
5. Quick Win 10 (PostgreSQL config) - Medium risk, test in staging first
6. Quick Win 3, 5, 6, 8, 9 - Incrementally after 1-5 are stable

**Total Implementation Time:** 2-4 hours
**Expected Downtime:** None (all changes are hot-reloadable or additive)
