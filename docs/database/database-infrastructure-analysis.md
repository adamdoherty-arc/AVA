# Database Infrastructure Analysis & Optimization Recommendations

**Magnus (AVA Trading Platform) - Database Optimizer Review**
**Date:** 2025-12-04
**Reviewed Files:**
- `backend/infrastructure/database.py` (Primary async database)
- `backend/infrastructure/async_db.py` (Alternative async implementation)
- `backend/database/connection.py` (Legacy sync pool)
- `backend/config.py` (Configuration)
- `backend/infrastructure/observability.py` (Monitoring)

---

## Executive Summary

The Magnus platform uses a **dual database infrastructure** with a modern async pool (`database.py`) as the primary access pattern and a legacy sync pool (`connection.py`) for backward compatibility. The infrastructure demonstrates solid fundamentals with connection pooling, retry logic, and basic health checks.

**Current State: 7/10 (Good)**

**Key Strengths:**
- Modern async/await with asyncpg
- Connection pooling (5-50 connections)
- Retry logic with exponential backoff
- Query statistics tracking
- OpenTelemetry integration hooks
- Transaction context managers

**Critical Gaps:**
- No prepared statement management for frequently-used queries
- Missing connection leak detection/prevention
- No query timeout granularity (per-query timeouts)
- Health checks don't test connection pool exhaustion
- No automatic slow query logging to database
- Missing connection pool metrics export
- No circuit breaker for database failures
- Limited query result caching

---

## 1. Connection Pool Configuration Analysis

### Current Configuration

#### Primary Pool (`backend/infrastructure/database.py`)
```python
@dataclass(frozen=True)
class DatabaseConfig:
    # Pool settings
    min_pool_size: int = 5
    max_pool_size: int = 50

    # Timeouts (seconds)
    connect_timeout: float = 10.0
    command_timeout: float = 60.0
    statement_timeout: float = 30.0
```

#### Legacy Pool (`backend/database/connection.py`)
```python
cls._pool = pool.ThreadedConnectionPool(
    minconn=5,
    maxconn=50,
    connect_timeout=10,
    options='-c statement_timeout=30000'  # 30 seconds
)
```

### Assessment: **Good Configuration**

**Strengths:**
- **Appropriate pool size** (5-50) for a medium-scale trading application
- **Separate timeouts** for connection vs. query execution
- **Pool warmup capability** (`warmup_pool()`) to prevent cold-start latency
- **Per-connection statement timeout** set via `_setup_connection()`

**Recommendations:**

#### 1.1 Add Dynamic Pool Sizing

**Problem:** Fixed pool sizes don't adapt to load variations (market hours vs. off-hours).

**Solution:** Implement adaptive pool sizing based on connection utilization.

```python
@dataclass(frozen=True)
class DatabaseConfig:
    # ... existing fields ...

    # Adaptive pool sizing
    enable_adaptive_pool: bool = True
    target_utilization: float = 0.7  # Scale up at 70% utilization
    scale_up_threshold: int = 45     # Scale when used > 45 of 50
    scale_down_threshold: int = 10    # Scale down when used < 10
    scale_interval: float = 60.0      # Check every 60 seconds

    # Pool limits
    absolute_max_pool_size: int = 100  # Hard ceiling
```

**Implementation:**
```python
async def _adaptive_pool_manager(self) -> None:
    """Background task to adjust pool size based on utilization."""
    while self._connected:
        await asyncio.sleep(self.config.scale_interval)

        if not self.config.enable_adaptive_pool:
            continue

        self._update_pool_stats()
        utilization = self._pool_stats.used_size / self._pool_stats.max_size

        if utilization > self.config.target_utilization:
            # Scale up
            new_max = min(
                self._pool_stats.max_size + 10,
                self.config.absolute_max_pool_size
            )
            logger.info(
                "pool_scaling_up",
                current_max=self._pool_stats.max_size,
                new_max=new_max,
                utilization=utilization
            )
            # asyncpg doesn't support dynamic resizing
            # Recommendation: Use pgbouncer for true dynamic pooling

        elif utilization < 0.3 and self._pool_stats.max_size > self.config.min_pool_size:
            # Scale down
            logger.info("pool_scale_down_suggested", utilization=utilization)
```

**Better Approach:** Use **PgBouncer** for dynamic connection pooling at the infrastructure level.

#### 1.2 Add Per-Priority Connection Pools

**Problem:** AI predictions and live trading share the same connection pool. A slow AI query can block a critical trade execution.

**Solution:** Segment the pool by priority.

```python
class PriorityDatabaseManager:
    """Multi-pool manager with priority tiers."""

    def __init__(self, config: DatabaseConfig):
        self.critical_pool = AsyncDatabaseManager(
            config.with_overrides(min_pool_size=3, max_pool_size=10)
        )
        self.high_pool = AsyncDatabaseManager(
            config.with_overrides(min_pool_size=5, max_pool_size=20)
        )
        self.normal_pool = AsyncDatabaseManager(
            config.with_overrides(min_pool_size=5, max_pool_size=30)
        )
        self.background_pool = AsyncDatabaseManager(
            config.with_overrides(min_pool_size=2, max_pool_size=10)
        )

    def get_pool(self, priority: str = "normal") -> AsyncDatabaseManager:
        pools = {
            "critical": self.critical_pool,  # Trade execution
            "high": self.high_pool,          # Real-time quotes
            "normal": self.normal_pool,      # User queries
            "background": self.background_pool  # Analytics, reports
        }
        return pools.get(priority, self.normal_pool)
```

#### 1.3 Add Connection Lifetime Management

**Problem:** Long-lived connections can accumulate cruft and degrade performance.

**Solution:** Set `max_connection_lifetime` and `max_idle_time`.

```python
@dataclass(frozen=True)
class DatabaseConfig:
    # ... existing fields ...

    # Connection lifecycle
    max_connection_age: float = 3600.0      # 1 hour max lifetime
    max_idle_time: float = 300.0            # 5 minutes max idle
    connection_recycling_enabled: bool = True
```

**Note:** `asyncpg` doesn't natively support this. Implement via background task:

```python
async def _connection_lifecycle_manager(self) -> None:
    """Periodically recycle old connections."""
    while self._connected:
        await asyncio.sleep(60)  # Check every minute

        # Force pool to create new connections
        # (asyncpg limitation: can't inspect connection age)
        logger.info("connection_pool_recycle_check")
```

**Better Approach:** Use **PgBouncer** with `server_lifetime` setting.

---

## 2. Health Check Mechanisms

### Current Implementation

#### Primary Health Check (`database.py`)
```python
async def health_check(self) -> bool:
    """Perform database health check."""
    try:
        result = await self.fetchval("SELECT 1")
        self._pool_stats.last_health_check = datetime.now()
        self._update_pool_stats()
        return result == 1
    except Exception:
        return False
```

#### Background Health Check Loop
```python
async def _health_check_loop(self) -> None:
    """Background task for periodic health checks."""
    while self._connected:
        try:
            await asyncio.sleep(self.config.health_check_interval)  # 30s
            healthy = await self.health_check()
            if not healthy:
                self._pool_stats.health_check_failures += 1
```

### Assessment: **Basic but Functional**

**Strengths:**
- Continuous background health checks (every 30 seconds)
- Failure counting
- Integration with observability (`HealthChecker.check_database()`)

**Weaknesses:**
- Only tests basic connectivity, not pool exhaustion
- No circuit breaker pattern
- No alerting on repeated failures
- Doesn't validate transaction capabilities

**Recommendations:**

#### 2.1 Comprehensive Health Check

Replace the basic `SELECT 1` with a multi-faceted health check:

```python
@dataclass
class DatabaseHealthReport:
    """Detailed health check result."""
    is_healthy: bool
    latency_ms: float
    pool_utilization: float
    slow_queries_count: int
    failed_queries_count: int
    connection_errors: int
    transaction_test_passed: bool
    details: Dict[str, Any]

async def comprehensive_health_check(self) -> DatabaseHealthReport:
    """Multi-faceted database health check."""
    start = time.perf_counter()
    checks = []

    # 1. Connectivity test
    try:
        result = await self.fetchval("SELECT 1")
        checks.append(("connectivity", result == 1))
    except Exception as e:
        logger.error("health_check_connectivity_failed", error=str(e))
        checks.append(("connectivity", False))

    # 2. Pool utilization check
    self._update_pool_stats()
    utilization = self._pool_stats.used_size / self._pool_stats.max_size
    checks.append(("pool_utilization", utilization < 0.9))

    # 3. Transaction test
    try:
        async with self.transaction() as conn:
            await conn.execute("SELECT 1")
        checks.append(("transaction", True))
    except Exception as e:
        logger.error("health_check_transaction_failed", error=str(e))
        checks.append(("transaction", False))

    # 4. Query statistics check
    checks.append(("query_failure_rate", self._stats.failed_queries / max(self._stats.total_queries, 1) < 0.05))
    checks.append(("slow_queries", self._stats.slow_queries < 100))

    # 5. Table access test (critical tables)
    try:
        await self.fetchval("SELECT COUNT(*) FROM positions LIMIT 1")
        checks.append(("critical_tables", True))
    except Exception as e:
        logger.error("health_check_tables_failed", error=str(e))
        checks.append(("critical_tables", False))

    latency = (time.perf_counter() - start) * 1000
    is_healthy = all(passed for _, passed in checks)

    return DatabaseHealthReport(
        is_healthy=is_healthy,
        latency_ms=round(latency, 2),
        pool_utilization=round(utilization, 2),
        slow_queries_count=self._stats.slow_queries,
        failed_queries_count=self._stats.failed_queries,
        connection_errors=self._pool_stats.health_check_failures,
        transaction_test_passed=dict(checks).get("transaction", False),
        details={name: passed for name, passed in checks}
    )
```

#### 2.2 Health Check Circuit Breaker

Implement a circuit breaker to stop health checks when the database is clearly down:

```python
class HealthCheckCircuitBreaker:
    """Circuit breaker for database health checks."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_attempts: int = 3
    ):
        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: Optional[float] = None
        self.half_open_attempts = 0
        self.half_open_max_attempts = half_open_max_attempts

    async def call(self, health_check_fn: Callable) -> bool:
        """Execute health check with circuit breaker logic."""

        # Open state: Skip health checks during cooldown
        if self.state == "open":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half_open"
                self.half_open_attempts = 0
                logger.info("circuit_breaker_half_open")
            else:
                return False

        # Half-open state: Test recovery
        if self.state == "half_open":
            if self.half_open_attempts >= self.half_open_max_attempts:
                self.state = "closed"
                self.failure_count = 0
                logger.info("circuit_breaker_closed")
            self.half_open_attempts += 1

        # Execute health check
        try:
            healthy = await health_check_fn()
            if healthy:
                self.failure_count = 0
                if self.state == "half_open":
                    self.state = "closed"
                return True
            else:
                self._record_failure()
                return False
        except Exception as e:
            logger.error("health_check_error", error=str(e))
            self._record_failure()
            return False

    def _record_failure(self) -> None:
        """Record a health check failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                recovery_timeout=self.recovery_timeout
            )
```

#### 2.3 Readiness vs. Liveness Checks

Distinguish between "is the service alive?" and "is the service ready to handle traffic?".

```python
async def liveness(self) -> bool:
    """Kubernetes liveness probe: Is the process alive?"""
    return self._connected and self._pool is not None

async def readiness(self) -> bool:
    """Kubernetes readiness probe: Is the service ready for traffic?"""
    if not self.liveness():
        return False

    # Check pool has available connections
    self._update_pool_stats()
    if self._pool_stats.free_size == 0:
        logger.warning("readiness_failed_no_free_connections")
        return False

    # Check recent query success rate
    if self._stats.total_queries > 100:
        success_rate = self._stats.successful_queries / self._stats.total_queries
        if success_rate < 0.95:
            logger.warning("readiness_failed_low_success_rate", rate=success_rate)
            return False

    return True
```

---

## 3. Auto-Reconnection Logic

### Current Implementation

#### Retry Logic with Backoff
```python
async def _execute_with_retry(
    self,
    operation: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute operation with retry logic and instrumentation."""
    last_error = None
    start_time = time.perf_counter()

    for attempt in range(self.config.max_retries):  # max_retries = 3
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
                delay = self.config.retry_delay * (
                    self.config.retry_backoff ** attempt
                )
                # delay = 0.5 * (2 ** attempt) = 0.5s, 1s, 2s
                logger.warning(
                    "database_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)
```

### Assessment: **Good but Limited**

**Strengths:**
- Exponential backoff prevents thundering herd
- Only retries connection-related errors (not logic errors)
- Logs retry attempts

**Weaknesses:**
- Only 3 retries (4 total attempts) - may be insufficient for brief network hiccups
- No jitter in backoff (can cause synchronized retries across multiple clients)
- Doesn't distinguish between transient vs. permanent failures
- No connection pool recreation on catastrophic failure

**Recommendations:**

#### 3.1 Add Jittered Exponential Backoff

Prevent synchronized retries across multiple API workers:

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
            asyncpg.TooManyConnectionsError,
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
            else:
                logger.error(
                    "database_retry_exhausted",
                    attempts=self.config.max_retries,
                    error=str(e)
                )
```

#### 3.2 Implement Connection Pool Resurrection

Detect and recover from catastrophic pool failures:

```python
async def _ensure_connected(self) -> None:
    """Ensure we have a valid connection pool."""
    if not self._connected or not self._pool:
        logger.warning("pool_not_connected_attempting_reconnect")
        await self.connect()
        return

    # Check if pool is still healthy
    try:
        # Test pool by checking size
        size = self._pool.get_size()
        if size == 0:
            logger.error("pool_empty_forcing_reconnect")
            await self._force_reconnect()
    except Exception as e:
        logger.error("pool_unhealthy_forcing_reconnect", error=str(e))
        await self._force_reconnect()

async def _force_reconnect(self) -> None:
    """Force pool reconnection after catastrophic failure."""
    try:
        # Close existing pool
        if self._pool:
            await self._pool.close()

        self._pool = None
        self._connected = False

        # Wait before reconnecting
        await asyncio.sleep(5)

        # Reconnect
        await self.connect()
        logger.info("pool_reconnect_successful")
    except Exception as e:
        logger.error("pool_reconnect_failed", error=str(e))
        raise
```

#### 3.3 Add Connection Validation

Validate connections before use (especially after long idle periods):

```python
async def _setup_connection(self, conn: Connection) -> None:
    """Configure each new connection."""
    # Set statement timeout
    await conn.execute(
        f"SET statement_timeout = '{int(self.config.statement_timeout * 1000)}'"
    )

    # Set idle_in_transaction_session_timeout (prevent abandoned transactions)
    await conn.execute("SET idle_in_transaction_session_timeout = '60s'")

    # Set application name for debugging
    await conn.execute("SET application_name = 'ava_trading_platform'")

    # Register a connection init callback
    await conn.add_termination_listener(self._on_connection_termination)

def _on_connection_termination(self, conn: Connection) -> None:
    """Called when a connection is terminated."""
    logger.warning("connection_terminated", conn_id=id(conn))
```

---

## 4. Query Timeout Handling

### Current Implementation

#### Global Statement Timeout
```python
async def _setup_connection(self, conn: Connection) -> None:
    """Configure each new connection."""
    await conn.execute(
        f"SET statement_timeout = '{int(self.config.statement_timeout * 1000)}'"
    )
    # Default: 30 seconds
```

#### Per-Query Timeout (Optional)
```python
async def fetch(
    self,
    query: str,
    *args: Any,
    timeout: Optional[float] = None,  # Can override global timeout
) -> List[Record]:
    await self._ensure_connected()

    async def _fetch() -> List[Record]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)

    return await self._execute_with_retry(_fetch)
```

### Assessment: **Good Baseline**

**Strengths:**
- Statement timeout prevents runaway queries
- Per-query timeout override available
- Connection-level timeout prevents hung connections

**Weaknesses:**
- No query classification (fast vs. slow expected queries)
- No timeout monitoring/alerting
- No adaptive timeout based on query history

**Recommendations:**

#### 4.1 Query Classification System

Categorize queries by expected execution time:

```python
class QueryClass(Enum):
    """Query classification by expected execution time."""
    INSTANT = "instant"          # < 100ms (single-row lookups)
    FAST = "fast"                # < 500ms (indexed queries)
    NORMAL = "normal"            # < 2s (most queries)
    SLOW = "slow"                # < 10s (reports, aggregations)
    BATCH = "batch"              # < 60s (bulk operations)

@dataclass
class QueryConfig:
    """Per-class query configuration."""
    timeout: float
    retry_enabled: bool
    cache_ttl: Optional[int]
    alert_threshold: float

QUERY_CLASS_CONFIG = {
    QueryClass.INSTANT: QueryConfig(
        timeout=0.2,
        retry_enabled=True,
        cache_ttl=60,
        alert_threshold=0.1
    ),
    QueryClass.FAST: QueryConfig(
        timeout=1.0,
        retry_enabled=True,
        cache_ttl=30,
        alert_threshold=0.5
    ),
    QueryClass.NORMAL: QueryConfig(
        timeout=5.0,
        retry_enabled=True,
        cache_ttl=None,
        alert_threshold=2.0
    ),
    QueryClass.SLOW: QueryConfig(
        timeout=15.0,
        retry_enabled=False,
        cache_ttl=300,
        alert_threshold=10.0
    ),
    QueryClass.BATCH: QueryConfig(
        timeout=120.0,
        retry_enabled=False,
        cache_ttl=None,
        alert_threshold=60.0
    ),
}

async def fetch_classified(
    self,
    query: str,
    *args: Any,
    query_class: QueryClass = QueryClass.NORMAL,
) -> List[Record]:
    """Execute query with class-specific configuration."""
    config = QUERY_CLASS_CONFIG[query_class]

    start = time.perf_counter()
    try:
        result = await self.fetch(query, *args, timeout=config.timeout)
        duration = time.perf_counter() - start

        # Alert on slow queries
        if duration > config.alert_threshold:
            logger.warning(
                "query_slow",
                query_class=query_class.value,
                duration_ms=duration * 1000,
                threshold_ms=config.alert_threshold * 1000,
                query=query[:200]  # Log first 200 chars
            )

        return result
    except asyncio.TimeoutError:
        logger.error(
            "query_timeout",
            query_class=query_class.value,
            timeout=config.timeout,
            query=query[:200]
        )
        raise
```

#### 4.2 Adaptive Timeout System

Learn query execution times and adjust timeouts dynamically:

```python
class AdaptiveTimeoutManager:
    """Manages per-query adaptive timeouts based on historical performance."""

    def __init__(self):
        self.query_history: Dict[str, List[float]] = {}
        self.max_history = 100

    def _get_query_signature(self, query: str) -> str:
        """Get a normalized signature for the query."""
        # Normalize query by removing literals
        import re
        normalized = re.sub(r'\$\d+', '$N', query)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized[:200]

    def record_execution(self, query: str, duration: float) -> None:
        """Record query execution time."""
        sig = self._get_query_signature(query)
        if sig not in self.query_history:
            self.query_history[sig] = []

        self.query_history[sig].append(duration)
        if len(self.query_history[sig]) > self.max_history:
            self.query_history[sig] = self.query_history[sig][-self.max_history:]

    def get_recommended_timeout(
        self,
        query: str,
        default_timeout: float = 30.0,
        percentile: float = 0.95
    ) -> float:
        """Get recommended timeout based on historical performance."""
        sig = self._get_query_signature(query)
        history = self.query_history.get(sig, [])

        if not history:
            return default_timeout

        # Use 95th percentile + 50% buffer
        import statistics
        p95 = statistics.quantiles(history, n=20)[18]  # 95th percentile
        recommended = p95 * 1.5

        # Clamp to reasonable bounds
        return max(1.0, min(recommended, 120.0))
```

#### 4.3 Query Timeout Monitoring Dashboard

Expose timeout metrics for monitoring:

```python
def get_timeout_stats(self) -> Dict[str, Any]:
    """Get query timeout statistics."""
    return {
        "total_queries": self._stats.total_queries,
        "timeout_count": self._timeout_count,
        "timeout_rate": self._timeout_count / max(self._stats.total_queries, 1),
        "avg_query_time_ms": self._stats.avg_time_ms,
        "p95_query_time_ms": self._get_p95_time(),
        "p99_query_time_ms": self._get_p99_time(),
        "slow_queries_by_class": self._get_slow_queries_by_class(),
    }
```

---

## 5. Transaction Support

### Current Implementation

```python
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
```

### Assessment: **Solid Foundation**

**Strengths:**
- Context manager with automatic rollback on exception
- Configurable isolation levels
- Async-native

**Weaknesses:**
- No transaction timeout
- No nested transaction support (savepoints)
- No transaction statistics/monitoring
- No deadlock detection/retry

**Recommendations:**

#### 5.1 Add Transaction Timeout

Prevent long-running transactions from blocking other queries:

```python
@asynccontextmanager
async def transaction(
    self,
    isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    timeout: Optional[float] = None,
) -> AsyncGenerator[Connection, None]:
    """
    Context manager for database transactions with timeout.

    Args:
        isolation: Transaction isolation level
        timeout: Max transaction duration (seconds)
    """
    await self._ensure_connected()

    timeout = timeout or self.config.transaction_timeout  # Default: 30s
    start_time = time.perf_counter()

    async with self.pool.acquire() as conn:
        try:
            async with asyncio.timeout(timeout):
                async with conn.transaction(isolation=isolation.value):
                    yield conn

            duration = (time.perf_counter() - start_time) * 1000
            logger.info("transaction_completed", duration_ms=round(duration, 2))

        except asyncio.TimeoutError:
            duration = (time.perf_counter() - start_time) * 1000
            logger.error(
                "transaction_timeout",
                duration_ms=round(duration, 2),
                timeout_s=timeout
            )
            raise
```

#### 5.2 Add Savepoint Support for Nested Transactions

Enable partial rollbacks within a transaction:

```python
@asynccontextmanager
async def savepoint(
    self,
    conn: Connection,
    name: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Create a savepoint within a transaction.

    Usage:
        async with db.transaction() as conn:
            await conn.execute("INSERT INTO orders ...")

            async with db.savepoint(conn, "before_risky_op") as sp:
                try:
                    await conn.execute("UPDATE inventory ...")
                except Exception:
                    # Rolls back to savepoint only
                    raise

            await conn.execute("INSERT INTO audit_log ...")
    """
    savepoint_name = name or f"sp_{uuid.uuid4().hex[:8]}"

    await conn.execute(f"SAVEPOINT {savepoint_name}")
    try:
        yield savepoint_name
    except Exception:
        await conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
        raise
    else:
        await conn.execute(f"RELEASE SAVEPOINT {savepoint_name}")
```

#### 5.3 Add Deadlock Detection & Retry

Automatically retry transactions that fail due to deadlocks:

```python
async def transaction_with_retry(
    self,
    operation: Callable[[Connection], Awaitable[T]],
    max_attempts: int = 3,
    isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
) -> T:
    """
    Execute a transaction with automatic deadlock retry.

    Usage:
        async def update_inventory(conn: Connection):
            await conn.execute("UPDATE inventory SET qty = qty - 1 WHERE id = $1", item_id)
            await conn.execute("INSERT INTO orders ...")

        result = await db.transaction_with_retry(update_inventory)
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            async with self.transaction(isolation=isolation) as conn:
                result = await operation(conn)
                return result

        except asyncpg.DeadlockDetectedError as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                logger.warning(
                    "transaction_deadlock_retry",
                    attempt=attempt + 1,
                    delay=delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "transaction_deadlock_exhausted",
                    attempts=max_attempts
                )

        except asyncpg.SerializationError as e:
            # Serialization failures in SERIALIZABLE isolation
            last_error = e
            if attempt < max_attempts - 1:
                delay = 0.05 * (2 ** attempt)
                logger.warning(
                    "transaction_serialization_retry",
                    attempt=attempt + 1,
                    delay=delay
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "transaction_serialization_exhausted",
                    attempts=max_attempts
                )

    raise last_error or Exception("Transaction failed")
```

#### 5.4 Add Transaction Statistics

Monitor transaction performance:

```python
@dataclass
class TransactionStats:
    """Transaction statistics."""
    total_transactions: int = 0
    committed: int = 0
    rolled_back: int = 0
    deadlocks: int = 0
    timeouts: int = 0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0

def get_transaction_stats(self) -> Dict[str, Any]:
    """Get transaction performance statistics."""
    return {
        "total": self._tx_stats.total_transactions,
        "committed": self._tx_stats.committed,
        "rolled_back": self._tx_stats.rolled_back,
        "commit_rate": self._tx_stats.committed / max(self._tx_stats.total_transactions, 1),
        "deadlocks": self._tx_stats.deadlocks,
        "timeouts": self._tx_stats.timeouts,
        "avg_duration_ms": round(self._tx_stats.avg_duration_ms, 2),
        "max_duration_ms": round(self._tx_stats.max_duration_ms, 2),
    }
```

---

## 6. Prepared Statement Support

### Current State: **NOT IMPLEMENTED**

The current implementation has a placeholder for prepared statements:

```python
self._prepared_statements: Dict[str, str] = {}
```

But there's **no actual prepared statement caching logic**.

### Assessment: **Critical Missing Feature**

**Impact:**
- **Performance loss:** Repeatedly parsing the same queries wastes CPU cycles
- **Security:** Increases SQL injection risk if not using parameterized queries properly
- **Scalability:** PostgreSQL query planner overhead on every execution

**Recommendation: HIGH PRIORITY**

#### 6.1 Implement Prepared Statement Cache

```python
class PreparedStatementCache:
    """LRU cache for prepared statements."""

    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, str] = {}  # query -> statement_name
        self.access_count: Dict[str, int] = {}
        self.max_size = max_size

    def _get_statement_name(self, query: str) -> str:
        """Generate unique statement name."""
        import hashlib
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"stmt_{query_hash}"

    def get_or_create(self, query: str) -> str:
        """Get existing or generate new statement name."""
        if query in self.cache:
            self.access_count[query] += 1
            return self.cache[query]

        # Evict LRU if cache full
        if len(self.cache) >= self.max_size:
            lru_query = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_query]
            del self.access_count[lru_query]

        stmt_name = self._get_statement_name(query)
        self.cache[query] = stmt_name
        self.access_count[query] = 1
        return stmt_name

class AsyncDatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        # ... existing code ...
        self._prepared_cache = PreparedStatementCache(max_size=200)
        self._prepared_lock = asyncio.Lock()

    async def fetch_prepared(
        self,
        query: str,
        *args: Any,
        timeout: Optional[float] = None,
    ) -> List[Record]:
        """
        Execute query using prepared statement.

        Automatically prepares and caches statements for reuse.
        """
        await self._ensure_connected()

        stmt_name = self._prepared_cache.get_or_create(query)

        async def _fetch_prepared() -> List[Record]:
            async with self.pool.acquire() as conn:
                # Check if statement exists
                try:
                    return await conn.fetch(f"EXECUTE {stmt_name} ({', '.join('$' + str(i+1) for i in range(len(args)))})", *args, timeout=timeout)
                except asyncpg.InvalidSQLStatementNameError:
                    # Statement doesn't exist, prepare it
                    async with self._prepared_lock:
                        try:
                            await conn.execute(
                                f"PREPARE {stmt_name} AS {query}"
                            )
                        except asyncpg.DuplicatePreparedStatementError:
                            # Another coroutine prepared it
                            pass

                    # Execute prepared statement
                    return await conn.fetch(f"EXECUTE {stmt_name} ({', '.join('$' + str(i+1) for i in range(len(args)))})", *args, timeout=timeout)

        return await self._execute_with_retry(_fetch_prepared)
```

**Better Approach:** Use `asyncpg`'s built-in prepared statement support:

```python
async def fetch_with_prepare(
    self,
    query: str,
    *args: Any,
    prepare_threshold: int = 5,
) -> List[Record]:
    """
    Execute query with automatic preparation after threshold.

    asyncpg automatically prepares statements that are executed
    frequently on the same connection.

    Args:
        query: SQL query
        *args: Query parameters
        prepare_threshold: Execute this many times before preparing
    """
    await self._ensure_connected()

    # Track query frequency
    query_sig = self._get_query_signature(query)
    self._query_frequency[query_sig] = self._query_frequency.get(query_sig, 0) + 1

    should_prepare = self._query_frequency[query_sig] >= prepare_threshold

    async def _fetch() -> List[Record]:
        async with self.pool.acquire() as conn:
            if should_prepare:
                # asyncpg will auto-prepare after first execution on this connection
                return await conn.fetch(query, *args)
            else:
                # Execute without preparation
                return await conn.fetch(query, *args)

    return await self._execute_with_retry(_fetch)
```

#### 6.2 Prepared Statement Monitoring

Track which queries benefit from preparation:

```python
def get_prepared_statement_stats(self) -> Dict[str, Any]:
    """Get prepared statement cache statistics."""
    total_prepared = len(self._prepared_cache.cache)
    top_queries = sorted(
        self._prepared_cache.access_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return {
        "total_prepared": total_prepared,
        "cache_size_limit": self._prepared_cache.max_size,
        "cache_utilization": total_prepared / self._prepared_cache.max_size,
        "top_10_queries": [
            {
                "query": query[:100],
                "execution_count": count
            }
            for query, count in top_queries
        ]
    }
```

---

## 7. Connection Leak Prevention

### Current State: **BASIC PROTECTION**

The current implementation uses context managers which help prevent leaks:

```python
async def fetch(self, query: str, *args: Any) -> List[Record]:
    async def _fetch() -> List[Record]:
        async with self.pool.acquire() as conn:  # Auto-releases
            return await conn.fetch(query, *args)
    return await self._execute_with_retry(_fetch)
```

**However:**
- No leak detection
- No connection lifetime tracking
- No warning on long-held connections

### Assessment: **Needs Improvement**

**Recommendations:**

#### 7.1 Connection Leak Detector

Track connection acquisitions and detect leaks:

```python
@dataclass
class ConnectionLease:
    """Tracks a connection lease."""
    conn_id: int
    acquired_at: float
    stack_trace: str
    released: bool = False

class ConnectionLeakDetector:
    """Detects connection leaks in the pool."""

    def __init__(self, leak_threshold: float = 60.0):
        self.leak_threshold = leak_threshold
        self.active_leases: Dict[int, ConnectionLease] = {}
        self._lock = asyncio.Lock()

    async def track_acquisition(self, conn: Connection) -> None:
        """Track connection acquisition."""
        import traceback

        conn_id = id(conn)
        lease = ConnectionLease(
            conn_id=conn_id,
            acquired_at=time.time(),
            stack_trace=''.join(traceback.format_stack()),
            released=False
        )

        async with self._lock:
            self.active_leases[conn_id] = lease

    async def track_release(self, conn: Connection) -> None:
        """Track connection release."""
        conn_id = id(conn)

        async with self._lock:
            if conn_id in self.active_leases:
                self.active_leases[conn_id].released = True
                del self.active_leases[conn_id]

    async def detect_leaks(self) -> List[ConnectionLease]:
        """Detect potential connection leaks."""
        leaks = []
        current_time = time.time()

        async with self._lock:
            for lease in self.active_leases.values():
                if not lease.released:
                    age = current_time - lease.acquired_at
                    if age > self.leak_threshold:
                        leaks.append(lease)
                        logger.warning(
                            "connection_leak_detected",
                            conn_id=lease.conn_id,
                            age_seconds=round(age, 2),
                            stack_trace=lease.stack_trace
                        )

        return leaks

# Add to AsyncDatabaseManager
class AsyncDatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        # ... existing code ...
        self._leak_detector = ConnectionLeakDetector(leak_threshold=60.0)
        self._leak_check_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        # ... existing connection code ...

        # Start leak detection background task
        self._leak_check_task = asyncio.create_task(
            self._leak_detection_loop()
        )

    async def _leak_detection_loop(self) -> None:
        """Background task to detect connection leaks."""
        while self._connected:
            await asyncio.sleep(30)  # Check every 30 seconds
            leaks = await self._leak_detector.detect_leaks()
            if leaks:
                logger.error(
                    "connection_leaks_found",
                    count=len(leaks),
                    pool_size=self._pool_stats.size,
                    free=self._pool_stats.free_size
                )
```

#### 7.2 Connection Timeout Enforcement

Automatically reclaim connections held too long:

```python
@asynccontextmanager
async def acquire(
    self,
    timeout: float = 300.0  # 5 minutes max
) -> AsyncGenerator[Connection, None]:
    """
    Acquire a connection from the pool with lease timeout.

    Args:
        timeout: Max time connection can be held (seconds)
    """
    await self._ensure_connected()

    async with self.pool.acquire() as conn:
        await self._leak_detector.track_acquisition(conn)
        start = time.time()

        try:
            # Use asyncio.timeout to enforce max lease time
            async with asyncio.timeout(timeout):
                yield conn
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.error(
                "connection_lease_timeout",
                elapsed=elapsed,
                timeout=timeout
            )
            raise
        finally:
            await self._leak_detector.track_release(conn)
```

#### 7.3 Connection Usage Monitoring

Expose metrics on connection usage patterns:

```python
def get_connection_usage_stats(self) -> Dict[str, Any]:
    """Get connection usage statistics."""
    self._update_pool_stats()

    return {
        "pool_size": self._pool_stats.size,
        "free_connections": self._pool_stats.free_size,
        "used_connections": self._pool_stats.used_size,
        "utilization": self._pool_stats.used_size / max(self._pool_stats.size, 1),
        "active_leases": len(self._leak_detector.active_leases),
        "leaked_connections": len([
            lease for lease in self._leak_detector.active_leases.values()
            if time.time() - lease.acquired_at > self._leak_detector.leak_threshold
        ]),
        "avg_connection_age_ms": self._calculate_avg_connection_age(),
    }
```

---

## 8. Metrics and Observability Hooks

### Current Implementation

#### Query Statistics
```python
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
```

#### OpenTelemetry Integration
```python
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
```

#### Prometheus Metrics (from `observability.py`)
```python
# Database metrics
self.db_query_count = Counter(
    "ava_db_queries_total",
    "Total database queries",
    ["operation", "status"],
)

self.db_query_latency = Histogram(
    "ava_db_query_duration_seconds",
    "Database query latency",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

self.db_pool_size = Gauge(
    "ava_db_pool_connections",
    "Database connection pool size",
    ["state"],  # "active", "idle", "total"
)
```

### Assessment: **Good but Incomplete**

**Strengths:**
- Basic query statistics tracked
- OpenTelemetry tracing support
- Prometheus metrics defined in observability module

**Weaknesses:**
- Query statistics not exposed via Prometheus metrics
- No per-table or per-query-type breakdowns
- No slow query logging to database
- Missing key metrics (connection wait time, pool exhaustion events)

**Recommendations:**

#### 8.1 Comprehensive Metrics Export

Connect database stats to Prometheus metrics:

```python
class AsyncDatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        # ... existing code ...
        self._metrics_export_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        # ... existing connection code ...

        # Start metrics export background task
        self._metrics_export_task = asyncio.create_task(
            self._metrics_export_loop()
        )

    async def _metrics_export_loop(self) -> None:
        """Background task to export metrics to Prometheus."""
        from backend.infrastructure.observability import metrics

        while self._connected:
            await asyncio.sleep(10)  # Export every 10 seconds

            try:
                self._update_pool_stats()

                # Export pool metrics
                metrics.db_pool_size.labels(state="total").set(
                    self._pool_stats.size
                )
                metrics.db_pool_size.labels(state="idle").set(
                    self._pool_stats.free_size
                )
                metrics.db_pool_size.labels(state="active").set(
                    self._pool_stats.used_size
                )

                # Export query stats
                # (Already tracked via record_db_query in observability.py)

            except Exception as e:
                logger.error("metrics_export_failed", error=str(e))
```

#### 8.2 Per-Table Query Metrics

Track which tables are accessed most frequently:

```python
class TableAccessTracker:
    """Tracks per-table query statistics."""

    def __init__(self):
        self.table_stats: Dict[str, Dict[str, int]] = {}
        self._lock = asyncio.Lock()

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        import re
        # Simple regex to find table names after FROM/JOIN/INTO/UPDATE
        pattern = r'\b(?:FROM|JOIN|INTO|UPDATE)\s+([a-z_][a-z0-9_]*)'
        matches = re.findall(pattern, query.lower())
        return list(set(matches))

    async def record_access(self, query: str, operation: str) -> None:
        """Record table access."""
        tables = self._extract_tables(query)

        async with self._lock:
            for table in tables:
                if table not in self.table_stats:
                    self.table_stats[table] = {
                        "select": 0,
                        "insert": 0,
                        "update": 0,
                        "delete": 0,
                    }

                op = operation.lower().split()[0]
                if op in self.table_stats[table]:
                    self.table_stats[table][op] += 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get per-table statistics."""
        return self.table_stats.copy()
```

#### 8.3 Slow Query Logger (Database Persistence)

Log slow queries to a database table for analysis:

```python
async def _log_slow_query(
    self,
    query: str,
    args: Tuple[Any, ...],
    duration_ms: float,
    threshold_ms: float = 100.0
) -> None:
    """Log slow query to database for analysis."""
    if duration_ms < threshold_ms:
        return

    try:
        # Use a separate connection to avoid recursion
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO slow_query_log
                (query, args, duration_ms, threshold_ms, timestamp)
                VALUES ($1, $2, $3, $4, $5)
                """,
                query[:5000],  # Truncate long queries
                str(args)[:1000],
                duration_ms,
                threshold_ms,
                datetime.now()
            )
    except Exception as e:
        # Don't fail the original query if logging fails
        logger.warning("slow_query_log_failed", error=str(e))
```

**Migration for slow query log table:**

```sql
CREATE TABLE IF NOT EXISTS slow_query_log (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    args TEXT,
    duration_ms NUMERIC(10, 2) NOT NULL,
    threshold_ms NUMERIC(10, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    query_signature TEXT GENERATED ALWAYS AS (
        -- Normalized query for grouping
        regexp_replace(query, '\$\d+', '$N', 'g')
    ) STORED
);

CREATE INDEX idx_slow_query_log_timestamp ON slow_query_log(timestamp DESC);
CREATE INDEX idx_slow_query_log_duration ON slow_query_log(duration_ms DESC);
CREATE INDEX idx_slow_query_log_signature ON slow_query_log(query_signature);
```

#### 8.4 Connection Wait Time Metrics

Track how long queries wait for available connections:

```python
async def fetch_with_wait_tracking(
    self,
    query: str,
    *args: Any,
) -> List[Record]:
    """Execute query with connection wait time tracking."""
    await self._ensure_connected()

    wait_start = time.perf_counter()

    async def _fetch() -> List[Record]:
        # Track time waiting for connection
        async with self.pool.acquire() as conn:
            wait_time_ms = (time.perf_counter() - wait_start) * 1000

            if wait_time_ms > 100:  # Warn if waiting > 100ms
                logger.warning(
                    "connection_wait_slow",
                    wait_time_ms=round(wait_time_ms, 2),
                    pool_free=self._pool_stats.free_size,
                    pool_used=self._pool_stats.used_size
                )

            # Track in metrics
            from backend.infrastructure.observability import metrics
            metrics.db_connection_wait_time.observe(wait_time_ms / 1000)

            # Execute query
            return await conn.fetch(query, *args)

    return await self._execute_with_retry(_fetch)
```

**Add to Prometheus metrics:**

```python
# In observability.py
self.db_connection_wait_time = Histogram(
    "ava_db_connection_wait_seconds",
    "Time spent waiting for database connection",
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
)

self.db_pool_exhausted_total = Counter(
    "ava_db_pool_exhausted_total",
    "Number of times connection pool was exhausted"
)
```

---

## 9. Modern Best Practices Recommendations

### Priority 1: Critical (Implement Immediately)

#### 9.1 Add PgBouncer for Connection Pooling

**Problem:** Application-level pooling has limitations (no dynamic sizing, no connection reuse across application instances).

**Solution:** Deploy PgBouncer as a connection pooler between the application and PostgreSQL.

**Configuration:**

```ini
# /etc/pgbouncer/pgbouncer.ini

[databases]
ava = host=localhost port=5432 dbname=ava

[pgbouncer]
listen_addr = 127.0.0.1
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Pool settings
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
reserve_pool_size = 10
reserve_pool_timeout = 3

# Connection lifetime
server_lifetime = 3600
server_idle_timeout = 600

# Logging
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
```

**Update application config:**

```python
@dataclass(frozen=True)
class DatabaseConfig:
    # Change to point to PgBouncer
    host: str = "localhost"
    port: int = 6432  # PgBouncer port instead of 5432

    # Reduce application pool size (PgBouncer handles pooling)
    min_pool_size: int = 2
    max_pool_size: int = 10
```

**Benefits:**
- Connection reuse across application instances
- Dynamic pool sizing
- Transaction-level pooling
- Better connection lifecycle management

#### 9.2 Add pg_stat_statements for Query Analysis

**Problem:** No visibility into actual query performance in production.

**Solution:** Enable `pg_stat_statements` extension.

**Migration:**

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Query to find slow queries
SELECT
    calls,
    total_exec_time / 1000 AS total_seconds,
    mean_exec_time AS avg_ms,
    stddev_exec_time AS stddev_ms,
    min_exec_time AS min_ms,
    max_exec_time AS max_ms,
    rows / calls AS avg_rows,
    query
FROM pg_stat_statements
WHERE calls > 100
ORDER BY total_exec_time DESC
LIMIT 20;
```

**Add to health check:**

```python
async def check_query_performance(self) -> HealthCheck:
    """Check for slow queries using pg_stat_statements."""
    try:
        slow_queries = await self.fetch(
            """
            SELECT
                query,
                calls,
                mean_exec_time,
                total_exec_time
            FROM pg_stat_statements
            WHERE mean_exec_time > 100
                AND calls > 10
            ORDER BY total_exec_time DESC
            LIMIT 10
            """
        )

        if slow_queries:
            return HealthCheck(
                name="query_performance",
                status=HealthStatus.DEGRADED,
                message=f"{len(slow_queries)} slow queries detected",
                details={"slow_queries": slow_queries}
            )

        return HealthCheck(
            name="query_performance",
            status=HealthStatus.HEALTHY,
            message="No slow queries detected"
        )
    except Exception as e:
        return HealthCheck(
            name="query_performance",
            status=HealthStatus.UNHEALTHY,
            message=f"Error checking query performance: {str(e)}"
        )
```

#### 9.3 Add Statement Timeout Monitoring

**Problem:** No alerting when queries approach timeout limits.

**Solution:** Track queries that take >80% of timeout threshold.

```python
async def fetch_with_timeout_monitoring(
    self,
    query: str,
    *args: Any,
    timeout: Optional[float] = None,
) -> List[Record]:
    """Execute query with timeout monitoring."""
    timeout = timeout or self.config.statement_timeout
    start = time.perf_counter()

    try:
        result = await self.fetch(query, *args, timeout=timeout)
        duration = time.perf_counter() - start

        # Alert if query took >80% of timeout
        if duration > (timeout * 0.8):
            logger.warning(
                "query_near_timeout",
                duration_s=duration,
                timeout_s=timeout,
                utilization_pct=(duration / timeout) * 100,
                query=query[:200]
            )

        return result
    except asyncio.TimeoutError:
        logger.error(
            "query_timeout",
            timeout_s=timeout,
            query=query[:200]
        )
        raise
```

### Priority 2: Important (Implement Within 2 Weeks)

#### 9.4 Add Query Result Caching

**Problem:** Expensive queries executed repeatedly with same parameters.

**Solution:** Implement query result caching with TTL.

```python
class QueryResultCache:
    """LRU cache for query results."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = asyncio.Lock()

    def _get_cache_key(self, query: str, args: Tuple[Any, ...]) -> str:
        """Generate cache key from query and args."""
        import hashlib
        import json

        # Normalize query
        normalized = ' '.join(query.split())

        # Serialize args
        args_json = json.dumps(args, default=str)

        # Hash
        key = f"{normalized}:{args_json}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def get(self, query: str, args: Tuple[Any, ...]) -> Optional[Any]:
        """Get cached result if available and fresh."""
        key = self._get_cache_key(query, args)

        async with self._lock:
            if key in self.cache:
                result, expires_at = self.cache[key]
                if time.time() < expires_at:
                    return result
                else:
                    del self.cache[key]

        return None

    async def set(
        self,
        query: str,
        args: Tuple[Any, ...],
        result: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Cache query result."""
        key = self._get_cache_key(query, args)
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        async with self._lock:
            # Evict LRU if cache full
            if len(self.cache) >= self.max_size:
                # Simple eviction: remove expired entries first
                expired = [
                    k for k, (_, exp) in self.cache.items()
                    if time.time() >= exp
                ]
                for k in expired:
                    del self.cache[k]

                # If still full, remove oldest
                if len(self.cache) >= self.max_size:
                    oldest = min(self.cache.items(), key=lambda x: x[1][1])
                    del self.cache[oldest[0]]

            self.cache[key] = (result, expires_at)

# Usage
async def fetch_cached(
    self,
    query: str,
    *args: Any,
    cache_ttl: Optional[int] = None,
) -> List[Record]:
    """Execute query with result caching."""

    # Check cache
    cached = await self._query_cache.get(query, args)
    if cached is not None:
        logger.debug("query_cache_hit", query=query[:100])
        return cached

    # Execute query
    result = await self.fetch(query, *args)

    # Cache result
    await self._query_cache.set(query, args, result, ttl=cache_ttl)

    return result
```

#### 9.5 Add Connection Pool Prewarming on Startup

**Problem:** First requests after deployment are slow due to connection establishment.

**Solution:** Prewarm the pool during application startup.

```python
# In FastAPI startup event
from backend.infrastructure.database import get_database

@app.on_event("startup")
async def startup_event():
    db = await get_database()

    # Warm up the pool
    warmed = await db.warmup_pool(target_connections=10)
    logger.info("database_pool_warmed", connections=warmed)
```

#### 9.6 Add Automatic Index Recommendations

**Problem:** No automated detection of missing indexes.

**Solution:** Analyze pg_stat_user_tables and suggest indexes.

```python
async def get_index_recommendations(self) -> List[Dict[str, Any]]:
    """Get index recommendations based on table statistics."""
    recommendations = await self.fetch(
        """
        SELECT
            schemaname,
            tablename,
            seq_scan,
            seq_tup_read,
            idx_scan,
            n_live_tup,
            CASE
                WHEN seq_scan > 0
                THEN ROUND((seq_tup_read::numeric / seq_scan), 2)
                ELSE 0
            END AS avg_seq_read
        FROM pg_stat_user_tables
        WHERE seq_scan > 100
            AND idx_scan < seq_scan
            AND n_live_tup > 1000
        ORDER BY seq_tup_read DESC
        LIMIT 20;
        """
    )

    return [
        {
            "table": f"{row['schemaname']}.{row['tablename']}",
            "sequential_scans": row['seq_scan'],
            "avg_rows_per_scan": row['avg_seq_read'],
            "recommendation": (
                f"Consider adding index on {row['tablename']} - "
                f"table has {row['n_live_tup']:,} rows and "
                f"{row['seq_scan']} sequential scans"
            )
        }
        for row in recommendations
    ]
```

### Priority 3: Nice to Have (Future Enhancements)

#### 9.7 Add Query Explain Analyzer

Automatically run EXPLAIN ANALYZE on slow queries:

```python
async def analyze_slow_query(self, query: str, *args: Any) -> Dict[str, Any]:
    """Run EXPLAIN ANALYZE on a query."""
    explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

    result = await self.fetchval(explain_query, *args)
    return result[0]  # JSON result
```

#### 9.8 Add Connection Pooling Per Tenant

For multi-tenant scenarios:

```python
class MultiTenantDatabaseManager:
    """Manage separate connection pools per tenant."""

    def __init__(self):
        self.pools: Dict[str, AsyncDatabaseManager] = {}

    async def get_pool(self, tenant_id: str) -> AsyncDatabaseManager:
        if tenant_id not in self.pools:
            config = DatabaseConfig.from_env()
            config = config.with_overrides(
                database=f"ava_{tenant_id}",
                max_pool_size=20
            )
            self.pools[tenant_id] = AsyncDatabaseManager(config)
            await self.pools[tenant_id].connect()

        return self.pools[tenant_id]
```

#### 9.9 Add Read Replica Support

Separate read-only queries to replicas:

```python
class ReplicatedDatabaseManager:
    """Manage primary and read replica connections."""

    def __init__(self, config: DatabaseConfig):
        self.primary = AsyncDatabaseManager(config)

        # Configure read replicas
        replica_config = config.with_overrides(
            host=os.getenv("DB_REPLICA_HOST", config.host),
            max_pool_size=30  # More connections for read-heavy workload
        )
        self.replica = AsyncDatabaseManager(replica_config)

    async def fetch(
        self,
        query: str,
        *args: Any,
        use_replica: bool = True
    ) -> List[Record]:
        """Execute SELECT query, optionally on replica."""
        pool = self.replica if use_replica else self.primary
        return await pool.fetch(query, *args)

    async def execute(
        self,
        query: str,
        *args: Any
    ) -> str:
        """Execute write query (always on primary)."""
        return await self.primary.execute(query, *args)
```

---

## 10. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)

**Goal:** Improve observability and prevent failures

1. ✅ Add comprehensive health checks (Section 2.1)
2. ✅ Add circuit breaker for health checks (Section 2.2)
3. ✅ Add jittered exponential backoff (Section 3.1)
4. ✅ Add connection leak detector (Section 7.1)
5. ✅ Deploy PgBouncer (Section 9.1)

**Files to modify:**
- `backend/infrastructure/database.py`
- `backend/infrastructure/observability.py`
- `backend/config.py`

### Phase 2: Performance Optimizations (Week 2-3)

**Goal:** Improve query performance and reduce latency

1. ✅ Implement prepared statement cache (Section 6.1)
2. ✅ Add query result caching (Section 9.4)
3. ✅ Add query timeout monitoring (Section 9.3)
4. ✅ Enable pg_stat_statements (Section 9.2)
5. ✅ Add slow query logging (Section 8.3)

**Files to create:**
- `backend/infrastructure/query_cache.py`
- `migrations/add_slow_query_log_table.sql`

### Phase 3: Advanced Features (Week 4+)

**Goal:** Production-grade robustness

1. ✅ Add transaction deadlock retry (Section 5.3)
2. ✅ Add per-priority connection pools (Section 1.2)
3. ✅ Add connection pool prewarming (Section 9.5)
4. ✅ Add index recommendations (Section 9.6)
5. ✅ Add read replica support (Section 9.9)

---

## 11. Metrics Dashboard

### Key Metrics to Monitor

#### Connection Pool Health
```promql
# Pool utilization
ava_db_pool_connections{state="active"} /
ava_db_pool_connections{state="total"}

# Connection wait time (p99)
histogram_quantile(0.99,
  rate(ava_db_connection_wait_seconds_bucket[5m]))

# Pool exhaustion events
rate(ava_db_pool_exhausted_total[5m])
```

#### Query Performance
```promql
# Query latency (p95)
histogram_quantile(0.95,
  rate(ava_db_query_duration_seconds_bucket[5m]))

# Query failure rate
rate(ava_db_queries_total{status="error"}[5m]) /
rate(ava_db_queries_total[5m])

# Slow queries per second
rate(ava_db_slow_queries_total[5m])
```

#### Health Check Status
```promql
# Database availability
ava_database_health{check="connectivity"} == 1

# Health check failures
rate(ava_database_health_check_failures_total[5m])
```

---

## 12. Testing Recommendations

### Load Testing Scenarios

#### Scenario 1: Connection Pool Exhaustion
```python
# Test with more concurrent requests than pool size
async def test_pool_exhaustion():
    db = await get_database()

    # Start 100 concurrent queries (pool max = 50)
    tasks = [
        db.fetch("SELECT pg_sleep(5)")
        for _ in range(100)
    ]

    start = time.time()
    results = await asyncio.gather(*tasks)
    duration = time.time() - start

    # Should not timeout, should queue gracefully
    assert duration < 10  # Should complete within 10 seconds
```

#### Scenario 2: Query Timeout
```python
async def test_query_timeout():
    db = await get_database()

    with pytest.raises(asyncio.TimeoutError):
        await db.fetch(
            "SELECT pg_sleep(60)",
            timeout=1.0  # 1 second timeout
        )
```

#### Scenario 3: Deadlock Recovery
```python
async def test_deadlock_recovery():
    db = await get_database()

    async def update_a_then_b(conn):
        await conn.execute("UPDATE accounts SET balance = balance + 1 WHERE id = 1")
        await asyncio.sleep(0.1)
        await conn.execute("UPDATE accounts SET balance = balance + 1 WHERE id = 2")

    async def update_b_then_a(conn):
        await conn.execute("UPDATE accounts SET balance = balance + 1 WHERE id = 2")
        await asyncio.sleep(0.1)
        await conn.execute("UPDATE accounts SET balance = balance + 1 WHERE id = 1")

    # Should retry and succeed
    result = await db.transaction_with_retry(update_a_then_b)
```

---

## 13. Migration Plan

### Step 1: Deploy Monitoring (No Risk)

Add observability without changing behavior:

```bash
# 1. Enable pg_stat_statements
psql -d ava -c "CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"

# 2. Create slow query log table
psql -d ava -f migrations/add_slow_query_log_table.sql

# 3. Deploy updated database.py with metrics
git checkout feature/db-monitoring
python -m pytest backend/tests/test_database.py
# Deploy to staging
# Monitor for 24 hours
# Deploy to production
```

### Step 2: Deploy PgBouncer (Medium Risk)

Add connection pooler:

```bash
# 1. Install PgBouncer
sudo apt-get install pgbouncer

# 2. Configure PgBouncer
sudo cp config/pgbouncer.ini /etc/pgbouncer/

# 3. Update application config (point to port 6432)
# In .env:
# DB_PORT=6432
# DB_POOL_MAX=10  # Reduce app pool size

# 4. Restart services
sudo systemctl restart pgbouncer
sudo systemctl restart ava-backend

# 5. Monitor pool metrics
curl http://localhost:9090/metrics | grep db_pool
```

### Step 3: Deploy Performance Features (Low Risk)

Add caching and prepared statements:

```bash
# 1. Deploy code changes
git checkout feature/db-performance
python -m pytest backend/tests/

# 2. Gradual rollout (feature flag)
# In config.py:
ENABLE_QUERY_CACHE=True
ENABLE_PREPARED_STATEMENTS=True

# 3. Monitor cache hit rate
curl http://localhost:9090/metrics | grep cache_hits
```

---

## 14. Alerting Rules

### Prometheus Alert Rules

```yaml
groups:
  - name: database
    interval: 30s
    rules:
      # Connection pool exhaustion
      - alert: DatabasePoolExhausted
        expr: ava_db_pool_connections{state="free"} < 2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "Only {{ $value }} free connections remaining"

      # High query latency
      - alert: DatabaseSlowQueries
        expr: histogram_quantile(0.95, rate(ava_db_query_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Database queries are slow"
          description: "p95 query latency is {{ $value }}s"

      # Query failure rate
      - alert: DatabaseQueryFailures
        expr: rate(ava_db_queries_total{status="error"}[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High database query failure rate"
          description: "{{ $value }} queries/sec failing"

      # Health check failures
      - alert: DatabaseUnhealthy
        expr: ava_database_health{check="connectivity"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is unhealthy"
          description: "Database health check failing"
```

---

## 15. Summary & Scoring

### Current State: 7/10

**Strengths:**
- ✅ Modern async architecture with asyncpg
- ✅ Connection pooling with reasonable defaults
- ✅ Retry logic with exponential backoff
- ✅ Basic health checks
- ✅ Query statistics tracking
- ✅ OpenTelemetry integration hooks

**Critical Gaps:**
- ❌ No prepared statement caching (performance loss)
- ❌ No connection leak detection (stability risk)
- ❌ No query result caching (unnecessary load)
- ❌ Limited observability (hard to debug issues)
- ❌ No circuit breaker (cascading failures)
- ❌ No per-query timeout monitoring (hard to optimize)

### Target State: 9.5/10

After implementing Priority 1 and Priority 2 recommendations:

- ✅ Comprehensive health checks with circuit breaker
- ✅ Connection leak detection and prevention
- ✅ Prepared statement caching
- ✅ Query result caching
- ✅ PgBouncer deployment
- ✅ pg_stat_statements enabled
- ✅ Slow query logging
- ✅ Full Prometheus metrics export
- ✅ Connection pool prewarming
- ✅ Jittered exponential backoff

**Remaining for 10/10:**
- Read replica support (for scale)
- Multi-tenant pooling (if needed)
- Query explain analyzer (for optimization)

---

## Appendix A: Configuration Reference

### Recommended Production Configuration

```python
@dataclass(frozen=True)
class DatabaseConfig:
    # Connection
    host: str = "localhost"
    port: int = 6432  # PgBouncer port
    database: str = "ava"
    user: str = "ava_app"
    password: str = "<from-env>"

    # Pool settings (with PgBouncer)
    min_pool_size: int = 5
    max_pool_size: int = 20  # Reduced (PgBouncer handles pooling)

    # Timeouts
    connect_timeout: float = 5.0
    command_timeout: float = 30.0
    statement_timeout: float = 30.0
    transaction_timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.5
    retry_backoff: float = 2.0
    retry_jitter: float = 0.3  # +/- 30%

    # Health check
    health_check_interval: float = 30.0
    health_check_timeout: float = 5.0

    # Connection lifecycle
    max_connection_age: float = 3600.0  # 1 hour
    max_idle_time: float = 300.0        # 5 minutes

    # Observability
    enable_slow_query_log: bool = True
    slow_query_threshold_ms: float = 100.0
    enable_query_cache: bool = True
    query_cache_ttl: int = 60

    # Features
    enable_prepared_statements: bool = True
    enable_leak_detection: bool = True
    leak_detection_threshold: float = 60.0
```

---

## Appendix B: File Structure

Recommended file organization:

```
backend/
├── infrastructure/
│   ├── database.py            # Main async database manager (ENHANCED)
│   ├── database_metrics.py    # Metrics export logic (NEW)
│   ├── query_cache.py         # Query result caching (NEW)
│   ├── connection_leak.py     # Leak detection (NEW)
│   ├── async_db.py            # Alternative implementation (keep for now)
│   └── observability.py       # Prometheus metrics (ENHANCED)
├── database/
│   └── connection.py          # Legacy sync pool (DEPRECATED)
├── migrations/
│   ├── add_slow_query_log.sql        # New table (NEW)
│   └── add_pg_stat_statements.sql    # Extension (NEW)
└── tests/
    ├── test_database.py       # Unit tests (ENHANCED)
    └── test_database_load.py  # Load tests (NEW)
```

---

**End of Report**

This analysis provides a comprehensive roadmap for optimizing the Magnus database infrastructure from a solid 7/10 to a production-grade 9.5/10 system with world-class observability, performance, and reliability.
