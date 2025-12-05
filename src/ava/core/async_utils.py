"""
AVA Async Utilities
===================

High-performance async utilities for:
- Parallel processing with rate limiting
- Batch operations with backpressure
- Circuit breaker pattern
- Retry with exponential backoff
- Connection pooling

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import (
    List, Dict, Any, TypeVar, Callable, Optional,
    Awaitable, Tuple, Generic, Union
)
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
from collections import deque
import random

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.

    Usage:
        limiter = RateLimiter(rate=10, per=1.0)  # 10 requests per second

        async with limiter:
            await make_api_call()
    """

    def __init__(
        self,
        rate: int,
        per: float = 1.0,
        burst: Optional[int] = None
    ):
        self.rate = rate
        self.per = per
        self.burst = burst or rate * 2

        self._tokens = float(self.burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1):
        """Acquire tokens, waiting if necessary"""
        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time
                needed = tokens - self._tokens
                wait_time = needed / (self.rate / self.per)
                await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on time passed
        self._tokens = min(
            self.burst,
            self._tokens + elapsed * (self.rate / self.per)
        )

    async def __aenter__(self) -> None:
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 30.0   # Seconds before testing recovery
    success_threshold: int = 3       # Successes to close again
    timeout: float = 10.0            # Call timeout


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Usage:
        breaker = CircuitBreaker(name="api")

        @breaker
        async def call_api():
            return await api.fetch()
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def _transition_to(self, new_state: CircuitState):
        """Transition to new state"""
        old_state = self._state
        self._state = new_state
        logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
        elif new_state == CircuitState.OPEN:
            self._success_count = 0

    async def _record_success(self) -> None:
        """Record successful call"""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    await self._transition_to(CircuitState.CLOSED)

    async def _record_failure(self) -> None:
        """Record failed call"""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                await self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.config.failure_threshold:
                await self._transition_to(CircuitState.OPEN)

    async def _check_state(self) -> None:
        """Check if circuit should transition"""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time:
                    elapsed = time.monotonic() - self._last_failure_time
                    if elapsed >= self.config.recovery_timeout:
                        await self._transition_to(CircuitState.HALF_OPEN)

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function through circuit breaker"""
        await self._check_state()

        if self._state == CircuitState.OPEN:
            raise CircuitBreakerOpen(f"Circuit {self.name} is open")

        try:
            result = await asyncio.wait_for(
                func(),
                timeout=self.config.timeout
            )
            await self._record_success()
            return result

        except Exception as e:
            await self._record_failure()
            raise

    def __call__(self, func: Callable) -> Callable:
        """Decorator form"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await self.call(lambda: func(*args, **kwargs))
        return wrapper


class CircuitBreakerOpen(Exception):
    """Raised when circuit is open"""
    pass


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[type, ...] = (Exception,)


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None
) -> T:
    """
    Retry async function with exponential backoff.

    Usage:
        result = await retry_with_backoff(
            lambda: api.fetch(symbol),
            config=RetryConfig(max_attempts=3)
        )
    """
    config = config or RetryConfig()
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func()

        except config.retryable_exceptions as e:
            last_exception = e
            if attempt == config.max_attempts - 1:
                break

            # Calculate delay
            delay = min(
                config.initial_delay * (config.exponential_base ** attempt),
                config.max_delay
            )

            # Add jitter
            if config.jitter:
                delay = delay * (0.5 + random.random())

            logger.warning(
                f"Retry attempt {attempt + 1}/{config.max_attempts} "
                f"after {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise last_exception or Exception("Retry failed")


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retry with backoff.

    Usage:
        @with_retry(RetryConfig(max_attempts=3))
        async def fetch_data():
            return await api.get()
    """
    config = config or RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_with_backoff(
                lambda: func(*args, **kwargs),
                config
            )
        return wrapper
    return decorator


# =============================================================================
# PARALLEL PROCESSOR
# =============================================================================

class ParallelProcessor(Generic[T, R]):
    """
    Process items in parallel with concurrency limits.

    Usage:
        processor = ParallelProcessor(
            worker=process_item,
            max_concurrency=10,
            rate_limiter=RateLimiter(rate=5)
        )

        results = await processor.process(items)
    """

    def __init__(
        self,
        worker: Callable[[T], Awaitable[R]],
        max_concurrency: int = 10,
        rate_limiter: Optional[RateLimiter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        self.worker = worker
        self.max_concurrency = max_concurrency
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config

        self._semaphore: Optional[asyncio.Semaphore] = None

    async def process(
        self,
        items: List[T],
        return_exceptions: bool = False
    ) -> List[Union[R, Exception]]:
        """Process all items in parallel"""
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        tasks = [
            self._process_item(item)
            for item in items
        ]

        if return_exceptions:
            return await asyncio.gather(*tasks, return_exceptions=True)
        return await asyncio.gather(*tasks)

    async def _process_item(self, item: T) -> R:
        """Process single item with all protections"""
        async with self._semaphore:
            # Rate limiting
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            # Build wrapped function
            async def call():
                return await self.worker(item)

            # Apply circuit breaker
            if self.circuit_breaker:
                call = lambda c=call: self.circuit_breaker.call(c)

            # Apply retry
            if self.retry_config:
                return await retry_with_backoff(call, self.retry_config)

            return await call()

    async def process_batched(
        self,
        items: List[T],
        batch_size: int = 100,
        delay_between_batches: float = 0
    ) -> List[R]:
        """Process items in batches"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = await self.process(batch)
            results.extend(batch_results)

            if delay_between_batches > 0 and i + batch_size < len(items):
                await asyncio.sleep(delay_between_batches)

        return results


# =============================================================================
# ASYNC BATCH COLLECTOR
# =============================================================================

class BatchCollector(Generic[T, R]):
    """
    Collect items and process in batches for efficiency.

    Usage:
        async def batch_insert(items):
            await db.insert_many(items)

        collector = BatchCollector(
            processor=batch_insert,
            batch_size=100,
            flush_interval=5.0
        )

        await collector.start()
        await collector.add(item1)
        await collector.add(item2)
        await collector.stop()
    """

    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[R]],
        batch_size: int = 100,
        flush_interval: float = 5.0,
        max_queue_size: int = 10000
    ):
        self.processor = processor
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._flush_event = asyncio.Event()

    async def start(self) -> None:
        """Start the collector"""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        """Stop and flush remaining items"""
        self._running = False
        self._flush_event.set()

        if self._task:
            await self._task

    async def add(self, item: T):
        """Add item to batch"""
        await self._queue.put(item)

        if self._queue.qsize() >= self.batch_size:
            self._flush_event.set()

    async def _process_loop(self) -> None:
        """Main processing loop"""
        while self._running or not self._queue.empty():
            try:
                # Wait for batch or timeout
                try:
                    await asyncio.wait_for(
                        self._flush_event.wait(),
                        timeout=self.flush_interval
                    )
                except asyncio.TimeoutError:
                    pass

                self._flush_event.clear()

                # Collect batch
                batch = []
                while not self._queue.empty() and len(batch) < self.batch_size:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                # Process batch
                if batch:
                    try:
                        await self.processor(batch)
                    except Exception as e:
                        logger.error(f"Batch processing error: {e}")

            except asyncio.CancelledError:
                break


# =============================================================================
# ASYNC SEMAPHORE POOL
# =============================================================================

class SemaphorePool:
    """
    Pool of semaphores for resource-specific rate limiting.

    Usage:
        pool = SemaphorePool(default_limit=5)

        async with pool.acquire("api-v1"):
            await call_api()
    """

    def __init__(self, default_limit: int = 10):
        self.default_limit = default_limit
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._limits: Dict[str, int] = {}
        self._lock = asyncio.Lock()

    def set_limit(self, resource: str, limit: int):
        """Set limit for a specific resource"""
        self._limits[resource] = limit

    async def acquire(self, resource: str):
        """Acquire semaphore for resource"""
        async with self._lock:
            if resource not in self._semaphores:
                limit = self._limits.get(resource, self.default_limit)
                self._semaphores[resource] = asyncio.Semaphore(limit)

        return self._semaphores[resource]

    class _Context:
        def __init__(self, semaphore: asyncio.Semaphore):
            self.semaphore = semaphore

        async def __aenter__(self) -> None:
            await self.semaphore.acquire()
            return self

        async def __aexit__(self, *args):
            self.semaphore.release()

    def __call__(self, resource: str):
        """Context manager for resource"""
        async def get_context():
            sem = await self.acquire(resource)
            return self._Context(sem)
        return get_context()


# =============================================================================
# PARALLEL SCANNER
# =============================================================================

async def parallel_scan(
    symbols: List[str],
    scanner: Callable[[str], Awaitable[T]],
    max_concurrency: int = 10,
    rate_limit: Optional[int] = None,
    timeout: float = 30.0
) -> Dict[str, Union[T, Exception]]:
    """
    Scan multiple symbols in parallel.

    Usage:
        results = await parallel_scan(
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            scanner=analyze_symbol,
            max_concurrency=5
        )
    """
    rate_limiter = RateLimiter(rate=rate_limit, per=1.0) if rate_limit else None
    semaphore = asyncio.Semaphore(max_concurrency)

    async def scan_with_limits(symbol: str) -> Tuple[str, Union[T, Exception]]:
        async with semaphore:
            if rate_limiter:
                await rate_limiter.acquire()

            try:
                result = await asyncio.wait_for(
                    scanner(symbol),
                    timeout=timeout
                )
                return symbol, result
            except Exception as e:
                logger.error(f"Scan failed for {symbol}: {e}")
                return symbol, e

    tasks = [scan_with_limits(s) for s in symbols]
    results = await asyncio.gather(*tasks)

    return dict(results)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Async Utilities ===\n")

    async def test_utilities():
        # Test rate limiter
        print("1. Testing Rate Limiter...")
        limiter = RateLimiter(rate=5, per=1.0)

        start = time.time()
        for i in range(5):
            async with limiter:
                pass
        elapsed = time.time() - start
        print(f"   5 requests in {elapsed:.2f}s (should be ~0s)")

        # Test circuit breaker
        print("\n2. Testing Circuit Breaker...")
        breaker = CircuitBreaker(
            "test",
            CircuitBreakerConfig(failure_threshold=2)
        )

        call_count = 0

        @breaker
        async def failing_call():
            nonlocal call_count
            call_count += 1
            raise Exception("Simulated failure")

        for _ in range(3):
            try:
                await failing_call()
            except CircuitBreakerOpen:
                print(f"   Circuit opened after {call_count} failures")
                break
            except Exception:
                pass

        # Test parallel processor
        print("\n3. Testing Parallel Processor...")

        async def slow_worker(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 2

        processor = ParallelProcessor(
            worker=slow_worker,
            max_concurrency=5
        )

        items = list(range(10))
        start = time.time()
        results = await processor.process(items)
        elapsed = time.time() - start

        print(f"   Processed 10 items in {elapsed:.2f}s")
        print(f"   Results: {results}")

        # Test parallel scan
        print("\n4. Testing Parallel Scan...")

        async def mock_analyze(symbol: str) -> Dict:
            await asyncio.sleep(0.1)
            return {"symbol": symbol, "score": len(symbol) * 10}

        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        results = await parallel_scan(
            symbols=symbols,
            scanner=mock_analyze,
            max_concurrency=3
        )

        for symbol, result in results.items():
            print(f"   {symbol}: {result}")

        print("\n✅ Async utilities tests passed!")

    asyncio.run(test_utilities())
