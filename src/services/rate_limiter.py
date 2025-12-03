"""
Rate Limiter Service
Thread-safe token bucket rate limiter with decorator support
"""

import time
import threading
import logging
from typing import Dict, Callable, Optional
from functools import wraps

from src.services.config import ServiceConfig, get_service_config

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket rate limiter - thread-safe implementation

    Allows bursts up to max_calls, then refills at a steady rate.
    More flexible than fixed window rate limiting.
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize token bucket

        Args:
            max_calls: Maximum number of calls allowed in time_window
            time_window: Time window in seconds
        """
        self.capacity = max_calls
        self.tokens = max_calls
        self.refill_rate = max_calls / time_window  # tokens per second
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from bucket

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False if not enough tokens
        """
        with self.lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_and_acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available, then acquire

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return False

            # Calculate wait time until next token
            with self.lock:
                self._refill()
                tokens_needed = tokens - self.tokens
                if tokens_needed > 0:
                    wait_time = tokens_needed / self.refill_rate
                    wait_time = min(wait_time, 1.0)  # Cap at 1 second
                else:
                    wait_time = 0.1

            time.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """Get number of available tokens"""
        with self.lock:
            self._refill()
            return self.tokens

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for tokens

        Args:
            tokens: Number of tokens needed

        Returns:
            Estimated wait time in seconds
        """
        with self.lock:
            self._refill()
            tokens_needed = max(0, tokens - self.tokens)
            return tokens_needed / self.refill_rate


class RateLimiter:
    """
    Global rate limiter managing multiple services
    Thread-safe singleton pattern
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, '_initialized'):
            self._buckets: Dict[str, TokenBucket] = {}
            self._bucket_lock = threading.Lock()
            self._initialized = True
            logger.info("Rate limiter initialized")

    def _get_bucket(self, service_name: str) -> TokenBucket:
        """Get or create token bucket for service"""
        if service_name not in self._buckets:
            with self._bucket_lock:
                if service_name not in self._buckets:
                    # Get service config
                    try:
                        config = get_service_config(service_name)
                        rate_limit = config.rate_limit

                        self._buckets[service_name] = TokenBucket(
                            max_calls=rate_limit.max_calls,
                            time_window=rate_limit.time_window
                        )

                        logger.info(
                            f"Created rate limiter for {service_name}: "
                            f"{rate_limit.max_calls} calls per {rate_limit.time_window}s"
                        )
                    except ValueError:
                        # Unknown service, use default
                        logger.warning(f"Unknown service {service_name}, using default rate limit")
                        self._buckets[service_name] = TokenBucket(
                            max_calls=60,
                            time_window=60
                        )

        return self._buckets[service_name]

    def check_limit(self, service_name: str, tokens: int = 1) -> bool:
        """
        Check if request is allowed under rate limit

        Args:
            service_name: Name of the service
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if allowed, False if rate limit exceeded
        """
        bucket = self._get_bucket(service_name)
        allowed = bucket.acquire(tokens)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {service_name}. "
                f"Wait time: {bucket.get_wait_time(tokens):.2f}s"
            )

        return allowed

    def wait_if_needed(self, service_name: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait if needed to respect rate limit

        Args:
            service_name: Name of the service
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait

        Returns:
            True if acquired, False if timeout
        """
        bucket = self._get_bucket(service_name)

        # Check if we need to wait
        if bucket.acquire(tokens):
            return True

        wait_time = bucket.get_wait_time(tokens)
        logger.info(f"Rate limit reached for {service_name}, waiting {wait_time:.2f}s")

        return bucket.wait_and_acquire(tokens, timeout)

    def get_stats(self, service_name: str) -> Dict[str, float]:
        """
        Get rate limit statistics for a service

        Args:
            service_name: Name of the service

        Returns:
            Dict with available_tokens, capacity, and wait_time
        """
        bucket = self._get_bucket(service_name)

        return {
            "available_tokens": bucket.get_available_tokens(),
            "capacity": bucket.capacity,
            "wait_time": bucket.get_wait_time(1),
            "utilization": 1.0 - (bucket.get_available_tokens() / bucket.capacity)
        }

    def reset(self, service_name: str):
        """Reset rate limiter for a service"""
        with self._bucket_lock:
            if service_name in self._buckets:
                del self._buckets[service_name]
                logger.info(f"Reset rate limiter for {service_name}")

    def reset_all(self) -> None:
        """Reset all rate limiters"""
        with self._bucket_lock:
            self._buckets.clear()
            logger.info("Reset all rate limiters")


# Singleton instance
_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    return _rate_limiter


# =============================================================================
# Decorators
# =============================================================================

def rate_limit(service_name: str, tokens: int = 1, timeout: Optional[float] = None):
    """
    Decorator to apply rate limiting to a function

    Args:
        service_name: Name of the service to rate limit
        tokens: Number of tokens to consume per call
        timeout: Maximum time to wait for rate limit (None = wait forever)

    Example:
        @rate_limit("robinhood", tokens=1, timeout=30)
        def get_positions():
            return rh.get_positions()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Wait if needed
            acquired = limiter.wait_if_needed(service_name, tokens, timeout)

            if not acquired:
                raise TimeoutError(
                    f"Rate limit timeout exceeded for {service_name} "
                    f"(waited {timeout}s)"
                )

            # Execute function
            return func(*args, **kwargs)

        return wrapper
    return decorator


def rate_limit_async(service_name: str, tokens: int = 1, timeout: Optional[float] = None):
    """
    Async decorator to apply rate limiting to an async function

    Args:
        service_name: Name of the service to rate limit
        tokens: Number of tokens to consume per call
        timeout: Maximum time to wait for rate limit

    Example:
        @rate_limit_async("robinhood", tokens=1)
        async def get_positions():
            return await async_get_positions()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Wait if needed (in a thread to not block event loop)
            import asyncio
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None,
                limiter.wait_if_needed,
                service_name,
                tokens,
                timeout
            )

            if not acquired:
                raise TimeoutError(
                    f"Rate limit timeout exceeded for {service_name} "
                    f"(waited {timeout}s)"
                )

            # Execute function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import concurrent.futures

    print("Testing Rate Limiter")
    print("=" * 60)

    limiter = get_rate_limiter()

    # Test 1: Basic rate limiting
    print("\nTest 1: Basic rate limiting (5 calls/second)")

    @rate_limit("test_service", tokens=1)
    def test_function():
        return f"Called at {time.time():.3f}"

    # Create a test service config
    from src.services.config import SERVICE_CONFIGS, ServiceConfig, ServiceRateLimit, RetryPolicy
    SERVICE_CONFIGS["test_service"] = ServiceConfig(
        name="test_service",
        rate_limit=ServiceRateLimit(max_calls=5, time_window=1),
        timeout=10,
        retry_policy=RetryPolicy()
    )

    start = time.time()
    for i in range(10):
        result = test_function()
        elapsed = time.time() - start
        print(f"  Call {i+1}: {result} (elapsed: {elapsed:.3f}s)")

    # Test 2: Concurrent access
    print("\nTest 2: Concurrent access (10 threads)")

    def worker(worker_id: int):
        try:
            result = test_function()
            return f"Worker {worker_id}: {result}"
        except Exception as e:
            return f"Worker {worker_id}: Error - {e}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, i) for i in range(10)]
        for future in concurrent.futures.as_completed(futures):
            print(f"  {future.result()}")

    # Test 3: Check stats
    print("\nTest 3: Rate limiter stats")
    stats = limiter.get_stats("test_service")
    print(f"  Available tokens: {stats['available_tokens']:.2f}")
    print(f"  Capacity: {stats['capacity']}")
    print(f"  Wait time: {stats['wait_time']:.2f}s")
    print(f"  Utilization: {stats['utilization']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Rate limiter tests complete!")
