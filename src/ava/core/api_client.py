"""
Robust API Client
=================

Production-grade HTTP client with retry logic, circuit breaker,
caching, and comprehensive error handling.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
import asyncio
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

import aiohttp
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class APIClientConfig:
    """Configuration for API client"""
    # Retry settings
    max_retries: int = int(os.getenv("API_MAX_RETRIES", "3"))
    retry_backoff_factor: float = float(os.getenv("API_RETRY_BACKOFF", "0.5"))
    retry_statuses: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

    # Timeout settings
    connect_timeout: float = float(os.getenv("API_CONNECT_TIMEOUT", "5.0"))
    read_timeout: float = float(os.getenv("API_READ_TIMEOUT", "30.0"))

    # Circuit breaker settings
    circuit_failure_threshold: int = int(os.getenv("API_CIRCUIT_FAILURES", "5"))
    circuit_recovery_timeout: float = float(os.getenv("API_CIRCUIT_RECOVERY", "60.0"))

    # Rate limiting
    rate_limit_per_second: float = float(os.getenv("API_RATE_LIMIT", "10.0"))

    # Caching
    cache_enabled: bool = os.getenv("API_CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("API_CACHE_TTL", "300"))


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for API resilience.

    Prevents cascading failures by temporarily blocking
    requests to a failing service.
    """
    failure_threshold: int = 5
    recovery_timeout: float = 60.0

    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[float] = field(default=None, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)

    def record_success(self) -> None:
        """Record a successful request"""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed request"""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self._failure_count} failures"
            )

    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker HALF_OPEN - testing recovery")
                    return True
            return False

        # HALF_OPEN - allow one request to test
        return True

    @property
    def state(self) -> CircuitState:
        return self._state


# =============================================================================
# RESPONSE CACHE
# =============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    expires_at: float


class ResponseCache:
    """Simple in-memory response cache with TTL"""

    def __init__(self, default_ttl: int = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        entry = self._cache.get(key)
        if entry and time.time() < entry.expires_at:
            return entry.value
        elif entry:
            del self._cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cache value with TTL"""
        ttl = ttl or self._default_ttl
        self._cache[key] = CacheEntry(
            value=value,
            expires_at=time.time() + ttl
        )

    def invalidate(self, key: str) -> None:
        """Remove key from cache"""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries, return count removed"""
        now = time.time()
        expired = [k for k, v in self._cache.items() if now >= v.expires_at]
        for key in expired:
            del self._cache[key]
        return len(expired)

    @staticmethod
    def make_key(url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and params"""
        key_data = url
        if params:
            key_data += str(sorted(params.items()))
        return hashlib.md5(key_data.encode()).hexdigest()


# =============================================================================
# RATE LIMITER
# =============================================================================

class TokenBucketRateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate  # Tokens per second
        self.burst = burst  # Max tokens
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until tokens are available"""
        async with self._lock:
            while True:
                now = time.time()
                elapsed = now - self._last_update
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_update = now

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Wait for tokens to replenish
                wait_time = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait_time)


# =============================================================================
# API CLIENT
# =============================================================================

class RobustAPIClient:
    """
    Production-grade API client with retry, circuit breaker, and caching.

    Usage:
        client = RobustAPIClient()

        # Sync request
        response = client.get("https://api.example.com/data")

        # Async request
        response = await client.async_get("https://api.example.com/data")

        # With caching
        response = client.get("https://api.example.com/data", cache=True)
    """

    def __init__(self, config: Optional[APIClientConfig] = None):
        self.config = config or APIClientConfig()

        # Circuit breakers per host
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Response cache
        self._cache = ResponseCache(self.config.cache_ttl_seconds)

        # Rate limiter
        self._rate_limiter = TokenBucketRateLimiter(
            rate=self.config.rate_limit_per_second
        )

        # Sync session with retry
        self._sync_session = self._create_sync_session()

        # Async session (created lazily)
        self._async_session: Optional[aiohttp.ClientSession] = None

        # Stats
        self._stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "cache_hits": 0,
            "circuit_breaks": 0
        }

    def _create_sync_session(self) -> requests.Session:
        """Create requests session with retry adapter"""
        session = requests.Session()

        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff_factor,
            status_forcelist=self.config.retry_statuses,
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS"],
            raise_on_status=False
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async session"""
        if self._async_session is None or self._async_session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=self.config.connect_timeout,
                total=self.config.read_timeout
            )
            self._async_session = aiohttp.ClientSession(timeout=timeout)
        return self._async_session

    def _get_circuit_breaker(self, url: str) -> CircuitBreaker:
        """Get circuit breaker for URL host"""
        from urllib.parse import urlparse
        host = urlparse(url).netloc

        if host not in self._circuit_breakers:
            self._circuit_breakers[host] = CircuitBreaker(
                failure_threshold=self.config.circuit_failure_threshold,
                recovery_timeout=self.config.circuit_recovery_timeout
            )
        return self._circuit_breakers[host]

    # -------------------------------------------------------------------------
    # SYNC METHODS
    # -------------------------------------------------------------------------

    def get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache: bool = False,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Sync GET request with retry and optional caching"""
        return self._sync_request(
            "GET", url, params=params, headers=headers,
            cache=cache, cache_ttl=cache_ttl, **kwargs
        )

    def post(
        self,
        url: str,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Sync POST request with retry"""
        return self._sync_request(
            "POST", url, data=data, json=json, headers=headers, **kwargs
        )

    def _sync_request(
        self,
        method: str,
        url: str,
        cache: bool = False,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute sync HTTP request with all protections"""
        self._stats["requests"] += 1

        # Check cache
        cache_key = None
        if cache and method == "GET" and self.config.cache_enabled:
            cache_key = ResponseCache.make_key(url, kwargs.get("params"))
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        # Check circuit breaker
        circuit = self._get_circuit_breaker(url)
        if not circuit.can_execute():
            self._stats["circuit_breaks"] += 1
            raise CircuitBreakerOpen(f"Circuit breaker open for {url}")

        # Execute request
        try:
            response = self._sync_session.request(
                method,
                url,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
                **kwargs
            )

            response.raise_for_status()
            result = response.json()

            # Record success
            circuit.record_success()
            self._stats["successes"] += 1

            # Cache result
            if cache_key:
                self._cache.set(cache_key, result, cache_ttl)

            return result

        except requests.exceptions.RequestException as e:
            circuit.record_failure()
            self._stats["failures"] += 1
            logger.error(f"API request failed: {method} {url} - {e}")
            raise APIRequestError(f"Request failed: {e}") from e

    # -------------------------------------------------------------------------
    # ASYNC METHODS
    # -------------------------------------------------------------------------

    async def async_get(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache: bool = False,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async GET request with retry and optional caching"""
        return await self._async_request(
            "GET", url, params=params, headers=headers,
            cache=cache, cache_ttl=cache_ttl, **kwargs
        )

    async def async_post(
        self,
        url: str,
        data: Optional[Dict] = None,
        json: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async POST request with retry"""
        return await self._async_request(
            "POST", url, data=data, json=json, headers=headers, **kwargs
        )

    async def _async_request(
        self,
        method: str,
        url: str,
        cache: bool = False,
        cache_ttl: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute async HTTP request with all protections"""
        self._stats["requests"] += 1

        # Rate limiting
        await self._rate_limiter.acquire()

        # Check cache
        cache_key = None
        if cache and method == "GET" and self.config.cache_enabled:
            cache_key = ResponseCache.make_key(url, kwargs.get("params"))
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return cached

        # Check circuit breaker
        circuit = self._get_circuit_breaker(url)
        if not circuit.can_execute():
            self._stats["circuit_breaks"] += 1
            raise CircuitBreakerOpen(f"Circuit breaker open for {url}")

        # Execute with retry
        session = await self._get_async_session()
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                async with session.request(method, url, **kwargs) as response:
                    if response.status in self.config.retry_statuses:
                        if attempt < self.config.max_retries:
                            wait = self.config.retry_backoff_factor * (2 ** attempt)
                            self._stats["retries"] += 1
                            logger.warning(
                                f"Retry {attempt + 1}/{self.config.max_retries} "
                                f"for {url} (status {response.status}), waiting {wait}s"
                            )
                            await asyncio.sleep(wait)
                            continue
                        else:
                            response.raise_for_status()

                    response.raise_for_status()
                    result = await response.json()

                    # Record success
                    circuit.record_success()
                    self._stats["successes"] += 1

                    # Cache result
                    if cache_key:
                        self._cache.set(cache_key, result, cache_ttl)

                    return result

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    wait = self.config.retry_backoff_factor * (2 ** attempt)
                    self._stats["retries"] += 1
                    logger.warning(
                        f"Retry {attempt + 1}/{self.config.max_retries} "
                        f"for {url} ({e}), waiting {wait}s"
                    )
                    await asyncio.sleep(wait)
                    continue

        # All retries failed
        circuit.record_failure()
        self._stats["failures"] += 1
        logger.error(f"API request failed after retries: {method} {url} - {last_error}")
        raise APIRequestError(f"Request failed after {self.config.max_retries} retries: {last_error}")

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    async def close(self) -> None:
        """Close async session"""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self._stats,
            "cache_size": len(self._cache._cache),
            "circuit_breakers": {
                host: cb.state.value
                for host, cb in self._circuit_breakers.items()
            }
        }

    def reset_stats(self) -> None:
        """Reset statistics"""
        self._stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "retries": 0,
            "cache_hits": 0,
            "circuit_breaks": 0
        }


# =============================================================================
# EXCEPTIONS
# =============================================================================

class APIClientError(Exception):
    """Base API client error"""
    pass


class APIRequestError(APIClientError):
    """Request failed"""
    pass


class CircuitBreakerOpen(APIClientError):
    """Circuit breaker is open"""
    pass


class RateLimitExceeded(APIClientError):
    """Rate limit exceeded"""
    pass


# =============================================================================
# SINGLETON
# =============================================================================

_api_client: Optional[RobustAPIClient] = None


def get_api_client() -> RobustAPIClient:
    """Get singleton API client instance"""
    global _api_client
    if _api_client is None:
        _api_client = RobustAPIClient()
    return _api_client


# =============================================================================
# DECORATOR
# =============================================================================

def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for adding retry logic to any function.

    Usage:
        @with_retry(max_retries=3, exceptions=(ValueError, ConnectionError))
        def my_function():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                        )
                        time.sleep(wait)
            raise last_error

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                        )
                        await asyncio.sleep(wait)
            raise last_error

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Robust API Client ===\n")

    async def test_client():
        client = RobustAPIClient()

        try:
            # Test sync GET with caching
            print("1. Testing sync GET with caching...")
            result1 = client.get(
                "https://httpbin.org/get",
                params={"test": "value"},
                cache=True
            )
            print(f"   ✅ First request: {result1.get('args', {})}")

            result2 = client.get(
                "https://httpbin.org/get",
                params={"test": "value"},
                cache=True
            )
            print(f"   ✅ Cached request (cache hits: {client._stats['cache_hits']})")

            # Test async GET
            print("\n2. Testing async GET...")
            result3 = await client.async_get(
                "https://httpbin.org/get",
                params={"async": "test"}
            )
            print(f"   ✅ Async request: {result3.get('args', {})}")

            # Print stats
            print("\n3. Client Statistics:")
            stats = client.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

        finally:
            await client.close()

        print("\n✅ API Client tests complete!")

    asyncio.run(test_client())
