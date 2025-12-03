"""
Rate Limiting Infrastructure

Provides:
- Per-endpoint rate limiting
- Per-user rate limiting
- API quota management for Robinhood
- Sliding window algorithm
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Callable
from dataclasses import dataclass
from functools import wraps
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int  # Number of requests allowed
    window: int    # Time window in seconds

    def __str__(self):
        return f"{self.requests}/{self.window}s"


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    def __init__(self, retry_after: float, limit: str):
        self.retry_after = retry_after
        self.limit = limit
        super().__init__(f"Rate limit exceeded: {limit}. Retry after {retry_after:.1f}s")


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    More accurate than fixed window, prevents burst at window boundaries.

    Usage:
        limiter = SlidingWindowRateLimiter()

        # Check and consume
        if await limiter.check("user:123", RateLimitConfig(10, 60)):
            await limiter.consume("user:123", RateLimitConfig(10, 60))
            # Process request
        else:
            # Rate limited
    """

    def __init__(self):
        # key -> list of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check(self, key: str, config: RateLimitConfig) -> bool:
        """Check if request is allowed (doesn't consume)"""
        async with self._lock:
            return self._is_allowed(key, config)

    async def consume(self, key: str, config: RateLimitConfig) -> bool:
        """Consume one request from quota"""
        async with self._lock:
            if self._is_allowed(key, config):
                self._requests[key].append(time.time())
                return True
            return False

    async def check_and_consume(self, key: str, config: RateLimitConfig) -> tuple[bool, float]:
        """
        Check and consume in one operation.

        Returns:
            (allowed, retry_after) - retry_after is 0 if allowed
        """
        async with self._lock:
            self._cleanup_old_requests(key, config.window)

            current_count = len(self._requests[key])

            if current_count < config.requests:
                self._requests[key].append(time.time())
                return (True, 0.0)

            # Calculate retry after
            oldest = self._requests[key][0] if self._requests[key] else time.time()
            retry_after = oldest + config.window - time.time()
            return (False, max(0.0, retry_after))

    def _is_allowed(self, key: str, config: RateLimitConfig) -> bool:
        """Internal check without lock"""
        self._cleanup_old_requests(key, config.window)
        return len(self._requests[key]) < config.requests

    def _cleanup_old_requests(self, key: str, window: int):
        """Remove requests outside the window"""
        cutoff = time.time() - window
        self._requests[key] = [
            ts for ts in self._requests[key]
            if ts > cutoff
        ]

    async def get_remaining(self, key: str, config: RateLimitConfig) -> int:
        """Get remaining requests in current window"""
        async with self._lock:
            self._cleanup_old_requests(key, config.window)
            return max(0, config.requests - len(self._requests[key]))

    async def reset(self, key: str):
        """Reset rate limit for a key"""
        async with self._lock:
            self._requests.pop(key, None)


class APIQuotaManager:
    """
    Manages API quota for external services like Robinhood.

    Robinhood limits:
    - ~100 requests/hour for basic accounts
    - ~300 requests/hour for Gold accounts

    Usage:
        quota = APIQuotaManager("robinhood", hourly_limit=100)

        if await quota.can_make_request(10):  # Need 10 API calls
            await quota.record_usage(10)
            # Make API calls
        else:
            # Use cached data
    """

    def __init__(
        self,
        name: str,
        hourly_limit: int = 100,
        reserve_pct: float = 0.1  # Keep 10% for emergencies
    ):
        self.name = name
        self.hourly_limit = hourly_limit
        self.reserve = int(hourly_limit * reserve_pct)
        self._usage: list[float] = []
        self._lock = asyncio.Lock()

    async def can_make_request(self, calls_needed: int = 1) -> bool:
        """Check if we have quota for the needed API calls"""
        async with self._lock:
            self._cleanup_old_usage()
            available = self.hourly_limit - len(self._usage) - self.reserve
            return calls_needed <= available

    async def record_usage(self, calls: int = 1):
        """Record API call usage"""
        async with self._lock:
            now = time.time()
            for _ in range(calls):
                self._usage.append(now)

    def _cleanup_old_usage(self):
        """Remove usage older than 1 hour"""
        cutoff = time.time() - 3600
        self._usage = [ts for ts in self._usage if ts > cutoff]

    async def get_status(self) -> dict:
        """Get current quota status"""
        async with self._lock:
            self._cleanup_old_usage()
            used = len(self._usage)
            available = self.hourly_limit - used

            return {
                "name": self.name,
                "hourly_limit": self.hourly_limit,
                "used_this_hour": used,
                "available": available,
                "reserve": self.reserve,
                "pct_used": round(used / self.hourly_limit * 100, 1)
            }

    async def reset(self):
        """Reset quota (use carefully)"""
        async with self._lock:
            self._usage.clear()


# =============================================================================
# Rate Limiting Decorator
# =============================================================================

_global_limiter = SlidingWindowRateLimiter()


def rate_limited(
    requests: int,
    window: int,
    key_func: Optional[Callable] = None
):
    """
    Decorator for rate limiting endpoints.

    Usage:
        @rate_limited(10, 60)  # 10 requests per minute
        async def get_positions():
            ...

        @rate_limited(5, 60, key_func=lambda req: req.client.host)
        async def get_analysis(request: Request):
            ...
    """
    config = RateLimitConfig(requests, window)

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build rate limit key
            if key_func:
                key = f"{func.__name__}:{key_func(*args, **kwargs)}"
            else:
                key = func.__name__

            allowed, retry_after = await _global_limiter.check_and_consume(key, config)

            if not allowed:
                raise RateLimitExceeded(retry_after, str(config))

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Pre-configured Rate Limiters
# =============================================================================

# Robinhood API quota manager
robinhood_quota = APIQuotaManager("robinhood", hourly_limit=100)

# Per-endpoint rate limits
ENDPOINT_LIMITS = {
    "positions": RateLimitConfig(20, 60),      # 20/min
    "deep_analysis": RateLimitConfig(5, 60),   # 5/min (expensive)
    "portfolio_analysis": RateLimitConfig(3, 60),  # 3/min (very expensive)
    "metadata": RateLimitConfig(30, 60),       # 30/min
    "default": RateLimitConfig(60, 60)         # 60/min default
}


def get_endpoint_limit(endpoint: str) -> RateLimitConfig:
    """Get rate limit config for an endpoint"""
    return ENDPOINT_LIMITS.get(endpoint, ENDPOINT_LIMITS["default"])
