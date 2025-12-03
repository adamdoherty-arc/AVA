"""
AVA Caching Layer
=================

Multi-tier caching with:
- In-memory LRU cache for hot data
- Redis cache for distributed/persistent caching
- Automatic TTL management
- Cache invalidation patterns

Author: AVA Trading Platform
Created: 2025-11-28
"""

import asyncio
import json
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, TypeVar, Callable, Generic
from functools import wraps
from collections import OrderedDict
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# CACHE ENTRY
# =============================================================================

@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with TTL tracking"""
    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self):
        """Update access time and count"""
        self.last_accessed = datetime.now()
        self.access_count += 1


# =============================================================================
# IN-MEMORY LRU CACHE
# =============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Usage:
        cache = LRUCache(max_size=1000, default_ttl=300)
        cache.set("key", value, ttl=60)
        result = cache.get("key")
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: int = 300,
        cleanup_interval: int = 60
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }

        # Start cleanup task
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task] = None

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1

            return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl > 0 else None

        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at
            )
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None
    ) -> T:
        """Get value or compute and cache it"""
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        value = factory()
        self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Async version of get_or_set"""
        value = self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        self.set(key, value, ttl)
        return value

    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats["expirations"] += 1

    @property
    def stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total if total > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self._stats["evictions"],
                "expirations": self._stats["expirations"]
            }


# =============================================================================
# REDIS CACHE
# =============================================================================

class RedisCache:
    """
    Redis-backed cache with async support.

    Usage:
        cache = RedisCache(url="redis://localhost:6379/0")
        await cache.connect()
        await cache.set("key", value, ttl=60)
        result = await cache.get("key")
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        default_ttl: int = 300,
        prefix: str = "ava:",
        pool_size: int = 10
    ):
        self.url = url
        self.default_ttl = default_ttl
        self.prefix = prefix
        self.pool_size = pool_size
        self._redis = None
        self._connected = False

    async def connect(self):
        """Connect to Redis"""
        try:
            import redis.asyncio as redis
            self._redis = await redis.from_url(
                self.url,
                encoding="utf-8",
                decode_responses=False,  # We handle serialization
                max_connections=self.pool_size
            )
            self._connected = True
            logger.info(f"Connected to Redis: {self.url}")
        except ImportError:
            logger.warning("redis package not installed, Redis cache disabled")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self._connected = False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            await self._redis.close()
            self._connected = False

    def _make_key(self, key: str) -> str:
        """Create prefixed key"""
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage"""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        return pickle.loads(data)

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self._connected:
            return None

        try:
            data = await self._redis.get(self._make_key(key))
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ):
        """Set value in Redis"""
        if not self._connected:
            return

        ttl = ttl or self.default_ttl

        try:
            data = self._serialize(value)
            await self._redis.setex(
                self._make_key(key),
                ttl,
                data
            )
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self._connected:
            return False

        try:
            result = await self._redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear_pattern(self, pattern: str):
        """Clear keys matching pattern"""
        if not self._connected:
            return

        try:
            full_pattern = self._make_key(pattern)
            async for key in self._redis.scan_iter(match=full_pattern):
                await self._redis.delete(key)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None
    ) -> Any:
        """Get value or compute and cache it"""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value


# =============================================================================
# MULTI-TIER CACHE
# =============================================================================

class TieredCache:
    """
    Multi-tier cache combining in-memory and Redis.

    - L1: In-memory LRU cache (fastest, limited size)
    - L2: Redis cache (persistent, distributed)

    Usage:
        cache = TieredCache()
        await cache.initialize()
        await cache.set("key", value, ttl=60)
        result = await cache.get("key")
    """

    def __init__(
        self,
        l1_max_size: int = 1000,
        l1_default_ttl: int = 60,
        redis_url: Optional[str] = None,
        redis_default_ttl: int = 300,
        prefix: str = "ava:"
    ):
        self.l1 = LRUCache(
            max_size=l1_max_size,
            default_ttl=l1_default_ttl
        )
        self.l2: Optional[RedisCache] = None

        if redis_url:
            self.l2 = RedisCache(
                url=redis_url,
                default_ttl=redis_default_ttl,
                prefix=prefix
            )

    async def initialize(self):
        """Initialize cache connections"""
        if self.l2:
            await self.l2.connect()

    async def shutdown(self):
        """Shutdown cache connections"""
        if self.l2:
            await self.l2.disconnect()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 first, then L2)"""
        # Try L1
        value = self.l1.get(key)
        if value is not None:
            return value

        # Try L2
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                # Promote to L1
                self.l1.set(key, value)
                return value

        return None

    async def set(
        self,
        key: str,
        value: Any,
        l1_ttl: Optional[int] = None,
        l2_ttl: Optional[int] = None
    ):
        """Set value in both cache tiers"""
        self.l1.set(key, value, l1_ttl)

        if self.l2:
            await self.l2.set(key, value, l2_ttl)

    async def delete(self, key: str):
        """Delete from both tiers"""
        self.l1.delete(key)

        if self.l2:
            await self.l2.delete(key)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        l1_ttl: Optional[int] = None,
        l2_ttl: Optional[int] = None
    ) -> Any:
        """Get or compute value"""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, l1_ttl, l2_ttl)
        return value


# =============================================================================
# CACHE DECORATORS
# =============================================================================

def cached(
    ttl: int = 300,
    key_prefix: str = "",
    cache_instance: Optional[LRUCache] = None
):
    """
    Decorator for caching function results.

    Usage:
        @cached(ttl=60, key_prefix="prices")
        def get_price(symbol: str) -> float:
            return fetch_price(symbol)
    """
    _cache = cache_instance or LRUCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


def async_cached(
    ttl: int = 300,
    key_prefix: str = "",
    cache_instance: Optional[LRUCache] = None
):
    """
    Decorator for caching async function results.

    Usage:
        @async_cached(ttl=60, key_prefix="chains")
        async def get_chain(symbol: str) -> Dict:
            return await fetch_chain(symbol)
    """
    _cache = cache_instance or LRUCache()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = ":".join(key_parts)

            # Check cache
            result = _cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = await func(*args, **kwargs)
            _cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


# =============================================================================
# SPECIALIZED CACHES
# =============================================================================

class OptionChainCache(TieredCache):
    """Specialized cache for option chains"""

    def __init__(self, redis_url: Optional[str] = None):
        super().__init__(
            l1_max_size=100,
            l1_default_ttl=30,  # 30 seconds for hot data
            redis_url=redis_url,
            redis_default_ttl=60,  # 1 minute for Redis
            prefix="ava:chains:"
        )

    def _make_key(self, symbol: str, expiration: Optional[str] = None) -> str:
        key = symbol.upper()
        if expiration:
            key += f":{expiration}"
        return key

    async def get_chain(
        self,
        symbol: str,
        expiration: Optional[str] = None
    ) -> Optional[Dict]:
        key = self._make_key(symbol, expiration)
        return await self.get(key)

    async def set_chain(
        self,
        symbol: str,
        chain: Dict,
        expiration: Optional[str] = None
    ):
        key = self._make_key(symbol, expiration)
        await self.set(key, chain)


class GreeksCache(LRUCache):
    """Specialized cache for Greeks calculations"""

    def __init__(self):
        super().__init__(
            max_size=5000,
            default_ttl=5  # Greeks change rapidly
        )

    def _make_key(
        self,
        underlying_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str
    ) -> str:
        # Round to reduce cache misses from tiny price changes
        return f"{underlying_price:.2f}:{strike:.2f}:{dte}:{iv:.3f}:{option_type}"

    def get_greeks(
        self,
        underlying_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str
    ) -> Optional[Dict]:
        key = self._make_key(underlying_price, strike, dte, iv, option_type)
        return self.get(key)

    def set_greeks(
        self,
        underlying_price: float,
        strike: float,
        dte: int,
        iv: float,
        option_type: str,
        greeks: Dict
    ):
        key = self._make_key(underlying_price, strike, dte, iv, option_type)
        self.set(key, greeks)


# =============================================================================
# GLOBAL CACHE INSTANCES
# =============================================================================

# Singleton caches
_option_chain_cache: Optional[OptionChainCache] = None
_greeks_cache: Optional[GreeksCache] = None
_general_cache: Optional[TieredCache] = None


def get_option_chain_cache() -> OptionChainCache:
    """Get singleton option chain cache"""
    global _option_chain_cache
    if _option_chain_cache is None:
        _option_chain_cache = OptionChainCache()
    return _option_chain_cache


def get_greeks_cache() -> GreeksCache:
    """Get singleton Greeks cache"""
    global _greeks_cache
    if _greeks_cache is None:
        _greeks_cache = GreeksCache()
    return _greeks_cache


def get_general_cache() -> TieredCache:
    """Get singleton general cache"""
    global _general_cache
    if _general_cache is None:
        _general_cache = TieredCache()
    return _general_cache


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Cache Layer ===\n")

    async def test_caches():
        # Test LRU Cache
        print("1. Testing LRU Cache...")
        lru = LRUCache(max_size=5, default_ttl=10)

        lru.set("a", 1)
        lru.set("b", 2)
        lru.set("c", 3)

        assert lru.get("a") == 1
        assert lru.get("b") == 2
        print(f"   Stats: {lru.stats}")

        # Test eviction
        lru.set("d", 4)
        lru.set("e", 5)
        lru.set("f", 6)  # Should evict 'c' (least recently used after a, b)

        print(f"   After eviction: {lru.stats}")

        # Test tiered cache
        print("\n2. Testing Tiered Cache...")
        tiered = TieredCache(l1_max_size=10)
        await tiered.initialize()

        await tiered.set("key1", {"data": "test"})
        result = await tiered.get("key1")
        assert result == {"data": "test"}
        print(f"   Retrieved: {result}")

        # Test get_or_set
        call_count = 0

        def expensive_compute():
            nonlocal call_count
            call_count += 1
            return {"computed": True}

        result1 = await tiered.get_or_set("computed", expensive_compute)
        result2 = await tiered.get_or_set("computed", expensive_compute)

        assert call_count == 1  # Should only compute once
        print(f"   Compute calls: {call_count} (expected 1)")

        # Test Greeks cache
        print("\n3. Testing Greeks Cache...")
        greeks_cache = GreeksCache()

        greeks_data = {
            "delta": 0.45,
            "gamma": 0.02,
            "theta": -0.05,
            "vega": 0.15
        }

        greeks_cache.set_greeks(100.0, 100.0, 30, 0.25, "call", greeks_data)
        retrieved = greeks_cache.get_greeks(100.0, 100.0, 30, 0.25, "call")
        assert retrieved == greeks_data
        print(f"   Greeks retrieved: {retrieved}")

        await tiered.shutdown()
        print("\n✅ Cache tests passed!")

    asyncio.run(test_caches())
