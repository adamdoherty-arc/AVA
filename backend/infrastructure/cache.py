"""
Distributed Caching Infrastructure

Provides:
- Redis-based distributed cache with TTL
- In-memory fallback for development
- Cache invalidation patterns
- Automatic serialization/deserialization
"""

import os
import json
import logging
import asyncio
from typing import Any, Awaitable, Optional, Dict, Callable, TypeVar
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

T = TypeVar('T')


class KeyedLockManager:
    """
    Manages per-key locks to prevent cache stampede.

    Uses lazy initialization for asyncio.Lock to avoid
    issues when instantiated before event loop exists.
    """

    def __init__(self) -> None:
        self._locks: Dict[str, asyncio.Lock] = {}
        self._meta_lock: Optional[asyncio.Lock] = None  # Lazy init

    def _get_meta_lock(self) -> asyncio.Lock:
        """Lazily create the meta lock when first needed (in async context)."""
        if self._meta_lock is None:
            self._meta_lock = asyncio.Lock()
        return self._meta_lock

    async def acquire(self, key: str) -> asyncio.Lock:
        """Get or create a lock for a key"""
        meta_lock = self._get_meta_lock()
        async with meta_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
            lock = self._locks[key]

        await lock.acquire()
        return lock

    def release(self, key: str, lock: asyncio.Lock) -> None:
        """Release a lock"""
        lock.release()

    async def cleanup_unused(self) -> int:
        """Remove locks that aren't currently held"""
        async with self._meta_lock:
            to_remove = [k for k, v in self._locks.items() if not v.locked()]
            for key in to_remove:
                del self._locks[key]
            return len(to_remove)


@dataclass
class CacheStats:
    """Track cache performance metrics"""
    hits: int = 0
    misses: int = 0
    errors: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class InMemoryCache:
    """
    In-memory cache with TTL support and LRU eviction.
    Used as fallback when Redis is unavailable.

    Features:
    - Max size limit to prevent unbounded memory growth
    - LRU eviction when limit is reached
    - TTL-based expiration
    """

    def __init__(self, default_ttl: int = 300, max_size: int = 10000):
        self._cache: Dict[str, tuple[Any, datetime, float]] = {}  # value, expires_at, last_access
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        async with self._lock:
            if key in self._cache:
                value, expires_at, _ = self._cache[key]
                now = datetime.now()
                if now < expires_at:
                    # Update last access time for LRU
                    self._cache[key] = (value, expires_at, now.timestamp())
                    self.stats.hits += 1
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    self.stats.evictions += 1

            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        async with self._lock:
            # Evict if at max size (before adding new entry)
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_lru_unlocked()

            ttl = ttl or self._default_ttl
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl)
            self._cache[key] = (value, expires_at, now.timestamp())
            return True

    def _evict_lru_unlocked(self) -> None:
        """Evict least recently used entries (must hold lock)"""
        if not self._cache:
            return

        # Remove expired entries first
        now = datetime.now()
        expired_keys = [k for k, (_, exp, _) in self._cache.items() if now >= exp]
        for key in expired_keys:
            del self._cache[key]
            self.stats.evictions += 1

        # If still at max, evict LRU entries
        while len(self._cache) >= self._max_size:
            # Find least recently used
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k][2])
            del self._cache[lru_key]
            self.stats.evictions += 1

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern (simple prefix match)"""
        async with self._lock:
            keys_to_delete = [
                k for k in self._cache.keys()
                if k.startswith(pattern.replace('*', ''))
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    async def clear(self) -> bool:
        """Clear entire cache"""
        async with self._lock:
            self._cache.clear()
            return True

    async def health_check(self) -> bool:
        """Check if cache is healthy"""
        return True

    @property
    def size(self) -> int:
        """Current number of entries in cache"""
        return len(self._cache)

    @property
    def max_size(self) -> int:
        """Maximum cache size"""
        return self._max_size


class RedisCache:
    """
    Redis-based distributed cache with automatic fallback.

    Features:
    - Redis primary with in-memory fallback
    - Cache stampede protection via per-key locking
    - TTL-based expiration
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 300,
        prefix: str = "ava:"
    ):
        self._redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
        self._default_ttl = default_ttl
        self._prefix = prefix
        self._redis = None
        self._connected = False
        self._fallback = InMemoryCache(default_ttl)
        self.stats = CacheStats()
        self._lock_manager = KeyedLockManager()  # Stampede protection

    async def connect(self) -> bool:
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self._redis = await redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._redis_url}")
            return True
        except ImportError:
            logger.warning("redis package not installed, using in-memory cache")
            return False
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}, using in-memory fallback")
            return False

    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        full_key = self._make_key(key)

        if self._connected and self._redis:
            try:
                value = await self._redis.get(full_key)
                if value:
                    self.stats.hits += 1
                    return json.loads(value)
                self.stats.misses += 1
                return None
            except Exception as e:
                logger.error(f"Redis get error: {e}")
                self.stats.errors += 1
                # Fallback to in-memory
                return await self._fallback.get(key)

        return await self._fallback.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL"""
        full_key = self._make_key(key)
        ttl = ttl or self._default_ttl

        if self._connected and self._redis:
            try:
                serialized = json.dumps(value, default=str)
                await self._redis.setex(full_key, ttl, serialized)
                return True
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                self.stats.errors += 1
                return await self._fallback.set(key, value, ttl)

        return await self._fallback.set(key, value, ttl)

    async def get_or_fetch(
        self,
        key: str,
        fetch_func: Callable[[], Awaitable[T]],
        ttl: Optional[int] = None
    ) -> T:
        """
        Get from cache or fetch with stampede protection.

        Prevents cache stampede by using per-key locking. When multiple
        concurrent requests hit a cache miss, only ONE request calls
        fetch_func while others wait for the result.

        Args:
            key: Cache key
            fetch_func: Async function to call on cache miss
            ttl: Optional TTL override

        Returns:
            Cached or freshly fetched value
        """
        # Fast path: check cache without lock
        cached = await self.get(key)
        if cached is not None:
            return cached

        # Slow path: acquire per-key lock to prevent stampede
        lock = await self._lock_manager.acquire(key)
        try:
            # Double-check cache after acquiring lock
            cached = await self.get(key)
            if cached is not None:
                return cached

            # We are the single fetcher - call the function
            result = await fetch_func()

            # Cache the result
            await self.set(key, result, ttl)

            return result
        finally:
            self._lock_manager.release(key, lock)

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        full_key = self._make_key(key)

        if self._connected and self._redis:
            try:
                await self._redis.delete(full_key)
                return True
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
                return await self._fallback.delete(key)

        return await self._fallback.delete(key)

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        full_pattern = self._make_key(pattern)

        if self._connected and self._redis:
            try:
                cursor = 0
                deleted = 0
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=full_pattern)
                    if keys:
                        await self._redis.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                return deleted
            except Exception as e:
                logger.error(f"Redis invalidate error: {e}")
                return await self._fallback.invalidate_pattern(pattern)

        return await self._fallback.invalidate_pattern(pattern)

    async def clear(self) -> bool:
        """Clear all cache with our prefix"""
        return await self.invalidate_pattern("*") >= 0

    async def health_check(self) -> bool:
        """Check if cache is healthy"""
        if self._connected and self._redis:
            try:
                await self._redis.ping()
                return True
            except Exception:
                return False
        return await self._fallback.health_check()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "connected": self._connected,
            "backend": "redis" if self._connected else "in-memory",
            **asdict(self.stats),
            "hit_rate": self.stats.hit_rate,
        }
        # Add fallback cache info
        if not self._connected:
            stats["fallback_size"] = self._fallback.size
            stats["fallback_max_size"] = self._fallback.max_size
            stats["fallback_hit_rate"] = self._fallback.stats.hit_rate
        return stats


# =============================================================================
# Caching Decorators
# =============================================================================

def cached(
    ttl: int = 300,
    key_builder: Optional[Callable[..., str]] = None,
    cache_none: bool = False
):
    """
    Decorator for caching async function results.

    Usage:
        @cached(ttl=60)
        async def get_positions(user_id: str):
            ...

        @cached(ttl=300, key_builder=lambda symbol: f"metadata:{symbol}")
        async def get_metadata(symbol: str):
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key: function name + hash of args
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Try to get from cache
            cache = get_cache()
            cached_value = await cache.get(cache_key)
            if cached_value is not None or (cached_value is None and cache_none):
                return cached_value

            # Call function and cache result
            result = await func(*args, **kwargs)

            if result is not None or cache_none:
                await cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


# =============================================================================
# Singleton Cache Instance
# =============================================================================

import threading

_cache_instance: Optional[RedisCache] = None
_cache_lock = threading.Lock()
_cache_connected = False  # Track connection state separately


async def init_cache() -> RedisCache:
    """
    Initialize and connect the global cache instance.

    This should be called during app startup (in lifespan).
    Thread-safe with proper async connection.
    """
    global _cache_instance, _cache_connected

    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = RedisCache()

    # Connect outside the lock (async operation)
    if not _cache_connected:
        try:
            await _cache_instance.connect()
            _cache_connected = True
            logger.info("Cache initialized and connected")
        except Exception as e:
            logger.warning(f"Cache connection failed, using in-memory fallback: {e}")
            _cache_connected = True  # Mark as "initialized" even if Redis failed

    return _cache_instance


def get_cache() -> RedisCache:
    """
    Get the global cache instance (thread-safe).

    NOTE: Returns cache that may not be connected yet if called before
    init_cache(). The cache will gracefully fall back to in-memory storage.
    For proper initialization, call init_cache() during app startup.
    """
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            # Double-check pattern for thread safety
            if _cache_instance is None:
                _cache_instance = RedisCache()
                logger.info("Cache instance created (connection deferred to init_cache)")
    return _cache_instance


def is_cache_connected() -> bool:
    """Check if cache has been properly initialized."""
    return _cache_connected
