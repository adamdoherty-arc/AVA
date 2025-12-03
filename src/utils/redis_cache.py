"""
Redis Caching Layer (Optional)
For distributed caching across multiple server instances
"""
import os
import pickle
import logging
from typing import Any, Callable, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Check if Redis is available
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not installed. Distributed caching disabled.")

class RedisCache:
    """Redis cache wrapper with fallback"""

    def __init__(self) -> None:
        self.client = None
        self.enabled = False

        if REDIS_AVAILABLE and os.getenv('REDIS_ENABLED', 'false').lower() == 'true':
            try:
                self.client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=int(os.getenv('REDIS_DB', 0)),
                    decode_responses=False,
                    socket_timeout=2
                )
                # Test connection
                self.client.ping()
                self.enabled = True
                logger.info("✅ Redis cache enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using local cache only.")
                self.enabled = False

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.enabled:
            return None

        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")

        return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in Redis with TTL"""
        if not self.enabled:
            return False

        try:
            self.client.setex(
                key,
                ttl,
                pickle.dumps(value)
            )
            return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
            return False

    def delete(self, key: str):
        """Delete key from Redis"""
        if not self.enabled:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern (e.g., 'market_data:*')"""
        if not self.enabled:
            return False

        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.warning(f"Redis clear pattern error: {e}")
            return False

    def get_stats(self) -> dict:
        """Get Redis cache statistics"""
        if not self.enabled:
            return {
                "enabled": False,
                "status": "Redis not available"
            }

        try:
            info = self.client.info()
            return {
                "enabled": True,
                "keys": self.client.dbsize(),
                "memory_used": info.get('used_memory_human', 'N/A'),
                "uptime_seconds": info.get('uptime_in_seconds', 0),
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            return {
                "enabled": False,
                "error": str(e)
            }

# Global Redis cache instance
_redis_cache = RedisCache()

def cache_with_redis(key_prefix: str, ttl: int = 300):
    """
    Decorator for Redis-backed caching with Streamlit fallback.

    Args:
        key_prefix: Prefix for cache key
        ttl: Time to live in seconds

    Example:
        @cache_with_redis("market_data", ttl=60)
        def get_market_data(symbol):
            return api.fetch(symbol)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            try:
                cache_key = f"{key_prefix}:{func.__name__}:{hash(pickle.dumps((args, kwargs)))}"
            except:
                # Fallback if hashing fails
                cache_key = f"{key_prefix}:{func.__name__}"

            # Try Redis first
            if _redis_cache.enabled:
                cached = _redis_cache.get(cache_key)
                if cached is not None:
                    logger.debug(f"Redis cache hit: {func.__name__}")
                    return cached

            # Compute value
            result = func(*args, **kwargs)

            # Cache in Redis if enabled
            if _redis_cache.enabled and result is not None:
                _redis_cache.set(cache_key, result, ttl)
                logger.debug(f"Redis cache set: {func.__name__}")

            return result

        return wrapper
    return decorator

def get_redis_cache() -> RedisCache:
    """Get the global Redis cache instance"""
    return _redis_cache

def clear_redis_cache(pattern: str = "*"):
    """
    Clear Redis cache by pattern.

    Args:
        pattern: Pattern to match keys (default: "*" for all)
    """
    return _redis_cache.clear_pattern(pattern)

def get_cache_stats() -> dict:
    """Get Redis cache statistics"""
    return _redis_cache.get_stats()
