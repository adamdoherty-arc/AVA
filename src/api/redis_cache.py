"""
Redis Cache Manager
Provides async Redis caching with TTL support
"""

import redis.asyncio as redis
from typing import Optional, Any, Dict
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Async Redis cache manager with JSON serialization
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 1800,  # 30 minutes
        decode_responses: bool = True
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            default_ttl: Default TTL in seconds (30 min)
            decode_responses: Auto-decode responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.decode_responses = decode_responses
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish Redis connection"""
        if self._client is None:
            try:
                self._client = await redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=self.decode_responses
                )
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {str(e)}")
                raise

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    async def _get_client(self) -> redis.Redis:
        """Get Redis client, connecting if necessary"""
        if self._client is None:
            await self.connect()
        return self._client

    async def ping(self) -> bool:
        """
        Check if Redis is responsive

        Returns:
            True if Redis is healthy
        """
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value as dict, or None if not found
        """
        try:
            client = await self._get_client()
            value = await client.get(key)

            if value:
                # Parse JSON string to dict
                return json.loads(value)
            return None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON for key {key}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Failed to get key {key}: {str(e)}")
            return None

    async def set(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache with TTL

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable dict)
            ttl: Time-to-live in seconds (uses default_ttl if None)

        Returns:
            True if successful
        """
        try:
            client = await self._get_client()

            # Serialize dict to JSON string
            value_str = json.dumps(value, default=str)

            ttl = ttl or self.default_ttl

            await client.setex(
                name=key,
                time=ttl,
                value=value_str
            )

            logger.debug(f"Cached key {key} with TTL {ttl}s")
            return True

        except TypeError as e:
            logger.error(f"Failed to serialize value for key {key}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to set key {key}: {str(e)}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache

        Args:
            key: Cache key to delete

        Returns:
            True if key was deleted
        """
        try:
            client = await self._get_client()
            result = await client.delete(key)
            logger.debug(f"Deleted key {key}: {result}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete key {key}: {str(e)}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            client = await self._get_client()
            result = await client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to check existence of key {key}: {str(e)}")
            return False

    async def get_ttl(self, key: str) -> int:
        """
        Get remaining TTL for a key

        Args:
            key: Cache key

        Returns:
            Remaining seconds, -1 if no expiry, -2 if key doesn't exist
        """
        try:
            client = await self._get_client()
            ttl = await client.ttl(key)
            return ttl
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {str(e)}")
            return -2

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Update TTL for existing key

        Args:
            key: Cache key
            ttl: New TTL in seconds

        Returns:
            True if TTL was updated
        """
        try:
            client = await self._get_client()
            result = await client.expire(key, ttl)
            return result
        except Exception as e:
            logger.error(f"Failed to update TTL for key {key}: {str(e)}")
            return False

    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment integer value at key

        Args:
            key: Cache key
            amount: Amount to increment by

        Returns:
            New value after increment
        """
        try:
            client = await self._get_client()
            return await client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Failed to increment key {key}: {str(e)}")
            raise

    async def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement integer value at key

        Args:
            key: Cache key
            amount: Amount to decrement by

        Returns:
            New value after decrement
        """
        try:
            client = await self._get_client()
            return await client.decrby(key, amount)
        except Exception as e:
            logger.error(f"Failed to decrement key {key}: {str(e)}")
            raise

    async def get_many(self, *keys: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Get multiple keys at once

        Args:
            keys: Cache keys to retrieve

        Returns:
            Dict mapping keys to their values (or None)
        """
        try:
            client = await self._get_client()
            values = await client.mget(*keys)

            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = None
                else:
                    result[key] = None

            return result

        except Exception as e:
            logger.error(f"Failed to get multiple keys: {str(e)}")
            return {key: None for key in keys}

    async def set_many(
        self,
        mapping: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple keys at once

        Args:
            mapping: Dict of key-value pairs to cache
            ttl: TTL in seconds (applied to all keys)

        Returns:
            True if all keys were set successfully
        """
        try:
            client = await self._get_client()
            ttl = ttl or self.default_ttl

            # Use pipeline for atomic operations
            async with client.pipeline(transaction=True) as pipe:
                for key, value in mapping.items():
                    value_str = json.dumps(value, default=str)
                    pipe.setex(key, ttl, value_str)

                await pipe.execute()

            logger.debug(f"Cached {len(mapping)} keys with TTL {ttl}s")
            return True

        except Exception as e:
            logger.error(f"Failed to set multiple keys: {str(e)}")
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern

        Args:
            pattern: Redis key pattern (e.g., 'research:*')

        Returns:
            Number of keys deleted
        """
        try:
            client = await self._get_client()
            keys = []

            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await client.delete(*keys)
                logger.info(f"Deleted {deleted} keys matching pattern '{pattern}'")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to clear pattern '{pattern}': {str(e)}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get Redis cache statistics

        Returns:
            Dict with cache statistics
        """
        try:
            client = await self._get_client()
            info = await client.info()

            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {str(e)}")
            return {}


# Singleton instance
_redis_cache_instance: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    """
    Get singleton Redis cache instance

    Returns:
        RedisCache instance
    """
    global _redis_cache_instance
    if _redis_cache_instance is None:
        _redis_cache_instance = RedisCache()
    return _redis_cache_instance
