"""
Query Cache Module
Provides in-memory caching with TTL for database queries to reduce load
"""

import time
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QueryCache:
    """
    Simple in-memory cache with Time-To-Live (TTL) support
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryCache, cls).__new__(cls)
            cls._instance._cache = {}
            cls._instance._stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0
            }
        return cls._instance

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if it exists and hasn't expired
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                self._stats['hits'] += 1
                return value
            else:
                # Expired
                del self._cache[key]
                self._stats['evictions'] += 1
        
        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 60):
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds (default 60)
        """
        expiry = time.time() + ttl_seconds
        self._cache[key] = (value, expiry)
        
        # Periodic cleanup (simple implementation)
        if len(self._cache) > 1000:
            self._cleanup()

    def invalidate(self, key_pattern: str = None):
        """
        Invalidate cache keys matching a pattern (or all if None)
        
        Args:
            key_pattern: String pattern to match (startswith)
        """
        if key_pattern is None:
            self._cache.clear()
            logger.info("Cache cleared completely")
        else:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(key_pattern)]
            for k in keys_to_remove:
                del self._cache[k]
            logger.info(f"Invalidated {len(keys_to_remove)} keys matching '{key_pattern}'")

    def _cleanup(self):
        """Remove expired items"""
        now = time.time()
        keys_to_remove = [k for k, (_, expiry) in self._cache.items() if now >= expiry]
        for k in keys_to_remove:
            del self._cache[k]
        self._stats['evictions'] += len(keys_to_remove)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            **self._stats,
            'size': len(self._cache)
        }

# Global instance
query_cache = QueryCache()
