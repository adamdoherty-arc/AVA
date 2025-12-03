"""
Base Connector for Magnus Financial Assistant
Provides unified interface for accessing all Magnus features
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseConnector(ABC):
    """
    Base class for all Magnus feature connectors.

    Features:
    - Unified interface for data access
    - Built-in caching with TTL
    - Error handling and logging
    - Response validation
    - Rate limiting support
    """

    def __init__(self, cache_ttl: int = 300):
        """
        Initialize connector.

        Args:
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.cache = {}
        self.cache_ttl = cache_ttl
        self.cache_timestamps = {}
        self.connector_name = self.__class__.__name__
        logger.info(f"Initialized {self.connector_name} (cache TTL: {cache_ttl}s)")

    @abstractmethod
    def get_data(self, **kwargs) -> Dict[str, Any]:
        """
        Fetch data from the Magnus feature.

        Args:
            **kwargs: Feature-specific parameters

        Returns:
            Dictionary containing the requested data

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    def validate_response(self, data: Dict[str, Any]) -> bool:
        """
        Validate the response data.

        Args:
            data: Response data to validate

        Returns:
            True if valid, False otherwise

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False

        if cache_key not in self.cache_timestamps:
            return False

        age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_ttl

    def get_cached_or_fetch(
        self,
        cache_key: str,
        fetch_func,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get from cache or fetch fresh data.

        Args:
            cache_key: Unique key for this data
            fetch_func: Function to call if cache miss
            *args: Arguments for fetch function
            **kwargs: Keyword arguments for fetch function

        Returns:
            Cached or fresh data
        """
        # Check cache
        if self._is_cache_valid(cache_key):
            logger.debug(f"{self.connector_name}: Cache hit for {cache_key}")
            return self.cache[cache_key]

        # Cache miss - fetch fresh data
        logger.debug(f"{self.connector_name}: Cache miss for {cache_key}, fetching...")

        try:
            data = fetch_func(*args, **kwargs)

            # Validate response
            if not self.validate_response(data):
                logger.error(f"{self.connector_name}: Invalid response for {cache_key}")
                return {"error": "Invalid response", "cache_key": cache_key}

            # Store in cache
            self.cache[cache_key] = data
            self.cache_timestamps[cache_key] = datetime.now()

            logger.info(f"{self.connector_name}: Fetched and cached {cache_key}")
            return data

        except Exception as e:
            logger.error(f"{self.connector_name}: Error fetching {cache_key}: {e}")
            return {"error": str(e), "cache_key": cache_key}

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache.

        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        if cache_key:
            if cache_key in self.cache:
                del self.cache[cache_key]
                del self.cache_timestamps[cache_key]
                logger.info(f"{self.connector_name}: Cleared cache for {cache_key}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info(f"{self.connector_name}: Cleared all cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        valid_keys = [k for k in self.cache.keys() if self._is_cache_valid(k)]

        return {
            "connector": self.connector_name,
            "cache_size": len(self.cache),
            "valid_cache_entries": len(valid_keys),
            "cache_ttl": self.cache_ttl,
            "cache_keys": list(self.cache.keys())
        }

    def format_error_response(
        self,
        error: Exception,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Format error response consistently.

        Args:
            error: Exception that occurred
            context: Additional context about the error

        Returns:
            Formatted error dictionary
        """
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__,
            "connector": self.connector_name,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }

    def format_success_response(
        self,
        data: Any,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Format success response consistently.

        Args:
            data: Response data
            metadata: Optional metadata to include

        Returns:
            Formatted success dictionary
        """
        response = {
            "success": True,
            "data": data,
            "connector": self.connector_name,
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            response["metadata"] = metadata

        return response


class ConnectorRegistry:
    """
    Registry for all Magnus connectors.
    Provides centralized access to all feature connectors.
    """

    def __init__(self) -> None:
        """Initialize connector registry."""
        self.connectors: Dict[str, BaseConnector] = {}
        logger.info("Initialized ConnectorRegistry")

    def register(self, name: str, connector: BaseConnector):
        """
        Register a connector.

        Args:
            name: Connector name
            connector: Connector instance
        """
        self.connectors[name] = connector
        logger.info(f"Registered connector: {name}")

    def get(self, name: str) -> Optional[BaseConnector]:
        """
        Get a connector by name.

        Args:
            name: Connector name

        Returns:
            Connector instance or None if not found
        """
        return self.connectors.get(name)

    def list_connectors(self) -> List[str]:
        """Get list of all registered connectors."""
        return list(self.connectors.keys())

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connectors."""
        return {
            "total_connectors": len(self.connectors),
            "connectors": {
                name: connector.get_cache_stats()
                for name, connector in self.connectors.items()
            }
        }

    def clear_all_caches(self) -> None:
        """Clear caches for all connectors."""
        for connector in self.connectors.values():
            connector.clear_cache()
        logger.info("Cleared all connector caches")


# Global registry instance
_global_registry = None


def get_registry() -> ConnectorRegistry:
    """Get the global connector registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ConnectorRegistry()
    return _global_registry


def register_connector(name: str, connector: BaseConnector):
    """
    Register a connector in the global registry.

    Args:
        name: Connector name
        connector: Connector instance
    """
    registry = get_registry()
    registry.register(name, connector)


def get_connector(name: str) -> Optional[BaseConnector]:
    """
    Get a connector from the global registry.

    Args:
        name: Connector name

    Returns:
        Connector instance or None
    """
    registry = get_registry()
    return registry.get(name)
