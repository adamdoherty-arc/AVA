"""
HTTP Client Wrapper
===================

Easy-to-use wrapper around RobustAPIClient for migration.
Drop-in replacement for direct requests usage.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from functools import lru_cache

from .api_client import (
    RobustAPIClient,
    APIClientConfig,
    APIRequestError,
    CircuitBreakerOpen,
    RateLimitExceeded,
    get_api_client
)

logger = logging.getLogger(__name__)


# =============================================================================
# EASY-TO-USE HTTP FUNCTIONS
# =============================================================================

def http_get(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    cache: bool = False,
    cache_ttl: int = 300,
    timeout: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple GET request with retry and circuit breaker.

    Drop-in replacement for requests.get().json()

    Usage:
        # Old way:
        response = requests.get(url, params=params)
        data = response.json()

        # New way:
        data = http_get(url, params=params)

    Args:
        url: Request URL
        params: Query parameters
        headers: Request headers
        cache: Enable response caching
        cache_ttl: Cache TTL in seconds
        timeout: Request timeout

    Returns:
        Response JSON as dict

    Raises:
        APIRequestError: If request fails after retries
        CircuitBreakerOpen: If service is unavailable
    """
    client = get_api_client()
    return client.get(
        url,
        params=params,
        headers=headers,
        cache=cache,
        cache_ttl=cache_ttl,
        **kwargs
    )


def http_post(
    url: str,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Simple POST request with retry and circuit breaker.

    Drop-in replacement for requests.post().json()

    Usage:
        # Old way:
        response = requests.post(url, json=payload)
        data = response.json()

        # New way:
        data = http_post(url, json=payload)
    """
    client = get_api_client()
    return client.post(
        url,
        data=data,
        json=json,
        headers=headers,
        **kwargs
    )


async def async_http_get(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    cache: bool = False,
    cache_ttl: int = 300,
    **kwargs
) -> Dict[str, Any]:
    """
    Async GET request with retry, rate limiting, and circuit breaker.

    Usage:
        data = await async_http_get(url, params=params)
    """
    client = get_api_client()
    return await client.async_get(
        url,
        params=params,
        headers=headers,
        cache=cache,
        cache_ttl=cache_ttl,
        **kwargs
    )


async def async_http_post(
    url: str,
    data: Optional[Dict] = None,
    json: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Async POST request with retry, rate limiting, and circuit breaker.

    Usage:
        data = await async_http_post(url, json=payload)
    """
    client = get_api_client()
    return await client.async_post(
        url,
        data=data,
        json=json,
        headers=headers,
        **kwargs
    )


# =============================================================================
# SERVICE-SPECIFIC CLIENTS
# =============================================================================

class ServiceClient:
    """
    Base class for service-specific API clients.

    Provides automatic base URL handling, auth headers,
    and service-specific configuration.

    Usage:
        class KalshiService(ServiceClient):
            def __init__(self):
                super().__init__(
                    base_url="https://api.kalshi.com/v2",
                    service_name="kalshi"
                )
    """

    def __init__(
        self,
        base_url: str,
        service_name: str,
        default_headers: Optional[Dict] = None,
        cache_enabled: bool = True,
        rate_limit: float = 10.0
    ):
        self.base_url = base_url.rstrip('/')
        self.service_name = service_name
        self.default_headers = default_headers or {}
        self.cache_enabled = cache_enabled

        # Create service-specific client
        config = APIClientConfig(
            rate_limit_per_second=rate_limit,
            cache_enabled=cache_enabled
        )
        self._client = RobustAPIClient(config)
        self._auth_token: Optional[str] = None

    def set_auth_token(self, token: str) -> None:
        """Set authentication token for requests"""
        self._auth_token = token

    def _get_headers(self, extra_headers: Optional[Dict] = None) -> Dict:
        """Build request headers"""
        headers = {**self.default_headers}
        if self._auth_token:
            headers['Authorization'] = self._auth_token
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint"""
        endpoint = endpoint.lstrip('/')
        return f"{self.base_url}/{endpoint}"

    def get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache: bool = False,
        cache_ttl: int = 300
    ) -> Dict[str, Any]:
        """Make GET request to service"""
        return self._client.get(
            self._build_url(endpoint),
            params=params,
            headers=self._get_headers(headers),
            cache=cache and self.cache_enabled,
            cache_ttl=cache_ttl
        )

    def post(
        self,
        endpoint: str,
        json: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make POST request to service"""
        return self._client.post(
            self._build_url(endpoint),
            json=json,
            data=data,
            headers=self._get_headers(headers)
        )

    async def async_get(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        cache: bool = False,
        cache_ttl: int = 300
    ) -> Dict[str, Any]:
        """Make async GET request to service"""
        return await self._client.async_get(
            self._build_url(endpoint),
            params=params,
            headers=self._get_headers(headers),
            cache=cache and self.cache_enabled,
            cache_ttl=cache_ttl
        )

    async def async_post(
        self,
        endpoint: str,
        json: Optional[Dict] = None,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make async POST request to service"""
        return await self._client.async_post(
            self._build_url(endpoint),
            json=json,
            data=data,
            headers=self._get_headers(headers)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        stats = self._client.get_stats()
        stats['service'] = self.service_name
        return stats

    async def close(self) -> None:
        """Close async session"""
        await self._client.close()


# =============================================================================
# PRE-CONFIGURED SERVICE CLIENTS
# =============================================================================

@lru_cache(maxsize=1)
def get_polygon_client() -> ServiceClient:
    """Get Polygon.io API client"""
    api_key = os.getenv('POLYGON_API_KEY', '')
    return ServiceClient(
        base_url="https://api.polygon.io",
        service_name="polygon",
        default_headers={'Authorization': f'Bearer {api_key}'} if api_key else {},
        cache_enabled=True,
        rate_limit=5.0  # Polygon has rate limits
    )


@lru_cache(maxsize=1)
def get_tradier_client() -> ServiceClient:
    """Get Tradier API client"""
    api_key = os.getenv('TRADIER_API_KEY', '')
    return ServiceClient(
        base_url="https://api.tradier.com/v1",
        service_name="tradier",
        default_headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        } if api_key else {},
        cache_enabled=True,
        rate_limit=10.0
    )


@lru_cache(maxsize=1)
def get_fred_client() -> ServiceClient:
    """Get FRED (Federal Reserve) API client"""
    api_key = os.getenv('FRED_API_KEY', '')
    return ServiceClient(
        base_url="https://api.stlouisfed.org/fred",
        service_name="fred",
        default_headers={},
        cache_enabled=True,
        rate_limit=2.0  # FRED has lower limits
    )


@lru_cache(maxsize=1)
def get_kalshi_client() -> ServiceClient:
    """Get Kalshi API client"""
    return ServiceClient(
        base_url="https://api.elections.kalshi.com/trade-api/v2",
        service_name="kalshi",
        default_headers={
            'accept': 'application/json',
            'content-type': 'application/json'
        },
        cache_enabled=True,
        rate_limit=10.0
    )


# =============================================================================
# EXCEPTION HELPERS
# =============================================================================

def safe_request(func):
    """
    Decorator to safely handle API request exceptions.

    Usage:
        @safe_request
        def fetch_data():
            return http_get("https://api.example.com/data")
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CircuitBreakerOpen as e:
            logger.warning(f"Service unavailable (circuit open): {e}")
            return None
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return None
        except APIRequestError as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper


def safe_async_request(func):
    """Async version of safe_request decorator"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except CircuitBreakerOpen as e:
            logger.warning(f"Service unavailable (circuit open): {e}")
            return None
        except RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return None
        except APIRequestError as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Simple functions
    'http_get',
    'http_post',
    'async_http_get',
    'async_http_post',
    # Service client
    'ServiceClient',
    # Pre-configured clients
    'get_polygon_client',
    'get_tradier_client',
    'get_fred_client',
    'get_kalshi_client',
    # Decorators
    'safe_request',
    'safe_async_request',
    # Exceptions
    'APIRequestError',
    'CircuitBreakerOpen',
    'RateLimitExceeded',
]
