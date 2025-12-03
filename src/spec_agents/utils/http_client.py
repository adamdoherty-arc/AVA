"""
HTTP Client Wrapper for SpecAgents

Provides:
- Async HTTP client with retry logic
- Response validation
- Error handling
- Request/response logging
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HTTPResponse:
    """Wrapper for HTTP response data"""
    status_code: int
    headers: Dict[str, str]
    json_data: Optional[Any] = None
    text: Optional[str] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class SpecHttpClient:
    """
    HTTP client for SpecAgent API testing.

    Features:
    - Async request handling
    - Automatic retry on failure
    - Response validation
    - Request timing
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8002/api",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize HTTP client

        Args:
            base_url: Base URL for API requests
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client = None

    async def _get_client(self) -> None:
        """Get or create httpx async client"""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HTTPResponse:
        """
        Make GET request

        Args:
            path: API path
            params: Query parameters
            headers: Request headers

        Returns:
            HTTPResponse with response data
        """
        return await self._request('GET', path, params=params, headers=headers)

    async def post(
        self,
        path: str,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> HTTPResponse:
        """
        Make POST request

        Args:
            path: API path
            json_data: JSON body data
            params: Query parameters
            headers: Request headers

        Returns:
            HTTPResponse with response data
        """
        return await self._request('POST', path, json=json_data, params=params, headers=headers)

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> HTTPResponse:
        """
        Make HTTP request with retry logic

        Args:
            method: HTTP method
            path: API path
            **kwargs: Additional request arguments

        Returns:
            HTTPResponse with response data
        """
        import time

        client = await self._get_client()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = await client.request(method, path, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                # Parse response
                json_data = None
                text = None

                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    try:
                        json_data = response.json()
                    except Exception:
                        text = response.text
                else:
                    text = response.text

                return HTTPResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    json_data=json_data,
                    text=text,
                    elapsed_ms=elapsed_ms,
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{method} {path} - {e}"
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)

        # All retries failed
        return HTTPResponse(
            status_code=0,
            headers={},
            error=last_error,
        )

    async def close(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # Convenience methods

    async def health_check(self, path: str = "/health") -> bool:
        """Check if API is healthy"""
        response = await self.get(path)
        return response.status_code == 200

    async def get_json(self, path: str) -> Optional[Any]:
        """Get JSON data from path, returns None on error"""
        response = await self.get(path)
        if response.status_code == 200 and response.json_data is not None:
            return response.json_data
        return None

    def validate_response(
        self,
        response: HTTPResponse,
        expected_status: int = 200,
        required_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Validate HTTP response

        Args:
            response: HTTPResponse to validate
            expected_status: Expected status code
            required_fields: Fields that must be present in JSON response

        Returns:
            List of validation issues
        """
        issues = []

        # Check status code
        if response.status_code != expected_status:
            issues.append({
                'type': 'status_code',
                'expected': expected_status,
                'actual': response.status_code,
                'message': f"Expected status {expected_status}, got {response.status_code}"
            })

        # Check for error
        if response.error:
            issues.append({
                'type': 'request_error',
                'error': response.error,
                'message': f"Request failed: {response.error}"
            })

        # Check required fields
        if required_fields and response.json_data:
            data = response.json_data
            for field in required_fields:
                if field not in data:
                    issues.append({
                        'type': 'missing_field',
                        'field': field,
                        'message': f"Missing required field: {field}"
                    })

        return issues
