"""
API Endpoints Check Module

Tests ALL FastAPI endpoints to ensure:
1. They return 200 status codes
2. They return real data (not mock/dummy)
3. Response times are acceptable
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available")


class APIEndpointsCheck(BaseCheck):
    """
    Tests all FastAPI endpoints for availability and data quality.

    CRITICAL check - API endpoints must work and return real data.
    """

    # Default endpoints to test - VERIFIED against actual FastAPI routes (Dec 2025)
    # All endpoints verified returning 200 status
    DEFAULT_ENDPOINTS = [
        # System endpoints (CRITICAL)
        ("GET", "/api/system/health", "System health check"),
        ("GET", "/api/system/services", "System services status"),
        # Portfolio endpoints (HIGH PRIORITY)
        ("GET", "/api/portfolio/positions", "Portfolio positions"),
        ("GET", "/api/portfolio/summary", "Portfolio summary"),
        # Dashboard endpoints
        ("GET", "/api/dashboard/summary", "Dashboard summary"),
        # Sports endpoints
        ("GET", "/api/sports/games", "Sports games"),
        # Predictions endpoints
        ("GET", "/api/predictions/kalshi", "Kalshi markets"),
        # Scanner endpoints (HIGH PRIORITY)
        ("GET", "/api/scanner/quick-scan", "Quick scan"),
        ("GET", "/api/scanner/stored-premiums", "Stored premiums"),
        ("GET", "/api/scanner/ai-picks", "AI picks"),
        # Chat endpoints
        ("GET", "/api/chat/history", "Chat history"),
        # Options endpoints (HIGH PRIORITY)
        ("GET", "/api/options/analysis", "Options analysis"),
        # Earnings endpoints
        ("GET", "/api/earnings/calendar", "Earnings calendar"),
        # Watchlist endpoints
        ("GET", "/api/xtrades/watchlist", "XTrades watchlist"),
        ("GET", "/api/watchlist/all", "All watchlists"),
    ]

    # Patterns that indicate mock/dummy data
    MOCK_DATA_PATTERNS = [
        'mock', 'dummy', 'fake', 'test_data', 'sample',
        'placeholder', 'lorem', 'todo',
    ]

    def __init__(self, base_url: str = "http://localhost:8002",
                 timeout: int = 30, endpoints: List[Tuple] = None):
        """
        Initialize API endpoints check.

        Args:
            base_url: Base URL of the API server
            timeout: Request timeout in seconds
            endpoints: List of (method, path, name) tuples to test
        """
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        self.endpoints = endpoints or self.DEFAULT_ENDPOINTS

    @property
    def name(self) -> str:
        return "api_endpoints"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "server_available",
            "endpoint_responses",
            "response_times",
            "data_quality",
        ]

    def run(self) -> ModuleCheckResult:
        """Run API endpoint checks."""
        self._start_module()

        if not REQUESTS_AVAILABLE:
            self._error("server_available", "requests library not installed")
            return self._end_module()

        # Check if server is available
        server_up = self._check_server_available()
        if not server_up:
            self._fail(
                "server_available",
                f"API server not available at {self.base_url}",
                details={"base_url": self.base_url},
            )
            self._skip("endpoint_responses", "Server not available")
            self._skip("response_times", "Server not available")
            self._skip("data_quality", "Server not available")
            return self._end_module()

        self._pass("server_available", f"API server responding at {self.base_url}")

        # Test each endpoint
        failing_endpoints = []
        slow_endpoints = []
        mock_data_endpoints = []

        for method, path, name in self.endpoints:
            result = self._test_endpoint(method, path, name)

            if not result['success']:
                failing_endpoints.append({
                    'path': path,
                    'name': name,
                    'status': result.get('status'),
                    'error': result.get('error'),
                })
            elif result.get('response_time_ms', 0) > 1000:
                slow_endpoints.append({
                    'path': path,
                    'name': name,
                    'response_time_ms': result['response_time_ms'],
                })

            if result.get('has_mock_data'):
                mock_data_endpoints.append({
                    'path': path,
                    'name': name,
                    'patterns_found': result.get('mock_patterns', []),
                })

        # Report endpoint responses
        if not failing_endpoints:
            self._pass(
                "endpoint_responses",
                f"All {len(self.endpoints)} endpoints responding correctly"
            )
        else:
            self._fail(
                "endpoint_responses",
                f"{len(failing_endpoints)} of {len(self.endpoints)} endpoints failing",
                details={'failing': failing_endpoints},
            )

        # Report response times
        if not slow_endpoints:
            self._pass(
                "response_times",
                "All endpoints responding within acceptable time (<1s)"
            )
        else:
            self._warn(
                "response_times",
                f"{len(slow_endpoints)} endpoints responding slowly (>1s)",
                details={'slow': slow_endpoints},
            )

        # Report data quality
        if not mock_data_endpoints:
            self._pass(
                "data_quality",
                "No mock/dummy data detected in API responses"
            )
        else:
            self._fail(
                "data_quality",
                f"{len(mock_data_endpoints)} endpoints returning mock data!",
                details={'mock_data': mock_data_endpoints},
                auto_fixable=False,
            )

        return self._end_module()

    def _check_server_available(self) -> bool:
        """Check if the API server is available."""
        try:
            response = requests.get(
                f"{self.base_url}/api/system/health",
                timeout=5,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # Try legacy health endpoint
            try:
                response = requests.get(f"{self.base_url}/api/health", timeout=5)
                return response.status_code in (200, 404)
            except requests.exceptions.RequestException:
                return False

    def _test_endpoint(self, method: str, path: str, name: str) -> Dict[str, Any]:
        """Test a single endpoint."""
        url = f"{self.base_url}{path}"

        try:
            start_time = time.time()

            if method.upper() == "GET":
                response = requests.get(url, timeout=self.timeout)
            elif method.upper() == "POST":
                response = requests.post(url, json={}, timeout=self.timeout)
            else:
                return {'success': False, 'error': f'Unsupported method: {method}'}

            response_time_ms = (time.time() - start_time) * 1000

            # Check status
            if response.status_code != 200:
                return {
                    'success': False,
                    'status': response.status_code,
                    'error': f'HTTP {response.status_code}',
                    'response_time_ms': response_time_ms,
                }

            # Check for mock data in response
            response_text = response.text.lower()
            mock_patterns_found = [
                p for p in self.MOCK_DATA_PATTERNS
                if p in response_text
            ]

            return {
                'success': True,
                'status': 200,
                'response_time_ms': response_time_ms,
                'has_mock_data': bool(mock_patterns_found),
                'mock_patterns': mock_patterns_found,
            }

        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Request timeout',
                'status': None,
            }
        except requests.exceptions.ConnectionError as e:
            return {
                'success': False,
                'error': f'Connection error: {str(e)[:100]}',
                'status': None,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)[:100],
                'status': None,
            }

    def can_auto_fix(self, check_name: str) -> bool:
        """API endpoint issues cannot be auto-fixed."""
        return False
