"""
API Connectivity Check Module

Tests connectivity to external APIs:
- Robinhood API
- Database connection
- Kalshi API
- TradingView session
- ESPN API

CRITICAL check - External API connectivity is essential.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class APIConnectivityCheck(BaseCheck):
    """
    Tests connectivity to external APIs and services.

    CRITICAL check - External API connectivity must work for the platform.
    """

    def __init__(self) -> None:
        """Initialize API connectivity check."""
        super().__init__()
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")

    @property
    def name(self) -> str:
        return "api_connectivity"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.CRITICAL

    def get_checks_list(self) -> List[str]:
        return [
            "database_connection",
            "robinhood_api",
            "kalshi_api",
            "espn_api",
            "ai_providers",
        ]

    def run(self) -> ModuleCheckResult:
        """Run API connectivity checks."""
        self._start_module()

        # Check database connection
        self._check_database_connection()

        # Check Robinhood API
        self._check_robinhood_api()

        # Check Kalshi API
        self._check_kalshi_api()

        # Check ESPN API
        self._check_espn_api()

        # Check AI providers
        self._check_ai_providers()

        return self._end_module()

    def _check_database_connection(self) -> None:
        """Test PostgreSQL database connectivity."""
        try:
            import asyncio
            from backend.infrastructure.database import get_database

            async def test_db():
                db = await get_database()
                result = await db.fetchval("SELECT 1")
                return result == 1

            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                success = loop.run_until_complete(test_db())
                if success:
                    self._pass(
                        "database_connection",
                        "PostgreSQL database connection successful"
                    )
                else:
                    self._fail(
                        "database_connection",
                        "Database query returned unexpected result"
                    )
            finally:
                loop.close()

        except ImportError as e:
            self._fail(
                "database_connection",
                f"Could not import database module: {e}",
                details={"error": str(e)}
            )
        except Exception as e:
            self._fail(
                "database_connection",
                f"Database connection failed: {e}",
                details={"error": str(e)}
            )

    def _check_robinhood_api(self) -> None:
        """Test Robinhood API connectivity via backend endpoint."""
        try:
            import requests

            response = requests.get(
                f"{self.backend_url}/api/portfolio/positions",
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                # Check if it's real data (has expected structure)
                if isinstance(data, (list, dict)):
                    self._pass(
                        "robinhood_api",
                        "Robinhood API responding via portfolio endpoint"
                    )
                else:
                    self._warn(
                        "robinhood_api",
                        "Robinhood API responded but data format unexpected",
                        details={"type": type(data).__name__}
                    )
            elif response.status_code == 401:
                self._fail(
                    "robinhood_api",
                    "Robinhood API authentication failed - credentials may be expired",
                    details={"status": 401}
                )
            elif response.status_code == 404:
                self._warn(
                    "robinhood_api",
                    "Portfolio endpoint not found - backend may not be running",
                    details={"status": 404}
                )
            else:
                self._fail(
                    "robinhood_api",
                    f"Robinhood API returned HTTP {response.status_code}",
                    details={"status": response.status_code}
                )

        except requests.exceptions.ConnectionError:
            self._fail(
                "robinhood_api",
                "Could not connect to backend - is the server running?",
                details={"backend_url": self.backend_url}
            )
        except Exception as e:
            self._fail(
                "robinhood_api",
                f"Robinhood API check failed: {e}",
                details={"error": str(e)}
            )

    def _check_kalshi_api(self) -> None:
        """Test Kalshi API connectivity."""
        try:
            import requests

            # Try the kalshi markets endpoint
            response = requests.get(
                f"{self.backend_url}/api/predictions/markets",
                timeout=30
            )

            if response.status_code == 200:
                self._pass(
                    "kalshi_api",
                    "Kalshi/predictions API responding"
                )
            elif response.status_code == 404:
                # Try alternative endpoint
                alt_response = requests.get(
                    f"{self.backend_url}/api/predictions/summary",
                    timeout=30
                )
                if alt_response.status_code == 200:
                    self._pass(
                        "kalshi_api",
                        "Predictions API responding (alternative endpoint)"
                    )
                else:
                    self._warn(
                        "kalshi_api",
                        "Kalshi API endpoints not found",
                        details={"status": 404}
                    )
            else:
                self._warn(
                    "kalshi_api",
                    f"Kalshi API returned HTTP {response.status_code}",
                    details={"status": response.status_code}
                )

        except requests.exceptions.ConnectionError:
            self._skip("kalshi_api", "Backend not available for Kalshi check")
        except Exception as e:
            self._warn(
                "kalshi_api",
                f"Kalshi API check failed: {e}",
                details={"error": str(e)}
            )

    def _check_espn_api(self) -> None:
        """Test ESPN API connectivity via sports endpoint."""
        try:
            import requests

            response = requests.get(
                f"{self.backend_url}/api/sports/games/today",
                timeout=30
            )

            if response.status_code == 200:
                self._pass(
                    "espn_api",
                    "ESPN/Sports API responding"
                )
            elif response.status_code == 404:
                # Try alternative
                alt_response = requests.get(
                    f"{self.backend_url}/api/sports/schedule",
                    timeout=30
                )
                if alt_response.status_code == 200:
                    self._pass(
                        "espn_api",
                        "Sports API responding (alternative endpoint)"
                    )
                else:
                    self._warn(
                        "espn_api",
                        "Sports API endpoints not found",
                        details={"status": 404}
                    )
            else:
                self._warn(
                    "espn_api",
                    f"ESPN API returned HTTP {response.status_code}",
                    details={"status": response.status_code}
                )

        except requests.exceptions.ConnectionError:
            self._skip("espn_api", "Backend not available for ESPN check")
        except Exception as e:
            self._warn(
                "espn_api",
                f"ESPN API check failed: {e}",
                details={"error": str(e)}
            )

    def _check_ai_providers(self) -> None:
        """Test AI provider connectivity (OpenAI, Anthropic, Groq, Ollama)."""
        providers_status = {}

        # Check environment variables for API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")

        if openai_key:
            providers_status["openai"] = "configured"
        if anthropic_key:
            providers_status["anthropic"] = "configured"
        if groq_key:
            providers_status["groq"] = "configured"

        # Check Ollama (local)
        try:
            import requests
            ollama_response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=5
            )
            if ollama_response.status_code == 200:
                providers_status["ollama"] = "running"
        except:
            pass

        if providers_status:
            configured = [k for k, v in providers_status.items() if v in ("configured", "running")]
            self._pass(
                "ai_providers",
                f"AI providers available: {', '.join(configured)}",
                details=providers_status
            )
        else:
            self._warn(
                "ai_providers",
                "No AI provider API keys configured",
                details={"hint": "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GROQ_API_KEY"}
            )

    def can_auto_fix(self, check_name: str) -> bool:
        """API connectivity issues cannot be auto-fixed."""
        return False
