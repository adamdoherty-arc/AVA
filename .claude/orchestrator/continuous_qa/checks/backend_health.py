"""
Backend Health Check Module

Tests backend health:
- Import all routers without errors
- AI client connectivity
- Backend process status
- Router registration

HIGH priority check - Backend health is critical for the platform.
"""

import os
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class BackendHealthCheck(BaseCheck):
    """
    Tests backend health including router imports and AI clients.

    HIGH priority check - Backend must be healthy.
    """

    # Expected routers from backend/main.py
    EXPECTED_ROUTERS = [
        'sports', 'predictions', 'dashboard', 'chat', 'research',
        'portfolio', 'strategy', 'scanner', 'agents', 'earnings',
        'xtrades', 'system', 'technicals', 'knowledge', 'enhancements',
        'analytics', 'options', 'subscriptions', 'cache', 'discord',
        'settings', 'integration_test', 'watchlist', 'qa_dashboard',
        'goals', 'briefings', 'automations', 'smart_money',
        'advanced_technicals', 'options_indicators', 'sports_streaming',
        'sports_v2', 'notifications', 'portfolio_v2', 'orchestrator',
        'stocks_tiles', 'reasoning'
    ]

    def __init__(self) -> None:
        """Initialize backend health check."""
        super().__init__()
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")

    @property
    def name(self) -> str:
        return "backend_health"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.HIGH

    def get_checks_list(self) -> List[str]:
        return [
            "router_imports",
            "ai_clients",
            "backend_server",
        ]

    def run(self) -> ModuleCheckResult:
        """Run backend health checks."""
        self._start_module()

        # Check router imports
        self._check_router_imports()

        # Check AI client availability
        self._check_ai_clients()

        # Check backend server is running
        self._check_backend_server()

        return self._end_module()

    def _check_router_imports(self) -> None:
        """Test that all routers can be imported without errors."""
        import_errors = []
        successful_imports = []

        for router_name in self.EXPECTED_ROUTERS:
            try:
                module_path = f"backend.routers.{router_name}"
                module = importlib.import_module(module_path)

                # Check if router attribute exists
                if hasattr(module, 'router'):
                    successful_imports.append(router_name)
                else:
                    import_errors.append({
                        'router': router_name,
                        'error': 'No router attribute found'
                    })

            except ImportError as e:
                import_errors.append({
                    'router': router_name,
                    'error': str(e)
                })
            except Exception as e:
                import_errors.append({
                    'router': router_name,
                    'error': f"Unexpected error: {e}"
                })

        if import_errors:
            self._fail(
                "router_imports",
                f"{len(import_errors)} of {len(self.EXPECTED_ROUTERS)} routers failed to import",
                details={
                    'failed': import_errors[:10],  # Limit to 10
                    'successful_count': len(successful_imports)
                }
            )
        else:
            self._pass(
                "router_imports",
                f"All {len(successful_imports)} routers imported successfully"
            )

    def _check_ai_clients(self) -> None:
        """Check AI client availability and configuration."""
        ai_status = {}

        # Check OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            ai_status["openai"] = "configured"

        # Check Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            ai_status["anthropic"] = "configured"

        # Check Groq
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            ai_status["groq"] = "configured"

        # Check Ollama (local)
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                ai_status["ollama"] = f"running ({len(models)} models)"
        except:
            pass

        # Try to import backend AI client
        try:
            from backend.infrastructure.ai_client import get_ai_client
            ai_status["backend_ai_client"] = "available"
        except ImportError as e:
            ai_status["backend_ai_client"] = f"import error: {e}"
        except Exception as e:
            ai_status["backend_ai_client"] = f"error: {e}"

        if ai_status:
            working = [k for k, v in ai_status.items() if 'configured' in str(v) or 'running' in str(v) or 'available' in str(v)]
            if working:
                self._pass(
                    "ai_clients",
                    f"AI clients available: {', '.join(working)}",
                    details=ai_status
                )
            else:
                self._warn(
                    "ai_clients",
                    "No AI clients fully configured",
                    details=ai_status
                )
        else:
            self._fail(
                "ai_clients",
                "No AI clients configured",
                details={"hint": "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama"}
            )

    def _check_backend_server(self) -> None:
        """Check if backend server is running."""
        try:
            import requests

            # Try system status endpoint
            response = requests.get(
                f"{self.backend_url}/api/system/status",
                timeout=10
            )

            if response.status_code == 200:
                self._pass(
                    "backend_server",
                    f"Backend server running at {self.backend_url}"
                )
            elif response.status_code == 404:
                # Try health endpoint
                health_response = requests.get(
                    f"{self.backend_url}/api/health",
                    timeout=10
                )
                if health_response.status_code == 200:
                    self._pass(
                        "backend_server",
                        f"Backend server running at {self.backend_url}"
                    )
                else:
                    self._warn(
                        "backend_server",
                        f"Backend responding but status endpoint not found",
                        details={"status": response.status_code}
                    )
            else:
                self._fail(
                    "backend_server",
                    f"Backend returned HTTP {response.status_code}",
                    details={"status": response.status_code}
                )

        except requests.exceptions.ConnectionError:
            self._fail(
                "backend_server",
                f"Cannot connect to backend at {self.backend_url}",
                details={"hint": "Start backend with: cd backend && uvicorn main:app --port 8002"}
            )
        except Exception as e:
            self._fail(
                "backend_server",
                f"Backend check failed: {e}",
                details={"error": str(e)}
            )

    def can_auto_fix(self, check_name: str) -> bool:
        """Backend health issues cannot be auto-fixed."""
        return False
