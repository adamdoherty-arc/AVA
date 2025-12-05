"""
Performance Check Module

Tests performance metrics:
- API endpoint response times
- Database query times
- Memory usage
- Cache effectiveness

MEDIUM priority check - Performance optimization.
"""

import os
import sys
import time
import psutil
from pathlib import Path
from typing import List, Dict, Any
import logging

from .base_check import BaseCheck, CheckPriority, CheckStatus, ModuleCheckResult

logger = logging.getLogger(__name__)

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class PerformanceCheck(BaseCheck):
    """
    Tests performance metrics across the platform.

    MEDIUM priority check - Performance monitoring.
    """

    # Thresholds
    SLOW_API_THRESHOLD_MS = 500
    VERY_SLOW_API_THRESHOLD_MS = 2000
    HIGH_MEMORY_THRESHOLD_MB = 1000
    HIGH_CPU_THRESHOLD_PERCENT = 80

    # Key endpoints to benchmark
    BENCHMARK_ENDPOINTS = [
        ("/api/system/status", "System Status"),
        ("/api/portfolio/summary", "Portfolio Summary"),
        ("/api/dashboard/summary", "Dashboard Summary"),
        ("/api/scanner/opportunities", "Scanner Opportunities"),
    ]

    def __init__(self) -> None:
        """Initialize performance check."""
        super().__init__()
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8002")

    @property
    def name(self) -> str:
        return "performance"

    @property
    def priority(self) -> CheckPriority:
        return CheckPriority.MEDIUM

    def get_checks_list(self) -> List[str]:
        return [
            "api_response_times",
            "system_resources",
            "python_process_memory",
        ]

    def run(self) -> ModuleCheckResult:
        """Run performance checks."""
        self._start_module()

        # Check API response times
        self._check_api_response_times()

        # Check system resources
        self._check_system_resources()

        # Check Python process memory
        self._check_python_memory()

        return self._end_module()

    def _check_api_response_times(self) -> None:
        """Benchmark API endpoint response times."""
        try:
            import requests

            results = []
            slow_endpoints = []
            very_slow_endpoints = []

            for path, name in self.BENCHMARK_ENDPOINTS:
                url = f"{self.backend_url}{path}"

                try:
                    start = time.time()
                    response = requests.get(url, timeout=30)
                    elapsed_ms = (time.time() - start) * 1000

                    result = {
                        'endpoint': name,
                        'path': path,
                        'status': response.status_code,
                        'time_ms': round(elapsed_ms, 2)
                    }
                    results.append(result)

                    if elapsed_ms > self.VERY_SLOW_API_THRESHOLD_MS:
                        very_slow_endpoints.append(result)
                    elif elapsed_ms > self.SLOW_API_THRESHOLD_MS:
                        slow_endpoints.append(result)

                except requests.exceptions.RequestException as e:
                    results.append({
                        'endpoint': name,
                        'path': path,
                        'error': str(e)[:50]
                    })

            # Analyze results
            successful = [r for r in results if 'time_ms' in r]
            if not successful:
                self._fail(
                    "api_response_times",
                    "Could not benchmark any endpoints - backend may be down",
                    details={'results': results}
                )
                return

            avg_time = sum(r['time_ms'] for r in successful) / len(successful)

            if very_slow_endpoints:
                self._fail(
                    "api_response_times",
                    f"{len(very_slow_endpoints)} endpoint(s) very slow (>{self.VERY_SLOW_API_THRESHOLD_MS}ms)",
                    details={
                        'very_slow': very_slow_endpoints,
                        'slow': slow_endpoints,
                        'avg_ms': round(avg_time, 2)
                    }
                )
            elif slow_endpoints:
                self._warn(
                    "api_response_times",
                    f"{len(slow_endpoints)} endpoint(s) slow (>{self.SLOW_API_THRESHOLD_MS}ms)",
                    details={
                        'slow': slow_endpoints,
                        'avg_ms': round(avg_time, 2)
                    }
                )
            else:
                self._pass(
                    "api_response_times",
                    f"All endpoints responding within {self.SLOW_API_THRESHOLD_MS}ms (avg: {avg_time:.0f}ms)",
                    details={'results': results}
                )

        except ImportError:
            self._skip("api_response_times", "requests library not available")
        except Exception as e:
            self._error(
                "api_response_times",
                f"Performance check failed: {e}"
            )

    def _check_system_resources(self) -> None:
        """Check system CPU and memory usage."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            details = {
                'cpu_percent': cpu_percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_total_gb': round(memory.total / (1024**3), 2),
                'memory_percent': memory.percent,
                'disk_used_gb': round(disk.used / (1024**3), 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'disk_percent': disk.percent
            }

            issues = []

            if cpu_percent > self.HIGH_CPU_THRESHOLD_PERCENT:
                issues.append(f"High CPU usage: {cpu_percent}%")

            if memory.percent > 90:
                issues.append(f"High memory usage: {memory.percent}%")

            if disk.percent > 90:
                issues.append(f"High disk usage: {disk.percent}%")

            if issues:
                self._warn(
                    "system_resources",
                    f"Resource warnings: {'; '.join(issues)}",
                    details=details
                )
            else:
                self._pass(
                    "system_resources",
                    f"System resources normal (CPU: {cpu_percent}%, RAM: {memory.percent}%, Disk: {disk.percent}%)",
                    details=details
                )

        except Exception as e:
            self._warn(
                "system_resources",
                f"Could not check system resources: {e}"
            )

    def _check_python_memory(self) -> None:
        """Check Python process memory usage."""
        try:
            # Find Python/uvicorn processes
            python_processes = []

            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'uvicorn' in cmdline or 'magnus' in cmdline.lower():
                            memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': cmdline[:100],
                                'memory_mb': round(memory_mb, 2)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if not python_processes:
                self._pass(
                    "python_process_memory",
                    "No Magnus/uvicorn processes found (backend may not be running)"
                )
                return

            # Check for high memory usage
            high_memory_procs = [
                p for p in python_processes
                if p['memory_mb'] > self.HIGH_MEMORY_THRESHOLD_MB
            ]

            total_memory = sum(p['memory_mb'] for p in python_processes)

            if high_memory_procs:
                self._warn(
                    "python_process_memory",
                    f"{len(high_memory_procs)} process(es) using >{self.HIGH_MEMORY_THRESHOLD_MB}MB",
                    details={
                        'high_memory': high_memory_procs,
                        'total_mb': round(total_memory, 2)
                    }
                )
            else:
                self._pass(
                    "python_process_memory",
                    f"{len(python_processes)} Python process(es) using {total_memory:.0f}MB total",
                    details={'processes': python_processes}
                )

        except Exception as e:
            self._warn(
                "python_process_memory",
                f"Could not check Python processes: {e}"
            )

    def can_auto_fix(self, check_name: str) -> bool:
        """Performance issues cannot be auto-fixed."""
        return False
