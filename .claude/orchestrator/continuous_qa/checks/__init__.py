"""
Check modules for Magnus QA system.

Available checks:
- APIEndpointsCheck: Tests FastAPI endpoints for availability and data quality
- APIConnectivityCheck: Tests external API connections (Robinhood, Kalshi, etc.)
- DatabaseHealthCheck: Tests database health and performance
- BackendHealthCheck: Tests backend router imports and AI clients
- PerformanceCheck: Tests API response times and system resources
- CodeQualityCheck: Checks code rules and style compliance
- DummyDataDetectorCheck: Detects mock/dummy/fake data patterns
- UIComponentsCheck: Tests React frontend build and components
- DataSyncCheck: Validates data freshness from all sources
- SharedCodeAnalyzerCheck: Finds duplicate code patterns
"""

from .base_check import (
    BaseCheck,
    CheckPriority,
    CheckStatus,
    CheckResult,
    ModuleCheckResult,
    CheckRunner,
)
from .api_endpoints import APIEndpointsCheck
from .api_connectivity import APIConnectivityCheck
from .database_health import DatabaseHealthCheck
from .backend_health import BackendHealthCheck
from .performance_check import PerformanceCheck
from .code_quality import CodeQualityCheck
from .dummy_data_detector import DummyDataDetectorCheck
from .ui_components import UIComponentsCheck
from .data_sync import DataSyncCheck
from .shared_code_analyzer import SharedCodeAnalyzerCheck

__all__ = [
    # Base classes
    "BaseCheck",
    "CheckPriority",
    "CheckStatus",
    "CheckResult",
    "ModuleCheckResult",
    "CheckRunner",
    # Check modules
    "APIEndpointsCheck",
    "APIConnectivityCheck",
    "DatabaseHealthCheck",
    "BackendHealthCheck",
    "PerformanceCheck",
    "CodeQualityCheck",
    "DummyDataDetectorCheck",
    "UIComponentsCheck",
    "DataSyncCheck",
    "SharedCodeAnalyzerCheck",
]
