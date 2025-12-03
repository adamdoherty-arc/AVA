"""
Check modules for Magnus QA system.

Available checks:
- APIEndpointsCheck: Tests FastAPI endpoints for availability and data quality
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
    "CodeQualityCheck",
    "DummyDataDetectorCheck",
    "UIComponentsCheck",
    "DataSyncCheck",
    "SharedCodeAnalyzerCheck",
]
