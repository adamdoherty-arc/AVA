"""
Magnus Continuous QA & Enhancement System

Runs every 20 minutes to check and improve the Magnus codebase.
Auto-starts as a Windows Service.

Components:
- AccomplishmentsTracker: Logs all accomplishments (JSONL, append-only)
- QARunner: Main entry point for QA cycles
- HealthScorer: Calculates multi-dimensional health scores
- EnhancementEngine: Proactively improves code
- PatternLearner: Learns from recurring issues
- TelegramNotifier: Sends notifications via AVA
- ContinuousQAIntegration: Integrates with Main Orchestrator

Check Modules:
- APIEndpointsCheck: Tests FastAPI endpoints
- CodeQualityCheck: Checks code rules and quality
- DummyDataDetectorCheck: Detects mock/dummy data
- UIComponentsCheck: Tests React frontend
- DataSyncCheck: Validates data freshness
- SharedCodeAnalyzerCheck: Finds duplicate code
"""

from .accomplishments_tracker import AccomplishmentsTracker, Accomplishment
from .qa_runner import QARunner
from .health_scorer import HealthScorer
from .enhancement_engine import EnhancementEngine
from .pattern_learner import PatternLearner
from .telegram_notifier import QATelegramNotifier as TelegramNotifier
from .orchestrator_integration import ContinuousQAIntegration, integrate_with_orchestrator

__all__ = [
    # Core components
    "AccomplishmentsTracker",
    "Accomplishment",
    "QARunner",
    "HealthScorer",
    "EnhancementEngine",
    "PatternLearner",
    "TelegramNotifier",
    # Integration
    "ContinuousQAIntegration",
    "integrate_with_orchestrator",
]

__version__ = "1.0.0"
