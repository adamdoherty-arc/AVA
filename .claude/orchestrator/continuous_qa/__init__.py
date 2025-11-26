"""
Magnus Continuous QA & Enhancement System

Runs every 20 minutes to check and improve the Magnus codebase.
Auto-starts as a Windows Service.
"""

from .accomplishments_tracker import AccomplishmentsTracker, Accomplishment
from .qa_runner import QARunner
from .health_scorer import HealthScorer

__all__ = [
    "AccomplishmentsTracker",
    "Accomplishment",
    "QARunner",
    "HealthScorer",
]

__version__ = "1.0.0"
