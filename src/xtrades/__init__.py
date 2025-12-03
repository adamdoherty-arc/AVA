"""
Xtrades Modern Module - AI-Powered Trade Alert System
======================================================

Modern, async-first implementation with:
- Pydantic models for data validation
- AI-powered trade analysis using LangChain
- Async/concurrent processing
- Tenacity retry with exponential backoff
- Structured logging with structlog
"""

from .models import (
    XtradeProfile,
    XtradeAlert,
    TradeSignal,
    SyncResult,
    AIAnalysis
)
from .scraper import ModernXtradesScraper
from .analyzer import AITradeAnalyzer
from .sync_service import ModernSyncService

__all__ = [
    'XtradeProfile',
    'XtradeAlert',
    'TradeSignal',
    'SyncResult',
    'AIAnalysis',
    'ModernXtradesScraper',
    'AITradeAnalyzer',
    'ModernSyncService'
]

__version__ = '2.0.0'
