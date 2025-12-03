"""
SpecAgents - Feature-specific testing agents

Each agent deeply understands its feature and performs:
- API endpoint testing
- UI component testing (Playwright)
- Business logic validation
- Data consistency checks
"""

# Tier 1 - Critical Features
from .positions_spec_agent import PositionsSpecAgent
from .dashboard_spec_agent import DashboardSpecAgent
from .premium_scanner_spec_agent import PremiumScannerSpecAgent
from .options_analysis_spec_agent import OptionsAnalysisSpecAgent
from .game_cards_spec_agent import GameCardsSpecAgent

# Tier 2 - Important Features
from .ava_chatbot_spec_agent import AVAChatbotSpecAgent
from .kalshi_markets_spec_agent import KalshiMarketsSpecAgent
from .earnings_calendar_spec_agent import EarningsCalendarSpecAgent
from .xtrades_watchlists_spec_agent import XTradesWatchlistsSpecAgent
from .dte_scanner_spec_agent import DTEScannerSpecAgent

# Tier 3 - Additional Features
from .best_bets_spec_agent import BestBetsSpecAgent
from .technical_indicators_spec_agent import TechnicalIndicatorsSpecAgent
from .calendar_spreads_spec_agent import CalendarSpreadsSpecAgent
from .signal_dashboard_spec_agent import SignalDashboardSpecAgent
from .qa_dashboard_spec_agent import QADashboardSpecAgent
from .research_spec_agent import ResearchSpecAgent

__all__ = [
    # Tier 1
    'PositionsSpecAgent',
    'DashboardSpecAgent',
    'PremiumScannerSpecAgent',
    'OptionsAnalysisSpecAgent',
    'GameCardsSpecAgent',
    # Tier 2
    'AVAChatbotSpecAgent',
    'KalshiMarketsSpecAgent',
    'EarningsCalendarSpecAgent',
    'XTradesWatchlistsSpecAgent',
    'DTEScannerSpecAgent',
    # Tier 3
    'BestBetsSpecAgent',
    'TechnicalIndicatorsSpecAgent',
    'CalendarSpreadsSpecAgent',
    'SignalDashboardSpecAgent',
    'QADashboardSpecAgent',
    'ResearchSpecAgent',
]
