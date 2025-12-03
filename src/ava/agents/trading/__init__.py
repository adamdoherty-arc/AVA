"""Trading Agents Package"""

# Legacy agents
from .market_data_agent import MarketDataAgent
from .options_analysis_agent import OptionsAnalysisAgent
from .strategy_agent import StrategyAgent
from .risk_management_agent import RiskManagementAgent
from .portfolio_agent import PortfolioAgent

# Modern AI-powered agents
from .strategy_recommendation_agent import StrategyRecommendationAgent, QuickStrategySelector
from .ai_risk_agent import AIRiskManagementAgent, QuickRiskChecker
from .ai_options_analysis_agent import AIOptionsAnalysisAgent, QuickOptionsScorer
from .ai_premium_scanner_agent import AIPremiumScannerAgent, QuickPremiumScanner
from .ai_earnings_agent import AIEarningsAgent, QuickEarningsChecker

__all__ = [
    # Legacy
    "MarketDataAgent",
    "OptionsAnalysisAgent",
    "StrategyAgent",
    "RiskManagementAgent",
    "PortfolioAgent",
    # Modern AI agents
    "StrategyRecommendationAgent",
    "QuickStrategySelector",
    "AIRiskManagementAgent",
    "QuickRiskChecker",
    "AIOptionsAnalysisAgent",
    "QuickOptionsScorer",
    "AIPremiumScannerAgent",
    "QuickPremiumScanner",
    "AIEarningsAgent",
    "QuickEarningsChecker",
]

