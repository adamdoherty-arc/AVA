"""Analysis Agents Package

Modern AI-powered analysis agents with LLM integration
for comprehensive trading analysis.
"""

# Legacy agents
from .fundamental_agent import FundamentalAnalysisAgent
from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .supply_demand_agent import SupplyDemandAgent

# Modern AI-powered agents
from .ai_fundamental_agent import AIFundamentalAgent, QuickFundamentalChecker
from .ai_technical_agent import AITechnicalAgent, QuickTechnicalCalculator
from .ai_sentiment_agent import AISentimentAgent, QuickSentimentChecker
from .ai_options_flow_agent import AIOptionsFlowAgent, QuickFlowAnalyzer
from .ai_supply_demand_agent import AISupplyDemandAgent, QuickZoneFinder

__all__ = [
    # Legacy
    "FundamentalAnalysisAgent",
    "TechnicalAnalysisAgent",
    "SentimentAnalysisAgent",
    "SupplyDemandAgent",
    # Modern AI agents
    "AIFundamentalAgent",
    "QuickFundamentalChecker",
    "AITechnicalAgent",
    "QuickTechnicalCalculator",
    "AISentimentAgent",
    "QuickSentimentChecker",
    "AIOptionsFlowAgent",
    "QuickFlowAnalyzer",
    "AISupplyDemandAgent",
    "QuickZoneFinder",
]

