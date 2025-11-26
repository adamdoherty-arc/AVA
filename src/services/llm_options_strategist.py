"""
LLM Options Strategist Service
Provides AI-powered options trading strategies using Local LLM
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from src.magnus_local_llm import get_magnus_llm, TaskComplexity

logger = logging.getLogger(__name__)


class MarketOutlook(str, Enum):
    """Market outlook options for strategy generation"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"


class RiskTolerance(str, Enum):
    """Risk tolerance levels for strategy generation"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class LLMOptionsStrategist:
    """
    Generates options trading strategies based on market data and risk profile
    """

    def __init__(self):
        self.llm = get_magnus_llm()

    def generate_strategy(self, 
                         symbol: str, 
                         market_data: Dict[str, Any],
                         risk_profile: str = "moderate",
                         outlook: str = "neutral") -> Dict[str, Any]:
        """
        Generate options strategy for a symbol
        
        Args:
            symbol: Stock symbol
            market_data: Dictionary containing price, IV, Greeks, etc.
            risk_profile: 'conservative', 'moderate', 'aggressive'
            outlook: 'bullish', 'bearish', 'neutral', 'volatile'
            
        Returns:
            Dictionary with strategy recommendation and reasoning
        """
        try:
            # Format context
            price = market_data.get('price', 0)
            iv = market_data.get('iv', 0)
            iv_rank = market_data.get('iv_rank', 0)
            
            prompt = f"""Generate an options trading strategy for {symbol}:

CONTEXT:
- Current Price: ${price:.2f}
- Implied Volatility: {iv:.1%}
- IV Rank: {iv_rank}
- Market Outlook: {outlook.upper()}
- Risk Profile: {risk_profile.upper()}

Please recommend a specific options strategy (e.g., Iron Condor, Credit Spread, Covered Call, etc.) that aligns with these parameters.

Include:
1. Strategy Name
2. Specific Strikes (relative to current price) and Expiration
3. Entry Criteria
4. Exit Plan (Profit Target & Stop Loss)
5. Risk/Reward Analysis
6. Why this strategy fits the current IV environment

Format the response as JSON with keys: 'strategy_name', 'setup', 'entry', 'exit', 'risk_analysis', 'reasoning'.
"""
            
            response = self.llm.query(
                prompt=prompt,
                complexity=TaskComplexity.CREATIVE,
                use_trading_context=True,
                max_tokens=1500
            )
            
            # Parse JSON response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                elif "{" in response and "}" in response:
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    json_str = response[start:end]
                    return json.loads(json_str)
                else:
                    return {
                        'strategy_name': 'Custom Analysis',
                        'reasoning': response
                    }
            except json.JSONDecodeError:
                return {
                    'strategy_name': 'Error parsing',
                    'reasoning': response
                }
                
        except Exception as e:
            logger.error(f"Error generating strategy for {symbol}: {e}")
            return {'error': str(e)}

    def analyze_existing_position(self, position: Dict[str, Any]) -> str:
        """Analyze an existing options position and suggest adjustments"""
        # Implementation for position adjustment analysis
        pass
