"""
Fundamental Analyst Agent
=========================

Analyzes company fundamentals and valuation.
"""

from datetime import datetime
from typing import Any, Optional

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class FundamentalAnalyst:
    """
    Fundamental analysis agent that evaluates:
    - Earnings and revenue trends
    - Valuation metrics (P/E, P/S, etc.)
    - Balance sheet health
    - Competitive position
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Fundamental Analyst"

    async def analyze(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> AgentAnalysis:
        """Perform fundamental analysis"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        risks = []
        score = 0

        symbol = context.symbol.upper()

        # Simplified fundamental analysis based on available context
        # In production, would fetch from financial APIs

        # Large cap vs small cap consideration
        large_caps = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
        if symbol in large_caps:
            bullish_factors.append("Large-cap with strong market position")
            score += 10
        else:
            neutral_factors.append("Mid/small cap - higher growth potential but more risk")

        # ETF vs individual stock
        etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'XLE', 'XLK', 'XLV']
        if symbol in etfs:
            bullish_factors.append("Diversified ETF reduces single-stock risk")
            score += 5
        else:
            risks.append("Single stock concentration risk")

        # Earnings proximity analysis
        if context.days_to_earnings is not None:
            if context.days_to_earnings < 7:
                bearish_factors.append(f"Earnings in {context.days_to_earnings} days - high IV crush risk")
                risks.append("Imminent earnings creates uncertainty")
                score -= 20
            elif context.days_to_earnings < 14:
                bearish_factors.append(f"Earnings approaching ({context.days_to_earnings} days)")
                risks.append("Pre-earnings volatility")
                score -= 10
            elif context.days_to_earnings < 30:
                neutral_factors.append(f"Earnings in {context.days_to_earnings} days - monitor")
                score -= 5
            else:
                bullish_factors.append(f"Clear of earnings ({context.days_to_earnings} days away)")
                score += 10

        # Sector analysis (simplified)
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'QQQ']
        defensive_stocks = ['JNJ', 'PG', 'KO', 'PEP', 'XLV', 'XLP']

        if symbol in tech_stocks:
            neutral_factors.append("Technology sector - growth focus")
            if context.vix and context.vix < 20:
                bullish_factors.append("Low VIX favorable for growth stocks")
                score += 5
        elif symbol in defensive_stocks:
            bullish_factors.append("Defensive sector - stable in volatility")
            score += 5

        # Determine conviction
        if abs(score) >= 20:
            conviction = Conviction.HIGH
        elif abs(score) >= 10:
            conviction = Conviction.MODERATE
        else:
            conviction = Conviction.LOW

        # Generate summary
        if score > 15:
            summary = f"Fundamentals are SUPPORTIVE for {symbol}. Company/sector shows strength."
        elif score < -15:
            summary = f"Fundamentals show CONCERNS for {symbol}. Risks identified."
        else:
            summary = f"Fundamentals are NEUTRAL for {symbol}. Mixed signals."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="fundamental",
            timestamp=datetime.now(),
            summary=summary,
            score=score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            risks=risks,
            confidence=0.6,  # Lower confidence without actual financial data
            metrics={
                'days_to_earnings': context.days_to_earnings,
                'is_etf': symbol in etfs,
                'is_large_cap': symbol in large_caps
            }
        )
