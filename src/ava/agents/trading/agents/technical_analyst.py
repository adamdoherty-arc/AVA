"""
Technical Analyst Agent
=======================

Analyzes price action, chart patterns, and technical indicators.
"""

from datetime import datetime
from typing import Any, Optional
import numpy as np

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class TechnicalAnalyst:
    """
    Technical analysis agent that evaluates:
    - Price trends and momentum
    - Support/resistance levels
    - Chart patterns
    - Technical indicators (RSI, MACD, etc.)
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Technical Analyst"

    async def analyze(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> AgentAnalysis:
        """Perform technical analysis"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        risks = []
        score = 0

        # Price trend analysis
        price_change = context.daily_change_pct
        if price_change > 0.02:
            bullish_factors.append(f"Strong uptrend: +{price_change:.1%} today")
            score += 15
        elif price_change > 0:
            bullish_factors.append(f"Mild bullish: +{price_change:.1%}")
            score += 5
        elif price_change < -0.02:
            bearish_factors.append(f"Strong downtrend: {price_change:.1%} today")
            score -= 15
        elif price_change < 0:
            bearish_factors.append(f"Mild bearish: {price_change:.1%}")
            score -= 5
        else:
            neutral_factors.append("Flat price action")

        # RSI analysis
        rsi = context.rsi_14
        if rsi > 70:
            bearish_factors.append(f"Overbought RSI: {rsi:.0f}")
            score -= 10
            risks.append("RSI overbought - potential reversal")
        elif rsi < 30:
            bullish_factors.append(f"Oversold RSI: {rsi:.0f}")
            score += 10
        elif 40 <= rsi <= 60:
            neutral_factors.append(f"Neutral RSI: {rsi:.0f}")
        else:
            neutral_factors.append(f"RSI: {rsi:.0f}")

        # Moving average analysis
        if context.sma_50 > 0 and context.sma_200 > 0:
            if context.current_price > context.sma_50 > context.sma_200:
                bullish_factors.append("Price above rising MAs (bullish alignment)")
                score += 15
            elif context.current_price < context.sma_50 < context.sma_200:
                bearish_factors.append("Price below falling MAs (bearish alignment)")
                score -= 15
            elif context.current_price > context.sma_200:
                bullish_factors.append("Price above 200 SMA (long-term bullish)")
                score += 5
            else:
                bearish_factors.append("Price below 200 SMA (long-term bearish)")
                score -= 5

        # 52-week range analysis
        if context.high_52w > 0 and context.low_52w > 0:
            range_position = (context.current_price - context.low_52w) / (context.high_52w - context.low_52w)
            if range_position > 0.9:
                bearish_factors.append(f"Near 52-week high ({range_position:.0%} of range)")
                risks.append("Near resistance at 52-week high")
                score -= 5
            elif range_position < 0.1:
                bullish_factors.append(f"Near 52-week low ({range_position:.0%} of range)")
                score += 5
            else:
                neutral_factors.append(f"Mid-range position: {range_position:.0%}")

        # Volume analysis
        if context.volume > 0 and context.avg_volume > 0:
            vol_ratio = context.volume / context.avg_volume
            if vol_ratio > 1.5:
                if price_change > 0:
                    bullish_factors.append(f"High volume on up day ({vol_ratio:.1f}x avg)")
                    score += 10
                else:
                    bearish_factors.append(f"High volume on down day ({vol_ratio:.1f}x avg)")
                    score -= 10
            elif vol_ratio < 0.5:
                neutral_factors.append("Low volume - weak conviction")

        # Determine conviction
        if abs(score) >= 30:
            conviction = Conviction.HIGH
        elif abs(score) >= 15:
            conviction = Conviction.MODERATE
        else:
            conviction = Conviction.LOW

        # Generate summary
        if score > 20:
            summary = f"Technical outlook is BULLISH for {context.symbol}. Price action and indicators suggest upside potential."
        elif score < -20:
            summary = f"Technical outlook is BEARISH for {context.symbol}. Price action and indicators suggest downside risk."
        else:
            summary = f"Technical outlook is NEUTRAL for {context.symbol}. Mixed signals from price action and indicators."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="technical",
            timestamp=datetime.now(),
            summary=summary,
            score=score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            risks=risks,
            confidence=0.7 + (len(bullish_factors) + len(bearish_factors)) * 0.02,
            metrics={
                'rsi': context.rsi_14,
                'sma_50': context.sma_50,
                'sma_200': context.sma_200,
                'price_change': price_change
            }
        )
