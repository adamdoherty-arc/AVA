"""
Sentiment Analyst Agent
=======================

Analyzes market sentiment and positioning.
"""

from datetime import datetime
from typing import Any, Optional

from src.ava.strategies.base import StrategySetup, MarketContext
from ..orchestrator import AgentAnalysis, Conviction


class SentimentAnalyst:
    """
    Sentiment analysis agent that evaluates:
    - VIX and fear/greed indicators
    - Market breadth
    - Options flow and put/call ratios
    - Institutional positioning
    """

    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.name = "Sentiment Analyst"

    async def analyze(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> AgentAnalysis:
        """Perform sentiment analysis"""

        bullish_factors = []
        bearish_factors = []
        neutral_factors = []
        risks = []
        score = 0

        # VIX Analysis (Fear Index)
        vix = context.vix
        if vix > 0:
            if vix < 15:
                bullish_factors.append(f"Low VIX ({vix:.1f}) - market complacency")
                score += 10
                risks.append("Low VIX can precede volatility spikes")
            elif vix < 20:
                bullish_factors.append(f"Normal VIX ({vix:.1f}) - stable sentiment")
                score += 5
            elif vix < 25:
                neutral_factors.append(f"Elevated VIX ({vix:.1f}) - some fear")
            elif vix < 30:
                bearish_factors.append(f"High VIX ({vix:.1f}) - fear present")
                score -= 10
            else:
                bearish_factors.append(f"Very high VIX ({vix:.1f}) - extreme fear")
                risks.append("Extreme fear can signal capitulation or more downside")
                score -= 15

        # IV vs HV Analysis (Implied vs Realized)
        iv = context.implied_volatility
        hv = context.historical_volatility

        if iv > 0 and hv > 0:
            iv_premium = (iv - hv) / hv
            if iv_premium > 0.3:
                bullish_factors.append(f"IV premium high ({iv_premium:.0%}) - options expensive")
                score += 10 if setup.net_theta > 0 else -5
            elif iv_premium > 0.1:
                neutral_factors.append(f"Normal IV premium ({iv_premium:.0%})")
            elif iv_premium < -0.1:
                bearish_factors.append(f"IV discount ({iv_premium:.0%}) - options cheap")
                score += 5 if setup.net_theta < 0 else -5

        # Daily change sentiment
        daily_change = context.daily_change_pct
        if daily_change > 0.03:
            bullish_factors.append(f"Strong bullish momentum (+{daily_change:.1%})")
            score += 15
        elif daily_change > 0.01:
            bullish_factors.append(f"Positive sentiment (+{daily_change:.1%})")
            score += 5
        elif daily_change < -0.03:
            bearish_factors.append(f"Strong bearish momentum ({daily_change:.1%})")
            score -= 15
        elif daily_change < -0.01:
            bearish_factors.append(f"Negative sentiment ({daily_change:.1%})")
            score -= 5
        else:
            neutral_factors.append(f"Flat sentiment ({daily_change:.1%})")

        # Market regime inference
        if vix > 0 and context.iv_rank > 0:
            if vix < 18 and context.iv_rank < 30:
                bullish_factors.append("Low volatility regime - favorable for selling premium")
                score += 10
            elif vix > 25 and context.iv_rank > 70:
                neutral_factors.append("High volatility regime - elevated premiums")
                if setup.net_theta > 0:
                    bullish_factors.append("High IV environment good for credit strategies")
                    score += 10
                else:
                    risks.append("High IV can crush debit positions")

        # Overall market sentiment inference
        # In production, would integrate news sentiment, social media, etc.
        neutral_factors.append("Note: Full sentiment analysis requires external data feeds")

        # Determine conviction
        if abs(score) >= 25:
            conviction = Conviction.HIGH
        elif abs(score) >= 15:
            conviction = Conviction.MODERATE
        else:
            conviction = Conviction.LOW

        # Generate summary
        if score >= 20:
            summary = f"Market sentiment is BULLISH. Low fear and positive momentum support the trade."
        elif score > 5:
            summary = f"Market sentiment is SLIGHTLY BULLISH. Generally favorable conditions."
        elif score > -5:
            summary = f"Market sentiment is NEUTRAL. No strong directional bias."
        elif score > -20:
            summary = f"Market sentiment is SLIGHTLY BEARISH. Some caution warranted."
        else:
            summary = f"Market sentiment is BEARISH. High fear and negative momentum present."

        return AgentAnalysis(
            agent_name=self.name,
            agent_type="sentiment",
            timestamp=datetime.now(),
            summary=summary,
            score=score,
            conviction=conviction,
            bullish_factors=bullish_factors,
            bearish_factors=bearish_factors,
            neutral_factors=neutral_factors,
            risks=risks,
            confidence=0.6,  # Lower confidence without full sentiment data
            metrics={
                'vix': vix,
                'iv_hv_ratio': iv / hv if hv > 0 else 1,
                'daily_change': daily_change
            }
        )
