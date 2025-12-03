"""
Fundamental Analyst Agent
Analyzes company financials, earnings, valuation metrics
"""

import logging
from typing import Optional
import yfinance as yf
from datetime import datetime

from src.agents.ai_research.models import FundamentalAnalysis

logger = logging.getLogger(__name__)


class FundamentalAgent:
    """
    Fundamental analysis specialist

    Analyzes:
    - Revenue growth and earnings trends
    - Valuation metrics (P/E, P/B, etc.)
    - Financial health (debt, cash flow, ROE)
    - Earnings estimates and next earnings date
    """

    def __init__(self) -> None:
        self.api_calls = 0

    async def analyze(self, symbol: str) -> FundamentalAnalysis:
        """
        Perform fundamental analysis

        Args:
            symbol: Stock ticker symbol

        Returns:
            FundamentalAnalysis object
        """
        self.api_calls = 0
        logger.info(f"Starting fundamental analysis for {symbol}")

        try:
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            self.api_calls += 1

            info = ticker.info
            financials = ticker.financials
            self.api_calls += 1

            # Extract key metrics
            revenue_growth_yoy = self._calculate_revenue_growth(financials)
            earnings_beat_streak = self._get_earnings_beat_streak(ticker)

            pe_ratio = info.get('trailingPE', 0.0) or info.get('forwardPE', 0.0)
            sector = info.get('sector', 'Unknown')
            sector_avg_pe = self._get_sector_avg_pe(sector)

            pb_ratio = info.get('priceToBook', 0.0)
            debt_to_equity = info.get('debtToEquity', 0.0)
            roe = info.get('returnOnEquity', 0.0)
            free_cash_flow = info.get('freeCashflow', 0.0)
            dividend_yield = info.get('dividendYield', 0.0) or 0.0

            # Valuation assessment
            valuation = self._assess_valuation(pe_ratio, sector_avg_pe, pb_ratio)

            # Identify strengths and risks
            strengths = self._identify_strengths(info, revenue_growth_yoy, roe)
            risks = self._identify_risks(info, debt_to_equity)

            # Calculate score (0-100)
            score = self._calculate_score(
                revenue_growth_yoy, earnings_beat_streak, pe_ratio, sector_avg_pe,
                debt_to_equity, roe, free_cash_flow
            )

            # Get next earnings date
            next_earnings = self._get_next_earnings_date(info)

            return FundamentalAnalysis(
                score=score,
                revenue_growth_yoy=revenue_growth_yoy,
                earnings_beat_streak=earnings_beat_streak,
                pe_ratio=pe_ratio,
                sector_avg_pe=sector_avg_pe,
                pb_ratio=pb_ratio,
                debt_to_equity=debt_to_equity,
                roe=roe,
                free_cash_flow=free_cash_flow,
                dividend_yield=dividend_yield,
                valuation_assessment=valuation,
                key_strengths=strengths,
                key_risks=risks,
                next_earnings_date=next_earnings,
                analyst_estimates=None  # TODO: Add analyst estimates
            )

        except Exception as e:
            logger.error(f"Fundamental analysis failed for {symbol}: {str(e)}")
            raise

    def _calculate_revenue_growth(self, financials) -> float:
        """Calculate YoY revenue growth"""
        try:
            if financials is None or financials.empty:
                return 0.0

            if 'Total Revenue' in financials.index:
                revenues = financials.loc['Total Revenue']
                if len(revenues) >= 2:
                    latest = revenues.iloc[0]
                    previous = revenues.iloc[1]
                    if previous != 0:
                        return (latest - previous) / previous

            return 0.0
        except:
            return 0.0

    def _get_earnings_beat_streak(self, ticker) -> int:
        """Get consecutive earnings beat streak"""
        try:
            # TODO: Implement actual earnings beat tracking
            # For now, return 0 (requires earnings calendar API)
            return 0
        except:
            return 0

    def _get_sector_avg_pe(self, sector: str) -> float:
        """Get average P/E for sector"""
        # Historical sector averages (approximate)
        sector_pes = {
            'Technology': 28.0,
            'Healthcare': 24.0,
            'Financial Services': 15.0,
            'Consumer Cyclical': 20.0,
            'Industrials': 22.0,
            'Energy': 18.0,
            'Utilities': 18.0,
            'Real Estate': 35.0,
            'Consumer Defensive': 22.0,
            'Communication Services': 20.0,
            'Basic Materials': 16.0
        }
        return sector_pes.get(sector, 20.0)

    def _assess_valuation(self, pe: float, sector_pe: float, pb: float) -> str:
        """Assess if stock is undervalued, fairly valued, or overvalued"""
        if pe == 0 or sector_pe == 0:
            return "Unable to determine valuation"

        pe_discount = (pe - sector_pe) / sector_pe

        if pe_discount < -0.25:
            return "Significantly undervalued vs sector"
        elif pe_discount < -0.10:
            return "Moderately undervalued vs sector"
        elif pe_discount < 0.10:
            return "Fairly valued vs sector"
        elif pe_discount < 0.25:
            return "Moderately overvalued vs sector"
        else:
            return "Significantly overvalued vs sector"

    def _identify_strengths(self, info: dict, revenue_growth: float, roe: float) -> list:
        """Identify company strengths"""
        strengths = []

        if revenue_growth > 0.20:
            strengths.append(f"Strong revenue growth ({revenue_growth:.1%} YoY)")
        elif revenue_growth > 0.10:
            strengths.append(f"Solid revenue growth ({revenue_growth:.1%} YoY)")

        if roe > 0.15:
            strengths.append(f"High return on equity ({roe:.1%})")

        if info.get('profitMargins', 0) > 0.15:
            strengths.append(f"Strong profit margins ({info['profitMargins']:.1%})")

        cash = info.get('totalCash', 0)
        if cash > 1_000_000_000:
            strengths.append(f"Strong cash position (${cash/1e9:.1f}B)")

        if not strengths:
            strengths.append("Stable operations")

        return strengths[:5]  # Max 5 strengths

    def _identify_risks(self, info: dict, debt_to_equity: float) -> list:
        """Identify company risks"""
        risks = []

        if debt_to_equity > 2.0:
            risks.append(f"High debt-to-equity ratio ({debt_to_equity:.2f})")

        if info.get('profitMargins', 0) < 0:
            risks.append("Negative profit margins")

        quick_ratio = info.get('quickRatio', 1.0)
        if quick_ratio < 0.5:
            risks.append(f"Low liquidity (quick ratio: {quick_ratio:.2f})")

        if info.get('beta', 1.0) > 1.5:
            risks.append(f"High volatility (beta: {info['beta']:.2f})")

        if not risks:
            risks.append("Normal market risks")

        return risks[:5]  # Max 5 risks

    def _calculate_score(
        self,
        revenue_growth: float,
        earnings_beat_streak: int,
        pe: float,
        sector_pe: float,
        debt_to_equity: float,
        roe: float,
        free_cash_flow: float
    ) -> int:
        """Calculate fundamental score 0-100"""
        score = 50  # Start at neutral

        # Revenue growth (+/- 15 points)
        if revenue_growth > 0.20:
            score += 15
        elif revenue_growth > 0.10:
            score += 10
        elif revenue_growth > 0.05:
            score += 5
        elif revenue_growth < -0.10:
            score -= 15
        elif revenue_growth < -0.05:
            score -= 10

        # Earnings beat streak (+10 points)
        score += min(earnings_beat_streak * 2, 10)

        # Valuation (+/- 10 points)
        if pe > 0 and sector_pe > 0:
            pe_ratio = pe / sector_pe
            if pe_ratio < 0.8:
                score += 10
            elif pe_ratio < 0.9:
                score += 5
            elif pe_ratio > 1.3:
                score -= 10
            elif pe_ratio > 1.2:
                score -= 5

        # Debt level (+/- 10 points)
        if debt_to_equity < 0.5:
            score += 10
        elif debt_to_equity < 1.0:
            score += 5
        elif debt_to_equity > 2.5:
            score -= 10
        elif debt_to_equity > 2.0:
            score -= 5

        # ROE (+/- 10 points)
        if roe > 0.20:
            score += 10
        elif roe > 0.15:
            score += 5
        elif roe < 0.05:
            score -= 10
        elif roe < 0.10:
            score -= 5

        # Free cash flow (+/- 5 points)
        if free_cash_flow > 0:
            score += 5
        elif free_cash_flow < 0:
            score -= 5

        # Clamp to 0-100
        return max(0, min(100, score))

    def _get_next_earnings_date(self, info: dict) -> Optional[str]:
        """Get next earnings date"""
        try:
            # Try to get from earnings dates
            earnings_date = info.get('earningsDate')
            if earnings_date:
                if isinstance(earnings_date, list) and len(earnings_date) > 0:
                    return earnings_date[0].strftime('%Y-%m-%d')
                elif hasattr(earnings_date, 'strftime'):
                    return earnings_date.strftime('%Y-%m-%d')

            return None
        except:
            return None

    def get_api_call_count(self) -> int:
        """Get number of API calls made"""
        return self.api_calls
