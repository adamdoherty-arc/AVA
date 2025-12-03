"""
Sector Metrics Calculator
Comprehensive sector analysis with momentum, relative strength, breadth, and technical indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SectorMetricsCalculator:
    """
    Advanced sector metrics calculator implementing:
    - Multi-period momentum scoring
    - Relative strength analysis (RS-Ratio, RS-Momentum)
    - Breadth indicators (A/D, % above MAs)
    - Risk-adjusted returns (Sharpe, Beta)
    - Sector rotation signals
    """

    def __init__(self) -> None:
        """Initialize the sector metrics calculator"""
        self.risk_free_rate = 0.045  # 4.5% (3-month T-Bill approximate)

        # GICS Sector mapping to ETFs
        self.sector_etf_map = {
            'Information Technology': 'XLK',
            'Health Care': 'XLV',
            'Financials': 'XLF',
            'Communication Services': 'XLC',
            'Consumer Discretionary': 'XLY',
            'Industrials': 'XLI',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU'
        }

        # Sector characteristics
        self.sector_profiles = {
            'Information Technology': {
                'typical_beta': 1.2,
                'cyclicality': 'Pro-Cyclical',
                'defensive': False,
                'avg_dividend_yield': 1.5,
                'wheel_tier': 1,
                'optimal_dte': 30
            },
            'Health Care': {
                'typical_beta': 0.9,
                'cyclicality': 'Defensive',
                'defensive': True,
                'avg_dividend_yield': 2.5,
                'wheel_tier': 1,
                'optimal_dte': 45
            },
            'Financials': {
                'typical_beta': 1.1,
                'cyclicality': 'Highly Cyclical',
                'defensive': False,
                'avg_dividend_yield': 3.0,
                'wheel_tier': 2,
                'optimal_dte': 45
            },
            'Communication Services': {
                'typical_beta': 1.1,
                'cyclicality': 'Mixed',
                'defensive': False,
                'avg_dividend_yield': 1.5,
                'wheel_tier': 1,
                'optimal_dte': 30
            },
            'Consumer Discretionary': {
                'typical_beta': 1.2,
                'cyclicality': 'Highly Cyclical',
                'defensive': False,
                'avg_dividend_yield': 1.5,
                'wheel_tier': 1,
                'optimal_dte': 30
            },
            'Industrials': {
                'typical_beta': 1.1,
                'cyclicality': 'Cyclical',
                'defensive': False,
                'avg_dividend_yield': 2.5,
                'wheel_tier': 2,
                'optimal_dte': 45
            },
            'Consumer Staples': {
                'typical_beta': 0.7,
                'cyclicality': 'Defensive',
                'defensive': True,
                'avg_dividend_yield': 3.5,
                'wheel_tier': 3,
                'optimal_dte': 60
            },
            'Energy': {
                'typical_beta': 1.3,
                'cyclicality': 'Highly Cyclical',
                'defensive': False,
                'avg_dividend_yield': 4.0,
                'wheel_tier': 2,
                'optimal_dte': 30
            },
            'Materials': {
                'typical_beta': 1.2,
                'cyclicality': 'Cyclical',
                'defensive': False,
                'avg_dividend_yield': 2.5,
                'wheel_tier': 2,
                'optimal_dte': 45
            },
            'Real Estate': {
                'typical_beta': 1.0,
                'cyclicality': 'Rate Sensitive',
                'defensive': True,
                'avg_dividend_yield': 4.0,
                'wheel_tier': 3,
                'optimal_dte': 60
            },
            'Utilities': {
                'typical_beta': 0.6,
                'cyclicality': 'Defensive',
                'defensive': True,
                'avg_dividend_yield': 3.5,
                'wheel_tier': 3,
                'optimal_dte': 60
            }
        }

    # ========================================================================
    # MOMENTUM INDICATORS
    # ========================================================================

    def calculate_multi_period_momentum(
        self,
        returns_1m: float,
        returns_3m: float,
        returns_6m: float
    ) -> float:
        """
        Calculate multi-period momentum score with recent bias weighting

        Formula: Momentum = (0.5 × 1M) + (0.3 × 3M) + (0.2 × 6M)

        Args:
            returns_1m: 1-month return (%)
            returns_3m: 3-month return (%)
            returns_6m: 6-month return (%)

        Returns:
            Weighted momentum score
        """
        momentum_score = (0.5 * returns_1m) + (0.3 * returns_3m) + (0.2 * returns_6m)
        return momentum_score

    def calculate_rate_of_change(
        self,
        current_price: float,
        price_n_periods_ago: float
    ) -> float:
        """
        Calculate Rate of Change (ROC)

        Formula: ROC = ((Current - Past) / Past) × 100

        Args:
            current_price: Current price
            price_n_periods_ago: Price N periods ago (typically 90 days)

        Returns:
            Rate of change percentage
        """
        if price_n_periods_ago == 0:
            return 0.0

        roc = ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100
        return roc

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            prices: Price series
            period: RSI period (default 14)

        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    # ========================================================================
    # RELATIVE STRENGTH INDICATORS
    # ========================================================================

    def calculate_rs_ratio(
        self,
        sector_price: float,
        spy_price: float
    ) -> float:
        """
        Calculate RS-Ratio (Relative Strength vs S&P 500)

        Formula: RS-Ratio = (Sector_Price / SPY_Price) × 100

        Values > 100: Outperforming market
        Values < 100: Underperforming market

        Args:
            sector_price: Current sector ETF price
            spy_price: Current SPY price

        Returns:
            RS-Ratio value
        """
        if spy_price == 0:
            return 100.0

        rs_ratio = (sector_price / spy_price) * 100
        return rs_ratio

    def calculate_rs_momentum(
        self,
        current_rs_ratio: float,
        rs_ratio_20d_ago: float
    ) -> float:
        """
        Calculate RS-Momentum (leading indicator for rotation)

        Formula: RS-Momentum = Current_RS_Ratio - RS_Ratio_20_Days_Ago

        Positive values: Improving relative strength
        Negative values: Declining relative strength

        Args:
            current_rs_ratio: Current RS-Ratio
            rs_ratio_20d_ago: RS-Ratio 20 days ago

        Returns:
            RS-Momentum value
        """
        rs_momentum = current_rs_ratio - rs_ratio_20d_ago
        return rs_momentum

    def calculate_rrg_quadrant(
        self,
        rs_ratio: float,
        rs_momentum: float
    ) -> str:
        """
        Determine RRG (Relative Rotation Graph) quadrant

        Quadrants:
        - Leading: High RS-Ratio, Positive RS-Momentum (HOLD/BUY)
        - Improving: Low RS-Ratio, Positive RS-Momentum (BUY)
        - Lagging: Low RS-Ratio, Negative RS-Momentum (SELL)
        - Weakening: High RS-Ratio, Negative RS-Momentum (REDUCE)

        Args:
            rs_ratio: RS-Ratio value
            rs_momentum: RS-Momentum value

        Returns:
            Quadrant name
        """
        if rs_ratio > 100 and rs_momentum > 0:
            return "Leading"
        elif rs_ratio <= 100 and rs_momentum > 0:
            return "Improving"
        elif rs_ratio <= 100 and rs_momentum <= 0:
            return "Lagging"
        else:  # rs_ratio > 100 and rs_momentum <= 0
            return "Weakening"

    # ========================================================================
    # BREADTH INDICATORS
    # ========================================================================

    def calculate_advance_decline_ratio(
        self,
        advancing_stocks: int,
        declining_stocks: int
    ) -> float:
        """
        Calculate Advance/Decline Ratio

        Formula: A/D_Ratio = Advancing_Stocks / Declining_Stocks

        > 1.5: Strong breadth
        0.5-1.5: Neutral
        < 0.5: Weak breadth

        Args:
            advancing_stocks: Number of stocks rising
            declining_stocks: Number of stocks falling

        Returns:
            A/D Ratio
        """
        if declining_stocks == 0:
            return 999.0 if advancing_stocks > 0 else 1.0

        ad_ratio = advancing_stocks / declining_stocks
        return ad_ratio

    def calculate_breadth_score(
        self,
        stocks_above_50ma: int,
        total_stocks: int
    ) -> float:
        """
        Calculate breadth score (% of stocks above 50-day MA)

        > 70%: Strong sector health
        30-70%: Neutral
        < 30%: Weak sector health

        Args:
            stocks_above_50ma: Stocks above 50-day moving average
            total_stocks: Total stocks in sector

        Returns:
            Breadth score (0-100)
        """
        if total_stocks == 0:
            return 50.0

        breadth_score = (stocks_above_50ma / total_stocks) * 100
        return breadth_score

    # ========================================================================
    # RISK-ADJUSTED RETURNS
    # ========================================================================

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return)

        Formula: Sharpe = (Return - Risk_Free_Rate) / StdDev

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (default: 4.5% annual)

        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) < 2:
            return 0.0

        # Annualize returns (assuming daily)
        annual_return = returns.mean() * 252
        annual_std = returns.std() * np.sqrt(252)

        if annual_std == 0:
            return 0.0

        sharpe = (annual_return - risk_free_rate) / annual_std
        return sharpe

    def calculate_beta(
        self,
        sector_returns: pd.Series,
        market_returns: pd.Series
    ) -> float:
        """
        Calculate Beta (volatility relative to market)

        Formula: Beta = Covariance(Sector, Market) / Variance(Market)

        > 1: More volatile than market
        = 1: Same volatility as market
        < 1: Less volatile than market

        Args:
            sector_returns: Sector return series
            market_returns: Market (SPY) return series

        Returns:
            Beta coefficient
        """
        if len(sector_returns) < 2 or len(market_returns) < 2:
            return 1.0

        # Align series
        aligned_data = pd.DataFrame({
            'sector': sector_returns,
            'market': market_returns
        }).dropna()

        if len(aligned_data) < 2:
            return 1.0

        # Calculate covariance and variance
        covariance = aligned_data['sector'].cov(aligned_data['market'])
        market_variance = aligned_data['market'].var()

        if market_variance == 0:
            return 1.0

        beta = covariance / market_variance
        return beta

    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Calculate Sortino Ratio (downside risk-adjusted return)

        Like Sharpe but only penalizes downside volatility

        Args:
            returns: Return series
            risk_free_rate: Risk-free rate

        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        if len(returns) < 2:
            return 0.0

        # Annualize returns
        annual_return = returns.mean() * 252

        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0

        downside_std = downside_returns.std() * np.sqrt(252)

        if downside_std == 0:
            return 0.0

        sortino = (annual_return - risk_free_rate) / downside_std
        return sortino

    # ========================================================================
    # SECTOR ROTATION SIGNALS
    # ========================================================================

    def generate_rotation_signal(
        self,
        sector: str,
        momentum_score: float,
        rs_ratio: float,
        rs_momentum: float,
        breadth_score: float
    ) -> Dict[str, any]:
        """
        Generate comprehensive sector rotation signal

        Args:
            sector: Sector name
            momentum_score: Multi-period momentum
            rs_ratio: Relative strength ratio
            rs_momentum: RS momentum
            breadth_score: Breadth score

        Returns:
            Dictionary with signal, confidence, and reasoning
        """
        # Determine RRG quadrant
        quadrant = self.calculate_rrg_quadrant(rs_ratio, rs_momentum)

        # Calculate composite score (0-100)
        momentum_weight = 0.3
        rs_weight = 0.3
        breadth_weight = 0.4

        # Normalize scores
        momentum_normalized = min(max((momentum_score + 20) / 40 * 100, 0), 100)
        rs_normalized = (rs_ratio - 90) / 20 * 100  # 90-110 range
        rs_normalized = min(max(rs_normalized, 0), 100)

        composite_score = (
            (momentum_normalized * momentum_weight) +
            (rs_normalized * rs_weight) +
            (breadth_score * breadth_weight)
        )

        # Determine signal
        if quadrant == "Leading":
            if composite_score > 70:
                signal = "STRONG BUY"
                confidence = "High"
            elif composite_score > 55:
                signal = "BUY"
                confidence = "Medium"
            else:
                signal = "HOLD"
                confidence = "Medium"

        elif quadrant == "Improving":
            if rs_momentum > 2:
                signal = "BUY"
                confidence = "High"
            elif rs_momentum > 0:
                signal = "BUY"
                confidence = "Medium"
            else:
                signal = "HOLD"
                confidence = "Low"

        elif quadrant == "Weakening":
            if rs_momentum < -2:
                signal = "SELL"
                confidence = "High"
            elif rs_momentum < 0:
                signal = "REDUCE"
                confidence = "Medium"
            else:
                signal = "HOLD"
                confidence = "Medium"

        else:  # Lagging
            if composite_score < 30:
                signal = "STRONG SELL"
                confidence = "High"
            elif composite_score < 45:
                signal = "SELL"
                confidence = "Medium"
            else:
                signal = "HOLD"
                confidence = "Low"

        # Generate reasoning
        reasons = []
        if momentum_score > 10:
            reasons.append("Strong upward momentum")
        elif momentum_score < -10:
            reasons.append("Weak momentum")

        if rs_ratio > 105:
            reasons.append("Outperforming market")
        elif rs_ratio < 95:
            reasons.append("Underperforming market")

        if rs_momentum > 1:
            reasons.append("Improving relative strength")
        elif rs_momentum < -1:
            reasons.append("Declining relative strength")

        if breadth_score > 70:
            reasons.append("Strong sector breadth")
        elif breadth_score < 30:
            reasons.append("Weak sector breadth")

        return {
            'sector': sector,
            'signal': signal,
            'confidence': confidence,
            'quadrant': quadrant,
            'composite_score': round(composite_score, 1),
            'reasons': reasons
        }

    # ========================================================================
    # ECONOMIC CYCLE POSITIONING
    # ========================================================================

    def get_economic_cycle_recommendation(
        self,
        pmi: float,
        gdp_growth: float,
        unemployment_rate: float,
        fed_funds_rate: float
    ) -> Dict[str, List[str]]:
        """
        Determine sector recommendations based on economic cycle

        Args:
            pmi: Manufacturing PMI (50 = expansion threshold)
            gdp_growth: GDP growth rate (annual %)
            unemployment_rate: Unemployment rate (%)
            fed_funds_rate: Federal Funds Rate (%)

        Returns:
            Dictionary with overweight/underweight sectors
        """
        # Determine cycle phase
        if pmi > 55 and gdp_growth > 3:
            cycle = "Early Expansion"
            overweight = ['Industrials', 'Materials', 'Information Technology']
            underweight = ['Utilities', 'Consumer Staples']

        elif pmi > 50 and gdp_growth > 2:
            cycle = "Mid Expansion"
            overweight = ['Information Technology', 'Consumer Discretionary', 'Industrials']
            underweight = ['Utilities', 'Real Estate']

        elif pmi < 50 and gdp_growth < 2:
            cycle = "Late Cycle/Slowdown"
            overweight = ['Energy', 'Financials', 'Consumer Staples']
            underweight = ['Industrials', 'Materials']

        else:  # pmi < 45 or gdp_growth < 0
            cycle = "Recession"
            overweight = ['Utilities', 'Consumer Staples', 'Health Care']
            underweight = ['Consumer Discretionary', 'Industrials', 'Materials']

        # Interest rate adjustments
        if fed_funds_rate > 4.5:  # High rates
            if 'Financials' not in overweight:
                overweight.append('Financials')
            if 'Real Estate' not in underweight:
                underweight.append('Real Estate')

        return {
            'cycle': cycle,
            'overweight': overweight,
            'underweight': underweight,
            'pmi': pmi,
            'gdp_growth': gdp_growth
        }

    # ========================================================================
    # COMPREHENSIVE SECTOR SCORE
    # ========================================================================

    def calculate_comprehensive_score(
        self,
        sector: str,
        momentum_score: float,
        rs_ratio: float,
        rs_momentum: float,
        breadth_score: float,
        sharpe_ratio: float,
        beta: float,
        avg_premium_yield: float = 0.0
    ) -> Dict[str, any]:
        """
        Calculate comprehensive sector score (0-100)

        Combines:
        - Momentum (25%)
        - Relative Strength (25%)
        - Breadth (20%)
        - Risk-Adjusted Returns (15%)
        - Wheel Strategy Suitability (15%)

        Args:
            sector: Sector name
            momentum_score: Multi-period momentum
            rs_ratio: RS-Ratio
            rs_momentum: RS-Momentum
            breadth_score: Breadth score
            sharpe_ratio: Sharpe ratio
            beta: Beta
            avg_premium_yield: Average premium yield for wheel strategy

        Returns:
            Comprehensive score and breakdown
        """
        # Normalize components (0-100)

        # 1. Momentum (25%)
        momentum_normalized = min(max((momentum_score + 20) / 40 * 100, 0), 100)
        momentum_component = momentum_normalized * 0.25

        # 2. Relative Strength (25%)
        rs_normalized = (rs_ratio - 90) / 20 * 100  # 90-110 range
        rs_normalized = min(max(rs_normalized, 0), 100)
        rs_component = rs_normalized * 0.25

        # 3. Breadth (20%)
        breadth_component = breadth_score * 0.20

        # 4. Risk-Adjusted Returns (15%)
        sharpe_normalized = min(max((sharpe_ratio + 1) / 3 * 100, 0), 100)
        risk_component = sharpe_normalized * 0.15

        # 5. Wheel Strategy Suitability (15%)
        # Based on typical premium yields and sector profile
        profile = self.sector_profiles.get(sector, {})
        wheel_tier = profile.get('wheel_tier', 2)

        if wheel_tier == 1:
            wheel_score = 90
        elif wheel_tier == 2:
            wheel_score = 70
        else:
            wheel_score = 50

        # Boost score if actual premium yield is high
        if avg_premium_yield > 3:
            wheel_score = min(wheel_score + 10, 100)

        wheel_component = wheel_score * 0.15

        # Calculate total
        total_score = (
            momentum_component +
            rs_component +
            breadth_component +
            risk_component +
            wheel_component
        )

        return {
            'sector': sector,
            'overall_score': round(total_score, 1),
            'momentum_score': round(momentum_component, 1),
            'rs_score': round(rs_component, 1),
            'breadth_score': round(breadth_component, 1),
            'risk_score': round(risk_component, 1),
            'wheel_score': round(wheel_component, 1),
            'beta': round(beta, 2),
            'sharpe': round(sharpe_ratio, 2),
            'wheel_tier': wheel_tier
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def get_sector_profile(self, sector: str) -> Dict[str, any]:
        """Get sector profile with typical characteristics"""
        return self.sector_profiles.get(sector, {
            'typical_beta': 1.0,
            'cyclicality': 'Unknown',
            'defensive': False,
            'avg_dividend_yield': 2.0,
            'wheel_tier': 2,
            'optimal_dte': 45
        })

    def get_sector_etf(self, sector: str) -> str:
        """Get ETF ticker for a sector"""
        return self.sector_etf_map.get(sector, 'SPY')

    def get_all_sectors(self) -> List[str]:
        """Get list of all GICS sectors"""
        return list(self.sector_etf_map.keys())
