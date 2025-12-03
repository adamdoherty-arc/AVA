"""
Data models for calendar spread analysis
"""

from dataclasses import dataclass
from datetime import date
from typing import List, Optional, Tuple


@dataclass
class OptionContract:
    """Single option contract"""
    symbol: str
    strike: float
    expiration: date
    option_type: str  # 'call' or 'put'
    premium: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    theta: float
    gamma: float
    vega: float
    dte: int  # Days to expiration


@dataclass
class CalendarSpreadOpportunity:
    """Calendar spread opportunity with metrics"""
    # Basic info
    symbol: str
    stock_price: float

    # Near-term leg (sell)
    near_strike: float
    near_expiration: date
    near_dte: int
    near_premium: float
    near_iv: float
    near_theta: float
    near_volume: int

    # Far-term leg (buy)
    far_strike: float
    far_expiration: date
    far_dte: int
    far_premium: float
    far_iv: float
    far_theta: float
    far_volume: int

    # Spread metrics
    net_debit: float  # Cost to open (far_premium - near_premium)
    max_profit: float  # Maximum potential profit
    max_loss: float  # = net_debit
    profit_potential: float  # Max profit / max loss ratio
    probability_profit: float  # % chance of profit (0-100)
    breakeven_lower: float
    breakeven_upper: float

    # Greeks
    net_theta: float  # Theta advantage (near_theta - far_theta)
    net_vega: float  # Vega exposure

    # Quality metrics
    liquidity_score: float  # 0-100 based on volume/OI
    iv_differential: float  # Difference in IV between legs

    # Scoring
    opportunity_score: float  # 0-100 composite score
    rank: int = 0
    strategy_name: str = "Calendar Spread"

    def __post_init__(self) -> None:
        """Calculate derived metrics"""
        if self.net_debit > 0:
            self.profit_potential = self.max_profit / self.net_debit
        else:
            self.profit_potential = 0.0

        self.max_loss = abs(self.net_debit)
        self.net_theta = abs(self.near_theta) - abs(self.far_theta)
        self.iv_differential = abs(self.near_iv - self.far_iv)