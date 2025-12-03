"""
Base Options Strategy Framework
===============================

Abstract base class and data structures for all options strategies.

Author: AVA Trading Platform
Created: 2025-11-28
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any, Callable
import logging
import pandas as pd
import numpy as np

from src.ava.options.greeks_engine import (
    AdvancedGreeksEngine,
    GreeksResult,
    get_greeks_engine
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class StrategyType(Enum):
    """Categories of options strategies"""
    INCOME = "income"           # Theta positive, collect premium
    SPREAD = "spread"           # Defined risk spreads
    NEUTRAL = "neutral"         # Range-bound strategies
    DIRECTIONAL = "directional" # Bullish or bearish
    VOLATILITY = "volatility"   # Vol expansion/contraction plays
    ADVANCED = "advanced"       # Complex/0DTE strategies


class OptionSide(Enum):
    """Buy or sell"""
    BUY = "buy"
    SELL = "sell"


class OptionType(Enum):
    """Call or put"""
    CALL = "call"
    PUT = "put"


class MarketOutlook(Enum):
    """Expected market direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"     # Expecting big move, direction unknown
    LOW_VOL = "low_volatility"  # Expecting low volatility


class SignalStrength(Enum):
    """Strength of trading signal"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NONE = "none"


class ExitReason(Enum):
    """Reason for exiting a position"""
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"
    TIME_DECAY = "time_decay"
    DTE_THRESHOLD = "dte_threshold"
    DELTA_BREACH = "delta_breach"
    ASSIGNMENT_RISK = "assignment_risk"
    MANUAL = "manual"
    ROLL = "roll"
    EARNINGS = "earnings"
    EXPIRATION = "expiration"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptionLeg:
    """Represents a single leg of an options strategy"""
    option_type: OptionType
    side: OptionSide
    strike: float
    expiration: date
    quantity: int = 1

    # Market data (optional, filled at execution)
    bid: float = 0.0
    ask: float = 0.0
    last_price: float = 0.0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    @property
    def mid_price(self) -> float:
        """Mid price between bid and ask"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last_price

    @property
    def is_call(self) -> bool:
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        return self.option_type == OptionType.PUT

    @property
    def is_long(self) -> bool:
        return self.side == OptionSide.BUY

    @property
    def is_short(self) -> bool:
        return self.side == OptionSide.SELL

    @property
    def days_to_expiration(self) -> int:
        return (self.expiration - date.today()).days

    def to_dict(self) -> Dict:
        return {
            'option_type': self.option_type.value,
            'side': self.side.value,
            'strike': self.strike,
            'expiration': self.expiration.isoformat(),
            'quantity': self.quantity,
            'mid_price': self.mid_price,
            'iv': self.implied_volatility,
            'delta': self.delta
        }


@dataclass
class StrategySetup:
    """Complete setup for a strategy trade"""
    symbol: str
    strategy_name: str
    legs: List[OptionLeg]

    # Risk/Reward
    max_profit: float = 0.0
    max_loss: float = 0.0
    breakeven_prices: List[float] = field(default_factory=list)

    # Probabilities
    probability_of_profit: float = 0.0
    probability_of_max_profit: float = 0.0
    probability_of_max_loss: float = 0.0

    # Greeks (aggregate)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0

    # Scoring
    score: float = 0.0
    score_components: Dict[str, float] = field(default_factory=dict)

    # Market context
    underlying_price: float = 0.0
    iv_rank: float = 0.0
    iv_percentile: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""

    @property
    def risk_reward_ratio(self) -> float:
        """Risk to reward ratio (lower is better)"""
        if self.max_profit == 0:
            return float('inf')
        return abs(self.max_loss) / self.max_profit

    @property
    def expected_value(self) -> float:
        """Expected value of the trade"""
        return (self.probability_of_profit * self.max_profit -
                (1 - self.probability_of_profit) * abs(self.max_loss))

    @property
    def net_premium(self) -> float:
        """Net premium received (positive) or paid (negative)"""
        total = 0.0
        for leg in self.legs:
            value = leg.mid_price * leg.quantity * 100
            if leg.is_long:
                total -= value
            else:
                total += value
        return total

    @property
    def days_to_expiration(self) -> int:
        """DTE of the nearest expiration"""
        if not self.legs:
            return 0
        return min(leg.days_to_expiration for leg in self.legs)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'legs': [leg.to_dict() for leg in self.legs],
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'breakeven_prices': self.breakeven_prices,
            'probability_of_profit': self.probability_of_profit,
            'risk_reward_ratio': self.risk_reward_ratio,
            'expected_value': self.expected_value,
            'net_delta': self.net_delta,
            'net_theta': self.net_theta,
            'score': self.score,
            'underlying_price': self.underlying_price,
            'iv_rank': self.iv_rank,
            'dte': self.days_to_expiration,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class EntrySignal:
    """Signal to enter a trade"""
    should_enter: bool
    strength: SignalStrength
    setup: Optional[StrategySetup] = None
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_position_size: int = 1

    def to_dict(self) -> Dict:
        return {
            'should_enter': self.should_enter,
            'strength': self.strength.value,
            'setup': self.setup.to_dict() if self.setup else None,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'suggested_position_size': self.suggested_position_size
        }


@dataclass
class ExitSignal:
    """Signal to exit a trade"""
    should_exit: bool
    reason: ExitReason
    urgency: SignalStrength
    suggested_action: str = ""  # e.g., "close", "roll to next month"
    current_pnl: float = 0.0
    current_pnl_pct: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'should_exit': self.should_exit,
            'reason': self.reason.value,
            'urgency': self.urgency.value,
            'suggested_action': self.suggested_action,
            'current_pnl': self.current_pnl,
            'current_pnl_pct': self.current_pnl_pct,
            'notes': self.notes
        }


@dataclass
class StrategyResult:
    """Result of a completed strategy trade"""
    symbol: str
    strategy_name: str
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    quantity: int

    realized_pnl: float = 0.0
    realized_pnl_pct: float = 0.0
    days_held: int = 0
    exit_reason: ExitReason = ExitReason.MANUAL

    # Performance metrics
    max_drawdown: float = 0.0
    max_profit_seen: float = 0.0
    theta_collected: float = 0.0

    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'realized_pnl': self.realized_pnl,
            'realized_pnl_pct': self.realized_pnl_pct,
            'days_held': self.days_held,
            'exit_reason': self.exit_reason.value
        }


# =============================================================================
# MARKET DATA INTERFACE
# =============================================================================

@dataclass
class MarketContext:
    """Current market conditions for strategy evaluation"""
    symbol: str
    current_price: float
    previous_close: float

    # Volatility metrics
    implied_volatility: float = 0.0
    iv_rank: float = 0.0           # 0-100, current IV vs past year
    iv_percentile: float = 0.0     # 0-100, % of days with lower IV
    historical_volatility: float = 0.0
    vix: float = 0.0

    # Price movement
    daily_change: float = 0.0
    daily_change_pct: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0

    # Volume
    volume: int = 0
    avg_volume: int = 0

    # Events
    earnings_date: Optional[date] = None
    days_to_earnings: Optional[int] = None
    ex_dividend_date: Optional[date] = None

    # Technicals (optional)
    rsi_14: float = 50.0
    sma_50: float = 0.0
    sma_200: float = 0.0

    # Timestamp
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def is_high_iv(self) -> bool:
        """IV rank > 50 indicates elevated volatility"""
        return self.iv_rank > 50

    @property
    def is_low_iv(self) -> bool:
        """IV rank < 30 indicates low volatility"""
        return self.iv_rank < 30

    @property
    def days_to_earnings_safe(self) -> bool:
        """True if no earnings within 14 days"""
        if self.days_to_earnings is None:
            return True
        return self.days_to_earnings > 14

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'iv': self.implied_volatility,
            'iv_rank': self.iv_rank,
            'iv_percentile': self.iv_percentile,
            'daily_change_pct': self.daily_change_pct,
            'days_to_earnings': self.days_to_earnings,
            'is_high_iv': self.is_high_iv,
            'is_low_iv': self.is_low_iv
        }


@dataclass
class OptionsChain:
    """Options chain data for strategy evaluation"""
    symbol: str
    underlying_price: float
    expirations: List[date]
    calls: pd.DataFrame  # Columns: strike, expiration, bid, ask, last, iv, delta, gamma, theta, vega, oi, volume
    puts: pd.DataFrame

    @property
    def nearest_expiration(self) -> Optional[date]:
        if self.expirations:
            return min(self.expirations)
        return None

    def get_chain_for_expiration(self, expiration: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get calls and puts for a specific expiration"""
        calls = self.calls[self.calls['expiration'] == expiration] if not self.calls.empty else pd.DataFrame()
        puts = self.puts[self.puts['expiration'] == expiration] if not self.puts.empty else pd.DataFrame()
        return calls, puts

    def get_atm_strike(self, expiration: date) -> float:
        """Get the at-the-money strike for an expiration"""
        calls, _ = self.get_chain_for_expiration(expiration)
        if calls.empty:
            return self.underlying_price

        strikes = calls['strike'].values
        return float(strikes[np.argmin(np.abs(strikes - self.underlying_price))])

    def get_strike_by_delta(
        self,
        expiration: date,
        target_delta: float,
        option_type: OptionType
    ) -> Optional[float]:
        """Find strike with closest delta to target"""
        if option_type == OptionType.CALL:
            chain = self.calls[self.calls['expiration'] == expiration]
        else:
            chain = self.puts[self.puts['expiration'] == expiration]

        if chain.empty:
            return None

        # For puts, delta is negative, so compare absolute values
        target = abs(target_delta)
        deltas = chain['delta'].abs().values
        idx = np.argmin(np.abs(deltas - target))

        return float(chain.iloc[idx]['strike'])


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class OptionsStrategy(ABC):
    """
    Abstract base class for all options strategies.

    Subclasses must implement:
    - find_opportunities(): Find valid setups in options chain
    - entry_conditions(): Check if entry conditions are met
    - exit_conditions(): Check if exit conditions are met
    - calculate_position_size(): Determine optimal position size

    Optional overrides:
    - score_setup(): Custom scoring logic
    - adjust_position(): Adjustment recommendations
    """

    # Strategy metadata (override in subclasses)
    name: str = "Base Strategy"
    description: str = "Abstract base strategy"
    strategy_type: StrategyType = StrategyType.INCOME

    # Default parameters (override in subclasses)
    min_iv_rank: float = 0.0
    max_iv_rank: float = 100.0
    min_dte: int = 7
    max_dte: int = 60
    min_delta: float = 0.0
    max_delta: float = 1.0
    min_probability_of_profit: float = 0.0

    # Risk parameters
    max_position_size_pct: float = 0.05  # Max 5% of portfolio per trade
    max_loss_pct: float = 0.02           # Max 2% loss per trade
    profit_target_pct: float = 0.50      # Take profits at 50%
    stop_loss_pct: float = 2.0           # Stop at 200% of credit received

    def __init__(self, **kwargs):
        """
        Initialize strategy with optional parameter overrides.

        Args:
            **kwargs: Override any class-level parameters
        """
        self.greeks_engine = get_greeks_engine()

        # Apply parameter overrides
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown parameter: {key}")

        logger.info(f"Initialized {self.name} strategy")

    # =========================================================================
    # ABSTRACT METHODS (Must implement in subclasses)
    # =========================================================================

    @abstractmethod
    def find_opportunities(
        self,
        chain: OptionsChain,
        context: MarketContext
    ) -> List[StrategySetup]:
        """
        Find all valid strategy setups in the options chain.

        Args:
            chain: Options chain data
            context: Market context (price, IV, etc.)

        Returns:
            List of valid StrategySetup objects, sorted by score
        """
        pass

    @abstractmethod
    def entry_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext
    ) -> EntrySignal:
        """
        Check if entry conditions are met for a setup.

        Args:
            setup: Strategy setup to evaluate
            context: Current market context

        Returns:
            EntrySignal with recommendation
        """
        pass

    @abstractmethod
    def exit_conditions(
        self,
        setup: StrategySetup,
        context: MarketContext,
        entry_price: float,
        current_price: float
    ) -> ExitSignal:
        """
        Check if exit conditions are met for an open position.

        Args:
            setup: Original strategy setup
            context: Current market context
            entry_price: Price at entry (net premium)
            current_price: Current price (net premium)

        Returns:
            ExitSignal with recommendation
        """
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        setup: StrategySetup,
        account_value: float,
        current_positions: List[StrategySetup]
    ) -> int:
        """
        Calculate optimal position size for a setup.

        Args:
            setup: Strategy setup
            account_value: Total account value
            current_positions: Existing open positions

        Returns:
            Number of contracts to trade
        """
        pass

    # =========================================================================
    # OPTIONAL OVERRIDES
    # =========================================================================

    def score_setup(self, setup: StrategySetup, context: MarketContext) -> float:
        """
        Score a strategy setup (0-100). Higher is better.

        Default implementation uses:
        - Probability of profit (30%)
        - Risk/reward ratio (25%)
        - IV rank alignment (20%)
        - Days to expiration (15%)
        - Net theta (10%)
        """
        score = 0.0
        components = {}

        # Probability of profit (30%)
        pop_score = setup.probability_of_profit * 100 * 0.30
        components['pop'] = pop_score
        score += pop_score

        # Risk/reward (25%) - lower is better, max at 1:3
        rr = setup.risk_reward_ratio
        if rr > 0:
            rr_score = max(0, (3 - min(rr, 3)) / 3) * 25
        else:
            rr_score = 25  # Unlimited profit potential
        components['risk_reward'] = rr_score
        score += rr_score

        # IV rank alignment (20%)
        # High IV strategies (selling premium) want high IV rank
        # Low IV strategies (buying premium) want low IV rank
        if self.strategy_type == StrategyType.INCOME:
            iv_score = context.iv_rank / 100 * 20
        else:
            iv_score = (100 - context.iv_rank) / 100 * 20
        components['iv_rank'] = iv_score
        score += iv_score

        # DTE (15%) - prefer middle of range
        dte = setup.days_to_expiration
        optimal_dte = (self.min_dte + self.max_dte) / 2
        dte_diff = abs(dte - optimal_dte)
        dte_range = (self.max_dte - self.min_dte) / 2
        dte_score = max(0, (dte_range - dte_diff) / dte_range) * 15
        components['dte'] = dte_score
        score += dte_score

        # Net theta (10%) - positive theta is good for income strategies
        if setup.net_theta > 0:
            theta_score = min(setup.net_theta / 10, 1) * 10
        else:
            theta_score = 0
        components['theta'] = theta_score
        score += theta_score

        setup.score = round(score, 2)
        setup.score_components = components

        return score

    def adjust_position(
        self,
        setup: StrategySetup,
        context: MarketContext,
        current_pnl: float
    ) -> Optional[StrategySetup]:
        """
        Recommend position adjustment (e.g., roll, add leg).

        Default implementation: No adjustment.
        Override in subclasses for specific adjustment logic.

        Returns:
            New StrategySetup if adjustment recommended, None otherwise
        """
        return None

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def filter_by_iv_rank(self, context: MarketContext) -> bool:
        """Check if IV rank is within acceptable range"""
        return self.min_iv_rank <= context.iv_rank <= self.max_iv_rank

    def filter_by_dte(self, expiration: date) -> bool:
        """Check if expiration is within acceptable DTE range"""
        dte = (expiration - date.today()).days
        return self.min_dte <= dte <= self.max_dte

    def filter_by_delta(self, delta: float) -> bool:
        """Check if delta is within acceptable range"""
        return self.min_delta <= abs(delta) <= self.max_delta

    def calculate_breakeven(self, legs: List[OptionLeg], underlying_price: float) -> List[float]:
        """Calculate breakeven price(s) for a strategy"""
        # Simplified - override in subclasses for accurate calculations
        net_premium = sum(
            leg.mid_price * leg.quantity * (1 if leg.is_short else -1)
            for leg in legs
        )

        breakevens = []

        # Single leg or simple spread approximation
        if len(legs) == 1:
            leg = legs[0]
            if leg.is_call:
                breakevens.append(leg.strike + net_premium if leg.is_long else leg.strike - net_premium)
            else:
                breakevens.append(leg.strike - net_premium if leg.is_long else leg.strike + net_premium)
        else:
            # Multi-leg - return strikes as approximation
            strikes = [leg.strike for leg in legs]
            breakevens = [min(strikes), max(strikes)]

        return breakevens

    def calculate_max_profit_loss(
        self,
        legs: List[OptionLeg],
        underlying_price: float
    ) -> Tuple[float, float]:
        """
        Calculate max profit and max loss for a strategy.

        Returns:
            Tuple of (max_profit, max_loss)
        """
        # Net premium received (positive) or paid (negative)
        net_premium = sum(
            leg.mid_price * leg.quantity * 100 * (1 if leg.is_short else -1)
            for leg in legs
        )

        # Simple approximations - override in subclasses
        if net_premium > 0:
            # Credit strategy - max profit is premium, max loss is complex
            max_profit = net_premium

            # Calculate max loss based on strike widths
            strikes = sorted([leg.strike for leg in legs])
            if len(strikes) >= 2:
                max_width = (strikes[-1] - strikes[0]) * 100
                max_loss = max_width - net_premium
            else:
                max_loss = float('inf')  # Undefined for single leg
        else:
            # Debit strategy - max loss is premium paid, max profit varies
            max_loss = abs(net_premium)
            max_profit = float('inf')  # Often undefined

        return max_profit, max_loss

    def aggregate_greeks(self, legs: List[OptionLeg]) -> Dict[str, float]:
        """Calculate aggregate Greeks for all legs"""
        net_greeks = {
            'delta': 0.0,
            'gamma': 0.0,
            'theta': 0.0,
            'vega': 0.0
        }

        for leg in legs:
            multiplier = leg.quantity * (1 if leg.is_long else -1)
            net_greeks['delta'] += leg.delta * multiplier
            net_greeks['gamma'] += leg.gamma * multiplier
            net_greeks['theta'] += leg.theta * multiplier
            net_greeks['vega'] += leg.vega * multiplier

        return net_greeks

    def validate_setup(self, setup: StrategySetup) -> Tuple[bool, List[str]]:
        """
        Validate a strategy setup.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        if not setup.legs:
            errors.append("No legs in setup")

        if setup.max_loss == float('inf'):
            errors.append("Undefined max loss - check risk limits")

        if setup.days_to_expiration <= 0:
            errors.append("Option has expired")

        if setup.probability_of_profit < self.min_probability_of_profit:
            errors.append(f"POP {setup.probability_of_profit:.1%} below minimum {self.min_probability_of_profit:.1%}")

        # Check liquidity (bid-ask spread)
        for leg in setup.legs:
            if leg.ask > 0 and leg.bid > 0:
                spread_pct = (leg.ask - leg.bid) / leg.mid_price
                if spread_pct > 0.10:  # 10% spread
                    errors.append(f"Wide bid-ask spread ({spread_pct:.1%}) on {leg.strike} {leg.option_type.value}")

        return len(errors) == 0, errors

    def __str__(self) -> str:
        return f"{self.name} ({self.strategy_type.value})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


# =============================================================================
# STRATEGY REGISTRY
# =============================================================================

class StrategyRegistry:
    """Registry for available strategies"""

    _strategies: Dict[str, type] = {}

    @classmethod
    def register(cls, strategy_class: type):
        """Register a strategy class"""
        name = strategy_class.name if hasattr(strategy_class, 'name') else strategy_class.__name__
        cls._strategies[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a strategy class by name"""
        return cls._strategies.get(name)

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered strategy names"""
        return list(cls._strategies.keys())

    @classmethod
    def list_by_type(cls, strategy_type: StrategyType) -> List[str]:
        """List strategies of a specific type"""
        return [
            name for name, strat_cls in cls._strategies.items()
            if hasattr(strat_cls, 'strategy_type') and strat_cls.strategy_type == strategy_type
        ]

    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[OptionsStrategy]:
        """Create a strategy instance by name"""
        strategy_class = cls.get(name)
        if strategy_class:
            return strategy_class(**kwargs)
        return None


def register_strategy(cls):
    """Decorator to register a strategy class"""
    StrategyRegistry.register(cls)
    return cls


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Base Strategy Framework ===\n")

    # Test OptionLeg
    print("1. Testing OptionLeg")
    leg = OptionLeg(
        option_type=OptionType.PUT,
        side=OptionSide.SELL,
        strike=500.0,
        expiration=date.today() + timedelta(days=30),
        quantity=1,
        bid=4.50,
        ask=4.80,
        delta=-0.25,
        theta=0.15
    )
    print(f"   Leg: {leg.strike} {leg.option_type.value} {leg.side.value}")
    print(f"   Mid price: ${leg.mid_price:.2f}")
    print(f"   DTE: {leg.days_to_expiration}")

    # Test StrategySetup
    print("\n2. Testing StrategySetup")
    setup = StrategySetup(
        symbol='NVDA',
        strategy_name='Cash Secured Put',
        legs=[leg],
        max_profit=480.0,
        max_loss=49520.0,
        probability_of_profit=0.75,
        net_theta=15.0,
        underlying_price=520.0,
        iv_rank=65.0
    )
    print(f"   Strategy: {setup.strategy_name}")
    print(f"   Max profit: ${setup.max_profit:.2f}")
    print(f"   Risk/Reward: {setup.risk_reward_ratio:.2f}")
    print(f"   Expected value: ${setup.expected_value:.2f}")

    # Test MarketContext
    print("\n3. Testing MarketContext")
    context = MarketContext(
        symbol='NVDA',
        current_price=520.0,
        previous_close=515.0,
        implied_volatility=0.40,
        iv_rank=65.0,
        iv_percentile=70.0,
        earnings_date=date.today() + timedelta(days=45)
    )
    print(f"   Symbol: {context.symbol}")
    print(f"   Is high IV: {context.is_high_iv}")
    print(f"   Days to earnings safe: {context.days_to_earnings_safe}")

    # Test Registry
    print("\n4. Testing StrategyRegistry")
    print(f"   Registered strategies: {StrategyRegistry.list_all()}")

    print("\n✅ Base strategy framework ready!")
