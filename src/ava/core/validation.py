"""
AVA Core Validation Models
==========================

Pydantic models for comprehensive data validation throughout the platform.
Ensures type safety, bounds checking, and data integrity.

Author: AVA Trading Platform
Created: 2025-11-28
"""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Optional, Any, Union, Annotated
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    PositiveFloat,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt
)
import re


# =============================================================================
# ENUMS
# =============================================================================

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OrderAction(str, Enum):
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class StrategyType(str, Enum):
    INCOME = "income"
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"
    NEUTRAL = "neutral"
    HEDGE = "hedge"


class TradeAction(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    AVOID = "avoid"
    STRONG_AVOID = "strong_avoid"


class Conviction(str, Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"


# =============================================================================
# CUSTOM TYPES
# =============================================================================

# Bounded float types
Delta = Annotated[float, Field(ge=-1.0, le=1.0, description="Option delta (-1 to 1)")]
Gamma = Annotated[float, Field(ge=0, le=1.0, description="Option gamma (0 to 1)")]
Theta = Annotated[float, Field(description="Option theta (negative for long options)")]
Vega = Annotated[float, Field(ge=0, description="Option vega (non-negative)")]
IV = Annotated[float, Field(ge=0, le=5.0, description="Implied volatility (0 to 500%)")]
IVRank = Annotated[float, Field(ge=0, le=100, description="IV Rank (0 to 100)")]
Probability = Annotated[float, Field(ge=0, le=1.0, description="Probability (0 to 1)")]
Percentage = Annotated[float, Field(ge=-100, le=1000, description="Percentage")]


# =============================================================================
# BASE MODELS
# =============================================================================

class AVABaseModel(BaseModel):
    """Base model with common configuration"""
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True
    )


class TimestampedModel(AVABaseModel):
    """Model with automatic timestamp"""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def touch(self):
        self.updated_at = datetime.now()


# =============================================================================
# OPTION MODELS
# =============================================================================

class OptionLeg(AVABaseModel):
    """Validated option leg"""
    symbol: str = Field(..., min_length=1, max_length=50)
    underlying_symbol: str = Field(..., min_length=1, max_length=10)
    strike: PositiveFloat
    expiration: date
    option_type: OptionType
    quantity: int = Field(..., ne=0, description="Non-zero quantity")
    bid: NonNegativeFloat = 0
    ask: NonNegativeFloat = 0
    last: NonNegativeFloat = 0
    delta: Delta = 0
    gamma: Gamma = 0
    theta: Theta = 0
    vega: Vega = 0
    iv: IV = 0.25
    open_interest: NonNegativeInt = 0
    volume: NonNegativeInt = 0

    @field_validator('symbol')
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate option symbol format"""
        v = v.upper().strip()
        # Allow standard OCC format or simplified format
        if not re.match(r'^[A-Z]{1,6}', v):
            raise ValueError(f"Invalid option symbol: {v}")
        return v

    @field_validator('expiration')
    @classmethod
    def validate_expiration(cls, v: date) -> date:
        """Ensure expiration is not in the past"""
        if v < date.today():
            raise ValueError(f"Expiration {v} is in the past")
        return v

    @model_validator(mode='after')
    def validate_bid_ask(self):
        """Ensure bid <= ask"""
        if self.bid > self.ask > 0:
            raise ValueError(f"Bid ({self.bid}) cannot be greater than ask ({self.ask})")
        return self

    @property
    def mid_price(self) -> float:
        """Calculate mid price safely"""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last or 0

    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.ask - self.bid if self.ask > 0 else 0

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price"""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration"""
        return max(0, (self.expiration - date.today()).days)

    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money (requires underlying price)"""
        # This would need underlying price context
        return False

    def is_liquid(self, min_oi: int = 100, max_spread_pct: float = 10) -> bool:
        """Check if option meets liquidity requirements"""
        return (
            self.open_interest >= min_oi and
            self.spread_pct <= max_spread_pct and
            self.volume > 0
        )


class OptionChain(AVABaseModel):
    """Validated options chain"""
    underlying_symbol: str = Field(..., min_length=1, max_length=10)
    underlying_price: PositiveFloat
    expirations: List[date]
    calls: Dict[str, List[OptionLeg]] = Field(default_factory=dict)
    puts: Dict[str, List[OptionLeg]] = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.now)

    @field_validator('underlying_symbol')
    @classmethod
    def validate_underlying(cls, v: str) -> str:
        return v.upper().strip()

    def get_expiration_chain(self, expiration: date) -> tuple[List[OptionLeg], List[OptionLeg]]:
        """Get calls and puts for a specific expiration"""
        exp_str = expiration.isoformat()
        return (
            self.calls.get(exp_str, []),
            self.puts.get(exp_str, [])
        )

    def find_by_delta(
        self,
        option_type: OptionType,
        target_delta: float,
        expiration: date,
        tolerance: float = 0.05
    ) -> Optional[OptionLeg]:
        """Find option closest to target delta"""
        exp_str = expiration.isoformat()
        options = self.calls.get(exp_str, []) if option_type == OptionType.CALL else self.puts.get(exp_str, [])

        if not options:
            return None

        closest = min(
            options,
            key=lambda x: abs(x.delta - target_delta)
        )

        if abs(closest.delta - target_delta) <= tolerance:
            return closest
        return None


# =============================================================================
# STRATEGY MODELS
# =============================================================================

class StrategySetup(AVABaseModel):
    """Validated strategy setup"""
    symbol: str = Field(..., min_length=1, max_length=10)
    strategy_name: str = Field(..., min_length=1, max_length=100)
    strategy_type: StrategyType
    legs: List[OptionLeg] = Field(..., min_length=1, max_length=8)
    underlying_price: PositiveFloat

    # P&L metrics
    max_profit: float = Field(..., description="Maximum profit (can be infinite)")
    max_loss: float = Field(..., description="Maximum loss (negative)")
    break_even: Union[float, List[float]] = Field(..., description="Break-even point(s)")

    # Probability metrics
    probability_of_profit: Probability = Field(..., ge=0, le=1)
    expected_value: float = 0

    # Greeks
    net_delta: Delta = 0
    net_gamma: float = 0
    net_theta: float = 0
    net_vega: float = 0

    # Time factors
    days_to_expiration: NonNegativeInt
    iv_rank_at_entry: IVRank = 50

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    notes: Optional[str] = None

    @model_validator(mode='after')
    def validate_setup(self):
        """Comprehensive setup validation"""
        # Validate max_loss is negative or zero
        if self.max_loss > 0:
            raise ValueError("max_loss should be negative or zero")

        # Validate DTE against leg expirations
        if self.legs:
            max_leg_dte = max(leg.days_to_expiration for leg in self.legs)
            if self.days_to_expiration > max_leg_dte + 1:  # +1 for rounding
                raise ValueError("days_to_expiration exceeds leg expirations")

        return self

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio safely"""
        if self.max_loss == 0:
            return float('inf') if self.max_profit > 0 else 0
        return abs(self.max_profit / self.max_loss)

    @property
    def is_credit_strategy(self) -> bool:
        """Check if this is a credit strategy"""
        return self.net_theta > 0

    @property
    def total_premium(self) -> float:
        """Calculate total premium received/paid"""
        return sum(
            leg.mid_price * leg.quantity * 100
            for leg in self.legs
        )


class TradeSignal(AVABaseModel):
    """Validated trade signal"""
    symbol: str
    strategy_name: str
    action: TradeAction
    conviction: Conviction
    confidence: Probability

    # Position sizing
    recommended_contracts: PositiveInt = 1
    max_contracts: PositiveInt = 1

    # Risk parameters
    stop_loss_pct: Percentage = Field(default=50, ge=0, le=500)
    profit_target_pct: Percentage = Field(default=50, ge=0, le=500)
    max_days_to_hold: PositiveInt = 45

    # Analysis
    composite_score: float = Field(default=0, ge=-100, le=100)
    key_reasons: List[str] = Field(default_factory=list, max_length=10)
    warnings: List[str] = Field(default_factory=list, max_length=10)

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    @model_validator(mode='after')
    def set_expiry(self):
        """Set signal expiry if not provided"""
        if self.expires_at is None:
            # Signals expire at market close
            self.expires_at = datetime.combine(
                date.today(),
                time(16, 0)
            )
        return self


# =============================================================================
# ORDER MODELS
# =============================================================================

class OrderRequest(AVABaseModel):
    """Validated order request"""
    symbol: str = Field(..., min_length=1)
    action: OrderAction
    quantity: PositiveInt
    order_type: OrderType = OrderType.LIMIT
    limit_price: Optional[PositiveFloat] = None
    stop_price: Optional[PositiveFloat] = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Multi-leg support
    legs: Optional[List[Dict[str, Any]]] = None

    # Metadata
    client_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode='after')
    def validate_prices(self):
        """Validate price requirements based on order type"""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require limit_price")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require stop_price")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Stop-limit orders require both limit_price and stop_price")
        return self


class OrderResponse(AVABaseModel):
    """Validated order response"""
    order_id: str
    client_order_id: Optional[str] = None
    status: str
    filled_quantity: NonNegativeInt = 0
    average_fill_price: Optional[NonNegativeFloat] = None
    commission: NonNegativeFloat = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    message: Optional[str] = None


# =============================================================================
# POSITION MODELS
# =============================================================================

class Position(AVABaseModel):
    """Validated position"""
    id: str
    symbol: str
    underlying_symbol: str
    quantity: int = Field(..., ne=0)
    average_cost: NonNegativeFloat
    current_price: NonNegativeFloat = 0
    market_value: float = 0

    # Option-specific
    strike: Optional[PositiveFloat] = None
    expiration: Optional[date] = None
    option_type: Optional[OptionType] = None

    # Greeks
    delta: Delta = 0
    gamma: Gamma = 0
    theta: Theta = 0
    vega: Vega = 0
    iv: IV = 0.25

    # P&L
    unrealized_pnl: float = 0
    unrealized_pnl_pct: Percentage = 0
    realized_pnl: float = 0

    # Metadata
    opened_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_option(self) -> bool:
        return self.option_type is not None

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def days_to_expiration(self) -> Optional[int]:
        if self.expiration:
            return max(0, (self.expiration - date.today()).days)
        return None

    @property
    def position_delta(self) -> float:
        """Position delta (delta * quantity * multiplier)"""
        return self.delta * self.quantity * 100


# =============================================================================
# MARKET DATA MODELS
# =============================================================================

class MarketContext(AVABaseModel):
    """Validated market context"""
    symbol: str
    underlying_price: PositiveFloat

    # Volatility
    iv_rank: IVRank = 50
    iv_percentile: IVRank = 50
    hv_20: IV = 0.20
    hv_60: IV = 0.20
    vix: NonNegativeFloat = 15

    # Fundamentals
    sector: Optional[str] = None
    market_cap: Optional[PositiveFloat] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[NonNegativeFloat] = None

    # Events
    days_to_earnings: Optional[NonNegativeInt] = None
    earnings_date: Optional[date] = None
    ex_dividend_date: Optional[date] = None

    # Technical
    trend: Optional[str] = None
    support_level: Optional[PositiveFloat] = None
    resistance_level: Optional[PositiveFloat] = None
    rsi: Optional[float] = Field(default=None, ge=0, le=100)

    # Timestamp
    as_of: datetime = Field(default_factory=datetime.now)

    @property
    def iv_premium(self) -> float:
        """IV premium over historical vol"""
        if self.hv_20 > 0:
            return (self.iv_rank / 100 * 0.5 - self.hv_20) / self.hv_20 * 100
        return 0

    @property
    def has_upcoming_earnings(self) -> bool:
        """Check if earnings within 14 days"""
        return self.days_to_earnings is not None and self.days_to_earnings <= 14


# =============================================================================
# RISK MODELS
# =============================================================================

class RiskLimits(AVABaseModel):
    """Validated risk limits"""
    # Portfolio limits
    max_portfolio_delta: float = Field(default=500, ge=0, le=10000)
    max_portfolio_gamma: float = Field(default=100, ge=0, le=1000)
    max_portfolio_theta: float = Field(default=-500, le=0)
    max_portfolio_vega: float = Field(default=1000, ge=0, le=10000)

    # Position limits
    max_position_size_pct: Percentage = Field(default=5, ge=0.1, le=25)
    max_single_underlying_pct: Percentage = Field(default=20, ge=1, le=50)
    max_sector_exposure_pct: Percentage = Field(default=40, ge=5, le=80)

    # Risk limits
    max_var_95_pct: Percentage = Field(default=3, ge=0.1, le=20)
    max_daily_loss_pct: Percentage = Field(default=2, ge=0.1, le=10)
    max_weekly_loss_pct: Percentage = Field(default=5, ge=0.5, le=25)

    # Strategy limits
    max_dte: PositiveInt = Field(default=60, le=365)
    min_iv_rank: IVRank = Field(default=20, ge=0)
    max_bid_ask_spread_pct: Percentage = Field(default=10, ge=0.5, le=50)
    min_open_interest: NonNegativeInt = Field(default=100, ge=0)
    min_volume: NonNegativeInt = Field(default=10, ge=0)


class RiskAnalysis(AVABaseModel):
    """Validated risk analysis result"""
    portfolio_value: PositiveFloat

    # Greeks
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float

    # VaR
    var_95: float
    var_99: float
    var_95_pct: Percentage
    var_99_pct: Percentage

    # Stress tests
    stress_test_results: Dict[str, float] = Field(default_factory=dict)
    max_stress_loss: float = 0
    max_stress_loss_pct: Percentage = 0

    # Violations
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Recommendations
    recommended_hedges: List[str] = Field(default_factory=list)
    position_adjustments: List[str] = Field(default_factory=list)

    # Metadata
    analyzed_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_within_limits(self) -> bool:
        return len(self.violations) == 0


# =============================================================================
# BACKTEST MODELS
# =============================================================================

class BacktestConfig(AVABaseModel):
    """Validated backtest configuration"""
    start_date: date
    end_date: date
    initial_capital: PositiveFloat = 100000

    # Costs
    slippage_pct: Percentage = Field(default=0.1, ge=0, le=5)
    commission_per_contract: NonNegativeFloat = Field(default=0.65, ge=0, le=10)

    # Filters
    min_iv_rank: IVRank = Field(default=20, ge=0)
    max_iv_rank: IVRank = Field(default=80, le=100)
    min_dte: NonNegativeInt = Field(default=21, ge=0)
    max_dte: NonNegativeInt = Field(default=45, le=365)
    min_open_interest: NonNegativeInt = Field(default=100, ge=0)
    max_bid_ask_spread_pct: Percentage = Field(default=10, ge=0)

    # Risk
    max_position_size_pct: Percentage = Field(default=5, ge=0.1, le=50)
    profit_target_pct: Percentage = Field(default=50, ge=1, le=200)
    stop_loss_pct: Percentage = Field(default=200, ge=10, le=500)

    # Simulation
    monte_carlo_runs: NonNegativeInt = Field(default=0, ge=0, le=10000)
    random_seed: Optional[int] = None

    @model_validator(mode='after')
    def validate_dates(self):
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.min_dte > self.max_dte:
            raise ValueError("min_dte cannot exceed max_dte")
        if self.min_iv_rank > self.max_iv_rank:
            raise ValueError("min_iv_rank cannot exceed max_iv_rank")
        return self


class BacktestTrade(AVABaseModel):
    """Validated backtest trade"""
    trade_id: str
    symbol: str
    strategy_name: str
    entry_date: date
    exit_date: date
    entry_price: PositiveFloat
    exit_price: NonNegativeFloat
    quantity: PositiveInt
    realized_pnl: float
    realized_pnl_pct: Percentage
    max_profit: float
    max_loss: float
    commission: NonNegativeFloat = 0
    slippage: NonNegativeFloat = 0
    exit_reason: str = "unknown"

    @model_validator(mode='after')
    def validate_dates(self):
        if self.entry_date > self.exit_date:
            raise ValueError("entry_date cannot be after exit_date")
        return self


class BacktestResult(AVABaseModel):
    """Validated backtest result"""
    config: BacktestConfig
    trades: List[BacktestTrade]

    # Performance
    total_return: Percentage
    annual_return: Percentage
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: Percentage
    max_drawdown_duration_days: NonNegativeInt

    # Statistics
    total_trades: NonNegativeInt
    winning_trades: NonNegativeInt
    losing_trades: NonNegativeInt
    win_rate: Probability
    avg_win: float
    avg_loss: float
    profit_factor: NonNegativeFloat

    # Time analysis
    avg_days_in_trade: NonNegativeFloat
    avg_days_to_profit_target: Optional[NonNegativeFloat] = None

    # Risk metrics
    var_95: float = 0
    expected_shortfall: float = 0

    # Metadata
    completed_at: datetime = Field(default_factory=datetime.now)
    duration_seconds: NonNegativeFloat = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_option_symbol(symbol: str) -> bool:
    """Validate OCC option symbol format"""
    # OCC format: ROOT + YYMMDD + C/P + STRIKE (8 digits)
    pattern = r'^[A-Z]{1,6}\d{6}[CP]\d{8}$'
    return bool(re.match(pattern, symbol.upper()))


def parse_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Parse OCC option symbol into components"""
    pattern = r'^([A-Z]{1,6})(\d{6})([CP])(\d{8})$'
    match = re.match(pattern, symbol.upper())

    if not match:
        return None

    root, date_str, opt_type, strike_str = match.groups()

    return {
        'underlying': root,
        'expiration': datetime.strptime(date_str, '%y%m%d').date(),
        'option_type': 'call' if opt_type == 'C' else 'put',
        'strike': int(strike_str) / 1000  # OCC uses 8 digits with 3 decimals
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Validation Models ===\n")

    # Test OptionLeg
    print("1. Testing OptionLeg validation...")
    try:
        leg = OptionLeg(
            symbol="AAPL241220C00230000",
            underlying_symbol="AAPL",
            strike=230.0,
            expiration=date(2024, 12, 20),
            option_type=OptionType.CALL,
            quantity=1,
            bid=5.50,
            ask=5.60,
            delta=0.45
        )
        print(f"   Valid leg created: {leg.symbol}")
        print(f"   Mid price: ${leg.mid_price:.2f}")
        print(f"   Spread: {leg.spread_pct:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")

    # Test invalid data
    print("\n2. Testing validation errors...")
    try:
        bad_leg = OptionLeg(
            symbol="AAPL",
            underlying_symbol="AAPL",
            strike=-100,  # Invalid
            expiration=date(2020, 1, 1),  # Past date
            option_type=OptionType.CALL,
            quantity=0,  # Invalid
            bid=10,
            ask=5  # Bid > Ask
        )
    except Exception as e:
        print(f"   Caught expected error: {type(e).__name__}")

    # Test RiskLimits
    print("\n3. Testing RiskLimits...")
    limits = RiskLimits()
    print(f"   Max portfolio delta: {limits.max_portfolio_delta}")
    print(f"   Max position size: {limits.max_position_size_pct}%")

    print("\n✅ Validation models ready!")
