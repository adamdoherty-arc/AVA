"""
Modern Data Validation Infrastructure
======================================

Production-grade validation using Pydantic v2 with:
- Type-safe request/response models
- Custom validators
- Field constraints
- Serialization helpers
- Trading-specific validators

Author: AVA Trading Platform
Updated: 2025-11-29
"""

import re
from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)

# =============================================================================
# Custom Types & Annotations
# =============================================================================

# Stock symbol (1-5 uppercase letters)
StockSymbol = Annotated[
    str,
    Field(
        min_length=1,
        max_length=5,
        pattern=r"^[A-Z]{1,5}$",
        description="Stock ticker symbol (e.g., AAPL, MSFT)",
    ),
]

# Option symbol (OCC format)
OptionSymbol = Annotated[
    str,
    Field(
        min_length=15,
        max_length=21,
        description="OCC option symbol",
    ),
]

# Price with precision
Price = Annotated[
    Decimal,
    Field(ge=0, decimal_places=2, description="Price in USD"),
]

# Percentage (-100 to 100)
Percentage = Annotated[
    float,
    Field(ge=-100, le=100, description="Percentage value"),
]

# Score (0 to 100)
Score = Annotated[
    float,
    Field(ge=0, le=100, description="Score from 0 to 100"),
]

# Confidence (0 to 1)
Confidence = Annotated[
    float,
    Field(ge=0, le=1, description="Confidence score from 0 to 1"),
]

# Delta (-1 to 1)
Delta = Annotated[
    float,
    Field(ge=-1, le=1, description="Option delta"),
]

# IV (0 to 500%)
ImpliedVolatility = Annotated[
    float,
    Field(ge=0, le=500, description="Implied volatility percentage"),
]


# =============================================================================
# Enums
# =============================================================================


class OptionType(str, Enum):
    """Option type."""

    CALL = "call"
    PUT = "put"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Time in force."""

    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class PositionType(str, Enum):
    """Position type."""

    STOCK = "stock"
    OPTION = "option"
    CRYPTO = "crypto"


class StrategyType(str, Enum):
    """Options strategy type."""

    CASH_SECURED_PUT = "cash_secured_put"
    COVERED_CALL = "covered_call"
    IRON_CONDOR = "iron_condor"
    CREDIT_SPREAD = "credit_spread"
    DEBIT_SPREAD = "debit_spread"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    COLLAR = "collar"


class Trend(str, Enum):
    """Market trend."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class RiskLevel(str, Enum):
    """Risk tolerance level."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Sport(str, Enum):
    """Supported sports."""

    NFL = "nfl"
    NBA = "nba"
    MLB = "mlb"
    NHL = "nhl"
    NCAAF = "ncaaf"
    NCAAB = "ncaab"
    SOCCER = "soccer"


class BetType(str, Enum):
    """Bet type."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"
    PROP = "prop"
    PARLAY = "parlay"
    TEASER = "teaser"


# =============================================================================
# Base Models
# =============================================================================

T = TypeVar("T")


class AVABaseModel(BaseModel):
    """
    Base model for all AVA API models.

    Features:
    - Strict validation by default
    - JSON-compatible serialization
    - Exclude None values by default
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        use_enum_values=True,
        populate_by_name=True,
        json_schema_extra={"examples": []},
    )


class TimestampedModel(AVABaseModel):
    """Base model with automatic timestamps."""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class PaginatedResponse(AVABaseModel, Generic[T]):
    """Generic paginated response."""

    items: List[T]
    total: int
    page: int = 1
    page_size: int = 20
    has_more: bool = False

    @model_validator(mode="after")
    def set_has_more(self) -> "PaginatedResponse[T]":
        """Calculate has_more based on total and pagination."""
        self.has_more = (self.page * self.page_size) < self.total
        return self


# =============================================================================
# Trading Models
# =============================================================================


class StockQuote(AVABaseModel):
    """Real-time stock quote."""

    symbol: StockSymbol
    price: float = Field(ge=0, description="Current price")
    change: float = Field(description="Price change from previous close")
    change_percent: float = Field(description="Percentage change")
    volume: int = Field(ge=0, description="Trading volume")
    high: float = Field(ge=0, description="Day high")
    low: float = Field(ge=0, description="Day low")
    open: float = Field(ge=0, description="Open price")
    previous_close: float = Field(ge=0, description="Previous close")
    timestamp: datetime

    @field_validator("symbol", mode="before")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper() if isinstance(v, str) else v


class OptionContract(AVABaseModel):
    """Option contract details."""

    symbol: str = Field(description="OCC option symbol")
    underlying: StockSymbol
    option_type: OptionType
    strike: float = Field(gt=0, description="Strike price")
    expiration: date
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    last: Optional[float] = Field(ge=0, default=None)
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)

    # Greeks
    delta: Delta = 0.0
    gamma: float = Field(ge=0, default=0.0)
    theta: float = Field(default=0.0, description="Theta - negative for long options, positive for short")
    vega: float = Field(ge=0, default=0.0)
    rho: float = 0.0
    iv: ImpliedVolatility = 0.0

    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_percent(self) -> float:
        """Calculate spread as percentage of mid price."""
        mid = self.mid_price
        return (self.spread / mid * 100) if mid > 0 else 0

    @property
    def days_to_expiration(self) -> int:
        """Calculate days to expiration."""
        return (self.expiration - date.today()).days


class Position(AVABaseModel):
    """Trading position."""

    symbol: str
    position_type: PositionType
    quantity: int
    average_cost: float = Field(ge=0)
    current_price: float = Field(ge=0)
    market_value: float = Field(ge=0)
    unrealized_pnl: float
    unrealized_pnl_percent: float
    day_change: float = 0.0
    day_change_percent: float = 0.0

    # Option-specific
    option_type: Optional[OptionType] = None
    strike: Optional[float] = None
    expiration: Optional[date] = None
    delta: Optional[float] = None
    theta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None

    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable."""
        return self.unrealized_pnl > 0


class PortfolioSummary(AVABaseModel):
    """Portfolio summary."""

    total_value: float = Field(ge=0)
    cash: float = Field(ge=0)
    buying_power: float = Field(ge=0)
    day_change: float
    day_change_percent: float
    total_unrealized_pnl: float
    total_realized_pnl: float

    # Greeks aggregate
    total_delta: float = 0.0
    total_theta: float = 0.0
    total_gamma: float = 0.0
    total_vega: float = 0.0

    # Position counts
    stock_positions: int = 0
    option_positions: int = 0

    timestamp: datetime = Field(default_factory=datetime.now)


class TradeOrder(AVABaseModel):
    """Trade order request."""

    symbol: str
    side: OrderSide
    quantity: PositiveInt
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    limit_price: Optional[PositiveFloat] = None
    stop_price: Optional[PositiveFloat] = None

    @model_validator(mode="after")
    def validate_prices(self) -> "TradeOrder":
        """Validate limit/stop prices based on order type."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit price required for limit orders")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop price required for stop orders")
        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None or self.stop_price is None:
                raise ValueError("Both limit and stop prices required for stop-limit orders")
        return self


# =============================================================================
# AI Score Models
# =============================================================================


class StockAIScore(AVABaseModel):
    """AI-generated stock score."""

    symbol: StockSymbol
    company_name: str
    sector: str = "Unknown"

    # Price data
    current_price: float = Field(ge=0)
    daily_change_pct: float

    # AI Score
    ai_score: Score
    recommendation: Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
    confidence: Confidence

    # Components
    prediction_score: Score = 50.0
    technical_score: Score = 50.0
    sentiment_score: Score = 50.0
    volatility_score: Score = 50.0

    # Trend
    trend: Trend
    trend_strength: Confidence

    # Technicals
    rsi_14: float = Field(ge=0, le=100, default=50.0)
    macd_histogram: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0

    # Volatility
    iv_estimate: ImpliedVolatility = 30.0
    vol_regime: Literal["low", "normal", "elevated", "extreme"] = "normal"

    # Predictions
    predicted_change_1d: float = 0.0
    predicted_change_5d: float = 0.0

    # Levels
    support_price: float = 0.0
    resistance_price: float = 0.0

    # Meta
    market_cap: Optional[float] = None
    calculated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Sports Betting Models
# =============================================================================


class GameInfo(AVABaseModel):
    """Sports game information."""

    game_id: str
    sport: Sport
    home_team: str
    away_team: str
    start_time: datetime
    status: Literal["scheduled", "live", "final", "postponed"]

    # Scores (if live or final)
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # Venue
    venue: Optional[str] = None
    weather: Optional[str] = None


class GameOdds(AVABaseModel):
    """Game odds from sportsbook."""

    game_id: str

    # Moneyline
    home_ml: int = Field(description="Home team moneyline (e.g., -150)")
    away_ml: int = Field(description="Away team moneyline (e.g., +130)")

    # Spread
    spread: float = Field(description="Home team spread (e.g., -3.5)")
    spread_home_odds: int = -110
    spread_away_odds: int = -110

    # Total
    total: float = Field(description="Over/under total")
    over_odds: int = -110
    under_odds: int = -110

    # Source
    sportsbook: str = "consensus"
    updated_at: datetime = Field(default_factory=datetime.now)


class BetPrediction(AVABaseModel):
    """AI prediction for a bet."""

    game_id: str
    bet_type: BetType

    # Prediction
    predicted_winner: Optional[str] = None
    predicted_spread_cover: Optional[str] = None
    predicted_total: Optional[Literal["over", "under"]] = None

    # Probabilities
    win_probability: Confidence
    confidence: Confidence

    # Value
    edge: float = Field(description="Estimated edge vs. market odds")
    kelly_fraction: float = Field(ge=0, le=1, default=0.0)
    recommended_units: float = Field(ge=0, le=10, default=0.0)

    # Analysis
    key_factors: List[str] = Field(default_factory=list)
    reasoning: str

    generated_at: datetime = Field(default_factory=datetime.now)


class BetSlipLeg(AVABaseModel):
    """Single leg of a bet slip."""

    game_id: str
    bet_type: BetType
    selection: str = Field(description="Selected outcome")
    odds: int = Field(description="American odds")
    stake: Optional[float] = Field(ge=0, default=None)

    @property
    def decimal_odds(self) -> float:
        """Convert American odds to decimal."""
        if self.odds > 0:
            return (self.odds / 100) + 1
        return (100 / abs(self.odds)) + 1

    @property
    def implied_probability(self) -> float:
        """Calculate implied probability from odds."""
        decimal = self.decimal_odds
        return 1 / decimal if decimal > 0 else 0


class BetSlip(AVABaseModel):
    """Complete bet slip."""

    legs: List[BetSlipLeg] = Field(min_length=1, max_length=15)
    bet_mode: Literal["singles", "parlay"] = "singles"
    total_stake: float = Field(ge=0)

    @property
    def total_decimal_odds(self) -> float:
        """Calculate combined decimal odds for parlay."""
        if self.bet_mode == "parlay":
            odds = 1.0
            for leg in self.legs:
                odds *= leg.decimal_odds
            return odds
        return sum(leg.decimal_odds for leg in self.legs) / len(self.legs)

    @property
    def potential_payout(self) -> float:
        """Calculate potential payout."""
        if self.bet_mode == "parlay":
            return self.total_stake * self.total_decimal_odds
        return sum(
            (leg.stake or 0) * leg.decimal_odds for leg in self.legs
        )


# =============================================================================
# Request/Response Models
# =============================================================================


class SymbolRequest(AVABaseModel):
    """Request with stock symbol."""

    symbol: StockSymbol


class SymbolListRequest(AVABaseModel):
    """Request with list of symbols."""

    symbols: List[StockSymbol] = Field(min_length=1, max_length=100)


class DateRangeRequest(AVABaseModel):
    """Request with date range."""

    start_date: date
    end_date: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def validate_range(self) -> "DateRangeRequest":
        """Ensure start_date is before end_date."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
        return self


class PaginationRequest(AVABaseModel):
    """Pagination parameters."""

    page: PositiveInt = 1
    page_size: PositiveInt = Field(default=20, le=100)


class SearchRequest(AVABaseModel):
    """Search request."""

    query: str = Field(min_length=1, max_length=200)
    filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "desc"


class APIResponse(AVABaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(AVABaseModel):
    """Error response model."""

    success: bool = False
    error: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# =============================================================================
# Utility Functions
# =============================================================================


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize a stock symbol.

    Args:
        symbol: Stock symbol to validate

    Returns:
        Normalized uppercase symbol

    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol:
        raise ValueError("Symbol cannot be empty")

    symbol = symbol.strip().upper()

    if not re.match(r"^[A-Z]{1,5}$", symbol):
        raise ValueError(f"Invalid symbol format: {symbol}")

    return symbol


def validate_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse and validate an OCC option symbol.

    Format: AAPL  240119C00150000
            ^     ^     ^  ^
            |     |     |  Strike price * 1000
            |     |     C=Call, P=Put
            |     Expiration YYMMDD
            Underlying (6 chars, space-padded)

    Args:
        symbol: OCC option symbol

    Returns:
        Dict with underlying, expiration, type, strike

    Raises:
        ValueError: If symbol is invalid
    """
    if len(symbol) < 15:
        raise ValueError(f"Invalid option symbol length: {symbol}")

    try:
        underlying = symbol[:6].strip()
        exp_str = symbol[6:12]
        option_type = symbol[12]
        strike_str = symbol[13:]

        expiration = datetime.strptime(exp_str, "%y%m%d").date()
        strike = int(strike_str) / 1000

        return {
            "underlying": underlying,
            "expiration": expiration,
            "option_type": "call" if option_type == "C" else "put",
            "strike": strike,
        }
    except Exception as e:
        raise ValueError(f"Invalid option symbol: {symbol}") from e


def calculate_kelly_criterion(
    probability: float,
    odds: int,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly Criterion bet size.

    Args:
        probability: Win probability (0-1)
        odds: American odds
        fraction: Fraction of full Kelly to use (default 0.25 = quarter Kelly)

    Returns:
        Recommended bet fraction of bankroll
    """
    if probability <= 0 or probability >= 1:
        return 0.0

    # Convert American odds to decimal
    if odds > 0:
        decimal_odds = (odds / 100) + 1
    else:
        decimal_odds = (100 / abs(odds)) + 1

    b = decimal_odds - 1
    p = probability
    q = 1 - p

    # Kelly formula: f* = (bp - q) / b
    kelly = (b * p - q) / b if b > 0 else 0

    # Apply fraction and ensure non-negative
    return max(0, kelly * fraction)


def american_to_implied_probability(odds: int) -> float:
    """
    Convert American odds to implied probability.

    Args:
        odds: American odds (e.g., -150, +130)

    Returns:
        Implied probability (0-1)
    """
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def implied_probability_to_american(probability: float) -> int:
    """
    Convert implied probability to American odds.

    Args:
        probability: Implied probability (0-1)

    Returns:
        American odds
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")

    if probability >= 0.5:
        return int(-(probability / (1 - probability)) * 100)
    return int(((1 - probability) / probability) * 100)
