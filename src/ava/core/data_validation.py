"""
Data Validation Layer
=====================

Centralized validation for all external data sources.
Ensures data integrity across the platform.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int"""
    if value is None:
        return default
    try:
        return int(float(value))  # Handle "1.0" strings
    except (ValueError, TypeError):
        return default


def safe_decimal(value: Any, default: Decimal = Decimal('0')) -> Decimal:
    """Safely convert value to Decimal for precise financial calculations"""
    if value is None:
        return default
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return default


def safe_date(value: Any, formats: List[str] = None) -> Optional[date]:
    """Safely parse date from various formats"""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    formats = formats or ['%Y-%m-%d', '%m/%d/%Y', '%Y%m%d', '%d-%m-%Y']
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def safe_datetime(value: Any, formats: List[str] = None) -> Optional[datetime]:
    """Safely parse datetime from various formats"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    formats = formats or [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d',
        '%m/%d/%Y %H:%M:%S'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    return None


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(max_val, value))


def validate_percentage(value: Any, allow_negative: bool = False) -> float:
    """Validate percentage value (0-100 or 0-1 format)"""
    val = safe_float(value)

    # Handle 0-1 format (convert to percentage)
    if 0 <= val <= 1:
        val = val * 100

    # Clamp to valid range
    if allow_negative:
        return clamp(val, -100, 1000)  # Allow up to 1000% for some metrics
    return clamp(val, 0, 1000)


def validate_price(value: Any) -> float:
    """Validate price value (must be positive or zero)"""
    val = safe_float(value)
    return max(0.0, val)


def validate_quantity(value: Any) -> int:
    """Validate quantity (must be non-negative integer)"""
    val = safe_int(value)
    return max(0, val)


# =============================================================================
# VALIDATED DATA MODELS
# =============================================================================

class ValidatedOptionData(BaseModel):
    """Validated options contract data"""
    symbol: str
    underlying: str = ""
    strike: float = Field(ge=0)
    expiration: date
    option_type: str = Field(pattern="^(call|put|C|P)$")

    # Prices (all non-negative)
    bid: float = Field(ge=0, default=0.0)
    ask: float = Field(ge=0, default=0.0)
    last: float = Field(ge=0, default=0.0)
    mark: float = Field(ge=0, default=0.0)

    # Volume and OI
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)

    # Greeks (with reasonable bounds)
    delta: float = Field(ge=-1, le=1, default=0.0)
    gamma: float = Field(ge=0, default=0.0)
    theta: float = Field(le=0, default=0.0)  # Theta is always negative or zero
    vega: float = Field(ge=0, default=0.0)
    rho: float = Field(default=0.0)

    # IV
    iv: float = Field(ge=0, le=10, default=0.0)  # 0-1000%

    @field_validator('option_type')
    @classmethod
    def normalize_option_type(cls, v: str) -> str:
        return 'call' if v.upper() in ('C', 'CALL') else 'put'

    @field_validator('iv')
    @classmethod
    def convert_iv_to_decimal(cls, v: float) -> float:
        # If IV > 10, assume it's in percentage form (e.g., 45.5%)
        if v > 10:
            return v / 100
        return v

    @model_validator(mode='after')
    def calculate_mark_if_missing(self):
        if self.mark == 0 and (self.bid > 0 or self.ask > 0):
            self.mark = (self.bid + self.ask) / 2
        return self

    @property
    def dte(self) -> int:
        """Days to expiration"""
        return (self.expiration - date.today()).days

    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.ask - self.bid if self.ask > self.bid else 0

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mark"""
        if self.mark > 0:
            return (self.spread / self.mark) * 100
        return 0


class ValidatedStockData(BaseModel):
    """Validated stock/underlying data"""
    symbol: str
    price: float = Field(ge=0)

    # OHLC
    open: float = Field(ge=0, default=0.0)
    high: float = Field(ge=0, default=0.0)
    low: float = Field(ge=0, default=0.0)
    close: float = Field(ge=0, default=0.0)
    previous_close: float = Field(ge=0, default=0.0)

    # Volume
    volume: int = Field(ge=0, default=0)
    avg_volume: int = Field(ge=0, default=0)

    # Changes
    change: float = 0.0
    change_pct: float = 0.0

    # Market data
    market_cap: float = Field(ge=0, default=0.0)
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None

    # Timestamps
    timestamp: Optional[datetime] = None

    @model_validator(mode='after')
    def calculate_change(self):
        if self.previous_close > 0 and self.change == 0:
            self.change = self.price - self.previous_close
            self.change_pct = (self.change / self.previous_close) * 100
        return self


class ValidatedEarningsData(BaseModel):
    """Validated earnings event data"""
    symbol: str
    earnings_date: date

    # Timing
    before_market: bool = False
    after_market: bool = False

    # Estimates
    eps_estimate: Optional[float] = None
    revenue_estimate: Optional[float] = None

    # Actual (after announcement)
    eps_actual: Optional[float] = None
    revenue_actual: Optional[float] = None

    # Historical
    eps_surprise_pct: Optional[float] = None
    avg_move_pct: Optional[float] = None

    @property
    def days_until(self) -> int:
        return (self.earnings_date - date.today()).days

    @property
    def has_reported(self) -> bool:
        return self.eps_actual is not None


class ValidatedPredictionMarket(BaseModel):
    """Validated prediction market data (Kalshi, etc.)"""
    market_id: str
    title: str

    # Prices (0-100 or 0-1)
    yes_price: float = Field(ge=0, le=100)
    no_price: float = Field(ge=0, le=100)

    # Volume
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)

    # Status
    status: str = "open"
    close_time: Optional[datetime] = None

    # Metadata
    category: str = ""
    subcategory: str = ""

    @field_validator('yes_price', 'no_price')
    @classmethod
    def normalize_price(cls, v: float) -> float:
        # Convert 0-1 to 0-100 if needed
        if 0 <= v <= 1:
            return v * 100
        return v

    @property
    def implied_prob_yes(self) -> float:
        """Implied probability of YES outcome"""
        return self.yes_price / 100

    @property
    def implied_prob_no(self) -> float:
        """Implied probability of NO outcome"""
        return self.no_price / 100


class ValidatedSportsGame(BaseModel):
    """Validated sports game data"""
    game_id: str
    league: str

    home_team: str
    away_team: str

    # Game time
    game_time: Optional[datetime] = None
    is_live: bool = False

    # Score (if in progress or finished)
    home_score: int = Field(ge=0, default=0)
    away_score: int = Field(ge=0, default=0)

    # Odds (American format)
    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None
    spread_home: Optional[float] = None
    over_under: Optional[float] = None

    # Game state
    quarter: Optional[int] = None
    time_remaining: Optional[str] = None
    possession: Optional[str] = None

    @property
    def implied_prob_home(self) -> Optional[float]:
        """Convert moneyline to implied probability"""
        if self.moneyline_home is None:
            return None
        if self.moneyline_home > 0:
            return 100 / (self.moneyline_home + 100)
        else:
            return abs(self.moneyline_home) / (abs(self.moneyline_home) + 100)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_option_chain(raw_data: List[Dict]) -> List[ValidatedOptionData]:
    """Validate entire option chain"""
    validated = []
    for item in raw_data:
        try:
            # Map common field names
            mapped = {
                'symbol': item.get('symbol', item.get('contractSymbol', '')),
                'underlying': item.get('underlying', item.get('underlyingSymbol', '')),
                'strike': safe_float(item.get('strike', item.get('strikePrice', 0))),
                'expiration': safe_date(item.get('expiration', item.get('expirationDate'))),
                'option_type': item.get('option_type', item.get('type', item.get('putCall', 'call'))),
                'bid': safe_float(item.get('bid', 0)),
                'ask': safe_float(item.get('ask', 0)),
                'last': safe_float(item.get('last', item.get('lastPrice', 0))),
                'mark': safe_float(item.get('mark', 0)),
                'volume': safe_int(item.get('volume', 0)),
                'open_interest': safe_int(item.get('open_interest', item.get('openInterest', 0))),
                'delta': safe_float(item.get('delta', 0)),
                'gamma': safe_float(item.get('gamma', 0)),
                'theta': safe_float(item.get('theta', 0)),
                'vega': safe_float(item.get('vega', 0)),
                'iv': safe_float(item.get('iv', item.get('impliedVolatility', 0))),
            }

            if mapped['expiration'] is None:
                continue  # Skip invalid dates

            validated.append(ValidatedOptionData(**mapped))
        except Exception as e:
            logger.debug(f"Skipping invalid option: {e}")
            continue

    return validated


def validate_stock_quote(raw_data: Dict) -> Optional[ValidatedStockData]:
    """Validate stock quote data"""
    try:
        mapped = {
            'symbol': raw_data.get('symbol', ''),
            'price': safe_float(raw_data.get('price', raw_data.get('regularMarketPrice', 0))),
            'open': safe_float(raw_data.get('open', raw_data.get('regularMarketOpen', 0))),
            'high': safe_float(raw_data.get('high', raw_data.get('regularMarketDayHigh', 0))),
            'low': safe_float(raw_data.get('low', raw_data.get('regularMarketDayLow', 0))),
            'close': safe_float(raw_data.get('close', raw_data.get('regularMarketPrice', 0))),
            'previous_close': safe_float(raw_data.get('previous_close', raw_data.get('regularMarketPreviousClose', 0))),
            'volume': safe_int(raw_data.get('volume', raw_data.get('regularMarketVolume', 0))),
            'avg_volume': safe_int(raw_data.get('avg_volume', raw_data.get('averageVolume', 0))),
            'market_cap': safe_float(raw_data.get('market_cap', raw_data.get('marketCap', 0))),
        }
        return ValidatedStockData(**mapped)
    except Exception as e:
        logger.warning(f"Failed to validate stock data: {e}")
        return None


def validate_earnings_calendar(raw_data: List[Dict]) -> List[ValidatedEarningsData]:
    """Validate earnings calendar data"""
    validated = []
    for item in raw_data:
        try:
            earnings_date = safe_date(item.get('date', item.get('earningsDate')))
            if earnings_date is None:
                continue

            # Determine timing
            time_str = str(item.get('time', item.get('earningsTime', ''))).lower()
            before_market = 'before' in time_str or 'bmo' in time_str or 'pre' in time_str
            after_market = 'after' in time_str or 'amc' in time_str or 'post' in time_str

            validated.append(ValidatedEarningsData(
                symbol=item.get('symbol', ''),
                earnings_date=earnings_date,
                before_market=before_market,
                after_market=after_market,
                eps_estimate=safe_float(item.get('eps_estimate', item.get('epsEstimate'))) or None,
                revenue_estimate=safe_float(item.get('revenue_estimate', item.get('revenueEstimate'))) or None,
                avg_move_pct=safe_float(item.get('avg_move', item.get('avgMove'))) or None,
            ))
        except Exception as e:
            logger.debug(f"Skipping invalid earnings: {e}")
            continue

    return validated


def validate_response(
    raw_data: Dict[str, Any],
    model_class: Type[T],
    field_mapping: Optional[Dict[str, str]] = None
) -> Optional[T]:
    """
    Generic response validator.

    Args:
        raw_data: Raw API response
        model_class: Pydantic model to validate against
        field_mapping: Optional field name remapping

    Returns:
        Validated model instance or None
    """
    try:
        if field_mapping:
            mapped = {}
            for model_field, api_field in field_mapping.items():
                if api_field in raw_data:
                    mapped[model_field] = raw_data[api_field]
                elif model_field in raw_data:
                    mapped[model_field] = raw_data[model_field]
            return model_class(**mapped)
        return model_class(**raw_data)
    except Exception as e:
        logger.warning(f"Validation failed for {model_class.__name__}: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Utilities
    'safe_float', 'safe_int', 'safe_decimal', 'safe_date', 'safe_datetime',
    'clamp', 'validate_percentage', 'validate_price', 'validate_quantity',

    # Models
    'ValidatedOptionData', 'ValidatedStockData', 'ValidatedEarningsData',
    'ValidatedPredictionMarket', 'ValidatedSportsGame',

    # Functions
    'validate_option_chain', 'validate_stock_quote', 'validate_earnings_calendar',
    'validate_response',
]
