"""
Pydantic Models for Xtrades Trade Alert System
===============================================

Modern, type-safe data models with validation using Pydantic v2.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class TradeStrategy(str, Enum):
    """Supported trading strategies."""
    CALL = "call"
    PUT = "put"
    STOCK = "stock"
    SPREAD = "spread"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    UNKNOWN = "unknown"


class TradeAction(str, Enum):
    """Trade action types."""
    BTO = "bto"  # Buy to Open
    STO = "sto"  # Sell to Open
    BTC = "btc"  # Buy to Close
    STC = "stc"  # Sell to Close
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class AlertType(str, Enum):
    """Types of alerts."""
    ENTRY = "entry"
    EXIT = "exit"
    UPDATE = "update"
    WATCHLIST = "watchlist"
    ANALYSIS = "analysis"
    OTHER = "other"


class SentimentLevel(str, Enum):
    """AI-determined sentiment levels."""
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class RiskLevel(str, Enum):
    """AI-determined risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class XtradeProfile(BaseModel):
    """
    Xtrades user profile with validation.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    username: str = Field(..., min_length=1, max_length=100)
    display_name: Optional[str] = Field(None, max_length=200)
    profile_url: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    is_active: bool = True
    last_sync: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Ensure username is clean."""
        return v.strip().lower()


class TradeSignal(BaseModel):
    """
    Extracted trade signal from alert text.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(..., min_length=1, max_length=10, pattern=r'^[A-Z]{1,10}$')
    strategy: TradeStrategy = TradeStrategy.UNKNOWN
    action: TradeAction = TradeAction.UNKNOWN
    strike_price: Optional[Decimal] = Field(None, ge=0)
    expiration_date: Optional[datetime] = None
    entry_price: Optional[Decimal] = Field(None, ge=0)
    target_price: Optional[Decimal] = Field(None, ge=0)
    stop_loss: Optional[Decimal] = Field(None, ge=0)
    quantity: Optional[int] = Field(None, ge=1)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    raw_text: str = ""

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Ensure ticker is uppercase."""
        return v.upper().strip()

    @model_validator(mode='after')
    def validate_prices(self) -> 'TradeSignal':
        """Validate price relationships."""
        if self.target_price and self.stop_loss and self.entry_price:
            # For long positions, target should be above entry, stop below
            if self.action in [TradeAction.BTO, TradeAction.BUY]:
                if self.target_price < self.entry_price:
                    # Log warning but don't fail - might be intentional
                    pass
        return self


class XtradeAlert(BaseModel):
    """
    Xtrades alert/post with full validation.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    id: Optional[int] = None
    profile_id: int
    alert_id: str = Field(..., min_length=1)  # Xtrades unique ID
    alert_text: str = Field(..., min_length=1)
    alert_type: AlertType = AlertType.OTHER
    posted_at: datetime
    scraped_at: datetime = Field(default_factory=datetime.utcnow)

    # Extracted trade data
    ticker: Optional[str] = Field(None, max_length=10)
    strategy: Optional[str] = None
    action: Optional[str] = None
    strike_price: Optional[Decimal] = Field(None, ge=0)
    expiration_date: Optional[datetime] = None
    entry_price: Optional[Decimal] = Field(None, ge=0)
    target_price: Optional[Decimal] = Field(None, ge=0)
    stop_loss: Optional[Decimal] = Field(None, ge=0)
    quantity: Optional[int] = Field(None, ge=1)

    # AI analysis fields
    sentiment: Optional[SentimentLevel] = None
    risk_level: Optional[RiskLevel] = None
    ai_summary: Optional[str] = None
    ai_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Metadata
    raw_html: Optional[str] = None
    extra_data: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        """Ensure ticker is uppercase if present."""
        if v:
            return v.upper().strip()
        return v

    def has_trade_data(self) -> bool:
        """Check if alert contains actionable trade data."""
        return bool(self.ticker and (self.strategy or self.action or self.entry_price))

    def to_signal(self) -> Optional[TradeSignal]:
        """Convert alert to trade signal if valid."""
        if not self.has_trade_data() or not self.ticker:
            return None

        return TradeSignal(
            ticker=self.ticker,
            strategy=TradeStrategy(self.strategy) if self.strategy else TradeStrategy.UNKNOWN,
            action=TradeAction(self.action) if self.action else TradeAction.UNKNOWN,
            strike_price=self.strike_price,
            expiration_date=self.expiration_date,
            entry_price=self.entry_price,
            target_price=self.target_price,
            stop_loss=self.stop_loss,
            quantity=self.quantity,
            raw_text=self.alert_text
        )


class AIAnalysis(BaseModel):
    """
    AI-generated analysis of a trade alert.
    """
    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str
    ticker: str

    # Sentiment analysis
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_reasoning: str = ""

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list)

    # Trade quality
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # AI-generated content
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    suggested_action: Optional[str] = None

    # Technical analysis hints
    support_levels: List[Decimal] = Field(default_factory=list)
    resistance_levels: List[Decimal] = Field(default_factory=list)

    # Metadata
    model_used: str = "unknown"
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: int = 0
    tokens_used: int = 0

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        return v.upper().strip()


class SyncResult(BaseModel):
    """
    Result of a sync operation.
    """
    success: bool
    profile_username: str
    start_time: datetime
    end_time: datetime = Field(default_factory=datetime.utcnow)

    # Counts
    alerts_found: int = 0
    alerts_new: int = 0
    alerts_updated: int = 0
    alerts_failed: int = 0

    # Trade extraction
    trades_extracted: int = 0
    trades_with_ai: int = 0

    # Errors
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    # Performance
    duration_seconds: float = 0.0
    pages_scraped: int = 0

    @model_validator(mode='after')
    def calculate_duration(self) -> 'SyncResult':
        """Calculate duration from start/end times."""
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        return self

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.success = False

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Sync {status} for @{self.profile_username}",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Alerts: {self.alerts_found} found, {self.alerts_new} new, {self.alerts_updated} updated",
            f"  Trades: {self.trades_extracted} extracted, {self.trades_with_ai} analyzed",
        ]
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        return "\n".join(lines)


class SyncBatchResult(BaseModel):
    """
    Result of syncing multiple profiles.
    """
    start_time: datetime
    end_time: datetime = Field(default_factory=datetime.utcnow)

    profile_results: List[SyncResult] = Field(default_factory=list)

    @property
    def total_profiles(self) -> int:
        return len(self.profile_results)

    @property
    def successful_profiles(self) -> int:
        return sum(1 for r in self.profile_results if r.success)

    @property
    def total_alerts(self) -> int:
        return sum(r.alerts_found for r in self.profile_results)

    @property
    def total_new_alerts(self) -> int:
        return sum(r.alerts_new for r in self.profile_results)

    @property
    def total_trades(self) -> int:
        return sum(r.trades_extracted for r in self.profile_results)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def to_summary(self) -> str:
        """Generate batch summary."""
        lines = [
            f"Batch Sync Complete",
            f"  Profiles: {self.successful_profiles}/{self.total_profiles} successful",
            f"  Duration: {self.duration_seconds:.1f}s",
            f"  Total Alerts: {self.total_alerts} ({self.total_new_alerts} new)",
            f"  Total Trades: {self.total_trades}",
        ]
        return "\n".join(lines)
