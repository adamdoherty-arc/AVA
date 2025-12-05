"""
AVA Centralized Configuration
=============================

Environment-based configuration system with validation.
All hardcoded values are moved here for easy customization.

Author: AVA Trading Platform
Created: 2025-11-28
"""

import os
from datetime import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from functools import lru_cache
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# BASE CONFIGURATION
# =============================================================================

class AVASettings(BaseSettings):
    """Base settings with environment variable support"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AVA_",
        case_sensitive=False,
        extra="ignore"
    )


# =============================================================================
# API CONFIGURATION
# =============================================================================

class APISettings(AVASettings):
    """API keys and endpoints"""

    # Robinhood
    robinhood_username: Optional[str] = Field(default=None, alias="ROBINHOOD_USERNAME")
    robinhood_password: Optional[str] = Field(default=None, alias="ROBINHOOD_PASSWORD")
    robinhood_mfa_code: Optional[str] = Field(default=None, alias="ROBINHOOD_MFA_CODE")

    # Market Data
    polygon_api_key: Optional[str] = Field(default=None, alias="POLYGON_API_KEY")
    alpha_vantage_key: Optional[str] = Field(default=None, alias="ALPHA_VANTAGE_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, alias="FINNHUB_API_KEY")

    # AI/LLM
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/ava",
        alias="DATABASE_URL"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL"
    )

    # Notifications
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")
    discord_webhook_url: Optional[str] = Field(default=None, alias="DISCORD_WEBHOOK_URL")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


# =============================================================================
# MARKET HOURS CONFIGURATION
# =============================================================================

class MarketHoursSettings(AVASettings):
    """Trading hours configuration"""

    # Regular market hours (Eastern Time)
    market_open: time = Field(default=time(9, 30))
    market_close: time = Field(default=time(16, 0))

    # Extended hours
    premarket_open: time = Field(default=time(4, 0))
    afterhours_close: time = Field(default=time(20, 0))

    # 0DTE specific
    zero_dte_entry_start: time = Field(default=time(9, 45))
    zero_dte_entry_end: time = Field(default=time(14, 30))
    zero_dte_force_close: time = Field(default=time(15, 45))

    # Timezone
    timezone: str = Field(default="America/New_York")


# =============================================================================
# STRATEGY CONFIGURATION
# =============================================================================

class StrategySettings(AVASettings):
    """Strategy parameters - previously hardcoded values"""

    # Supported symbols for various strategies
    supported_symbols: List[str] = Field(
        default=[
            "SPY", "QQQ", "IWM", "DIA",  # ETFs
            "AAPL", "MSFT", "NVDA", "AMD", "TSLA",  # Tech
            "AMZN", "GOOGL", "META",  # FAANG
            "JPM", "BAC", "GS",  # Finance
            "XOM", "CVX",  # Energy
        ]
    )

    # Zero DTE symbols
    zero_dte_symbols: List[str] = Field(
        default=["SPY", "QQQ", "SPX", "IWM", "DIA", "AAPL", "TSLA", "NVDA", "AMD", "AMZN"]
    )

    # Wheel Strategy
    wheel_target_delta: float = Field(default=0.30, ge=0.1, le=0.5)
    wheel_min_iv_rank: float = Field(default=30.0, ge=0, le=100)
    wheel_min_dte: int = Field(default=21, ge=7, le=90)
    wheel_max_dte: int = Field(default=45, ge=14, le=120)
    wheel_profit_target_pct: float = Field(default=50.0, ge=10, le=90)
    wheel_call_strike_above_cost: float = Field(default=0.02, ge=0, le=0.10)

    # Iron Condor
    ic_target_short_delta: float = Field(default=0.20, ge=0.10, le=0.35)
    ic_min_wing_width: float = Field(default=5.0, ge=1, le=50)
    ic_max_wing_width: float = Field(default=20.0, ge=5, le=100)
    ic_wing_width_pct: float = Field(default=0.02, ge=0.005, le=0.10)
    ic_min_iv_rank: float = Field(default=40.0, ge=0, le=100)
    ic_dte_close_threshold: int = Field(default=14, ge=1, le=30)
    ic_short_strike_buffer: float = Field(default=0.02, ge=0, le=0.10)
    ic_profit_target_pct: float = Field(default=50.0, ge=10, le=90)
    ic_stop_loss_pct: float = Field(default=200.0, ge=50, le=500)

    # Vertical Spreads
    vertical_short_delta: float = Field(default=0.30, ge=0.15, le=0.45)
    vertical_min_credit_pct: float = Field(default=0.25, ge=0.10, le=0.50)
    vertical_default_width: int = Field(default=5, ge=1, le=50)

    # Calendar Spreads
    calendar_front_dte_min: int = Field(default=14, ge=7, le=45)
    calendar_front_dte_max: int = Field(default=35, ge=14, le=60)
    calendar_back_dte_min: int = Field(default=40, ge=30, le=90)
    calendar_back_dte_max: int = Field(default=75, ge=45, le=120)

    # Straddle/Strangle
    straddle_min_iv_rank_short: float = Field(default=50.0, ge=0, le=100)
    straddle_max_iv_rank_long: float = Field(default=40.0, ge=0, le=100)
    strangle_wing_delta: float = Field(default=0.16, ge=0.05, le=0.30)

    @field_validator('supported_symbols', 'zero_dte_symbols', mode='before')
    @classmethod
    def parse_symbols(cls, v):
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(',')]
        return [s.upper() for s in v]


# =============================================================================
# RISK MANAGEMENT CONFIGURATION
# =============================================================================

class RiskSettings(AVASettings):
    """Risk management parameters"""

    # Portfolio limits
    max_portfolio_delta: float = Field(default=500.0, ge=0, le=10000)
    max_portfolio_gamma: float = Field(default=100.0, ge=0, le=1000)
    max_portfolio_theta: float = Field(default=-500.0, le=0)
    max_portfolio_vega: float = Field(default=1000.0, ge=0, le=10000)

    # Position limits
    max_position_size_pct: float = Field(default=5.0, ge=0.5, le=25)
    max_single_underlying_pct: float = Field(default=20.0, ge=5, le=50)
    max_sector_exposure_pct: float = Field(default=40.0, ge=10, le=80)
    max_total_positions: int = Field(default=50, ge=1, le=200)

    # Daily/Weekly limits
    max_daily_loss_pct: float = Field(default=2.0, ge=0.5, le=10)
    max_weekly_loss_pct: float = Field(default=5.0, ge=1, le=25)
    max_monthly_loss_pct: float = Field(default=10.0, ge=2, le=50)

    # VaR limits
    max_var_95_pct: float = Field(default=3.0, ge=0.5, le=20)
    max_var_99_pct: float = Field(default=5.0, ge=1, le=30)

    # Position sizing
    default_risk_per_trade_pct: float = Field(default=2.0, ge=0.5, le=10)
    kelly_fraction: float = Field(default=0.25, ge=0.1, le=1.0)

    # Trading days per year (for annualization)
    trading_days_per_year: int = Field(default=252, ge=250, le=260)

    # Risk-free rate
    risk_free_rate: float = Field(default=0.05, ge=0, le=0.20)


# =============================================================================
# BACKTEST CONFIGURATION
# =============================================================================

class BacktestSettings(AVASettings):
    """Backtesting parameters"""

    # Default values
    default_initial_capital: float = Field(default=100000.0, ge=1000, le=10000000)
    default_slippage_pct: float = Field(default=0.1, ge=0, le=5)
    default_commission: float = Field(default=0.65, ge=0, le=10)

    # Filters
    min_open_interest: int = Field(default=100, ge=0, le=10000)
    min_volume: int = Field(default=10, ge=0, le=1000)
    max_bid_ask_spread_pct: float = Field(default=10.0, ge=0.5, le=50)

    # Monte Carlo
    default_monte_carlo_runs: int = Field(default=1000, ge=0, le=100000)

    # Performance
    use_vectorized_calculations: bool = Field(default=True)
    parallel_workers: int = Field(default=4, ge=1, le=32)


# =============================================================================
# STREAMING CONFIGURATION
# =============================================================================

class StreamingSettings(AVASettings):
    """Real-time streaming parameters"""

    # Update intervals (seconds)
    position_update_interval: float = Field(default=1.0, ge=0.1, le=60)
    greeks_update_interval: float = Field(default=5.0, ge=1, le=300)
    portfolio_update_interval: float = Field(default=10.0, ge=5, le=600)
    price_update_interval: float = Field(default=1.0, ge=0.1, le=60)

    # WebSocket settings
    websocket_heartbeat_interval: float = Field(default=30.0, ge=10, le=120)
    websocket_max_message_size: int = Field(default=1048576, ge=1024, le=10485760)  # 1MB default
    websocket_max_connections: int = Field(default=100, ge=1, le=10000)

    # Buffer settings
    alert_queue_size: int = Field(default=1000, ge=100, le=100000)
    price_buffer_size: int = Field(default=100, ge=10, le=10000)


# =============================================================================
# AI/LLM CONFIGURATION
# =============================================================================

class AISettings(AVASettings):
    """AI and LLM configuration"""

    # Provider selection (ollama=FREE local, groq=FREE cloud, huggingface=FREE)
    # Options: "ollama", "groq", "huggingface", "anthropic", "openai"
    provider: str = Field(default="groq", alias="AVA_PROVIDER")

    # Model selection per provider
    # Ollama: llama3.2, mistral, codellama, phi3, gemma2
    # Groq: llama-3.3-70b-versatile, mixtral-8x7b-32768 (FREE tier!)
    # HuggingFace: mistralai/Mistral-7B-Instruct-v0.3
    # Anthropic: claude-sonnet-4-20250514
    # OpenAI: gpt-4-turbo
    default_model: str = Field(
        default="llama-3.3-70b-versatile",
        alias="AVA_DEFAULT_MODEL"
    )
    fast_model: str = Field(default="llama-3.3-70b-versatile")
    reasoning_model: str = Field(default="llama-3.3-70b-versatile")

    # Ollama settings (local LLM server)
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    # Token limits
    max_input_tokens: int = Field(default=100000, ge=1000, le=200000)
    max_output_tokens: int = Field(default=4096, ge=256, le=8192)

    # Temperature settings
    analysis_temperature: float = Field(default=0.3, ge=0, le=1)
    creative_temperature: float = Field(default=0.7, ge=0, le=1)

    # Retry settings
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_delay: float = Field(default=1.0, ge=0.1, le=30)

    # Caching
    cache_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    enable_response_caching: bool = Field(default=True)

    # Agent settings
    enable_debate: bool = Field(default=True)
    debate_rounds: int = Field(default=2, ge=1, le=5)
    parallel_agents: bool = Field(default=True)
    agent_timeout_seconds: float = Field(default=30.0, ge=5, le=120)


# =============================================================================
# CACHING CONFIGURATION
# =============================================================================

class CacheSettings(AVASettings):
    """Caching configuration"""

    # Enable/disable
    enable_redis: bool = Field(default=True)
    enable_memory_cache: bool = Field(default=True)

    # TTL settings (seconds)
    option_chain_ttl: int = Field(default=60, ge=10, le=3600)
    greeks_ttl: int = Field(default=30, ge=5, le=300)
    market_data_ttl: int = Field(default=5, ge=1, le=60)
    analysis_ttl: int = Field(default=300, ge=60, le=3600)

    # Memory cache limits
    max_memory_cache_size: int = Field(default=1000, ge=100, le=100000)
    memory_cache_policy: str = Field(default="lru")

    # Connection pool
    redis_pool_size: int = Field(default=10, ge=1, le=100)
    redis_timeout: float = Field(default=5.0, ge=1, le=30)


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseSettings(AVASettings):
    """Database configuration"""

    # Connection pool
    pool_size: int = Field(default=5, ge=1, le=50)
    max_overflow: int = Field(default=10, ge=0, le=100)
    pool_timeout: float = Field(default=30.0, ge=5, le=120)
    pool_recycle: int = Field(default=3600, ge=300, le=86400)

    # Query settings
    query_timeout: float = Field(default=30.0, ge=5, le=300)
    slow_query_threshold: float = Field(default=1.0, ge=0.1, le=10)

    # Batch operations
    batch_size: int = Field(default=1000, ge=100, le=100000)
    bulk_insert_chunk_size: int = Field(default=500, ge=50, le=10000)


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class LoggingSettings(AVASettings):
    """Logging configuration"""

    level: str = Field(default="INFO")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_to_file: bool = Field(default=True)
    log_file_path: str = Field(default="logs/ava.log")
    max_log_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=20)

    # Performance logging
    log_slow_operations: bool = Field(default=True)
    slow_operation_threshold_ms: int = Field(default=1000, ge=100, le=30000)


# =============================================================================
# AGGREGATED CONFIGURATION
# =============================================================================

class AVAConfig:
    """
    Aggregated configuration manager.

    Usage:
        config = get_config()
        print(config.strategy.wheel_target_delta)
        print(config.risk.max_portfolio_delta)
    """

    def __init__(self) -> None:
        self.api = APISettings()
        self.market_hours = MarketHoursSettings()
        self.strategy = StrategySettings()
        self.risk = RiskSettings()
        self.backtest = BacktestSettings()
        self.streaming = StreamingSettings()
        self.ai = AISettings()
        self.cache = CacheSettings()
        self.database = DatabaseSettings()
        self.logging = LoggingSettings()

    def to_dict(self) -> Dict[str, Any]:
        """Export all settings as dictionary"""
        return {
            "api": self.api.model_dump(exclude={"robinhood_password", "anthropic_api_key", "openai_api_key"}),
            "market_hours": self.market_hours.model_dump(),
            "strategy": self.strategy.model_dump(),
            "risk": self.risk.model_dump(),
            "backtest": self.backtest.model_dump(),
            "streaming": self.streaming.model_dump(),
            "ai": self.ai.model_dump(),
            "cache": self.cache.model_dump(),
            "database": self.database.model_dump(),
            "logging": self.logging.model_dump(),
        }

    def validate_all(self) -> List[str]:
        """Validate all settings and return any warnings"""
        warnings = []

        # Check API keys
        if not self.api.anthropic_api_key and not self.api.openai_api_key:
            warnings.append("No AI API key configured - LLM features will be disabled")

        if not self.api.robinhood_username:
            warnings.append("Robinhood credentials not configured - live trading disabled")

        # Check risk settings
        if self.risk.max_position_size_pct > 10:
            warnings.append(f"High position size limit: {self.risk.max_position_size_pct}%")

        if self.risk.max_daily_loss_pct > 5:
            warnings.append(f"High daily loss limit: {self.risk.max_daily_loss_pct}%")

        return warnings


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_config_instance: Optional[AVAConfig] = None


@lru_cache(maxsize=1)
def get_config() -> AVAConfig:
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AVAConfig()
    return _config_instance


def reload_config() -> AVAConfig:
    """Force reload configuration from environment"""
    global _config_instance
    get_config.cache_clear()
    _config_instance = None
    return get_config()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("\n=== Testing Configuration System ===\n")

    config = get_config()

    print("1. Strategy Settings:")
    print(f"   Wheel target delta: {config.strategy.wheel_target_delta}")
    print(f"   IC short delta: {config.strategy.ic_target_short_delta}")
    print(f"   Supported symbols: {len(config.strategy.supported_symbols)} symbols")

    print("\n2. Risk Settings:")
    print(f"   Max portfolio delta: {config.risk.max_portfolio_delta}")
    print(f"   Max position size: {config.risk.max_position_size_pct}%")
    print(f"   Max daily loss: {config.risk.max_daily_loss_pct}%")

    print("\n3. AI Settings:")
    print(f"   Default model: {config.ai.default_model}")
    print(f"   Enable debate: {config.ai.enable_debate}")
    print(f"   Parallel agents: {config.ai.parallel_agents}")

    print("\n4. Cache Settings:")
    print(f"   Enable Redis: {config.cache.enable_redis}")
    print(f"   Option chain TTL: {config.cache.option_chain_ttl}s")

    print("\n5. Validation Warnings:")
    warnings = config.validate_all()
    for w in warnings:
        print(f"   - {w}")

    print("\n✅ Configuration system ready!")
