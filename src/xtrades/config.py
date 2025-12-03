"""
Configuration for Xtrades Modern Module
========================================

Central configuration with:
- Environment variable loading
- Structured logging setup
- Database connection settings
- AI provider settings
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache

import structlog
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Database connection settings."""
    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    database: str = Field(default="trading_dashboard")
    user: str = Field(default="postgres")
    password: str = Field(default="")
    pool_min_size: int = Field(default=2)
    pool_max_size: int = Field(default=10)

    @property
    def url(self) -> str:
        """Build PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    @property
    def async_url(self) -> str:
        """Build async PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class AISettings(BaseModel):
    """AI/LLM provider settings."""
    enabled: bool = Field(default=True)
    provider: str = Field(default="auto")  # openai, anthropic, ollama, auto
    model: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=100, le=4000)
    timeout: int = Field(default=30, ge=5, le=120)
    max_concurrent: int = Field(default=5, ge=1, le=20)


class ScraperSettings(BaseModel):
    """Web scraper settings."""
    headless: bool = Field(default=True)
    timeout_ms: int = Field(default=30000)
    max_alerts_per_profile: int = Field(default=100)
    max_concurrent_profiles: int = Field(default=3)
    page_load_wait_ms: int = Field(default=5000)
    cache_dir: Optional[str] = Field(default=None)


class NotificationSettings(BaseModel):
    """Notification settings."""
    telegram_enabled: bool = Field(default=False)
    telegram_bot_token: Optional[str] = Field(default=None)
    telegram_chat_id: Optional[str] = Field(default=None)
    discord_webhook_url: Optional[str] = Field(default=None)


class XtradesSettings(BaseSettings):
    """
    Main settings class using Pydantic Settings.

    Loads configuration from environment variables with XTRADES_ prefix.
    """
    model_config = SettingsConfigDict(
        env_prefix="XTRADES_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Module settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    sync_interval_minutes: int = Field(default=5)

    # Sub-settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai: AISettings = Field(default_factory=AISettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)

    def __init__(self, **kwargs):
        """Initialize settings, loading from environment."""
        # Load database settings from standard env vars
        db_settings = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'trading_dashboard'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', ''),
        }

        # Load AI settings from env vars
        ai_settings = {
            'enabled': os.getenv('XTRADES_AI_ENABLED', 'true').lower() == 'true',
            'provider': os.getenv('LLM_PROVIDER', 'auto'),
        }
        if os.getenv('OPENAI_API_KEY'):
            ai_settings['provider'] = os.getenv('LLM_PROVIDER', 'openai')
        elif os.getenv('ANTHROPIC_API_KEY'):
            ai_settings['provider'] = os.getenv('LLM_PROVIDER', 'anthropic')

        # Load notification settings
        notif_settings = {
            'telegram_enabled': os.getenv('TELEGRAM_ENABLED', 'false').lower() == 'true',
            'telegram_bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
            'discord_webhook_url': os.getenv('DISCORD_WEBHOOK_URL'),
        }

        # Merge with provided kwargs
        merged = {
            'database': DatabaseSettings(**db_settings),
            'ai': AISettings(**ai_settings),
            'notifications': NotificationSettings(**notif_settings),
            **kwargs
        }

        super().__init__(**merged)


@lru_cache()
def get_settings() -> XtradesSettings:
    """Get cached settings instance."""
    return XtradesSettings()


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Configure structlog for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Output logs as JSON (for production)
        log_file: Optional file path for log output
    """
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper()),
        stream=sys.stdout
    )

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Pretty console output for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.rich_traceback
            )
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        logging.getLogger().addHandler(file_handler)


def get_cache_dir() -> Path:
    """Get the cache directory, creating if needed."""
    cache_dir = Path.home() / '.xtrades_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Initialize logging on import
configure_logging(level=os.getenv('LOG_LEVEL', 'INFO'))
