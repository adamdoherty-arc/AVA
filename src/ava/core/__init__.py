"""
AVA Core - Unified Chatbot Core Implementation
Modern architecture with LangGraph, structured outputs, and streaming

Enhanced with:
- Pydantic validation models
- Centralized configuration
- LLM-powered decision engine
- Multi-tier caching
- Async utilities with rate limiting
- Database connection pooling
- Comprehensive error handling
"""

# Original core components
from .ava_core import AVACore
from .state_manager import AVAStateManager
from .tool_registry import ToolRegistry
from .models import IntentResult, MessageResponse, ToolCall, AVAConfig as LegacyAVAConfig, ConversationState
from .agent_base import BaseAgent, AgentState
from .agent_registry import AgentRegistry
from .agent_initializer import initialize_all_agents, ensure_agents_initialized
from .multi_agent import AgentSupervisor, MultiAgentState
from .multi_agent_enhanced import EnhancedAgentSupervisor

# New validation models
from .validation import (
    OptionType, OrderAction, OrderType, TimeInForce, StrategyType, TradeAction, Conviction,
    AVABaseModel, TimestampedModel, OptionLeg, OptionChain, StrategySetup, TradeSignal,
    OrderRequest, OrderResponse, Position, MarketContext, RiskLimits, RiskAnalysis,
    BacktestConfig, BacktestTrade, BacktestResult,
)

# Configuration system
from .config import (
    AVAConfig, APISettings, StrategySettings, RiskSettings, AISettings,
    CacheSettings, StreamingSettings, get_config, reload_config,
)

# LLM Engine
from .llm_engine import (
    LLMClient, LLMProvider, LLMResponse, TradingAnalysisEngine,
    TradeAnalysis, StrategyRecommendationEngine,
)

# Caching layer
from .cache import (
    LRUCache, RedisCache, TieredCache, OptionChainCache, GreeksCache,
    cached, async_cached, get_option_chain_cache, get_greeks_cache, get_general_cache,
)

# Async utilities
from .async_utils import (
    RateLimiter, CircuitBreaker, CircuitBreakerOpen, CircuitState,
    ParallelProcessor, BatchCollector, SemaphorePool,
    RetryConfig, retry_with_backoff, with_retry, parallel_scan,
)

# Database utilities
from .database import (
    AsyncDatabaseManager, DatabaseConfig, BatchOperations, QueryBuilder,
    BaseRepository, TradeRepository, PositionRepository,
    get_database, close_database, check_database_health,
)

# Error handling
from .errors import (
    AVAError, ErrorCode, ErrorContext, ErrorHandler,
    ValidationError, ConfigurationError, APIError, RateLimitError,
    TradingError, InsufficientFundsError, OrderRejectedError, MarketClosedError,
    DataError, DataNotFoundError, StaleDataError,
    StrategyError, NoOpportunitiesError, RiskLimitExceededError, DatabaseError,
    RetryPolicy, should_retry, get_error_handler,
)

# API Client with retry, circuit breaker, and caching
from .api_client import (
    RobustAPIClient, APIClientConfig, ResponseCache,
    CircuitBreaker, CircuitState, TokenBucketRateLimiter,
    APIClientError, APIRequestError, CircuitBreakerOpen as APICircuitBreakerOpen,
    RateLimitExceeded, get_api_client, with_retry,
)

# HTTP Client helpers (easy migration from requests)
from .http_client import (
    http_get, http_post, async_http_get, async_http_post,
    ServiceClient, safe_request, safe_async_request,
    get_polygon_client, get_tradier_client, get_fred_client, get_kalshi_client,
)

# Data validation layer
from .data_validation import (
    safe_float, safe_int, safe_decimal, safe_date, safe_datetime,
    clamp, validate_percentage, validate_price, validate_quantity,
    ValidatedOptionData, ValidatedStockData, ValidatedEarningsData,
    ValidatedPredictionMarket, ValidatedSportsGame,
    validate_option_chain, validate_stock_quote, validate_earnings_calendar,
    validate_response,
)

__all__ = [
    # Original exports
    "AVACore", "AVAStateManager", "ToolRegistry",
    "IntentResult", "MessageResponse", "ToolCall", "LegacyAVAConfig", "ConversationState",
    "BaseAgent", "AgentState", "AgentRegistry",
    "initialize_all_agents", "ensure_agents_initialized",
    "AgentSupervisor", "MultiAgentState", "EnhancedAgentSupervisor",

    # Validation models
    "OptionType", "OrderAction", "OrderType", "TimeInForce",
    "StrategyType", "TradeAction", "Conviction",
    "AVABaseModel", "TimestampedModel",
    "OptionLeg", "OptionChain", "StrategySetup", "TradeSignal",
    "OrderRequest", "OrderResponse", "Position", "MarketContext",
    "RiskLimits", "RiskAnalysis",
    "BacktestConfig", "BacktestTrade", "BacktestResult",

    # Configuration
    "AVAConfig", "APISettings", "StrategySettings", "RiskSettings",
    "AISettings", "CacheSettings", "StreamingSettings",
    "get_config", "reload_config",

    # LLM Engine
    "LLMClient", "LLMProvider", "LLMResponse",
    "TradingAnalysisEngine", "TradeAnalysis", "StrategyRecommendationEngine",

    # Caching
    "LRUCache", "RedisCache", "TieredCache",
    "OptionChainCache", "GreeksCache",
    "cached", "async_cached",
    "get_option_chain_cache", "get_greeks_cache", "get_general_cache",

    # Async utilities
    "RateLimiter", "CircuitBreaker", "CircuitBreakerOpen", "CircuitState",
    "ParallelProcessor", "BatchCollector", "SemaphorePool",
    "RetryConfig", "retry_with_backoff", "with_retry", "parallel_scan",

    # Database
    "AsyncDatabaseManager", "DatabaseConfig", "BatchOperations", "QueryBuilder",
    "BaseRepository", "TradeRepository", "PositionRepository",
    "get_database", "close_database", "check_database_health",

    # Error handling
    "AVAError", "ErrorCode", "ErrorContext", "ErrorHandler",
    "ValidationError", "ConfigurationError", "APIError", "RateLimitError",
    "TradingError", "InsufficientFundsError", "OrderRejectedError", "MarketClosedError",
    "DataError", "DataNotFoundError", "StaleDataError",
    "StrategyError", "NoOpportunitiesError", "RiskLimitExceededError", "DatabaseError",
    "RetryPolicy", "should_retry", "get_error_handler",

    # API Client
    "RobustAPIClient", "APIClientConfig", "ResponseCache",
    "TokenBucketRateLimiter",
    "APIClientError", "APIRequestError", "APICircuitBreakerOpen", "RateLimitExceeded",
    "get_api_client",

    # HTTP Client helpers
    "http_get", "http_post", "async_http_get", "async_http_post",
    "ServiceClient", "safe_request", "safe_async_request",
    "get_polygon_client", "get_tradier_client", "get_fred_client", "get_kalshi_client",

    # Data validation
    "safe_float", "safe_int", "safe_decimal", "safe_date", "safe_datetime",
    "clamp", "validate_percentage", "validate_price", "validate_quantity",
    "ValidatedOptionData", "ValidatedStockData", "ValidatedEarningsData",
    "ValidatedPredictionMarket", "ValidatedSportsGame",
    "validate_option_chain", "validate_stock_quote", "validate_earnings_calendar",
    "validate_response",
]

