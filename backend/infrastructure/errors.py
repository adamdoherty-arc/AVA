"""
Modern Error Handling Infrastructure
=====================================

Production-grade error handling with:
- Typed exception hierarchy
- Error codes for API responses
- Retry policies
- Error context preservation
- Observability integration

Author: AVA Trading Platform
Updated: 2025-11-29
"""

import functools
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import structlog
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# =============================================================================
# Error Codes
# =============================================================================


class ErrorCode(IntEnum):
    """
    Standardized error codes for API responses.

    Ranges:
    - 1000-1999: Validation errors
    - 2000-2999: Authentication/Authorization
    - 3000-3999: Resource errors
    - 4000-4999: External service errors
    - 5000-5999: Internal errors
    - 6000-6999: Business logic errors
    """

    # Validation (1000-1999)
    VALIDATION_ERROR = 1000
    INVALID_INPUT = 1001
    MISSING_REQUIRED_FIELD = 1002
    INVALID_FORMAT = 1003
    VALUE_OUT_OF_RANGE = 1004
    INVALID_SYMBOL = 1005
    INVALID_DATE_RANGE = 1006

    # Authentication/Authorization (2000-2999)
    AUTHENTICATION_REQUIRED = 2000
    INVALID_CREDENTIALS = 2001
    TOKEN_EXPIRED = 2002
    INSUFFICIENT_PERMISSIONS = 2003
    ACCOUNT_LOCKED = 2004
    SESSION_EXPIRED = 2005

    # Resource (3000-3999)
    RESOURCE_NOT_FOUND = 3000
    SYMBOL_NOT_FOUND = 3001
    POSITION_NOT_FOUND = 3002
    ORDER_NOT_FOUND = 3003
    WATCHLIST_NOT_FOUND = 3004
    GAME_NOT_FOUND = 3005
    MARKET_NOT_FOUND = 3006

    # External Services (4000-4999)
    EXTERNAL_SERVICE_ERROR = 4000
    BROKER_CONNECTION_ERROR = 4001
    BROKER_API_ERROR = 4002
    MARKET_DATA_ERROR = 4003
    AI_SERVICE_ERROR = 4004
    DATABASE_ERROR = 4005
    CACHE_ERROR = 4006
    SPORTS_API_ERROR = 4007
    RATE_LIMIT_EXCEEDED = 4008
    CIRCUIT_BREAKER_OPEN = 4009

    # Internal (5000-5999)
    INTERNAL_ERROR = 5000
    CONFIGURATION_ERROR = 5001
    SERIALIZATION_ERROR = 5002
    INITIALIZATION_ERROR = 5003

    # Business Logic (6000-6999)
    BUSINESS_LOGIC_ERROR = 6000
    INSUFFICIENT_FUNDS = 6001
    MARKET_CLOSED = 6002
    TRADING_HALTED = 6003
    POSITION_LIMIT_EXCEEDED = 6004
    INVALID_ORDER = 6005
    RISK_LIMIT_EXCEEDED = 6006
    DUPLICATE_ORDER = 6007


# HTTP status code mapping
ERROR_CODE_TO_HTTP: Dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.MISSING_REQUIRED_FIELD: 400,
    ErrorCode.INVALID_FORMAT: 400,
    ErrorCode.VALUE_OUT_OF_RANGE: 400,
    ErrorCode.INVALID_SYMBOL: 400,
    ErrorCode.INVALID_DATE_RANGE: 400,
    ErrorCode.AUTHENTICATION_REQUIRED: 401,
    ErrorCode.INVALID_CREDENTIALS: 401,
    ErrorCode.TOKEN_EXPIRED: 401,
    ErrorCode.INSUFFICIENT_PERMISSIONS: 403,
    ErrorCode.ACCOUNT_LOCKED: 403,
    ErrorCode.SESSION_EXPIRED: 401,
    ErrorCode.RESOURCE_NOT_FOUND: 404,
    ErrorCode.SYMBOL_NOT_FOUND: 404,
    ErrorCode.POSITION_NOT_FOUND: 404,
    ErrorCode.ORDER_NOT_FOUND: 404,
    ErrorCode.WATCHLIST_NOT_FOUND: 404,
    ErrorCode.GAME_NOT_FOUND: 404,
    ErrorCode.MARKET_NOT_FOUND: 404,
    ErrorCode.EXTERNAL_SERVICE_ERROR: 502,
    ErrorCode.BROKER_CONNECTION_ERROR: 502,
    ErrorCode.BROKER_API_ERROR: 502,
    ErrorCode.MARKET_DATA_ERROR: 502,
    ErrorCode.AI_SERVICE_ERROR: 502,
    ErrorCode.DATABASE_ERROR: 503,
    ErrorCode.CACHE_ERROR: 503,
    ErrorCode.SPORTS_API_ERROR: 502,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.CIRCUIT_BREAKER_OPEN: 503,
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.CONFIGURATION_ERROR: 500,
    ErrorCode.SERIALIZATION_ERROR: 500,
    ErrorCode.INITIALIZATION_ERROR: 500,
    ErrorCode.BUSINESS_LOGIC_ERROR: 422,
    ErrorCode.INSUFFICIENT_FUNDS: 422,
    ErrorCode.MARKET_CLOSED: 422,
    ErrorCode.TRADING_HALTED: 422,
    ErrorCode.POSITION_LIMIT_EXCEEDED: 422,
    ErrorCode.INVALID_ORDER: 422,
    ErrorCode.RISK_LIMIT_EXCEEDED: 422,
    ErrorCode.DUPLICATE_ORDER: 409,
}


# =============================================================================
# Exception Hierarchy
# =============================================================================


class AVAError(Exception):
    """
    Base exception for all AVA errors.

    All custom exceptions should inherit from this class.
    Provides structured error information for API responses.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()

        # Capture stack trace
        self.traceback = traceback.format_exc() if cause else None

    @property
    def http_status(self) -> int:
        """Get corresponding HTTP status code."""
        return ERROR_CODE_TO_HTTP.get(self.code, 500)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "error": {
                "code": self.code.value,
                "name": self.code.name,
                "message": self.message,
                "details": self.details,
                "timestamp": self.timestamp.isoformat(),
            }
        }

    def __str__(self) -> str:
        return f"{self.code.name}: {self.message}"


class ValidationError(AVAError):
    """Validation error for invalid input."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
    ):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        super().__init__(message, code, details)
        self.field = field
        self.value = value


class AuthenticationError(AVAError):
    """Authentication error."""

    def __init__(
        self,
        message: str = "Authentication required",
        code: ErrorCode = ErrorCode.AUTHENTICATION_REQUIRED,
    ):
        super().__init__(message, code)


class AuthorizationError(AVAError):
    """Authorization error."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
    ):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        super().__init__(message, ErrorCode.INSUFFICIENT_PERMISSIONS, details)


class ResourceNotFoundError(AVAError):
    """Resource not found error."""

    def __init__(
        self,
        resource_type: str,
        resource_id: Any,
        code: ErrorCode = ErrorCode.RESOURCE_NOT_FOUND,
    ):
        message = f"{resource_type} not found: {resource_id}"
        details = {"resource_type": resource_type, "resource_id": str(resource_id)}
        super().__init__(message, code, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ExternalServiceError(AVAError):
    """Error from external service."""

    def __init__(
        self,
        service: str,
        message: str,
        code: ErrorCode = ErrorCode.EXTERNAL_SERVICE_ERROR,
        cause: Optional[Exception] = None,
    ):
        details = {"service": service}
        if cause:
            details["original_error"] = str(cause)
        super().__init__(message, code, details, cause)
        self.service = service


class BrokerError(ExternalServiceError):
    """Error from broker (Robinhood, etc.)."""

    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__("robinhood", message, ErrorCode.BROKER_API_ERROR, cause)


class DatabaseError(AVAError):
    """Database error."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        details = {}
        if operation:
            details["operation"] = operation
        super().__init__(message, ErrorCode.DATABASE_ERROR, details, cause)


class RateLimitError(AVAError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, ErrorCode.RATE_LIMIT_EXCEEDED, details)
        self.retry_after = retry_after


class CircuitBreakerError(AVAError):
    """Circuit breaker is open."""

    def __init__(
        self,
        service: str,
        retry_after: Optional[int] = None,
    ):
        message = f"Service temporarily unavailable: {service}"
        details = {"service": service}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, ErrorCode.CIRCUIT_BREAKER_OPEN, details)
        self.service = service
        self.retry_after = retry_after


class BusinessLogicError(AVAError):
    """Business logic error."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.BUSINESS_LOGIC_ERROR,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, code, details)


class InsufficientFundsError(BusinessLogicError):
    """Insufficient funds for operation."""

    def __init__(
        self,
        required: float,
        available: float,
    ):
        message = f"Insufficient funds: required ${required:.2f}, available ${available:.2f}"
        details = {"required": required, "available": available}
        super().__init__(message, ErrorCode.INSUFFICIENT_FUNDS, details)


class RiskLimitError(BusinessLogicError):
    """Risk limit exceeded."""

    def __init__(
        self,
        limit_type: str,
        current: float,
        limit: float,
    ):
        message = f"Risk limit exceeded: {limit_type}"
        details = {"limit_type": limit_type, "current": current, "limit": limit}
        super().__init__(message, ErrorCode.RISK_LIMIT_EXCEEDED, details)


# =============================================================================
# Error Response Model
# =============================================================================


class ErrorDetail(BaseModel):
    """Error detail for API response."""

    code: int
    name: str
    message: str
    details: Dict[str, Any] = {}
    timestamp: str


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: ErrorDetail


# =============================================================================
# FastAPI Exception Handlers
# =============================================================================


async def ava_exception_handler(request: Request, exc: AVAError) -> JSONResponse:
    """Handle AVAError exceptions."""
    logger.error(
        "ava_error",
        error_code=exc.code.name,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
        headers=_get_error_headers(exc),
    )


async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(
        "validation_error",
        field=exc.field,
        value=exc.value,
        message=exc.message,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTPException."""
    # Convert to AVAError format
    error = AVAError(
        message=str(exc.detail),
        code=_http_status_to_error_code(exc.status_code),
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=error.to_dict(),
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    logger.exception(
        "unhandled_exception",
        exception_type=type(exc).__name__,
        message=str(exc),
        path=request.url.path,
        method=request.method,
    )

    error = AVAError(
        message="An unexpected error occurred",
        code=ErrorCode.INTERNAL_ERROR,
        cause=exc,
    )

    return JSONResponse(
        status_code=500,
        content=error.to_dict(),
    )


def _get_error_headers(exc: AVAError) -> Dict[str, str]:
    """Get additional headers for error response."""
    headers = {}

    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    if isinstance(exc, CircuitBreakerError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return headers


def _http_status_to_error_code(status: int) -> ErrorCode:
    """Map HTTP status to error code."""
    mapping = {
        400: ErrorCode.VALIDATION_ERROR,
        401: ErrorCode.AUTHENTICATION_REQUIRED,
        403: ErrorCode.INSUFFICIENT_PERMISSIONS,
        404: ErrorCode.RESOURCE_NOT_FOUND,
        409: ErrorCode.DUPLICATE_ORDER,
        422: ErrorCode.BUSINESS_LOGIC_ERROR,
        429: ErrorCode.RATE_LIMIT_EXCEEDED,
        500: ErrorCode.INTERNAL_ERROR,
        502: ErrorCode.EXTERNAL_SERVICE_ERROR,
        503: ErrorCode.DATABASE_ERROR,
    }
    return mapping.get(status, ErrorCode.INTERNAL_ERROR)


# =============================================================================
# Retry Policies
# =============================================================================


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    backoff_multiplier: float = 2.0
    retryable_exceptions: List[Type[Exception]] = field(
        default_factory=lambda: [
            ExternalServiceError,
            DatabaseError,
            ConnectionError,
            TimeoutError,
        ]
    )
    retryable_codes: List[ErrorCode] = field(
        default_factory=lambda: [
            ErrorCode.EXTERNAL_SERVICE_ERROR,
            ErrorCode.BROKER_CONNECTION_ERROR,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.CACHE_ERROR,
            ErrorCode.RATE_LIMIT_EXCEEDED,
        ]
    )

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        """Check if exception should be retried."""
        if attempt >= self.max_retries:
            return False

        # Check exception type
        for exc_type in self.retryable_exceptions:
            if isinstance(exc, exc_type):
                return True

        # Check AVAError code
        if isinstance(exc, AVAError) and exc.code in self.retryable_codes:
            return True

        return False

    def get_delay(self, attempt: int) -> float:
        """Get delay for attempt (exponential backoff)."""
        delay = self.initial_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)


# Default policies
DEFAULT_RETRY_POLICY = RetryPolicy()
AGGRESSIVE_RETRY_POLICY = RetryPolicy(
    max_retries=5,
    initial_delay=0.25,
    backoff_multiplier=1.5,
)
CONSERVATIVE_RETRY_POLICY = RetryPolicy(
    max_retries=2,
    initial_delay=1.0,
    backoff_multiplier=3.0,
)


# =============================================================================
# Decorators
# =============================================================================


def handle_errors(
    default_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
    log_level: str = "error",
) -> Callable[[F], F]:
    """
    Decorator to handle errors and convert to AVAError.

    Usage:
        @handle_errors(ErrorCode.EXTERNAL_SERVICE_ERROR)
        async def fetch_data():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except AVAError:
                raise
            except Exception as e:
                log_func = getattr(logger, log_level)
                log_func(
                    "error_handled",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                raise AVAError(
                    message=str(e),
                    code=default_code,
                    cause=e,
                )

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except AVAError:
                raise
            except Exception as e:
                log_func = getattr(logger, log_level)
                log_func(
                    "error_handled",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                raise AVAError(
                    message=str(e),
                    code=default_code,
                    cause=e,
                )

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


# =============================================================================
# Registration Helper
# =============================================================================


def register_exception_handlers(app: Any) -> None:
    """Register all exception handlers with FastAPI app."""
    from fastapi import FastAPI

    if not isinstance(app, FastAPI):
        raise TypeError("app must be a FastAPI instance")

    app.add_exception_handler(AVAError, ava_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    logger.info("exception_handlers_registered")


# =============================================================================
# Safe Error Helpers
# =============================================================================


def raise_safe_http_error(
    status_code: int,
    user_message: str,
    exception: Optional[Exception] = None,
    log_error: bool = True,
) -> None:
    """
    Raise HTTPException with a safe user-facing message while logging the actual error.

    Use this instead of `raise HTTPException(status_code=500, detail=str(e))`
    to avoid exposing internal error details to users.

    Args:
        status_code: HTTP status code
        user_message: Safe message to show to users
        exception: Original exception (logged but not exposed)
        log_error: Whether to log the exception

    Raises:
        HTTPException with user_message as detail

    Usage:
        try:
            await do_something()
        except Exception as e:
            raise_safe_http_error(500, "Failed to process request", e)
    """
    if log_error and exception:
        logger.error(
            "safe_http_error",
            status_code=status_code,
            user_message=user_message,
            error_type=type(exception).__name__,
            error_detail=str(exception),
        )
    raise HTTPException(status_code=status_code, detail=user_message)


def safe_internal_error(
    exception: Exception,
    operation: str = "process request",
) -> None:
    """
    Helper for 500 errors - logs detail but shows generic message.

    Usage:
        except Exception as e:
            safe_internal_error(e, "fetch positions")
    """
    raise_safe_http_error(
        status_code=500,
        user_message=f"Failed to {operation}. Please try again later.",
        exception=exception,
    )


def safe_service_error(
    exception: Exception,
    service_name: str,
) -> None:
    """
    Helper for external service errors.

    Usage:
        except Exception as e:
            safe_service_error(e, "Robinhood")
    """
    raise_safe_http_error(
        status_code=502,
        user_message=f"{service_name} service is temporarily unavailable.",
        exception=exception,
    )


# =============================================================================
# AI-Powered Error Classification
# =============================================================================


class ErrorClassifier:
    """
    AI-powered error classification and correlation system.

    Features:
    - Automatic error categorization using pattern matching
    - Error correlation to identify related issues
    - Smart suggestions for resolution
    - Error trend analysis
    """

    # Error pattern definitions for classification
    ERROR_PATTERNS = {
        ErrorCode.DATABASE_ERROR: [
            "connection", "postgres", "asyncpg", "pool", "timeout", "deadlock",
            "constraint", "duplicate key", "foreign key", "transaction"
        ],
        ErrorCode.BROKER_API_ERROR: [
            "robinhood", "broker", "authentication", "token", "session",
            "trading", "order", "position", "401", "403"
        ],
        ErrorCode.MARKET_DATA_ERROR: [
            "yfinance", "yahoo", "quote", "ticker", "market data", "price",
            "historical", "options chain", "expiration"
        ],
        ErrorCode.AI_SERVICE_ERROR: [
            "llm", "ollama", "openai", "anthropic", "model", "inference",
            "embedding", "rag", "vector", "chromadb"
        ],
        ErrorCode.RATE_LIMIT_EXCEEDED: [
            "rate limit", "too many requests", "429", "throttle", "quota"
        ],
        ErrorCode.SPORTS_API_ERROR: [
            "espn", "kalshi", "odds", "game", "sport", "betting"
        ],
        ErrorCode.CACHE_ERROR: [
            "redis", "cache", "memcached", "ttl", "expire"
        ],
        ErrorCode.VALIDATION_ERROR: [
            "validation", "invalid", "required", "format", "type error",
            "pydantic", "schema"
        ],
    }

    # Suggested actions for each error code
    RESOLUTION_HINTS = {
        ErrorCode.DATABASE_ERROR: [
            "Check database connection pool status",
            "Verify PostgreSQL is running",
            "Review connection string in .env",
            "Check for long-running queries or deadlocks"
        ],
        ErrorCode.BROKER_API_ERROR: [
            "Re-authenticate with Robinhood",
            "Check if session token is expired",
            "Verify trading hours",
            "Review rate limits"
        ],
        ErrorCode.MARKET_DATA_ERROR: [
            "Verify symbol is valid and tradeable",
            "Check if market is open",
            "Try refreshing data cache",
            "Verify yfinance installation"
        ],
        ErrorCode.AI_SERVICE_ERROR: [
            "Check if Ollama is running",
            "Verify model is downloaded",
            "Review API key configuration",
            "Check GPU/memory availability"
        ],
        ErrorCode.RATE_LIMIT_EXCEEDED: [
            "Wait before retrying",
            "Implement request caching",
            "Reduce request frequency",
            "Check API quota usage"
        ],
        ErrorCode.SPORTS_API_ERROR: [
            "Verify ESPN API is accessible",
            "Check Kalshi connection",
            "Verify game IDs are valid",
            "Review sports data cache"
        ],
        ErrorCode.CACHE_ERROR: [
            "Check Redis connection",
            "Verify cache key format",
            "Review TTL settings",
            "Fall back to in-memory cache"
        ],
    }

    def __init__(self) -> None:
        self._error_history: List[Dict[str, Any]] = []
        self._correlation_window = 300  # 5 minutes

    def classify(self, exception: Exception) -> ErrorCode:
        """
        Classify an exception into an appropriate error code using AI pattern matching.

        Args:
            exception: The exception to classify

        Returns:
            ErrorCode that best matches the exception
        """
        error_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Check each pattern category
        scores: Dict[ErrorCode, int] = {}

        for code, patterns in self.ERROR_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if pattern in error_str or pattern in exception_type:
                    score += 1
            if score > 0:
                scores[code] = score

        # Return highest scoring code, or INTERNAL_ERROR if no match
        if scores:
            return max(scores, key=scores.get)
        return ErrorCode.INTERNAL_ERROR

    def get_resolution_hints(self, code: ErrorCode) -> List[str]:
        """Get suggested resolution steps for an error code."""
        return self.RESOLUTION_HINTS.get(code, [
            "Check application logs for more details",
            "Verify all services are running",
            "Contact support if issue persists"
        ])

    def record_error(
        self,
        exception: Exception,
        code: ErrorCode,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record an error for correlation analysis.

        Returns:
            error_id for tracking
        """
        import uuid
        from datetime import datetime

        error_id = str(uuid.uuid4())[:8]

        self._error_history.append({
            "error_id": error_id,
            "timestamp": datetime.now(),
            "code": code,
            "exception_type": type(exception).__name__,
            "message": str(exception)[:500],
            "context": context or {},
        })

        # Keep only recent errors (memory management)
        cutoff = datetime.now() - timedelta(seconds=self._correlation_window * 2)
        self._error_history = [
            e for e in self._error_history
            if e["timestamp"] > cutoff
        ]

        return error_id

    def find_correlated_errors(self, error_id: str) -> List[Dict[str, Any]]:
        """Find errors that may be related to a given error."""
        from datetime import datetime

        # Find the target error
        target = next(
            (e for e in self._error_history if e["error_id"] == error_id),
            None
        )
        if not target:
            return []

        # Find errors within correlation window with same code
        cutoff_start = target["timestamp"] - timedelta(seconds=self._correlation_window)
        cutoff_end = target["timestamp"] + timedelta(seconds=self._correlation_window)

        correlated = [
            e for e in self._error_history
            if e["error_id"] != error_id
            and cutoff_start <= e["timestamp"] <= cutoff_end
            and e["code"] == target["code"]
        ]

        return correlated

    def get_error_trends(self) -> Dict[str, Any]:
        """Analyze recent error trends."""
        from collections import Counter
        from datetime import datetime

        if not self._error_history:
            return {"total_errors": 0, "trends": {}}

        # Count by code
        code_counts = Counter(e["code"].name for e in self._error_history)

        # Count by exception type
        type_counts = Counter(e["exception_type"] for e in self._error_history)

        # Calculate error rate (per minute)
        if len(self._error_history) >= 2:
            time_range = (
                self._error_history[-1]["timestamp"] -
                self._error_history[0]["timestamp"]
            ).total_seconds() / 60
            error_rate = len(self._error_history) / max(time_range, 1)
        else:
            error_rate = 0

        return {
            "total_errors": len(self._error_history),
            "error_rate_per_minute": round(error_rate, 2),
            "by_code": dict(code_counts.most_common(5)),
            "by_type": dict(type_counts.most_common(5)),
            "window_seconds": self._correlation_window,
        }


# Global error classifier instance
_error_classifier: Optional[ErrorClassifier] = None


def get_error_classifier() -> ErrorClassifier:
    """Get the global error classifier instance."""
    global _error_classifier
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    return _error_classifier


def smart_error_response(
    exception: Exception,
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    include_hints: bool = False,
) -> None:
    """
    AI-enhanced error response with classification and correlation.

    Features:
    - Automatic error code classification
    - Error correlation tracking
    - Optional resolution hints in dev mode

    Usage:
        except Exception as e:
            smart_error_response(e, "fetch portfolio", {"user_id": user_id})
    """
    classifier = get_error_classifier()

    # Classify the error
    code = classifier.classify(exception)

    # Record for correlation
    error_id = classifier.record_error(exception, code, context)

    # Log with full context
    logger.error(
        "smart_error_classified",
        error_id=error_id,
        operation=operation,
        error_code=code.name,
        exception_type=type(exception).__name__,
        message=str(exception),
        context=context,
        correlated_count=len(classifier.find_correlated_errors(error_id)),
    )

    # Build user message
    user_message = f"Failed to {operation}. Please try again later."

    # Add hints in development mode
    if include_hints:
        hints = classifier.get_resolution_hints(code)
        if hints:
            user_message += f" (Error ID: {error_id})"

    # Get HTTP status from error code
    http_status = ERROR_CODE_TO_HTTP.get(code, 500)

    raise HTTPException(
        status_code=http_status,
        detail=user_message,
        headers={"X-Error-ID": error_id, "X-Error-Code": code.name}
    )
