"""
AVA Error Handling
==================

Comprehensive error handling with:
- Custom exception hierarchy
- Error codes and messages
- Retry policies
- Error recovery strategies
- Logging and monitoring

Author: AVA Trading Platform
Created: 2025-11-28
"""

import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, Callable, TypeVar, List
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# ERROR CODES
# =============================================================================

class ErrorCode(Enum):
    """Error codes for categorization"""
    # General errors (1xxx)
    UNKNOWN = 1000
    VALIDATION_ERROR = 1001
    CONFIGURATION_ERROR = 1002
    TIMEOUT_ERROR = 1003

    # API errors (2xxx)
    API_ERROR = 2000
    API_RATE_LIMIT = 2001
    API_AUTHENTICATION = 2002
    API_NOT_FOUND = 2003
    API_SERVER_ERROR = 2004

    # Trading errors (3xxx)
    TRADING_ERROR = 3000
    INSUFFICIENT_FUNDS = 3001
    INVALID_ORDER = 3002
    ORDER_REJECTED = 3003
    POSITION_NOT_FOUND = 3004
    MARKET_CLOSED = 3005

    # Data errors (4xxx)
    DATA_ERROR = 4000
    DATA_NOT_FOUND = 4001
    DATA_STALE = 4002
    DATA_INVALID = 4003

    # Strategy errors (5xxx)
    STRATEGY_ERROR = 5000
    NO_OPPORTUNITIES = 5001
    INVALID_SETUP = 5002
    RISK_LIMIT_EXCEEDED = 5003

    # System errors (6xxx)
    SYSTEM_ERROR = 6000
    DATABASE_ERROR = 6001
    CACHE_ERROR = 6002
    NETWORK_ERROR = 6003


# =============================================================================
# BASE EXCEPTIONS
# =============================================================================

class AVAError(Exception):
    """Base exception for all AVA errors"""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code.value,
            "code_name": self.code.name,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }

    def __str__(self) -> str:
        return f"[{self.code.name}] {self.message}"


# =============================================================================
# SPECIFIC EXCEPTIONS
# =============================================================================

class ValidationError(AVAError):
    """Data validation failed"""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code=ErrorCode.VALIDATION_ERROR,
            details={"field": field, **kwargs}
        )
        self.field = field


class ConfigurationError(AVAError):
    """Configuration is invalid or missing"""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            code=ErrorCode.CONFIGURATION_ERROR,
            details={"config_key": config_key, **kwargs}
        )


class APIError(AVAError):
    """External API error"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        code = ErrorCode.API_ERROR
        if status_code == 429:
            code = ErrorCode.API_RATE_LIMIT
        elif status_code == 401:
            code = ErrorCode.API_AUTHENTICATION
        elif status_code == 404:
            code = ErrorCode.API_NOT_FOUND
        elif status_code and status_code >= 500:
            code = ErrorCode.API_SERVER_ERROR

        super().__init__(
            message,
            code=code,
            details={
                "status_code": status_code,
                "response_body": response_body,
                **kwargs
            }
        )
        self.status_code = status_code


class RateLimitError(APIError):
    """API rate limit exceeded"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, status_code=429, **kwargs)
        self.code = ErrorCode.API_RATE_LIMIT
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class TradingError(AVAError):
    """Trading operation error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=ErrorCode.TRADING_ERROR, **kwargs)


class InsufficientFundsError(TradingError):
    """Insufficient funds for trade"""

    def __init__(
        self,
        message: str = "Insufficient funds",
        required: Optional[float] = None,
        available: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.INSUFFICIENT_FUNDS
        self.details.update({
            "required": required,
            "available": available
        })


class OrderRejectedError(TradingError):
    """Order was rejected"""

    def __init__(
        self,
        message: str,
        order_id: Optional[str] = None,
        reason: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.ORDER_REJECTED
        self.details.update({
            "order_id": order_id,
            "reason": reason
        })


class MarketClosedError(TradingError):
    """Market is closed"""

    def __init__(self, message: str = "Market is closed", **kwargs):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.MARKET_CLOSED


class DataError(AVAError):
    """Data-related error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, code=ErrorCode.DATA_ERROR, **kwargs)


class DataNotFoundError(DataError):
    """Requested data not found"""

    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        identifier: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.DATA_NOT_FOUND
        self.details.update({
            "resource": resource,
            "identifier": identifier
        })


class StaleDataError(DataError):
    """Data is too old"""

    def __init__(
        self,
        message: str,
        data_age_seconds: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.DATA_STALE
        self.details.update({
            "data_age_seconds": data_age_seconds,
            "max_age_seconds": max_age_seconds
        })


class StrategyError(AVAError):
    """Strategy-related error"""

    def __init__(self, message: str, strategy_name: Optional[str] = None, **kwargs):
        super().__init__(message, code=ErrorCode.STRATEGY_ERROR, **kwargs)
        self.details["strategy_name"] = strategy_name


class NoOpportunitiesError(StrategyError):
    """No valid opportunities found"""

    def __init__(
        self,
        message: str = "No valid opportunities found",
        filters_applied: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.NO_OPPORTUNITIES
        self.details["filters_applied"] = filters_applied or []


class RiskLimitExceededError(StrategyError):
    """Risk limit exceeded"""

    def __init__(
        self,
        message: str,
        limit_name: str,
        current_value: float,
        limit_value: float,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.code = ErrorCode.RISK_LIMIT_EXCEEDED
        self.details.update({
            "limit_name": limit_name,
            "current_value": current_value,
            "limit_value": limit_value
        })


class DatabaseError(AVAError):
    """Database operation error"""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(message, code=ErrorCode.DATABASE_ERROR, **kwargs)
        self.details["query"] = query


# =============================================================================
# ERROR HANDLING UTILITIES
# =============================================================================

@dataclass
class ErrorContext:
    """Context information for error handling"""
    operation: str
    component: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorHandler:
    """
    Centralized error handler with logging and recovery.

    Usage:
        handler = ErrorHandler()

        @handler.handle(fallback=default_value)
        async def risky_operation():
            ...
    """

    def __init__(
        self,
        notify_on_critical: bool = True,
        log_level: int = logging.ERROR
    ):
        self.notify_on_critical = notify_on_critical
        self.log_level = log_level
        self._error_counts: Dict[ErrorCode, int] = {}
        self._callbacks: List[Callable[[AVAError], None]] = []

    def register_callback(self, callback: Callable[[AVAError], None]):
        """Register error callback for notifications"""
        self._callbacks.append(callback)

    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None
    ) -> AVAError:
        """Handle an error with logging and callbacks"""
        # Wrap in AVAError if needed
        if not isinstance(error, AVAError):
            ava_error = AVAError(
                message=str(error),
                cause=error
            )
        else:
            ava_error = error

        # Update error counts
        self._error_counts[ava_error.code] = self._error_counts.get(ava_error.code, 0) + 1

        # Build log message
        log_data = ava_error.to_dict()
        if context:
            log_data["context"] = {
                "operation": context.operation,
                "component": context.component,
                "user_id": context.user_id,
                "request_id": context.request_id,
                "metadata": context.metadata
            }

        # Log error
        logger.log(self.log_level, f"Error: {ava_error}", extra=log_data)

        # Call callbacks
        for callback in self._callbacks:
            try:
                callback(ava_error)
            except Exception as e:
                logger.warning(f"Error callback failed: {e}")

        return ava_error

    def handle(
        self,
        fallback: Optional[T] = None,
        reraise: bool = False,
        context: Optional[ErrorContext] = None
    ):
        """
        Decorator for error handling.

        Usage:
            @handler.handle(fallback=[], reraise=False)
            async def fetch_data():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, context)
                    if reraise:
                        raise
                    return fallback

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.handle_error(e, context)
                    if reraise:
                        raise
                    return fallback

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return {
            code.name: count
            for code, count in self._error_counts.items()
        }


# =============================================================================
# RETRY POLICIES
# =============================================================================

@dataclass
class RetryPolicy:
    """Retry policy configuration"""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    retryable_errors: tuple = (
        APIError,
        DatabaseError,
        TimeoutError,
        ConnectionError
    )
    non_retryable_errors: tuple = (
        ValidationError,
        InsufficientFundsError,
        OrderRejectedError
    )


def should_retry(error: Exception, policy: RetryPolicy) -> bool:
    """Determine if error should be retried"""
    # Never retry these
    if isinstance(error, policy.non_retryable_errors):
        return False

    # Always retry these
    if isinstance(error, policy.retryable_errors):
        return True

    # Rate limit - retry after delay
    if isinstance(error, RateLimitError):
        return True

    return False


async def with_retry(
    func: Callable[[], T],
    policy: Optional[RetryPolicy] = None
) -> T:
    """
    Execute function with retry policy.

    Usage:
        result = await with_retry(
            lambda: api.fetch(symbol),
            policy=RetryPolicy(max_attempts=3)
        )
    """
    policy = policy or RetryPolicy()
    last_error = None

    for attempt in range(policy.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            return func()

        except Exception as e:
            last_error = e

            if not should_retry(e, policy):
                raise

            if attempt == policy.max_attempts - 1:
                raise

            # Calculate delay
            delay = min(
                policy.initial_delay * (policy.exponential_base ** attempt),
                policy.max_delay
            )

            # Handle rate limit retry-after
            if isinstance(e, RateLimitError) and e.retry_after:
                delay = max(delay, e.retry_after)

            logger.warning(
                f"Retry {attempt + 1}/{policy.max_attempts} "
                f"after {delay:.1f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise last_error or Exception("Retry failed")


# =============================================================================
# ERROR RECOVERY
# =============================================================================

class RecoveryStrategy:
    """Base class for error recovery strategies"""

    async def recover(self, error: AVAError, context: Dict) -> bool:
        """
        Attempt recovery from error.

        Returns True if recovery successful, False otherwise.
        """
        raise NotImplementedError


class RetryRecovery(RecoveryStrategy):
    """Recovery by retrying operation"""

    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()

    async def recover(self, error: AVAError, context: Dict) -> bool:
        if should_retry(error, self.policy):
            return True
        return False


class FallbackRecovery(RecoveryStrategy):
    """Recovery by using fallback value"""

    def __init__(self, fallback_func: Callable):
        self.fallback_func = fallback_func

    async def recover(self, error: AVAError, context: Dict) -> bool:
        try:
            context["result"] = await self.fallback_func()
            return True
        except Exception:
            return False


class CircuitBreakerRecovery(RecoveryStrategy):
    """Recovery by opening circuit breaker"""

    def __init__(self, breaker_name: str):
        self.breaker_name = breaker_name

    async def recover(self, error: AVAError, context: Dict) -> bool:
        # Would integrate with circuit breaker
        logger.warning(f"Opening circuit breaker: {self.breaker_name}")
        return False


# =============================================================================
# GLOBAL ERROR HANDLER
# =============================================================================

_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Error Handling ===\n")

    async def test_errors():
        handler = get_error_handler()

        # Test validation error
        print("1. Testing ValidationError...")
        try:
            raise ValidationError(
                "Invalid strike price",
                field="strike",
                value=-100
            )
        except ValidationError as e:
            print(f"   Caught: {e}")
            print(f"   Dict: {e.to_dict()}")

        # Test API error
        print("\n2. Testing APIError...")
        try:
            raise APIError(
                "Failed to fetch option chain",
                status_code=429,
                response_body="Rate limit exceeded"
            )
        except APIError as e:
            print(f"   Caught: {e}")
            print(f"   Status: {e.status_code}")

        # Test with retry
        print("\n3. Testing retry...")
        call_count = 0

        async def flaky_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise APIError("Temporary failure", status_code=500)
            return "success"

        try:
            result = await with_retry(flaky_call, RetryPolicy(max_attempts=3))
            print(f"   Result after {call_count} attempts: {result}")
        except Exception as e:
            print(f"   Failed: {e}")

        # Test decorated handler
        print("\n4. Testing decorated handler...")

        @handler.handle(fallback="default", reraise=False)
        async def risky_operation():
            raise DataNotFoundError("Symbol not found", resource="options", identifier="INVALID")

        result = await risky_operation()
        print(f"   Result with fallback: {result}")

        print("\n5. Error statistics:")
        stats = handler.get_error_stats()
        for code, count in stats.items():
            print(f"   {code}: {count}")

        print("\n✅ Error handling tests passed!")

    asyncio.run(test_errors())
