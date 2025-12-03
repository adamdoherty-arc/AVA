"""
Circuit Breaker Pattern Implementation

Provides resilience for external API calls:
- Automatic failure detection
- Graceful degradation
- Self-healing after timeout
- Fallback support
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Track circuit breaker metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None


class CircuitBreakerError(Exception):
    """Raised when circuit is open"""
    pass


class CircuitBreaker(Generic[T]):
    """
    Circuit breaker for external API resilience.

    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Too many failures, reject all calls immediately
    - HALF_OPEN: Testing recovery, allow limited calls

    Usage:
        breaker = CircuitBreaker(
            name="robinhood",
            failure_threshold=5,
            recovery_timeout=60,
            fallback=lambda: cached_positions
        )

        result = await breaker.call(get_positions)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
        fallback: Optional[Callable[[], T]] = None
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.fallback = fallback

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
        self.stats = CircuitBreakerStats()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, auto-transitioning if needed"""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self.stats.state_changes += 1
                    logger.info(f"Circuit {self.name}: OPEN -> HALF_OPEN (recovery timeout)")

        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.

        Raises:
            CircuitBreakerError: If circuit is open and no fallback
        """
        async with self._lock:
            self.stats.total_calls += 1

            current_state = self.state

            # OPEN: Reject or fallback
            if current_state == CircuitState.OPEN:
                self.stats.rejected_calls += 1
                logger.warning(f"Circuit {self.name} is OPEN, rejecting call")

                if self.fallback:
                    return self.fallback()
                raise CircuitBreakerError(f"Circuit {self.name} is open")

            # HALF_OPEN: Check if we can make test call
            if current_state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self.stats.rejected_calls += 1
                    if self.fallback:
                        return self.fallback()
                    raise CircuitBreakerError(f"Circuit {self.name} half-open limit reached")
                self._half_open_calls += 1

        # Execute the call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            self.stats.successful_calls += 1
            self.stats.last_success_time = time.time()
            self._success_count += 1

            if self._state == CircuitState.HALF_OPEN:
                # Successful test call, close circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self.stats.state_changes += 1
                logger.info(f"Circuit {self.name}: HALF_OPEN -> CLOSED (recovery successful)")

            # Reset failure count on success in closed state
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, error: Exception):
        """Handle failed call"""
        async with self._lock:
            self.stats.failed_calls += 1
            self.stats.last_failure_time = time.time()
            self._failure_count += 1
            self._last_failure_time = time.time()

            logger.warning(f"Circuit {self.name} failure #{self._failure_count}: {error}")

            if self._state == CircuitState.HALF_OPEN:
                # Failed test call, reopen circuit
                self._state = CircuitState.OPEN
                self.stats.state_changes += 1
                logger.warning(f"Circuit {self.name}: HALF_OPEN -> OPEN (test call failed)")

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self.stats.state_changes += 1
                    logger.warning(f"Circuit {self.name}: CLOSED -> OPEN (threshold reached)")

    def get_stats(self) -> dict:
        """Get circuit breaker statistics"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            **{
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "rejected_calls": self.stats.rejected_calls,
                "state_changes": self.stats.state_changes
            }
        }

    def reset(self):
        """Manually reset circuit to closed state"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info(f"Circuit {self.name} manually reset to CLOSED")


# =============================================================================
# Circuit Breaker Decorator
# =============================================================================

def circuit_protected(
    breaker: CircuitBreaker,
    fallback: Optional[Callable[..., Any]] = None
):
    """
    Decorator to protect a function with circuit breaker.

    Usage:
        rh_breaker = CircuitBreaker("robinhood", failure_threshold=5)

        @circuit_protected(rh_breaker)
        async def get_positions():
            return rh.get_open_stock_positions()
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await breaker.call(func, *args, **kwargs)
            except CircuitBreakerError:
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise
        return wrapper
    return decorator


# =============================================================================
# Pre-configured Circuit Breakers
# =============================================================================

# Robinhood API circuit breaker
robinhood_breaker = CircuitBreaker(
    name="robinhood",
    failure_threshold=5,
    recovery_timeout=60,
    half_open_max_calls=2
)

# YFinance API circuit breaker
yfinance_breaker = CircuitBreaker(
    name="yfinance",
    failure_threshold=10,
    recovery_timeout=120,
    half_open_max_calls=3
)

# LLM API circuit breaker
llm_breaker = CircuitBreaker(
    name="llm",
    failure_threshold=3,
    recovery_timeout=30,
    half_open_max_calls=1
)


def get_all_breaker_stats() -> dict:
    """Get stats for all circuit breakers"""
    return {
        "robinhood": robinhood_breaker.get_stats(),
        "yfinance": yfinance_breaker.get_stats(),
        "llm": llm_breaker.get_stats()
    }
