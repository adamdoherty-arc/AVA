"""
ESPN API Rate Limiter
Prevents IP bans by limiting requests to ESPN's unofficial API
"""

import time
import logging
from functools import wraps
from typing import Callable, Any

logger = logging.getLogger(__name__)


class ESPNRateLimiter:
    """
    Rate limiter for ESPN API calls

    Prevents excessive requests that could lead to IP bans
    Default: 60 calls per minute (conservative estimate)
    """

    def __init__(self, max_calls_per_minute: int = 60):
        """
        Initialize rate limiter

        Args:
            max_calls_per_minute: Maximum API calls allowed per minute
        """
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.window_seconds = 60

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator to apply rate limiting to a function

        Usage:
            rate_limiter = ESPNRateLimiter(max_calls_per_minute=60)

            @rate_limiter
            def fetch_scoreboard():
                return espn_client.get_scoreboard()
        """
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            now = time.time()

            # Remove calls older than the time window
            self.calls = [c for c in self.calls if now - c < self.window_seconds]

            # Check if we're at the limit
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = self.calls[0]
                sleep_time = self.window_seconds - (now - oldest_call)

                if sleep_time > 0:
                    logger.warning(
                        f"Rate limit reached ({self.max_calls} calls/{self.window_seconds}s). "
                        f"Sleeping for {sleep_time:.2f} seconds"
                    )
                    time.sleep(sleep_time)

                    # Clear old calls after sleeping
                    now = time.time()
                    self.calls = [c for c in self.calls if now - c < self.window_seconds]

            # Record this call
            self.calls.append(time.time())

            # Execute the function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in rate-limited function {func.__name__}: {e}")
                raise

        return wrapper

    def get_current_rate(self) -> int:
        """
        Get current number of calls in the time window

        Returns:
            Number of API calls made in the last minute
        """
        now = time.time()
        self.calls = [c for c in self.calls if now - c < self.window_seconds]
        return len(self.calls)

    def reset(self) -> None:
        """Reset the rate limiter (clear all tracked calls)"""
        self.calls = []
        logger.info("Rate limiter reset")


# Global rate limiter instance (shared across application)
espn_rate_limiter = ESPNRateLimiter(max_calls_per_minute=60)


# Convenience decorator using the global instance
def rate_limited(func: Callable) -> Callable:
    """
    Convenience decorator using global rate limiter

    Usage:
        from src.espn_rate_limiter import rate_limited

        @rate_limited
        def fetch_data():
            return espn_client.get_scoreboard()
    """
    return espn_rate_limiter(func)
