"""
AVA Rate Limiter
================

Thread-safe rate limiting for Telegram bot to prevent abuse and ensure fair usage.

Features:
- Per-user rate limits (10 requests/minute)
- Global rate limits (100 requests/minute)
- Daily limits (1000 requests/day per user)
- Automatic cleanup of expired entries
- User-friendly error messages
- Thread-safe operations

Usage:
    from src.ava.rate_limiter import get_rate_limiter, RateLimitExceeded

    limiter = get_rate_limiter()

    try:
        limiter.check_rate_limit(user_id=12345)
        # Process request
    except RateLimitExceeded as e:
        await update.message.reply_text(str(e))
"""

import time
import threading
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded"""
    pass


@dataclass
class UserLimitState:
    """Tracks rate limit state for a single user"""
    minute_requests: list = field(default_factory=list)  # Timestamps of requests in current minute
    daily_requests: int = 0  # Count of requests today
    daily_reset_time: datetime = field(default_factory=datetime.now)  # When daily counter resets


class RateLimiter:
    """
    Thread-safe rate limiter for Telegram bot.

    Implements multiple rate limiting strategies:
    - Per-user per-minute limit
    - Per-user daily limit
    - Global per-minute limit
    """

    # Configuration
    PER_USER_MINUTE_LIMIT = 10  # requests per minute per user
    PER_USER_DAILY_LIMIT = 1000  # requests per day per user
    GLOBAL_MINUTE_LIMIT = 100  # total requests per minute across all users

    # Cleanup configuration
    CLEANUP_INTERVAL = 300  # seconds (5 minutes)

    def __init__(self) -> None:
        """Initialize rate limiter"""
        self._user_states: Dict[int, UserLimitState] = {}
        self._global_requests: list = []  # Timestamps of all requests in current minute
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

        logger.info(
            f"Rate limiter initialized: "
            f"{self.PER_USER_MINUTE_LIMIT} req/min/user, "
            f"{self.PER_USER_DAILY_LIMIT} req/day/user, "
            f"{self.GLOBAL_MINUTE_LIMIT} req/min global"
        )

    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries to prevent memory leaks"""
        current_time = time.time()

        # Only cleanup every CLEANUP_INTERVAL seconds
        if current_time - self._last_cleanup < self.CLEANUP_INTERVAL:
            return

        with self._lock:
            # Cleanup global requests older than 1 minute
            one_minute_ago = current_time - 60
            self._global_requests = [
                ts for ts in self._global_requests if ts > one_minute_ago
            ]

            # Cleanup user states
            users_to_remove = []
            for user_id, state in self._user_states.items():
                # Remove minute requests older than 1 minute
                state.minute_requests = [
                    ts for ts in state.minute_requests if ts > one_minute_ago
                ]

                # Reset daily counter if needed
                if datetime.now() > state.daily_reset_time:
                    state.daily_requests = 0
                    state.daily_reset_time = datetime.now() + timedelta(days=1)

                # Remove user state if no recent activity
                if not state.minute_requests and state.daily_requests == 0:
                    users_to_remove.append(user_id)

            # Remove inactive users
            for user_id in users_to_remove:
                del self._user_states[user_id]

            self._last_cleanup = current_time

            if users_to_remove:
                logger.debug(f"Cleaned up {len(users_to_remove)} inactive user states")

    def _get_user_state(self, user_id: int) -> UserLimitState:
        """Get or create user state"""
        if user_id not in self._user_states:
            self._user_states[user_id] = UserLimitState(
                daily_reset_time=datetime.now() + timedelta(days=1)
            )
        return self._user_states[user_id]

    def _check_per_user_minute_limit(self, user_id: int, current_time: float) -> Tuple[bool, Optional[str]]:
        """
        Check per-user per-minute rate limit.

        Returns:
            (is_allowed, error_message)
        """
        state = self._get_user_state(user_id)

        # Remove requests older than 1 minute
        one_minute_ago = current_time - 60
        state.minute_requests = [ts for ts in state.minute_requests if ts > one_minute_ago]

        # Check limit
        if len(state.minute_requests) >= self.PER_USER_MINUTE_LIMIT:
            oldest_request = min(state.minute_requests)
            wait_seconds = int(60 - (current_time - oldest_request))
            return False, (
                f"⏱️ Rate limit exceeded. Please wait {wait_seconds} seconds before trying again.\n"
                f"Limit: {self.PER_USER_MINUTE_LIMIT} requests per minute."
            )

        return True, None

    def _check_per_user_daily_limit(self, user_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check per-user daily rate limit.

        Returns:
            (is_allowed, error_message)
        """
        state = self._get_user_state(user_id)

        # Reset daily counter if needed
        if datetime.now() > state.daily_reset_time:
            state.daily_requests = 0
            state.daily_reset_time = datetime.now() + timedelta(days=1)

        # Check limit
        if state.daily_requests >= self.PER_USER_DAILY_LIMIT:
            hours_until_reset = int((state.daily_reset_time - datetime.now()).total_seconds() / 3600)
            return False, (
                f"📊 Daily limit reached. Please try again in {hours_until_reset} hours.\n"
                f"Limit: {self.PER_USER_DAILY_LIMIT} requests per day."
            )

        return True, None

    def _check_global_minute_limit(self, current_time: float) -> Tuple[bool, Optional[str]]:
        """
        Check global per-minute rate limit.

        Returns:
            (is_allowed, error_message)
        """
        # Remove requests older than 1 minute
        one_minute_ago = current_time - 60
        self._global_requests = [ts for ts in self._global_requests if ts > one_minute_ago]

        # Check limit
        if len(self._global_requests) >= self.GLOBAL_MINUTE_LIMIT:
            oldest_request = min(self._global_requests)
            wait_seconds = int(60 - (current_time - oldest_request))
            return False, (
                f"🚦 System is currently busy. Please try again in {wait_seconds} seconds.\n"
                f"Thank you for your patience!"
            )

        return True, None

    def check_rate_limit(self, user_id: int) -> None:
        """
        Check if user is within rate limits.

        Args:
            user_id: Telegram user ID

        Raises:
            RateLimitExceeded: If any rate limit is exceeded
        """
        current_time = time.time()

        # Periodic cleanup
        self._cleanup_expired_entries()

        with self._lock:
            # Check per-user minute limit
            allowed, error_msg = self._check_per_user_minute_limit(user_id, current_time)
            if not allowed:
                logger.warning(f"Per-user minute limit exceeded for user {user_id}")
                raise RateLimitExceeded(error_msg)

            # Check per-user daily limit
            allowed, error_msg = self._check_per_user_daily_limit(user_id)
            if not allowed:
                logger.warning(f"Per-user daily limit exceeded for user {user_id}")
                raise RateLimitExceeded(error_msg)

            # Check global minute limit
            allowed, error_msg = self._check_global_minute_limit(current_time)
            if not allowed:
                logger.warning(f"Global minute limit exceeded (triggered by user {user_id})")
                raise RateLimitExceeded(error_msg)

            # All checks passed - record the request
            state = self._get_user_state(user_id)
            state.minute_requests.append(current_time)
            state.daily_requests += 1
            self._global_requests.append(current_time)

            logger.debug(
                f"Request allowed for user {user_id}: "
                f"{len(state.minute_requests)}/{self.PER_USER_MINUTE_LIMIT} per-minute, "
                f"{state.daily_requests}/{self.PER_USER_DAILY_LIMIT} daily, "
                f"{len(self._global_requests)}/{self.GLOBAL_MINUTE_LIMIT} global"
            )

    def get_user_stats(self, user_id: int) -> Dict[str, any]:
        """
        Get rate limit statistics for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            if user_id not in self._user_states:
                return {
                    "minute_requests": 0,
                    "minute_limit": self.PER_USER_MINUTE_LIMIT,
                    "daily_requests": 0,
                    "daily_limit": self.PER_USER_DAILY_LIMIT,
                    "daily_reset_time": (datetime.now() + timedelta(days=1)).isoformat()
                }

            state = self._user_states[user_id]

            # Clean up old minute requests
            current_time = time.time()
            one_minute_ago = current_time - 60
            state.minute_requests = [ts for ts in state.minute_requests if ts > one_minute_ago]

            # Reset daily counter if needed
            if datetime.now() > state.daily_reset_time:
                state.daily_requests = 0
                state.daily_reset_time = datetime.now() + timedelta(days=1)

            return {
                "minute_requests": len(state.minute_requests),
                "minute_limit": self.PER_USER_MINUTE_LIMIT,
                "daily_requests": state.daily_requests,
                "daily_limit": self.PER_USER_DAILY_LIMIT,
                "daily_reset_time": state.daily_reset_time.isoformat()
            }

    def get_global_stats(self) -> Dict[str, any]:
        """
        Get global rate limit statistics.

        Returns:
            Dictionary with global statistics
        """
        with self._lock:
            # Clean up old requests
            current_time = time.time()
            one_minute_ago = current_time - 60
            self._global_requests = [ts for ts in self._global_requests if ts > one_minute_ago]

            return {
                "global_minute_requests": len(self._global_requests),
                "global_minute_limit": self.GLOBAL_MINUTE_LIMIT,
                "total_active_users": len(self._user_states)
            }

    def reset_user_limits(self, user_id: int):
        """
        Reset all limits for a specific user (admin function).

        Args:
            user_id: Telegram user ID
        """
        with self._lock:
            if user_id in self._user_states:
                del self._user_states[user_id]
                logger.info(f"Reset rate limits for user {user_id}")


# Global singleton instance
_rate_limiter_instance: Optional[RateLimiter] = None
_init_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """
    Get the global rate limiter instance.

    Thread-safe singleton accessor.

    Returns:
        RateLimiter instance
    """
    global _rate_limiter_instance

    if _rate_limiter_instance is None:
        with _init_lock:
            if _rate_limiter_instance is None:
                _rate_limiter_instance = RateLimiter()

    return _rate_limiter_instance


# Example usage
if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Testing AVA Rate Limiter...\n")

    limiter = get_rate_limiter()
    test_user_id = 12345

    try:
        # Test 1: Normal requests
        print("Test 1: Normal requests (should succeed)")
        for i in range(5):
            limiter.check_rate_limit(test_user_id)
            print(f"Request {i+1} allowed")

        # Test 2: Check stats
        print("\nTest 2: User statistics")
        stats = limiter.get_user_stats(test_user_id)
        print(f"User stats: {stats}")

        # Test 3: Check global stats
        print("\nTest 3: Global statistics")
        global_stats = limiter.get_global_stats()
        print(f"Global stats: {global_stats}")

        # Test 4: Rapid requests (should hit per-minute limit)
        print("\nTest 4: Rapid requests (testing rate limit)")
        try:
            for i in range(15):  # Exceeds PER_USER_MINUTE_LIMIT (10)
                limiter.check_rate_limit(test_user_id)
                print(f"Request {i+1} allowed")
        except RateLimitExceeded as e:
            print(f"✅ Rate limit triggered as expected: {e}")

        # Test 5: Reset user limits
        print("\nTest 5: Reset user limits")
        limiter.reset_user_limits(test_user_id)
        stats = limiter.get_user_stats(test_user_id)
        print(f"Stats after reset: {stats}")

        print("\n✅ All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
