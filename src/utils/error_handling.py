"""
Enhanced Error Handling Utilities
Provides graceful error handling with user-friendly messages

Features:
- Consistent error logging
- User-friendly error messages
- Automatic retry logic
- Error metrics tracking
- Streamlit integration
- Async support

Author: Magnus Enhancement Team
"""
import streamlit as st
import logging
import time
import traceback
from typing import Callable, Any, Optional
from functools import wraps
from datetime import datetime

logger = logging.getLogger(__name__)

# Global error metrics
_error_metrics = {
    'total_errors': 0,
    'errors_by_type': {},
    'errors_by_function': {},
    'last_error_time': None
}

class APIRateLimitError(Exception):
    """Raised when API rate limit is exceeded"""
    pass

class DataNotAvailableError(Exception):
    """Raised when data is temporarily unavailable"""
    pass

def with_error_handling(
    fallback_value: Any = None,
    show_error: bool = True,
    error_message: Optional[str] = None
):
    """
    Decorator for graceful error handling with user feedback.

    Args:
        fallback_value: Value to return on error
        show_error: Whether to display error to user
        error_message: Custom error message (None = use exception message)

    Example:
        @with_error_handling(fallback_value=[])
        def get_data():
            return api.fetch()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except APIRateLimitError as e:
                if show_error:
                    st.warning(
                        f"⚠️ API rate limit reached. "
                        f"Showing cached data. Resets in 60 seconds."
                    )
                logger.warning(f"Rate limit in {func.__name__}: {e}")
                return fallback_value

            except DataNotAvailableError as e:
                if show_error:
                    st.info(
                        f"ℹ️ Data temporarily unavailable. "
                        f"Please try again in a moment."
                    )
                logger.info(f"Data unavailable in {func.__name__}: {e}")
                return fallback_value

            except ConnectionError as e:
                if show_error:
                    st.error(
                        f"❌ Connection error. "
                        f"Please check your internet connection and try again."
                    )
                logger.error(f"Connection error in {func.__name__}: {e}")
                return fallback_value

            except Exception as e:
                msg = error_message or f"An error occurred: {str(e)}"
                if show_error:
                    st.error(f"❌ {msg}")
                logger.exception(f"Error in {func.__name__}")
                return fallback_value

        return wrapper
    return decorator

def safe_cache_data(ttl: int = 300, **cache_kwargs):
    """
    Wrapper combining @st.cache_data with error handling.

    Args:
        ttl: Time to live in seconds
        **cache_kwargs: Additional arguments for @st.cache_data
    """
    def decorator(func: Callable) -> Callable:
        # Apply error handling first
        func_with_errors = with_error_handling(fallback_value=None)(func)
        # Then apply caching
        return st.cache_data(ttl=ttl, **cache_kwargs)(func_with_errors)
    return decorator


# Enhanced error handling with metrics tracking

def handle_errors(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    notify_user: bool = True,
    reraise: bool = False,
    custom_message: Optional[str] = None
):
    """
    Enhanced decorator for standardized error handling with metrics

    Args:
        default_return: Value to return on error (default: None)
        log_level: Logging level for errors (default: ERROR)
        notify_user: Show error to user via streamlit (default: True)
        reraise: Re-raise exception after handling (default: False)
        custom_message: Custom error message for user (default: auto-generated)

    Example:
        @handle_errors(default_return=[], notify_user=True)
        def get_all_stocks():
            return db.execute("SELECT * FROM stocks").fetchall()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                # Track error metrics
                _track_error(func.__name__, type(e).__name__)

                # Log error
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )

                # Notify user if requested
                if notify_user:
                    _notify_user_error(func.__name__, e, custom_message)

                # Re-raise if requested
                if reraise:
                    raise

                # Return default value
                return default_return

        return wrapper
    return decorator


def async_handle_errors(
    default_return: Any = None,
    log_level: int = logging.ERROR,
    notify_user: bool = True,
    reraise: bool = False,
    custom_message: Optional[str] = None
):
    """
    Async version of handle_errors decorator

    Args:
        Same as handle_errors

    Example:
        @async_handle_errors(default_return={})
        async def fetch_data():
            return await api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)

            except Exception as e:
                # Track error metrics
                _track_error(func.__name__, type(e).__name__)

                # Log error
                logger.log(
                    log_level,
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True
                )

                # Notify user if requested
                if notify_user:
                    _notify_user_error(func.__name__, e, custom_message)

                # Re-raise if requested
                if reraise:
                    raise

                # Return default value
                return default_return

        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry function on error with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry

    Example:
        @retry_on_error(max_retries=3, delay=1.0)
        def fetch_api_data():
            return requests.get(url).json()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}: {e}"
                        )

            # If we get here, all retries failed
            raise last_exception

        return wrapper
    return decorator


def async_retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Async version of retry_on_error decorator

    Args:
        Same as retry_on_error

    Example:
        @async_retry_on_error(max_retries=3)
        async def fetch_async_data():
            return await api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio

            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}: {e}"
                        )

            # If we get here, all retries failed
            raise last_exception

        return wrapper
    return decorator


def _track_error(function_name: str, error_type: str):
    """Track error metrics"""
    global _error_metrics

    _error_metrics['total_errors'] += 1
    _error_metrics['last_error_time'] = datetime.now()

    # Track by error type
    if error_type not in _error_metrics['errors_by_type']:
        _error_metrics['errors_by_type'][error_type] = 0
    _error_metrics['errors_by_type'][error_type] += 1

    # Track by function
    if function_name not in _error_metrics['errors_by_function']:
        _error_metrics['errors_by_function'][function_name] = 0
    _error_metrics['errors_by_function'][function_name] += 1


def _notify_user_error(function_name: str, error: Exception, custom_message: Optional[str] = None):
    """Notify user of error via streamlit"""
    try:
        if custom_message:
            message = custom_message
        else:
            # Generate user-friendly message
            error_type = type(error).__name__

            if 'connection' in str(error).lower() or 'timeout' in str(error).lower():
                message = "Connection error - please check your network connection and try again"
            elif 'database' in str(error).lower() or 'sql' in str(error).lower():
                message = "Database error - data may be temporarily unavailable"
            elif 'api' in str(error).lower() or 'http' in str(error).lower():
                message = "API error - external service may be temporarily unavailable"
            else:
                message = f"An error occurred: {error_type}"

        st.error(f"⚠️ {message}")

        # Show details in expander for debugging
        with st.expander("Error Details"):
            st.code(f"Function: {function_name}\nError: {str(error)}\n\n{traceback.format_exc()}")

    except Exception as e:
        # Don't let error notification cause additional errors
        logger.error(f"Error showing error notification: {e}")


def get_error_metrics() -> dict:
    """Get error metrics for monitoring"""
    return _error_metrics.copy()


def reset_error_metrics():
    """Reset error metrics (useful for testing)"""
    global _error_metrics
    _error_metrics = {
        'total_errors': 0,
        'errors_by_type': {},
        'errors_by_function': {},
        'last_error_time': None
    }


class ErrorContext:
    """
    Context manager for error handling

    Example:
        with ErrorContext("Loading data", notify_user=True):
            data = fetch_data()
    """

    def __init__(
        self,
        operation_name: str,
        notify_user: bool = True,
        reraise: bool = True,
        log_level: int = logging.ERROR
    ):
        self.operation_name = operation_name
        self.notify_user = notify_user
        self.reraise = reraise
        self.log_level = log_level

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Track error
            _track_error(self.operation_name, exc_type.__name__)

            # Log error
            logger.log(
                self.log_level,
                f"Error in {self.operation_name}: {exc_val}",
                exc_info=True
            )

            # Notify user
            if self.notify_user:
                _notify_user_error(self.operation_name, exc_val)

            # Return False to re-raise, True to suppress
            return not self.reraise

        return True


# Convenience decorators for common scenarios

def database_error_handler(func: Callable) -> Callable:
    """Decorator for database operation error handling"""
    return handle_errors(
        default_return=None,
        log_level=logging.ERROR,
        notify_user=True,
        custom_message="Database operation failed - data may be temporarily unavailable"
    )(func)


def api_error_handler(func: Callable) -> Callable:
    """Decorator for API call error handling"""
    return handle_errors(
        default_return=None,
        log_level=logging.WARNING,
        notify_user=True,
        custom_message="API call failed - service may be temporarily unavailable"
    )(func)


def critical_error_handler(func: Callable) -> Callable:
    """Decorator for critical operation error handling (re-raises after logging)"""
    return handle_errors(
        default_return=None,
        log_level=logging.CRITICAL,
        notify_user=True,
        reraise=True,
        custom_message="Critical error occurred"
    )(func)


def silent_error_handler(func: Callable) -> Callable:
    """Decorator for silent error handling (logs only, no user notification)"""
    return handle_errors(
        default_return=None,
        log_level=logging.WARNING,
        notify_user=False
    )(func)
