"""
Application Performance Monitoring (APM) with Sentry Integration

This module provides comprehensive performance monitoring, error tracking,
and observability for the AVA trading platform using Sentry.

Features:
- Automatic error tracking and reporting
- Performance transaction monitoring
- Custom performance metrics
- Database query tracking
- API call monitoring
- User context and breadcrumbs
- Release and environment tracking

Usage:
    from src.utils.apm_monitoring import init_sentry, track_performance, capture_exception

    # Initialize on app startup
    init_sentry()

    # Track performance
    @track_performance("load_positions")
    def load_positions():
        # Your code here
        pass

    # Manual error capture
    try:
        risky_operation()
    except Exception as e:
        capture_exception(e, context={"user_action": "manual_trade"})

Configuration (.env):
    SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id
    SENTRY_ENVIRONMENT=production  # or development, staging
    SENTRY_TRACES_SAMPLE_RATE=0.1  # 10% of transactions
    SENTRY_PROFILES_SAMPLE_RATE=0.1  # 10% of transactions for profiling
    SENTRY_ENABLED=true  # Enable/disable Sentry
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import Sentry
SENTRY_AVAILABLE = False
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    logger.warning("Sentry SDK not installed. Install with: pip install sentry-sdk")


class APMMonitor:
    """
    Application Performance Monitoring wrapper for Sentry.

    Provides graceful degradation if Sentry is not available or not configured.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.sentry_initialized = False

    def initialize(
        self,
        dsn: Optional[str] = None,
        environment: Optional[str] = None,
        traces_sample_rate: float = 0.1,
        profiles_sample_rate: float = 0.1,
        release: Optional[str] = None
    ) -> bool:
        """
        Initialize Sentry APM monitoring.

        Args:
            dsn: Sentry DSN (auto-loaded from env if not provided)
            environment: Environment name (production, staging, development)
            traces_sample_rate: Percentage of transactions to track (0.0 to 1.0)
            profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
            release: Release version string

        Returns:
            True if initialized successfully, False otherwise
        """
        # Check if Sentry is enabled
        if not SENTRY_AVAILABLE:
            logger.info("APM: Sentry SDK not available - monitoring disabled")
            return False

        enabled = os.getenv('SENTRY_ENABLED', 'false').lower() == 'true'
        if not enabled:
            logger.info("APM: Sentry disabled via SENTRY_ENABLED=false")
            return False

        # Get configuration from environment
        dsn = dsn or os.getenv('SENTRY_DSN')
        if not dsn:
            logger.warning("APM: SENTRY_DSN not configured - monitoring disabled")
            return False

        environment = environment or os.getenv('SENTRY_ENVIRONMENT', 'development')
        traces_sample_rate = float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', traces_sample_rate))
        profiles_sample_rate = float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', profiles_sample_rate))

        # Auto-detect release from git if available
        if not release:
            try:
                import subprocess
                result = subprocess.run(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    release = f"ava@{result.stdout.strip()}"
            except Exception:
                release = "ava@unknown"

        try:
            # Configure logging integration
            logging_integration = LoggingIntegration(
                level=logging.INFO,        # Capture info and above as breadcrumbs
                event_level=logging.ERROR  # Send errors as events
            )

            # Initialize Sentry
            sentry_sdk.init(
                dsn=dsn,
                environment=environment,
                release=release,
                traces_sample_rate=traces_sample_rate,
                profiles_sample_rate=profiles_sample_rate,
                integrations=[
                    logging_integration,
                    SqlalchemyIntegration(),
                ],
                # Set custom tags
                before_send=self._before_send,
                # Ignore certain errors
                ignore_errors=[
                    KeyboardInterrupt,
                    SystemExit,
                ]
            )

            self.enabled = True
            self.sentry_initialized = True

            logger.info(
                f"✅ APM initialized: environment={environment}, "
                f"traces={traces_sample_rate*100}%, profiles={profiles_sample_rate*100}%"
            )

            # Set global tags
            sentry_sdk.set_tag("platform", "streamlit")
            sentry_sdk.set_tag("application", "ava-trading")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
            return False

    def _before_send(self, event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Pre-process events before sending to Sentry.

        Use this to filter sensitive data, add custom context, etc.
        """
        # Remove sensitive data from event
        if 'request' in event:
            headers = event['request'].get('headers', {})
            # Remove authorization headers
            headers.pop('Authorization', None)
            headers.pop('X-API-Key', None)

        return event

    def set_user(self, user_id: Optional[str] = None, email: Optional[str] = None, **kwargs):
        """
        Set user context for error tracking.

        Args:
            user_id: User identifier
            email: User email
            **kwargs: Additional user properties
        """
        if not self.enabled:
            return

        user_data = {}
        if user_id:
            user_data['id'] = user_id
        if email:
            user_data['email'] = email
        user_data.update(kwargs)

        sentry_sdk.set_user(user_data)

    def set_context(self, key: str, value: Dict[str, Any]):
        """
        Add custom context to events.

        Args:
            key: Context key (e.g., "trade", "portfolio")
            value: Context data dictionary
        """
        if not self.enabled:
            return

        sentry_sdk.set_context(key, value)

    def add_breadcrumb(self, message: str, category: str = "default", level: str = "info", data: Optional[Dict] = None):
        """
        Add a breadcrumb for debugging.

        Args:
            message: Breadcrumb message
            category: Category (e.g., "navigation", "api", "database")
            level: Severity level (debug, info, warning, error)
            data: Additional data
        """
        if not self.enabled:
            return

        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )

    def capture_exception(self, exception: Exception, **kwargs):
        """
        Manually capture an exception.

        Args:
            exception: The exception to capture
            **kwargs: Additional context (tags, contexts, etc.)
        """
        if not self.enabled:
            logger.error(f"Exception (APM disabled): {exception}", exc_info=True)
            return

        # Add any custom context
        if 'tags' in kwargs:
            for key, value in kwargs['tags'].items():
                sentry_sdk.set_tag(key, value)

        if 'context' in kwargs:
            for key, value in kwargs['context'].items():
                sentry_sdk.set_context(key, value)

        sentry_sdk.capture_exception(exception)

    def capture_message(self, message: str, level: str = "info", **kwargs):
        """
        Capture a message event.

        Args:
            message: Message to capture
            level: Severity level (debug, info, warning, error, fatal)
            **kwargs: Additional context
        """
        if not self.enabled:
            logger.log(getattr(logging, level.upper(), logging.INFO), message)
            return

        sentry_sdk.capture_message(message, level=level)

    @contextmanager
    def transaction(self, name: str, op: str = "function"):
        """
        Context manager for performance transactions.

        Usage:
            with apm.transaction("load_positions", op="database"):
                positions = load_positions_from_db()

        Args:
            name: Transaction name
            op: Operation type (function, database, api, etc.)
        """
        if not self.enabled:
            # No-op if disabled
            yield None
            return

        transaction = sentry_sdk.start_transaction(name=name, op=op)
        try:
            yield transaction
        finally:
            transaction.finish()

    @contextmanager
    def span(self, description: str, op: str = "function"):
        """
        Context manager for performance spans within a transaction.

        Usage:
            with apm.transaction("load_page"):
                with apm.span("query_database", op="database"):
                    data = query_db()
                with apm.span("render_chart", op="render"):
                    chart = create_chart(data)

        Args:
            description: Span description
            op: Operation type
        """
        if not self.enabled:
            yield None
            return

        with sentry_sdk.start_span(op=op, description=description) as span:
            yield span


# Global APM instance
apm = APMMonitor()


def init_sentry(**kwargs) -> bool:
    """
    Initialize Sentry APM monitoring (convenience function).

    Returns:
        True if initialized successfully
    """
    return apm.initialize(**kwargs)


def track_performance(transaction_name: str, op: str = "function"):
    """
    Decorator to track function performance.

    Usage:
        @track_performance("load_positions", op="database")
        def load_positions():
            return db.query()

    Args:
        transaction_name: Name for the transaction
        op: Operation type (function, database, api, etc.)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not apm.enabled:
                # No monitoring overhead if disabled
                return func(*args, **kwargs)

            with apm.transaction(transaction_name, op=op):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def capture_exception(exception: Exception, **kwargs):
    """
    Capture an exception (convenience function).

    Args:
        exception: Exception to capture
        **kwargs: Additional context (tags, context, etc.)
    """
    apm.capture_exception(exception, **kwargs)


def measure_time(func_name: str):
    """
    Decorator to measure and log function execution time.

    This provides basic timing even when Sentry is disabled.

    Usage:
        @measure_time("expensive_query")
        def expensive_query():
            return db.query()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start

                if duration > 1.0:  # Log slow operations
                    logger.warning(f"⏱️ {func_name} took {duration:.2f}s")
                else:
                    logger.debug(f"⏱️ {func_name} took {duration:.3f}s")

                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"❌ {func_name} failed after {duration:.2f}s: {e}")
                raise

        return wrapper
    return decorator


# Convenience exports
__all__ = [
    'apm',
    'init_sentry',
    'track_performance',
    'capture_exception',
    'measure_time',
    'APMMonitor',
]
